"""
Ingestion Script - Load Part 2 event extractions, chunk with LangChain, embed with OpenAI,
and store in ChromaDB Cloud.
"""

import os
import sys
import time
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple
from dotenv import load_dotenv

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


# ---------------------------
# ENV LOADING (similar style)
# ---------------------------
project_root = Path(__file__).parent.parent.parent

# Try src/.env first (like your reference), else fallback to project_root/.env
# env_path_src = project_root / "src" / ".env"
env_path_root = project_root / ".env"

# env_path = env_path_src if env_path_src.exists() else env_path_root
env_path = env_path_root
print(f"Loading .env from: {env_path}")
load_dotenv(dotenv_path=env_path, override=True)


def clean_env_value(value):
    """Remove quotes from environment variable values."""
    if value is None:
        return None
    value = value.strip()
    if (value.startswith("'") and value.endswith("'")) or (value.startswith('"') and value.endswith('"')):
        value = value[1:-1]
    return value


# ---------------------------
# DATA LOADING / NORMALIZING
# ---------------------------
def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def flatten_by_event(by_event: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Converts extractions_by_event.json shape into a flat list of source records.
    """
    rows: List[Dict[str, Any]] = []
    for event_id, payload in by_event.items():
        # payload: {event_name, event_date, lincoln_claims:{sources:[]}, other_author_claims:{sources:[]}}
        lin_sources = (payload.get("lincoln_claims") or {}).get("sources") or []
        oth_sources = (payload.get("other_author_claims") or {}).get("sources") or []

        for r in lin_sources + oth_sources:
            # ensure event id exists
            if "event" not in r:
                r = {**r, "event": event_id}
            rows.append(r)
    return rows


def load_part2_rows(part2_dir: Path) -> List[Dict[str, Any]]:
    """
    Priority:
      1) extractions_all.json (already flat)
      2) extractions_by_event.json (needs flattening)
      3) combine extractions_lincoln.json + extractions_others.json
    """
    p_all = part2_dir / "extractions_all.json"
    p_by  = part2_dir / "extractions_by_event.json"
    p_l   = part2_dir / "extractions_lincoln.json"
    p_o   = part2_dir / "extractions_others.json"

    if p_all.exists():
        print(f"✓ Found: {p_all}")
        data = load_json(p_all)
        if isinstance(data, list):
            return data

    if p_by.exists():
        print(f"✓ Found: {p_by}")
        data = load_json(p_by)
        if isinstance(data, dict):
            return flatten_by_event(data)

    rows: List[Dict[str, Any]] = []
    if p_l.exists():
        print(f"✓ Found: {p_l}")
        dl = load_json(p_l)
        if isinstance(dl, list):
            rows.extend(dl)
    if p_o.exists():
        print(f"✓ Found: {p_o}")
        do = load_json(p_o)
        if isinstance(do, list):
            rows.extend(do)

    return rows


def build_source_text(row: Dict[str, Any]) -> str:
    """
    Convert one Part-2 extraction record into a text block suitable for chunking + retrieval.
    (We store claims/quotes as text; metadata keeps event/author/source_type, etc.)
    """
    event = row.get("event", "")
    event_name = row.get("event_name", "")
    source_title = row.get("source_title", "")
    source_id = row.get("source_id", "")
    author = row.get("author", "")
    source_type = row.get("source_type", "")
    temporal = row.get("temporal_details") or {}
    tone = row.get("tone")
    confidence = row.get("confidence")

    claims = row.get("claims") or []
    quotes = row.get("quotes") or []

    parts = []
    parts.append(f"Event: {event_name} ({event})")
    parts.append(f"Source Title: {source_title}")
    parts.append(f"Source ID: {source_id}")
    parts.append(f"Author: {author}")
    parts.append(f"Source Type: {source_type}")
    if temporal:
        parts.append(f"Temporal Details: {json.dumps(temporal)}")
    if tone is not None:
        parts.append(f"Tone: {tone}")
    if confidence is not None:
        parts.append(f"Confidence: {confidence}")

    if claims:
        parts.append("\nClaims:")
        for c in claims:
            c = str(c).strip()
            if c:
                parts.append(f"- {c}")

    if quotes:
        parts.append("\nQuotes:")
        for q in quotes:
            q = str(q).strip()
            if q:
                parts.append(f"> {q}")

    return "\n".join(parts).strip()


# ---------------------------
# CHROMA VECTOR STORE WRAPPER
# ---------------------------
class Part2VectorStore:
    def __init__(
        self,
        api_key: str,
        tenant: str,
        database: str,
        openai_api_key: str,
        collection_name: str,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        embedding_model: str = "text-embedding-3-small",
    ):
        self.client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant,
            database=database,
        )

        # OpenAI embeddings (like your reference)
        self.embeddings = OpenAIEmbeddings(
            api_key=openai_api_key,
            model=embedding_model,
        )

        # LangChain chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def reset_collection(self):
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass
        self.collection = self.client.get_or_create_collection(name=self.collection_name)

    def ingest_part2_rows(self, rows: List[Dict[str, Any]], force_refresh: bool = False) -> Dict[str, Any]:
        if force_refresh:
            print("🧹 Force refresh enabled: deleting & recreating collection...")
            self.reset_collection()

        sources_processed = 0
        chunks_created = 0
        chunks_stored = 0
        errors: List[str] = []

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []

        for row in rows:
            try:
                sources_processed += 1

                # Skip empty extractions (no claims and no quotes)
                if not (row.get("claims") or row.get("quotes")):
                    continue

                text = build_source_text(row)
                chunks = self.splitter.split_text(text)

                event = row.get("event", "")
                source_id = row.get("source_id", "unknown_source")
                source_type = row.get("source_type", "")

                for i, ch in enumerate(chunks):
                    chunks_created += 1

                    doc_id = f"{event}|{source_id}|{source_type}|chunk_{i}"
                    metadata = {
                        "event": row.get("event", ""),
                        "event_name": row.get("event_name", ""),
                        "source_id": row.get("source_id", ""),
                        "source_title": row.get("source_title", ""),
                        "author": row.get("author", ""),
                        "source_type": row.get("source_type", ""),
                        "tone": row.get("tone", ""),
                        "confidence": row.get("confidence", None),
                        "temporal_date": (row.get("temporal_details") or {}).get("date", ""),
                        "chunk_index": i,
                    }

                    ids.append(doc_id)
                    docs.append(ch)
                    metas.append(metadata)

            except Exception as e:
                errors.append(f"{row.get('source_id','unknown')}: {str(e)}")

        # Embed + upsert in batches
        BATCH = 64
        for start in range(0, len(docs), BATCH):
            batch_docs = docs[start:start + BATCH]
            batch_ids = ids[start:start + BATCH]
            batch_metas = metas[start:start + BATCH]

            try:
                vectors = self.embeddings.embed_documents(batch_docs)
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_docs,
                    metadatas=batch_metas,
                    embeddings=vectors,
                )
                chunks_stored += len(batch_docs)
            except Exception as e:
                errors.append(f"Upsert batch {start}-{start+BATCH}: {str(e)}")

        return {
            "sources_processed": sources_processed,
            "chunks_created": chunks_created,
            "chunks_stored": chunks_stored,
            "errors": errors,
            "collection": self.collection_name,
            "chunk_size": self.splitter._chunk_size,
            "chunk_overlap": self.splitter._chunk_overlap,
            "embedding_model": getattr(self.embeddings, "model", "unknown"),
        }


def main():
    print("=" * 70)
    print(" Part 2 Ingestion: LangChain Chunking + OpenAI Embeddings + Chroma Cloud")
    print("=" * 70)

    # Read env
    chroma_api_key = clean_env_value(os.getenv("CHROMA_API_KEY"))
    chroma_tenant = clean_env_value(os.getenv("CHROMA_TENANT"))
    chroma_db = clean_env_value(os.getenv("CHROMA_DB"))

    openai_api_key = clean_env_value(os.getenv("OPENAI_API_KEY")) or clean_env_value(os.getenv("OPENAI_KEY"))

    collection_name = clean_env_value(os.getenv("COLLECTION_NAME")) or "historical_divergence_part2_v1"
    part2_path =str(project_root/"src"/"Part2_Event_Extraction" /"data" / "extractions")
    part2_dir = Path(part2_path)
    print(str(project_root/"src"/"Part2_Event_Extraction" /"data" / "extractions"))

    print("\n🔍 Environment Variables:")
    print(f"  CHROMA_API_KEY: {'✓ Set' if chroma_api_key else '✗ Missing'}")
    print(f"  CHROMA_TENANT: {'✓ Set' if chroma_tenant else '✗ Missing'}")
    print(f"  CHROMA_DB: {'✓ Set' if chroma_db else '✗ Missing'}")
    print(f"  OPENAI_API_KEY: {'✓ Set' if openai_api_key else '✗ Missing'}")
    print(f"  PART2_PATH: {part2_dir}")
    print(f"  COLLECTION_NAME: {collection_name}")

    if not all([chroma_api_key, chroma_tenant, chroma_db, openai_api_key]):
        print("\n❌ Missing required credentials in .env")
        print("Required: CHROMA_API_KEY, CHROMA_TENANT, CHROMA_DB, OPENAI_API_KEY")
        sys.exit(1)

    if not part2_dir.exists():
        print(f"\n❌ PART2_PATH does not exist: {part2_dir}")
        sys.exit(1)

    # Load rows
    print("\n📥 Loading Part 2 extraction files...")
    rows = load_part2_rows(part2_dir)
    if not rows:
        print("❌ No extraction rows found. Ensure files exist in PART2_PATH.")
        sys.exit(1)

    print(f"✓ Loaded {len(rows)} extraction records")

    # Confirm
    print("\n⚠️  This will embed text using OpenAI embeddings (cost may apply).")
    resp = input("Continue? (yes/no): ").strip().lower()
    if resp not in ["yes", "y"]:
        print("Cancelled.")
        sys.exit(0)

    refresh_resp = input("Force refresh (delete existing collection)? (yes/no): ").strip().lower()
    force_refresh = refresh_resp in ["yes", "y"]

    # Init store
    try:
        print("\n🔌 Initializing Chroma Cloud + LangChain splitter + OpenAI embeddings...")
        store = Part2VectorStore(
            api_key=chroma_api_key,
            tenant=chroma_tenant,
            database=chroma_db,
            openai_api_key=openai_api_key,
            collection_name=collection_name,
            chunk_size=1000,
            chunk_overlap=200,
            embedding_model="text-embedding-3-small",
        )
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # Ingest
    print("\n" + "=" * 70)
    print("Starting ingestion...")
    print("=" * 70)

    start_time = time.time()
    stats = store.ingest_part2_rows(rows, force_refresh=force_refresh)
    elapsed = time.time() - start_time

    print("\n" + "=" * 70)
    print("✅ Ingestion Complete")
    print("=" * 70)
    print(f"  Collection: {stats['collection']}")
    print(f"  Sources processed: {stats['sources_processed']}")
    print(f"  Chunks created: {stats['chunks_created']}")
    print(f"  Chunks stored: {stats['chunks_stored']}")
    print(f"  Embedding model: {stats['embedding_model']}")
    print(f"  Chunk size/overlap: {stats['chunk_size']}/{stats['chunk_overlap']}")
    print(f"  Time: {elapsed:.2f}s ({elapsed/60:.2f} min)")

    if stats["errors"]:
        print(f"\n⚠️  Errors: {len(stats['errors'])}")
        for err in stats["errors"][:5]:
            print(f"  - {err}")


if __name__ == "__main__":
    main()
