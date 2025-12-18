"""
Vector Store Module for Lincoln Historical Divergence - RAG (LangChain + OpenAI + Chroma Cloud)
Indexes Part 2 extracted claims/quotes for fast retrieval by event and corpus (Lincoln vs Others).
"""

import os
import json
import hashlib
from typing import List, Dict, Optional
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings


def load_part2_extractions(part2_dir: str) -> List[Dict]:
    """
    Loads extractions_all.json (flat list) if present; otherwise builds from by_event.
    """
    part2 = Path(part2_dir)
    p_all = part2 / "extractions_all.json"
    p_by = part2 / "extractions_by_event.json"

    if p_all.exists():
        with open(p_all, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []

    if p_by.exists():
        with open(p_by, "r", encoding="utf-8") as f:
            by_event = json.load(f)

        rows = []
        for event_id, payload in by_event.items():
            lin = (payload.get("lincoln_claims") or {}).get("sources") or []
            oth = (payload.get("other_author_claims") or {}).get("sources") or []
            for r in lin + oth:
                if "event" not in r:
                    r = {**r, "event": event_id}
                rows.append(r)
        return rows

    return []


class LincolnVectorStore:
    def __init__(
        self,
        chroma_api_key: str,
        chroma_tenant: str,
        chroma_db: str,
        openai_api_key: str,
        collection_name: str = "historical_divergence_part2_v1",
        chunk_size: int = 800,
        chunk_overlap: int = 150,
    ):
        # Chroma Cloud
        self.client = chromadb.CloudClient(
            api_key=chroma_api_key,
            tenant=chroma_tenant,
            database=chroma_db
        )
        self.collection_name = collection_name
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"description": "Lincoln Historical Divergence - Part2 claims/quotes"}
        )

        # Chunker (mostly for long claim sets; single claims usually don’t need chunking)
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
            keep_separator=True
        )

        # Embeddings
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        )

        print(f"✓ Connected to collection: {self.collection_name}")

    def _stable_id(self, *parts: str) -> str:
        base = "|".join(parts)
        return hashlib.md5(base.encode("utf-8")).hexdigest()

    def ingest_part2(self, rows: List[Dict], force_refresh: bool = False) -> Dict:
        stats = {"sources_processed": 0, "chunks_created": 0, "chunks_stored": 0, "errors": []}

        if force_refresh:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass
            self.collection = self.client.get_or_create_collection(name=self.collection_name)

        all_texts, all_metas, all_ids = [], [], []

        for r in rows:
            try:
                stats["sources_processed"] += 1

                event = r.get("event", "")
                event_name = r.get("event_name", "")
                source_type = r.get("source_type", "")  # lincoln / other_author
                source_id = r.get("source_id", "")
                source_title = r.get("source_title", "")
                author = r.get("author", "")
                tone = r.get("tone", "")
                confidence = r.get("confidence", None)

                claims = r.get("claims") or []
                quotes = r.get("quotes") or []

                # Store each claim as its own "document"
                for i, claim in enumerate(claims):
                    claim = str(claim).strip()
                    if len(claim) < 10:
                        continue

                    chunks = self.splitter.split_text(claim)
                    for j, ch in enumerate(chunks):
                        stats["chunks_created"] += 1
                        _id = self._stable_id("claim", event, source_id, str(i), str(j))

                        meta = {
                            "type": "claim",
                            "event": event,
                            "event_name": event_name,
                            "source_type": source_type,
                            "source_id": source_id,
                            "source_title": source_title,
                            "author": author,
                            "tone": tone,
                            "confidence": confidence,
                            "claim_index": i,
                            "chunk_index": j,
                        }

                        all_ids.append(_id)
                        all_texts.append(ch)
                        all_metas.append(meta)

                # Optional: store quotes too
                for i, quote in enumerate(quotes):
                    quote = str(quote).strip()
                    if len(quote) < 10:
                        continue
                    _id = self._stable_id("quote", event, source_id, str(i))

                    all_ids.append(_id)
                    all_texts.append(quote)
                    all_metas.append({
                        "type": "quote",
                        "event": event,
                        "event_name": event_name,
                        "source_type": source_type,
                        "source_id": source_id,
                        "source_title": source_title,
                        "author": author,
                        "quote_index": i,
                    })

            except Exception as e:
                stats["errors"].append(str(e))

        if all_texts:
            vectors = self.embeddings.embed_documents(all_texts)
            self.collection.upsert(ids=all_ids, documents=all_texts, metadatas=all_metas, embeddings=vectors)
            stats["chunks_stored"] = len(all_texts)

        return stats

    def search(
        self,
        query: str,
        top_k: int = 6,
        event: Optional[str] = None,
        source_type: Optional[str] = None
    ) -> List[Dict]:
        q_vec = self.embeddings.embed_query(query)

        where = {}
        if event:
            where["event"] = event
        if source_type:
            where["source_type"] = source_type

        results = self.collection.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            where=where if where else None
        )

        out = []
        for i, doc in enumerate(results["documents"][0]):
            out.append({
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if "distances" in results else None
            })
        return out
