import os
import json
import logging
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict, field
from dotenv import load_dotenv
from openai import OpenAI

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(env_path, override=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

GUTENBERG_AUTHORS = {
    "6811": "John G. Nicolay & John Hay",
    "6812": "Ida M. Tarbell",
    "12801": "Ward Hill Lamon",
    "14004": "Henry Ketcham",
    "18379": "Lord Charnwood",
}


def get_author_from_id(source_id: str) -> Optional[str]:
    numbers = re.findall(r'\d+', str(source_id))
    for num in numbers:
        if num in GUTENBERG_AUTHORS:
            return GUTENBERG_AUTHORS[num]
    return None


def get_author(doc: Dict, source_type: str) -> str:
    if doc.get("from") and doc["from"].strip():
        return doc["from"].strip()
    
    if doc.get("author") and doc["author"].strip():
        return doc["author"].strip()
    
    if doc.get("from_person") and doc["from_person"].strip():
        return doc["from_person"].strip()
    
    source_id = doc.get("id", "")
    author_from_id = get_author_from_id(source_id)
    if author_from_id:
        return author_from_id
    
    if source_type == "lincoln":
        return "Abraham Lincoln"
    else:
        return "Unknown Author"


EVENTS = {
    "election_night_1860": {
        "name": "Election Night 1860",
        "date": "November 6, 1860",
        "keywords": ["election", "1860", "november", "springfield", "telegraph", 
                     "republican", "electoral", "votes", "douglas", "elected", "president-elect"]
    },
    "fort_sumter": {
        "name": "Fort Sumter Decision",
        "date": "April 12-14, 1861",
        "keywords": ["fort sumter", "sumter", "charleston", "anderson", "beauregard",
                     "resupply", "cabinet", "bombardment", "april 1861", "first shot"]
    },
    "gettysburg_address": {
        "name": "Gettysburg Address",
        "date": "November 19, 1863",
        "keywords": ["gettysburg", "cemetery", "four score", "november 1863", 
                     "dedication", "battlefield", "consecrate", "perish"]
    },
    "second_inaugural": {
        "name": "Second Inaugural Address",
        "date": "March 4, 1865",
        "keywords": ["inaugural", "second inaugural", "march 1865", "malice toward none",
                     "charity", "oath", "second term", "inauguration"]
    },
    "ford_theatre": {
        "name": "Ford's Theatre Assassination",
        "date": "April 14-15, 1865",
        "keywords": ["ford's theatre", "ford theatre", "booth", "assassination", "shot", 
                     "april 14", "april 15", "petersen", "died", "assassin", "murdered"]
    }
}

EXTRACTION_PROMPT = """You are a historical document analyst. Extract information about "{event_name}" from the text below.

EVENT: {event_name} ({event_date})
SOURCE: {source_title} by {author}

TEXT:
{text}

Extract ONLY information explicitly stated in the text. Return JSON:
{{
    "event_found": true,
    "claims": ["specific factual claim 1", "specific factual claim 2"],
    "quotes": ["any direct quotes related to this event"],
    "temporal_details": {{"date": "specific date if mentioned", "time": "specific time if mentioned"}},
    "tone": "sympathetic/critical/neutral/reverential/analytical",
    "confidence": 0.0-1.0
}}

If the event is NOT mentioned in this text, return:
{{"event_found": false, "claims": [], "quotes": [], "temporal_details": {{}}, "tone": null, "confidence": 0.0}}

Return ONLY valid JSON, no other text."""


@dataclass
class EventExtraction:
    event: str
    event_name: str
    source_id: str
    source_title: str
    author: str
    source_type: str
    claims: List[str] = field(default_factory=list)
    quotes: List[str] = field(default_factory=list)
    temporal_details: Dict = field(default_factory=dict)
    tone: Optional[str] = None
    confidence: float = 0.0
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    def has_content(self) -> bool:
        return len(self.claims) > 0 or len(self.quotes) > 0


def chunk_text(text: str, chunk_size: int = 6000, overlap: int = 500) -> List[str]:
    if not text or not text.strip():
        return []
    
    words = text.split()
    if len(words) <= chunk_size:
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(' '.join(words[start:end]))
        start = end - overlap
        if start >= len(words) - overlap:
            break
    
    return chunks


def find_relevant_chunks(chunks: List[str], event_id: str, max_chunks: int = 5) -> List[str]:
    if not chunks:
        return []
    
    keywords = EVENTS[event_id]["keywords"]
    scored = []
    
    for chunk in chunks:
        chunk_lower = chunk.lower()
        score = sum(
            2 if ' ' in kw else 1 
            for kw in keywords 
            if kw.lower() in chunk_lower
        )
        if score > 0:
            scored.append((score, chunk))
    
    scored.sort(key=lambda x: x[0], reverse=True)
    return [chunk for _, chunk in scored[:max_chunks]]


def extract_with_llm(client: OpenAI, text: str, event_id: str, 
                     source_title: str, author: str, model: str) -> Dict:
    event_info = EVENTS[event_id]
    
    prompt = EXTRACTION_PROMPT.format(
        event_name=event_info["name"],
        event_date=event_info["date"],
        source_title=source_title,
        author=author,
        text=text[:15000]
    )
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1000,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content)
    except Exception as e:
        logger.error(f"LLM error: {e}")
        return {"event_found": False, "claims": [], "quotes": [], 
                "temporal_details": {}, "tone": None, "confidence": 0.0}


def merge_extractions(extractions: List[Dict]) -> Dict:
    valid = [e for e in extractions if e.get("event_found")]
    
    if not valid:
        return {"event_found": False, "claims": [], "quotes": [], 
                "temporal_details": {}, "tone": None, "confidence": 0.0}
    
    all_claims, all_quotes = [], []
    temporal = {}
    tones = []
    max_conf = 0.0
    
    for ext in valid:
        all_claims.extend(ext.get("claims", []))
        all_quotes.extend(ext.get("quotes", []))
        td = ext.get("temporal_details", {})
        if td.get("date"): temporal["date"] = td["date"]
        if td.get("time"): temporal["time"] = td["time"]
        if ext.get("tone"): tones.append(ext["tone"])
        max_conf = max(max_conf, ext.get("confidence", 0))
    
    return {
        "event_found": True,
        "claims": list(dict.fromkeys(all_claims)),
        "quotes": list(dict.fromkeys(all_quotes)),
        "temporal_details": temporal,
        "tone": tones[0] if tones else None,
        "confidence": max_conf
    }


datapath = Path(__file__).parent.parent.parent / 'src' / 'part1_data_acquisition/data/processed'


class EventExtractionPipeline:
    
    def __init__(self, data_dir: str = datapath, output_dir: str = "data/extractions"):
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        
        self.documents: List[Dict] = []
        self.results: List[EventExtraction] = []
        
        logger.info(f"Pipeline initialized with model: {self.model}")
    
    def load_documents(self) -> int:
        firstperson_path = self.data_dir / "dataset_firstperson.json"
        if firstperson_path.exists():
            with open(firstperson_path, 'r', encoding='utf-8') as f:
                docs = json.load(f)
                for doc in docs:
                    author = get_author(doc, "lincoln")
                    self.documents.append({
                        "id": doc.get("id", "unknown"),
                        "title": doc.get("title", "Unknown"),
                        "author": author,
                        "content": doc.get("content", ""),
                        "source_type": "lincoln"
                    })
            logger.info(f"Loaded {len(docs)} first-person documents")
        else:
            logger.warning(f"First-person file not found: {firstperson_path}")
        
        otherauthors_path = self.data_dir / "dataset_otherauthors.json"
        if otherauthors_path.exists():
            with open(otherauthors_path, 'r', encoding='utf-8') as f:
                docs = json.load(f)
                for doc in docs:
                    author = get_author(doc, "other_author")
                    self.documents.append({
                        "id": doc.get("id", "unknown"),
                        "title": doc.get("title", "Unknown"),
                        "author": author,
                        "content": doc.get("content", ""),
                        "source_type": "other_author"
                    })
            logger.info(f"Loaded {len(docs)} other-author documents")
        else:
            logger.warning(f"Other authors file not found: {otherauthors_path}")
        
        if self.documents:
            print("\n" + "=" * 60)
            print("AUTHOR DETECTION FROM 'from' FIELD")
            print("=" * 60)
            for doc in self.documents:
                label = "LINCOLN" if doc["source_type"] == "lincoln" else "OTHER"
                print(f"  [{label}] {doc['id']:<30} -> {doc['author']}")
            print("=" * 60)
        
        logger.info(f"Total documents loaded: {len(self.documents)}")
        return len(self.documents)
    
    def process_document_event(self, doc: Dict, event_id: str) -> EventExtraction:
        event_info = EVENTS[event_id]
        
        chunks = chunk_text(doc["content"])
        
        if not chunks:
            return EventExtraction(
                event=event_id,
                event_name=event_info["name"],
                source_id=doc["id"],
                source_title=doc["title"],
                author=doc["author"],
                source_type=doc["source_type"]
            )
        
        logger.debug(f"  Created {len(chunks)} chunks")
        
        relevant = find_relevant_chunks(chunks, event_id, max_chunks=5)
        
        if not relevant:
            return EventExtraction(
                event=event_id,
                event_name=event_info["name"],
                source_id=doc["id"],
                source_title=doc["title"],
                author=doc["author"],
                source_type=doc["source_type"]
            )
        
        logger.info(f"  Found {len(relevant)} relevant chunks")
        
        extractions = [
            extract_with_llm(self.client, chunk, event_id, 
                           doc["title"], doc["author"], self.model)
            for chunk in relevant
        ]
        
        merged = merge_extractions(extractions)
        
        return EventExtraction(
            event=event_id,
            event_name=event_info["name"],
            source_id=doc["id"],
            source_title=doc["title"],
            author=doc["author"],
            source_type=doc["source_type"],
            claims=merged.get("claims", []),
            quotes=merged.get("quotes", []),
            temporal_details=merged.get("temporal_details", {}),
            tone=merged.get("tone"),
            confidence=merged.get("confidence", 0.0)
        )
    
    def run(self) -> List[EventExtraction]:
        print("\n" + "=" * 60)
        print("PART 2: EVENT EXTRACTION PIPELINE")
        print("=" * 60)
        
        if not self.documents:
            self.load_documents()
        
        total_pairs = len(self.documents) * len(EVENTS)
        current = 0
        
        for doc in self.documents:
            source_label = "LINCOLN" if doc["source_type"] == "lincoln" else "OTHER"
            print(f"\n[{source_label}] {doc['author']} | {doc['title'][:40]}...")
            
            for event_id in EVENTS:
                current += 1
                event_name = EVENTS[event_id]["name"]
                print(f"  [{current}/{total_pairs}] Extracting: {event_name}")
                
                extraction = self.process_document_event(doc, event_id)
                self.results.append(extraction)
                
                if extraction.has_content():
                    print(f"    Found {len(extraction.claims)} claims")
                else:
                    print(f"    No information found")
        
        self.save_results()
        self.print_summary()
        
        return self.results
    
    def save_results(self):
        lincoln_results = [r for r in self.results if r.source_type == "lincoln"]
        others_results = [r for r in self.results if r.source_type == "other_author"]
        
        all_path = self.output_dir / "extractions_all.json"
        with open(all_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in self.results], f, indent=2, ensure_ascii=False)
        print(f"Saved: {all_path}")
        
        lincoln_path = self.output_dir / "extractions_lincoln.json"
        with open(lincoln_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in lincoln_results], f, indent=2, ensure_ascii=False)
        print(f"Saved: {lincoln_path}")
        
        others_path = self.output_dir / "extractions_others.json"
        with open(others_path, 'w', encoding='utf-8') as f:
            json.dump([r.to_dict() for r in others_results], f, indent=2, ensure_ascii=False)
        print(f"Saved: {others_path}")
        
        by_event = {}
        for event_id, event_info in EVENTS.items():
            lincoln_for_event = [
                r.to_dict() for r in lincoln_results 
                if r.event == event_id and r.has_content()
            ]
            others_for_event = [
                r.to_dict() for r in others_results 
                if r.event == event_id and r.has_content()
            ]
            
            by_event[event_id] = {
                "event_name": event_info["name"],
                "event_date": event_info["date"],
                "lincoln_claims": {
                    "count": len(lincoln_for_event),
                    "sources": lincoln_for_event
                },
                "other_author_claims": {
                    "count": len(others_for_event),
                    "sources": others_for_event
                }
            }
        
        by_event_path = self.output_dir / "extractions_by_event.json"
        with open(by_event_path, 'w', encoding='utf-8') as f:
            json.dump(by_event, f, indent=2, ensure_ascii=False)
        print(f"Saved: {by_event_path}")
        
        print(f"\nAll files saved to: {self.output_dir}/")
    
    def print_summary(self):
        print("\n" + "=" * 60)
        print("EXTRACTION SUMMARY")
        print("=" * 60)
        
        lincoln_results = [r for r in self.results if r.source_type == "lincoln"]
        others_results = [r for r in self.results if r.source_type == "other_author"]
        
        print("\nAUTHORS DETECTED:")
        authors = {}
        for r in self.results:
            if r.author not in authors:
                authors[r.author] = {"total": 0, "with_content": 0}
            authors[r.author]["total"] += 1
            if r.has_content():
                authors[r.author]["with_content"] += 1
        
        for author, counts in sorted(authors.items()):
            print(f"  {author}: {counts['with_content']}/{counts['total']} extractions with content")
        
        print(f"\n{'Event':<30} {'Lincoln':<10} {'Others':<10}")
        print("-" * 50)
        
        for event_id, info in EVENTS.items():
            lincoln_claims = sum(
                len(r.claims) for r in lincoln_results 
                if r.event == event_id
            )
            other_claims = sum(
                len(r.claims) for r in others_results 
                if r.event == event_id
            )
            print(f"{info['name']:<30} {lincoln_claims:<10} {other_claims:<10}")
        
        print("\n" + "=" * 60)
        print("OUTPUT FILES GENERATED:")
        print("  1. extractions_all.json      - All extractions unified")
        print("  2. extractions_lincoln.json  - Lincoln's first-person only")
        print("  3. extractions_others.json   - Other authors only")
        print("  4. extractions_by_event.json - Grouped for Part 3 Judge")
        print("=" * 60)


def main():
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found!")
        print("\nCreate a .env file with:")
        print("  OPENAI_API_KEY=sk-your-key-here")
        return 1
    
    pipeline = EventExtractionPipeline()
    
    if pipeline.load_documents() == 0:
        print("No documents found in part1_data_acquisition/data/processed/")
        print("Run Part 1 first to collect documents.")
        return 1
    
    lincoln_count = sum(1 for d in pipeline.documents if d["source_type"] == "lincoln")
    others_count = sum(1 for d in pipeline.documents if d["source_type"] == "other_author")
    
    print(f"\nDocuments Found:")
    print(f"  Lincoln (first-person): {lincoln_count}")
    print(f"  Other authors: {others_count}")
    print(f"  Total: {len(pipeline.documents)}")
    print(f"\nModel: {pipeline.model}")
    print(f"Events to extract: {len(EVENTS)}")
    print(f"Estimated API calls: ~{len(pipeline.documents) * len(EVENTS) * 3}")
    
    response = input("\nProceed with extraction? [y/N]: ")
    if response.lower() != 'y':
        print("Aborted.")
        return 0
    
    pipeline.run()
    
    print("\nPart 2 complete! Ready for Part 3 (LLM Judge)")
    return 0


if __name__ == "__main__":
    exit(main())