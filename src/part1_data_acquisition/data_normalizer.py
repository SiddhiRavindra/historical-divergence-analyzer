import re
import logging
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from scrapers.document_models import DocumentData

logger = logging.getLogger(__name__)

class DataNormalizer:
    
    
    def __init__(self):
        self.required_fields = ["id", "title", "reference", "document_type", "content"]
        self.optional_fields = ["date", "place", "from", "to"]
        
    def normalize_document(self, doc: DocumentData) -> DocumentData:
        
        
       
        normalized_doc = DocumentData(
            id=self._normalize_id(doc.id),
            title=self._normalize_title(doc.title),
            reference=self._normalize_reference(doc.reference),
            document_type=self._normalize_document_type(doc.document_type),
            date=self._normalize_date(doc.date),
            place=self._normalize_place(doc.place),
            from_person=self._normalize_person(doc.from_person),
            to_person=self._normalize_person(doc.to_person),
            content=self._normalize_content(doc.content),
            source=doc.source
        )
        
        return normalized_doc
    
    def normalize_document_list(self, documents: List[DocumentData]) -> List[DocumentData]:
        
        normalized_docs = []
        
        for i, doc in enumerate(documents):
            try:
                normalized_doc = self.normalize_document(doc)
                normalized_docs.append(normalized_doc)
                logger.debug(f"Normalized document {i+1}/{len(documents)}: {doc.title}")
            except Exception as e:
                logger.error(f"Failed to normalize document {i+1}: {str(e)}")
                
                normalized_docs.append(doc)
        
        return normalized_docs
    
    def validate_document(self, doc: DocumentData) -> Tuple[bool, List[str]]:
        
        errors = []
        
        # Check required fields
        if not doc.id or not doc.id.strip():
            errors.append("Missing or empty ID field")
        
        if not doc.title or not doc.title.strip():
            errors.append("Missing or empty title field")
        
        if not doc.reference or not doc.reference.strip():
            errors.append("Missing or empty reference field")
        
        if not doc.document_type or not doc.document_type.strip():
            errors.append("Missing or empty document_type field")
        
        if not doc.content or not doc.content.strip():
            errors.append("Missing or empty content field")
        
        # Check ID format
        if doc.id and not re.match(r'^[a-zA-Z0-9_-]+$', doc.id):
            errors.append("ID contains invalid characters (use only letters, numbers, underscore, dash)")
        
        # Check document type is valid
        valid_types = ["Letter", "Speech", "Note", "Book", "Document"]
        if doc.document_type and doc.document_type not in valid_types:
            errors.append(f"Invalid document_type: {doc.document_type}. Must be one of: {valid_types}")
        
        # Check content length
        if doc.content and len(doc.content.strip()) < 10:
            errors.append("Content is too short (less than 10 characters)")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def validate_document_list(self, documents: List[DocumentData]) -> Dict:
        
        validation_report = {
            "total_documents": len(documents),
            "valid_documents": 0,
            "invalid_documents": 0,
            "validation_errors": [],
            "duplicate_ids": [],
            "empty_content_count": 0,
            "missing_dates": 0,
            "missing_metadata": 0
        }
        
        seen_ids = set()
        
        for i, doc in enumerate(documents):
            is_valid, errors = self.validate_document(doc)
            
            if is_valid:
                validation_report["valid_documents"] += 1
            else:
                validation_report["invalid_documents"] += 1
                validation_report["validation_errors"].append({
                    "document_index": i,
                    "document_id": doc.id,
                    "errors": errors
                })
            
            # Check for duplicate IDs
            if doc.id in seen_ids:
                validation_report["duplicate_ids"].append(doc.id)
            else:
                seen_ids.add(doc.id)
            
            # Count quality metrics
            if not doc.content or len(doc.content.strip()) == 0:
                validation_report["empty_content_count"] += 1
            
            if not doc.date:
                validation_report["missing_dates"] += 1
            
            if not doc.place and not doc.from_person:
                validation_report["missing_metadata"] += 1
        
        validation_report["validation_success_rate"] = (
            validation_report["valid_documents"] / validation_report["total_documents"] * 100
            if validation_report["total_documents"] > 0 else 0
        )
        
        return validation_report
    
    def to_json_format(self, doc: DocumentData) -> Dict:
        
        return {
            "id": doc.id,
            "title": doc.title,
            "reference": doc.reference,
            "document_type": doc.document_type,
            "date": doc.date if doc.date else "Date unknown",
            "place": doc.place if doc.place else "",
            "from": doc.from_person if doc.from_person else "",
            "to": doc.to_person if doc.to_person else "",
            "content": doc.content
        }
    
    def documents_to_json_list(self, documents: List[DocumentData]) -> List[Dict]:
        
        return [self.to_json_format(doc) for doc in documents]
    
    def _normalize_id(self, doc_id: str) -> str:
        
        if not doc_id:
            return f"unknown_{int(datetime.now().timestamp())}"
        
        # Clean ID: only letters, numbers, underscore, dash
        clean_id = re.sub(r'[^a-zA-Z0-9_-]', '_', doc_id.strip())
        return clean_id[:50]  # Limit length
    
    def _normalize_title(self, title: str) -> str:
        
        if not title:
            return "Untitled Document"
        
        # Clean up whitespace and limit length
        normalized = ' '.join(title.strip().split())
        return normalized[:200]  
    
    def _normalize_reference(self, reference: str) -> str:
        
        if not reference:
            return ""
        
        return reference.strip()
    
    def _normalize_document_type(self, doc_type: str) -> str:
       
        if not doc_type:
            return "Document"
        
        # Standardize common types
        type_mapping = {
            "letter": "Letter",
            "speech": "Speech", 
            "note": "Note",
            "book": "Book",
            "document": "Document",
            "correspondence": "Letter",
            "address": "Speech",
            "manuscript": "Document"
        }
        
        normalized = doc_type.strip().lower()
        return type_mapping.get(normalized, doc_type.title())
    
    def _normalize_date(self, date: str) -> Optional[str]:
        
        if not date:
            return None
        
        # clean whitespace
        return date.strip()
    
    def _normalize_place(self, place: str) -> Optional[str]:
        
        if not place:
            return None
        
        # Clean whitespace and standardize
        normalized = ' '.join(place.strip().split())
        return normalized if normalized else None
    
    def _normalize_person(self, person: str) -> Optional[str]:
       
        if not person:
            return None
        
        # Clean whitespace and standardize
        normalized = ' '.join(person.strip().split())
        return normalized if normalized else None
    
    def _normalize_content(self, content: str) -> str:
        
        if not content:
            return ""
       
        normalized = content.strip()
        
        
        normalized = normalized.replace('\r\n', '\n').replace('\r', '\n')
        
        
        normalized = re.sub(r'\n{4,}', '\n\n\n', normalized)
        
        return normalized
    
    def generate_quality_metrics(self, documents: List[DocumentData]) -> Dict:
       
        if not documents:
            return {"error": "No documents provided"}
        
        metrics = {
            "total_documents": len(documents),
            "content_statistics": {
                "avg_content_length": sum(len(doc.content) for doc in documents) / len(documents),
                "min_content_length": min(len(doc.content) for doc in documents),
                "max_content_length": max(len(doc.content) for doc in documents),
                "empty_content_count": sum(1 for doc in documents if not doc.content.strip())
            },
            "metadata_completeness": {
                "documents_with_dates": sum(1 for doc in documents if doc.date),
                "documents_with_places": sum(1 for doc in documents if doc.place),
                "documents_with_from": sum(1 for doc in documents if doc.from_person),
                "documents_with_to": sum(1 for doc in documents if doc.to_person)
            },
            "source_distribution": {}
        }
        
        # Count by source
        for doc in documents:
            source = doc.source or "unknown"
            if source not in metrics["source_distribution"]:
                metrics["source_distribution"][source] = 0
            metrics["source_distribution"][source] += 1
        
        # Calculate completeness percentages
        total = metrics["total_documents"]
        completeness = metrics["metadata_completeness"]
        completeness["date_completeness"] = f"{completeness['documents_with_dates']}/{total} ({completeness['documents_with_dates']/total*100:.1f}%)"
        completeness["place_completeness"] = f"{completeness['documents_with_places']}/{total} ({completeness['documents_with_places']/total*100:.1f}%)"
        
        return metrics

def test_normalizer():
    
    test_doc = DocumentData(
        id="test@doc#1!",
        title="  Test   Document  Title  ",
        reference="http://example.com/test",
        document_type="letter",
        date="  1860-11-06  ",
        place="  Springfield,   Illinois  ",
        from_person="  Abraham   Lincoln  ",
        to_person="",
        content="  This is test content\n\n\n\n\nWith excessive whitespace  ",
        source="test"
    )
    
    normalizer = DataNormalizer()
    
    
    normalized = normalizer.normalize_document(test_doc)
    print("Original ID:", repr(test_doc.id))
    print("Normalized ID:", repr(normalized.id))
    print("Original title:", repr(test_doc.title))
    print("Normalized title:", repr(normalized.title))
    
    is_valid, errors = normalizer.validate_document(normalized)
    print("Is valid:", is_valid)
    print("Validation errors:", errors)
    
    json_format = normalizer.to_json_format(normalized)
    print("JSON format keys:", list(json_format.keys()))

if __name__ == "__main__":
    test_normalizer()