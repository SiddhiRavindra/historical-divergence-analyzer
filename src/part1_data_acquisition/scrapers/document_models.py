from dataclasses import dataclass
from typing import Optional

@dataclass
class DocumentData:
    """Standard document data structure"""
    id: str
    title: str
    reference: str
    document_type: str
    date: Optional[str] = None
    place: Optional[str] = None
    from_person: Optional[str] = None
    to_person: Optional[str] = None
    content: str = ""
    source: str = ""  # 'loc' or 'gutenberg'
