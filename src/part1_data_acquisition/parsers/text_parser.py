import re
import logging
from typing import Dict, Optional, List, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class TextParser:
    """Parser for Project Gutenberg and other plain text sources"""
    
    def __init__(self):
        # Gutenberg header/footer patterns
        self.gutenberg_start_patterns = [
            r'\*\*\* START OF TH(IS|E) PROJECT GUTENBERG',
            r'\*\*\* START OF THE PROJECT GUTENBERG',
            r'START OF TH(IS|E) PROJECT GUTENBERG',
            r'PRODUCED BY.*',
            r'PROJECT GUTENBERG.*EBOOK'
        ]
        
        self.gutenberg_end_patterns = [
            r'\*\*\* END OF TH(IS|E) PROJECT GUTENBERG',
            r'\*\*\* END OF THE PROJECT GUTENBERG', 
            r'END OF TH(IS|E) PROJECT GUTENBERG',
            r'END OF PROJECT GUTENBERG'
        ]
        
        # Common metadata patterns in text files
        self.metadata_patterns = {
            'title': [
                r'Title:\s*(.+)',
                r'THE\s+(.+)\s*\n',
                r'^(.+)\s*\nby\s+',
                r'Project Gutenberg.*\n\n(.+)\s*\n'
            ],
            'author': [
                r'Author:\s*(.+)',
                r'by\s+(.+)',
                r'By\s+(.+)',
                r'\nby\s+(.+)\s*\n'
            ],
            'date': [
                r'Release Date:\s*(.+)',
                r'Posted:\s*(.+)',
                r'Date:\s*(.+)',
                r'(\d{4}-\d{2}-\d{2})',
                r'(\w+\s+\d{1,2},\s+\d{4})'
            ],
            'language': [
                r'Language:\s*(.+)',
                r'Lang:\s*(.+)'
            ]
        }
    
    def parse_text_content(self, raw_text: str, source_url: str = "") -> Dict:
        """Parse raw text and extract metadata and clean content"""
        
        if not raw_text:
            return {"error": "Empty text provided"}
        
        try:
           
            metadata = self.extract_metadata(raw_text)
         
            if "gutenberg" in source_url.lower():
                clean_content = self.clean_gutenberg_text(raw_text)
            else:
                clean_content = self.clean_generic_text(raw_text)
            
            # Combine results
            result = {
                "content": clean_content,
                "metadata": metadata,
                "original_length": len(raw_text),
                "cleaned_length": len(clean_content),
                "source_type": "gutenberg" if "gutenberg" in source_url.lower() else "generic"
            }
            
            logger.info(f"Parsed text: {len(clean_content)} chars from {len(raw_text)} original")
            return result
            
        except Exception as e:
            logger.error(f"Error parsing text content: {str(e)}")
            return {"error": str(e), "content": raw_text}
    
    def extract_metadata(self, text: str) -> Dict:
        
        metadata = {}
     
        for field, patterns in self.metadata_patterns.items():
            for pattern in patterns:
                match = re.search(pattern, text[:2000], re.IGNORECASE | re.MULTILINE)
                if match:
                    value = match.group(1).strip()
                    if value and len(value) < 200:  # Reasonable length limit
                        metadata[field] = value
                        break
        
        
        if 'title' not in metadata:
            title = self.extract_title_heuristic(text)
            if title:
                metadata['title'] = title
        
        # Clean up extracted metadata
        for key, value in metadata.items():
            if isinstance(value, str):
                # Remove excessive whitespace and control characters
                cleaned = ' '.join(value.split())
                metadata[key] = cleaned
        
        return metadata
    
    def extract_title_heuristic(self, text: str) -> Optional[str]:
       
        lines = text.split('\n')[:50]  # Look at first 50 lines
        
        # Look for lines that might be titles
        potential_titles = []
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines and very short lines
            if not line or len(line) < 3:
                continue
            
            # Skip lines that look like metadata
            if any(pattern in line.lower() for pattern in ['project gutenberg', 'release date', 'produced by', 'language:']):
                continue
            
            # Lines in ALL CAPS might be titles
            if line.isupper() and 5 <= len(line) <= 100:
                potential_titles.append((line, 'caps'))
            
            # Lines that are longer than surrounding lines might be titles
            if 10 <= len(line) <= 100 and not line.endswith('.'):
                potential_titles.append((line, 'length'))
        
        # Return the most likely title
        if potential_titles:
            # Prefer ALL CAPS titles first
            caps_titles = [t for t in potential_titles if t[1] == 'caps']
            if caps_titles:
                return caps_titles[0][0].title()  # Convert to title case
            else:
                return potential_titles[0][0]
        
        return None
    
    def clean_gutenberg_text(self, raw_text: str) -> str:
       
        if not raw_text:
            return ""
        
        lines = raw_text.split('\n')
        start_idx = 0
        end_idx = len(lines)
        
        # Find start of actual content
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            for pattern in self.gutenberg_start_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    start_idx = i + 1
                    break
            if start_idx > 0:
                break
        
      
        for i in range(len(lines) - 1, -1, -1):
            line_lower = lines[i].lower().strip()
            for pattern in self.gutenberg_end_patterns:
                if re.search(pattern, lines[i], re.IGNORECASE):
                    end_idx = i
                    break
            if end_idx < len(lines):
                break
        
       
        content_lines = lines[start_idx:end_idx]
        
        # Join and clean
        content = '\n'.join(content_lines)
        
 
        content = self.clean_generic_text(content)
        
        return content
    
    def clean_generic_text(self, text: str) -> str:
       
        if not text:
            return ""
        
      
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        
        text = re.sub(r'\n{3,}', '\n\n', text)
        
       
        lines = []
        for line in text.split('\n'):
            lines.append(line.rstrip())
        
        
        text = '\n'.join(lines)
        
        
        text = text.strip()
        
        return text
    
    def split_into_chapters(self, text: str) -> List[Dict]:
       
        if not text:
            return []
      
        chapter_patterns = [
            r'^CHAPTER\s+[IVXLCDM]+',  # Roman numerals
            r'^CHAPTER\s+\d+',         # Arabic numerals
            r'^Chapter\s+\d+',         # Capitalized
            r'^PART\s+[IVXLCDM]+',     # Parts with Roman numerals
            r'^PART\s+\d+',            # Parts with numbers
            r'^\d+\.\s*[A-Z][a-z]'     # Numbered sections
        ]
        
        chapters = []
        lines = text.split('\n')
        current_chapter = []
        chapter_title = "Beginning"
        
        for line in lines:
          
            is_chapter_start = False
            for pattern in chapter_patterns:
                if re.match(pattern, line.strip()):
                   
                    if current_chapter:
                        content = '\n'.join(current_chapter).strip()
                        if content:
                            chapters.append({
                                "title": chapter_title,
                                "content": content,
                                "length": len(content)
                            })
                  
                    chapter_title = line.strip()
                    current_chapter = []
                    is_chapter_start = True
                    break
            
            if not is_chapter_start:
                current_chapter.append(line)
        
      
        if current_chapter:
            content = '\n'.join(current_chapter).strip()
            if content:
                chapters.append({
                    "title": chapter_title,
                    "content": content,
                    "length": len(content)
                })
        
       
        if not chapters and text.strip():
            chapters.append({
                "title": "Full Text",
                "content": text.strip(),
                "length": len(text.strip())
            })
        
        logger.info(f"Split text into {len(chapters)} sections")
        return chapters
    
    def extract_text_statistics(self, text: str) -> Dict:
        
        if not text:
            return {"error": "No text provided"}
        
        lines = text.split('\n')
        words = text.split()
        
        # Count paragraphs
        paragraphs = re.split(r'\n\s*\n', text.strip())
        paragraphs = [p for p in paragraphs if p.strip()]
        
        stats = {
            "character_count": len(text),
            "word_count": len(words),
            "line_count": len(lines),
            "paragraph_count": len(paragraphs),
            "average_words_per_paragraph": len(words) / len(paragraphs) if paragraphs else 0,
            "average_characters_per_word": len(text) / len(words) if words else 0,
            "blank_line_count": sum(1 for line in lines if not line.strip())
        }
        
        return stats

def test_text_parser():
    sample_text = """Project Gutenberg's Lincoln Biography

Title: Abraham Lincoln: A Biography
Author: John Smith  
Release Date: January 1, 2020
Language: English

*** START OF THIS PROJECT GUTENBERG EBOOK ABRAHAM LINCOLN ***

ABRAHAM LINCOLN
A Biography

by John Smith

CHAPTER I
EARLY LIFE

Abraham Lincoln was born in a log cabin in Kentucky...

This is the main content of the biography.
It contains multiple paragraphs and important information.

CHAPTER II
POLITICAL CAREER

Lincoln entered politics in his twenties...

*** END OF THIS PROJECT GUTENBERG EBOOK ABRAHAM LINCOLN ***
"""
    
    parser = TextParser()
    
    # Test parsing
    result = parser.parse_text_content(sample_text, "https://gutenberg.org/test")
    
    print("Metadata extracted:")
    for key, value in result.get("metadata", {}).items():
        print(f"  {key}: {value}")
    
    print(f"\nContent length: {len(result.get('content', ''))}")
    print(f"Original length: {result.get('original_length', 0)}")
    
    # Test chapter splitting
    chapters = parser.split_into_chapters(result.get("content", ""))
    print(f"\nFound {len(chapters)} chapters:")
    for chapter in chapters:
        print(f"  - {chapter['title']} ({chapter['length']} chars)")
    
    # Test statistics
    stats = parser.extract_text_statistics(result.get("content", ""))
    print(f"\nText statistics:")
    print(f"  Words: {stats['word_count']}")
    print(f"  Paragraphs: {stats['paragraph_count']}")

if __name__ == "__main__":
    test_text_parser()