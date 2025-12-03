from typing import Dict, Optional
import re
import logging
from bs4 import BeautifulSoup
from scrapers.document_models import DocumentData

logger = logging.getLogger(__name__)


class GutenbergScraper:
    
    KNOWN_BOOKS = {
        "6811": {
            "author": "Francis Fisher Browne",
            "title": "The Every-day Life of Abraham Lincoln",
            "place": "New York"
        },
        "6812": {
            "author": "Francis Fisher Browne",
            "title": "The Every-day Life of Abraham Lincoln",
            "place": "New York"
        },
        "12801": {
            "author": "Noah Brooks",
            "title": "Abraham Lincoln and the Downfall of American Slavery",
            "place": "New York"
        },
        "14004": {
            "author": "Isaac N. Arnold",
            "title": "The Life of Abraham Lincoln",
            "place": "Chicago"
        },
        "18379": {
            "author": "Lord Charnwood",
            "title": "Abraham Lincoln",
            "place": "London"
        }
    }
    
    def __init__(self, base_scraper, url_discovery):
        self.scraper = base_scraper
        self.discovery = url_discovery
    
    def scrape_gutenberg_book(self, gutenberg_url: str) -> Optional[DocumentData]:
        logger.info(f"Scraping Gutenberg book from: {gutenberg_url}")
        
        book_id = self._extract_book_id(gutenberg_url)
        
        metadata = self._extract_book_metadata(gutenberg_url, book_id)
        if not metadata:
            metadata = {}
        
        text_url = self.discovery.find_gutenberg_text_url(gutenberg_url)
        if not text_url:
            logger.error(f"Could not find text URL for {gutenberg_url}")
            return None
        
        result = self.scraper.fetch_url(text_url)
        if not result.success:
            logger.error(f"Failed to fetch text from {text_url}")
            return None
        
        cleaned_content = self._clean_gutenberg_text(result.content)
        
        if not metadata.get('author') or metadata.get('author') == 'Unknown':
            header_author = self._extract_author_from_text_header(result.content)
            if header_author:
                metadata['author'] = header_author
        
        doc_data = DocumentData(
            id=f"gutenberg_{book_id}" if book_id else f"gutenberg_unknown",
            title=metadata.get('title') if metadata.get('title') else "Unknown Title",
            reference=text_url if text_url else gutenberg_url,
            document_type='Book',
            date=metadata.get('release_date') if metadata.get('release_date') else None,
            place=metadata.get('place') if metadata.get('place') else None,
            from_person=metadata.get('author') if metadata.get('author') else "Unknown",
            to_person=None,
            content=cleaned_content if cleaned_content else "",
            source='gutenberg'
        )
        
        logger.info(f"Successfully scraped: {doc_data.title} by {doc_data.from_person}")
        return doc_data
    
    def _extract_book_id(self, gutenberg_url: str) -> str:
        match = re.search(r'/ebooks?/(\d+)', gutenberg_url)
        if match:
            return match.group(1)
        return "unknown"
    
    def _extract_book_metadata(self, gutenberg_url: str, book_id: str) -> Optional[Dict]:
        result = self.scraper.fetch_url(gutenberg_url)
        if not result.success:
            return self._get_fallback_metadata(book_id)
        
        soup = BeautifulSoup(result.content, 'html.parser')
        metadata = {}
        
        bibrec_table = soup.find('table', class_='bibrec')
        if bibrec_table:
            bibrec_data = self._parse_bibrec_table(bibrec_table)
            metadata.update(bibrec_data)
        
        if not metadata.get('author'):
            author_elem = soup.find('a', itemprop='creator')
            if author_elem:
                metadata['author'] = self._clean_author_name(author_elem.get_text(strip=True))
        
        if not metadata.get('author'):
            author_patterns = [
                r'/ebooks/author/\d+',
                r'/browse/authors/',
                r'/author/'
            ]
            for pattern in author_patterns:
                author_elem = soup.find('a', href=re.compile(pattern))
                if author_elem:
                    metadata['author'] = self._clean_author_name(author_elem.get_text(strip=True))
                    break
        
        if not metadata.get('author'):
            header_author = self._extract_author_from_page_header(soup)
            if header_author:
                metadata['author'] = header_author
        
        if not metadata.get('author') or metadata.get('author') == 'Unknown':
            fallback = self._get_fallback_metadata(book_id)
            if fallback:
                if not metadata.get('author') or metadata.get('author') == 'Unknown':
                    metadata['author'] = fallback.get('author', 'Unknown')
                if not metadata.get('title'):
                    metadata['title'] = fallback.get('title')
                if not metadata.get('place'):
                    metadata['place'] = fallback.get('place')
        
        if not metadata.get('title'):
            title_elem = soup.find('h1', itemprop='name')
            if not title_elem:
                title_elem = soup.find('h1')
            if title_elem:
                metadata['title'] = title_elem.get_text(strip=True)
        
        return metadata
    
    def _parse_bibrec_table(self, table) -> Dict:
        metadata = {}
        
        for row in table.find_all('tr'):
            header = row.find('th')
            data = row.find('td')
            
            if not header or not data:
                continue
            
            header_text = header.get_text(strip=True).lower()
            
            if header_text in ['author', 'creator', 'by']:
                author_link = data.find('a')
                if author_link:
                    author_name = author_link.get_text(strip=True)
                else:
                    author_name = data.get_text(strip=True)
                metadata['author'] = self._clean_author_name(author_name)
            
            elif header_text == 'title':
                metadata['title'] = data.get_text(strip=True)
            
            elif 'release date' in header_text or 'posted' in header_text:
                metadata['release_date'] = data.get_text(strip=True)
            
            elif header_text == 'language':
                metadata['language'] = data.get_text(strip=True)
            
            elif header_text in ['subject', 'subjects']:
                metadata['subject'] = data.get_text(strip=True)
            
            elif header_text in ['loc class', 'location', 'place', 'publisher']:
                metadata['place'] = data.get_text(strip=True)
            
            elif 'original publication' in header_text or 'note' in header_text:
                note_text = data.get_text(strip=True)
                if not metadata.get('place'):
                    place = self._extract_place_from_note(note_text)
                    if place:
                        metadata['place'] = place
        
        return metadata
    
    def _extract_place_from_note(self, note_text: str) -> Optional[str]:
        if not note_text:
            return None
        
        match = re.search(r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)\s*[:\,]', note_text)
        if match:
            place = match.group(1)
            if place.lower() not in ['the', 'a', 'an', 'this', 'volume']:
                return place
        
        match = re.search(r'[Pp]ublished\s+(?:in|at)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)', note_text)
        if match:
            return match.group(1)
        
        return None
    
    def _extract_author_from_page_header(self, soup) -> Optional[str]:
        for container in [soup.find('div', id='content'), soup.find('body')]:
            if not container:
                continue
            
            text_content = container.get_text()
            
            by_match = re.search(
                r'\bby\s+([A-Z][a-zA-Z\s\.\-\']+?)(?:\n|,|\d{4}|$)',
                text_content[:2000]
            )
            if by_match:
                author = by_match.group(1).strip()
                if len(author) > 3 and len(author) < 100:
                    return self._clean_author_name(author)
        
        return None
    
    def _extract_author_from_text_header(self, raw_text: str) -> Optional[str]:
        if not raw_text:
            return None
        
        header = raw_text[:3000]
        
        match = re.search(
            r'Project Gutenberg (?:EBook|eBook) of .+?,\s*by\s+([^\n\r]+)',
            header,
            re.IGNORECASE
        )
        if match:
            return self._clean_author_name(match.group(1))
        
        match = re.search(r'^Author:\s*(.+)$', header, re.MULTILINE | re.IGNORECASE)
        if match:
            return self._clean_author_name(match.group(1))
        
        match = re.search(r'\n\s*by\s+([A-Z][^\n]+)\n', header)
        if match:
            return self._clean_author_name(match.group(1))
        
        return None
    
    def _clean_author_name(self, name: str) -> str:
        if not name:
            return "Unknown"
        
        name = re.sub(r',?\s*\d{4}\s*-\s*\d{4}', '', name)
        name = re.sub(r',?\s*\d{4}\s*-\s*$', '', name)
        name = re.sub(r',?\s*\(\d{4}-\d{4}\)', '', name)
        name = re.sub(r',?\s*[bd]\.\s*\d{4}', '', name)
        name = re.sub(r',\s*(Jr\.|Sr\.|III|II|IV)?\s*$', '', name)
        
        name = name.strip(' ,.-:')
        name = ' '.join(name.split())
        
        return name if name else "Unknown"
    
    def _get_fallback_metadata(self, book_id: str) -> Optional[Dict]:
        if book_id in self.KNOWN_BOOKS:
            return self.KNOWN_BOOKS[book_id].copy()
        return None
    
    def _clean_gutenberg_text(self, raw_text: str) -> str:
        if not raw_text:
            return ""
        
        lines = raw_text.split('\n')
        
        start_idx = 0
        for i, line in enumerate(lines):
            line_lower = line.lower()
            if any(marker in line_lower for marker in [
                '*** start of this project gutenberg',
                '*** start of the project gutenberg',
                '***start of',
                'produced by'
            ]):
                start_idx = i + 1
                while start_idx < len(lines) and not lines[start_idx].strip():
                    start_idx += 1
                break
        
        end_idx = len(lines)
        for i in range(len(lines) - 1, max(0, len(lines) - 500), -1):
            line_lower = lines[i].lower()
            if any(marker in line_lower for marker in [
                '*** end of this project gutenberg',
                '*** end of the project gutenberg',
                '***end of',
                'end of project gutenberg'
            ]):
                end_idx = i
                break
        
        content_lines = lines[start_idx:end_idx]
        content = '\n'.join(content_lines)
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        return content.strip()
    
    def _generate_gutenberg_id(self, gutenberg_url: str) -> str:
        match = re.search(r'/ebooks?/(\d+)', gutenberg_url)
        if match:
            return f"gutenberg_{match.group(1)}"
        return f"gutenberg_{abs(hash(gutenberg_url)) % 10000:04d}"
    