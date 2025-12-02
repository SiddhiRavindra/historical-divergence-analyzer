from typing import Dict, Optional
import re
import logging
from bs4 import BeautifulSoup
from scrapers.document_models import DocumentData

logger = logging.getLogger(__name__)

class GutenbergScraper:
    
    def __init__(self, base_scraper, url_discovery):
        self.scraper = base_scraper
        self.discovery = url_discovery
    
    def scrape_gutenberg_book(self, gutenberg_url: str) -> Optional[DocumentData]:
        logger.info(f"Scraping Gutenberg book from: {gutenberg_url}")
        
        
        metadata = self._extract_book_metadata(gutenberg_url)
        if not metadata:
            logger.error(f"Could not extract metadata for {gutenberg_url}")
            return None
        
        
        text_url = self.discovery.find_gutenberg_text_url(gutenberg_url)
        if not text_url:
            logger.error(f"Could not find text URL for {gutenberg_url}")
            return None
        
        
        result = self.scraper.fetch_url(text_url)
        if not result.success:
            logger.error(f"Failed to fetch text from {text_url}")
            return None
        
        
        cleaned_content = self._clean_gutenberg_text(result.content)
        
        
        doc_data = DocumentData(
            id=self._generate_gutenberg_id(gutenberg_url),
            title=metadata.get('title', 'Unknown Title'),
            reference=text_url,
            document_type='Book',
            date=metadata.get('date'),
            place=metadata.get('place'),
            from_person=metadata.get('author'),
            to_person=None,
            content=cleaned_content,
            source='gutenberg'
        )
        
        logger.info(f"Successfully scraped: {doc_data.title} by {doc_data.from_person}")
        return doc_data
    
    def _extract_book_metadata(self, gutenberg_url: str) -> Optional[Dict]:
        
        result = self.scraper.fetch_url(gutenberg_url)
        if not result.success:
            return None
        
        soup = BeautifulSoup(result.content, 'html.parser')
        metadata = {}
     
        title_elem = soup.find('h1')
        if title_elem:
            metadata['title'] = title_elem.get_text(strip=True)
       
        author_elem = soup.find('a', href=re.compile(r'/browse/authors/'))
        if author_elem:
            metadata['author'] = author_elem.get_text(strip=True)
        
       
        bibrec_table = soup.find('table', class_='bibrec')
        if bibrec_table:
            for row in bibrec_table.find_all('tr'):
                header = row.find('th')
                data = row.find('td')
                
                if header and data:
                    header_text = header.get_text(strip=True).lower()
                    data_text = data.get_text(strip=True)
                    
                    if 'release date' in header_text or 'posted' in header_text:
                        metadata['date'] = data_text
                    elif 'language' in header_text:
                        metadata['language'] = data_text
        
        return metadata
    
    def _clean_gutenberg_text(self, raw_text: str) -> str:
       
        if not raw_text:
            return ""
        
        lines = raw_text.split('\n')
        
        
        start_idx = 0
        for i, line in enumerate(lines):
            if any(marker in line.lower() for marker in [
                '*** start of this project gutenberg',
                '*** start of the project gutenberg',
                'produced by'
            ]):
                start_idx = i + 1
                break
       
        end_idx = len(lines)
        for i in range(len(lines) - 1, -1, -1):
            if any(marker in lines[i].lower() for marker in [
                '*** end of this project gutenberg',
                '*** end of the project gutenberg',
                'end of project gutenberg'
            ]):
                end_idx = i
                break
        
        content_lines = lines[start_idx:end_idx]
        
        
        content = '\n'.join(content_lines)
        
        
        content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
        
        return content.strip()
    
    def _generate_gutenberg_id(self, gutenberg_url: str) -> str:
        
        match = re.search(r'/ebooks/(\d+)', gutenberg_url)
        if match:
            return f"gutenberg_{match.group(1)}"
        else:
            return f"gutenberg_{abs(hash(gutenberg_url)) % 10000:04d}"