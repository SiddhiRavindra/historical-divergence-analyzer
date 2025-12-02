"""
Library of Congress XML scraper with metadata extraction
"""

import xml.etree.ElementTree as ET
from typing import Dict, Optional
import re
import logging
from scrapers.document_models import DocumentData

logger = logging.getLogger(__name__)

class LocScraper:
    """Library of Congress XML scraper with metadata extraction"""
    
    def __init__(self, base_scraper, url_discovery):
        self.scraper = base_scraper
        self.discovery = url_discovery
    
    def scrape_loc_document(self, loc_url: str, document_info: Dict) -> Optional[DocumentData]:
        """
        Scrape a single LoC document
        document_info should contain: {'name': 'Election Night 1860', 'type': 'Letter'}
        """
        logger.info(f"Scraping LoC document: {document_info['name']}")
        
        # Step 1: Find XML download URL
        xml_url = self.discovery.find_loc_xml_url(loc_url)
        if not xml_url:
            logger.error(f"Could not find XML URL for {loc_url}")
            return None
        
        # Step 2: Fetch XML content
        result = self.scraper.fetch_url(xml_url)
        if not result.success:
            logger.error(f"Failed to fetch XML from {xml_url}")
            return None
        
        # Step 3: Parse XML and extract metadata
        try:
            metadata = self._parse_loc_xml(result.content, xml_url)
            if not metadata:
                logger.error(f"Failed to parse XML from {xml_url}")
                return None
            
            # Step 4: Create standardized document
            doc_data = DocumentData(
                id=self._generate_loc_id(loc_url),
                title=metadata.get('title', document_info['name']),
                reference=xml_url,
                document_type=document_info.get('type', 'Document'),
                date=metadata.get('date'),
                place=metadata.get('place'),
                from_person=metadata.get('from', 'Abraham Lincoln'),
                to_person=metadata.get('to'),
                content=metadata.get('content', ''),
                source='loc'
            )
            
            logger.info(f"Successfully scraped: {doc_data.title}")
            return doc_data
            
        except Exception as e:
            logger.error(f"Error processing LoC document {loc_url}: {str(e)}")
            return None
    
    def _parse_loc_xml(self, xml_content: str, xml_url: str) -> Optional[Dict]:
        """Parse LoC XML and extract all available metadata"""
        try:
            # Handle potential encoding issues
            if isinstance(xml_content, bytes):
                xml_content = xml_content.decode('utf-8', errors='ignore')
            
            root = ET.fromstring(xml_content)
            metadata = {}
            
            # Extract title (try multiple XML paths)
            title_paths = [
                ".//title",
                ".//titlestmt/title", 
                ".//filedesc/titlestmt/title",
                ".//tei2/teiheader/filedesc/titlestmt/title"
            ]
            
            for path in title_paths:
                title_elem = root.find(path)
                if title_elem is not None and title_elem.text:
                    metadata['title'] = title_elem.text.strip()
                    break
            
            # Extract date information
            date_paths = [
                ".//date",
                ".//publicationstmt/date",
                ".//profiledesc/creation/date"
            ]
            
            for path in date_paths:
                date_elem = root.find(path)
                if date_elem is not None:
                    # Get date from text or attributes
                    date_text = date_elem.text or date_elem.get('when') or date_elem.get('value')
                    if date_text:
                        metadata['date'] = date_text.strip()
                        break
            
            # Extract place information
            place_paths = [
                ".//placename",
                ".//place",
                ".//profiledesc/creation/placename"
            ]
            
            for path in place_paths:
                place_elem = root.find(path)
                if place_elem is not None and place_elem.text:
                    metadata['place'] = place_elem.text.strip()
                    break
            
            # Extract correspondence information (from/to)
            corr_paths = [
                (".//persname[@type='addressee']", 'to'),
                (".//persname[@type='sender']", 'from'),
                (".//name[@type='recipient']", 'to'),
                (".//name[@type='sender']", 'from')
            ]
            
            for path, field in corr_paths:
                elem = root.find(path)
                if elem is not None and elem.text:
                    metadata[field] = elem.text.strip()
            
            # Extract main content text
            content_paths = [
                ".//text",
                ".//body",
                ".//div[@type='letter']",
                ".//div[@type='speech']",
                ".//p"  # Fallback to paragraphs
            ]
            
            content_parts = []
            for path in content_paths:
                elements = root.findall(path)
                if elements:
                    for elem in elements:
                        text = self._extract_text_from_element(elem)
                        if text.strip():
                            content_parts.append(text)
                    break  # Use first successful path
            
            metadata['content'] = '\n\n'.join(content_parts) if content_parts else ''
            
            # If no content found, try to get all text
            if not metadata['content']:
                all_text = []
                for elem in root.iter():
                    if elem.text and elem.text.strip():
                        all_text.append(elem.text.strip())
                metadata['content'] = ' '.join(all_text)
            
            logger.info(f"Extracted metadata: title='{metadata.get('title', 'N/A')}', date='{metadata.get('date', 'N/A')}'")
            return metadata
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error for {xml_url}: {str(e)}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error parsing XML {xml_url}: {str(e)}")
            return None
    
    def _extract_text_from_element(self, element) -> str:
        """Recursively extract text from XML element"""
        text_parts = []
        
        if element.text:
            text_parts.append(element.text)
        
        for child in element:
            child_text = self._extract_text_from_element(child)
            if child_text:
                text_parts.append(child_text)
            if child.tail:
                text_parts.append(child.tail)
        
        return ' '.join(text_parts).strip()
    
    def _generate_loc_id(self, loc_url: str) -> str:
        """Generate unique ID from LoC URL"""
        # Extract meaningful identifier from URL
        match = re.search(r'/(mal\d+|mal\.\d+|[^/]+)/?$', loc_url)
        if match:
            return f"loc_{match.group(1).replace('.', '_')}"
        else:
            # Fallback: use hash of URL
            return f"loc_{abs(hash(loc_url)) % 10000:04d}"