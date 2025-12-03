import logging
import json
from pathlib import Path
from typing import List, Dict, Optional

from scrapers.base_scraper import BaseScraper, URLDiscovery
from scrapers.loc_scraper import LocScraper
from scrapers.gutenberg_scraper import GutenbergScraper
from scrapers.document_models import DocumentData

from parsers.text_parser import TextParser
from parsers.xml_parser import XMLParser
from data_normalizer import DataNormalizer

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class EnhancedPart1DataCollector:
    
    def __init__(self):
        self.base_scraper = BaseScraper(rate_limit_seconds=2.5)
        self.discovery = URLDiscovery(self.base_scraper)
        self.loc_scraper = LocScraper(self.base_scraper, self.discovery)
        self.gutenberg_scraper = GutenbergScraper(self.base_scraper, self.discovery)
        
        self.text_parser = TextParser()
        self.xml_parser = XMLParser()
        self.data_normalizer = DataNormalizer()
        
        logger.info("Initialized data collector")
        
        self.loc_sources = {
            "Election Night 1860": {
                "url": "https://www.loc.gov/item/mal0440500/",
                "type": "Letter"
            },
            "Fort Sumter Decision": {
                "url": "https://www.loc.gov/resource/mal.0882800",
                "type": "Letter"
            },
            "Gettysburg Address": {
                "url": "https://www.loc.gov/exhibits/gettysburg-address/ext/trans-nicolay-copy.html",
                "type": "Speech"
            },
            "Second Inaugural Address": {
                "url": "https://www.loc.gov/resource/mal.4361300",
                "type": "Speech"
            },
            "Last Public Address": {
                "url": "https://www.loc.gov/resource/mal.4361800/",
                "type": "Speech"
            }
        }
        
        self.gutenberg_sources = [
            "https://www.gutenberg.org/ebooks/6812",
            "https://www.gutenberg.org/ebooks/6811",
            "https://www.gutenberg.org/ebooks/12801/",
            "https://www.gutenberg.org/ebooks/14004/",
            "https://www.gutenberg.org/ebooks/18379"
        ]
        
        self.output_dir = Path("data/processed")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def scrape_and_parse_loc_document(self, loc_url: str, document_info: Dict) -> Optional[DocumentData]:
        logger.info(f"Processing LoC document: {document_info['name']}")
        
        try:
            special_case_key = self.discovery.is_special_case(loc_url)
            
            if special_case_key:
                logger.info(f"Special case detected: {special_case_key}")
                
                doc = self.discovery.scrape_special_document(
                    loc_url, 
                    {'name': document_info['name'], 'type': document_info['type']}
                )
                
                if doc:
                    normalized_doc = self.data_normalizer.normalize_document(doc)
                    logger.info(f"Special case SUCCESS: {normalized_doc.title} ({len(normalized_doc.content)} chars)")
                    return normalized_doc
                else:
                    logger.error(f"Special case FAILED: {document_info['name']}")
                    return None
            
            logger.info(f"Using standard XML extraction")
            
            xml_url = self.discovery.find_loc_xml_url(loc_url)
            if not xml_url:
                logger.error(f"Could not find XML URL for {loc_url}")
                return None
            
            result = self.base_scraper.fetch_url(xml_url)
            if not result.success:
                logger.error(f"Failed to fetch XML: {xml_url}")
                return None
            
            parse_result = self.xml_parser.parse_xml_content(result.content, xml_url)
            
            if not parse_result.get("parsing_success"):
                logger.error(f"XML parsing failed")
                return None
            
            parsed_metadata = parse_result.get("metadata", {})
            
            doc_data = DocumentData(
                id=self._generate_loc_id(loc_url),
                title=parsed_metadata.get('title') if parsed_metadata.get('title') else document_info['name'],
                reference=xml_url if xml_url else loc_url,
                document_type=document_info.get('type') if document_info.get('type') else 'Document',
                date=parsed_metadata.get('date') if parsed_metadata.get('date') else None,
                place=parsed_metadata.get('place') if parsed_metadata.get('place') else None,
                from_person=parsed_metadata.get('from_person') if parsed_metadata.get('from_person') else 'Abraham Lincoln',
                to_person=parsed_metadata.get('to_person') if parsed_metadata.get('to_person') else None,
                content=parse_result.get("content") if parse_result.get("content") else "",
                source='loc'
            )
            
            normalized_doc = self.data_normalizer.normalize_document(doc_data)
            logger.info(f"XML SUCCESS: {normalized_doc.title} ({len(normalized_doc.content)} chars)")
            return normalized_doc
            
        except Exception as e:
            logger.error(f"Error processing {loc_url}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def scrape_and_parse_gutenberg_book(self, gutenberg_url: str) -> Optional[DocumentData]:
        logger.info(f"Processing Gutenberg: {gutenberg_url}")
        
        try:
            doc = self.gutenberg_scraper.scrape_gutenberg_book(gutenberg_url)
            
            if doc:
                normalized_doc = self.data_normalizer.normalize_document(doc)
                logger.info(f"Gutenberg SUCCESS: {normalized_doc.title} by {normalized_doc.from_person} ({len(normalized_doc.content)} chars)")
                return normalized_doc
            else:
                logger.error(f"Gutenberg scraping returned None for {gutenberg_url}")
                return None
            
        except Exception as e:
            logger.error(f"Error processing Gutenberg {gutenberg_url}: {str(e)}")
            import traceback
            traceback.print_exc()
            return None
    
    def collect_firstperson_documents(self) -> List[DocumentData]:
        logger.info("=" * 60)
        logger.info("COLLECTING FIRST-PERSON DOCUMENTS (Library of Congress)")
        logger.info("=" * 60)
        
        documents = []
        
        for i, (name, info) in enumerate(self.loc_sources.items(), 1):
            logger.info(f"[{i}/{len(self.loc_sources)}] {name}")
            
            doc = self.scrape_and_parse_loc_document(
                info["url"],
                {'name': name, 'type': info["type"]}
            )
            
            if doc:
                documents.append(doc)
            else:
                logger.error(f"FAILED: {name}")
        
        logger.info(f"First-person: {len(documents)}/{len(self.loc_sources)} collected")
        return documents
    
    def collect_otherauthor_documents(self) -> List[DocumentData]:
        logger.info("=" * 60)
        logger.info("COLLECTING OTHER-AUTHOR DOCUMENTS (Project Gutenberg)")
        logger.info("=" * 60)
        
        documents = []
        
        for i, url in enumerate(self.gutenberg_sources, 1):
            logger.info(f"[{i}/{len(self.gutenberg_sources)}] {url}")
            
            doc = self.scrape_and_parse_gutenberg_book(url)
            
            if doc:
                documents.append(doc)
            else:
                logger.error(f"FAILED: Book {i}")
        
        logger.info(f"Other-author: {len(documents)}/{len(self.gutenberg_sources)} collected")
        return documents
    
    def export_to_json(self, documents: List[DocumentData], filename: str) -> bool:
        try:
            json_data = self.data_normalizer.documents_to_json_list(documents)
            output_path = self.output_dir / filename
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(json_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Exported {len(json_data)} docs to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Export failed: {str(e)}")
            return False
    
    def run_enhanced_data_acquisition(self) -> bool:
        logger.info("=" * 60)
        logger.info("PART 1: DATA ACQUISITION")
        logger.info("=" * 60)
        
        try:
            firstperson_docs = self.collect_firstperson_documents()
            otherauthor_docs = self.collect_otherauthor_documents()
            
            logger.info("=" * 60)
            logger.info("EXPORTING")
            logger.info("=" * 60)
            
            self.export_to_json(firstperson_docs, "dataset_firstperson.json")
            self.export_to_json(otherauthor_docs, "dataset_otherauthors.json")
            
            total = len(firstperson_docs) + len(otherauthor_docs)
            target = len(self.loc_sources) + len(self.gutenberg_sources)
            
            print("\n" + "=" * 50)
            print("COLLECTION SUMMARY")
            print("=" * 50)
            print(f"First-person (LoC):    {len(firstperson_docs)}/{len(self.loc_sources)}")
            print(f"Other-author (Gut):    {len(otherauthor_docs)}/{len(self.gutenberg_sources)}")
            print(f"Total:                 {total}/{target} ({total/target*100:.0f}%)")
            print("=" * 50)
            
            if otherauthor_docs:
                print("\nGutenberg Authors Extracted:")
                for doc in otherauthor_docs:
                    author = doc.from_person if doc.from_person else "Unknown"
                    place = doc.place if doc.place else "N/A"
                    print(f"  - {doc.id}: {author} (Place: {place})")
            
            return total >= 8
            
        except Exception as e:
            logger.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()
            return False
    
    def _generate_loc_id(self, url: str) -> str:
        import re
        match = re.search(r'/(mal\d+|mal\.\d+|[^/]+)/?$', url)
        if match:
            return f"loc_{match.group(1).replace('.', '_')}"
        return f"loc_{abs(hash(url)) % 10000:04d}"
    
    def _generate_gutenberg_id(self, url: str) -> str:
        import re
        match = re.search(r'/ebooks/(\d+)', url)
        if match:
            return f"gutenberg_{match.group(1)}"
        return f"gutenberg_{abs(hash(url)) % 10000:04d}"


def main():
    collector = EnhancedPart1DataCollector()
    success = collector.run_enhanced_data_acquisition()
    
    if success:
        print("\nPart 1 completed successfully!")
        return 0
    else:
        print("\nPart 1 completed with errors")
        return 1


if __name__ == "__main__":
    exit(main())