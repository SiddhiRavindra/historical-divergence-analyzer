import time
import requests
import re
import random
import logging
from typing import Optional, Dict, List
from urllib.parse import urljoin, urlparse
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
from dataclasses import dataclass

from scrapers.document_models import DocumentData

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class ScrapingResult:
    
    success: bool
    content: Optional[str] = None
    metadata: Optional[Dict] = None
    error: Optional[str] = None
    url: Optional[str] = None


class BaseScraper:
   
    
    def __init__(self, rate_limit_seconds: float = 5.0, max_retries: int = 3):
        self.rate_limit_seconds = rate_limit_seconds
        self.max_retries = max_retries
        self.last_request_time = 0
        self.session = self._create_session()
        logger.info(f"Initialized BaseScraper with {rate_limit_seconds}s rate limit")
        
    def _create_session(self) -> requests.Session:
       
        session = requests.Session()
        
        retry_strategy = Retry(
            total=self.max_retries,
            backoff_factor=2,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["HEAD", "GET", "OPTIONS"]
        )
        
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        
        return session
    
    def _get_headers(self, url: str) -> Dict[str, str]:
        
        user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0'
        ]
        
        headers = {
            'User-Agent': random.choice(user_agents),
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'DNT': '1',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'no-cache',
            'Pragma': 'no-cache',
        }
        
        parsed_url = urlparse(url)
        if 'loc.gov' in parsed_url.netloc:
            headers['Referer'] = 'https://www.google.com/'
            headers['Sec-Fetch-Site'] = 'cross-site'
        
        return headers
    
    def _enforce_rate_limit(self):
        
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        jitter = random.uniform(0.5, 1.5)
        adjusted_rate_limit = self.rate_limit_seconds * jitter
        
        if time_since_last < adjusted_rate_limit:
            sleep_time = adjusted_rate_limit - time_since_last
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def fetch_url(self, url: str, timeout: int = 30, allow_redirects: bool = True) -> ScrapingResult:
        
        self._enforce_rate_limit()
        
        try:
            logger.info(f"Fetching: {url}")
            headers = self._get_headers(url)
            
            response = self.session.get(
                url, 
                timeout=timeout, 
                allow_redirects=allow_redirects,
                headers=headers,
                verify=True
            )
            
            response.raise_for_status()
            
            return ScrapingResult(
                success=True,
                content=response.text,
                url=response.url,
                metadata={
                    'status_code': response.status_code, 
                    'encoding': response.encoding,
                    'content_type': response.headers.get('Content-Type', ''),
                    'final_url': response.url
                }
            )
            
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 403:
                logger.warning(f"403 Forbidden for {url} - LoC may be blocking automated access")
                return ScrapingResult(
                    success=False, 
                    error=f"403 Forbidden - {url}", 
                    url=url,
                    metadata={'status_code': 403}
                )
            error_msg = f"HTTP error {e.response.status_code} for {url}: {e.response.reason}"
            logger.error(error_msg)
            return ScrapingResult(success=False, error=error_msg, url=url)
            
        except Exception as e:
            error_msg = f"Error fetching {url}: {type(e).__name__}: {str(e)}"
            logger.error(error_msg)
            return ScrapingResult(success=False, error=error_msg, url=url)


class URLDiscovery:
    """Get URL for LoC and Gutenberg with special case handling"""
    
    SPECIAL_CASES = {
        "gettysburg-address": {
            "url_pattern": r"exhibits/gettysburg-address",
            "method": "exhibit_html",
            "doc_type": "Speech"
        },
        "mal4361800": {
            "url_pattern": r"mal\.?4361800",
            "method": "resource_download",
            "doc_type": "Speech"
        }
    }
    
    def __init__(self, base_scraper: BaseScraper):
        self.scraper = base_scraper
    
    def is_special_case(self, url: str) -> Optional[str]:
        """Check if URL requires special handling (not standard XML)"""
        for case_key, case_info in self.SPECIAL_CASES.items():
            if re.search(case_info["url_pattern"], url, re.IGNORECASE):
                logger.info(f"URL matched special case: {case_key}")
                return case_key
        return None
    
    def handle_special_case(self, url: str, case_key: str) -> Optional[Dict]:
        """Handle special case URL and return parsed content"""
        case_info = self.SPECIAL_CASES.get(case_key)
        if not case_info:
            return None
        
        method = case_info["method"]
        logger.info(f"Handling special case '{case_key}' using method: {method}")
        
        if method == "exhibit_html":
            return self._extract_exhibit_transcription(url)
        elif method == "resource_download":
            return self._extract_resource_content(url)
        
        return None
    
    def scrape_special_document(self, url: str, document_info: Dict) -> Optional[DocumentData]:
        """
        Full scraping method for special cases that returns DocumentData
        
        Args:
            url: The LoC URL
            document_info: Dict with 'name' and 'type' keys
            
        Returns:
            DocumentData or None
        """
        case_key = self.is_special_case(url)
        if not case_key:
            logger.warning(f"URL is not a recognized special case: {url}")
            return None
        
        result = self.handle_special_case(url, case_key)
        
        if not result or not result.get("parsing_success"):
            logger.error(f"Special case handling failed for: {url}")
            return None
        
        metadata = result.get("metadata", {})
        
        doc_data = DocumentData(
            id=self._generate_special_id(url),
            title=metadata.get('title', document_info.get('name', 'Unknown')),
            reference=url,
            document_type=document_info.get('type', 'Document'),
            date=metadata.get('date'),
            place=metadata.get('place'),
            from_person='Abraham Lincoln',
            to_person=None,
            content=result.get('content', ''),
            source='loc'
        )
        
        logger.info(f"Created DocumentData: {doc_data.title} ({len(doc_data.content)} chars)")
        return doc_data
    
    def _extract_exhibit_transcription(self, url: str) -> Optional[Dict]:
        """Extract transcription from LoC exhibit HTML page"""
        logger.info(f"Extracting exhibit transcription from: {url}")
        
        result = self.scraper.fetch_url(url)
        if not result.success:
            logger.error(f"Failed to fetch exhibit page: {result.error}")
            return None
        
        soup = BeautifulSoup(result.content, 'html.parser')
        
        # Find tag <h2>Transcription</h2>
        h2_transcription = soup.find('h2', string=re.compile(r'^Transcription$', re.I))
        if not h2_transcription:
            h2_transcription = soup.find('h2', string=re.compile(r'Transcription', re.I))
        
        if not h2_transcription:
            logger.error("Could not find <h2>Transcription</h2> in HTML")
            return None
        
        logger.info("Found <h2>Transcription</h2> element")
        
        # Get parent div (col2_equal)
        parent_div = h2_transcription.find_parent('div', class_='col2_equal')
        if not parent_div:
            parent_div = h2_transcription.find_parent('div')
        
        paragraphs = []
        
        # Extract <p> tags
        if parent_div:
            for p_tag in parent_div.find_all('p'):
                text = p_tag.get_text(separator=' ', strip=True)
                
                # Skip editorial notes
                if text.startswith('(') or 'Differences between' in text:
                    continue
                
                if text:
                    paragraphs.append(text)
        
        # Fallback: sibling <p> tags
        if not paragraphs:
            for sibling in h2_transcription.find_next_siblings('p'):
                text = sibling.get_text(separator=' ', strip=True)
                if text and not text.startswith('('):
                    paragraphs.append(text)
        
        if not paragraphs:
            logger.error("No paragraphs found")
            return None
        
        content = '\n\n'.join(paragraphs)
        title = self._extract_page_title(soup, "Gettysburg Address")
        
        logger.info(f"Extracted {len(paragraphs)} paragraphs ({len(content)} chars)")
        
        return {
            "parsing_success": True,
            "content": content,
            "metadata": {
                "title": title,
                "date": "November 19, 1863",
                "place": "Gettysburg, Pennsylvania",
                "source_url": url,
                "extraction_method": "html_h2_transcription_p_tags",
                "paragraphs_extracted": len(paragraphs)
            }
        }
    
    def _extract_resource_content(self, url: str) -> Optional[Dict]:
        """Extract content from LoC resource page via download form"""
        logger.info(f"Extracting resource content from: {url}")
        
        # Normalize URL
        base_url = re.sub(r'\?.*$', '', url)
        
        result = self.scraper.fetch_url(base_url)
        
        if not result.success:
            logger.warning(f"Resource page fetch failed, trying constructed URLs")
            return self._try_constructed_download_urls(url)
        
        soup = BeautifulSoup(result.content, 'html.parser')
        
        # Find download <select> tag
        download_select = soup.find('select', {'id': 'download'})
        if not download_select:
            download_select = soup.find('select', {'name': 'download'})
        
        download_urls = []
        
        if download_select:
            logger.info("Found download <select> element")
            
            for option in download_select.find_all('option'):
                value = option.get('value', '')
                data_type = option.get('data-file-download', '').lower()
                label = option.get_text(strip=True).lower()
                
                if value.startswith('http'):
                    url_info = {
                        'url': value,
                        'is_txt': '.txt' in value or 'text' in data_type or 'text' in label,
                        'is_complete': 'complete' in label,
                    }
                    download_urls.append(url_info)
                    logger.info(f"Found download: {value}")
        
        # Try download URLs (prioritize TXT, then complete)
        if download_urls:
            download_urls.sort(key=lambda x: (not x['is_txt'], not x['is_complete']))
            
            for url_info in download_urls:
                content_result = self._fetch_and_parse_download(url_info['url'])
                if content_result:
                    content_result['metadata']['source_url'] = url
                    content_result['metadata']['title'] = self._extract_page_title(soup, "Last Public Address")
                    return content_result
        
        # Fallback to constructed URLs
        return self._try_constructed_download_urls(url)
    
    def _try_constructed_download_urls(self, url: str) -> Optional[Dict]:
        """Construct download URLs from resource ID pattern"""
        match = re.search(r'mal\.?(\d+)', url)
        if not match:
            logger.error("Could not extract resource ID")
            return None
        
        resource_id = match.group(1)
        prefix = resource_id[:3]
        
        logger.info(f"Constructing URLs for resource: {resource_id}")
        
        # PreDEfined Known LoC URL patterns
        constructed_urls = [
            f"https://tile.loc.gov/storage-services/service/gdc/gdccrowd/mss/mal/{prefix}/{resource_id}/{resource_id}.txt",
            f"https://tile.loc.gov/storage-services/service/gdc/gdccrowd/mss/mal/{prefix}/{resource_id}/001.txt",
            f"https://tile.loc.gov/storage-services/service/mss/mal/{prefix}/{resource_id}/{resource_id}.xml",
        ]
        
        for download_url in constructed_urls:
            logger.info(f"Trying: {download_url}")
            content_result = self._fetch_and_parse_download(download_url)
            if content_result:
                content_result['metadata']['source_url'] = url
                content_result['metadata']['url_constructed'] = True
                content_result['metadata']['title'] = "Last Public Address"
                content_result['metadata']['date'] = "April 11, 1865"
                content_result['metadata']['place'] = "Washington, D.C."
                return content_result
        
        logger.error("All constructed URLs failed")
        return None
    
    def _fetch_and_parse_download(self, download_url: str) -> Optional[Dict]:
        """Fetch and parse content from download URL"""
        result = self.scraper.fetch_url(download_url)
        if not result.success:
            return None
        
        content = result.content
        file_type = "txt"
        
        if '.xml' in download_url.lower():
            file_type = "xml"
            content = self._parse_xml_to_text(content)
        
        content = self._clean_text_content(content)
        
        if content and len(content) > 50:
            logger.info(f"Extracted {len(content)} chars from {download_url}")
            return {
                "parsing_success": True,
                "content": content,
                "metadata": {
                    "download_url": download_url,
                    "extraction_method": f"resource_download_{file_type}",
                    "file_type": file_type
                }
            }
        return None
    
    def _parse_xml_to_text(self, xml_content: str) -> str:
        """Extract text from XML"""
        try:
            soup = BeautifulSoup(xml_content, 'lxml-xml')
            for tag in ['text', 'body', 'div']:
                elem = soup.find(tag)
                if elem:
                    return elem.get_text(separator='\n', strip=True)
            return soup.get_text(separator='\n', strip=True)
        except:
            return xml_content
    
    def _clean_text_content(self, content: str) -> str:
        """Clean extracted text"""
        if not content:
            return ""
        content = re.sub(r'\r\n', '\n', content)
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        lines = [line.strip() for line in content.split('\n')]
        while lines and not lines[0]:
            lines.pop(0)
        while lines and not lines[-1]:
            lines.pop()
        return '\n'.join(lines)
    
    def _extract_page_title(self, soup: BeautifulSoup, default: str) -> str:
        """Extract page title"""
        h1 = soup.find('h1')
        if h1:
            return h1.get_text(strip=True)
        title_tag = soup.find('title')
        if title_tag:
            return title_tag.get_text(strip=True).split('|')[0].strip()
        return default
    
    def _generate_special_id(self, url: str) -> str:
        """Generate ID for special case documents"""
        if 'gettysburg' in url.lower():
            return "loc_gettysburg_address"
        match = re.search(r'mal\.?(\d+)', url)
        if match:
            return f"loc_mal_{match.group(1)}"
        return f"loc_{abs(hash(url)) % 100000:05d}"
    
    def find_loc_xml_url(self, loc_url: str) -> Optional[str]:
        """Find XML download URL from LoC resource page"""
        logger.info(f"Searching for XML URL from: {loc_url}")
       
        constructed_url = self._construct_loc_xml_url_direct(loc_url)
        if constructed_url:
            return constructed_url
        
        # Fallback :
        result = self.scraper.fetch_url(loc_url)
        if result.success:
            soup = BeautifulSoup(result.content, 'html.parser')
            xml_urls = self._extract_xml_urls_from_page(soup, result.metadata['final_url'])
            if xml_urls:
                return xml_urls[0]
        
        logger.warning(f"No XML URL found for {loc_url}")
        return None
    
    def _construct_loc_xml_url_direct(self, loc_url: str) -> Optional[str]:
        """Construct XML URL directly from known patterns"""
        
        # Pattern: /item/mal0440500/
        item_match = re.search(r'/item/(mal\d+)', loc_url)
        if item_match:
            mal_id = item_match.group(1)
            numeric_part = mal_id.replace('mal', '')
            
            if len(numeric_part) >= 7:
                dir1 = numeric_part[:3]
                xml_url = f"https://tile.loc.gov/storage-services/service/mss/mal/{dir1}/{numeric_part}/{numeric_part}.xml"
                logger.info(f"Constructed XML URL: {xml_url}")
                
                test_result = self.scraper.fetch_url(xml_url)
                if test_result.success:
                    return xml_url
        
        # search for Pattern: /resource/mal.0882800
        resource_match = re.search(r'/resource/mal\.(\d+)', loc_url)
        if resource_match:
            numeric_part = resource_match.group(1)
            
            potential_urls = []
            if len(numeric_part) >= 7:
                dir1 = numeric_part[:3]
                potential_urls.append(
                    f"https://tile.loc.gov/storage-services/service/mss/mal/{dir1}/{numeric_part}/{numeric_part}.xml"
                )
            
            if len(numeric_part) == 6:
                numeric_part = '0' + numeric_part
                dir1 = numeric_part[:3]
                potential_urls.append(
                    f"https://tile.loc.gov/storage-services/service/mss/mal/{dir1}/{numeric_part}/{numeric_part}.xml"
                )
            
            for test_url in potential_urls:
                test_result = self.scraper.fetch_url(test_url)
                if test_result.success and 'xml' in test_result.content[:1000].lower():
                    return test_url
        
        return None
    
    def _extract_xml_urls_from_page(self, soup: BeautifulSoup, base_url: str) -> List[str]:
        """Extract XML URLs from HTML page"""
        xml_urls = []
        
        # Download forms
        forms = soup.find_all('form', class_='resource-download-form')
        for form in forms:
            for option in form.find_all('option'):
                value = option.get('value', '')
                if '.xml' in value or 'format=xml' in value:
                    xml_urls.append(value if value.startswith('http') else urljoin(base_url, value))
        
        # Direct links
        for link in soup.find_all('a', href=True):
            href = link['href']
            if '.xml' in href and ('tile.loc.gov' in href or '/storage-services/' in href):
                xml_urls.append(href if href.startswith('http') else urljoin(base_url, href))
        
        return list(set(xml_urls))
    
    def find_gutenberg_text_url(self, gutenberg_url: str) -> Optional[str]:
       
        logger.info(f"Searching for text URL from: {gutenberg_url}")
        
        result = self.scraper.fetch_url(gutenberg_url)
        if not result.success:
            return None
            
        soup = BeautifulSoup(result.content, 'html.parser')
        
        download_links = soup.find_all('a', class_='link', href=True)
        for link in download_links:
            if 'Plain Text UTF-8' in link.get_text():
                txt_url = urljoin(result.metadata['final_url'], link['href'])
                logger.info(f"Found text URL: {txt_url}")
                return txt_url
        
        # Fallback: construct from book ID
        id_match = re.search(r'/ebooks?/(\d+)', gutenberg_url)
        if id_match:
            book_id = id_match.group(1)
            potential_urls = [
                f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8",
                f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt",
            ]
            
            for test_url in potential_urls:
                test_result = self.scraper.fetch_url(test_url)
                if test_result.success:
                    return test_url
        
        return None