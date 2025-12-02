import xml.etree.ElementTree as ET
from typing import Dict, Optional, List, Tuple
import re
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class XMLParser:
    """Parser for Library of Congress XML documents and other XML sources"""
    
    def __init__(self):
        # Common XML namespaces used by LoC
        self.namespaces = {
            'tei': 'http://www.tei-c.org/ns/1.0',
            'mods': 'http://www.loc.gov/mods/v3',
            'dc': 'http://purl.org/dc/elements/1.1/',
            'mets': 'http://www.loc.gov/METS/',
            'xlink': 'http://www.w3.org/1999/xlink'
        }
        
        # XPath patterns for different metadata fields
        self.metadata_paths = {
            'title': [
                ".//title",
                ".//titlestmt/title",
                ".//filedesc/titlestmt/title",
                ".//tei2/teiheader/filedesc/titlestmt/title",
                ".//mods:title",
                ".//dc:title"
            ],
            'date': [
                ".//date",
                ".//publicationstmt/date", 
                ".//profiledesc/creation/date",
                ".//creation/date",
                ".//mods:dateCreated",
                ".//dc:date"
            ],
            'place': [
                ".//placename",
                ".//place",
                ".//profiledesc/creation/placename",
                ".//creation/placename",
                ".//mods:place",
                ".//dc:coverage"
            ],
            'from_person': [
                ".//persname[@type='sender']",
                ".//name[@type='sender']", 
                ".//author",
                ".//creator",
                ".//mods:name[@type='personal']",
                ".//dc:creator"
            ],
            'to_person': [
                ".//persname[@type='addressee']",
                ".//persname[@type='recipient']",
                ".//name[@type='recipient']",
                ".//name[@type='addressee']"
            ]
        }
        
        # XPath patterns for content extraction
        self.content_paths = [
            ".//text",
            ".//body",
            ".//div[@type='letter']",
            ".//div[@type='speech']", 
            ".//div[@type='document']",
            ".//div",
            ".//p"
        ]
    
    def parse_xml_content(self, xml_content: str, source_url: str = "") -> Dict:
        """Parse XML content and extract metadata and text"""
        
        if not xml_content:
            return {"error": "Empty XML content provided"}
        
        try:
           
            if isinstance(xml_content, bytes):
                xml_content = xml_content.decode('utf-8', errors='ignore')
            
            # Parse XML
            root = ET.fromstring(xml_content)
            
          
            metadata = self.extract_metadata_from_xml(root)
            
            content = self.extract_content_from_xml(root)
            
            
            structure_info = self.analyze_xml_structure(root)
            
            result = {
                "metadata": metadata,
                "content": content,
                "structure": structure_info,
                "source_url": source_url,
                "parsing_success": True
            }
            
            logger.info(f"Successfully parsed XML: {len(content)} chars content, {len(metadata)} metadata fields")
            return result
            
        except ET.ParseError as e:
            logger.error(f"XML parsing error: {str(e)}")
            return {
                "error": f"XML parsing failed: {str(e)}",
                "parsing_success": False
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing XML: {str(e)}")
            return {
                "error": f"XML processing failed: {str(e)}",
                "parsing_success": False
            }
    
    def extract_metadata_from_xml(self, root: ET.Element) -> Dict:
        """Extract metadata from XML root element"""
        metadata = {}
        
        
        for field, path_list in self.metadata_paths.items():
            value = self.extract_field_with_fallbacks(root, path_list)
            if value:
                metadata[field] = value
        
        
        if 'date' not in metadata or not metadata['date']:
            date_value = self.extract_date_with_attributes(root)
            if date_value:
                metadata['date'] = date_value
        
        
        additional_metadata = self.extract_loc_specific_metadata(root)
        metadata.update(additional_metadata)
      
        cleaned_metadata = {}
        for key, value in metadata.items():
            if value and isinstance(value, str):
                cleaned_value = self.clean_metadata_value(value)
                if cleaned_value:
                    cleaned_metadata[key] = cleaned_value
        
        return cleaned_metadata
    
    def extract_content_from_xml(self, root: ET.Element) -> str:
        """Extract main text content from XML"""
        content_parts = []
        
      
        for path in self.content_paths:
            elements = root.findall(path)
            if elements:
                for elem in elements:
                    text = self.extract_text_from_element(elem)
                    if text and text.strip():
                        content_parts.append(text.strip())
                
               
                if content_parts:
                    break
        
       
        if not content_parts:
            all_text = self.extract_all_text_from_xml(root)
            if all_text:
                content_parts.append(all_text)
        
        
        full_content = '\n\n'.join(content_parts) if content_parts else ''
        
        
        cleaned_content = self.clean_xml_content(full_content)
        
        return cleaned_content
    
    def extract_text_from_element(self, element: ET.Element) -> str:
        
        text_parts = []
        
        
        if element.text:
            text_parts.append(element.text)
        
        
        for child in element:
            child_text = self.extract_text_from_element(child)
            if child_text:
                text_parts.append(child_text)
            
            
            if child.tail:
                text_parts.append(child.tail)
        
        # Join and clean
        result = ' '.join(text_parts)
        return result.strip()
    
    def extract_all_text_from_xml(self, root: ET.Element) -> str:
       
        all_text_parts = []
        
        for elem in root.iter():
            if elem.text and elem.text.strip():
                
                if elem.tag.lower() not in ['title', 'author', 'date', 'creator']:
                    all_text_parts.append(elem.text.strip())
        
        return ' '.join(all_text_parts)
    
    def extract_field_with_fallbacks(self, root: ET.Element, path_list: List[str]) -> Optional[str]:
      
        for path in path_list:
            try:
                # Try with and without namespaces
                element = root.find(path)
                if element is not None and element.text:
                    return element.text.strip()
                
                # Try with namespaces
                for ns_prefix, ns_uri in self.namespaces.items():
                    namespaced_path = path.replace(f"{ns_prefix}:", f"{{{ns_uri}}}")
                    element = root.find(namespaced_path)
                    if element is not None and element.text:
                        return element.text.strip()
                        
            except Exception as e:
                logger.debug(f"Failed to extract with path {path}: {str(e)}")
                continue
        
        return None
    
    def extract_date_with_attributes(self, root: ET.Element) -> Optional[str]:
       
        date_elements = root.findall(".//date") + root.findall(".//creation/date")
        
        for elem in date_elements:
            # Check common date attributes
            for attr in ['when', 'value', 'notBefore', 'notAfter']:
                if attr in elem.attrib:
                    date_value = elem.attrib[attr]
                    if date_value:
                        return date_value.strip()
        
        return None
    
    def extract_loc_specific_metadata(self, root: ET.Element) -> Dict:
        
        loc_metadata = {}
        
        collection_elem = root.find(".//collection")
        if collection_elem is not None and collection_elem.text:
            loc_metadata['collection'] = collection_elem.text.strip()
        
     
        id_elements = root.findall(".//identifier") + root.findall(".//idno")
        for elem in id_elements:
            if elem.text:
                id_type = elem.get('type', 'identifier')
                loc_metadata[f'loc_{id_type}'] = elem.text.strip()
        
       
        lang_elem = root.find(".//language")
        if lang_elem is not None and lang_elem.text:
            loc_metadata['language'] = lang_elem.text.strip()
        
       
        repo_elem = root.find(".//repository")
        if repo_elem is not None and repo_elem.text:
            loc_metadata['repository'] = repo_elem.text.strip()
        
        return loc_metadata
    
    def clean_metadata_value(self, value: str) -> str:
        """Clean and normalize metadata values"""
        if not value:
            return ""
        
        
        cleaned = ' '.join(value.split())
        
        # Remove common XML artifacts
        cleaned = re.sub(r'[\[\]<>]', '', cleaned)
        
        
        if len(cleaned) > 500:
            cleaned = cleaned[:500] + "..."
        
        return cleaned.strip()
    
    def clean_xml_content(self, content: str) -> str:
   
        if not content:
            return ""
        
        # Normalize whitespace
        content = re.sub(r'\s+', ' ', content)
        
       
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'&[a-zA-Z0-9]+;', ' ', content)
        
      
        content = re.sub(r'(\.)(\s*)([A-Z])', r'\1\n\n\3', content)
        
        # Clean up excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        content = re.sub(r'[ \t]+', ' ', content)
        
        return content.strip()
    
    def analyze_xml_structure(self, root: ET.Element) -> Dict:
        
        structure = {
            "root_tag": root.tag,
            "namespace": root.tag.split('}')[0].strip('{') if '}' in root.tag else None,
            "element_count": len(list(root.iter())),
            "unique_tags": list(set(elem.tag for elem in root.iter())),
            "has_text_content": bool(root.text and root.text.strip()),
            "attributes": dict(root.attrib) if root.attrib else {}
        }
        
        
        tag_counts = {}
        for elem in root.iter():
            tag_name = elem.tag.split('}')[-1] if '}' in elem.tag else elem.tag
            tag_counts[tag_name] = tag_counts.get(tag_name, 0) + 1
        
        structure["tag_counts"] = tag_counts
        
        return structure
    
    def validate_xml_structure(self, xml_content: str) -> Tuple[bool, List[str]]:
       
        issues = []
        
        if not xml_content:
            return False, ["Empty XML content"]
        
        try:
            root = ET.fromstring(xml_content)
            
            
            if len(list(root.iter())) < 2:
                issues.append("XML has very few elements")
            
           
            has_text = any(elem.text and elem.text.strip() for elem in root.iter())
            if not has_text:
                issues.append("No text content found in XML")
            
          
            common_elements = ['title', 'date', 'text', 'body']
            found_elements = set(elem.tag.split('}')[-1].lower() for elem in root.iter())
            
            if not any(elem in found_elements for elem in common_elements):
                issues.append("No common document elements found")
            
            is_valid = len(issues) == 0
            return is_valid, issues
            
        except ET.ParseError as e:
            return False, [f"XML parsing error: {str(e)}"]
        except Exception as e:
            return False, [f"Validation error: {str(e)}"]

def test_xml_parser():
    
    sample_xml = """<?xml version="1.0" encoding="UTF-8"?>
    <tei2>
        <teiheader>
            <filedesc>
                <titlestmt>
                    <title>Abraham Lincoln letter to Truman Smith, Saturday, November 10, 1860</title>
                </titlestmt>
                <publicationstmt>
                    <date>1999-06-05</date>
                </publicationstmt>
            </filedesc>
            <profiledesc>
                <creation>
                    <date when="1860-11-10">November 10, 1860</date>
                    <placename>Springfield, Illinois</placename>
                </creation>
            </profiledesc>
        </teiheader>
        <text>
            <body>
                <div type="letter">
                    <p>Dear Sir,</p>
                    <p>I acknowledge receipt of your letter regarding the election results. The matter you mention requires careful consideration.</p>
                    <p>Very respectfully yours,</p>
                    <p>A. Lincoln</p>
                </div>
            </body>
        </text>
    </tei2>"""
    
    parser = XMLParser()
    
    
    result = parser.parse_xml_content(sample_xml)
    
    if result.get("parsing_success"):
        print("Metadata extracted:")
        for key, value in result.get("metadata", {}).items():
            print(f"  {key}: {value}")
        
        print(f"\nContent length: {len(result.get('content', ''))}")
        print("Content preview:")
        print(result.get('content', '')[:200] + "...")
        
        print(f"\nXML structure:")
        structure = result.get('structure', {})
        print(f"  Root tag: {structure.get('root_tag')}")
        print(f"  Element count: {structure.get('element_count')}")
        print(f"  Unique tags: {len(structure.get('unique_tags', []))}")
    else:
        print(f"Parsing failed: {result.get('error')}")
    
    # Test validation
    is_valid, issues = parser.validate_xml_structure(sample_xml)
    print(f"\nValidation: {'PASS' if is_valid else 'FAIL'}")
    if issues:
        print("Issues found:")
        for issue in issues:
            print(f"  - {issue}")

if __name__ == "__main__":
    test_xml_parser()