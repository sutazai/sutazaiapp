import os
import json
import logging
from typing import Dict, Any, Optional

import fitz  # PyMuPDF
import docx2txt
from loguru import logger

class DocumentParser:
    """
    Offline document parsing module for PDF and DOCX files
    """
    
    def __init__(self, 
                 output_dir: str = "/opt/sutazaiapp/doc_data/parsed",
                 max_file_size_mb: int = 50):
        """
        Initialize DocumentParser
        
        Args:
            output_dir (str): Directory to save parsed documents
            max_file_size_mb (int): Maximum allowed file size in MB
        """
        self.output_dir = output_dir
        self.max_file_size_mb = max_file_size_mb
        
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            filename=os.path.join(output_dir, "doc_parsing.log"),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s: %(message)s'
        )
    
    def _validate_file(self, file_path: str) -> bool:
        """
        Validate file before parsing
        
        Args:
            file_path (str): Path to the file
        
        Returns:
            bool: Whether file is valid for parsing
        """
        # Check file existence
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return False
        
        # Check file size
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.max_file_size_mb:
            logger.error(f"File too large: {file_path} ({file_size_mb} MB)")
            return False
        
        return True
    
    def parse_pdf(self, file_path: str) -> Dict[str, Any]:
        """
        Parse PDF file and extract text
        
        Args:
            file_path (str): Path to PDF file
        
        Returns:
            Dict containing parsed document information
        """
        if not self._validate_file(file_path):
            return {"error": "Invalid file"}
        
        try:
            doc = fitz.open(file_path)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                pages.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "num_words": len(text.split()),
                    "num_characters": len(text)
                })
            
            # Save parsed content
            output_file = os.path.join(
                self.output_dir, 
                f"{os.path.basename(file_path)}_parsed.json"
            )
            
            result = {
                "filename": os.path.basename(file_path),
                "total_pages": len(doc),
                "pages": pages
            }
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"PDF parsed successfully: {file_path}")
            return result
        
        except Exception as e:
            logger.exception(f"PDF parsing error: {e}")
            return {"error": str(e)}
    
    def parse_docx(self, file_path: str) -> Dict[str, Any]:
        """
        Parse DOCX file and extract text
        
        Args:
            file_path (str): Path to DOCX file
        
        Returns:
            Dict containing parsed document information
        """
        if not self._validate_file(file_path):
            return {"error": "Invalid file"}
        
        try:
            text = docx2txt.process(file_path)
            
            # Basic text analysis
            paragraphs = text.split('\n\n')
            
            # Save parsed content
            output_file = os.path.join(
                self.output_dir, 
                f"{os.path.basename(file_path)}_parsed.json"
            )
            
            result = {
                "filename": os.path.basename(file_path),
                "total_paragraphs": len(paragraphs),
                "num_words": len(text.split()),
                "num_characters": len(text),
                "content": text,
                "paragraphs": paragraphs
            }
            
            with open(output_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            logger.info(f"DOCX parsed successfully: {file_path}")
            return result
        
        except Exception as e:
            logger.exception(f"DOCX parsing error: {e}")
            return {"error": str(e)}

def main():
    """
    Example usage and testing
    """
    parser = DocumentParser()
    
    # Example PDF parsing
    pdf_result = parser.parse_pdf("/path/to/sample.pdf")
    print(json.dumps(pdf_result, indent=2))
    
    # Example DOCX parsing
    docx_result = parser.parse_docx("/path/to/sample.docx")
    print(json.dumps(docx_result, indent=2))

if __name__ == "__main__":
    main() 