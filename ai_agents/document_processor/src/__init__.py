import os
import logging
from typing import Dict, Any, List, Optional
import fitz  # PyMuPDF
import cv2
import numpy as np
import pytesseract
from loguru import logger

from ai_agents.base_agent import BaseAgent, AgentError

class DocumentProcessorAgent(BaseAgent):
    """
    Advanced Document Processing Agent
    
    Capabilities:
    - PDF text extraction
    - Image-based text recognition
    - Document structure analysis
    - Metadata extraction
    """
    
    def __init__(self, 
                 temp_dir: str = '/opt/sutazai_project/SutazAI/doc_data/temp',
                 **kwargs):
        """
        Initialize Document Processor Agent
        
        Args:
            temp_dir (str): Temporary directory for processing documents
            **kwargs: Additional configuration parameters
        """
        super().__init__(agent_name='document_processor', **kwargs)
        
        # Temporary processing directory
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        
        # Logging configuration
        logger.info(f"📄 Document Processor Agent initialized")
    
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute document processing tasks
        
        Args:
            task (Dict): Processing task specification
        
        Returns:
            Dict: Processing results
        
        Raises:
            AgentError: For processing failures
        """
        try:
            task_type = task.get('type', 'extract_text')
            document_path = task.get('document')
            
            if not document_path:
                raise AgentError("No document path provided")
            
            # Task routing
            processing_methods = {
                'extract_text': self._extract_text,
                'ocr_processing': self._ocr_processing,
                'document_analysis': self._document_analysis
            }
            
            method = processing_methods.get(task_type)
            if not method:
                raise AgentError(f"Unsupported task type: {task_type}")
            
            result = method(document_path, **task.get('params', {}))
            
            # Log performance
            self._log_performance(task, result)
            
            return result
        
        except Exception as e:
            logger.error(f"Document processing error: {e}")
            raise AgentError(f"Document processing failed: {e}", task=task)
    
    def _extract_text(self, 
                      document_path: str, 
                      pages: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Extract text from PDF documents
        
        Args:
            document_path (str): Path to the document
            pages (List[int], optional): Specific pages to extract
        
        Returns:
            Dict: Extracted text and metadata
        """
        try:
            doc = fitz.open(document_path)
            
            # Page selection
            page_range = pages or range(len(doc))
            
            extracted_text = []
            for page_num in page_range:
                page = doc[page_num]
                extracted_text.append(page.get_text())
            
            return {
                'status': 'success',
                'text': extracted_text,
                'total_pages': len(doc),
                'metadata': {
                    'file_path': document_path,
                    'extracted_pages': list(page_range)
                }
            }
        
        except Exception as e:
            logger.error(f"Text extraction failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _ocr_processing(self, 
                        image_path: str, 
                        languages: List[str] = ['eng']) -> Dict[str, Any]:
        """
        Perform Optical Character Recognition (OCR)
        
        Args:
            image_path (str): Path to the image
            languages (List[str]): OCR languages
        
        Returns:
            Dict: OCR processing results
        """
        try:
            # Read image
            image = cv2.imread(image_path)
            
            # Preprocess image
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # OCR processing
            ocr_result = pytesseract.image_to_string(
                gray, 
                lang='+'.join(languages)
            )
            
            return {
                'status': 'success',
                'text': ocr_result,
                'metadata': {
                    'image_path': image_path,
                    'languages': languages
                }
            }
        
        except Exception as e:
            logger.error(f"OCR processing failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }
    
    def _document_analysis(self, 
                           document_path: str, 
                           analysis_type: str = 'structure') -> Dict[str, Any]:
        """
        Perform advanced document analysis
        
        Args:
            document_path (str): Path to the document
            analysis_type (str): Type of analysis to perform
        
        Returns:
            Dict: Document analysis results
        """
        try:
            doc = fitz.open(document_path)
            
            analysis_results = {
                'total_pages': len(doc),
                'text_blocks': [],
                'images': [],
                'tables': []
            }
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                
                # Text block extraction
                blocks = page.get_text('dict')['blocks']
                analysis_results['text_blocks'].extend([
                    {
                        'text': block.get('lines', []),
                        'bbox': block.get('bbox')
                    } for block in blocks if block['type'] == 0
                ])
                
                # Image detection
                images = page.get_images()
                analysis_results['images'].extend([
                    {
                        'xref': img[0],
                        'bbox': page.get_image_bbox(img[0])
                    } for img in images
                ])
            
            return {
                'status': 'success',
                'analysis': analysis_results,
                'metadata': {
                    'file_path': document_path,
                    'analysis_type': analysis_type
                }
            }
        
        except Exception as e:
            logger.error(f"Document analysis failed: {e}")
            return {
                'status': 'failed',
                'error': str(e)
            }

def main():
    """Demonstration of Document Processor Agent"""
    agent = DocumentProcessorAgent()
    
    # Example tasks
    pdf_task = {
        'type': 'extract_text',
        'document': '/opt/sutazai_project/SutazAI/doc_data/sample.pdf',
        'params': {'pages': [0, 1]}
    }
    
    ocr_task = {
        'type': 'ocr_processing',
        'document': '/opt/sutazai_project/SutazAI/doc_data/sample_image.png',
        'params': {'languages': ['eng', 'fra']}
    }
    
    try:
        pdf_result = agent.execute(pdf_task)
        print("PDF Extraction Result:", pdf_result)
        
        ocr_result = agent.execute(ocr_task)
        print("OCR Result:", ocr_result)
    
    except AgentError as e:
        print(f"Agent Error: {e}")

if __name__ == "__main__":
    main() 