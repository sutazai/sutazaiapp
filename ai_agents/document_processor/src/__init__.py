#!/usr/bin/env python3.11
"""
Document Processor Agent for AI Agents System

This module provides document processing capabilities including:
- PDF text extraction
- Image-based text recognition (OCR)
- Document structure analysis
- Metadata extraction
"""

import logging
import os
from collections.abc import Iterator, Sequence
from typing import Any, Dict, List, Optional, Union, overload, cast

try:
    import cv2
    from cv2 import error as cv2_error
except ImportError:
    cv2 = None
    cv2_error = Exception

try:
    import fitz  # PyMuPDF
    FileDataError = fitz.FileDataError
except ImportError:
    fitz = None
    FileDataError = Exception

try:
    import numpy as np
except ImportError:
    np = None

try:
    import pytesseract
except ImportError:
    pytesseract = None

from loguru import logger
from ai_agents.base_agent import AgentError, BaseAgent
from ai_agents.exceptions import PDFExtractionError


class Document(Sequence[str]):
    """Document class representing a sequence of pages."""
    
    def __init__(self, pages: List[str]):
        """Initialize document with pages."""
        self.pages = pages

    def __iter__(self) -> Iterator[str]:
        """Return iterator over pages."""
        return iter(self.pages)

    @overload
    def __getitem__(self, index: int) -> str:
        """Get single page by index."""
        ...

    @overload
    def __getitem__(self, s: slice) -> List[str]:
        """Get multiple pages by slice."""
        ...

    def __getitem__(self, key: Union[int, slice]) -> Union[str, List[str]]:
        """Get page(s) by index or slice."""
        return self.pages[key]

    def __len__(self) -> int:
        """Return number of pages."""
        return len(self.pages)


class DocumentProcessorAgent(BaseAgent):
    """
    Advanced Document Processing Agent
    
    Capabilities:
    - PDF text extraction
    - Image-based text recognition
    - Document structure analysis  
    - Metadata extraction
    """
    
    def __init__(
        self,
        temp_dir: str = "/tmp/doc_processor",
        **kwargs,
    ):
        """
        Initialize Document Processor Agent
        
        Args:
            temp_dir (str): Temporary directory for processing documents
            **kwargs: Additional configuration parameters
        """
        super().__init__(agent_name="document_processor", **kwargs)
        
        # Temporary processing directory
        self.temp_dir = temp_dir
        os.makedirs(temp_dir, exist_ok=True)
        
        # Logging configuration
        logger.info("📄 Document Processor Agent initialized")

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute document processing task
        
        Args:
            task (Dict): Task details including:
                - document_path: Path to the document
                - operation: Operation to perform (extract_text, ocr, analyze)
                - parameters: Operation-specific parameters
                
        Returns:
            Dict containing operation results
        """
        try:
            document_path = task.get("document_path")
            operation = task.get("operation", "extract_text")
            parameters = task.get("parameters", {})
            
            if not document_path or not os.path.exists(document_path):
                return {
                    "status": "failed",
                    "error": f"Document not found: {document_path}",
                }
            
            result: Dict[str, Any] = {"status": "success"}
            
            if operation == "extract_text":
                pages = parameters.get("pages")
                result.update(self._extract_text(document_path, pages))
            elif operation == "ocr":
                languages = parameters.get("languages", ["eng"])
                result.update(self._ocr_processing(document_path, languages))
            elif operation == "analyze":
                analysis_type = parameters.get("analysis_type", "structure")
                result.update(self._document_analysis(document_path, analysis_type))
            else:
                result = {
                    "status": "failed",
                    "error": f"Unsupported operation: {operation}",
                }
            
            # Log performance
            self._log_performance(task, result)
            return result
            
        except Exception as e:
            error_result = {"status": "failed", "error": str(e)}
            self._log_performance(task, error_result)
            raise AgentError(
                f"Document processing failed: {e}",
                task=task,
            ) from e

    def _extract_text(
        self,
        document_path: str,
        pages: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Extract text from PDF documents
        
        Args:
            document_path (str): Path to the document
            pages (List[int], optional): Specific pages to extract
            
        Returns:
            Dict: Extracted text and metadata
        """
        if not fitz:
            return {"status": "failed", "error": "PyMuPDF not installed"}
        
        try:
            doc = fitz.open(document_path)
            
            # Page selection
            page_range = pages or range(len(doc))
            extracted_text = []
            
            for page_num in page_range:
                if page_num < len(doc):
                    page = doc[page_num]
                    extracted_text.append(page.get_text("text"))
            
            doc.close()
            
            return {
                "status": "success",
                "text": extracted_text,
                "total_pages": len(doc),
                "metadata": {
                    "file_path": document_path,
                    "extracted_pages": list(page_range),
                },
            }
            
        except Exception as e:
            logger.error("Text extraction failed: %s", e)
            return {"status": "failed", "error": str(e)}

    def _ocr_processing(
        self,
        image_path: str,
        languages: List[str] = ["eng"],
    ) -> Dict[str, Any]:
        """
        Perform OCR on an image
        
        Args:
            image_path (str): Path to the image
            languages (List[str]): OCR languages
            
        Returns:
            Dict: OCR processing results
        """
        if not cv2 or not pytesseract:
            return {"status": "failed", "error": "OpenCV or pytesseract not installed"}
        
        try:
            # Read image
            img = cv2.imread(image_path)
            
            if img is None:
                raise FileNotFoundError(f"Could not read image at {image_path}")
            
            # Preprocess image
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # OCR processing
            ocr_result = pytesseract.image_to_string(
                gray,
                lang="+".join(languages),
            )
            
            return {
                "status": "success",
                "text": ocr_result,
                "metadata": {"image_path": image_path, "languages": languages},
            }
            
        except Exception as e:
            logger.error("OCR processing error: %s", str(e))
            return {"status": "failed", "error": str(e)}

    def _document_analysis(
        self,
        document_path: str,
        analysis_type: str = "structure",
    ) -> Dict[str, Any]:
        """
        Perform advanced document analysis
        
        Args:
            document_path (str): Path to the document
            analysis_type (str): Type of analysis to perform
            
        Returns:
            Dict: Document analysis results
        """
        if not fitz:
            return {"status": "failed", "error": "PyMuPDF not installed"}
        
        try:
            doc = fitz.open(document_path)
            analysis_results: Dict[str, Any] = {
                "total_pages": len(doc),
                "text_blocks": [],
                "images": [],
                "tables": [],
            }
            
            for page_num, page in enumerate(doc):
                # Text block extraction
                text_dict = page.get_text("dict")
                
                if "blocks" in text_dict:
                    blocks = text_dict["blocks"]
                    text_blocks = []
                    
                    for block in blocks:
                        if block.get("type", -1) == 0:  # Text blocks have type 0
                            text_blocks.append({
                                "text": block.get("lines", []),
                                "bbox": block.get("bbox"),
                            })
                    
                    analysis_results["text_blocks"].extend(text_blocks)
                
                # Image detection
                try:
                    images = page.get_images()
                    image_info = []
                    
                    for img in images:
                        try:
                            bbox = page.get_image_bbox(img[0])
                            image_info.append({"xref": img[0], "bbox": bbox})
                        except Exception as img_err:
                            logger.warning("Error getting image bbox: %s", img_err)
                    
                    analysis_results["images"].extend(image_info)
                    
                except Exception as img_ex:
                    logger.warning(
                        "Error processing images on page %s: %s",
                        page_num,
                        img_ex,
                    )
            
            doc.close()
            
            return {
                "status": "success",
                "analysis": analysis_results,
                "metadata": {
                    "file_path": document_path,
                    "analysis_type": analysis_type,
                },
            }
            
        except Exception as e:
            logger.error("Document analysis failed: %s", e)
            return {"status": "failed", "error": str(e)}

    def _log_performance(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        """Log performance metrics for the task."""
        operation = task.get("operation", "unknown")
        status = result.get("status", "unknown")
        logger.info(f"Document processing - Operation: {operation}, Status: {status}")


def main():
    """Demonstration of Document Processor Agent"""
    agent = DocumentProcessorAgent()
    
    # Example tasks
    pdf_task = {
        "document_path": "/tmp/sample.pdf",
        "operation": "extract_text",
        "parameters": {"pages": [0, 1]},
    }
    
    ocr_task = {
        "document_path": "/tmp/sample_image.png",
        "operation": "ocr",
        "parameters": {"languages": ["eng", "fra"]},
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
