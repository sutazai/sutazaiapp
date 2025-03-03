#!/usr/bin/env python3.11
"""Document Processor Module

This module provides the DocumentProcessor agent for processing various document types.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from loguru import logger

from ai_agents.base_agent import BaseAgent
from typing import Union
from typing import Optional


class DocumentProcessor(BaseAgent):
    """Document processing agent that handles various document types.
    
    Supported document types:
    - PDF
    - DOCX
    - TXT
    - Images (diagrams)
    """
    
    def __init__(
        self,
        name: str | None = None,
        config: Optional[Dict[str, Any]] = None,
    ):
        """Initialize the document processor agent.
        
        Args:
            name: Optional name for the agent
            config: Optional configuration dictionary
        """
        super().__init__(name=name, config=config)
        
        # Set up document processing directories
        self.input_dir = Path(self.config.get("input_dir", "data/input"))
        self.output_dir = Path(self.config.get("output_dir", "data/output"))
        self.temp_dir = Path(self.config.get("temp_dir", "data/temp"))
        
        # Create directories if they don't exist
        for directory in [self.input_dir, self.output_dir, self.temp_dir]:
            directory.mkdir(parents=True, exist_ok=True)
            
        # Initialize supported file types
        self.supported_types = {
            ".pdf": self._process_pdf,
            ".docx": self._process_docx,
            ".txt": self._process_text,
            ".png": self._process_diagram,
            ".jpg": self._process_diagram,
            ".jpeg": self._process_diagram,
        }
        
    def initialize(self) -> bool:
        """Initialize the document processor.
        
        Returns:
            bool: Whether initialization was successful
        """
        try:
            # Verify directories are writable
            for directory in [self.input_dir, self.output_dir, self.temp_dir]:
                test_file = directory / ".test"
                test_file.touch()
                test_file.unlink()
                
            # Initialize any required resources
            self._initialize_resources()
            
            self.logger.info("Document processor initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize document processor: {e}")
            return False
            
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a document processing task.
        
        Args:
            task: Dictionary containing task details
                Required fields:
                - file_path: Path to the document to process
                - output_format: Desired output format
                
        Returns:
            Dict containing processing results
        """
        try:
            file_path = Path(task["file_path"])
            if not file_path.exists():
                raise FileNotFoundError(f"File not found: {file_path}")
                
            file_type = file_path.suffix.lower()
            if file_type not in self.supported_types:
                raise ValueError(f"Unsupported file type: {file_type}")
                
            # Process the document using the appropriate handler
            processor = self.supported_types[file_type]
            result = processor(file_path)
            
            self.logger.info(f"Successfully processed {file_path}")
            return {
                "status": "success",
                "file_path": str(file_path),
                "result": result,
            }
            
        except Exception as e:
            self.logger.error(f"Failed to process document: {e}")
            return {
                "status": "error",
                "error": str(e),
            }
            
    def _initialize_resources(self) -> None:
        """Initialize required resources for document processing."""
        # Import optional dependencies only when needed
        try:
            import fitz  # PyMuPDF
            import python_docx  # python-docx
            import cv2
            import numpy as np
            
            self.logger.info("Successfully loaded document processing dependencies")
            
        except ImportError as e:
            self.logger.warning(f"Optional dependency not available: {e}")
            
    def _process_pdf(self, file_path: Path) -> Dict[str, Any]:
        """Process a PDF document.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Dict containing extracted content and metadata
        """
        import fitz  # PyMuPDF
        
        try:
            # Convert Path to string if needed
            file_path_str = str(file_path)
            doc = fitz.open(file_path_str)
            pages = []
            
            for page_num in range(len(doc)):
                page = doc[page_num]
                text = page.get_text()
                pages.append({
                    "page_number": page_num + 1,
                    "text": text,
                    "num_words": len(text.split()),
                    "num_characters": len(text),
                })
                
            # Save parsed content
            output_file = self.output_dir / f"{file_path.stem}_parsed.json"
            result = {
                "filename": file_path.name,
                "total_pages": len(doc),
                "pages": pages,
            }
            
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
                
            self.logger.info(f"PDF parsed successfully: {file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"PDF parsing error: {e}")
            return {"error": str(e)}
            
    def _process_docx(self, file_path: Path) -> Dict[str, Any]:
        """Process a DOCX document.
        
        Args:
            file_path: Path to the DOCX file
            
        Returns:
            Dict containing extracted content and metadata
        """
        from docx import Document
        
        try:
            doc = Document(file_path)
            text = "\n".join(paragraph.text for paragraph in doc.paragraphs)
            
            # Basic text analysis
            paragraphs = text.split("\n\n")
            
            # Save parsed content
            output_file = self.output_dir / f"{file_path.stem}_parsed.json"
            result = {
                "filename": file_path.name,
                "total_paragraphs": len(paragraphs),
                "num_words": len(text.split()),
                "num_characters": len(text),
                "content": text,
                "paragraphs": paragraphs,
            }
            
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
                
            self.logger.info(f"DOCX parsed successfully: {file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"DOCX parsing error: {e}")
            return {"error": str(e)}
            
    def _process_text(self, file_path: Path) -> Dict[str, Any]:
        """Process a text document.
        
        Args:
            file_path: Path to the text file
            
        Returns:
            Dict containing extracted content and metadata
        """
        try:
            with open(file_path, "r") as f:
                text = f.read()
                
            # Basic text analysis
            lines = text.split("\n")
            paragraphs = text.split("\n\n")
            
            result = {
                "filename": file_path.name,
                "num_lines": len(lines),
                "num_paragraphs": len(paragraphs),
                "num_words": len(text.split()),
                "num_characters": len(text),
                "content": text,
            }
            
            # Save parsed content
            output_file = self.output_dir / f"{file_path.stem}_parsed.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
                
            self.logger.info(f"Text file parsed successfully: {file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Text parsing error: {e}")
            return {"error": str(e)}
            
    def _process_diagram(self, file_path: Path) -> Dict[str, Any]:
        """Process an image/diagram.
        
        Args:
            file_path: Path to the image file
            
        Returns:
            Dict containing extracted content and metadata
        """
        import cv2
        import numpy as np
        
        try:
            # Read image
            image = cv2.imread(str(file_path))
            
            # Detect contours
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            
            # Extract entities (shapes, text regions, etc.)
            entities = []
            for contour in contours:
                if cv2.contourArea(contour) > 100:  # Filter small noise
                    x, y, w, h = cv2.boundingRect(contour)
                    entities.append({
                        "type": "region",
                        "x": int(x),
                        "y": int(y),
                        "width": int(w),
                        "height": int(h),
                        "area": float(cv2.contourArea(contour)),
                    })
                    
            # Save analysis results
            output_file = self.output_dir / f"{file_path.stem}_analysis.json"
            result = {
                "filename": file_path.name,
                "image_dimensions": {
                    "width": image.shape[1],
                    "height": image.shape[0],
                    "channels": image.shape[2],
                },
                "entities": entities,
                "total_entities": len(entities),
            }
            
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
                
            self.logger.info(f"Diagram analyzed successfully: {file_path}")
            return result
            
        except Exception as e:
            self.logger.error(f"Diagram analysis error: {e}")
            return {"error": str(e)}


def main():
    """Example usage of the DocumentProcessor agent."""
    # Create and initialize the agent
    processor = DocumentProcessor()
    if not processor.initialize():
        logger.error("Failed to initialize document processor")
        return
        
    # Process a sample document
    task = {
        "file_path": "data/sample.pdf",
        "output_format": "json",
    }
    
    result = processor.execute(task)
    logger.info(f"Processing result: {result}")


if __name__ == "__main__":
    main()