"""
Document Agent Module

This module provides a specialized agent for document processing tasks.
It includes capabilities for text extraction, OCR, and document analysis.
"""

import logging
from typing import Dict, Any
from pathlib import Path
import pytesseract
from PIL import Image
import pdf2image
import numpy as np

from .base_agent import BaseAgent, AgentError

logger = logging.getLogger(__name__)


class DocumentAgent(BaseAgent):
    """Agent specialized for document processing tasks."""

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the document agent.

        Args:
            config: Agent configuration dictionary
        """
        super().__init__(config)
        self.supported_formats = ["pdf", "png", "jpg", "jpeg", "tiff"]
        self.tesseract_config = config.get("tesseract_config", {})
        self._check_dependencies()

    def _check_dependencies(self) -> None:
        """
        Check if required dependencies are available.

        Raises:
            AgentError: If dependencies are missing
        """
        try:
            # Check Tesseract
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.warning(f"Tesseract not available: {str(e)}")
            self.tesseract_available = False
        else:
            self.tesseract_available = True

        try:
            # Check pdf2image
            pdf2image.__version__
        except Exception as e:
            logger.warning(f"pdf2image not available: {str(e)}")
            self.pdf2image_available = False
        else:
            self.pdf2image_available = True

    def _initialize(self) -> None:
        """Initialize the document agent."""
        # Create temporary directory for processing
        self.temp_dir = Path("/tmp/document_agent")
        self.temp_dir.mkdir(parents=True, exist_ok=True)

        # Initialize OCR if available
        if self.tesseract_available:
            logger.info("Tesseract OCR initialized")

        # Initialize PDF processing if available
        if self.pdf2image_available:
            logger.info("PDF processing initialized")

    def _execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a document processing task."""
        task_type = task["type"]
        params = task["parameters"]

        if task_type == "extract_text":
            return self._extract_text(params)
        elif task_type == "analyze_document":
            return self._analyze_document(params)
        else:
            raise AgentError(f"Unsupported task type: {task_type}")

    def _cleanup(self) -> None:
        """Clean up document agent resources."""
        # Remove temporary files
        try:
            for file in self.temp_dir.glob("*"):
                file.unlink()
            self.temp_dir.rmdir()
        except Exception as e:
            logger.warning(f"Error cleaning up temporary files: {str(e)}")

    def _extract_text(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract text from a document.

        Args:
            params: Task parameters

        Returns:
            Dict[str, Any]: Extracted text and metadata

        Raises:
            AgentError: If text extraction fails
        """
        file_path = params.get("file_path")
        if not file_path:
            raise AgentError("Missing file_path parameter")

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise AgentError(f"File not found: {file_path}")

            # Get file extension
            ext = file_path.suffix.lower().lstrip(".")
            if ext not in self.supported_formats:
                raise AgentError(f"Unsupported file format: {ext}")

            # Process based on file type
            if ext == "pdf":
                return self._extract_text_from_pdf(file_path)
            else:
                return self._extract_text_from_image(file_path)

        except Exception as e:
            logger.error(f"Error extracting text: {str(e)}")
            raise AgentError(f"Failed to extract text: {str(e)}")

    def _extract_text_from_pdf(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from a PDF file.

        Args:
            file_path: Path to PDF file

        Returns:
            Dict[str, Any]: Extracted text and metadata
        """
        if not self.pdf2image_available:
            raise AgentError("PDF processing not available")

        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(file_path)

            # Extract text from each page
            pages = []
            for i, image in enumerate(images):
                if self.tesseract_available:
                    text = pytesseract.image_to_string(image, **self.tesseract_config)
                else:
                    text = ""  # No OCR available

                pages.append(
                    {
                        "page_number": i + 1,
                        "text": text,
                        "word_count": len(text.split()),
                    }
                )

            return {
                "file_path": str(file_path),
                "file_type": "pdf",
                "page_count": len(pages),
                "pages": pages,
                "total_word_count": sum(p["word_count"] for p in pages),
            }

        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise AgentError(f"Failed to process PDF: {str(e)}")

    def _extract_text_from_image(self, file_path: Path) -> Dict[str, Any]:
        """
        Extract text from an image file.

        Args:
            file_path: Path to image file

        Returns:
            Dict[str, Any]: Extracted text and metadata
        """
        if not self.tesseract_available:
            raise AgentError("OCR not available")

        try:
            # Open image
            image = Image.open(file_path)

            # Extract text
            text = pytesseract.image_to_string(image, **self.tesseract_config)

            return {
                "file_path": str(file_path),
                "file_type": file_path.suffix.lower().lstrip("."),
                "page_count": 1,
                "pages": [
                    {"page_number": 1, "text": text, "word_count": len(text.split())}
                ],
                "total_word_count": len(text.split()),
            }

        except Exception as e:
            logger.error(f"Error processing image: {str(e)}")
            raise AgentError(f"Failed to process image: {str(e)}")

    def _analyze_document(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a document for various features.

        Args:
            params: Task parameters

        Returns:
            Dict[str, Any]: Analysis results

        Raises:
            AgentError: If analysis fails
        """
        file_path = params.get("file_path")
        if not file_path:
            raise AgentError("Missing file_path parameter")

        try:
            file_path = Path(file_path)
            if not file_path.exists():
                raise AgentError(f"File not found: {file_path}")

            # Get file extension
            ext = file_path.suffix.lower().lstrip(".")
            if ext not in self.supported_formats:
                raise AgentError(f"Unsupported file format: {ext}")

            # Extract text first
            text_result = self._extract_text({"file_path": str(file_path)})

            # Perform analysis
            analysis = {
                "file_path": str(file_path),
                "file_type": ext,
                "file_size": file_path.stat().st_size,
                "page_count": text_result["page_count"],
                "total_word_count": text_result["total_word_count"],
                "average_words_per_page": text_result["total_word_count"]
                / text_result["page_count"],
                "pages": [],
            }

            # Analyze each page
            for page in text_result["pages"]:
                page_analysis = {
                    "page_number": page["page_number"],
                    "word_count": page["word_count"],
                    "character_count": len(page["text"]),
                    "line_count": len(page["text"].splitlines()),
                    "average_line_length": np.mean(
                        [len(line) for line in page["text"].splitlines()]
                    )
                    if page["text"]
                    else 0,
                }
                analysis["pages"].append(page_analysis)

            return analysis

        except Exception as e:
            logger.error(f"Error analyzing document: {str(e)}")
            raise AgentError(f"Failed to analyze document: {str(e)}")

    def _update_metrics(self, execution_time: float) -> None:
        """
        Update agent metrics.

        Args:
            execution_time: Task execution time in seconds
        """
        # Add custom metrics tracking here
        pass
