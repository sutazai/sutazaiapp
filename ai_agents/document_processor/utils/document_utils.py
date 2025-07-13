#!/usr/bin/env python3.11
# pylint: disable=no-member
import hashlib
import os
from typing import Any, Dict, List

from loguru import logger

from ai_agents.exceptions import DocumentProcessingError


class DocumentUtils:
    """
    Advanced Document Processing Utility Functions
    Provides comprehensive document analysis and preprocessing utilities
    """

    @staticmethod
    def validate_document(file_path: str) -> Dict[str, Any]:
        """
        Comprehensive document validation

        Args:
            file_path (str): Path to the document

        Returns:
            Dict: Document validation results
        """
        try:
            # File existence check
            if not os.path.exists(file_path):
                return {"status": "error", "message": "File does not exist"}

            # MIME type detection
            mime = magic.Magic(mime=True)
            file_type = mime.from_file(file_path)

            # File size check
            file_size = os.path.getsize(file_path)

            # File hash generation
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()

            return {
                "status": "success",
                "metadata": {
                    "file_path": file_path,
                    "mime_type": file_type,
                    "file_size_bytes": file_size,
                    "file_hash": file_hash,
                },
            }
        except FileNotFoundError as e:
            logger.error("Document validation error: %s", e)
            return {"status": "error", "message": str(e)}

    @staticmethod
    def preprocess_image(image_path: str) -> "np.ndarray":
        """
        Advanced image preprocessing for OCR

        Args:
            image_path (str): Path to the image file

        Returns:
            np.ndarray: Preprocessed image
        """
        try:
            img = cv2.imread(image_path, cv2.IMREAD_COLOR)  # type: ignore
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # type: ignore
            return cv2.adaptiveThreshold(  # type: ignore
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # type: ignore
                cv2.THRESH_BINARY,
                11,
                2,
            )
        except cv2.error as e:  # type: ignore
            raise DocumentProcessingError(f"CV2 processing error: {e}") from e

    @staticmethod
    def extract_document_metadata(document_path: str) -> Dict[str, Any]:
        """
        Extract metadata from a document

        Args:
            document_path (str): Path to the document

        Returns:
            Dict: Extracted document metadata
        """
        try:
            doc = fitz.open(document_path)
            # Initialize metadata dictionary with defaults
            metadata: Dict[str, Any] = {
                "total_pages": len(doc),
                "title": "Unknown",
                "author": "Unknown",
                "creator": "Unknown",
                "producer": "Unknown",
                "creation_date": "Unknown",
                "modification_date": "Unknown",
                "keywords": [],
                "page_dimensions": [],
            }

            # Safely get metadata if available
            if hasattr(doc, "metadata") and doc.metadata is not None:
                metadata.update(
                    {
                        "title": doc.metadata.get("title", "Unknown"),
                        "author": doc.metadata.get("author", "Unknown"),
                        "creator": doc.metadata.get("creator", "Unknown"),
                        "producer": doc.metadata.get("producer", "Unknown"),
                        "creation_date": doc.metadata.get("creationDate", "Unknown"),
                        "modification_date": doc.metadata.get("modDate", "Unknown"),
                        "keywords": doc.metadata.get("keywords", []),
                    },
                )

            # Page dimension extraction
            page_dimensions: List[Dict[str, Any]] = []

            for page_num, page in enumerate(doc):
                rect = page.rect
                page_dimensions.append(
                    {
                        "page": page_num,
                        "width": rect.width,
                        "height": rect.height,
                    },
                )

            metadata["page_dimensions"] = page_dimensions
            return {"status": "success", "metadata": metadata}
        except fitz.fitz.FileDataError as e:
            logger.error("Metadata extraction error: %s", e)
            return {"status": "error", "message": str(e)}

    @staticmethod
    def detect_document_language(text: str) -> List[str]:
        """
        Detect the language of a document text

        Args:
            text (str): Document text content

        Returns:
            List[str]: List of detected languages with confidence scores
        """
        try:
            # Detect languages with confidence scores
            langs = detect_langs(text)
            return [str(lang) for lang in langs]
        except Exception as e:
            logger.exception(f"Language detection failed: {e}")
            return ["unknown"]

    @staticmethod
    def detect_language(text: str) -> Dict[str, float]:
        """
        Detect language with confidence scores

        Args:
            text (str): Text to analyze

        Returns:
            Dict[str, float]: Language codes mapped to confidence scores
        """
        return {result.lang: result.prob for result in detect_langs(text)}

    """"""
