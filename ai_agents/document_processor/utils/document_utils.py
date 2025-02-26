import hashlib
import os
from typing import Any, Dict, List, Optional

import cv2
import fitz
import magic
import numpy as np
from loguru import logger


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

        except Exception as e:
            logger.error(f"Document validation error: {e}")
            return {"status": "error", "message": str(e)}

    @staticmethod
    def preprocess_image(image_path: str) -> Optional[np.ndarray]:
        """
        Advanced image preprocessing for OCR

        Args:
            image_path (str): Path to the image file

        Returns:
            Optional[np.ndarray]: Preprocessed image
        """
        try:
            # Read image
            image = cv2.imread(image_path)

            # Grayscale conversion
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Noise reduction
            denoised = cv2.fastNlMeansDenoising(gray)

            # Binarization
            _, binary = cv2.threshold(
                denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )

            return binary

        except Exception as e:
            logger.error(f"Image preprocessing error: {e}")
            return None

    @staticmethod
    def extract_document_metadata(document_path: str) -> Dict[str, Any]:
        """
        Extract comprehensive document metadata

        Args:
            document_path (str): Path to the document

        Returns:
            Dict: Extracted document metadata
        """
        try:
            doc = fitz.open(document_path)

            metadata = {
                "total_pages": len(doc),
                "title": doc.metadata.get("title", "Unknown"),
                "author": doc.metadata.get("author", "Unknown"),
                "creator": doc.metadata.get("creator", "Unknown"),
                "producer": doc.metadata.get("producer", "Unknown"),
                "creation_date": doc.metadata.get("creationDate", "Unknown"),
                "modification_date": doc.metadata.get("modDate", "Unknown"),
                "keywords": doc.metadata.get("keywords", []),
                "page_dimensions": [],
            }

            # Page dimension extraction
            for page_num in range(len(doc)):
                page = doc[page_num]
                rect = page.rect
                metadata["page_dimensions"].append(
                    {
                        "page": page_num,
                        "width": rect.width,
                        "height": rect.height,
                    }
                )

            return {"status": "success", "metadata": metadata}

        except Exception as e:
            logger.error(f"Metadata extraction error: {e}")
            return {"status": "error", "message": str(e)}

    @staticmethod
    def detect_document_language(text: str) -> List[str]:
        """
        Detect document languages

        Args:
            text (str): Document text

        Returns:
            List[str]: Detected languages
        """
        try:
            from langdetect import detect_langs

            languages = detect_langs(text)
            return [
                lang.lang
                for lang in languages
                if lang.prob > 0.1  # Confidence threshold
            ]

        except ImportError:
            logger.warning(
                "langdetect not installed. Skipping language detection."
            )
            return []
        except Exception as e:
            logger.error(f"Language detection error: {e}")
            return []


def main():
    """
    Demonstration of Document Utility Functions
    """
    # Example usage
    document_path = "/opt/sutazaiapp/doc_data/sample.pdf"

    # Document Validation
    validation_result = DocumentUtils.validate_document(document_path)
    print("Document Validation:", validation_result)

    # Metadata Extraction
    metadata_result = DocumentUtils.extract_document_metadata(document_path)
    print("Document Metadata:", metadata_result)


if __name__ == "__main__":
    main()
