"""
SutazAI Document Parsing Service
Provides a unified interface for parsing and analyzing various document formats
"""

import os
import logging
import mimetypes
from typing import Dict, Any, Optional, BinaryIO

# Import individual parsers
from .pdf_parser import PDFParser
from .docx_parser import DocxParser

# Configure logging
logger = logging.getLogger("document_service")


class DocumentService:
    """
    Service for parsing and analyzing various document formats
    Provides a unified interface for all supported document types
    """

    def __init__(self):
        """Initialize the document service"""
        # Initialize the different parsers with graceful fallbacks
        self.pdf_parser = None
        self.docx_parser = None

        # Initialize PDF parser
        try:
            self.pdf_parser = PDFParser(enable_ocr=True, extract_tables=True)
            logger.info("PDF parser initialized with full functionality")
        except Exception as e:
            logger.warning(
                f"PDF parser initialized with limited functionality: {str(e)}"
            )
            try:
                # Fallback to basic parser without OCR and tables
                self.pdf_parser = PDFParser(enable_ocr=False, extract_tables=False)
                logger.info(
                    "PDF parser initialized with basic functionality (no OCR, no tables)"
                )
            except Exception as inner_e:
                logger.error(f"Failed to initialize PDF parser: {str(inner_e)}")
                # We'll handle missing parser in the parse_document method

        # Initialize DOCX parser
        try:
            self.docx_parser = DocxParser(extract_images=True)
            logger.info("DOCX parser initialized with full functionality")
        except Exception as e:
            logger.warning(
                f"DOCX parser initialized with limited functionality: {str(e)}"
            )
            try:
                # Fallback to basic parser without images
                self.docx_parser = DocxParser(extract_images=False)
                logger.info(
                    "DOCX parser initialized with basic functionality (no images)"
                )
            except Exception as inner_e:
                logger.error(f"Failed to initialize DOCX parser: {str(inner_e)}")
                # We'll handle missing parser in the parse_document method

        # Register supported file types
        self.supported_extensions = {
            ".pdf": self._parse_pdf,
            ".docx": self._parse_docx,
            ".doc": self._parse_docx,  # Attempt to parse DOC as DOCX
        }

        # Register supported MIME types
        self.supported_mimetypes = {
            "application/pdf": self._parse_pdf,
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": self._parse_docx,
            "application/msword": self._parse_docx,
        }

    def parse_document(
        self,
        file_path: Optional[str] = None,
        file_content: Optional[bytes] = None,
        file_type: Optional[str] = None,
        file_obj: Optional[BinaryIO] = None,
        filename: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Parse a document file using the appropriate parser

        Args:
            file_path: Path to the document file (optional)
            file_content: Raw file content as bytes (optional)
            file_type: File type or extension (optional, used if file_path not provided)
            file_obj: File-like object (optional)
            filename: Original filename (optional, for reference)

        Returns:
            Dictionary with parsed document information

        Note:
            You must provide either file_path, file_content, or file_obj
        """
        # Determine file type
        doc_type = file_type
        actual_filename = filename or (
            os.path.basename(file_path) if file_path else None
        )

        # Load content from file path if provided
        if file_path:
            if not os.path.exists(file_path):
                logger.error(f"File not found: {file_path}")
                return {"error": f"File not found: {file_path}"}

            # Determine file type from extension if not provided
            if not doc_type:
                _, ext = os.path.splitext(file_path)
                doc_type = ext.lower()

            # Read file content
            try:
                with open(file_path, "rb") as f:
                    file_content = f.read()
            except Exception as e:
                logger.error(f"Error reading file {file_path}: {str(e)}")
                return {"error": f"Error reading file: {str(e)}"}

        # Load content from file object if provided
        elif file_obj:
            try:
                file_content = file_obj.read()

                # Try to determine file type from file object if possible
                if hasattr(file_obj, "name") and not doc_type:
                    _, ext = os.path.splitext(file_obj.name)
                    doc_type = ext.lower()
            except Exception as e:
                logger.error(f"Error reading from file object: {str(e)}")
                return {"error": f"Error reading from file object: {str(e)}"}

        # Ensure we have content to parse
        if not file_content:
            logger.error("No file content provided")
            return {"error": "No file content provided"}

        # Try to determine file type from content if still unknown
        if not doc_type:
            mime_type = self._guess_mimetype(file_content, actual_filename)
            if mime_type in self.supported_mimetypes:
                parser_func = self.supported_mimetypes[mime_type]
                return parser_func(file_content, actual_filename)
            else:
                logger.error(f"Unsupported or unknown file type: {mime_type}")
                return {"error": f"Unsupported or unknown file type: {mime_type}"}

        # Use the appropriate parser based on file extension
        if doc_type in self.supported_extensions:
            parser_func = self.supported_extensions[doc_type]
            return parser_func(file_content, actual_filename)
        else:
            logger.error(f"Unsupported file type: {doc_type}")
            return {"error": f"Unsupported file type: {doc_type}"}

    def _parse_pdf(
        self, content: bytes, filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Parse PDF content"""
        if self.pdf_parser is None:
            logger.error("PDF parser not available")
            return {"error": "PDF parsing is not available due to missing dependencies"}
        return self.pdf_parser.parse_bytes(content, filename=filename)

    def _parse_docx(
        self, content: bytes, filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """Parse DOCX content"""
        if self.docx_parser is None:
            logger.error("DOCX parser not available")
            return {
                "error": "DOCX parsing is not available due to missing dependencies"
            }
        return self.docx_parser.parse_bytes(content, filename=filename)

    def _guess_mimetype(self, content: bytes, filename: Optional[str] = None) -> str:
        """
        Guess the MIME type of file content

        Args:
            content: File content as bytes
            filename: Original filename (optional)

        Returns:
            Guessed MIME type
        """
        # Try to guess from filename first if available
        if filename:
            mime_type, _ = mimetypes.guess_type(filename)
            if mime_type:
                return mime_type

        # Check for PDF signature
        if content.startswith(b"%PDF-"):
            return "application/pdf"

        # Check for Office Open XML (DOCX) signature
        if content.startswith(b"PK\x03\x04"):
            # This is a ZIP file, which could be DOCX or other Office formats
            # More detailed checking would require extracting the ZIP and checking contents
            return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"

        # Default to binary data
        return "application/octet-stream"

    def extract_text(
        self,
        file_path: Optional[str] = None,
        file_content: Optional[bytes] = None,
        file_type: Optional[str] = None,
    ) -> str:
        """
        Extract text content from a document

        Args:
            file_path: Path to the document file (optional)
            file_content: Raw file content as bytes (optional)
            file_type: File type or extension (optional)

        Returns:
            Extracted text content
        """
        result = self.parse_document(file_path, file_content, file_type)

        if "error" in result:
            logger.error(f"Error extracting text: {result['error']}")
            return ""

        if "full_text" in result:
            return result["full_text"]

        return ""

    def extract_metadata(
        self,
        file_path: Optional[str] = None,
        file_content: Optional[bytes] = None,
        file_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Extract metadata from a document

        Args:
            file_path: Path to the document file (optional)
            file_content: Raw file content as bytes (optional)
            file_type: File type or extension (optional)

        Returns:
            Dictionary with document metadata
        """
        result = self.parse_document(file_path, file_content, file_type)

        if "error" in result:
            logger.error(f"Error extracting metadata: {result['error']}")
            return {}

        if "metadata" in result:
            return result["metadata"]

        return {}

    def get_document_structure(
        self,
        file_path: Optional[str] = None,
        file_content: Optional[bytes] = None,
        file_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Get document structure information

        Args:
            file_path: Path to the document file (optional)
            file_content: Raw file content as bytes (optional)
            file_type: File type or extension (optional)

        Returns:
            Dictionary with document structure information
        """
        result = self.parse_document(file_path, file_content, file_type)

        if "error" in result:
            logger.error(f"Error extracting structure: {result['error']}")
            return {}

        if "structure" in result:
            return result["structure"]

        return {}


# Create a singleton instance for easy import
document_service = DocumentService()


# Helper functions for easier usage
def parse_document(file_path: str) -> Dict[str, Any]:
    """
    Parse a document file using the appropriate parser

    Args:
        file_path: Path to the document file

    Returns:
        Dictionary with parsed document information
    """
    return document_service.parse_document(file_path=file_path)


def extract_text(file_path: str) -> str:
    """
    Extract text content from a document

    Args:
        file_path: Path to the document file

    Returns:
        Extracted text content
    """
    return document_service.extract_text(file_path=file_path)


def extract_metadata(file_path: str) -> Dict[str, Any]:
    """
    Extract metadata from a document

    Args:
        file_path: Path to the document file

    Returns:
        Dictionary with document metadata
    """
    return document_service.extract_metadata(file_path=file_path)


def get_document_structure(file_path: str) -> Dict[str, Any]:
    """
    Get document structure information

    Args:
        file_path: Path to the document file

    Returns:
        Dictionary with document structure information
    """
    return document_service.get_document_structure(file_path=file_path)
