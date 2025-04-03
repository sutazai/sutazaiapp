"""
SutazAI PDF Parser Module
Extracts text, tables, and identifies structural elements from PDF documents
"""

import io
import os
import re
import tempfile
import logging
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass
from pathlib import PurePath

# Required dependencies (pip install pypdf pillow)
from pypdf import PdfReader
from PIL import Image as PILImage
from PIL.Image import Image

# Optional dependencies with fallbacks
TESSERACT_AVAILABLE = False
try:
    import pytesseract  # For OCR on scanned PDFs

    TESSERACT_AVAILABLE = True
except ImportError:
    pytesseract = None

PDF2IMAGE_AVAILABLE = False
# Define types for optional imports before try block
ConvertFromPathType = Optional[Callable[..., List[Image]]]
ConvertFromBytesType = Optional[Callable[..., List[Image]]]

convert_from_path: ConvertFromPathType = None
convert_from_bytes: ConvertFromBytesType = None
try:
    from pdf2image import (
        convert_from_path as pdf2image_convert_from_path,
        convert_from_bytes as pdf2image_convert_from_bytes,
    )  # For converting PDF pages to images
    convert_from_path = pdf2image_convert_from_path
    convert_from_bytes = pdf2image_convert_from_bytes
    PDF2IMAGE_AVAILABLE = True
except ImportError:
    # Variables remain None as initialized
    pass

TABULA_AVAILABLE = False
try:
    import tabula  # For table extraction

    TABULA_AVAILABLE = True
except ImportError:
    tabula = None


# Configure logging
logger = logging.getLogger("pdf_parser")


@dataclass
class PDFPage:
    """Data class to store information about a single PDF page"""

    number: int
    text: str
    tables: List[Dict[str, Any]]
    images: List[Dict[str, Any]]
    headers: List[str]
    footers: List[str]


class PDFParser:
    """
    Parser for extracting information from PDF documents including
    text, tables, images, and document structure.
    """

    def __init__(
        self,
        enable_ocr: bool = False,
        ocr_language: str = "eng",
        extract_tables: bool = True,
        extract_images: bool = False,
    ):
        """
        Initialize the PDF parser

        Args:
            enable_ocr: Whether to use OCR for scanned documents
            ocr_language: Language for OCR processing
            extract_tables: Whether to extract tables
            extract_images: Whether to extract images
        """
        # Check and adjust based on available dependencies
        self.enable_ocr = enable_ocr and TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE
        self.ocr_language = ocr_language
        self.extract_tables = extract_tables and TABULA_AVAILABLE
        self.extract_images = extract_images

        # Log warnings about missing dependencies
        if enable_ocr and not (TESSERACT_AVAILABLE and PDF2IMAGE_AVAILABLE):
            logger.warning(
                "OCR requested but dependencies not available. OCR will be disabled."
            )

        if extract_tables and not TABULA_AVAILABLE:
            logger.warning(
                "Table extraction requested but tabula not available. Table extraction will be disabled."
            )

        # Check OCR availability if requested
        if self.enable_ocr:
            self._check_tesseract()

    def _check_tesseract(self):
        """Check if Tesseract OCR is installed"""
        if not TESSERACT_AVAILABLE:
            logger.error("pytesseract module not installed")
            logger.warning("OCR functionality will be disabled")
            self.enable_ocr = False
            return False

        try:
            tesseract_version = pytesseract.get_tesseract_version()
            logger.info(f"Tesseract OCR version: {tesseract_version}")
            return True
        except Exception:
            logger.error("Tesseract OCR not found in PATH")
            logger.warning("OCR functionality will be disabled")
            self.enable_ocr = False
            return False

    def parse_file(self, file_path: str) -> Dict[str, Any]:
        """
        Parse a PDF file from disk

        Args:
            file_path: Path to the PDF file

        Returns:
            Dictionary with parsed information
        """
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return {"error": f"File not found: {file_path}"}

        try:
            with open(file_path, "rb") as f:
                pdf_data = f.read()
            return self.parse_bytes(pdf_data, filename=os.path.basename(file_path))
        except Exception as e:
            logger.error(f"Error parsing PDF file {file_path}: {str(e)}")
            return {"error": f"Failed to parse PDF: {str(e)}"}

    def parse_bytes(
        self, pdf_data: bytes, filename: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Parse a PDF from bytes

        Args:
            pdf_data: PDF file content as bytes
            filename: Optional filename for reference

        Returns:
            Dictionary with parsed information
        """
        try:
            # Use PyPDF to extract basic information
            pdf_stream = io.BytesIO(pdf_data)
            pdf = PdfReader(pdf_stream)

            # Extract metadata
            metadata = self._extract_metadata(pdf)

            # Process each page
            pages = []
            for i, page in enumerate(pdf.pages):
                page_num = i + 1

                # Basic text extraction
                text = page.extract_text()

                # Check if OCR is needed
                if self.enable_ocr and (not text or len(text.strip()) < 50):
                    logger.info(f"Page {page_num} has little text, attempting OCR")
                    text = self._perform_ocr_on_page(pdf_data, page_num)

                # Extract tables if enabled
                tables = []
                if self.extract_tables:
                    tables = self._extract_tables(pdf_data, page_num)

                # Extract images if enabled
                images = []
                if self.extract_images:
                    images = self._extract_images(pdf_data, page_num)

                # Extract headers and footers
                headers = self._identify_headers(text)
                footers = self._identify_footers(text)

                # Create page object
                pdf_page = PDFPage(
                    number=page_num,
                    text=text,
                    tables=tables,
                    images=images,
                    headers=headers,
                    footers=footers,
                )

                pages.append(vars(pdf_page))

            # Organize full document text
            full_text = self._assemble_full_text(pages)

            # Identify document structure
            structure = self._identify_document_structure(full_text)

            return {
                "success": True,
                "metadata": metadata,
                "pages": pages,
                "num_pages": len(pdf.pages),
                "full_text": full_text,
                "structure": structure,
                "tables": [table for page in pages for table in page["tables"]],
                "filename": filename,
            }

        except Exception as e:
            logger.error(f"Error parsing PDF data: {str(e)}")
            return {"success": False, "error": f"Failed to parse PDF: {str(e)}"}

    def _extract_metadata(self, pdf: PdfReader) -> Dict[str, Any]:
        """
        Extract metadata from PDF

        Args:
            pdf: PdfReader object

        Returns:
            Dictionary with metadata
        """
        metadata = {}

        # Extract standard metadata
        if pdf.metadata:
            for key, value in pdf.metadata.items():
                # Convert from PDF object format to string if needed
                if hasattr(value, "original_bytes"):
                    value = value.original_bytes.decode("utf-8", errors="ignore")
                if isinstance(value, (list, dict)) and not value:
                    continue
                # Remove the leading slash in metadata keys
                clean_key = key[1:] if key.startswith("/") else key
                metadata[clean_key] = value

        # Add document info
        metadata["page_count"] = len(pdf.pages)

        # Try to extract PDF version
        try:
            metadata["pdf_version"] = pdf.pdf_header
        except Exception as e:
            logger.debug(f"Could not extract pdf_version metadata: {e}")
            pass

        return metadata

    def _perform_ocr_on_page(self, pdf_data: bytes, page_num: int) -> str:
        """
        Perform OCR on a single PDF page

        Args:
            pdf_data: PDF file content as bytes
            page_num: Page number to process (1-based)

        Returns:
            Extracted text from the page
        """
        if not self.enable_ocr or not TESSERACT_AVAILABLE or not PDF2IMAGE_AVAILABLE or convert_from_bytes is None:
            logger.warning("OCR requested but not available. Returning empty text.")
            return ""

        try:
            # Convert PDF page to image (check if callable)
            images = convert_from_bytes(
                pdf_data, first_page=page_num, last_page=page_num
            )
            if not images:
                logger.warning(f"Could not convert page {page_num} to image for OCR")
                return ""

            # Perform OCR on the image
            text = pytesseract.image_to_string(images[0], lang=self.ocr_language)
            return text.strip()
        except Exception as e:
            logger.error(f"Error performing OCR on page {page_num}: {str(e)}")
            return ""

    def _extract_tables(self, pdf_data: bytes, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract tables from a PDF page

        Args:
            pdf_data: PDF file content as bytes
            page_num: Page number to extract tables from (1-based)

        Returns:
            List of extracted tables as dictionaries
        """
        if not self.extract_tables or not TABULA_AVAILABLE:
            logger.debug("Table extraction not available or disabled")
            return []

        try:
            # Create a temporary file for tabula to process
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(pdf_data)

            try:
                # Extract tables using tabula
                tables = tabula.read_pdf(
                    temp_path, pages=page_num, multiple_tables=True, guess=True
                )

                # Process the extracted tables
                result_tables = []
                for i, table in enumerate(tables):
                    # Convert DataFrame to dict
                    table_dict = {
                        "table_id": f"page_{page_num}_table_{i + 1}",
                        "page": page_num,
                        "data": table.to_dict(orient="records"),
                        "columns": table.columns.tolist(),
                        "rows": len(table),
                        "cols": len(table.columns),
                    }
                    result_tables.append(table_dict)

                return result_tables

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            logger.error(f"Error extracting tables from page {page_num}: {str(e)}")
            return []

    def _extract_images(self, pdf_data: bytes, page_num: int) -> List[Dict[str, Any]]:
        """
        Extract images from a PDF page

        Args:
            pdf_data: PDF file content as bytes
            page_num: Page number to extract images from (1-based)

        Returns:
            List of extracted images as dictionaries
        """
        if not self.extract_images or not PDF2IMAGE_AVAILABLE or convert_from_bytes is None:
            logger.debug("Image extraction not available or disabled")
            return []

        try:
            # This is a simplified implementation - in a real system,
            # you would extract the actual embedded images from the PDF
            # using a library like PyMuPDF

            # Convert the page to an image (check if callable)
            images = convert_from_bytes(
                pdf_data, first_page=page_num, last_page=page_num
            )
            if not images:
                return []

            # For now, just return the whole page as an image
            result = []
            for i, img in enumerate(images):
                # Save image to a bytes buffer
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")
                img_data = img_buffer.getvalue()

                # Add image info
                image_info = {
                    "image_id": f"page_{page_num}_image_{i + 1}",
                    "page": page_num,
                    "width": img.width,
                    "height": img.height,
                    "format": "PNG",
                    "size_bytes": len(img_data),
                    # "data": base64.b64encode(img_data).decode('utf-8')  # Uncomment to include base64 data
                }
                result.append(image_info)

            return result
        except Exception as e:
            logger.error(f"Error extracting images from page {page_num}: {str(e)}")
            return []

    def _identify_headers(self, text: str) -> List[str]:
        """
        Identify potential headers in page text

        Args:
            text: Text content of the page

        Returns:
            List of potential headers
        """
        headers = [] # type: ignore[var-annotated]

        if not text:
            return headers

        # Split text into lines
        lines = text.split("\n")

        # Check the first few lines for potential headers
        for i, line in enumerate(lines[:3]):
            line = line.strip()
            if line and len(line) < 100:  # Reasonable length for a header
                headers.append(line)

        return headers

    def _identify_footers(self, text: str) -> List[str]:
        """
        Identify potential footers in page text

        Args:
            text: Text content of the page

        Returns:
            List of potential footers
        """
        footers = [] # type: ignore[var-annotated]

        if not text:
            return footers

        # Split text into lines
        lines = text.split("\n")

        # Check the last few lines for potential footers
        for i, line in enumerate(lines[-3:]):
            line = line.strip()
            if line and len(line) < 100:  # Reasonable length for a footer
                # Look for page numbers, dates, or copyright info
                if (
                    re.search(r"\d+\s*of\s*\d+", line)
                    or re.search(r"©|copyright|page|p\.", line.lower())
                    or re.search(r"\d{1,2}/\d{1,2}/\d{2,4}", line)
                ):
                    footers.append(line)
                else:
                    # If it's very short, it's likely a footer
                    if len(line) < 30:
                        footers.append(line)

        return footers

    def _assemble_full_text(self, pages: List[Dict[str, Any]]) -> str:
        """
        Assemble full document text from individual pages

        Args:
            pages: List of page information

        Returns:
            Full document text
        """
        text_parts = []

        for page in pages:
            text_parts.append(page["text"])

        return "\n\n".join(text_parts)

    def _identify_document_structure(self, text: str) -> Dict[str, Any]:
        """
        Identify structure elements like sections, titles, etc.

        Args:
            text: Full document text

        Returns:
            Dictionary with document structure information
        """
        structure: Dict[str, Union[str, List[Dict[str, Any]]]] = {
            "title": "", "sections": [], "possible_headings": []
        }

        if not text:
            return structure

        # Split into lines
        lines = text.split("\n")

        # Try to identify document title (usually one of the first non-empty lines)
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) < 200 and len(line) > 5:
                # Check if line is all caps or has larger font
                if line.isupper() or re.match(r"^[A-Z0-9\s,.:-]+$", line):
                    structure["title"] = line
                    break

        # Find possible section headings (looking for numbered sections, all caps lines, etc.)
        possible_headings = []

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Check for numbered sections (e.g., "1. Introduction", "1.2 Background")
            if re.match(r"^(\d+\.)+\s+\S+", line) or re.match(r"^\d+\.\s+\S+", line):
                possible_headings.append(
                    {"text": line, "line_number": i, "type": "numbered"}
                )

            # Check for all caps headings
            elif line.isupper() and 3 <= len(line) <= 100:
                possible_headings.append(
                    {"text": line, "line_number": i, "type": "all_caps"}
                )

            # Check for headings with trailing colon
            elif re.match(r"^[A-Z][^.?!:]*:$", line) and len(line) < 100:
                possible_headings.append(
                    {"text": line, "line_number": i, "type": "colon_ending"}
                )

        structure["possible_headings"] = possible_headings

        # Attempt to organize text into sections
        if possible_headings:
            sections = []

            for i in range(len(possible_headings)):
                heading = possible_headings[i]

                # Determine section content (text until next heading)
                line_num = heading.get("line_number")
                assert isinstance(line_num, int)
                start_line = line_num + 1 # Use asserted int
                end_line = len(lines)

                if i < len(possible_headings) - 1:
                    next_line_num = possible_headings[i + 1].get("line_number")
                    assert isinstance(next_line_num, int)
                    end_line = next_line_num # Use asserted int

                section_content = "\n".join(lines[start_line:end_line])

                # Assert heading['text'] is a string before splitting
                heading_text = heading.get("text", "")
                assert isinstance(heading_text, str)
                level = 1 if heading.get("type") == "numbered" and len(heading_text.split(".")[0]) == 1 else 2

                sections.append(
                    {
                        "heading": heading_text,
                        "content": section_content.strip(),
                        "level": level,
                    }
                )

            structure["sections"] = sections

        return structure


# Helper functions for easier usage
def parse_pdf_file(
    file_path: str, enable_ocr: bool = False, extract_tables: bool = True
) -> Dict[str, Any]:
    """
    Parse PDF file and extract information

    Args:
        file_path: Path to the PDF file
        enable_ocr: Whether to use OCR for scanned documents
        extract_tables: Whether to extract tables

    Returns:
        Dictionary with parsed information
    """
    parser = PDFParser(enable_ocr=enable_ocr, extract_tables=extract_tables)
    return parser.parse_file(file_path)


def extract_pdf_text(file_path: str, enable_ocr: bool = False) -> str:
    """
    Extract only text content from a PDF file

    Args:
        file_path: Path to the PDF file
        enable_ocr: Whether to use OCR for scanned documents

    Returns:
        Extracted text content
    """
    result = parse_pdf_file(file_path, enable_ocr=enable_ocr, extract_tables=False)
    if result.get("success", False):
        return result["full_text"]
    return ""


def extract_pdf_tables(file_path: str) -> List[Dict[str, Any]]:
    """
    Extract only tables from a PDF file

    Args:
        file_path: Path to the PDF file

    Returns:
        List of extracted tables
    """
    result = parse_pdf_file(file_path, extract_tables=True)
    if result.get("success", False):
        return result["tables"]
    return []
