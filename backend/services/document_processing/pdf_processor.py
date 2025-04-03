import os
import sys
import logging
import time
from datetime import datetime
import pytesseract
from PIL import Image

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("pdf_processor")

logger.info("Starting PDF Processor initialization...")
logger.info(f"Python sys.executable: {sys.executable}")
logger.info(f"Python sys.path: {sys.path}")

# Add venv site-packages to Python path if needed
venv_path = os.path.join("/opt/sutazaiapp/venv/lib/python3.11/site-packages")
if venv_path not in sys.path:
    sys.path.append(venv_path)
    logger.info(f"Added {venv_path} to Python path: {venv_path}")
    logger.info(f"Updated sys.path: {sys.path}")  # Log updated path immediately

try:
    import fitz  # PyMuPDF

    logger.info(
        f"Successfully imported fitz. Version: {fitz.__version__}, File: {fitz.__file__}"
    )  # Log successful import
except ImportError as e:
    logger.error(f"Initial fitz import failed: {e}")
    logger.info("Attempting to fix import path...")

    # Try to find PyMuPDF in alternative locations
    potential_paths = [
        "/opt/sutazaiapp/venv/lib/python3.11/site-packages",
        "/usr/local/lib/python3.11/site-packages",
        "/usr/lib/python3.11/site-packages",
    ]

    for path in potential_paths:
        if path not in sys.path:
            sys.path.append(path)
            logger.info(f"Added {path} to Python path")

    try:
        import fitz

        logger.info(
            "Successfully imported fitz after path adjustment. Version: {fitz.__version__}, File: {fitz.__file__}"
        )  # Log successful import after fix
    except ImportError as e:
        logger.error(f"Still unable to import fitz: {e}")
        raise ImportError("PyMuPDF (fitz) module is required but could not be imported")


class PDFProcessor:
    def __init__(self, ocr_enabled=True, document_store_path=None):
        self.ocr_enabled = ocr_enabled
        self.document_store_path = document_store_path
        self._verify_tesseract()

    def _verify_tesseract(self):
        """Ensure Tesseract OCR is properly configured"""
        try:
            pytesseract.get_tesseract_version()
        except EnvironmentError:
            logger.error("Tesseract OCR not found in PATH")
            if self.ocr_enabled:
                raise RuntimeError("Tesseract required for OCR processing")

    def process_pdf(self, file_path: str, save_processed=True) -> dict:
        """Process a PDF file and extract text and metadata."""
        start_time = time.time()
        try:
            pdf_document = fitz.open(file_path)

            # Extract basic metadata
            metadata = {
                "title": pdf_document.metadata.get("title", ""),
                "author": pdf_document.metadata.get("author", ""),
                "subject": pdf_document.metadata.get("subject", ""),
                "page_count": len(pdf_document),
                "file_size_bytes": os.path.getsize(file_path),
                "processed_at": datetime.now().isoformat(),
            }

            # Extract text from each page
            text_content = []
            image_count = 0

            for page_num, page in enumerate(pdf_document):
                # Get text directly from PDF
                text = page.get_text()

                # If page has little text and OCR is enabled, try OCR
                if len(text.strip()) < 100 and self.ocr_enabled:
                    # Convert page to image
                    pix = page.get_pixmap(matrix=fitz.Matrix(300 / 72, 300 / 72))
                    img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)

                    # Perform OCR
                    ocr_text = pytesseract.image_to_string(img)

                    # Use OCR text if it found more content
                    if len(ocr_text.strip()) > len(text.strip()):
                        text = ocr_text
                        logger.info(f"Used OCR for page {page_num + 1} in {file_path}")

                # Count images on page
                image_list = page.get_images(full=True)
                image_count += len(image_list)

                # Add page content
                text_content.append({"page_num": page_num + 1, "text": text})

            # Combine all text content
            full_text = "\n\n".join([page["text"] for page in text_content])

            # Save processed document if requested
            if save_processed and self.document_store_path:
                self._save_processed_document(
                    file_path,
                    {
                        "metadata": metadata,
                        "pages": text_content,
                        "full_text": full_text,
                    },
                )

            # Calculate processing time
            processing_time_ms = int((time.time() - start_time) * 1000)

            result = {
                "metadata": metadata,
                "pages": text_content,
                "full_text": full_text,
                "image_count": image_count,
                "processing_time_ms": processing_time_ms,
            }

            logger.info(
                f"Successfully processed PDF: {file_path} ({len(pdf_document)} pages, {processing_time_ms}ms)"
            )
            return result

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {str(e)}")
            raise

    def _save_processed_document(self, original_path, processed_data):
        """Save processed document data to disk."""
        try:
            if not self.document_store_path:
                return

            # Create directory if it doesn't exist
            os.makedirs(self.document_store_path, exist_ok=True)

            # Generate a filename based on original filename
            base_filename = os.path.basename(original_path)
            filename_without_ext = os.path.splitext(base_filename)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            processed_filename = f"{filename_without_ext}_{timestamp}.json"

            # Save as JSON
            import json

            save_path = os.path.join(self.document_store_path, processed_filename)
            with open(save_path, "w") as f:
                json.dump(processed_data, f, ensure_ascii=False, indent=2)

            logger.info(f"Saved processed document to {save_path}")

        except Exception as e:
            logger.error(f"Error saving processed document: {str(e)}")
