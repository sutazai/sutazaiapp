import logging
import os
import time
from datetime import datetime
from docx import Document

logger = logging.getLogger("docx_processor")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("/opt/sutazaiapp/logs/docx_processor.log"),
        logging.StreamHandler(),
    ],
)


class DOCXProcessor:
    def __init__(self, doc_store=None, document_store_path=None):
        self.doc_store = doc_store
        self.document_store_path = document_store_path
        logger.info("DOCX Processor initialized")

    def process_docx(self, file_path: str) -> dict:
        start_time = time.time()
        try:
            doc = Document(file_path)
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs)

            metadata = {
                "page_count": len(doc.sections),
                "file_size_bytes": os.path.getsize(file_path),
                "processed_at": datetime.now().isoformat(),
            }

            processing_time_ms = int((time.time() - start_time) * 1000)
            logger.info(f"Processed DOCX '{file_path}' in {processing_time_ms}ms")
            return {
                "metadata": metadata,
                "paragraphs": paragraphs,
                "full_text": full_text,
            }
        except Exception as e:
            logger.error(f"DOCX processing error for {file_path}: {e}")
            raise
