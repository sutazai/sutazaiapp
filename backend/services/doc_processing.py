"""
SutazAI Document Processing Service
Handles document uploads and extraction of information from documents
"""

from fastapi import (
    APIRouter,
    UploadFile,
    File,
    Depends,
    HTTPException,
    BackgroundTasks,
    status,
)
from sqlalchemy.orm import Session
import os
import shutil
import uuid
import logging
from typing import List, Optional
from datetime import datetime
import pytesseract
import fitz

from backend.core.config import get_settings
from backend.core.database import get_db
from backend.models.base_models import Document, DocumentChunk
from backend.services.document_processing.pdf_processor import PDFProcessor
from backend.services.document_processing.docx_processor import DOCXProcessor
from backend.services.vector_store.vector_service import VectorStore

router = APIRouter()
settings = get_settings()
logger = logging.getLogger("doc_processing")

# Initialize processors and vector store
pdf_processor = PDFProcessor(
    ocr_enabled=settings.OCR_ENABLED, document_store_path=settings.DOCUMENT_STORE_PATH
)

docx_processor = DOCXProcessor(document_store_path=settings.DOCUMENT_STORE_PATH)

vector_store = VectorStore(host=settings.QDRANT_HOST, port=settings.QDRANT_PORT)


# F821: Added basic ProcessingError class definition
class ProcessingError(Exception):
    """Custom exception for document processing errors."""

    pass


@router.post("/upload", status_code=status.HTTP_202_ACCEPTED)
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    process_now: bool = False,
    db: Session = Depends(get_db),
):
    """
    Upload a document for processing.

    If process_now is True, the document will be processed immediately.
    Otherwise, it will be queued for background processing.
    """
    # Check if file is supported
    supported_types = {
        "application/pdf": ".pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    }

    if file.content_type not in supported_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Supported types are PDF and DOCX.",
        )

    # Create unique filename with proper extension
    original_filename = file.filename
    ext = supported_types[file.content_type]
    unique_filename = f"{uuid.uuid4()}{ext}"

    # Create upload directory if it doesn't exist
    upload_dir = os.path.join(settings.DOCUMENT_STORE_PATH, "uploads")
    os.makedirs(upload_dir, exist_ok=True)

    # Save the file
    file_path = os.path.join(upload_dir, unique_filename)
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Create database record
    document = Document(
        filename=original_filename,
        content_type=file.content_type,
        size=os.path.getsize(file_path),
        path=file_path,
        processed=False,
        metadata={},
    )

    db.add(document)
    db.commit()
    db.refresh(document)

    # Process document immediately or in background
    if process_now:
        process_document(document.id, file_path, file.content_type, db)
    else:
        background_tasks.add_task(
            process_document, document.id, file_path, file.content_type, db
        )

    return {
        "document_id": document.id,
        "filename": original_filename,
        "status": "processing" if process_now else "queued",
    }


@router.get("/documents", response_model=List[dict])
async def list_documents(
    limit: Optional[int] = 50, skip: Optional[int] = 0, db: Session = Depends(get_db)
):
    """List all uploaded documents."""
    documents = (
        db.query(Document)
        .order_by(Document.created_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "content_type": doc.content_type,
            "size": doc.size,
            "processed": doc.processed,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
        }
        for doc in documents
    ]


@router.get("/{document_id}", response_model=dict)
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """Get document details and metadata."""
    document = db.query(Document).filter(Document.id == document_id).first()
    if not document:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": document.id,
        "filename": document.filename,
        "content_type": document.content_type,
        "size": document.size,
        "processed": document.processed,
        "metadata": document.doc_metadata,
        "created_at": document.created_at,
        "updated_at": document.updated_at,
        "project_id": document.project_id,
    }


@router.delete("/documents/{document_id}")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """Delete a document and its chunks."""
    document = db.query(Document).filter(Document.id == document_id).first()

    if not document:
        raise HTTPException(
            status_code=404, detail=f"Document with ID {document_id} not found"
        )

    # Delete file if it exists
    if document.path and os.path.exists(document.path):
        try:
            os.remove(document.path)
            logger.info(f"Deleted file: {document.path}")
        except Exception as e:
            logger.error(f"Error deleting file {document.path}: {str(e)}")

    # Delete from vector store
    try:
        vector_store.delete_document(str(document_id))
    except Exception as e:
        logger.error(f"Error deleting document from vector store: {str(e)}")

    # Delete from database
    db.query(DocumentChunk).filter(DocumentChunk.document_id == document_id).delete()
    db.delete(document)
    db.commit()

    return {"status": "success", "message": f"Document {document_id} deleted"}


@router.post("/search")
async def search_documents(
    query: str, limit: Optional[int] = 5, db: Session = Depends(get_db)
):
    """Search for documents using vector similarity."""
    try:
        # Search in vector store
        results = vector_store.search(query, limit=limit)

        # Enhance results with document information
        enhanced_results = []
        for result in results:
            document_id = result.get("document_id")
            if document_id:
                # Get document from database
                document = db.query(Document).filter(Document.id == document_id).first()
                if document:
                    enhanced_results.append(
                        {
                            **result,
                            "document": {
                                "id": document.id,
                                "filename": document.filename,
                                "content_type": document.content_type,
                            },
                        }
                    )
                else:
                    enhanced_results.append(result)
            else:
                enhanced_results.append(result)

        return {"query": query, "results": enhanced_results}
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(
            status_code=500, detail=f"Error searching documents: {str(e)}"
        )


def process_document(document_id: int, file_path: str, content_type: str, db: Session):
    """Process a document and store its chunks in the vector store."""
    try:
        logger.info(f"Processing document {document_id} ({file_path})")

        # Get document from database
        document = db.query(Document).filter(Document.id == document_id).first()
        if not document:
            logger.error(f"Document {document_id} not found in database")
            return

        # Process document based on content type
        if content_type == "application/pdf":
            processed_data = pdf_processor.process_pdf(file_path)
        elif (
            content_type
            == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        ):
            processed_data = docx_processor.process_docx(file_path)
        else:
            logger.error(f"Unsupported content type: {content_type}")
            return

        # Save metadata to document
        document.doc_metadata = processed_data.get("metadata", {})
        document.processed = True
        db.commit()

        # Create document chunks
        chunks = vector_store._create_text_chunks(processed_data.get("full_text", ""))

        # Store chunks in database
        db_chunks = []
        for i, chunk in enumerate(chunks):
            db_chunk = DocumentChunk(
                document_id=document_id, content=chunk, chunk_index=i, metadata={}
            )
            db.add(db_chunk)
            db_chunks.append(db_chunk)

        db.commit()

        # Store in vector store
        processed_data["document_id"] = str(document_id)
        vector_store.store_document(processed_data)

        logger.info(
            f"Successfully processed document {document_id} with {len(chunks)} chunks"
        )

        document.doc_metadata = {
            **(document.doc_metadata or {}),
            "processing_started_at": datetime.now().isoformat(),
            "status": "processing",
        }

    except Exception as e:
        logger.error(f"Error processing document {document_id}: {str(e)}")
        # Update document status to indicate error
        document = db.query(Document).filter(Document.id == document_id).first()
        if document:
            document.doc_metadata = {
                **(document.doc_metadata or {}),
                "processing_error": str(e),
                "error_time": datetime.now().isoformat(),
            }
            db.commit()


class PDFProcessor:
    def __init__(self, ocr_enabled=True, document_store_path=None):
        self.ocr_enabled = ocr_enabled
        self.document_store_path = (
            document_store_path or "/opt/sutazaiapp/data/documents"
        )
        os.makedirs(self.document_store_path, exist_ok=True)

        # Validate Tesseract installation
        try:
            pytesseract.get_tesseract_version()
        except Exception as e:
            logger.error(f"Tesseract not properly installed: {e}")
            self.ocr_enabled = False

    def process_pdf(self, file_path: str) -> dict:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"PDF file not found: {file_path}")

        doc = None
        try:
            doc = fitz.open(file_path)
            pages = doc.page_count
            logger.info(f"Processing PDF: {file_path}, pages: {pages}")

            # ... (rest of the try block)

        except Exception as e:
            logger.error(f"Error processing PDF {file_path}: {e}")
            raise ProcessingError(f"Failed to process PDF: {e}")
        finally:
            if doc:
                try:
                    doc.close()
                except Exception as e:
                    logger.warning(f"Could not close PDF document {file_path}: {e}")
                    pass
