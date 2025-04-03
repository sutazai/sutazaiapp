"""
Document Router

This module provides API routes for document uploading, processing, and retrieval.
"""

import os
import uuid
import shutil
from typing import List, Optional, Dict, Any
import logging
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    status,
    UploadFile,
    File,
    Form,
    Query,
)
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from ..core.database import get_db
from ..models.base_models import Document
from ..core.config import get_settings
from ..core.storage import get_document_store_path

# Set up logging
logger = logging.getLogger("document_router")

# Create router
router = APIRouter()

# Get application settings
settings = get_settings()


@router.get("/", summary="List all documents", response_model=List[Dict[str, Any]])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db),
):
    """
    Retrieve a list of uploaded documents.
    """
    documents = db.query(Document).offset(skip).limit(limit).all()
    return [
        {
            "id": doc.id,
            "filename": doc.filename,
            "content_type": doc.content_type,
            "size": doc.size,
            "processed": doc.processed,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "metadata": doc.doc_metadata,
        }
        for doc in documents
    ]


@router.post(
    "/upload", summary="Upload a document", status_code=status.HTTP_201_CREATED
)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: Session = Depends(get_db),
):
    """
    Upload a new document to the system.
    """
    # Ensure filename exists
    if file.filename is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Filename cannot be empty."
        )

    # Verify file type is allowed
    file_extension = os.path.splitext(file.filename)[1].lower().lstrip(".")
    supported_extensions = settings.SUPPORTED_DOC_TYPES.split(',')
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Supported types: {', '.join(supported_extensions)}",
        )

    # Generate a unique filename
    unique_filename = f"{uuid.uuid4().hex}.{file_extension}"

    # Get document store path
    doc_store_path = get_document_store_path()
    file_path = os.path.join(doc_store_path, unique_filename)

    # Ensure directory exists
    os.makedirs(doc_store_path, exist_ok=True)

    # Save file
    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Error saving file: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not save the file",
        )

    # Create document record
    doc = Document(
        filename=file.filename,
        content_type=file.content_type,
        size=os.path.getsize(file_path),
        path=file_path,
        processed=False,
        doc_metadata={
            "original_filename": file.filename,
            "file_type": file_extension,
            "title": title or file.filename,
            "description": description,
        },
    )

    try:
        db.add(doc)
        db.commit()
        db.refresh(doc)
    except Exception as e:
        logger.error(f"Error creating document record: {str(e)}")
        os.unlink(file_path)  # Delete the file if record creation fails
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating document record",
        )

    # Return the created document
    return {
        "message": "Document uploaded successfully",
        "document": {
            "id": doc.id,
            "filename": doc.filename,
            "content_type": doc.content_type,
            "size": doc.size,
            "processed": doc.processed,
            "created_at": doc.created_at,
            "updated_at": doc.updated_at,
            "metadata": doc.doc_metadata,
        },
    }


@router.get("/{document_id}", summary="Get document details")
async def get_document(document_id: int, db: Session = Depends(get_db)):
    """
    Retrieve a specific document by ID.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    return {
        "id": doc.id,
        "filename": doc.filename,
        "content_type": doc.content_type,
        "size": doc.size,
        "processed": doc.processed,
        "created_at": doc.created_at,
        "updated_at": doc.updated_at,
        "metadata": doc.doc_metadata,
    }


@router.get("/download/{document_id}", summary="Download a document")
async def download_document(document_id: int, db: Session = Depends(get_db)):
    """
    Download a document file by ID.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    # Check if path exists in DB record
    if doc.path is None:
        logger.error(f"Document record {document_id} has no associated file path.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Document file path missing in database",
        )

    if not os.path.exists(doc.path):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document file not found on disk"
        )

    assert doc.path is not None # Ensure path is not None for FileResponse
    return FileResponse(doc.path, filename=doc.filename, media_type=doc.content_type)


@router.delete("/{document_id}", summary="Delete a document")
async def delete_document(document_id: int, db: Session = Depends(get_db)):
    """
    Delete a document and its file.
    """
    doc = db.query(Document).filter(Document.id == document_id).first()
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    # Delete file if path exists in DB record and file exists on disk
    if doc.path and os.path.exists(doc.path):
        try:
            # We already confirmed doc.path exists, assert for mypy
            assert doc.path is not None
            os.unlink(doc.path)
        except Exception as e:
            logger.error(f"Error deleting file {doc.path}: {str(e)}")
            # Continue with record deletion even if file deletion fails

    # Delete record
    try:
        db.delete(doc)
        db.commit()
    except Exception as e:
        logger.error(f"Error deleting document record: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error deleting document record",
        )

    return {"message": "Document deleted successfully"}
