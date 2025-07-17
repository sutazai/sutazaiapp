"""
Document Router

This module provides API routes for document uploading, processing, and retrieval.
"""

import os
import uuid
import shutil
from typing import List, Optional
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
from sqlmodel import Session as SQLModelSession, select as sqlmodel_select

from backend.core.database import get_db
from backend.models.base_models import Document, User
from backend.core.config import get_settings
from backend.dependencies import get_current_active_user
from backend.schemas import DocumentRead
from backend.crud import document_crud

# Set up logging
logger = logging.getLogger("document_router")

# Create router
router = APIRouter()

# Get application settings
settings = get_settings()

# Use UPLOAD_DIR from settings
# UPLOAD_DIR = settings.UPLOAD_DIR # No longer needed directly here if CRUD handles paths


@router.get("/", summary="List all documents", response_model=List[DocumentRead])
async def list_documents(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    db: SQLModelSession = Depends(get_db),
    project_id: Optional[int] = Query(None),
    q: Optional[str] = Query(None),
    tags: Optional[List[str]] = Query(None),
):
    """
    Retrieve a list of uploaded documents, optionally filtered.
    """
    query = sqlmodel_select(Document)
    if project_id:
        query = query.where(Document.project_id == project_id)

    if q and Document.filename is not None:
        search_pattern = f"%{q}%"
        query = query.where(Document.filename.ilike(search_pattern)) # type: ignore[attr-defined]

    if tags:
        # Assuming tags are stored in metadata (e.g., metadata["tags"]) - adjust field name
        pass

    documents = db.exec(query.offset(skip).limit(limit)).all()
    return documents # Return SQLModel objects directly, FastAPI handles validation via response_model


@router.post(
    "/upload", 
    summary="Upload a document", 
    response_model=DocumentRead, # Return created document info
    status_code=status.HTTP_201_CREATED
)
async def upload_document(
    file: UploadFile = File(...),
    title: Optional[str] = Form(None),
    description: Optional[str] = Form(None),
    db: SQLModelSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user)
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
    original_filename = file.filename
    file_extension = os.path.splitext(original_filename)[1].lower().lstrip(".")
    supported_extensions = settings.SUPPORTED_DOC_TYPES.split(',')
    if file_extension not in supported_extensions:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"File type not allowed. Supported types: {', '.join(supported_extensions)}",
        )

    # Generate a unique filename
    unique_filename = f"{uuid.uuid4().hex}.{file_extension}"

    # Use DOCUMENT_DIR from settings for storage location
    file_path = settings.DOCUMENT_DIR / unique_filename 

    # Ensure directory exists
    settings.DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)

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

    # Create document record using CRUD
    doc_data = Document(
        filename=original_filename, # Store original filename
        content_type=file.content_type,
        size=os.path.getsize(file_path), # Get size after saving
        path=str(file_path), # Store path as string
        processed=False,
        doc_metadata={
            "saved_filename": unique_filename, # Store the unique name used on disk
            "file_type": file_extension,
            "title": title or original_filename,
            "description": description,
            "uploader_id": current_user.id # Link to uploading user
        },
        owner_id=current_user.id # Set owner
    )

    try:
        created_doc = document_crud.create_document(db=db, document=doc_data)
    except Exception as e:
        logger.error(f"Error creating document record: {str(e)}")
        os.unlink(file_path)  # Delete the file if record creation fails
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error creating document record",
        )

    return created_doc # Return the created document (validated by response_model)


@router.get("/{document_id}", summary="Get document details", response_model=DocumentRead)
async def get_document(document_id: int, db: SQLModelSession = Depends(get_db)):
    """
    Retrieve a specific document by ID.
    """
    doc = document_crud.get_document(db=db, document_id=document_id) # Use CRUD
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    return doc # Return the document object (validated by response_model)


@router.get("/download/{document_id}", summary="Download a document")
async def download_document(document_id: int, db: SQLModelSession = Depends(get_db)):
    """
    Download a document file by ID.
    """
    doc = document_crud.get_document(db=db, document_id=document_id)
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

    assert doc.path is not None
    return FileResponse(doc.path, filename=doc.filename, media_type=doc.content_type)


@router.delete("/{document_id}", summary="Delete a document", status_code=status.HTTP_204_NO_CONTENT)
async def delete_document(document_id: int, db: SQLModelSession = Depends(get_db)):
    """
    Delete a document and its file.
    """
    doc = document_crud.get_document(db=db, document_id=document_id) # Use CRUD
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    # Delete file if path exists in DB record and file exists on disk
    if doc.path and os.path.exists(doc.path):
        try:
            assert doc.path is not None
            os.unlink(doc.path)
        except Exception as e:
            logger.error(f"Error deleting file {doc.path}: {str(e)}")
            # Continue with record deletion even if file deletion fails

    # Delete record using CRUD
    deleted = document_crud.delete_document(db=db, document_id=document_id)
    if not deleted:
        # This case might happen if the document was deleted between get and delete calls
        logger.warning(f"Attempted to delete document {document_id}, but it was already gone from DB.")
        # Return success anyway as the desired state (deleted) is achieved
        
    # No content is returned on success due to status_code=204
    return None

# TODO: Add endpoints for updating and deleting document metadata if needed.
# TODO: Consider adding an endpoint specifically for uploading the document file content.
