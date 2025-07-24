from typing import List, Optional

# Use sqlmodel Session and select
from sqlmodel import Session, select
from sqlalchemy.exc import SQLAlchemyError # Import for specific exceptions
import logging

# Import models from base_models
from backend.models.base_models import Document, DocumentChunk # Import from base_models
# Import update schema
from backend.schemas import DocumentUpdate

logger = logging.getLogger(__name__)

def get_document(db: Session, document_id: int) -> Optional[Document]:
    """Retrieve a single document by its ID."""
    try:
        # Use db.get() for primary key lookup for efficiency
        document = db.get(Document, document_id)
        return document
    except SQLAlchemyError as e:
        logger.error(f"Database error retrieving document {document_id}: {e}")
        raise # Re-raise after logging
    except Exception as e:
        logger.error(f"Unexpected error retrieving document {document_id}: {e}")
        raise

def get_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
    statement = select(Document).offset(skip).limit(limit)
    results = db.exec(statement).all()
    return list(results)

def create_document(db: Session, document: Document) -> Document:
    """Create a new document record."""
    try:
        db.add(document)
        db.commit()
        db.refresh(document)
        logger.info(f"Document created with ID: {document.id}")
        return document
    except SQLAlchemyError as e:
        logger.error(f"Database error creating document: {e}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error creating document: {e}")
        db.rollback()
        raise

def update_document(db: Session, db_document: Document, document_in: DocumentUpdate) -> Document:
    """Update an existing document record."""
    # Assuming document_in is a Pydantic model or dict
    update_data = document_in.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(db_document, key, value)
    try:
        db.add(db_document)
        db.commit()
        db.refresh(db_document)
        return db_document
    except SQLAlchemyError as e:
        logger.error(f"Database error updating document {db_document.id}: {e}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error updating document {db_document.id}: {e}")
        db.rollback()
        raise

def delete_document(db: Session, document_id: int) -> bool:
    """Delete a document record."""
    try:
        document = db.get(Document, document_id)
        if document:
            db.delete(document)
            db.commit()
            return True
        return False
    except SQLAlchemyError as e:
        logger.error(f"Database error deleting document {document_id}: {e}")
        db.rollback()
        raise
    except Exception as e:
        logger.error(f"Unexpected error deleting document {document_id}: {e}")
        db.rollback()
        raise

def create_document_chunk(db: Session, chunk: DocumentChunk) -> DocumentChunk:
    db.add(chunk)
    db.commit()
    db.refresh(chunk)
    return chunk

def get_document_chunks(db: Session, document_id: int) -> List[DocumentChunk]:
    statement = select(DocumentChunk).where(DocumentChunk.document_id == document_id)
    results = db.exec(statement).all()
    return list(results)

# Add functions for DocumentChunk if needed
# e.g., create_document_chunk, get_chunks_by_document_id