import logging
from typing import List, Optional

from sqlmodel import Session, select # Import Session and select

from backend.models.base_models import Document # Assuming Document model is defined here

logger = logging.getLogger(__name__)

def get_document(db: Session, document_id: int) -> Optional[Document]:
    statement = select(Document).where(Document.id == document_id)
    document = db.exec(statement).first()
    return document

def get_documents(db: Session, skip: int = 0, limit: int = 100) -> List[Document]:
    statement = select(Document).offset(skip).limit(limit)
    documents = db.exec(statement).all()
    return documents

def create_document(db: Session, document: Document) -> Document:
    # Assuming 'document' passed in is already a Document model instance
    db.add(document)
    db.commit()
    db.refresh(document)
    logger.info(f"Document created with ID: {document.id}")
    return document

# Add update and delete functions as needed
# def update_document(...)
# def delete_document(...) 