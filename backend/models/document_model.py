"""
Document model for the SutazAI application.
"""

import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean

from .base_models import Base


class Document(Base):
    """
    Document model for storing document metadata and content.
    """

    __tablename__ = "documents"
    __table_args__ = {"extend_existing": True}

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String(255), nullable=False)
    original_filename = Column(String(255), nullable=False)
    file_path = Column(String(512), nullable=False)
    file_size = Column(Integer, nullable=False)
    file_type = Column(String(50), nullable=False)
    content_type = Column(String(100), nullable=True)
    title = Column(String(255), nullable=True)
    description = Column(Text, nullable=True)
    text_content = Column(Text, nullable=True)
    is_processed = Column(Boolean, default=False)
    is_indexed = Column(Boolean, default=False)
    processing_status = Column(String(50), default="pending")
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    # Relationships can be added here as needed

    def __repr__(self):
        return f"<Document(id={self.id}, filename='{self.filename}')>"

    @property
    def as_dict(self):
        """
        Convert the document to a dictionary for API responses.
        """
        return {
            "id": self.id,
            "filename": self.filename,
            "original_filename": self.original_filename,
            "file_size": self.file_size,
            "file_type": self.file_type,
            "content_type": self.content_type,
            "title": self.title,
            "description": self.description,
            "is_processed": self.is_processed,
            "is_indexed": self.is_indexed,
            "processing_status": self.processing_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
