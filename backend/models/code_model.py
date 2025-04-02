"""
Code model for the SutazAI application.
"""

import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean

from .base_models import Base


class CodeSnippet(Base):
    """
    CodeSnippet model for storing generated code snippets.
    """

    __tablename__ = "code_snippets"

    id = Column(Integer, primary_key=True, index=True)
    title = Column(String(255), nullable=False)
    description = Column(Text, nullable=True)
    language = Column(String(50), nullable=False)
    code_content = Column(Text, nullable=False)
    user_id = Column(Integer, nullable=True)  # Optional link to user
    is_public = Column(Boolean, default=False)
    tags = Column(String(255), nullable=True)  # Comma-separated tags
    created_at = Column(DateTime, default=datetime.datetime.utcnow)
    updated_at = Column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    def __repr__(self):
        return f"<CodeSnippet(id={self.id}, title='{self.title}')>"

    @property
    def as_dict(self):
        """
        Convert the code snippet to a dictionary for API responses.
        """
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "code_content": self.code_content,
            "user_id": self.user_id,
            "is_public": self.is_public,
            "tags": self.tags.split(",") if self.tags else [],
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }
