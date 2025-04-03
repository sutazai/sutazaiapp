"""
Code model for the SutazAI application.
"""

import datetime
from sqlalchemy import Column, Integer, String, Text, DateTime, Boolean, ForeignKey
from sqlalchemy.orm import relationship, Mapped, mapped_column
from typing import Optional, List, Dict, Any
import os
import logging
from pydantic import BaseModel, Field

from .base_models import Base
from backend.services.code_analysis.language_detection import detect_language
from backend.services.code_analysis.comment_extraction import extract_comments
from backend.services.code_analysis.structure_analysis import analyze_structure

logger = logging.getLogger(__name__)

class CodeSnippet(Base):
    """
    CodeSnippet model for storing generated code snippets.
    """

    __tablename__ = "code_snippets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    title: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    language: Mapped[str] = mapped_column(String(50), nullable=False)
    code_content: Mapped[str] = mapped_column(Text, nullable=False)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # Optional link to user
    is_public: Mapped[bool] = mapped_column(Boolean, default=False)
    tags: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)  # Comma-separated tags
    created_at: Mapped[Optional[datetime.datetime]] = mapped_column(DateTime, default=datetime.datetime.utcnow)
    updated_at: Mapped[Optional[datetime.datetime]] = mapped_column(
        DateTime, default=datetime.datetime.utcnow, onupdate=datetime.datetime.utcnow
    )

    def __repr__(self):
        return f"<CodeSnippet(id={self.id}, title='{self.title}')>"

    @property
    def as_dict(self):
        """
        Convert the code snippet to a dictionary for API responses.
        """
        tag_list: List[str] = []
        if self.tags:
            tag_list = [tag.strip() for tag in self.tags.split(",") if tag.strip()]

        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "code_content": self.code_content,
            "user_id": self.user_id,
            "is_public": self.is_public,
            "tags": tag_list,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }

class CodeBlock(BaseModel):
    language: Optional[str] = None
    content: str
    start_line: int
    end_line: int

class CodeAnalysisResult(BaseModel):
    language: Optional[str] = None
    comments: List[str] = Field(default_factory=list)
    structure: Optional[Dict[str, Any]] = None
    code_blocks: List[CodeBlock] = Field(default_factory=list)

class CodeModel(Base):
    """Represents a document containing code, inheriting from Document model."""
    analysis: Optional[CodeAnalysisResult] = None

    class Settings:
        name = "code_documents"

    def analyze_code(self) -> CodeAnalysisResult:
        """Performs analysis on the code content.""" # type: ignore [annotation-unchecked]
        if not self.content:
            return CodeAnalysisResult()

        detected_language = detect_language(self.content)
        comments = extract_comments(self.content, language=detected_language)
        structure = analyze_structure(self.content, language=detected_language)

        # Placeholder for code block extraction
        code_blocks = [
            CodeBlock(language=detected_language, content=self.content, start_line=0, end_line=len(self.content.splitlines()))
        ]

        analysis_result = CodeAnalysisResult(
            language=detected_language,
            comments=comments,
            structure=structure,
            code_blocks=code_blocks
        )
        self.analysis = analysis_result
        return analysis_result

def process_code_document(doc: Base):
    """Processes a generic document, performs code analysis if applicable."""
    code_model = CodeModel(**doc.dict())
    if code_model.content: # Check if content exists
        # Simple heuristic: if it contains common code patterns, analyze
        code_patterns = ['def ', 'class ', 'import ', 'function ', 'public static', '=>']
        if any(pattern in code_model.content for pattern in code_patterns):
             logger.info(f"Analyzing code in document: {doc.id}")
             code_model.analyze_code()
    return code_model
