from sqlalchemy import (
    Column,
    Integer,
    String,
    DateTime,
    Boolean,
    ForeignKey,
    Text,
    JSON,
)
from sqlalchemy.sql import func
from sqlalchemy.orm import relationship, declarative_base, Mapped, mapped_column
from sqlalchemy.ext.declarative import declared_attr
from typing import Any, List, Optional, Dict

from backend.core.database import Base


class TimestampMixin:
    """Mixin for adding created_at and updated_at timestamps to models."""

    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())


class User(Base, TimestampMixin):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True)
    username = Column(String, unique=True, index=True)
    hashed_password = Column(String)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)

    # Relationships
    api_keys = relationship(
        "APIKey", back_populates="user", cascade="all, delete-orphan"
    )
    projects = relationship("Project", back_populates="owner")


class APIKey(Base, TimestampMixin):
    """API keys for user authentication."""

    __tablename__ = "api_keys"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String, unique=True, index=True)
    name = Column(String)
    user_id = Column(Integer, ForeignKey("users.id"))
    expires_at = Column(DateTime(timezone=True), nullable=True)
    is_active = Column(Boolean, default=True)

    # Relationships
    user = relationship("User", back_populates="api_keys")


class Project(Base, TimestampMixin):
    """Project model for organizing AI tasks and documents."""

    __tablename__ = "projects"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, index=True)
    description = Column(Text, nullable=True)
    owner_id = Column(Integer, ForeignKey("users.id"))

    # Relationships
    owner = relationship("User", back_populates="projects")
    documents = relationship(
        "Document", back_populates="project", cascade="all, delete-orphan"
    )
    code_generations = relationship(
        "CodeGeneration", back_populates="project", cascade="all, delete-orphan"
    )
    diagram_parses = relationship(
        "DiagramParse", back_populates="project", cascade="all, delete-orphan"
    )


class Document(Base, TimestampMixin):
    """Document model for storing uploaded documents."""

    __tablename__ = "documents"
    __table_args__ = {"extend_existing": True}

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    filename: Mapped[Optional[str]] = mapped_column(String)
    content_type: Mapped[Optional[str]] = mapped_column(String)
    size: Mapped[Optional[int]] = mapped_column(Integer)
    path: Mapped[Optional[str]] = mapped_column(String)
    processed: Mapped[bool] = mapped_column(Boolean, default=False)
    doc_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)
    project_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("projects.id"))

    # Relationships
    project: Mapped[Optional["Project"]] = relationship(back_populates="documents")
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class DocumentChunk(Base, TimestampMixin):
    """Document chunk for storing vector embeddings of document content."""

    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    document_id: Mapped[Optional[int]] = mapped_column(Integer, ForeignKey("documents.id"))
    content: Mapped[Optional[str]] = mapped_column(Text)
    chunk_index: Mapped[Optional[int]] = mapped_column(Integer)
    doc_metadata: Mapped[Optional[Dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Relationships
    document: Mapped[Optional["Document"]] = relationship(back_populates="chunks")


class CodeGeneration(Base, TimestampMixin):
    """Code generation task and results."""

    __tablename__ = "code_generations"

    id = Column(Integer, primary_key=True, index=True)
    prompt = Column(Text)
    generated_code = Column(Text, nullable=True)
    language = Column(String)
    model = Column(String)
    status = Column(String)  # pending, processing, completed, failed
    error = Column(Text, nullable=True)
    doc_metadata = Column(JSON, nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"))

    # Relationships
    project = relationship("Project", back_populates="code_generations")


class DiagramParse(Base, TimestampMixin):
    """Diagram parsing task and results."""

    __tablename__ = "diagram_parses"

    id = Column(Integer, primary_key=True, index=True)
    filename = Column(String)
    content_type = Column(String)
    path = Column(String)
    parsed_content = Column(JSON, nullable=True)
    diagram_type = Column(String)  # uml, flowchart, etc.
    status = Column(String)  # pending, processing, completed, failed
    error = Column(Text, nullable=True)
    project_id = Column(Integer, ForeignKey("projects.id"))

    # Relationships
    project = relationship("Project", back_populates="diagram_parses")
