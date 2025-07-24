from typing import List, Optional, Dict, Any
from sqlmodel import Field, Relationship, SQLModel, Column, JSON, func
from datetime import datetime
from enum import Enum


class TimestampMixin(SQLModel):
    """Mixin for adding created_at and updated_at timestamps to models."""

    created_at: Optional[datetime] = Field(default_factory=datetime.utcnow, sa_column_kwargs={"server_default": func.now()})
    updated_at: Optional[datetime] = Field(default=None, sa_column_kwargs={"onupdate": func.now()})


class User(TimestampMixin, table=True):
    """User model for authentication and authorization."""

    __tablename__ = "users"

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    email: str = Field(unique=True, index=True)
    username: str = Field(unique=True, index=True)
    hashed_password: str = Field()
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)

    # Relationships
    api_keys: List["APIKey"] = Relationship(back_populates="user")
    projects: List["Project"] = Relationship(back_populates="owner")


class APIKey(TimestampMixin, table=True):
    """API keys for user authentication."""

    __tablename__ = "api_keys"

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    key: str = Field(unique=True, index=True)
    name: str = Field()
    user_id: Optional[int] = Field(default=None, foreign_key="user.id")
    expires_at: Optional[datetime] = Field(default=None)
    is_active: bool = Field(default=True)

    # Relationships
    user: Optional[User] = Relationship(back_populates="api_keys")


class Project(TimestampMixin, table=True):
    """Project model for organizing AI tasks and documents."""

    __tablename__ = "projects"

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    name: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    owner_id: Optional[int] = Field(default=None, foreign_key="user.id")

    # Relationships
    owner: Optional[User] = Relationship(back_populates="projects")
    documents: List["Document"] = Relationship(back_populates="project")
    code_generations: List["CodeGeneration"] = Relationship(back_populates="project")
    diagram_parses: List["DiagramParse"] = Relationship(back_populates="project")


class Document(TimestampMixin, table=True):
    """Document model for storing uploaded documents."""

    __tablename__ = "documents"
    __table_args__ = {"extend_existing": True}

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    filename: Optional[str] = Field(default=None)
    content_type: Optional[str] = Field(default=None)
    size: Optional[int] = Field(default=None)
    path: Optional[str] = Field(default=None)
    processed: bool = Field(default=False)
    doc_metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")
    owner_id: Optional[int] = Field(default=None, foreign_key="user.id")

    # Relationships
    project: Optional[Project] = Relationship(back_populates="documents")
    chunks: List["DocumentChunk"] = Relationship(back_populates="document")
    owner: Optional[User] = Relationship()


class DocumentChunk(TimestampMixin, table=True):
    """Document chunk for storing vector embeddings of document content."""

    __tablename__ = "document_chunks"

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    document_id: Optional[int] = Field(default=None, foreign_key="document.id")
    content: Optional[str] = Field(default=None)
    chunk_index: Optional[int] = Field(default=None)
    doc_metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))

    # Relationships
    document: Optional[Document] = Relationship(back_populates="chunks")


# Define UserRole enum if not already defined globally
class UserRole(str, Enum):
    ADMIN = "admin"
    USER = "user"
    AGENT = "agent"


# Define Status enum (can be reused or specialized)
class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CodeGeneration(TimestampMixin, table=True):
    """Code generation task and results."""

    __tablename__ = "code_generations"

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    prompt: Optional[str] = Field(default=None)
    generated_code: Optional[str] = Field(default=None)
    language: Optional[str] = Field(default=None)
    model: Optional[str] = Field(default=None)
    status: Optional[TaskStatus] = Field(default=TaskStatus.PENDING) # Use Enum
    error: Optional[str] = Field(default=None)
    doc_metadata: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")

    # Relationships
    project: Optional[Project] = Relationship(back_populates="code_generations")


class DiagramParse(TimestampMixin, table=True):
    """Diagram parsing task and results."""

    __tablename__ = "diagram_parses"

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    filename: Optional[str] = Field(default=None)
    content_type: Optional[str] = Field(default=None)
    path: Optional[str] = Field(default=None)
    parsed_content: Optional[Dict[str, Any]] = Field(default=None, sa_column=Column(JSON))
    diagram_type: Optional[str] = Field(default=None)  # uml, flowchart, etc.
    status: Optional[TaskStatus] = Field(default=TaskStatus.PENDING) # Use Enum
    error: Optional[str] = Field(default=None)
    project_id: Optional[int] = Field(default=None, foreign_key="project.id")

    # Relationships
    project: Optional[Project] = Relationship(back_populates="diagram_parses")


class CodeSnippet(TimestampMixin, table=True):
    """Model for storing code snippets."""

    __tablename__ = "code_snippets"

    id: Optional[int] = Field(default=None, primary_key=True, index=True)
    title: str = Field(index=True)
    description: Optional[str] = Field(default=None)
    language: str = Field(index=True)
    code_content: str = Field()
    tags: Optional[List[str]] = Field(default=None, sa_column=Column(JSON)) # Store tags as JSON list
    is_public: bool = Field(default=False, index=True)
    owner_id: Optional[int] = Field(default=None, foreign_key="user.id", index=True)

    # Relationships
    owner: Optional[User] = Relationship()


# Replace Base with SQLModel for all models
