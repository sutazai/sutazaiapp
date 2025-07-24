"""
API Schema Definitions

Contains Pydantic models used for API request validation and response serialization.
"""

from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


# --- Authentication Schemas ---

class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Optional[str] = None


# --- User Schemas ---

# Base model for user properties shared across create/read/update
class UserBase(BaseModel):
    username: str
    email: EmailStr
    is_active: Optional[bool] = True
    is_superuser: bool = False


# Properties to receive via API on creation
class UserCreate(UserBase):
    password: str


# Properties to receive via API on update
class UserUpdate(BaseModel): # Can inherit UserBase if needed, or be separate
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    is_active: Optional[bool] = None
    is_superuser: Optional[bool] = None


# Properties to return via API (excluding sensitive info like password)
class UserRead(UserBase):
    id: int

    class Config:
        from_attributes = True # For compatibility with ORM models like SQLModel


# --- Document Schemas ---

# Base properties for a document
class DocumentBase(BaseModel):
    filename: str
    content_type: Optional[str] = None
    size: Optional[int] = None
    # Add other relevant metadata fields that are common
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    project_id: Optional[int] = None # If documents belong to projects


# Properties to receive via API on creation (e.g., when uploading metadata)
class DocumentCreate(DocumentBase):
    pass # Inherits all fields from DocumentBase


# Properties to receive via API on update
class DocumentUpdate(BaseModel):
    filename: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    tags: Optional[List[str]] = None
    # Add other updatable fields


# Properties to return via API
class DocumentRead(DocumentBase):
    id: int # Or str/UUID depending on the model
    path: Optional[str] = None # Maybe exclude internal path from API response?
    processed: bool = False
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    owner_id: Optional[int] = None # If linked to a user
    doc_metadata: Optional[Dict[str, Any]] = None # Catch-all for other metadata

    class Config:
        from_attributes = True

# --- Chat Schemas ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    agent: str # Agent identifier (e.g., config ID)
    messages: List[ChatMessage]
    parameters: Optional[Dict[str, Any]] = None
    # Add other relevant fields like session_id if needed

class ChatResponse(BaseModel):
    status: str
    response: str # The assistant's reply
    usage: Optional[Dict[str, Any]] = None
    agent: Optional[str] = None
    instance_id: Optional[str] = None
    error: Optional[str] = None


# --- Document Analysis Schemas ---

class DocumentAnalysisRequest(BaseModel):
    document_id: str # ID of the uploaded document
    analysis_type: str # e.g., "summary", "extraction", "qa"
    extraction_fields: Optional[List[str]] = None
    question: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None # Agent-specific params

# Define a response model for analysis if needed
# class DocumentAnalysisResponse(BaseModel): ...


# --- Code Generation Schemas ---

class CodeGenerationRequest(BaseModel):
    requirements: str
    language: str
    mode: str # e.g., "script", "function", "class"
    existing_code: Optional[str] = None
    generate_tests: Optional[bool] = False
    parameters: Optional[Dict[str, Any]] = None # Agent-specific params

# Define a response model for code generation if needed
# class CodeGenerationResponse(BaseModel): ...


# --- Code Execution Schemas ---

class CodeExecutionRequest(BaseModel):
    code: str
    language: str = "python"
    timeout: Optional[int] = 30

# Define a response model for code execution if needed
# class CodeExecutionResponse(BaseModel): ...


# --- System Control Schemas ---

class ServiceControlRequest(BaseModel):
    service: str
    action: str # e.g., "start", "stop", "restart"

class ModelControlRequest(BaseModel):
    model: str
    action: str # e.g., "load", "unload"

class LogRequest(BaseModel):
    service: str = "All"
    level: str = "ALL"
    lines: int = 50

# --- Code Snippet Schemas ---

class CodeSnippetBase(BaseModel):
    title: str
    description: Optional[str] = None
    language: str
    code_content: str
    is_public: bool = False
    tags: Optional[List[str]] = None # Expect list for creation/update

class CodeSnippetCreate(CodeSnippetBase):
    pass

class CodeSnippetUpdate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    language: Optional[str] = None
    code_content: Optional[str] = None
    is_public: Optional[bool] = None
    tags: Optional[List[str]] = None

class CodeSnippetRead(CodeSnippetBase):
    id: int
    owner_id: Optional[int] = None
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

# --- Model Management Schemas ---

class DownloadModelRequest(BaseModel):
    model_id: str = Field(..., description="The ID of the model to download")
    force: bool = Field(False, description="Force download even if model exists")

    # class Config:
        # protected_namespaces = () # No longer needed in Pydantic v2+

class ModelResponse(BaseModel):
    # Define fields expected in model list/status responses
    id: str
    name: Optional[str] = None
    type: Optional[str] = None # e.g., text-generation, code-generation
    is_local: Optional[bool] = None
    status: Optional[str] = None # e.g., available, loading, unavailable
    file_exists: Optional[bool] = None # For local models

class ModelStatusResponse(BaseModel):
    status: str
    loaded_models: Optional[List[str]] = None
    available_ram: Optional[str] = None
    gpu_available: Optional[bool] = None

class InferenceRequest(BaseModel):
    model_id: Optional[str] = None # Uses default if None
    prompt: str
    # Add other common inference parameters
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.7

class InferenceResponse(BaseModel):
    model: str
    prompt: str
    completion: str
    usage: Optional[Dict[str, int]] = None # e.g., {"prompt_tokens": 50, "completion_tokens": 100, "total_tokens": 150}


# Add other schemas as needed... 