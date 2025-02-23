from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """
    Schema for chat request with model selection and optional parameters.
    """
    model: str = Field(default="deepseek-coder", description="Model to use for chat")
    prompt: str = Field(..., min_length=1, max_length=2000, description="User's chat prompt")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context for the chat")
    temperature: float = Field(default=0.7, ge=0.0, le=1.0, description="Sampling temperature")
    max_tokens: int = Field(default=500, ge=10, le=2048, description="Maximum tokens to generate")

class ReportRequest(BaseModel):
    """
    Schema for report generation request.
    """
    format: str = Field(default="markdown", description="Output format for the report")
    topic: str = Field(..., min_length=1, max_length=200, description="Topic of the report")
    detail_level: str = Field(default="medium", description="Level of detail for the report")
    language: Optional[str] = Field(default="en", description="Language of the report")