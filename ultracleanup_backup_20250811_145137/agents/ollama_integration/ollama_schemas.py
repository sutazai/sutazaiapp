"""
Pydantic schemas for Ollama LLM integration.
Validates request/response payloads for the Ollama API.
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field, validator
from datetime import datetime


class OllamaGenerateRequest(BaseModel):
    """Schema for Ollama /api/generate endpoint request."""
    
    model: str = Field(
        default="tinyllama:latest",
        description="Model name to use for generation"
    )
    prompt: str = Field(
        ...,
        min_length=1,
        max_length=32768,
        description="Input prompt for text generation"
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream the response"
    )
    options: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Model-specific options"
    )
    
    # Generation parameters
    temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling threshold"
    )
    top_k: int = Field(
        default=40,
        ge=1,
        le=100,
        description="Top-k sampling parameter"
    )
    num_predict: int = Field(
        default=128,
        ge=1,
        le=2048,
        description="Maximum tokens to generate"
    )
    stop: Optional[List[str]] = Field(
        default=None,
        description="Stop sequences for generation"
    )
    seed: Optional[int] = Field(
        default=None,
        description="Random seed for reproducibility"
    )
    
    @validator('prompt')
    def validate_prompt(cls, v):
        """Ensure prompt is not just whitespace."""
        if not v.strip():
            raise ValueError("Prompt cannot be empty or whitespace only")
        return v
    
    @validator('stop')
    def validate_stop_sequences(cls, v):
        """Validate stop sequences if provided."""
        if v is not None:
            if len(v) > 5:
                raise ValueError("Maximum 5 stop sequences allowed")
            for seq in v:
                if len(seq) > 50:
                    raise ValueError("Stop sequence too long (max 50 chars)")
        return v
    
    def to_ollama_payload(self) -> Dict[str, Any]:
        """Convert to Ollama API payload format."""
        payload = {
            "model": self.model,
            "prompt": self.prompt,
            "stream": self.stream
        }
        
        # Add options if customized
        options = {}
        if self.temperature != 0.7:
            options["temperature"] = self.temperature
        if self.top_p != 0.9:
            options["top_p"] = self.top_p
        if self.top_k != 40:
            options["top_k"] = self.top_k
        if self.num_predict != 128:
            options["num_predict"] = self.num_predict
        if self.seed is not None:
            options["seed"] = self.seed
        if self.stop:
            options["stop"] = self.stop
            
        if options:
            payload["options"] = options
            
        return payload


class OllamaGenerateResponse(BaseModel):
    """Schema for Ollama /api/generate endpoint response."""
    
    model: str = Field(
        ...,
        description="Model used for generation"
    )
    created_at: str = Field(
        ...,
        description="Timestamp of generation"
    )
    response: str = Field(
        ...,
        description="Generated text response"
    )
    done: bool = Field(
        ...,
        description="Whether generation is complete"
    )
    context: Optional[List[int]] = Field(
        default=None,
        description="Context vector for conversation"
    )
    total_duration: Optional[int] = Field(
        default=None,
        description="Total duration in nanoseconds"
    )
    load_duration: Optional[int] = Field(
        default=None,
        description="Model load duration in nanoseconds"
    )
    prompt_eval_count: Optional[int] = Field(
        default=None,
        description="Number of tokens in prompt"
    )
    prompt_eval_duration: Optional[int] = Field(
        default=None,
        description="Prompt evaluation duration in nanoseconds"
    )
    eval_count: Optional[int] = Field(
        default=None,
        description="Number of tokens generated"
    )
    eval_duration: Optional[int] = Field(
        default=None,
        description="Generation duration in nanoseconds"
    )
    
    @property
    def tokens(self) -> int:
        """Total token count (prompt + generated)."""
        prompt_tokens = self.prompt_eval_count or 0
        generated_tokens = self.eval_count or 0
        return prompt_tokens + generated_tokens
    
    @property
    def latency_ms(self) -> float:
        """Total latency in milliseconds."""
        if self.total_duration:
            return self.total_duration / 1_000_000
        return 0.0
    
    @property
    def tokens_per_second(self) -> float:
        """Generation speed in tokens/second."""
        if self.eval_duration and self.eval_count:
            return self.eval_count / (self.eval_duration / 1_000_000_000)
        return 0.0
    
    def to_simple_response(self) -> Dict[str, Any]:
        """Convert to simplified response format."""
        return {
            "response": self.response,
            "tokens": self.tokens,
            "latency": self.latency_ms,
            "tokens_per_second": round(self.tokens_per_second, 2)
        }


class OllamaErrorResponse(BaseModel):
    """Schema for Ollama error responses."""
    
    error: str = Field(
        ...,
        description="Error message from Ollama"
    )
    code: Optional[int] = Field(
        default=None,
        description="HTTP status code"
    )
    timestamp: datetime = Field(
        default_factory=datetime.utcnow,
        description="When the error occurred"
    )
    request_hash: Optional[str] = Field(
        default=None,
        description="Hash of the failed request for tracking"
    )


class OllamaModelInfo(BaseModel):
    """Schema for model information from /api/tags."""
    
    name: str = Field(
        ...,
        description="Model name"
    )
    modified_at: str = Field(
        ...,
        description="Last modification timestamp"
    )
    size: int = Field(
        ...,
        description="Model size in bytes"
    )
    digest: str = Field(
        ...,
        description="Model digest/hash"
    )
    
    @property
    def size_mb(self) -> float:
        """Model size in megabytes."""
        return self.size / (1024 * 1024)


class OllamaModelsResponse(BaseModel):
    """Schema for /api/tags endpoint response."""
    
    models: List[OllamaModelInfo] = Field(
        ...,
        description="List of available models"
    )
    
    def has_model(self, model_name: str) -> bool:
        """Check if a specific model is available."""
        return any(
            model.name == model_name or 
            model.name.startswith(f"{model_name}:")
            for model in self.models
        )