"""
Custom exceptions for SutazAI
"""
from typing import Any, Dict, Optional

class SutazAIException(Exception):
    """Base exception for SutazAI"""
    
    def __init__(
        self,
        status_code: int,
        detail: str,
        headers: Optional[Dict[str, Any]] = None
    ):
        self.status_code = status_code
        self.detail = detail
        self.headers = headers
        super().__init__(detail)

class AgentNotFoundError(SutazAIException):
    """Raised when an agent is not found"""
    
    def __init__(self, agent_name: str):
        super().__init__(
            status_code=404,
            detail=f"Agent '{agent_name}' not found"
        )

class AgentExecutionError(SutazAIException):
    """Raised when agent execution fails"""
    
    def __init__(self, message: str):
        super().__init__(
            status_code=500,
            detail=f"Agent execution failed: {message}"
        )

class ModelNotFoundError(SutazAIException):
    """Raised when a model is not found"""
    
    def __init__(self, model_name: str):
        super().__init__(
            status_code=404,
            detail=f"Model '{model_name}' not found"
        )

class ResourceLimitError(SutazAIException):
    """Raised when resource limits are exceeded"""
    
    def __init__(self, resource: str, limit: Any):
        super().__init__(
            status_code=429,
            detail=f"Resource limit exceeded for {resource}: {limit}"
        )

class AuthenticationError(SutazAIException):
    """Raised for authentication failures"""
    
    def __init__(self, detail: str = "Authentication failed"):
        super().__init__(
            status_code=401,
            detail=detail,
            headers={"WWW-Authenticate": "Bearer"}
        )

class AuthorizationError(SutazAIException):
    """Raised for authorization failures"""
    
    def __init__(self, detail: str = "Not authorized"):
        super().__init__(
            status_code=403,
            detail=detail
        )

class ValidationError(SutazAIException):
    """Raised for validation errors"""
    
    def __init__(self, detail: str):
        super().__init__(
            status_code=422,
            detail=detail
        )