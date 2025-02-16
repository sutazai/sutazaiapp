from functools import lru_cache
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer
from .model_manager import ModelManager
from .monitoring import monitor_requests
from config import settings
from .code_generator import CODEGEN_MODEL

# Security configuration
security = HTTPBearer()

@lru_cache(maxsize=None)
def get_model_manager():
    """
    Cached model manager dependency
    
    Returns:
        ModelManager: Singleton instance of ModelManager
    """
    return ModelManager()

@lru_cache(maxsize=None)
def get_current_user(token: str = Depends(security)):
    """
    Authenticate and retrieve current user
    
    Args:
        token (str): Bearer token for authentication
    
    Returns:
        dict: User information
    
    Raises:
        HTTPException: If authentication fails
    """
    try:
        # Implement actual token validation logic
        user = validate_token(token)
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid authentication credentials")

def validate_token(token: str):
    """
    Token validation helper function
    
    Args:
        token (str): Authentication token
    
    Returns:
        dict: Validated user information
    """
    # Implement actual token validation logic
    # This is a placeholder and should be replaced with actual implementation
    if not token:
        raise ValueError("Invalid token")
    
    return {
        "user_id": "example_user",
        "permissions": ["read", "write"]
    }

def rate_limit_dependency():
    """
    Rate limiting dependency
    
    Returns:
        bool: Whether the request is allowed
    
    Raises:
        HTTPException: If rate limit is exceeded
    """
    # Implement rate limiting logic
    # This is a placeholder and should be replaced with actual implementation
    return True

def monitor_request_dependency():
    """
    Request monitoring dependency
    
    Returns:
        None
    """
    monitor_requests()

@lru_cache(maxsize=None)
def get_codegen_model():
    return CODEGEN_MODEL

@lru_cache(maxsize=None)
def get_monitoring():
    return monitor_requests

@lru_cache(maxsize=None)
def get_settings():
    return settings 