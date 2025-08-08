"""
Factory for creating code completion clients based on configuration
"""
import logging
from typing import Optional
from backend.app.core.config import Settings
from .interfaces import CodeCompletionClient
from .null_client import NullCodeCompletionClient
from .tabby_client import TabbyCodeCompletionClient

logger = logging.getLogger(__name__)

_completion_client: Optional[CodeCompletionClient] = None

def code_completion_factory(settings: Settings) -> CodeCompletionClient:
    """
    Factory function to create appropriate code completion client
    
    Args:
        settings: Application settings
        
    Returns:
        CodeCompletionClient implementation based on configuration
    """
    global _completion_client
    
    # Return cached client if available
    if _completion_client is not None:
        return _completion_client
    
    # Create appropriate client based on feature flag
    if settings.ENABLE_TABBY:
        logger.info("TabbyML code completion enabled")
        _completion_client = TabbyCodeCompletionClient(
            base_url=settings.TABBY_URL,
            api_key=settings.TABBY_API_KEY
        )
    else:
        logger.info("Code completion disabled - using null client")
        _completion_client = NullCodeCompletionClient()
    
    return _completion_client

def reset_completion_client():
    """
    Reset the cached completion client (useful for testing)
    """
    global _completion_client
    _completion_client = None