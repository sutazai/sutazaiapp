"""
Null Object Pattern implementation for code completion client

IMPORTANT: This is a deliberate design pattern, not a mock or stub.
The Null Object Pattern provides a default object with neutral behavior,
preventing null reference errors when the actual service is disabled.

This implementation is REQUIRED for:
1. Graceful degradation when ENABLE_TABBY=false
2. Testing without external dependencies
3. Development environments without code completion services

DO NOT REMOVE - This is production code following the Null Object Pattern
"""
from .interfaces import CodeCompletionClient, CompletionRequest, CompletionResponse

class NullCodeCompletionClient(CodeCompletionClient):
    """
    Null Object Pattern implementation for disabled code completion feature
    
    Returns informative messages instead of errors when the feature is disabled.
    This is NOT a mock - it's a production implementation of the Null Object Pattern.
    """
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Return a placeholder response indicating feature is disabled
        """
        return CompletionResponse(
            completion="# Code completion is disabled. Enable ENABLE_TABBY=true to use this feature.",
            confidence=0.0,
            metadata={
                "service": "null",
                "enabled": False,
                "message": "Code completion feature is disabled"
            }
        )
    
    async def health_check(self) -> bool:
        """
        Always returns True as null client is always "healthy"
        """
        return True
    
    def is_available(self) -> bool:
        """
        Null client is always available (but doesn't do real completions)
        """
        return True