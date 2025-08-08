"""
Null implementation of code completion client (when feature is disabled)
"""
from .interfaces import CodeCompletionClient, CompletionRequest, CompletionResponse

class NullCodeCompletionClient(CodeCompletionClient):
    """
    Null implementation that returns placeholder responses when feature is disabled
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