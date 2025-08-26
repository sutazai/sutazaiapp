"""
Code completion service interfaces
"""
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class CompletionRequest:
    """Request for code completion"""
    code: str
    language: Optional[str] = None
    context: Optional[str] = None
    max_tokens: int = 150
    temperature: float = 0.2
    
@dataclass
class CompletionResponse:
    """Response from code completion"""
    completion: str
    confidence: Optional[float] = None
    metadata: Optional[Dict[str, Any]] = None
    
class CodeCompletionClient(ABC):
    """Abstract base class for code completion clients"""
    
    @abstractmethod
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate code completion
        
        Args:
            request: Completion request parameters
            
        Returns:
            CompletionResponse with generated code
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> bool:
        """
        Check if the completion service is healthy
        
        Returns:
            True if service is available
        """
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """
        Check if this completion service is configured and available
        
        Returns:
            True if service can be used
        """
        pass