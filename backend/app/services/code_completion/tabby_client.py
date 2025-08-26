"""
TabbyML implementation of code completion client
"""
import httpx
import logging
from .interfaces import CodeCompletionClient, CompletionRequest, CompletionResponse

logger = logging.getLogger(__name__)

class TabbyCodeCompletionClient(CodeCompletionClient):
    """
    TabbyML implementation for code completion
    """
    
    def __init__(self, base_url: str, api_key: Optional[str] = None):
        """
        Initialize TabbyML client
        
        Args:
            base_url: Base URL for TabbyML service
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client = None
        self._available = None
    
    def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with lazy initialization"""
        if self._client is None:
            headers = {}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"
            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                headers=headers,
                timeout=httpx.Timeout(30.0)
            )
        return self._client
    
    async def complete(self, request: CompletionRequest) -> CompletionResponse:
        """
        Generate code completion using TabbyML
        """
        try:
            # Only import if we're actually using the feature
            client = self._get_client()
            
            # Prepare request payload for TabbyML API
            payload = {
                "prompt": request.code,
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }
            
            if request.language:
                payload["language"] = request.language
            
            # Call TabbyML completion endpoint
            response = await client.post("/v1/completions", json=payload)
            response.raise_for_status()
            
            result = response.json()
            
            # Extract completion from TabbyML response
            completion_text = result.get("choices", [{}])[0].get("text", "")
            
            return CompletionResponse(
                completion=completion_text,
                confidence=result.get("confidence"),
                metadata={
                    "service": "tabby",
                    "model": result.get("model"),
                    "language": request.language
                }
            )
            
        except httpx.HTTPError as e:
            logger.error(f"TabbyML HTTP error: {e}")
            return CompletionResponse(
                completion=f"# Error: TabbyML service unavailable - {str(e)}",
                confidence=0.0,
                metadata={"service": "tabby", "error": str(e)}
            )
        except Exception as e:
            logger.error(f"TabbyML completion error: {e}")
            return CompletionResponse(
                completion=f"# Error generating completion: {str(e)}",
                confidence=0.0,
                metadata={"service": "tabby", "error": str(e)}
            )
    
    async def health_check(self) -> bool:
        """
        Check if TabbyML service is healthy
        """
        try:
            client = self._get_client()
            response = await client.get("/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"TabbyML health check failed: {e}")
            return False
    
    def is_available(self) -> bool:
        """
        Check if TabbyML is configured and reachable
        """
        if self._available is None:
            # Cache availability check result
            import asyncio
            try:
                loop = asyncio.get_event_loop()
                self._available = loop.run_until_complete(self.health_check())
            except (IOError, OSError, FileNotFoundError) as e:
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                self._available = False
        return self._available
    
    async def __aenter__(self):
        """Async context manager entry"""
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit - cleanup client"""
        if self._client:
            await self._client.aclose()
