"""
Ollama Integration Agent
Handles all interactions with the Ollama LLM service.
Implements retry logic, error handling, and response validation.
"""

import asyncio
import hashlib
import json
import logging
import time
from typing import Optional, Dict, Any
from datetime import datetime

import aiohttp
from aiohttp import ClientTimeout
from pydantic import ValidationError

import sys
import os
sys.path.append('/opt/sutazaiapp')

from schemas.ollama_schemas import (
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaErrorResponse,
    OllamaModelsResponse
)

# Import metrics module
try:
    from agents.core.metrics import AgentMetrics, setup_metrics_endpoint, MetricsTimer
except ImportError:
    # Fallback if running in container
    sys.path.append('/app/agents')
    from core.metrics import AgentMetrics, setup_metrics_endpoint, MetricsTimer

# Configure structured logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class OllamaIntegrationAgent:
    """Agent for integrating with Ollama LLM service."""
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        timeout: int = 30,
        max_retries: int = 3,
        backoff_base: float = 2.0
    ):
        """
        Initialize Ollama integration agent.
        
        Args:
            base_url: Ollama API base URL
            timeout: Request timeout in seconds
            max_retries: Maximum retry attempts
            backoff_base: Base for exponential backoff calculation
        """
        self.base_url = base_url.rstrip('/')
        self.timeout = ClientTimeout(total=timeout)
        self.max_retries = max_retries
        self.backoff_base = backoff_base
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()
        
    async def start(self):
        """Start the HTTP session."""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
            logger.info(f"Started Ollama integration agent with base URL: {self.base_url}")
            
    async def close(self):
        """Close the HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
            logger.info("Closed Ollama integration agent")
            
    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay.
        
        Args:
            attempt: Current retry attempt (0-indexed)
            
        Returns:
            Delay in seconds
        """
        delay = self.backoff_base ** attempt
        # Add jitter to prevent thundering herd
        import random
        jitter = random.uniform(0, 0.1 * delay)
        return min(delay + jitter, 30)  # Cap at 30 seconds
        
    def _hash_request(self, request_data: Dict[str, Any]) -> str:
        """
        Create hash of request for tracking.
        
        Args:
            request_data: Request payload
            
        Returns:
            SHA256 hash of request
        """
        request_str = json.dumps(request_data, sort_keys=True)
        return hashlib.sha256(request_str.encode()).hexdigest()[:16]
        
    async def verify_model(self, model_name: str = "tinyllama") -> bool:
        """
        Verify that a specific model is available.
        
        Args:
            model_name: Name of model to check
            
        Returns:
            True if model is available
        """
        try:
            models = await self.list_models()
            has_model = models.has_model(model_name)
            
            if has_model:
                logger.info(f"Model '{model_name}' is available")
            else:
                logger.warning(f"Model '{model_name}' is not available")
                
            return has_model
            
        except Exception as e:
            logger.error(f"Failed to verify model: {e}")
            return False
            
    async def list_models(self) -> OllamaModelsResponse:
        """
        List available models from Ollama.
        
        Returns:
            Models response object
            
        Raises:
            Exception: If request fails after retries
        """
        if not self.session:
            await self.start()
            
        url = f"{self.base_url}/api/tags"
        
        async with self.session.get(url) as response:
            if response.status == 200:
                data = await response.json()
                return OllamaModelsResponse(**data)
            else:
                raise Exception(f"Failed to list models: HTTP {response.status}")
                
    async def generate(
        self,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 128,
        stop: Optional[list] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Generate text using Ollama.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop: Stop sequences
            **kwargs: Additional generation parameters
            
        Returns:
            Simplified response with text, tokens, and latency
            
        Raises:
            ValidationError: If request/response validation fails
            Exception: If all retries exhausted
        """
        # Create and validate request
        try:
            request = OllamaGenerateRequest(
                prompt=prompt,
                temperature=temperature,
                num_predict=max_tokens,
                stop=stop,
                **kwargs
            )
        except ValidationError as e:
            logger.error(f"Invalid request parameters: {e}")
            raise
            
        # Convert to Ollama payload format
        payload = request.to_ollama_payload()
        request_hash = self._hash_request(payload)
        
        logger.info(
            f"Generating text [hash={request_hash}] "
            f"prompt_length={len(prompt)} "
            f"max_tokens={max_tokens}"
        )
        
        # Attempt generation with retries
        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = await self._make_request(payload, request_hash, attempt)
                
                # Validate and return response
                ollama_response = OllamaGenerateResponse(**response)
                
                logger.info(
                    f"Generation successful [hash={request_hash}] "
                    f"tokens={ollama_response.tokens} "
                    f"latency={ollama_response.latency_ms:.2f}ms"
                )
                
                return ollama_response.to_simple_response()
                
            except aiohttp.ClientError as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    backoff = self._calculate_backoff(attempt)
                    logger.warning(
                        f"Request failed (attempt {attempt + 1}/{self.max_retries}) "
                        f"[hash={request_hash}]: {e}. "
                        f"Retrying in {backoff:.2f}s..."
                    )
                    await asyncio.sleep(backoff)
                else:
                    logger.error(
                        f"All retries exhausted [hash={request_hash}]: {e}"
                    )
                    
            except ValidationError as e:
                logger.error(
                    f"Invalid response from Ollama [hash={request_hash}]: {e}"
                )
                raise
                
        # Log failure with structured data
        error_log = OllamaErrorResponse(
            error=str(last_error),
            code=500,
            request_hash=request_hash
        )
        
        logger.error(
            f"Generation failed after {self.max_retries} attempts: "
            f"{json.dumps(error_log.dict())}"
        )
        
        raise Exception(f"Generation failed after {self.max_retries} retries: {last_error}")
        
    async def _make_request(
        self,
        payload: Dict[str, Any],
        request_hash: str,
        attempt: int
    ) -> Dict[str, Any]:
        """
        Make HTTP request to Ollama API.
        
        Args:
            payload: Request payload
            request_hash: Hash for tracking
            attempt: Current attempt number
            
        Returns:
            Response JSON
            
        Raises:
            aiohttp.ClientError: On HTTP errors
        """
        if not self.session:
            await self.start()
            
        url = f"{self.base_url}/api/generate"
        
        start_time = time.time()
        
        async with self.session.post(url, json=payload) as response:
            elapsed = (time.time() - start_time) * 1000
            
            # Handle different status codes
            if response.status == 200:
                return await response.json()
                
            elif response.status == 400:
                error_data = await response.text()
                logger.error(
                    f"Bad request [hash={request_hash}]: {error_data}"
                )
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=400,
                    message=f"Bad request: {error_data}"
                )
                
            elif response.status == 404:
                logger.error(
                    f"Model not found [hash={request_hash}]. "
                    "Ensure tinyllama:latest is pulled."
                )
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=404,
                    message="Model not found"
                )
                
            elif response.status == 429:
                # Rate limiting - check for Retry-After header
                retry_after = response.headers.get('Retry-After', '5')
                logger.warning(
                    f"Rate limited [hash={request_hash}]. "
                    f"Retry after {retry_after}s"
                )
                await asyncio.sleep(int(retry_after))
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=429,
                    message="Rate limited"
                )
                
            elif response.status >= 500:
                error_data = await response.text()
                logger.error(
                    f"Server error {response.status} [hash={request_hash}]: "
                    f"{error_data}"
                )
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"Server error: {error_data}"
                )
                
            else:
                raise aiohttp.ClientResponseError(
                    request_info=response.request_info,
                    history=response.history,
                    status=response.status,
                    message=f"Unexpected status: {response.status}"
                )


# FastAPI application for the agent
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse

app = FastAPI(title="Ollama Integration Agent")
agent: Optional[OllamaIntegrationAgent] = None
metrics: Optional[AgentMetrics] = None


@app.on_event("startup")
async def startup():
    """Initialize agent on startup."""
    global agent, metrics
    
    # Initialize metrics
    metrics = AgentMetrics("ollama_integration")
    setup_metrics_endpoint(app, metrics)
    
    # Get Ollama URL from environment variable, default to localhost
    ollama_url = os.environ.get('OLLAMA_BASE_URL', 'http://localhost:11434')
    logger.info(f"Initializing Ollama integration with URL: {ollama_url}")
    agent = OllamaIntegrationAgent(base_url=ollama_url)
    await agent.start()
    
    # Verify model is available
    if not await agent.verify_model("tinyllama"):
        logger.warning("TinyLlama model not found - generation may fail")
        metrics.set_health_status(False)
    else:
        metrics.set_health_status(True)
        

@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown."""
    global agent
    if agent:
        await agent.close()
        

@app.get("/health")
async def health():
    """Health check endpoint."""
    try:
        # Check if we can reach Ollama
        models = await agent.list_models()
        has_tinyllama = models.has_model("tinyllama")
        
        # Update health metric
        healthy = has_tinyllama
        metrics.set_health_status(healthy)
        
        return {
            "status": "healthy" if has_tinyllama else "degraded",
            "ollama_reachable": True,
            "tinyllama_available": has_tinyllama,
            "models_count": len(models.models)
        }
    except Exception as e:
        metrics.set_health_status(False)
        metrics.increment_error("health_check_error")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "ollama_reachable": False,
                "error": str(e)
            }
        )
        

@app.post("/generate")
async def generate(request: OllamaGenerateRequest):
    """
    Generate text using Ollama.
    
    Returns:
        response: Generated text
        tokens: Total token count
        latency: Response time in milliseconds
    """
    # Track metrics
    metrics.active_requests.labels(agent="ollama_integration").inc()
    metrics.last_request_timestamp.labels(agent="ollama_integration").set(time.time())
    
    start_time = time.time()
    status = "success"
    
    try:
        # Add queue latency simulation (time waiting for resources)
        queue_start = time.time()
        # In real scenario, this would be actual queue wait time
        await asyncio.sleep(0.001)  # Minimal delay
        metrics.record_queue_latency(time.time() - queue_start)
        
        result = await agent.generate(
            prompt=request.prompt,
            temperature=request.temperature,
            max_tokens=request.num_predict,
            stop=request.stop
        )
        return result
        
    except ValidationError as e:
        status = "error"
        metrics.increment_error("validation_error")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        status = "error"
        metrics.increment_error(type(e).__name__)
        logger.error(f"Generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Record request completion
        duration = time.time() - start_time
        metrics.request_count.labels(
            agent="ollama_integration",
            method="generate",
            status=status
        ).inc()
        metrics.processing_duration.labels(
            agent="ollama_integration",
            method="generate"
        ).observe(duration)
        metrics.active_requests.labels(agent="ollama_integration").dec()
        

@app.get("/models")
async def list_models():
    """List available Ollama models."""
    try:
        models = await agent.list_models()
        return {
            "models": [
                {
                    "name": m.name,
                    "size_mb": round(m.size_mb, 2)
                }
                for m in models.models
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8090)