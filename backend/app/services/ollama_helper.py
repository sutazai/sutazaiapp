"""
Ollama Helper Module
Provides utilities for interacting with Ollama service
"""

import httpx
import logging
import os
from typing import List, Dict, Optional, Any
from functools import lru_cache

from app.core.circuit_breaker import CircuitBreaker, CircuitBreakerOpen
from app.core.retry import async_retry

logger = logging.getLogger(__name__)

class OllamaHelper:
    """Helper class for Ollama operations"""
    
    def __init__(self):
        self.host = os.getenv("OLLAMA_HOST", "host.docker.internal")
        self.port = os.getenv("OLLAMA_PORT", "11434")
        self.base_url = f"http://{self.host}:{self.port}"
        self._available_models_cache = None
        # Circuit breaker for Ollama API calls
        self._circuit_breaker = CircuitBreaker(
            failure_threshold=5,
            timeout=30,  # 30 seconds for Ollama recovery
            expected_exception=(httpx.HTTPError, httpx.TimeoutException)
        )
        
    @async_retry(max_attempts=3, delay=1.0, backoff=2.0, exceptions=(httpx.HTTPError, httpx.TimeoutException))
    async def get_available_models(self) -> List[str]:
        """Get list of available Ollama models"""
        try:
            async def _get_models():
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(f"{self.base_url}/api/tags")
                    if response.status_code == 200:
                        data = response.json()
                        models = data.get("models", [])
                        return [model["name"] for model in models]
                    else:
                        logger.error(f"Failed to get models: {response.status_code}")
                        raise httpx.HTTPError(f"HTTP {response.status_code}")
            
            return await self._circuit_breaker.call(_get_models)
        except CircuitBreakerOpen:
            logger.warning("Circuit breaker open for Ollama - returning empty model list")
            return []
        except Exception as e:
            logger.error(f"Error getting Ollama models: {str(e)}")
            return []
    
    async def is_model_available(self, model_name: str) -> bool:
        """Check if a specific model is available"""
        available_models = await self.get_available_models()
        return model_name in available_models
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name}
                )
                if response.status_code == 200:
                    logger.info(f"Successfully pulled model: {model_name}")
                    return True
                else:
                    logger.error(f"Failed to pull model {model_name}: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            return False
    
    @async_retry(max_attempts=3, delay=2.0, backoff=2.0, exceptions=(httpx.HTTPError, httpx.TimeoutException))
    async def generate(
        self,
        model: str,
        prompt: str,
        temperature: float = 0.7,
        max_tokens: int = 2000,
        stream: bool = False
    ) -> Dict[str, Any]:
        """Generate text using Ollama model"""
        try:
            # Check if model is available
            if not await self.is_model_available(model):
                # Try to use a fallback model
                available = await self.get_available_models()
                if available:
                    logger.warning(f"Model {model} not found, using {available[0]}")
                    model = available[0]
                else:
                    raise Exception("No Ollama models available")
            
            async def _generate():
                async with httpx.AsyncClient(timeout=60.0) as client:
                    response = await client.post(
                        f"{self.base_url}/api/generate",
                        json={
                            "model": model,
                            "prompt": prompt,
                            "stream": stream,
                            "options": {
                                "temperature": temperature,
                                "num_predict": max_tokens
                            }
                        }
                    )
                    
                    if response.status_code == 200:
                        return response.json()
                    else:
                        raise httpx.HTTPError(f"Ollama API error: {response.status_code} - {response.text}")
            
            return await self._circuit_breaker.call(_generate)
                    
        except CircuitBreakerOpen:
            logger.error("Circuit breaker open for Ollama - cannot generate")
            raise Exception("Ollama service temporarily unavailable")
        except Exception as e:
            logger.error(f"Error generating with Ollama: {str(e)}")
            raise
    
    async def chat(
        self,
        model: str,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000
    ) -> str:
        """Chat completion using Ollama model"""
        try:
            # Check if model is available
            if not await self.is_model_available(model):
                available = await self.get_available_models()
                if available:
                    logger.warning(f"Model {model} not found, using {available[0]}")
                    model = available[0]
                else:
                    raise Exception("No Ollama models available")
            
            # Convert messages to prompt format
            prompt = self._messages_to_prompt(messages)
            
            result = await self.generate(
                model=model,
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return result.get("response", "")
            
        except Exception as e:
            logger.error(f"Error in Ollama chat: {str(e)}")
            raise
    
    def _messages_to_prompt(self, messages: List[Dict[str, str]]) -> str:
        """Convert chat messages to prompt format"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        # Add final assistant prompt
        prompt_parts.append("Assistant:")
        
        return "\n\n".join(prompt_parts)
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(
                    f"{self.base_url}/api/show",
                    json={"name": model_name}
                )
                if response.status_code == 200:
                    return response.json()
                else:
                    return None
        except Exception as e:
            logger.error(f"Error getting model info: {str(e)}")
            return None
    
    async def health_check(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            async def _check():
                async with httpx.AsyncClient(timeout=3.0) as client:
                    response = await client.get(f"{self.base_url}/api/tags")
                    if response.status_code == 200:
                        return True
                    raise httpx.HTTPError(f"HTTP {response.status_code}")
            
            return await self._circuit_breaker.call(_check)
        except Exception:
            return False

# Singleton instance
ollama_helper = OllamaHelper()