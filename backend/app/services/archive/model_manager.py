"""
Model Manager for SutazAI - Handles model loading, inference, and management
"""
import asyncio
import aiohttp
import json
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class ModelManager:
    """Manages AI models through Ollama"""
    
    def __init__(self):
        self.ollama_host = settings.OLLAMA_HOST
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        self.default_model = settings.DEFAULT_MODEL
        self.embedding_model = settings.EMBEDDING_MODEL
        self._session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize the model manager"""
        logger.info("Initializing Model Manager...")
        self._session = aiohttp.ClientSession()
        
        # Check Ollama connectivity
        if await self._check_ollama_health():
            logger.info("Ollama connection established")
            # List available models
            await self.list_models()
            
            # Preload default model for performance optimization
            if getattr(settings, 'MODEL_PRELOAD_ENABLED', True):
                logger.info(f"Preloading default model: {self.default_model}")
                await self._preload_model(self.default_model)
        else:
            logger.warning("Ollama service not available")
    
    async def cleanup(self):
        """Cleanup resources"""
        if self._session:
            await self._session.close()
    
    async def _check_ollama_health(self) -> bool:
        """Check if Ollama service is healthy"""
        try:
            async with self._session.get(f"{self.ollama_host}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            return False
    
    async def _preload_model(self, model_name: str) -> bool:
        """Preload a model to reduce cold start time"""
        try:
            logger.info(f"Warming up model: {model_name}")
            
            # Send a small test prompt to warm up the model
            warmup_data = {
                "model": model_name,
                "prompt": "Hello",
                "stream": False,
                "options": {
                    "num_predict": 1,  # Minimal prediction
                    "temperature": 0,
                    "top_p": 1,
                    "num_ctx": 512  # Small context window for warmup
                }
            }
            
            timeout = aiohttp.ClientTimeout(total=60)  # 60s timeout for warmup
            async with self._session.post(
                f"{self.ollama_host}/api/generate",
                json=warmup_data,
                timeout=timeout
            ) as response:
                if response.status == 200:
                    await response.json()  # Consume the response
                    logger.info(f"✅ Model {model_name} successfully preloaded and warmed up")
                    return True
                else:
                    logger.warning(f"⚠️ Model warmup failed for {model_name}: HTTP {response.status}")
                    return False
                    
        except asyncio.TimeoutError:
            logger.warning(f"⚠️ Model warmup timeout for {model_name}")
            return False
        except Exception as e:
            logger.error(f"❌ Error preloading model {model_name}: {e}")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models from Ollama"""
        try:
            async with self._session.get(f"{self.ollama_host}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    models = data.get("models", [])
                    
                    # Update loaded models cache
                    for model in models:
                        model_name = model.get("name", "")
                        self.loaded_models[model_name] = {
                            "name": model_name,
                            "size": model.get("size", 0),
                            "modified": model.get("modified_at", ""),
                            "loaded": True
                        }
                    
                    logger.info(f"Found {len(models)} models in Ollama")
                    return models
                return []
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model from Ollama registry"""
        try:
            logger.info(f"Pulling model: {model_name}")
            
            data = {"name": model_name}
            async with self._session.post(
                f"{self.ollama_host}/api/pull",
                json=data,
                timeout=aiohttp.ClientTimeout(total=3600)  # 1 hour timeout
            ) as response:
                if response.status == 200:
                    # Stream the response to track progress
                    async for line in response.content:
                        if line:
                            try:
                                progress = json.loads(line.decode())
                                if "status" in progress:
                                    logger.info(f"Pull progress: {progress.get('status')}")
                            except Exception as e:
                                # Suppressed exception (was bare except)
                                logger.debug(f"Suppressed exception: {e}")
                                pass
                    
                    logger.info(f"Successfully pulled model: {model_name}")
                    return True
                else:
                    logger.error(f"Failed to pull model: {response.status}")
                    return False
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def generate(self, prompt: str, model: Optional[str] = None, **kwargs) -> str:
        """Generate text using a model"""
        model = model or self.default_model
        
        try:
            # Add optimization parameters
            data = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "num_thread": 20,
                    "num_ctx": 2048,
                    "num_batch": 512,
                    "num_gpu": 0
                },
                **kwargs
            }
            
            async with self._session.post(
                f"{self.ollama_host}/api/generate",
                json=data,
                timeout=aiohttp.ClientTimeout(total=settings.MODEL_TIMEOUT)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("response", "")
                else:
                    logger.error(f"Generation failed with status: {response.status}")
                    return ""
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return ""
    
    async def chat(self, messages: List[Dict[str, str]], model: Optional[str] = None, **kwargs) -> str:
        """Chat with a model"""
        model = model or self.default_model
        
        try:
            # Add optimization parameters
            data = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": {
                    "num_thread": 20,
                    "num_ctx": 2048,
                    "num_batch": 512,
                    "num_gpu": 0
                },
                **kwargs
            }
            
            async with self._session.post(
                f"{self.ollama_host}/api/chat",
                json=data,
                timeout=aiohttp.ClientTimeout(total=settings.MODEL_TIMEOUT)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("message", {}).get("content", "")
                else:
                    logger.error(f"Chat failed with status: {response.status}")
                    return ""
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return ""
    
    async def embed(self, text: str, model: Optional[str] = None) -> List[float]:
        """Generate embeddings for text"""
        model = model or self.embedding_model
        
        try:
            data = {
                "model": model,
                "prompt": text
            }
            
            async with self._session.post(
                f"{self.ollama_host}/api/embeddings",
                json=data,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("embedding", [])
                else:
                    logger.error(f"Embedding failed with status: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            return []
    
    def get_status(self) -> Dict[str, Any]:
        """Get model manager status"""
        return {
            "status": "active" if self._session else "inactive",
            "loaded_count": len(self.loaded_models),
            "models": list(self.loaded_models.keys()),
            "default_model": self.default_model,
            "embedding_model": self.embedding_model
        }
    
    async def unload_model(self, model_name: str) -> bool:
        """Unload a model from memory"""
        try:
            data = {"name": model_name}
            async with self._session.delete(
                f"{self.ollama_host}/api/generate",
                json=data
            ) as response:
                if response.status == 200:
                    if model_name in self.loaded_models:
                        del self.loaded_models[model_name]
                    logger.info(f"Unloaded model: {model_name}")
                    return True
                return False
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
            return False