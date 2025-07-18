#!/usr/bin/env python3
"""
SutazAI Model Manager
Manages AI models across different services
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx
import json

from ..core.config import settings

logger = logging.getLogger(__name__)


class ModelManager:
    """Manages AI models and their lifecycle"""
    
    def __init__(self, http_client: httpx.AsyncClient):
        self.http_client = http_client
        self.loaded_models = {}
        self.model_configs = {}
        self._initialized = False
    
    async def initialize(self):
        """Initialize the model manager"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing Model Manager...")
            
            # Load model configurations
            await self._load_model_configs()
            
            # Check available models
            await self._discover_available_models()
            
            # Load default models
            await self._load_default_models()
            
            self._initialized = True
            logger.info("Model Manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Model Manager: {e}")
            raise
    
    async def _load_model_configs(self):
        """Load model configurations"""
        self.model_configs = {
            "ollama": {
                "url": settings.OLLAMA_URL,
                "models": [
                    "deepseek-coder:6.7b",
                    "deepseek-coder:33b",
                    "llama2:7b",
                    "llama2:13b",
                    "codellama:7b",
                    "codellama:13b",
                    "mistral:7b",
                    "phi:2.7b",
                    "qwen:7b",
                    "gemma:7b"
                ],
                "default_model": settings.DEFAULT_MODEL,
                "max_concurrent": settings.MAX_LOADED_MODELS
            },
            "pytorch": {
                "url": settings.PYTORCH_URL,
                "models": [
                    "bert-base-uncased",
                    "gpt2",
                    "t5-base",
                    "roberta-base",
                    "distilbert-base-uncased"
                ],
                "framework": "pytorch"
            },
            "tensorflow": {
                "url": settings.TENSORFLOW_URL,
                "models": [
                    "mobilenet_v2",
                    "resnet50",
                    "inception_v3",
                    "efficientnet_b0"
                ],
                "framework": "tensorflow"
            },
            "jax": {
                "url": settings.JAX_URL,
                "models": [
                    "flax-gpt2",
                    "flax-bert",
                    "flax-t5"
                ],
                "framework": "jax"
            }
        }
    
    async def _discover_available_models(self):
        """Discover available models from services"""
        for service_name, config in self.model_configs.items():
            try:
                response = await self.http_client.get(f"{config['url']}/models")
                if response.status_code == 200:
                    available_models = response.json()
                    config["available_models"] = available_models.get("models", [])
                    logger.info(f"Discovered {len(config['available_models'])} models in {service_name}")
            except Exception as e:
                logger.warning(f"Could not discover models from {service_name}: {e}")
                config["available_models"] = []
    
    async def _load_default_models(self):
        """Load default models"""
        try:
            # Load default Ollama model
            default_model = settings.DEFAULT_MODEL
            await self.load_model("ollama", default_model)
            
        except Exception as e:
            logger.warning(f"Could not load default models: {e}")
    
    async def shutdown(self):
        """Shutdown the model manager"""
        logger.info("Shutting down Model Manager...")
        
        # Unload all models
        for model_id in list(self.loaded_models.keys()):
            await self.unload_model(model_id)
        
        self._initialized = False
        logger.info("Model Manager shutdown complete")
    
    async def load_model(self, service: str, model_name: str, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Load a model"""
        try:
            if service not in self.model_configs:
                raise ValueError(f"Unknown service: {service}")
            
            service_config = self.model_configs[service]
            
            # Check if model is already loaded
            model_id = f"{service}:{model_name}"
            if model_id in self.loaded_models:
                return self.loaded_models[model_id]
            
            # Check service-specific limits
            if service == "ollama":
                loaded_ollama_count = len([m for m in self.loaded_models.values() if m["service"] == "ollama"])
                if loaded_ollama_count >= service_config["max_concurrent"]:
                    raise ValueError(f"Maximum concurrent Ollama models reached: {service_config['max_concurrent']}")
            
            # Load model
            load_request = {
                "model": model_name,
                "config": config or {}
            }
            
            response = await self.http_client.post(
                f"{service_config['url']}/load",
                json=load_request
            )
            
            if response.status_code == 200:
                model_data = {
                    "model_id": model_id,
                    "service": service,
                    "model_name": model_name,
                    "status": "loaded",
                    "loaded_at": datetime.utcnow().isoformat(),
                    "config": config or {},
                    "usage_count": 0,
                    "last_used": None
                }
                
                self.loaded_models[model_id] = model_data
                
                logger.info(f"Loaded model {model_name} on {service}")
                return model_data
            else:
                raise Exception(f"Failed to load model: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to load model {model_name} on {service}: {e}")
            raise
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model"""
        try:
            if model_id not in self.loaded_models:
                return False
            
            model_data = self.loaded_models[model_id]
            service_config = self.model_configs[model_data["service"]]
            
            # Unload model
            response = await self.http_client.post(
                f"{service_config['url']}/unload",
                json={"model": model_data["model_name"]}
            )
            
            # Remove from loaded models
            del self.loaded_models[model_id]
            
            logger.info(f"Unloaded model {model_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_id}: {e}")
            return False
    
    async def get_loaded_models(self) -> List[Dict[str, Any]]:
        """Get all loaded models"""
        return list(self.loaded_models.values())
    
    async def get_loaded_models_count(self) -> int:
        """Get count of loaded models"""
        return len(self.loaded_models)
    
    async def get_available_models(self, service: str = None) -> Dict[str, List[str]]:
        """Get available models by service"""
        if service:
            if service in self.model_configs:
                return {service: self.model_configs[service].get("available_models", [])}
            return {}
        
        available = {}
        for service_name, config in self.model_configs.items():
            available[service_name] = config.get("available_models", [])
        
        return available
    
    async def generate_text(self, model_id: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using a model"""
        try:
            if model_id not in self.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model_data = self.loaded_models[model_id]
            service_config = self.model_configs[model_data["service"]]
            
            # Generate text
            generation_request = {
                "model": model_data["model_name"],
                "prompt": prompt,
                **kwargs
            }
            
            response = await self.http_client.post(
                f"{service_config['url']}/generate",
                json=generation_request
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Update usage statistics
                model_data["usage_count"] += 1
                model_data["last_used"] = datetime.utcnow().isoformat()
                
                return result
            else:
                raise Exception(f"Generation failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to generate text with model {model_id}: {e}")
            raise
    
    async def chat_completion(self, model_id: str, messages: List[Dict[str, str]], **kwargs) -> Dict[str, Any]:
        """Chat completion using a model"""
        try:
            if model_id not in self.loaded_models:
                raise ValueError(f"Model {model_id} not loaded")
            
            model_data = self.loaded_models[model_id]
            service_config = self.model_configs[model_data["service"]]
            
            # Chat completion
            chat_request = {
                "model": model_data["model_name"],
                "messages": messages,
                **kwargs
            }
            
            response = await self.http_client.post(
                f"{service_config['url']}/chat",
                json=chat_request
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Update usage statistics
                model_data["usage_count"] += 1
                model_data["last_used"] = datetime.utcnow().isoformat()
                
                return result
            else:
                raise Exception(f"Chat completion failed: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Failed to complete chat with model {model_id}: {e}")
            raise
    
    async def get_model_stats(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model statistics"""
        if model_id not in self.loaded_models:
            return None
        
        model_data = self.loaded_models[model_id]
        service_config = self.model_configs[model_data["service"]]
        
        try:
            response = await self.http_client.get(
                f"{service_config['url']}/stats/{model_data['model_name']}"
            )
            
            if response.status_code == 200:
                stats = response.json()
                stats.update({
                    "usage_count": model_data["usage_count"],
                    "last_used": model_data["last_used"],
                    "loaded_at": model_data["loaded_at"]
                })
                return stats
        except Exception as e:
            logger.warning(f"Could not get stats for model {model_id}: {e}")
        
        return {
            "usage_count": model_data["usage_count"],
            "last_used": model_data["last_used"],
            "loaded_at": model_data["loaded_at"]
        }
    
    async def optimize_model_performance(self):
        """Optimize model performance"""
        try:
            logger.info("Optimizing model performance...")
            
            # Unload unused models
            current_time = datetime.utcnow()
            for model_id, model_data in list(self.loaded_models.items()):
                if model_data["last_used"]:
                    last_used = datetime.fromisoformat(model_data["last_used"])
                    if (current_time - last_used).total_seconds() > 3600:  # 1 hour
                        await self.unload_model(model_id)
                        logger.info(f"Unloaded unused model {model_id}")
            
            # Load frequently used models
            # This would be implemented based on usage patterns
            
            logger.info("Model performance optimization completed")
            
        except Exception as e:
            logger.error(f"Model optimization failed: {e}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for model manager"""
        try:
            health_data = {
                "status": "healthy",
                "loaded_models": len(self.loaded_models),
                "services": list(self.model_configs.keys()),
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Check service health
            service_health = {}
            for service_name, config in self.model_configs.items():
                try:
                    response = await self.http_client.get(f"{config['url']}/health")
                    service_health[service_name] = {
                        "status": "healthy" if response.status_code == 200 else "unhealthy",
                        "response_time": response.elapsed.total_seconds() if hasattr(response, 'elapsed') else 0
                    }
                except Exception as e:
                    service_health[service_name] = {
                        "status": "unreachable",
                        "error": str(e)
                    }
            
            health_data["service_health"] = service_health
            
            return health_data
            
        except Exception as e:
            logger.error(f"Model manager health check failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }