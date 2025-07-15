"""Ollama Integration for Local Models"""
import asyncio
import logging
import aiohttp
import json
from typing import Dict, List, Any, Optional

logger = logging.getLogger(__name__)

class OllamaManager:
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.available_models = []
        self.loaded_models = set()
    
    async def initialize(self):
        """Initialize Ollama manager"""
        logger.info("🦙 Initializing Ollama Manager")
        
        if await self._check_ollama_status():
            await self._refresh_model_list()
            logger.info("✅ Ollama Manager initialized")
        else:
            logger.warning("Ollama not available")
    
    async def _check_ollama_status(self) -> bool:
        """Check if Ollama is running"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags", timeout=5) as response:
                    return response.status == 200
        except Exception:
            return False
    
    async def _refresh_model_list(self):
        """Refresh available models"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.base_url}/api/tags") as response:
                    if response.status == 200:
                        data = await response.json()
                        self.available_models = data.get("models", [])
        except Exception as e:
            logger.error(f"Failed to refresh models: {e}")
    
    async def generate_text(self, model_name: str, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate text using Ollama model"""
        try:
            request_data = {
                "model": model_name,
                "prompt": prompt,
                "stream": False,
                **kwargs
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.base_url}/api/generate",
                    json=request_data
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return {
                            "text": result.get("response", ""),
                            "model": model_name
                        }
                    else:
                        return {"error": f"Generation failed: {response.status}"}
        except Exception as e:
            return {"error": str(e)}
    
    def list_available_models(self) -> List[str]:
        """List available models"""
        return [model.get("name", "") for model in self.available_models]
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        return {
            "ollama_available": len(self.available_models) > 0,
            "total_models": len(self.available_models),
            "available_models": self.list_available_models()
        }

# Global instance
ollama_manager = OllamaManager()
