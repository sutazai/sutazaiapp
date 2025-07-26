#!/usr/bin/env python3
"""
SutazAI LLM Service
Provides language model integration and management
"""

import asyncio
import requests
from typing import Dict, Any, Optional, List
from datetime import datetime


class LLMService:
    """Language model service for AI interactions"""
    
    def __init__(self, ollama_url: str = "http://localhost:11434"):
        self.ollama_url = ollama_url
        self.models = {}
        self.initialized = False
    
    async def initialize(self) -> None:
        """Initialize the LLM service"""
        try:
            await self._check_ollama_health()
            await self._load_available_models()
            self.initialized = True
        except Exception as e:
            print(f"LLM Service initialization warning: {e}")
            self.initialized = False
    
    async def _check_ollama_health(self) -> bool:
        """Check if Ollama is running and accessible"""
        try:
            response = requests.get(f"{self.ollama_url}/api/version", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    async def _load_available_models(self) -> None:
        """Load list of available models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models_data = response.json()
                for model in models_data.get("models", []):
                    self.models[model["name"]] = {
                        "name": model["name"],
                        "size": model.get("size", 0),
                        "modified": model.get("modified_at", ""),
                        "status": "available"
                    }
        except Exception as e:
            print(f"Could not load models: {e}")
    
    async def generate_response(self, prompt: str, model: str = "llama3.2:1b", **kwargs) -> Dict[str, Any]:
        """Generate response using specified model"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": kwargs.get("temperature", 0.7),
                    "top_p": kwargs.get("top_p", 0.9)
                }
            }
            
            response = requests.post(f"{self.ollama_url}/api/generate", json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                return {
                    "success": True,
                    "response": result.get("response", ""),
                    "model": model,
                    "timestamp": datetime.now().isoformat()
                }
            else:
                return {
                    "success": False,
                    "error": f"HTTP {response.status_code}",
                    "model": model
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "model": model
            }
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        return list(self.models.values())
    
    def is_model_available(self, model: str) -> bool:
        """Check if a specific model is available"""
        return model in self.models
    
    def cleanup(self) -> None:
        """Cleanup service on shutdown"""
        self.models.clear()
        self.initialized = False