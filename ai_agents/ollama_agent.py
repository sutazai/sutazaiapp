#!/usr/bin/env python3
"""
Ollama Local LLM Agent for SutazAI
Provides local LLM capabilities without external API dependencies
"""

import asyncio
import json
import aiohttp
import logging
from typing import Dict, List, Any, Optional, AsyncGenerator, Union
from dataclasses import dataclass
from enum import Enum
from backend.config import Config
from .base_agent import BaseAgent
from loguru import logger


class OllamaModelSize(str, Enum):
    """Ollama model size categories"""
    SMALL = "7b"
    MEDIUM = "13b" 
    LARGE = "70b"
    EXTRA_LARGE = "405b"


@dataclass
class OllamaModel:
    """Ollama model configuration"""
    name: str
    tag: str = "latest"
    size: OllamaModelSize = OllamaModelSize.SMALL
    description: str = ""
    context_length: int = 4096
    parameters: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.parameters is None:
            self.parameters = {}
    
    @property
    def full_name(self) -> str:
        """Get full model name with tag"""
        return f"{self.name}:{self.tag}"


class OllamaAgent(BaseAgent):
    """Local LLM agent using Ollama"""
    
    # Recommended models for different use cases
    RECOMMENDED_MODELS = {
        "general": OllamaModel("llama3.1", "8b", OllamaModelSize.SMALL, "General purpose LLM"),
        "coding": OllamaModel("codellama", "7b", OllamaModelSize.SMALL, "Code generation and analysis"),
        "chat": OllamaModel("mistral", "7b", OllamaModelSize.SMALL, "Conversational AI"),
        "analysis": OllamaModel("llama3.1", "70b", OllamaModelSize.LARGE, "Complex analysis and reasoning"),
        "lightweight": OllamaModel("phi3", "mini", OllamaModelSize.SMALL, "Fast, lightweight model"),
        "creative": OllamaModel("llama3.1", "8b", OllamaModelSize.SMALL, "Creative writing and content"),
        "technical": OllamaModel("deepseek-coder", "6.7b", OllamaModelSize.SMALL, "Technical documentation"),
        "embedding": OllamaModel("nomic-embed-text", "latest", OllamaModelSize.SMALL, "Text embeddings")
    }
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.agent_type = "ollama"
        self.name = "Ollama Local LLM Agent"
        self.description = "Local Large Language Model agent using Ollama"
        
        # Ollama configuration
        self.ollama_host = config.get("host", "localhost")
        self.ollama_port = config.get("port", 11434)
        self.base_url = f"http://{self.ollama_host}:{self.ollama_port}"
        
        # Model configuration
        self.default_model = config.get("default_model", "llama3.1:8b")
        self.max_retries = config.get("max_retries", 3)
        self.timeout = config.get("timeout", 30)
        
        # Generation parameters
        self.default_params = {
            "temperature": config.get("temperature", 0.7),
            "top_p": config.get("top_p", 0.9),
            "top_k": config.get("top_k", 40),
            "repeat_penalty": config.get("repeat_penalty", 1.1),
            "num_ctx": config.get("context_length", 4096),
            "num_predict": config.get("max_tokens", 512)
        }
        
        # Session for HTTP requests
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self) -> bool:
        """Initialize the Ollama agent"""
        try:
            self.session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # Check Ollama server connectivity
            if not await self.check_ollama_health():
                logger.error("Ollama server is not accessible")
                return False
            
            # Ensure default model is available
            await self.ensure_model_available(self.default_model)
            
            logger.info(f"Ollama agent initialized with server at {self.base_url}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Ollama agent: {e}")
            return False
    
    async def cleanup(self) -> None:
        """Cleanup resources"""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def check_ollama_health(self) -> bool:
        """Check if Ollama server is running and accessible"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                return response.status == 200
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List available models on Ollama server"""
        try:
            async with self.session.get(f"{self.base_url}/api/tags") as response:
                if response.status == 200:
                    data = await response.json()
                    return data.get("models", [])
                else:
                    logger.error(f"Failed to list models: {response.status}")
                    return []
        except Exception as e:
            logger.error(f"Error listing models: {e}")
            return []
    
    async def ensure_model_available(self, model_name: str) -> bool:
        """Ensure a model is available, pull if necessary"""
        try:
            models = await self.list_models()
            available_models = [model["name"] for model in models]
            
            if model_name in available_models:
                logger.info(f"Model {model_name} is already available")
                return True
            
            logger.info(f"Pulling model {model_name}...")
            await self.pull_model(model_name)
            return True
            
        except Exception as e:
            logger.error(f"Failed to ensure model availability: {e}")
            return False
    
    async def pull_model(self, model_name: str) -> bool:
        """Pull a model to Ollama server"""
        try:
            data = {"name": model_name}
            
            async with self.session.post(
                f"{self.base_url}/api/pull",
                json=data
            ) as response:
                if response.status == 200:
                    # Stream the response to show progress
                    async for line in response.content:
                        if line:
                            try:
                                status = json.loads(line)
                                if "status" in status:
                                    logger.info(f"Pulling {model_name}: {status['status']}")
                            except json.JSONDecodeError:
                                continue
                    
                    logger.info(f"Successfully pulled model {model_name}")
                    return True
                else:
                    logger.error(f"Failed to pull model {model_name}: {response.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            return False
    
    async def generate_text(
        self,
        prompt: str,
        model: str = None,
        system_prompt: str = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Generate text using Ollama"""
        model = model or self.default_model
        
        # Merge parameters
        params = {**self.default_params, **kwargs}
        
        # Prepare request data
        data = {
            "model": model,
            "prompt": prompt,
            "stream": stream,
            "options": params
        }
        
        if system_prompt:
            data["system"] = system_prompt
        
        try:
            if stream:
                return self._generate_stream(data)
            else:
                return await self._generate_single(data)
                
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            raise
    
    async def _generate_single(self, data: Dict[str, Any]) -> str:
        """Generate single response"""
        async with self.session.post(
            f"{self.base_url}/api/generate",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("response", "")
            else:
                error_text = await response.text()
                raise Exception(f"Generation failed: {response.status} - {error_text}")
    
    async def _generate_stream(self, data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Generate streaming response"""
        async with self.session.post(
            f"{self.base_url}/api/generate",
            json=data
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line)
                            if "response" in chunk:
                                yield chunk["response"]
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                error_text = await response.text()
                raise Exception(f"Streaming failed: {response.status} - {error_text}")
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        model: str = None,
        stream: bool = False,
        **kwargs
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Chat interface with conversation history"""
        model = model or self.default_model
        params = {**self.default_params, **kwargs}
        
        data = {
            "model": model,
            "messages": messages,
            "stream": stream,
            "options": params
        }
        
        try:
            if stream:
                return self._chat_stream(data)
            else:
                return await self._chat_single(data)
        except Exception as e:
            logger.error(f"Error in chat: {e}")
            raise
    
    async def _chat_single(self, data: Dict[str, Any]) -> str:
        """Single chat response"""
        async with self.session.post(
            f"{self.base_url}/api/chat",
            json=data
        ) as response:
            if response.status == 200:
                result = await response.json()
                return result.get("message", {}).get("content", "")
            else:
                error_text = await response.text()
                raise Exception(f"Chat failed: {response.status} - {error_text}")
    
    async def _chat_stream(self, data: Dict[str, Any]) -> AsyncGenerator[str, None]:
        """Streaming chat response"""
        async with self.session.post(
            f"{self.base_url}/api/chat",
            json=data
        ) as response:
            if response.status == 200:
                async for line in response.content:
                    if line:
                        try:
                            chunk = json.loads(line)
                            message = chunk.get("message", {})
                            if "content" in message:
                                yield message["content"]
                            if chunk.get("done", False):
                                break
                        except json.JSONDecodeError:
                            continue
            else:
                error_text = await response.text()
                raise Exception(f"Chat streaming failed: {response.status} - {error_text}")
    
    async def generate_embeddings(self, text: str, model: str = "nomic-embed-text") -> List[float]:
        """Generate embeddings for text"""
        data = {
            "model": model,
            "prompt": text
        }
        
        try:
            async with self.session.post(
                f"{self.base_url}/api/embeddings",
                json=data
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get("embedding", [])
                else:
                    error_text = await response.text()
                    raise Exception(f"Embedding generation failed: {response.status} - {error_text}")
        except Exception as e:
            logger.error(f"Error generating embeddings: {e}")
            raise
    
    async def execute_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a task using the Ollama agent"""
        task_type = task.get("type", "generate")
        
        try:
            if task_type == "generate":
                response = await self.generate_text(
                    prompt=task["prompt"],
                    model=task.get("model"),
                    system_prompt=task.get("system_prompt"),
                    **task.get("parameters", {})
                )
                
                return {
                    "status": "completed",
                    "result": response,
                    "model_used": task.get("model", self.default_model)
                }
            
            elif task_type == "chat":
                response = await self.chat(
                    messages=task["messages"],
                    model=task.get("model"),
                    **task.get("parameters", {})
                )
                
                return {
                    "status": "completed",
                    "result": response,
                    "model_used": task.get("model", self.default_model)
                }
            
            elif task_type == "embed":
                embeddings = await self.generate_embeddings(
                    text=task["text"],
                    model=task.get("model", "nomic-embed-text")
                )
                
                return {
                    "status": "completed",
                    "result": embeddings,
                    "model_used": task.get("model", "nomic-embed-text")
                }
            
            else:
                return {
                    "status": "error",
                    "error": f"Unsupported task type: {task_type}"
                }
                
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            return {
                "status": "error",
                "error": str(e)
            }
    
    async def get_health_status(self) -> Dict[str, Any]:
        """Get agent health status"""
        is_healthy = await self.check_ollama_health()
        models = await self.list_models() if is_healthy else []
        
        return {
            "status": "healthy" if is_healthy else "unhealthy",
            "ollama_server": f"{self.base_url}",
            "available_models": len(models),
            "default_model": self.default_model,
            "models": [model["name"] for model in models[:10]]  # Limit to first 10
        }


# Factory function for agent registration
def create_ollama_agent(config: Dict[str, Any] = None) -> OllamaAgent:
    """Factory function to create Ollama agent"""
    return OllamaAgent(config or {})