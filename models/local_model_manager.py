"""
SutazAI Local Model Management System
Advanced local LLM management with Ollama integration

This module provides comprehensive local model management capabilities
including model deployment, optimization, and intelligent switching.
"""

import os
import json
import logging
import asyncio
import subprocess
import time
import requests
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, AsyncIterator
from dataclasses import dataclass, field
from enum import Enum
import threading
import queue
from datetime import datetime, timedelta
import hashlib
import aiohttp
import aiofiles

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(Enum):
    """Supported model types"""
    CHAT = "chat"
    CODE = "code"
    EMBEDDING = "embedding"
    VISION = "vision"
    SPECIALIZED = "specialized"

class ModelStatus(Enum):
    """Model status states"""
    AVAILABLE = "available"
    DOWNLOADING = "downloading"
    INSTALLING = "installing"
    LOADED = "loaded"
    UNLOADED = "unloaded"
    ERROR = "error"
    UPDATING = "updating"

@dataclass
class ModelConfig:
    """Configuration for a local model"""
    name: str
    model_type: ModelType
    ollama_name: str
    description: str
    size_gb: float
    parameters: str
    capabilities: List[str] = field(default_factory=list)
    context_length: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 40
    repeat_penalty: float = 1.1
    system_prompt: str = ""
    performance_score: float = 0.0
    last_used: Optional[datetime] = None
    usage_count: int = 0
    status: ModelStatus = ModelStatus.AVAILABLE

@dataclass
class ModelResponse:
    """Response from a local model"""
    model_name: str
    response: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    model_info: Dict[str, Any]

class LocalModelManager:
    """Manages local LLM models with Ollama integration"""
    
    def __init__(self, models_dir: str = "/opt/sutazaiapp/models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        
        # Ollama configuration
        self.ollama_host = "http://localhost:11434"
        self.ollama_api = f"{self.ollama_host}/api"
        
        # Model registry
        self.model_registry: Dict[str, ModelConfig] = {}
        self.loaded_models: Dict[str, Dict[str, Any]] = {}
        
        # Performance tracking
        self.model_performance: Dict[str, Dict[str, Any]] = {}
        self.request_queue = queue.Queue()
        
        # Initialize system
        self._initialize_system()
        
        logger.info("Local Model Manager initialized")
    
    def _initialize_system(self):
        """Initialize the local model management system"""
        try:
            # Load model registry
            self._load_model_registry()
            
            # Initialize default models
            self._initialize_default_models()
            
            # Check Ollama availability
            self._check_ollama_availability()
            
            # Start background tasks
            self._start_background_tasks()
            
        except Exception as e:
            logger.error(f"Failed to initialize model system: {e}")
            raise
    
    def _load_model_registry(self):
        """Load model registry from disk"""
        registry_path = self.models_dir / "model_registry.json"
        
        if registry_path.exists():
            try:
                with open(registry_path, 'r') as f:
                    registry_data = json.load(f)
                
                # Convert to ModelConfig objects
                for model_name, model_data in registry_data.get("models", {}).items():
                    self.model_registry[model_name] = ModelConfig(
                        name=model_data["name"],
                        model_type=ModelType(model_data["type"]),
                        ollama_name=model_data.get("ollama_name", model_name),
                        description=model_data.get("description", ""),
                        size_gb=model_data.get("size_gb", 1.0),
                        parameters=model_data.get("parameters", "7B"),
                        capabilities=model_data.get("capabilities", []),
                        context_length=model_data.get("context_length", 4096),
                        temperature=model_data.get("temperature", 0.7),
                        status=ModelStatus(model_data.get("status", "available"))
                    )
                
                logger.info(f"Loaded {len(self.model_registry)} models from registry")
                
            except Exception as e:
                logger.error(f"Failed to load model registry: {e}")
        else:
            logger.info("No existing model registry found, creating new one")
    
    def _initialize_default_models(self):
        """Initialize default model configurations"""
        default_models = [
            ModelConfig(
                name="llama3.1",
                model_type=ModelType.CHAT,
                ollama_name="llama3.1:latest",
                description="Meta's Llama 3.1 model for general conversation",
                size_gb=4.7,
                parameters="8B",
                capabilities=["text_generation", "conversation", "reasoning"],
                context_length=8192,
                system_prompt="You are a helpful AI assistant."
            ),
            ModelConfig(
                name="codellama",
                model_type=ModelType.CODE,
                ollama_name="codellama:latest",
                description="Code Llama model for code generation and analysis",
                size_gb=3.8,
                parameters="7B",
                capabilities=["code_generation", "code_review", "debugging"],
                context_length=4096,
                system_prompt="You are an expert programming assistant."
            ),
            ModelConfig(
                name="mistral",
                model_type=ModelType.CHAT,
                ollama_name="mistral:latest",
                description="Mistral 7B model for efficient conversation",
                size_gb=4.1,
                parameters="7B",
                capabilities=["text_generation", "conversation", "analysis"],
                context_length=8192,
                system_prompt="You are a knowledgeable assistant."
            ),
            ModelConfig(
                name="neural-chat",
                model_type=ModelType.SPECIALIZED,
                ollama_name="neural-chat:latest",
                description="Specialized model for neural network discussions",
                size_gb=7.3,
                parameters="7B",
                capabilities=["neural_networks", "ai_research", "technical_discussion"],
                context_length=4096,
                system_prompt="You are an expert in neural networks and AI systems."
            )
        ]
        
        # Add default models to registry if they don't exist
        for model in default_models:
            if model.name not in self.model_registry:
                self.model_registry[model.name] = model
        
        # Save updated registry
        self._save_model_registry()
    
    def _save_model_registry(self):
        """Save model registry to disk"""
        registry_path = self.models_dir / "model_registry.json"
        
        try:
            registry_data = {
                "models": {},
                "initialized_at": datetime.now().isoformat()
            }
            
            for model_name, model in self.model_registry.items():
                registry_data["models"][model_name] = {
                    "name": model.name,
                    "type": model.model_type.value,
                    "ollama_name": model.ollama_name,
                    "description": model.description,
                    "size_gb": model.size_gb,
                    "parameters": model.parameters,
                    "capabilities": model.capabilities,
                    "context_length": model.context_length,
                    "temperature": model.temperature,
                    "status": model.status.value
                }
            
            with open(registry_path, 'w') as f:
                json.dump(registry_data, f, indent=2)
            
            logger.info("Model registry saved")
            
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def _check_ollama_availability(self):
        """Check if Ollama is available and install if needed"""
        try:
            # Check if Ollama is running
            response = requests.get(f"{self.ollama_host}/api/version", timeout=5)
            if response.status_code == 200:
                logger.info("Ollama is running")
                return True
        except:
            pass
        
        logger.info("Ollama not running, attempting to install and start")
        
        # Install Ollama if not available
        try:
            install_script = """
            curl -fsSL https://ollama.com/install.sh | sh
            """
            
            result = subprocess.run(
                install_script,
                shell=True,
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                logger.error(f"Failed to install Ollama: {result.stderr}")
                return False
            
            # Start Ollama service
            start_result = subprocess.run(
                ["systemctl", "start", "ollama"],
                capture_output=True,
                text=True
            )
            
            if start_result.returncode != 0:
                # Try starting directly
                subprocess.Popen(["ollama", "serve"], 
                               stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
            
            # Wait for service to be ready
            time.sleep(5)
            
            # Verify Ollama is running
            response = requests.get(f"{self.ollama_host}/api/version", timeout=10)
            if response.status_code == 200:
                logger.info("Ollama installed and started successfully")
                return True
            else:
                logger.error("Ollama installation failed")
                return False
                
        except Exception as e:
            logger.error(f"Failed to install Ollama: {e}")
            return False
    
    def _start_background_tasks(self):
        """Start background tasks for model management"""
        # Start model monitoring thread
        monitoring_thread = threading.Thread(
            target=self._model_monitoring_loop,
            daemon=True
        )
        monitoring_thread.start()
        
        # Start performance tracking thread
        performance_thread = threading.Thread(
            target=self._performance_tracking_loop,
            daemon=True
        )
        performance_thread.start()
    
    def _model_monitoring_loop(self):
        """Monitor model health and performance"""
        while True:
            try:
                # Check model health
                self._check_model_health()
                
                # Update model performance metrics
                self._update_performance_metrics()
                
                # Auto-unload unused models
                self._auto_unload_unused_models()
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Model monitoring error: {e}")
                time.sleep(60)  # Wait longer on error
    
    def _performance_tracking_loop(self):
        """Track model performance and usage"""
        while True:
            try:
                # Update usage statistics
                self._update_usage_statistics()
                
                # Optimize model selection
                self._optimize_model_selection()
                
                time.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Performance tracking error: {e}")
                time.sleep(600)  # Wait longer on error
    
    async def install_model(self, model_name: str) -> Dict[str, Any]:
        """Install a model via Ollama"""
        try:
            if model_name not in self.model_registry:
                raise ValueError(f"Model {model_name} not found in registry")
            
            model = self.model_registry[model_name]
            model.status = ModelStatus.DOWNLOADING
            
            logger.info(f"Installing model: {model_name}")
            
            # Pull model via Ollama API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_api}/pull",
                    json={"name": model.ollama_name}
                ) as response:
                    if response.status == 200:
                        # Stream the download progress
                        async for line in response.content:
                            if line:
                                progress_data = json.loads(line)
                                if "status" in progress_data:
                                    logger.info(f"Install progress: {progress_data['status']}")
                    else:
                        raise Exception(f"Failed to install model: {response.status}")
            
            model.status = ModelStatus.AVAILABLE
            self._save_model_registry()
            
            logger.info(f"Model {model_name} installed successfully")
            
            return {
                "status": "success",
                "model_name": model_name,
                "ollama_name": model.ollama_name,
                "message": "Model installed successfully"
            }
            
        except Exception as e:
            if model_name in self.model_registry:
                self.model_registry[model_name].status = ModelStatus.ERROR
            logger.error(f"Failed to install model {model_name}: {e}")
            raise
    
    async def load_model(self, model_name: str) -> Dict[str, Any]:
        """Load a model into memory"""
        try:
            if model_name not in self.model_registry:
                raise ValueError(f"Model {model_name} not found in registry")
            
            model = self.model_registry[model_name]
            
            # Check if model is already loaded
            if model_name in self.loaded_models:
                logger.info(f"Model {model_name} already loaded")
                return {"status": "already_loaded", "model_name": model_name}
            
            logger.info(f"Loading model: {model_name}")
            
            # Load model via Ollama API
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_api}/generate",
                    json={
                        "model": model.ollama_name,
                        "prompt": "Hello",
                        "stream": False
                    }
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        
                        # Store loaded model info
                        self.loaded_models[model_name] = {
                            "loaded_at": datetime.now(),
                            "model_info": result.get("model", {}),
                            "status": "loaded"
                        }
                        
                        model.status = ModelStatus.LOADED
                        self._save_model_registry()
                        
                        logger.info(f"Model {model_name} loaded successfully")
                        
                        return {
                            "status": "success",
                            "model_name": model_name,
                            "loaded_at": datetime.now().isoformat(),
                            "model_info": result.get("model", {})
                        }
                    else:
                        raise Exception(f"Failed to load model: {response.status}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a model from memory"""
        try:
            if model_name not in self.loaded_models:
                return {"status": "not_loaded", "model_name": model_name}
            
            logger.info(f"Unloading model: {model_name}")
            
            # Remove from loaded models
            del self.loaded_models[model_name]
            
            # Update model status
            if model_name in self.model_registry:
                self.model_registry[model_name].status = ModelStatus.AVAILABLE
                self._save_model_registry()
            
            logger.info(f"Model {model_name} unloaded successfully")
            
            return {
                "status": "success",
                "model_name": model_name,
                "message": "Model unloaded successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to unload model {model_name}: {e}")
            raise
    
    async def generate_response(self, 
                               model_name: str, 
                               prompt: str,
                               system_prompt: str = None,
                               temperature: float = None,
                               max_tokens: int = None,
                               stream: bool = False) -> Union[ModelResponse, AsyncIterator[str]]:
        """Generate response from a local model"""
        try:
            if model_name not in self.model_registry:
                raise ValueError(f"Model {model_name} not found in registry")
            
            model = self.model_registry[model_name]
            
            # Ensure model is loaded
            if model_name not in self.loaded_models:
                await self.load_model(model_name)
            
            # Prepare request parameters
            request_params = {
                "model": model.ollama_name,
                "prompt": prompt,
                "stream": stream,
                "options": {
                    "temperature": temperature or model.temperature,
                    "top_p": model.top_p,
                    "top_k": model.top_k,
                    "repeat_penalty": model.repeat_penalty
                }
            }
            
            # Add system prompt if provided
            if system_prompt or model.system_prompt:
                request_params["system"] = system_prompt or model.system_prompt
            
            # Add max tokens if provided
            if max_tokens:
                request_params["options"]["num_predict"] = max_tokens
            
            start_time = time.time()
            
            # Make request to Ollama
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.ollama_api}/generate",
                    json=request_params
                ) as response:
                    if response.status == 200:
                        if stream:
                            return self._stream_response(response)
                        else:
                            result = await response.json()
                            response_time = time.time() - start_time
                            
                            # Update model usage
                            self._update_model_usage(model_name, response_time)
                            
                            return ModelResponse(
                                model_name=model_name,
                                response=result.get("response", ""),
                                prompt_tokens=result.get("prompt_eval_count", 0),
                                completion_tokens=result.get("eval_count", 0),
                                total_tokens=result.get("prompt_eval_count", 0) + result.get("eval_count", 0),
                                response_time=response_time,
                                model_info=result.get("model", {})
                            )
                    else:
                        raise Exception(f"Model request failed: {response.status}")
            
        except Exception as e:
            logger.error(f"Failed to generate response from {model_name}: {e}")
            raise
    
    async def _stream_response(self, response) -> AsyncIterator[str]:
        """Stream response from model"""
        async for line in response.content:
            if line:
                try:
                    data = json.loads(line)
                    if "response" in data:
                        yield data["response"]
                except json.JSONDecodeError:
                    continue
    
    def _update_model_usage(self, model_name: str, response_time: float):
        """Update model usage statistics"""
        if model_name in self.model_registry:
            model = self.model_registry[model_name]
            model.usage_count += 1
            model.last_used = datetime.now()
            
            # Update performance score
            if model.performance_score == 0.0:
                model.performance_score = 1.0 / response_time
            else:
                # Weighted average of performance
                model.performance_score = (model.performance_score * 0.9) + (1.0 / response_time * 0.1)
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        models = []
        
        for model_name, model in self.model_registry.items():
            models.append({
                "name": model.name,
                "type": model.model_type.value,
                "description": model.description,
                "parameters": model.parameters,
                "size_gb": model.size_gb,
                "capabilities": model.capabilities,
                "context_length": model.context_length,
                "status": model.status.value,
                "loaded": model_name in self.loaded_models,
                "usage_count": model.usage_count,
                "last_used": model.last_used.isoformat() if model.last_used else None,
                "performance_score": model.performance_score
            })
        
        return models
    
    def get_model_info(self, model_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific model"""
        if model_name not in self.model_registry:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.model_registry[model_name]
        
        info = {
            "name": model.name,
            "type": model.model_type.value,
            "ollama_name": model.ollama_name,
            "description": model.description,
            "parameters": model.parameters,
            "size_gb": model.size_gb,
            "capabilities": model.capabilities,
            "context_length": model.context_length,
            "temperature": model.temperature,
            "status": model.status.value,
            "loaded": model_name in self.loaded_models,
            "usage_count": model.usage_count,
            "last_used": model.last_used.isoformat() if model.last_used else None,
            "performance_score": model.performance_score
        }
        
        # Add loaded model info if available
        if model_name in self.loaded_models:
            info["loaded_info"] = self.loaded_models[model_name]
        
        return info
    
    def recommend_model(self, task_type: str, requirements: Dict[str, Any] = None) -> str:
        """Recommend the best model for a given task"""
        requirements = requirements or {}
        
        # Filter models by type
        suitable_models = []
        
        for model_name, model in self.model_registry.items():
            if model.status in [ModelStatus.AVAILABLE, ModelStatus.LOADED]:
                # Check if model supports the task type
                if task_type in model.capabilities:
                    suitable_models.append((model_name, model))
        
        if not suitable_models:
            # Fallback to any available model
            suitable_models = [(name, model) for name, model in self.model_registry.items()
                             if model.status in [ModelStatus.AVAILABLE, ModelStatus.LOADED]]
        
        if not suitable_models:
            raise ValueError("No suitable models available")
        
        # Score models based on performance and requirements
        scored_models = []
        
        for model_name, model in suitable_models:
            score = model.performance_score
            
            # Boost score for loaded models
            if model_name in self.loaded_models:
                score *= 1.5
            
            # Boost score for recently used models
            if model.last_used:
                hours_since_use = (datetime.now() - model.last_used).total_seconds() / 3600
                if hours_since_use < 24:
                    score *= 1.2
            
            # Consider context length requirement
            if requirements.get("context_length"):
                if model.context_length >= requirements["context_length"]:
                    score *= 1.1
                else:
                    score *= 0.5
            
            scored_models.append((model_name, score))
        
        # Sort by score and return best model
        scored_models.sort(key=lambda x: x[1], reverse=True)
        
        return scored_models[0][0]
    
    def _check_model_health(self):
        """Check health of loaded models"""
        for model_name in list(self.loaded_models.keys()):
            try:
                # Simple health check - try to generate a short response
                asyncio.run(self.generate_response(model_name, "Hi", max_tokens=1))
            except Exception as e:
                logger.warning(f"Model {model_name} health check failed: {e}")
                # Remove unhealthy model
                if model_name in self.loaded_models:
                    del self.loaded_models[model_name]
    
    def _update_performance_metrics(self):
        """Update performance metrics for all models"""
        for model_name, model in self.model_registry.items():
            if model_name not in self.model_performance:
                self.model_performance[model_name] = {
                    "response_times": [],
                    "success_rate": 1.0,
                    "error_count": 0,
                    "total_requests": 0
                }
    
    def _auto_unload_unused_models(self):
        """Automatically unload models that haven't been used recently"""
        current_time = datetime.now()
        
        for model_name in list(self.loaded_models.keys()):
            if model_name in self.model_registry:
                model = self.model_registry[model_name]
                
                # Unload if not used in the last 2 hours
                if model.last_used and (current_time - model.last_used) > timedelta(hours=2):
                    logger.info(f"Auto-unloading unused model: {model_name}")
                    asyncio.run(self.unload_model(model_name))
    
    def _update_usage_statistics(self):
        """Update usage statistics for all models"""
        # This would typically involve more complex analytics
        pass
    
    def _optimize_model_selection(self):
        """Optimize model selection based on usage patterns"""
        # This would implement machine learning-based model selection
        pass
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        return {
            "total_models": len(self.model_registry),
            "loaded_models": len(self.loaded_models),
            "available_models": len([m for m in self.model_registry.values() 
                                   if m.status == ModelStatus.AVAILABLE]),
            "model_types": {model_type.value: len([m for m in self.model_registry.values() 
                                                 if m.model_type == model_type])
                           for model_type in ModelType},
            "ollama_status": self._get_ollama_status(),
            "performance_metrics": self.model_performance,
            "loaded_model_info": {name: info for name, info in self.loaded_models.items()}
        }
    
    def _get_ollama_status(self) -> Dict[str, Any]:
        """Get Ollama service status"""
        try:
            response = requests.get(f"{self.ollama_host}/api/version", timeout=5)
            if response.status_code == 200:
                return {
                    "status": "running",
                    "version": response.json().get("version", "unknown"),
                    "host": self.ollama_host
                }
            else:
                return {"status": "error", "message": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"status": "unavailable", "error": str(e)}

# Global model manager instance
_model_manager_instance = None

def get_model_manager() -> LocalModelManager:
    """Get the global model manager instance"""
    global _model_manager_instance
    if _model_manager_instance is None:
        _model_manager_instance = LocalModelManager()
    return _model_manager_instance

async def main():
    """Example usage of the Local Model Manager"""
    model_manager = get_model_manager()
    
    # List available models
    models = model_manager.get_available_models()
    print(f"Available models: {json.dumps(models, indent=2)}")
    
    # Install a model
    await model_manager.install_model("llama3.1")
    
    # Load a model
    await model_manager.load_model("llama3.1")
    
    # Generate response
    response = await model_manager.generate_response(
        "llama3.1",
        "Explain neural networks in simple terms"
    )
    print(f"Model response: {response.response}")
    
    # Get system status
    status = model_manager.get_system_status()
    print(f"System status: {json.dumps(status, indent=2)}")

if __name__ == "__main__":
    asyncio.run(main())