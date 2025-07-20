#!/usr/bin/env python3
"""
SutazAI Model Manager
Advanced AI model orchestration and optimization system
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
import httpx
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import psutil

from backend.core.config import get_settings
from backend.utils.logging_setup import get_api_logger

logger = get_api_logger()
settings = get_settings()

@dataclass
class ModelConfig:
    """Model configuration data structure"""
    name: str
    type: str  # "llm", "embedding", "multimodal", etc.
    endpoint: str
    capabilities: List[str] = field(default_factory=list)
    context_length: int = 4096
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    stop_sequences: List[str] = field(default_factory=list)
    system_prompt: str = ""
    warm_up_prompt: str = "Hello, world!"
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    is_loaded: bool = False
    load_time: Optional[datetime] = None
    last_used: Optional[datetime] = None

@dataclass
class ModelRequest:
    """Model inference request"""
    model_name: str
    prompt: Union[str, List[Dict[str, str]]]
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop_sequences: Optional[List[str]] = None
    stream: bool = False
    system_prompt: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ModelResponse:
    """Model inference response"""
    model_name: str
    content: str
    usage: Dict[str, int] = field(default_factory=dict)
    response_time: float = 0.0
    finish_reason: str = "stop"
    metadata: Dict[str, Any] = field(default_factory=dict)

class ModelPerformanceOptimizer:
    """Optimize model performance and resource usage"""
    
    def __init__(self):
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        self.optimization_rules: Dict[str, Dict[str, Any]] = {}
        
    async def optimize_model_config(self, model_name: str, config: ModelConfig) -> ModelConfig:
        """Optimize model configuration based on performance history"""
        history = self.performance_history.get(model_name, [])
        
        if len(history) < 10:  # Need sufficient data
            return config
        
        # Analyze response times for different parameters
        avg_response_times = {}
        
        for record in history[-50:]:  # Last 50 requests
            params_key = f"{record.get('temperature', 0.7)}-{record.get('max_tokens', 2048)}"
            if params_key not in avg_response_times:
                avg_response_times[params_key] = []
            avg_response_times[params_key].append(record.get('response_time', 0))
        
        # Find optimal parameters
        best_params = None
        best_avg_time = float('inf')
        
        for params_key, times in avg_response_times.items():
            avg_time = sum(times) / len(times)
            if avg_time < best_avg_time and len(times) >= 5:  # Minimum sample size
                best_avg_time = avg_time
                best_params = params_key
        
        if best_params:
            temp, max_tokens = best_params.split('-')
            config.temperature = float(temp)
            config.max_tokens = int(max_tokens)
            logger.info(f"Optimized {model_name}: temp={config.temperature}, max_tokens={config.max_tokens}")
        
        return config
    
    def record_performance(self, model_name: str, request: ModelRequest, response: ModelResponse):
        """Record performance metrics for analysis"""
        if model_name not in self.performance_history:
            self.performance_history[model_name] = []
        
        record = {
            'timestamp': datetime.now().isoformat(),
            'temperature': request.temperature,
            'max_tokens': request.max_tokens,
            'prompt_length': len(str(request.prompt)),
            'response_length': len(response.content),
            'response_time': response.response_time,
            'tokens_per_second': response.usage.get('completion_tokens', 0) / max(response.response_time, 0.01)
        }
        
        self.performance_history[model_name].append(record)
        
        # Keep only last 1000 records per model
        if len(self.performance_history[model_name]) > 1000:
            self.performance_history[model_name] = self.performance_history[model_name][-1000:]

class ModelLoadBalancer:
    """Intelligent load balancing across model instances"""
    
    def __init__(self):
        self.model_instances: Dict[str, List[Dict[str, Any]]] = {}
        self.load_metrics: Dict[str, Dict[str, float]] = {}
        
    async def get_best_instance(self, model_name: str) -> Optional[str]:
        """Get the best available instance for a model"""
        instances = self.model_instances.get(model_name, [])
        
        if not instances:
            return None
        
        # Simple round-robin with health checking
        best_instance = None
        lowest_load = float('inf')
        
        for instance in instances:
            if not instance.get('healthy', True):
                continue
                
            load = self.load_metrics.get(instance['endpoint'], {}).get('current_load', 0)
            if load < lowest_load:
                lowest_load = load
                best_instance = instance['endpoint']
        
        return best_instance
    
    async def update_load_metrics(self):
        """Update load metrics for all instances"""
        for model_name, instances in self.model_instances.items():
            for instance in instances:
                try:
                    async with httpx.AsyncClient(timeout=5.0) as client:
                        response = await client.get(f"{instance['endpoint']}/metrics")
                        if response.status_code == 200:
                            metrics = response.json()
                            self.load_metrics[instance['endpoint']] = metrics
                        else:
                            instance['healthy'] = False
                except Exception as e:
                    logger.warning(f"Failed to get metrics from {instance['endpoint']}: {e}")
                    instance['healthy'] = False

class EnhancedModelManager:
    """
    Advanced model management system for SutazAI
    Handles multiple model types, optimization, and intelligent routing
    """
    
    def __init__(self):
        self.models: Dict[str, ModelConfig] = {}
        self.model_cache: Dict[str, Any] = {}
        self.performance_optimizer = ModelPerformanceOptimizer()
        self.load_balancer = ModelLoadBalancer()
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        # Model endpoints and configurations
        self.default_models = {
            "deepseek-r1:8b": ModelConfig(
                name="deepseek-r1:8b",
                type="llm",
                endpoint=f"{settings.OLLAMA_URL}/api/generate",
                capabilities=["chat", "completion", "reasoning", "code"],
                context_length=8192,
                max_tokens=4096,
                system_prompt="You are DeepSeek R1, an advanced AI assistant capable of complex reasoning and problem-solving."
            ),
            "qwen3:8b": ModelConfig(
                name="qwen3:8b", 
                type="llm",
                endpoint=f"{settings.OLLAMA_URL}/api/generate",
                capabilities=["chat", "completion", "multilingual", "code"],
                context_length=8192,
                max_tokens=4096,
                system_prompt="You are Qwen3, a multilingual AI assistant with strong capabilities in various languages and domains."
            ),
            "codellama:7b": ModelConfig(
                name="codellama:7b",
                type="llm",
                endpoint=f"{settings.OLLAMA_URL}/api/generate", 
                capabilities=["code", "completion", "debugging"],
                context_length=4096,
                max_tokens=2048,
                system_prompt="You are CodeLlama, an AI assistant specialized in programming and code generation."
            ),
            "codellama:33b": ModelConfig(
                name="codellama:33b",
                type="llm", 
                endpoint=f"{settings.OLLAMA_URL}/api/generate",
                capabilities=["code", "completion", "debugging", "architecture"],
                context_length=8192,
                max_tokens=4096,
                system_prompt="You are CodeLlama 33B, an advanced AI assistant for complex programming tasks and software architecture."
            ),
            "llama2": ModelConfig(
                name="llama2",
                type="llm",
                endpoint=f"{settings.OLLAMA_URL}/api/generate",
                capabilities=["chat", "completion", "general"],
                context_length=4096,
                max_tokens=2048,
                system_prompt="You are Llama2, a helpful AI assistant."
            )
        }
        
        # Performance monitoring
        self.request_count = 0
        self.total_response_time = 0.0
        self.error_count = 0
        
    async def initialize(self):
        """Initialize the model manager"""
        logger.info("Initializing Enhanced Model Manager...")
        
        try:
            # Load default models
            for model_name, config in self.default_models.items():
                self.models[model_name] = config
            
            # Check Ollama availability
            await self.check_ollama_health()
            
            # Load available models from Ollama
            await self.discover_ollama_models()
            
            # Start background tasks
            asyncio.create_task(self.model_health_monitor())
            asyncio.create_task(self.performance_optimization_loop())
            asyncio.create_task(self.model_warm_up_loop())
            
            logger.info(f"Model Manager initialized with {len(self.models)} models")
            
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
            raise
    
    async def check_ollama_health(self):
        """Check if Ollama is healthy and responsive"""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{settings.OLLAMA_URL}/api/tags")
                if response.status_code == 200:
                    logger.info("Ollama is healthy and responsive")
                    return True
                else:
                    logger.error(f"Ollama health check failed: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Ollama health check failed: {e}")
            return False
    
    async def discover_ollama_models(self):
        """Discover available models in Ollama"""
        try:
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{settings.OLLAMA_URL}/api/tags")
                if response.status_code == 200:
                    data = response.json()
                    available_models = [model['name'] for model in data.get('models', [])]
                    
                    logger.info(f"Available Ollama models: {available_models}")
                    
                    # Update model availability
                    for model_name in available_models:
                        if model_name in self.models:
                            self.models[model_name].is_loaded = True
                            self.models[model_name].load_time = datetime.now()
                    
                    return available_models
                else:
                    logger.error("Failed to discover Ollama models")
                    return []
        except Exception as e:
            logger.error(f"Model discovery failed: {e}")
            return []
    
    async def ensure_model_loaded(self, model_name: str) -> bool:
        """Ensure a specific model is loaded and ready"""
        if model_name not in self.models:
            logger.error(f"Unknown model: {model_name}")
            return False
        
        config = self.models[model_name]
        
        if config.is_loaded:
            config.last_used = datetime.now()
            return True
        
        # Attempt to pull/load the model
        logger.info(f"Loading model: {model_name}")
        
        try:
            async with httpx.AsyncClient(timeout=300.0) as client:
                # Pull model if not available
                pull_data = {"name": model_name}
                response = await client.post(f"{settings.OLLAMA_URL}/api/pull", json=pull_data)
                
                if response.status_code == 200:
                    config.is_loaded = True
                    config.load_time = datetime.now()
                    logger.info(f"Model {model_name} loaded successfully")
                    
                    # Warm up the model
                    await self.warm_up_model(model_name)
                    return True
                else:
                    logger.error(f"Failed to load model {model_name}: {response.status_code}")
                    return False
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return False
    
    async def warm_up_model(self, model_name: str):
        """Warm up a model with a simple request"""
        try:
            config = self.models[model_name]
            warm_up_request = ModelRequest(
                model_name=model_name,
                prompt=config.warm_up_prompt,
                max_tokens=50,
                temperature=0.1
            )
            
            await self.generate_response(warm_up_request)
            logger.info(f"Model {model_name} warmed up successfully")
            
        except Exception as e:
            logger.warning(f"Failed to warm up model {model_name}: {e}")
    
    async def generate_response(self, request: ModelRequest) -> ModelResponse:
        """Generate response using the specified model"""
        start_time = time.time()
        
        try:
            # Ensure model is loaded
            if not await self.ensure_model_loaded(request.model_name):
                raise Exception(f"Failed to load model {request.model_name}")
            
            config = self.models[request.model_name]
            
            # Prepare request data
            request_data = {
                "model": request.model_name,
                "prompt": self._format_prompt(request.prompt, request.system_prompt or config.system_prompt),
                "stream": request.stream,
                "options": {
                    "temperature": request.temperature or config.temperature,
                    "top_p": request.top_p or config.top_p,
                    "top_k": request.top_k or config.top_k,
                    "num_predict": request.max_tokens or config.max_tokens,
                    "stop": request.stop_sequences or config.stop_sequences
                }
            }
            
            # Make request to Ollama
            async with httpx.AsyncClient(timeout=300.0) as client:
                response = await client.post(config.endpoint, json=request_data)
                
                if response.status_code == 200:
                    response_data = response.json()
                    response_time = time.time() - start_time
                    
                    # Parse response
                    content = response_data.get("response", "")
                    
                    # Calculate usage statistics
                    usage = {
                        "prompt_tokens": len(str(request.prompt).split()),
                        "completion_tokens": len(content.split()),
                        "total_tokens": len(str(request.prompt).split()) + len(content.split())
                    }
                    
                    model_response = ModelResponse(
                        model_name=request.model_name,
                        content=content,
                        usage=usage,
                        response_time=response_time,
                        finish_reason=response_data.get("done_reason", "stop"),
                        metadata={
                            "model_config": config.__dict__,
                            "request_params": request_data["options"]
                        }
                    )
                    
                    # Record performance metrics
                    self.performance_optimizer.record_performance(request.model_name, request, model_response)
                    
                    # Update usage statistics
                    config.last_used = datetime.now()
                    self.request_count += 1
                    self.total_response_time += response_time
                    
                    return model_response
                
                else:
                    self.error_count += 1
                    error_msg = f"Model request failed: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    raise Exception(error_msg)
        
        except Exception as e:
            self.error_count += 1
            logger.error(f"Error generating response: {e}")
            raise
    
    def _format_prompt(self, prompt: Union[str, List[Dict[str, str]]], system_prompt: str = "") -> str:
        """Format prompt for the model"""
        if isinstance(prompt, str):
            if system_prompt:
                return f"System: {system_prompt}\n\nUser: {prompt}\n\nAssistant:"
            return prompt
        
        # Handle chat format
        formatted_parts = []
        if system_prompt:
            formatted_parts.append(f"System: {system_prompt}")
        
        for message in prompt:
            role = message.get("role", "user")
            content = message.get("content", "")
            formatted_parts.append(f"{role.title()}: {content}")
        
        formatted_parts.append("Assistant:")
        return "\n\n".join(formatted_parts)
    
    async def get_model_recommendations(self, task_type: str, context: Dict[str, Any] = None) -> List[str]:
        """Get recommended models for a specific task type"""
        recommendations = []
        
        # Task-specific model recommendations
        task_model_mapping = {
            "code_generation": ["codellama:33b", "codellama:7b", "deepseek-r1:8b"],
            "chat": ["deepseek-r1:8b", "qwen3:8b", "llama2"],
            "reasoning": ["deepseek-r1:8b", "qwen3:8b"],
            "multilingual": ["qwen3:8b", "deepseek-r1:8b"],
            "general": ["deepseek-r1:8b", "llama2", "qwen3:8b"]
        }
        
        recommended_models = task_model_mapping.get(task_type, ["deepseek-r1:8b"])
        
        # Filter by availability and performance
        for model_name in recommended_models:
            if model_name in self.models and self.models[model_name].is_loaded:
                recommendations.append(model_name)
        
        # If no loaded models, return all recommendations
        if not recommendations:
            recommendations = recommended_models
        
        return recommendations
    
    async def get_model_info(self, model_name: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a model"""
        if model_name not in self.models:
            return None
        
        config = self.models[model_name]
        
        # Get performance metrics
        history = self.performance_optimizer.performance_history.get(model_name, [])
        avg_response_time = 0.0
        if history:
            avg_response_time = sum(r.get('response_time', 0) for r in history[-10:]) / min(len(history), 10)
        
        return {
            "name": config.name,
            "type": config.type,
            "capabilities": config.capabilities,
            "context_length": config.context_length,
            "max_tokens": config.max_tokens,
            "is_loaded": config.is_loaded,
            "load_time": config.load_time.isoformat() if config.load_time else None,
            "last_used": config.last_used.isoformat() if config.last_used else None,
            "performance": {
                "average_response_time": avg_response_time,
                "total_requests": len(history),
                "success_rate": 1.0 - (self.error_count / max(self.request_count, 1))
            },
            "resource_requirements": config.resource_requirements
        }
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """List all available models with their status"""
        models_info = []
        
        for model_name in self.models:
            info = await self.get_model_info(model_name)
            if info:
                models_info.append(info)
        
        return models_info
    
    async def get_system_stats(self) -> Dict[str, Any]:
        """Get model manager system statistics"""
        total_models = len(self.models)
        loaded_models = len([m for m in self.models.values() if m.is_loaded])
        
        avg_response_time = 0.0
        if self.request_count > 0:
            avg_response_time = self.total_response_time / self.request_count
        
        success_rate = 1.0 - (self.error_count / max(self.request_count, 1))
        
        # System resource usage
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            "total_models": total_models,
            "loaded_models": loaded_models,
            "total_requests": self.request_count,
            "total_errors": self.error_count,
            "average_response_time": avg_response_time,
            "success_rate": success_rate,
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_free_gb": disk.free / (1024**3)
            },
            "ollama_health": await self.check_ollama_health()
        }
    
    # Background tasks
    
    async def model_health_monitor(self):
        """Monitor model health and availability"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check Ollama health
                if not await self.check_ollama_health():
                    logger.warning("Ollama is not responding - attempting to reconnect")
                    await asyncio.sleep(10)
                    continue
                
                # Update model statuses
                await self.discover_ollama_models()
                
                # Check for models that haven't been used recently
                current_time = datetime.now()
                for model_name, config in self.models.items():
                    if config.last_used and (current_time - config.last_used) > timedelta(hours=1):
                        logger.info(f"Model {model_name} hasn't been used for over an hour")
                
            except Exception as e:
                logger.error(f"Model health monitor error: {e}")
    
    async def performance_optimization_loop(self):
        """Continuously optimize model performance"""
        while True:
            try:
                await asyncio.sleep(300)  # Optimize every 5 minutes
                
                for model_name, config in self.models.items():
                    optimized_config = await self.performance_optimizer.optimize_model_config(model_name, config)
                    self.models[model_name] = optimized_config
                
            except Exception as e:
                logger.error(f"Performance optimization loop error: {e}")
    
    async def model_warm_up_loop(self):
        """Periodically warm up models to keep them responsive"""
        while True:
            try:
                await asyncio.sleep(1800)  # Warm up every 30 minutes
                
                for model_name, config in self.models.items():
                    if config.is_loaded and config.last_used:
                        time_since_use = datetime.now() - config.last_used
                        if time_since_use > timedelta(minutes=20):  # Warm up if idle for 20+ minutes
                            logger.info(f"Warming up idle model: {model_name}")
                            await self.warm_up_model(model_name)
                
            except Exception as e:
                logger.error(f"Model warm-up loop error: {e}")

# Global model manager instance
model_manager = EnhancedModelManager()