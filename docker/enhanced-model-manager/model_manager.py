#!/usr/bin/env python3
"""
Enhanced Model Manager for advanced AI model handling
Optimized for DeepSeek-Coder, Llama 2, and other large language models
"""

import os
import json
import asyncio
import logging
import psutil
import torch
from pathlib import Path
from typing import Dict, List, Any, Optional
import time
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, pipeline
)
import gc

logger = logging.getLogger(__name__)

class EnhancedModelManager:
    """Advanced model management with optimization and caching"""
    
    def __init__(self, cache_dir: str = "/data/models"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.loaded_models = {}
        self.model_configs = {}
        self.tokenizers = {}
        self.pipelines = {}
        
        # Model registry with optimized configurations
        self.available_models = {
            "deepseek-coder-33b": {
                "hf_name": "deepseek-ai/deepseek-coder-33b-base",
                "type": "code_generation",
                "size": "33B",
                "quantization_supported": True,
                "gpu_memory_required": "24GB",
                "description": "Advanced code generation and understanding model"
            },
            "deepseek-coder-7b": {
                "hf_name": "deepseek-ai/deepseek-coder-7b-base",
                "type": "code_generation", 
                "size": "7B",
                "quantization_supported": True,
                "gpu_memory_required": "8GB",
                "description": "Smaller code generation model for faster inference"
            },
            "llama2-7b": {
                "hf_name": "meta-llama/Llama-2-7b-hf",
                "type": "general",
                "size": "7B",
                "quantization_supported": True,
                "gpu_memory_required": "8GB",
                "description": "General purpose language model"
            },
            "llama2-13b": {
                "hf_name": "meta-llama/Llama-2-13b-hf",
                "type": "general",
                "size": "13B", 
                "quantization_supported": True,
                "gpu_memory_required": "16GB",
                "description": "Larger general purpose language model"
            },
            "codellama-7b": {
                "hf_name": "codellama/CodeLlama-7b-Python-hf",
                "type": "code_generation",
                "size": "7B",
                "quantization_supported": True,
                "gpu_memory_required": "8GB",
                "description": "Code-specialized Llama model"
            }
        }
        
        self.start_time = time.time()
        self.stats = {
            "models_loaded": 0,
            "total_generations": 0,
            "cache_hits": 0,
            "memory_optimizations": 0
        }
    
    async def initialize(self):
        """Initialize the model manager"""
        logger.info("Initializing Enhanced Model Manager...")
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        if self.device == "cuda":
            logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
        
        # Load any pre-cached models
        await self.load_cached_models()
        
        logger.info("Enhanced Model Manager initialized")
    
    async def load_cached_models(self):
        """Load any models that are already cached"""
        cache_info_file = self.cache_dir / "cache_info.json"
        if cache_info_file.exists():
            with open(cache_info_file, 'r') as f:
                cache_info = json.load(f)
                logger.info(f"Found {len(cache_info)} cached models")
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available models"""
        return list(self.available_models.values())
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models"""
        return list(self.loaded_models.keys())
    
    async def load_model(self, model_name: str, model_path: Optional[str] = None, 
                        quantization: Optional[str] = None, device: str = "auto") -> Dict[str, Any]:
        """Load a model into memory with optimizations"""
        
        if model_name in self.loaded_models:
            logger.info(f"Model {model_name} already loaded")
            return {"status": "already_loaded", "model": model_name}
        
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        model_config = self.available_models[model_name]
        hf_name = model_path or model_config["hf_name"]
        
        logger.info(f"Loading model: {model_name} ({hf_name})")
        
        try:
            # Configure quantization if requested
            quantization_config = None
            if quantization == "4bit":
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            elif quantization == "8bit":
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            
            # Load tokenizer
            logger.info(f"Loading tokenizer for {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(
                hf_name,
                cache_dir=self.cache_dir,
                trust_remote_code=True
            )
            
            # Load model with optimizations
            logger.info(f"Loading model {model_name} with optimizations")
            model = AutoModelForCausalLM.from_pretrained(
                hf_name,
                cache_dir=self.cache_dir,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if device == "auto" else device,
                quantization_config=quantization_config,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # Create generation pipeline
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                device_map="auto" if device == "auto" else device,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32
            )
            
            # Store loaded components
            self.loaded_models[model_name] = model
            self.tokenizers[model_name] = tokenizer
            self.pipelines[model_name] = pipe
            self.model_configs[model_name] = model_config
            
            self.stats["models_loaded"] += 1
            
            logger.info(f"Successfully loaded model: {model_name}")
            
            return {
                "status": "loaded",
                "model": model_name,
                "quantization": quantization,
                "device": str(model.device) if hasattr(model, 'device') else device,
                "memory_usage": self.get_model_memory_usage(model_name)
            }
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    async def unload_model(self, model_name: str) -> Dict[str, Any]:
        """Unload a model from memory"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        logger.info(f"Unloading model: {model_name}")
        
        # Remove from memory
        del self.loaded_models[model_name]
        del self.tokenizers[model_name]
        del self.pipelines[model_name]
        if model_name in self.model_configs:
            del self.model_configs[model_name]
        
        # Force garbage collection
        gc.collect()
        if self.device == "cuda":
            torch.cuda.empty_cache()
        
        logger.info(f"Successfully unloaded model: {model_name}")
        
        return {"status": "unloaded", "model": model_name}
    
    async def generate(self, model_name: str, prompt: str, max_tokens: int = 2048,
                      temperature: float = 0.7, top_p: float = 0.9,
                      stop_sequences: Optional[List[str]] = None) -> str:
        """Generate text using a loaded model"""
        
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        pipeline = self.pipelines[model_name]
        
        # Configure generation parameters
        generation_kwargs = {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "do_sample": True,
            "pad_token_id": pipeline.tokenizer.eos_token_id,
        }
        
        if stop_sequences:
            generation_kwargs["stop_sequence"] = stop_sequences
        
        try:
            # Generate text
            result = pipeline(prompt, **generation_kwargs)
            generated_text = result[0]["generated_text"]
            
            # Remove the original prompt from the result
            if generated_text.startswith(prompt):
                generated_text = generated_text[len(prompt):].strip()
            
            self.stats["total_generations"] += 1
            
            return generated_text
            
        except Exception as e:
            logger.error(f"Generation failed for model {model_name}: {e}")
            raise
    
    async def batch_generate(self, model_name: str, prompts: List[str], 
                           max_tokens: int = 2048, temperature: float = 0.7) -> List[str]:
        """Generate text for multiple prompts in batch"""
        
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        results = []
        for prompt in prompts:
            result = await self.generate(model_name, prompt, max_tokens, temperature)
            results.append(result)
        
        return results
    
    async def generate_code(self, prompt: str, language: str = "python", 
                          model: str = "deepseek-coder-7b") -> str:
        """Generate code using code-specialized models"""
        
        code_prompt = f"# Generate {language} code for: {prompt}\n```{language}\n"
        
        if model not in self.loaded_models:
            # Auto-load if available
            if model in self.available_models:
                await self.load_model(model)
            else:
                raise ValueError(f"Code model {model} not available")
        
        generated = await self.generate(model, code_prompt, max_tokens=1024, temperature=0.2)
        
        # Extract code from markdown if present
        if "```" in generated:
            code_blocks = generated.split("```")
            if len(code_blocks) >= 2:
                generated = code_blocks[1].strip()
                # Remove language identifier if present
                lines = generated.split('\n')
                if lines[0].strip() in ['python', 'javascript', 'java', 'cpp', 'c', 'go']:
                    generated = '\n'.join(lines[1:])
        
        return generated.strip()
    
    async def complete_code(self, code: str, language: str = "python", 
                          model: str = "deepseek-coder-7b") -> str:
        """Complete partial code"""
        
        completion_prompt = f"# Complete this {language} code:\n```{language}\n{code}"
        
        if model not in self.loaded_models:
            if model in self.available_models:
                await self.load_model(model)
            else:
                raise ValueError(f"Code model {model} not available")
        
        completed = await self.generate(model, completion_prompt, max_tokens=512, temperature=0.1)
        
        return completed.strip()
    
    async def explain_code(self, code: str, language: str = "python", 
                         model: str = "deepseek-coder-7b") -> str:
        """Explain what a piece of code does"""
        
        explain_prompt = f"Explain what this {language} code does:\n```{language}\n{code}\n```\n\nExplanation:"
        
        if model not in self.loaded_models:
            if model in self.available_models:
                await self.load_model(model)
            else:
                raise ValueError(f"Code model {model} not available")
        
        explanation = await self.generate(model, explain_prompt, max_tokens=256, temperature=0.3)
        
        return explanation.strip()
    
    async def optimize_code(self, code: str, language: str = "python", 
                          model: str = "deepseek-coder-7b") -> str:
        """Optimize code for performance and readability"""
        
        optimize_prompt = f"Optimize this {language} code for better performance and readability:\n```{language}\n{code}\n```\n\nOptimized code:\n```{language}\n"
        
        if model not in self.loaded_models:
            if model in self.available_models:
                await self.load_model(model)
            else:
                raise ValueError(f"Code model {model} not available")
        
        optimized = await self.generate(model, optimize_prompt, max_tokens=1024, temperature=0.2)
        
        # Extract optimized code
        if "```" in optimized:
            code_blocks = optimized.split("```")
            if len(code_blocks) >= 2:
                optimized = code_blocks[1].strip()
        
        return optimized.strip()
    
    async def download_model(self, model_name: str):
        """Download a model from Hugging Face"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        model_config = self.available_models[model_name]
        hf_name = model_config["hf_name"]
        
        logger.info(f"Downloading model: {model_name} ({hf_name})")
        
        # Download tokenizer and model
        AutoTokenizer.from_pretrained(hf_name, cache_dir=self.cache_dir)
        AutoModelForCausalLM.from_pretrained(hf_name, cache_dir=self.cache_dir)
        
        logger.info(f"Successfully downloaded model: {model_name}")
    
    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get status of a specific model"""
        if model_name not in self.available_models:
            raise ValueError(f"Model {model_name} not available")
        
        is_loaded = model_name in self.loaded_models
        is_cached = self.is_model_cached(model_name)
        
        status = {
            "model_name": model_name,
            "is_loaded": is_loaded,
            "is_cached": is_cached,
            "config": self.available_models[model_name]
        }
        
        if is_loaded:
            status["memory_usage"] = self.get_model_memory_usage(model_name)
            status["device"] = str(self.loaded_models[model_name].device)
        
        return status
    
    def is_model_cached(self, model_name: str) -> bool:
        """Check if a model is cached locally"""
        if model_name not in self.available_models:
            return False
        
        hf_name = self.available_models[model_name]["hf_name"]
        model_path = self.cache_dir / f"models--{hf_name.replace('/', '--')}"
        
        return model_path.exists()
    
    def get_model_memory_usage(self, model_name: str) -> Dict[str, Any]:
        """Get memory usage information for a loaded model"""
        if model_name not in self.loaded_models:
            return {"error": "Model not loaded"}
        
        model = self.loaded_models[model_name]
        
        # Calculate model size
        param_count = sum(p.numel() for p in model.parameters())
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        
        memory_info = {
            "parameter_count": param_count,
            "parameter_size_mb": param_size / (1024 * 1024),
            "parameter_size_gb": param_size / (1024 * 1024 * 1024)
        }
        
        if self.device == "cuda":
            memory_info["gpu_memory_allocated"] = torch.cuda.memory_allocated() / (1024 * 1024)
            memory_info["gpu_memory_reserved"] = torch.cuda.memory_reserved() / (1024 * 1024)
        
        return memory_info
    
    async def optimize_model(self, model_name: str, optimization_type: str = "quantization") -> Dict[str, Any]:
        """Optimize a model for faster inference"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        logger.info(f"Optimizing model {model_name} with {optimization_type}")
        
        if optimization_type == "quantization":
            # Reload with quantization
            await self.unload_model(model_name)
            result = await self.load_model(model_name, quantization="4bit")
            self.stats["memory_optimizations"] += 1
            return result
        
        return {"status": "optimization_not_supported", "type": optimization_type}
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system resource usage and statistics"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        
        stats = {
            "uptime": time.time() - self.start_time,
            "cpu_usage_percent": cpu_percent,
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "loaded_models": len(self.loaded_models),
            "total_models_available": len(self.available_models),
            "generation_stats": self.stats
        }
        
        if self.device == "cuda":
            stats["gpu_memory_used_mb"] = torch.cuda.memory_allocated() / (1024 * 1024)
            stats["gpu_memory_total_mb"] = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        
        return stats
    
    async def benchmark_model(self, model_name: str, test_prompts: List[str]) -> Dict[str, Any]:
        """Benchmark model performance"""
        if model_name not in self.loaded_models:
            raise ValueError(f"Model {model_name} not loaded")
        
        logger.info(f"Benchmarking model: {model_name}")
        
        times = []
        token_counts = []
        
        for prompt in test_prompts:
            start_time = time.time()
            result = await self.generate(model_name, prompt, max_tokens=100)
            end_time = time.time()
            
            generation_time = end_time - start_time
            token_count = len(self.tokenizers[model_name].encode(result))
            
            times.append(generation_time)
            token_counts.append(token_count)
        
        avg_time = sum(times) / len(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        tokens_per_second = avg_tokens / avg_time
        
        return {
            "model": model_name,
            "test_count": len(test_prompts),
            "avg_generation_time": avg_time,
            "avg_tokens_generated": avg_tokens,
            "tokens_per_second": tokens_per_second,
            "total_time": sum(times)
        }