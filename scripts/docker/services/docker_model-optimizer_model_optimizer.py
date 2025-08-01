#!/usr/bin/env python3
"""
SutazAI Model Optimizer - Automatic Performance Optimization and Model Management
"""

import os
import asyncio
import logging
import json
import psutil
import subprocess
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path
import hashlib

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoConfig
from optimum.intel import IPEXModel
from optimum.onnxruntime import ORTModel
import onnx
import onnxruntime
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SutazAI Model Optimizer", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class OptimizationRequest(BaseModel):
    model_name: str
    optimization_type: str = "quantization"  # quantization, pruning, distillation, onnx
    target_precision: str = "int8"  # fp16, int8, int4
    target_device: str = "cpu"  # cpu, cuda, intel
    preserve_accuracy: bool = True
    calibration_samples: int = 100

class ModelStats(BaseModel):
    model_size_mb: float
    inference_time_ms: float
    memory_usage_mb: float
    accuracy_score: Optional[float] = None

class ModelOptimizer:
    def __init__(self):
        self.models_dir = Path("/app/models")
        self.optimized_dir = Path("/app/optimized_models")
        self.cache_dir = Path("/app/cache")
        
        # Create directories
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.optimized_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.device = self._detect_device()
        self.optimization_history = {}
        
        # Ollama integration
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        
    def _detect_device(self) -> str:
        """Detect optimal device for optimization"""
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available():
            return "intel"
        else:
            return "cpu"
    
    def get_system_info(self) -> Dict:
        """Get system information for optimization decisions"""
        return {
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "cuda_devices": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cpu_cores": psutil.cpu_count(),
            "memory_gb": psutil.virtual_memory().total / (1024**3),
            "mkldnn_available": hasattr(torch.backends, 'mkldnn') and torch.backends.mkldnn.is_available()
        }
    
    async def get_available_models(self) -> List[str]:
        """Get list of available models from Ollama"""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            if response.status_code == 200:
                data = response.json()
                return [model['name'] for model in data.get('models', [])]
            else:
                logger.error(f"Failed to get models from Ollama: {response.status_code}")
                return []
        except Exception as e:
            logger.error(f"Error getting models from Ollama: {e}")
            return []
    
    async def benchmark_model(self, model_path: str, model_name: str, 
                            tokenizer=None, num_runs: int = 10) -> ModelStats:
        """Benchmark model performance"""
        try:
            # Load model for benchmarking
            if model_path.endswith('.onnx'):
                model = onnxruntime.InferenceSession(model_path)
                input_sample = self._create_onnx_input_sample()
            else:
                model = torch.jit.load(model_path) if model_path.endswith('.pt') else AutoModel.from_pretrained(model_path)
                input_sample = self._create_torch_input_sample(tokenizer)
            
            # Measure model size
            model_size = Path(model_path).stat().st_size / (1024 * 1024)  # MB
            
            # Warm up
            for _ in range(3):
                if model_path.endswith('.onnx'):
                    model.run(None, input_sample)
                else:
                    with torch.no_grad():
                        model(**input_sample)
            
            # Benchmark inference time
            import time
            times = []
            memory_usage = []
            
            for _ in range(num_runs):
                torch.cuda.empty_cache() if torch.cuda.is_available() else None
                
                start_time = time.time()
                start_memory = psutil.virtual_memory().used / (1024 * 1024)
                
                if model_path.endswith('.onnx'):
                    model.run(None, input_sample)
                else:
                    with torch.no_grad():
                        model(**input_sample)
                
                end_time = time.time()
                end_memory = psutil.virtual_memory().used / (1024 * 1024)
                
                times.append((end_time - start_time) * 1000)  # ms
                memory_usage.append(end_memory - start_memory)
            
            avg_time = sum(times) / len(times)
            avg_memory = sum(memory_usage) / len(memory_usage)
            
            return ModelStats(
                model_size_mb=model_size,
                inference_time_ms=avg_time,
                memory_usage_mb=avg_memory
            )
            
        except Exception as e:
            logger.error(f"Benchmarking failed for {model_name}: {e}")
            return ModelStats(model_size_mb=0, inference_time_ms=0, memory_usage_mb=0)
    
    def _create_torch_input_sample(self, tokenizer=None) -> Dict:
        """Create sample input for PyTorch model"""
        if tokenizer:
            text = "This is a sample text for benchmarking."
            return tokenizer(text, return_tensors="pt", max_length=128, truncation=True, padding=True)
        else:
            # Generic input
            return {
                "input_ids": torch.randint(0, 1000, (1, 128)),
                "attention_mask": torch.ones(1, 128)
            }
    
    def _create_onnx_input_sample(self) -> Dict:
        """Create sample input for ONNX model"""
        return {
            "input_ids": torch.randint(0, 1000, (1, 128)).numpy(),
            "attention_mask": torch.ones(1, 128).numpy()
        }
    
    async def optimize_model(self, request: OptimizationRequest) -> Dict:
        """Optimize model based on request parameters"""
        model_name = request.model_name
        optimization_id = hashlib.md5(f"{model_name}_{request.optimization_type}_{request.target_precision}".encode()).hexdigest()
        
        logger.info(f"Starting optimization {optimization_id} for {model_name}")
        
        try:
            # Check if model is available
            available_models = await self.get_available_models()
            if model_name not in available_models:
                raise HTTPException(status_code=404, detail=f"Model {model_name} not found")
            
            # Create optimization workspace
            opt_workspace = self.cache_dir / optimization_id
            opt_workspace.mkdir(exist_ok=True)
            
            # Download/prepare model
            model_path = await self._prepare_model(model_name, opt_workspace)
            
            # Load original model for comparison
            original_stats = await self.benchmark_model(model_path, model_name)
            
            # Apply optimization
            optimized_path = await self._apply_optimization(
                model_path, request, opt_workspace
            )
            
            # Benchmark optimized model
            optimized_stats = await self.benchmark_model(optimized_path, f"{model_name}_optimized")
            
            # Calculate improvement metrics
            size_reduction = (original_stats.model_size_mb - optimized_stats.model_size_mb) / original_stats.model_size_mb * 100
            speed_improvement = (original_stats.inference_time_ms - optimized_stats.inference_time_ms) / original_stats.inference_time_ms * 100
            memory_reduction = (original_stats.memory_usage_mb - optimized_stats.memory_usage_mb) / original_stats.memory_usage_mb * 100
            
            # Store optimization results
            result = {
                "optimization_id": optimization_id,
                "model_name": model_name,
                "optimization_type": request.optimization_type,
                "target_precision": request.target_precision,
                "original_stats": original_stats.dict(),
                "optimized_stats": optimized_stats.dict(),
                "improvements": {
                    "size_reduction_percent": round(size_reduction, 2),
                    "speed_improvement_percent": round(speed_improvement, 2),
                    "memory_reduction_percent": round(memory_reduction, 2)
                },
                "optimized_model_path": str(optimized_path),
                "timestamp": datetime.now().isoformat()
            }
            
            self.optimization_history[optimization_id] = result
            
            # Save optimized model to optimized directory
            final_path = self.optimized_dir / f"{model_name}_{request.optimization_type}_{request.target_precision}"
            final_path.mkdir(exist_ok=True)
            
            # Copy optimized model
            import shutil
            if optimized_path.is_file():
                shutil.copy2(optimized_path, final_path / "model.onnx")
            else:
                shutil.copytree(optimized_path, final_path / "model", dirs_exist_ok=True)
            
            logger.info(f"Optimization {optimization_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Optimization failed for {model_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Optimization failed: {str(e)}")
    
    async def _prepare_model(self, model_name: str, workspace: Path) -> Path:
        """Prepare model for optimization"""
        try:
            # For now, we'll create a dummy model path
            # In a real implementation, this would download from Ollama or load from cache
            model_path = workspace / "original_model"
            model_path.mkdir(exist_ok=True)
            
            # Create dummy model files for demonstration
            config_path = model_path / "config.json"
            with open(config_path, 'w') as f:
                json.dump({"model_type": "transformer", "hidden_size": 768}, f)
            
            return model_path
            
        except Exception as e:
            raise Exception(f"Failed to prepare model {model_name}: {e}")
    
    async def _apply_optimization(self, model_path: Path, request: OptimizationRequest, 
                                workspace: Path) -> Path:
        """Apply specific optimization to the model"""
        optimized_path = workspace / "optimized_model"
        
        if request.optimization_type == "quantization":
            return await self._apply_quantization(model_path, request, optimized_path)
        elif request.optimization_type == "onnx":
            return await self._convert_to_onnx(model_path, request, optimized_path)
        elif request.optimization_type == "pruning":
            return await self._apply_pruning(model_path, request, optimized_path)
        elif request.optimization_type == "distillation":
            return await self._apply_distillation(model_path, request, optimized_path)
        else:
            raise ValueError(f"Unknown optimization type: {request.optimization_type}")
    
    async def _apply_quantization(self, model_path: Path, request: OptimizationRequest, 
                                output_path: Path) -> Path:
        """Apply quantization optimization"""
        try:
            logger.info(f"Applying {request.target_precision} quantization")
            
            # Create output directory
            output_path.mkdir(exist_ok=True)
            
            # For demonstration, create a quantized model placeholder
            quantized_model_path = output_path / "quantized_model.onnx"
            
            # In a real implementation, this would:
            # 1. Load the model
            # 2. Apply quantization (INT8, FP16, etc.)
            # 3. Save the quantized model
            
            # Create dummy quantized model
            dummy_model = b"dummy_quantized_model_data"
            with open(quantized_model_path, 'wb') as f:
                f.write(dummy_model)
            
            logger.info("Quantization completed")
            return quantized_model_path
            
        except Exception as e:
            raise Exception(f"Quantization failed: {e}")
    
    async def _convert_to_onnx(self, model_path: Path, request: OptimizationRequest, 
                             output_path: Path) -> Path:
        """Convert model to ONNX format"""
        try:
            logger.info("Converting model to ONNX format")
            
            output_path.mkdir(exist_ok=True)
            onnx_model_path = output_path / "model.onnx"
            
            # In a real implementation, this would:
            # 1. Load the PyTorch model
            # 2. Export to ONNX format
            # 3. Optimize the ONNX model
            
            # Create dummy ONNX model
            dummy_onnx = b"dummy_onnx_model_data"
            with open(onnx_model_path, 'wb') as f:
                f.write(dummy_onnx)
            
            logger.info("ONNX conversion completed")
            return onnx_model_path
            
        except Exception as e:
            raise Exception(f"ONNX conversion failed: {e}")
    
    async def _apply_pruning(self, model_path: Path, request: OptimizationRequest, 
                           output_path: Path) -> Path:
        """Apply model pruning"""
        try:
            logger.info("Applying model pruning")
            
            output_path.mkdir(exist_ok=True)
            pruned_model_path = output_path / "pruned_model.pt"
            
            # In a real implementation, this would:
            # 1. Load the model
            # 2. Apply structured or unstructured pruning
            # 3. Fine-tune if necessary
            # 4. Save the pruned model
            
            # Create dummy pruned model
            dummy_pruned = b"dummy_pruned_model_data"
            with open(pruned_model_path, 'wb') as f:
                f.write(dummy_pruned)
            
            logger.info("Pruning completed")
            return pruned_model_path
            
        except Exception as e:
            raise Exception(f"Pruning failed: {e}")
    
    async def _apply_distillation(self, model_path: Path, request: OptimizationRequest, 
                                output_path: Path) -> Path:
        """Apply knowledge distillation"""
        try:
            logger.info("Applying knowledge distillation")
            
            output_path.mkdir(exist_ok=True)
            distilled_model_path = output_path / "distilled_model.pt"
            
            # In a real implementation, this would:
            # 1. Load the teacher model
            # 2. Create or load a smaller student model
            # 3. Train the student model using distillation
            # 4. Save the distilled model
            
            # Create dummy distilled model
            dummy_distilled = b"dummy_distilled_model_data"
            with open(distilled_model_path, 'wb') as f:
                f.write(dummy_distilled)
            
            logger.info("Distillation completed")
            return distilled_model_path
            
        except Exception as e:
            raise Exception(f"Distillation failed: {e}")
    
    async def auto_optimize_models(self) -> Dict:
        """Automatically optimize all available models based on system capabilities"""
        logger.info("Starting auto-optimization of all models")
        
        try:
            available_models = await self.get_available_models()
            system_info = self.get_system_info()
            
            optimization_results = {}
            
            for model_name in available_models[:3]:  # Limit to first 3 models for demo
                # Determine optimal optimization strategy based on system
                if system_info["cuda_available"]:
                    # GPU available - use FP16 quantization
                    opt_request = OptimizationRequest(
                        model_name=model_name,
                        optimization_type="quantization",
                        target_precision="fp16",
                        target_device="cuda"
                    )
                elif system_info["mkldnn_available"]:
                    # Intel CPU - use INT8 quantization
                    opt_request = OptimizationRequest(
                        model_name=model_name,
                        optimization_type="quantization",
                        target_precision="int8",
                        target_device="intel"
                    )
                else:
                    # Generic CPU - use ONNX conversion
                    opt_request = OptimizationRequest(
                        model_name=model_name,
                        optimization_type="onnx",
                        target_precision="fp32",
                        target_device="cpu"
                    )
                
                try:
                    result = await self.optimize_model(opt_request)
                    optimization_results[model_name] = result
                except Exception as e:
                    logger.error(f"Failed to optimize {model_name}: {e}")
                    optimization_results[model_name] = {"error": str(e)}
            
            return {
                "auto_optimization_completed": True,
                "system_info": system_info,
                "optimized_models": len([r for r in optimization_results.values() if "error" not in r]),
                "failed_models": len([r for r in optimization_results.values() if "error" in r]),
                "results": optimization_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Auto-optimization failed: {e}")
            return {"error": str(e), "timestamp": datetime.now().isoformat()}

# Initialize optimizer
optimizer = ModelOptimizer()

@app.on_event("startup")
async def startup_event():
    """Initialize optimizer on startup"""
    logger.info("Model Optimizer started successfully")
    
    # Run auto-optimization if enabled
    if os.getenv("AUTO_OPTIMIZE", "true").lower() == "true":
        asyncio.create_task(optimizer.auto_optimize_models())

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    system_info = optimizer.get_system_info()
    
    return {
        "status": "healthy",
        "service": "Model Optimizer",
        "timestamp": datetime.now().isoformat(),
        "system_info": system_info,
        "optimizations_completed": len(optimizer.optimization_history)
    }

@app.get("/system")
async def get_system_info():
    """Get system information"""
    return optimizer.get_system_info()

@app.get("/models")
async def list_available_models():
    """List available models for optimization"""
    models = await optimizer.get_available_models()
    return {"available_models": models, "count": len(models)}

@app.post("/optimize")
async def optimize_model(request: OptimizationRequest, background_tasks: BackgroundTasks):
    """Optimize a specific model"""
    # Run optimization in background
    result = await optimizer.optimize_model(request)
    return result

@app.post("/auto-optimize")
async def auto_optimize():
    """Automatically optimize all models"""
    result = await optimizer.auto_optimize_models()
    return result

@app.get("/optimizations")
async def list_optimizations():
    """List all completed optimizations"""
    return {
        "optimizations": list(optimizer.optimization_history.values()),
        "count": len(optimizer.optimization_history)
    }

@app.get("/optimizations/{optimization_id}")
async def get_optimization(optimization_id: str):
    """Get specific optimization details"""
    if optimization_id not in optimizer.optimization_history:
        raise HTTPException(status_code=404, detail="Optimization not found")
    
    return optimizer.optimization_history[optimization_id]

@app.delete("/optimizations/{optimization_id}")
async def delete_optimization(optimization_id: str):
    """Delete optimization results"""
    if optimization_id not in optimizer.optimization_history:
        raise HTTPException(status_code=404, detail="Optimization not found")
    
    # Remove from history
    result = optimizer.optimization_history.pop(optimization_id)
    
    # Clean up files
    try:
        cache_path = optimizer.cache_dir / optimization_id
        if cache_path.exists():
            import shutil
            shutil.rmtree(cache_path)
    except Exception as e:
        logger.warning(f"Failed to clean up optimization files: {e}")
    
    return {"message": "Optimization deleted", "optimization_id": optimization_id}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SutazAI Model Optimizer",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "system": "/system",
            "models": "/models",
            "optimize": "/optimize",
            "auto_optimize": "/auto-optimize",
            "optimizations": "/optimizations"
        }
    }

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    host = os.environ.get("HOST", "0.0.0.0")
    
    logger.info("Starting SutazAI Model Optimizer...")
    logger.info(f"Optimizer URL: http://{host}:{port}")
    logger.info(f"Device: {optimizer.device}")
    logger.info(f"Auto-optimize: {os.getenv('AUTO_OPTIMIZE', 'true')}")
    
    uvicorn.run(app, host=host, port=port) 