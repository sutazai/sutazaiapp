#!/usr/bin/env python3
"""
Enhanced Model Management Service for SutazAI
Specialized management for GPT-OSS and other advanced models
"""

import os
import json
import logging
import asyncio
import aiohttp
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import time
import uvicorn
from pathlib import Path
from model_manager import EnhancedModelManager
from gpt_oss_integration import GPTOSSIntegration

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="SutazAI Enhanced Model Manager",
    description="Advanced model management with GPT-OSS and optimized inference",
    version="2.0.0"
)

class GenerationRequest(BaseModel):
    model_name: str
    prompt: str
    max_tokens: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    stop_sequences: Optional[List[str]] = None

class ModelLoadRequest(BaseModel):
    model_name: str
    model_path: Optional[str] = None
    quantization: Optional[str] = None
    device: str = "auto"

class BatchGenerationRequest(BaseModel):
    model_name: str
    prompts: List[str]
    max_tokens: int = 2048
    temperature: float = 0.7

# Initialize Enhanced Model Manager
model_manager = EnhancedModelManager()
gpt_oss_integration = GPTOSSIntegration()

@app.on_event("startup")
async def startup_event():
    """Initialize the enhanced model service"""
    logger.info("Starting Enhanced Model Manager...")
    await model_manager.initialize()
    await gpt_oss_integration.initialize()
    logger.info("Enhanced Model Manager initialized successfully")

@app.get("/")
async def root():
    return {
        "service": "SutazAI Enhanced Model Manager",
        "status": "running",
        "version": "2.0.0",
        "features": ["GPT-OSS", "Batch Processing", "Model Optimization"]
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "loaded_models": len(model_manager.loaded_models),
        "available_models": len(model_manager.available_models)
    }

@app.get("/models")
async def list_models():
    """List all available models"""
    models = model_manager.get_available_models()
    return {"models": models, "total": len(models)}

@app.get("/models/loaded")
async def list_loaded_models():
    """List currently loaded models"""
    loaded = model_manager.get_loaded_models()
    return {"loaded_models": loaded, "total": len(loaded)}

@app.post("/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a model into memory"""
    try:
        result = await model_manager.load_model(
            request.model_name,
            request.model_path,
            request.quantization,
            request.device
        )
        return {"message": f"Model {request.model_name} loaded successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_name}")
async def unload_model(model_name: str):
    """Unload a model from memory"""
    try:
        result = await model_manager.unload_model(model_name)
        return {"message": f"Model {model_name} unloaded successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
async def generate_text(request: GenerationRequest):
    """Generate text using a loaded model"""
    try:
        result = await model_manager.generate(
            request.model_name,
            request.prompt,
            request.max_tokens,
            request.temperature,
            request.top_p,
            request.stop_sequences
        )
        return {
            "generated_text": result,
            "model": request.model_name,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate/batch")
async def batch_generate(request: BatchGenerationRequest):
    """Generate text for multiple prompts in batch"""
    try:
        results = await model_manager.batch_generate(
            request.model_name,
            request.prompts,
            request.max_tokens,
            request.temperature
        )
        return {
            "results": results,
            "model": request.model_name,
            "batch_size": len(request.prompts),
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code/generate")
async def generate_code(prompt: str, language: str = "python", model: str = "tinyllama"):
    """Generate code using GPT-OSS or other code models"""
    try:
        if model == "tinyllama":
            result = await gpt_oss_integration.generate_code(prompt, language)
        else:
            result = await model_manager.generate_code(prompt, language, model)
        
        return {
            "generated_code": result,
            "language": language,
            "model": model,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code/complete")
async def complete_code(code: str, language: str = "python", model: str = "tinyllama"):
    """Complete partial code"""
    try:
        if model == "tinyllama":
            result = await gpt_oss_integration.complete_code(code, language)
        else:
            result = await model_manager.complete_code(code, language, model)
        
        return {
            "completed_code": result,
            "language": language,
            "model": model,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code/explain")
async def explain_code(code: str, language: str = "python", model: str = "tinyllama"):
    """Explain what a piece of code does"""
    try:
        if model == "tinyllama":
            explanation = await gpt_oss_integration.explain_code(code, language)
        else:
            explanation = await model_manager.explain_code(code, language, model)
        
        return {
            "explanation": explanation,
            "code": code,
            "language": language,
            "model": model,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/code/optimize")
async def optimize_code(code: str, language: str = "python", model: str = "tinyllama"):
    """Optimize code for performance and readability"""
    try:
        if model == "tinyllama":
            optimized = await gpt_oss_integration.optimize_code(code, language)
        else:
            optimized = await model_manager.optimize_code(code, language, model)
        
        return {
            "original_code": code,
            "optimized_code": optimized,
            "language": language,
            "model": model,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/models/download")
async def download_model(model_name: str, background_tasks: BackgroundTasks):
    """Download a model from Hugging Face"""
    try:
        background_tasks.add_task(model_manager.download_model, model_name)
        return {"message": f"Download started for {model_name}", "status": "downloading"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models/{model_name}/status")
async def get_model_status(model_name: str):
    """Get status of a specific model"""
    try:
        status = model_manager.get_model_status(model_name)
        return status
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))

@app.post("/models/optimize")
async def optimize_model(model_name: str, optimization_type: str = "quantization"):
    """Optimize a model for faster inference"""
    try:
        result = await model_manager.optimize_model(model_name, optimization_type)
        return {"message": f"Model {model_name} optimized successfully", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/stats")
async def get_system_stats():
    """Get system resource usage and model statistics"""
    stats = model_manager.get_system_stats()
    return {
        "system_stats": stats,
        "timestamp": time.time()
    }

@app.post("/models/benchmark")
async def benchmark_model(model_name: str, test_prompts: List[str]):
    """Benchmark model performance"""
    try:
        results = await model_manager.benchmark_model(model_name, test_prompts)
        return {
            "benchmark_results": results,
            "model": model_name,
            "timestamp": time.time()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "enhanced_model_service:app",
        host="0.0.0.0",
        port=8090,
        log_level="info",
        reload=False
    )