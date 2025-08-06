from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import requests
import os
import json
import asyncio
from typing import List, Dict, Any
import torch
from transformers import AutoTokenizer, AutoModel
import logging

app = FastAPI(title="Enhanced Model Manager", version="2.0.0")

class ModelRequest(BaseModel):
    model_name: str
    prompt: str
    parameters: Dict[str, Any] = {}

class ModelInfo(BaseModel):
    name: str
    status: str
    memory_usage: str
    quantization: str

ollama_url = os.environ.get('OLLAMA_URL', 'http://ollama:11434')
auto_pull_models = os.environ.get('AUTO_PULL_MODELS', 'true').lower() == 'true'
models_cache = {}

required_models = [
    "gpt-oss"
]

logging.basicConfig(level=logging.INFO)

@app.get("/health")
def health():
    return {"status": "healthy", "models_loaded": len(models_cache)}

@app.get("/models", response_model=List[ModelInfo])
def list_models():
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        ollama_models = response.json().get('models', [])
        
        model_list = []
        for model in ollama_models:
            model_list.append(ModelInfo(
                name=model['name'],
                status="available",
                memory_usage=f"{model['size'] / 1024**3:.1f}GB",
                quantization=model['details'].get('quantization_level', 'unknown')
            ))
        
        return model_list
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/generate")
def generate_text(request: ModelRequest):
    try:
        ollama_request = {
            "model": request.model_name,
            "prompt": request.prompt,
            "stream": False,
            "options": request.parameters
        }
        
        response = requests.post(f"{ollama_url}/api/generate", json=ollama_request)
        result = response.json()
        
        return {
            "response": result.get('response', ''),
            "model": request.model_name,
            "done": result.get('done', False),
            "total_duration": result.get('total_duration', 0),
            "load_duration": result.get('load_duration', 0),
            "prompt_eval_count": result.get('prompt_eval_count', 0),
            "eval_count": result.get('eval_count', 0)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/pull")
def pull_model(model_name: str):
    try:
        response = requests.post(f"{ollama_url}/api/pull", json={"name": model_name})
        return {"status": "pulling", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/models/{model_name}")
def delete_model(model_name: str):
    try:
        response = requests.delete(f"{ollama_url}/api/delete", json={"name": model_name})
        return {"status": "deleted", "model": model_name}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/stats")
def get_stats():
    try:
        # Get system stats
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "models_loaded": len(models_cache),
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    except Exception as e:
        return {"error": str(e)}

async def pull_model_async(model_name: str):
    try:
        logging.info(f"Starting to pull model: {model_name}")
        response = requests.post(f"{ollama_url}/api/pull", json={"name": model_name}, stream=True)
        for line in response.iter_lines():
            if line:
                logging.info(f"Pull status for {model_name}: {line.decode()}")
        logging.info(f"Successfully pulled model: {model_name}")
    except Exception as e:
        logging.error(f"Error pulling model {model_name}: {str(e)}")

@app.on_event("startup")
async def startup_event():
    if auto_pull_models:
        logging.info("Auto-pulling required models...")
        for model in required_models:
            try:
                response = requests.get(f"{ollama_url}/api/tags")
                existing_models = [m['name'] for m in response.json().get('models', [])]
                if model not in existing_models:
                    asyncio.create_task(pull_model_async(model))
            except Exception as e:
                logging.error(f"Error checking/pulling model {model}: {str(e)}")

@app.post("/auto-setup")
async def auto_setup_models(background_tasks: BackgroundTasks):
    """Manually trigger automatic model setup"""
    for model in required_models:
        background_tasks.add_task(pull_model_async, model)
    return {"status": "started", "models": required_models}

@app.get("/model-status")
def get_model_status():
    """Get status of all required models"""
    try:
        response = requests.get(f"{ollama_url}/api/tags")
        existing_models = [m['name'] for m in response.json().get('models', [])]
        
        status = {}
        for model in required_models:
            status[model] = "installed" if model in existing_models else "not_installed"
        
        return {
            "total_required": len(required_models),
            "installed": sum(1 for status in status.values() if status == "installed"),
            "models": status
        }
    except Exception as e:
        return {"error": str(e)}
