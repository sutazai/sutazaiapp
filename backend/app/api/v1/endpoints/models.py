"""Models endpoint to list available AI models"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict
import httpx
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def list_models() -> Dict:
    """List available AI models"""
    
    # Try to get models from Ollama
    ollama_models = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get("http://sutazai-ollama:11434/api/tags")
            if response.status_code == 200:
                data = response.json()
                ollama_models = [
                    {
                        "name": model.get("name", "unknown"),
                        "provider": "ollama",
                        "size": model.get("size", 0),
                        "available": True
                    }
                    for model in data.get("models", [])
                ]
    except Exception as e:
        logger.warning(f"Could not fetch Ollama models: {e}")
    
    # Static list of models (including both available and planned)
    static_models = [
        {"name": "gpt-4", "provider": "openai", "available": False},
        {"name": "claude-3", "provider": "anthropic", "available": False},
        {"name": "local", "provider": "system", "available": True},
    ]
    
    # Combine all models
    all_models = ollama_models + static_models
    
    # Extract just the model names for simple listing
    model_names = [m["name"] for m in all_models if m.get("available", False)]
    if not model_names:
        model_names = ["tinyllama:latest", "mistral:latest", "local"]
    
    return {
        "models": model_names,
        "detailed": all_models,
        "count": len(model_names)
    }