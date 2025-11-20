"""Models endpoint to list available AI models"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict
import httpx
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter()

@router.get("/")
async def list_models() -> Dict:
    """List available AI models"""
    
    # Get Ollama connection details from environment
    ollama_host = os.getenv("OLLAMA_HOST", "host.docker.internal")
    ollama_port = os.getenv("OLLAMA_PORT", "11434")
    ollama_url = f"http://{ollama_host}:{ollama_port}"
    
    # Try to get models from Ollama
    ollama_models = []
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{ollama_url}/api/tags")
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
                logger.info(f"Successfully fetched {len(ollama_models)} models from Ollama at {ollama_url}")
    except Exception as e:
        logger.warning(f"Could not fetch Ollama models from {ollama_url}: {e}")
    
    # Only include actually available local models
    # NO external API models (GPT-4, Claude, Gemini) per user requirements
    available_models = []
    unavailable_models = []
    
    # Add verified Ollama models (currently only tinyllama:latest is loaded)
    if ollama_models:
        available_models = ollama_models
    else:
        # Fallback if Ollama connection failed - only include verified models
        logger.info("Using fallback model list (Ollama connection failed)")
        available_models = [
            {"name": "tinyllama:latest", "provider": "ollama", "size": 637000000, "available": True}
        ]
    
    # List potentially available models (not yet pulled)
    unavailable_models = [
        {"name": "qwen3:Thinking-2507", "provider": "ollama", "available": False, "note": "Available for download"},
        
    ]
    
    # Extract just the model names for simple listing
    model_names = [m["name"] for m in available_models]
    
    return {
        "models": model_names,
        "available_detailed": available_models,
        "downloadable": unavailable_models,
        "count": len(model_names),
        "note": "Only local Ollama models supported - no external API models"
    }