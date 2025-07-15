"""Model Management API"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class GenerateRequest(BaseModel):
    prompt: str
    model_id: str = "llama2-7b"
    max_length: int = 100
    temperature: float = 0.7

class ModelResponse(BaseModel):
    success: bool
    data: Dict[str, Any] = {}
    error: Optional[str] = None

# Create router
models_router = APIRouter(prefix="/api/models", tags=["models"])

@models_router.post("/generate", response_model=ModelResponse)
async def generate_text(request: GenerateRequest):
    """Generate text using specified model"""
    try:
        from backend.ai.ollama_manager import ollama_manager
        
        result = await ollama_manager.generate_text(
            request.model_id,
            request.prompt,
            temperature=request.temperature
        )
        
        return ModelResponse(success=True, data=result)
        
    except Exception as e:
        logger.error(f"Text generation failed: {e}")
        return ModelResponse(success=False, error=str(e))

@models_router.get("/list", response_model=ModelResponse)
async def list_models():
    """List all available models"""
    try:
        from backend.ai.ollama_manager import ollama_manager
        from config.models.registry import model_registry
        
        models = {
            "ollama_models": ollama_manager.list_available_models(),
            "registry_models": [m.model_id for m in model_registry.list_models()]
        }
        
        return ModelResponse(success=True, data=models)
        
    except Exception as e:
        logger.error(f"Failed to list models: {e}")
        return ModelResponse(success=False, error=str(e))

@models_router.get("/status", response_model=ModelResponse)
async def get_model_status():
    """Get status of model management system"""
    try:
        from backend.ai.ollama_manager import ollama_manager
        
        status = {
            "ollama": ollama_manager.get_status(),
            "system_status": "operational"
        }
        
        return ModelResponse(success=True, data=status)
        
    except Exception as e:
        logger.error(f"Failed to get status: {e}")
        return ModelResponse(success=False, error=str(e))

@models_router.post("/initialize", response_model=ModelResponse)
async def initialize_models():
    """Initialize model management system"""
    try:
        from backend.ai.ollama_manager import ollama_manager
        
        await ollama_manager.initialize()
        
        return ModelResponse(
            success=True,
            data={"message": "Model management system initialized"}
        )
        
    except Exception as e:
        logger.error(f"Failed to initialize: {e}")
        return ModelResponse(success=False, error=str(e))
