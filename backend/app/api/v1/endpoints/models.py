"""
Models endpoint for SutazAI
"""
from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from app.services.model_manager import ModelManager
from app.core.dependencies import get_model_manager
from app.core.middleware import jwt_required
import logging

logger = logging.getLogger(__name__)

router = APIRouter()

class ModelInfo(BaseModel):
    name: str
    size: Optional[int] = None
    modified: Optional[str] = None
    loaded: bool = False

class ModelsResponse(BaseModel):
    models: List[ModelInfo]
    count: int

class PullModelRequest(BaseModel):
    name: str

class PullModelResponse(BaseModel):
    success: bool
    message: str

@router.get("/", response_model=ModelsResponse, dependencies=[Depends(jwt_required(scopes=["models:read"]))])
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    List available models
    """
    try:
        models = await model_manager.list_models()
        
        model_list = []
        for model in models:
            model_list.append(ModelInfo(
                name=model.get("name", ""),
                size=model.get("size", 0),
                modified=model.get("modified_at", ""),
                loaded=True
            ))
        
        return ModelsResponse(
            models=model_list,
            count=len(model_list)
        )
        
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pull", response_model=PullModelResponse, dependencies=[Depends(jwt_required(scopes=["models:write"]))])
async def pull_model(
    request: PullModelRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Pull a new model from Ollama registry
    """
    try:
        logger.info(f"Pulling model: {request.name}")
        
        success = await model_manager.pull_model(request.name)
        
        return PullModelResponse(
            success=success,
            message=f"Model {request.name} pulled successfully" if success else f"Failed to pull model {request.name}"
        )
        
    except Exception as e:
        logger.error(f"Error pulling model: {e}")
        raise HTTPException(status_code=500, detail=str(e))