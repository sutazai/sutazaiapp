"""
Model Router

This module provides API routes for AI model management and inference.
"""

import logging
from fastapi import APIRouter, HTTPException
from typing import List
from pathlib import Path
from fastapi import Depends

from backend.core.config import get_settings
from backend.schemas import DownloadModelRequest, ModelResponse, ModelStatusResponse, InferenceRequest, InferenceResponse, ModelDownloadResponse
from backend.ai_agents.model_manager import ModelManager
from backend.dependencies import get_model_manager

# Set up logging
logger = logging.getLogger("model_router")

# Create router
router = APIRouter()

# Get application settings
settings = get_settings()

# Define model directory using settings and pathlib
MODEL_DIR = Path(settings.DATA_DIR) / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


@router.get("/", summary="List available models", response_model=List[ModelResponse])
async def list_models(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    List all available AI models in the system.
    """
    # Get models from ModelManager
    try:
        # list_models is sync, no await needed
        models_data = model_manager.list_models()
        # Convert list of dicts to list of ModelResponse Pydantic models
        response_models = []
        for model_data in models_data:
            try:
                response_models.append(ModelResponse(**model_data))
            except Exception as pydantic_err:
                # Log error for specific model but continue processing others
                logger.error(f"Error validating model data for '{model_data.get('id', 'unknown')}' against ModelResponse schema: {pydantic_err}")
                # Optionally skip this model or add a placeholder with error info

        return response_models
    except Exception as e:
        logger.error(f"Error listing models: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.post("/inference", summary="Run inference with a specified model", response_model=InferenceResponse)
async def run_inference(
    request: InferenceRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Run inference on an AI model.
    """
    try:
        # Call ModelManager's inference method (now async)
        # model_manager.run_inference now returns a dict matching InferenceResponse
        result = await model_manager.run_inference(
            model_id=request.model_id,
            prompt=request.prompt,
            max_tokens=request.max_tokens,
            temperature=request.temperature
        )
        # Check if the manager returned an error internally
        if result.get("error"):
            logger.error(f"Inference failed internally for model '{request.model_id}': {result['error']}")
            raise HTTPException(status_code=500, detail=result['error'])

        # The manager should return a dict matching the schema
        return InferenceResponse(**result) # Validate and return
    except Exception as e:
        logger.error(f"Error during model inference: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")


@router.get("/status", summary="Check model service status", response_model=ModelStatusResponse)
async def model_status(
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Check the status of the model service.
    """
    try:
        # Call ModelManager's get_status method
        status_dict = await model_manager.get_status()

        # Convert the dictionary to the Pydantic model
        # Handle potential missing fields if necessary, though get_status should be reliable
        try:
            return ModelStatusResponse(**status_dict)
        except Exception as pydantic_err:
            logger.error(f"Error validating status data against ModelStatusResponse schema: {pydantic_err}. Data: {status_dict}", exc_info=True)
            raise HTTPException(status_code=500, detail="Failed to format model status response")

    except Exception as e:
        logger.error(f"Error getting model status: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Failed to get model status")


@router.post("/download", summary="Download a model", response_model=ModelDownloadResponse)
async def download_model(
    request: DownloadModelRequest,
    model_manager: ModelManager = Depends(get_model_manager)
):
    """
    Download a model for local use.
    """
    model_id = request.model_id
    force = request.force

    try:
        # Call ModelManager's async download method, which returns a dict
        result_dict = await model_manager.download_model(model_id=model_id, force=force)

        # Check status from the returned dictionary
        if result_dict.get("status") == "exists":
            logger.info(f"Model download skipped, already exists: {model_id}")
            raise HTTPException(status_code=409, detail=result_dict.get("message", "Model already exists"))
        elif result_dict.get("status") == "error":
            logger.error(f"Model download failed for {model_id}: {result_dict.get('message')}")
            raise HTTPException(status_code=500, detail=result_dict.get("message", "Download failed"))

        # If status is success, convert to response model
        return ModelDownloadResponse(**result_dict)

    except Exception as e:
        # Catch any other unexpected errors
        logger.error(f"Error downloading model {model_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to download model: {e}")
