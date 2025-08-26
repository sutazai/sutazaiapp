"""
Feature flags API endpoints
"""
from fastapi import APIRouter, Depends
from typing import Dict, Any
from app.core.config import get_settings, Settings

router = APIRouter(tags=["features"])

@router.get("/", response_model=Dict[str, Any])
async def get_feature_flags(settings: Settings = Depends(get_settings)):
    """
    Get current feature flag states
    
    Returns:
        Dict containing feature flag states and related configuration
    """
    return {
        "fsdp": {
            "enabled": settings.ENABLE_FSDP,
            "description": "Fully Sharded Data Parallel training support"
        },
        "tabby": {
            "enabled": settings.ENABLE_TABBY,
            "url": settings.TABBY_URL if settings.ENABLE_TABBY else None,
            "description": "TabbyML code completion service"
        },
        "gpu": {
            "enabled": settings.ENABLE_GPU,
            "description": "GPU acceleration support"
        },
        "monitoring": {
            "enabled": settings.ENABLE_MONITORING,
            "description": "System monitoring and metrics collection"
        }
    }