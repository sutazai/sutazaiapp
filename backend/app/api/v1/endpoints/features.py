"""
Features API endpoints for feature flag status
"""
from fastapi import APIRouter, Depends
from pydantic import BaseModel
from typing import Dict
from backend.app.core.config import get_settings, Settings

router = APIRouter()


class FeaturesResponse(BaseModel):
    """Response model for feature flags status"""
    features: Dict[str, bool]
    metadata: Dict[str, str]


@router.get("/", response_model=FeaturesResponse)
async def get_features(settings: Settings = Depends(get_settings)) -> FeaturesResponse:
    """
    Get current feature flags status
    
    Returns:
        FeaturesResponse with all feature flags and their states
    """
    return FeaturesResponse(
        features={
            "fsdp": settings.ENABLE_FSDP,
            "tabby": settings.ENABLE_TABBY,
            "gpu": settings.ENABLE_GPU,
            "monitoring": settings.ENABLE_MONITORING,
            "logging": settings.ENABLE_LOGGING,
            "health_checks": settings.ENABLE_HEALTH_CHECKS,
        },
        metadata={
            "tabby_url": settings.TABBY_URL if settings.ENABLE_TABBY else "",
            "environment": settings.SUTAZAI_ENV,
        }
    )