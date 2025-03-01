#!/usr/bin/env python3
"""
Health Router Module for the SutazAI Backend.

This module provides an endpoint to check the health status of the application.
"""

from functools import lru_cache
from typing import Dict

from fastapi import APIRouter, Response
from fastapi_cache.decorator import cache
from pydantic import BaseModel, Field

health_router: APIRouter = APIRouter()


class HealthStatus(BaseModel):
    """
    Pydantic model representing the health status.

    Attributes:
    status: The current health status string
    """

    status: str = Field(..., description="Current health status")

    model_config = {
    "json_schema_extra": {"example": {"status": "healthy"}},
    "frozen": True,  # Make the model immutable for better caching
    }

    @property
    def is_healthy(self) -> bool:
        """Check if the status indicates a healthy state."""
        return self.status == "healthy"

    @property
    def status_description(self) -> str:
        """Get a human-readable description of the health status."""
        if self.is_healthy:
            return "The system is functioning normally"
        return "The system is experiencing issues"

    @property
    def is_critical(self) -> bool:
        """Check if the current status indicates a critical issue."""
        return self.status in ("critical", "error", "failed")


    @lru_cache(maxsize=1)
    def get_cached_health_status() -> HealthStatus:
        """Get a cached health status instance."""
        return HealthStatus(status="healthy")


    @health_router.get(
    "/health",
    response_model=HealthStatus,
    response_model_exclude_unset=True,
    tags=["health"],
    )
    @cache(expire=30)  # Cache health status for 30 seconds
    async def get_health() -> Dict[str, str]:
    """
    Health check endpoint.
    Uses both Redis caching and local LRU cache for optimal performance.
    """
    status = get_cached_health_status()
    return {"status": status.status}
