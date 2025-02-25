#!/usr/bin/env python3
"""
Health Router Module for the SutazAI Backend.

This module provides an endpoint to check the health status of the application.
"""

from fastapi import APIRouter
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
    }

    def check_health(self) -> bool:
        """
        Check if the status indicates a healthy state.

        Returns:
            bool: True if healthy, False otherwise
        """
        return self.status == "healthy"


@health_router.get("/health", response_model=HealthStatus, tags=["health"])
async def get_health() -> HealthStatus:
    """
    Health check endpoint.

    Returns:
        HealthStatus: The current health status.
    """
    return HealthStatus(status="healthy")
