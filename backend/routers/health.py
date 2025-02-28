#!/usr/bin/env python3
"""
Health Router Module for the SutazAI Backend.

This module provides an endpoint to check the health status of the application.
"""

from fastapi import APIRouter, Response
from pydantic import BaseModel, Field

health_router: APIRouter = APIRouter()


class HealthStatus(BaseModel):
    """
    Pydantic model representing the health status.

    Attributes:
        status: The current health status string
    """

    status: str = Field(..., description="Current health status")

    class Config:
        json_schema_extra = {"example": {"status": "healthy"}}

    def check_health(self) -> bool:
        """
        Check if the status indicates a healthy state.

        Returns:
            bool: True if healthy, False otherwise
        """
        return self.status == "healthy"

    def get_status_description(self) -> str:
        """
        Get a human-readable description of the health status.

        Returns:
            str: A description of the current health status
        """
        if self.check_health():
            return "The system is functioning normally"
        return "The system is experiencing issues"

    def is_critical(self) -> bool:
        """
        Check if the current status indicates a critical issue.

        Returns:
            bool: True if the status indicates a critical issue, False otherwise
        """
        return self.status in ("critical", "error", "failed")


@health_router.get("/health", response_model=HealthStatus, response_model_exclude_unset=True, tags=["health"])
async def get_health() -> Response:
    """
    Health check endpoint.

    Returns:
        Response: The current health status wrapped in a Response.
    """
    return Response(content=HealthStatus(status="healthy").json())
