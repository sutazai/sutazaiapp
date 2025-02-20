#!/usr/bin/env python3
"""
Health Router Module for the SutazAI Backend.

This module provides an endpoint to check the health status of the application.
"""

from fastapi import APIRouter
from pydantic import BaseModel

health_router: APIRouter = APIRouter()


class HealthStatus(BaseModel):
    """
    Pydantic model representing the health status.
    """

    status: str


@health_router.get("/health", response_model=HealthStatus, tags=["health"])
async def get_health() -> HealthStatus:
    """
    Health check endpoint.

    Returns:
        HealthStatus: The current health status.
    """
    return HealthStatus(status="healthy")
