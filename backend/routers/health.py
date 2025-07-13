#!/usr/bin/env python3.11
"""Health Router Module

This module provides health check endpoints for the SutazAI backend.
"""

from typing import Dict

from fastapi import APIRouter
from loguru import logger

from backend.utils import cache

# Create router
health_router = APIRouter()


@health_router.get("/health")
@cache(expire=30)  # Cache health check for 30 seconds
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint.

    Returns:
        Dict containing health status and version
    """
    return {
        "status": "healthy",
        "version": "0.1.0",
    }


@health_router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Dict[str, str]]:
    """Detailed health check endpoint.

    Returns:
        Dict containing detailed health status for each component
    """
    try:
        # Check database connection
        db_status = "healthy"  # Replace with actual DB check

        # Check file system
        fs_status = "healthy"  # Replace with actual FS check

        return {
            "status": {
                "overall": "healthy",
                "database": db_status,
                "filesystem": fs_status,
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": {
                "overall": "unhealthy",
                "error": str(e),
            }
        }
