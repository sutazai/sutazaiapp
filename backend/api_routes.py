#!/usr/bin/env python3.11
"""API Routes Module

Provides centralized routing configuration and management.
"""

from typing import Any, Dict, Optional, List

from fastapi import APIRouter
from pydantic import BaseModel, Field, validator

# API version
API_VERSION = "0.1.0"


def get_api_version() -> str:
    """Get current API version.

    Returns:
        Current API version string
    """
    return API_VERSION


class APIRouteConfig(BaseModel):
    """API Route Configuration Model.

    Used to configure and validate API route settings.
    """
    prefix: str = Field(
        default="/api",
        description="API route prefix",
    )
    tags: list[str] = Field(
        default_factory=list,
        description="List of tags for API documentation",
    )
    enable_docs: bool = Field(
        default=True,
        description="Whether to enable API documentation for these routes",
    )
    rate_limit: int = Field(
        default=100,
        description="Rate limit for API calls per minute",
        ge=1,
        le=1000,
    )

    @validator("prefix")
    def prefix_must_start_with_slash(cls, v):
        """Validate that prefix starts with a slash."""
        if not v.startswith("/"):
            raise ValueError("prefix must start with '/'")
        return v


def create_api_router(
    prefix: str = "/api",
    tags: Optional[List[str]] = None,
    enable_docs: bool = True,
) -> APIRouter:
    """Create a configured API router.

    Args:
        prefix: API route prefix
        tags: List of tags for API documentation
        enable_docs: Whether to enable API documentation

    Returns:
        Configured APIRouter instance
    """
    tags = tags or ["api"]

    router = APIRouter(
        prefix=prefix,
        tags=tags,
    )

    # Add basic informational routes
    @router.get("/status")
    async def get_status() -> Dict[str, Any]:
        """Get current API status."""
        return {
            "status": "operational",
            "version": get_api_version(),
        }

    @router.get("/info")
    async def get_info() -> Dict[str, Any]:
        """Get API information and available endpoints."""
        return {
            "name": "SutazAI Backend API",
            "description": "Backend API for SutazAI autonomous development platform",
            "version": get_api_version(),
            "endpoints": [
                {"path": "/status", "method": "GET", "description": "Get API status"},
                {"path": "/info", "method": "GET",
                 "description": "Get API information"},
            ],
        }

    return router


def register_api_routes(app, routes_config=None):
    """Register API routes with the FastAPI application.

    Args:
        app: FastAPI application instance
        routes_config: Optional route configuration
    """
    # Default configuration if none provided
    if routes_config is None:
        routes_config = {
            "core": {
                "prefix": "/api/core",
                "tags": ["core"],
            },
            "users": {
                "prefix": "/api/users",
                "tags": ["users"],
            },
            "documents": {
                "prefix": "/api/documents",
                "tags": ["documents"],
            },
        }

    # Register each route group
    for route_group, config in routes_config.items():
        router = create_api_router(
            prefix=config.get("prefix", f"/api/{route_group}"),
            tags=config.get("tags", [route_group]),
            enable_docs=config.get("enable_docs", True),
        )

        app.include_router(router)
