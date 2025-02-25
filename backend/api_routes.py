#!/usr/bin/env python3
"""Minimal API routes for the SutazAI backend."""

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

router = APIRouter(tags=["api"])


class ApiStatus(BaseModel):
    """
    Pydantic model representing the API status.

    Attributes:
        status: The current API status string
        version: The API version string
    """

    status: str = Field(..., description="Current API status")
    version: str = Field(default="0.1.0", description="API version")

    model_config = {
        "json_schema_extra": {
            "example": {"status": "running", "version": "0.1.0"}
        },
    }


def get_api_version() -> str:
    """
    Get the current API version.

    Returns:
        str: The current API version string
    """
    # In a real app, this could come from a config file or package metadata
    return "0.1.0"


@router.get("/status", response_model=ApiStatus)
def get_status(version: str = Depends(get_api_version)) -> ApiStatus:
    """
    Get the current status of the API.

    Args:
        version: The API version from dependency

    Returns:
        ApiStatus: A model containing the API status information.
    """
    return ApiStatus(status="running", version=version)


@router.get("/info")
def get_info() -> dict[str, object]:
    """
    Get general information about the API.

    Returns:
        dict: Dictionary with API information
    """
    return {
        "name": "SutazAI Backend API",
        "description": "Backend API for SutazAI autonomous development platform",
        "version": get_api_version(),
        "endpoints": [
            {
                "path": "/status",
                "method": "GET",
                "description": "Get API status",
            },
            {
                "path": "/info",
                "method": "GET",
                "description": "Get API information",
            },
        ],
    }


# Additional API routes can be defined here as needed.
