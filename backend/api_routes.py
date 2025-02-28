#!/usr/bin/env python3
"""
API Routes Module for SutazAI Backend

Provides centralized routing configuration and management.
"""

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
        "json_schema_extra": {"example": {"status": "running", "version": "0.1.0"}},
    }

    def is_running(self) -> bool:
        """
        Check if the API is in running state.

        Returns:
            bool: True if the API is running, False otherwise
        """
        return self.status.lower() == "running"

    def get_version_info(self) -> dict[str, str]:
        """
        Get detailed version information.

        Returns:
            dict[str, str]: Dictionary containing version details
        """
        major, minor, patch = self.version.split(".")
        return {
            "version": self.version,
            "major": major,
            "minor": minor,
            "patch": patch,
        }

    def get_status_description(self) -> str:
        """
        Get a human-readable description of the API status.

        Returns:
            str: A description of the current API status
        """
        status_map = {
            "running": "The API is operational and accepting requests",
            "starting": "The API is currently starting up",
            "stopping": "The API is shutting down",
            "maintenance": "The API is undergoing maintenance",
        }
        return status_map.get(self.status.lower(), "Unknown API status")


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


class APIRouteConfig(BaseModel):
    """
    Configuration class for API routes.
    """

    enabled_routes: dict[str, bool] = Field(
        default_factory=lambda: {"status": True, "info": True},
        description="Dictionary of route names and their enabled status"
    )
    route_prefix: str = Field(
        default="/api/v1",
        description="Default route prefix for the API"
    )

    def get_route_prefix(self) -> str:
        """
        Get the default route prefix for the API.

        Returns:
            str: The default route prefix.
        """
        return self.route_prefix

    def is_route_enabled(self, route_name: str) -> bool:
        """
        Check if a specific route is enabled.

        Args:
            route_name: Name of the route to check.

        Returns:
            bool: True if the route is enabled, False otherwise.
        """
        return self.enabled_routes.get(route_name, True)

    def enable_route(self, route_name: str) -> None:
        """
        Enable a specific route.

        Args:
            route_name: Name of the route to enable.
        """
        self.enabled_routes[route_name] = True

    def disable_route(self, route_name: str) -> None:
        """
        Disable a specific route.

        Args:
            route_name: Name of the route to disable.
        """
        self.enabled_routes[route_name] = False


# Create an instance of the route configuration
route_config = APIRouteConfig()
