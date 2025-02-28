#!/usr/bin/env python3
"""
Core Router Module for SutazAI Backend

Provides core routing and system status endpoints with
comprehensive type annotations and error handling.
"""

from __future__ import annotations

# Standard Library Imports
from typing import Any, Callable, Dict, Generic, Type, TypeVar
import logging

# Third-Party Library Imports
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, ValidationError

# HTTP Status Codes
HTTP_400_BAD_REQUEST = 400
HTTP_422_UNPROCESSABLE_ENTITY = 422
HTTP_500_INTERNAL_SERVER_ERROR = 500

# Configure logging
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Initialize FastAPI Router
# -----------------------------------------------------------------------------
core_router: APIRouter = APIRouter()


# -----------------------------------------------------------------------------
# Pydantic Model Definitions and Helpers
# -----------------------------------------------------------------------------
class SystemStatus(BaseModel):
    """
    Pydantic model representing the system status.
    Provides a standardized way to represent and validate system status.
    """

    status: str = Field(..., description="Overall system status", min_length=1, max_length=50)

    class Config:
        json_schema_extra = {"example": {"status": "ok"}}
        extra = "forbid"  # Prevent additional fields

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """
        Return the dictionary representation of the model.

        Provides backward compatibility for different Pydantic versions.

        Args:
            *args: Variable length argument list
            **kwargs: Arbitrary keyword arguments

        Returns:
            Dict[str, Any]: Dictionary representation of the model
        """
        if hasattr(self, "dict"):
            return self.dict(*args, **kwargs)
        return super().model_dump(*args, **kwargs)

    def is_ok(self) -> bool:
        """
        Check if the status indicates an OK state.

        Returns:
            bool: True if the status is 'ok', False otherwise
        """
        return self.status.lower() == "ok"

    def get_status_level(self) -> str:
        """
        Get the severity level of the current status.

        Returns:
            str: The severity level (ok, warning, error, critical)
        """
        status_lower = self.status.lower()
        if status_lower == "ok":
            return "ok"
        if status_lower in ("warn", "warning"):
            return "warning"
        if status_lower in ("err", "error"):
            return "error"
        return "critical"


def validate_system_status(status_data: Any) -> SystemStatus:
    """
    Validates the input dictionary against the SystemStatus model.

    Performs comprehensive validation with detailed error handling.

    Args:
        status_data: The raw status data.

    Returns:
        SystemStatus: The validated SystemStatus instance.

    Raises:
        ValidationError: If the input data fails validation.
    """
    try:
        # Attempt to create a SystemStatus instance from the input data
        return SystemStatus(**status_data)
    except TypeError as e:
        # Catch type errors and attempt to parse as JSON
        try:
            return SystemStatus.model_validate_json(status_data)
        except ValidationError:
            raise
        except Exception:
            raise ValidationError(str(e), model=SystemStatus) from e
    except ValidationError:
        # Catch validation errors and attempt to parse as raw JSON
        try:
            return SystemStatus.parse_raw(status_data)
        except ValidationError:
            raise
        except Exception as e:
            raise ValidationError(str(e), model=SystemStatus) from e


# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------
@core_router.get(
    "/status",
    response_model=SystemStatus,
    responses={
        200: {"description": "Successful response"},
        400: {"description": "Bad request"},
        422: {"description": "Validation error"},
        500: {"description": "Internal server error"},
    },
)
async def get_status(_: Request) -> Dict[str, Any]:
    """
    Get the current system status.

    Performs comprehensive validation and error handling.

    Args:
        _: The incoming request (unused).

    Returns:
        Dict[str, Any]: The validated system status as a dictionary.

    Raises:
        HTTPException: If the status data fails validation.
    """
    try:
        # Simulate getting the raw status data (replace with actual logic)
        raw_status = {"status": "ok"}

        # Validate the status data
        validated_status = validate_system_status(raw_status)

        # Return the validated status as a dictionary
        return validated_status.model_dump()

    except ValidationError as e:
        # Handle validation errors
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(e),
        ) from e

    except Exception as e:
        # Handle all other exceptions
        logger.exception("Unexpected error: %s", e)
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred",
        ) from e


# -----------------------------------------------------------------------------
# Exception Handlers
# -----------------------------------------------------------------------------
@core_router.exception_handler(HTTPException)
async def custom_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    """
    Returns a custom JSON response for HTTP exceptions.

    Includes the exception detail in the response body.

    Args:
        _: The incoming request (unused).
        exc: The HTTPException instance.

    Returns:
        JSONResponse: The custom JSON response.
    """
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
        },
    )


# -----------------------------------------------------------------------------
# Router Creation and Configuration
# -----------------------------------------------------------------------------
def create_core_router() -> APIRouter:
    """
    Create and configure the core router.

    Returns:
        APIRouter: The configured core router.
    """
    core_router.include_router(health_router)
    return core_router


# -----------------------------------------------------------------------------
# Generic Router Handler
# -----------------------------------------------------------------------------
T = TypeVar("T", bound=BaseModel)


class RouterHandler(Generic[T]):
    """Generic router handler class for working with Pydantic models."""

    def __init__(self, router: APIRouter, model_type: Type[T]) -> None:
        """
        Initialize the router handler.

        Args:
            router: The APIRouter instance to use.
            model_type: The Pydantic model type to use.
        """
        self.router = router
        self.model_type = model_type

    def add_exception_handler(
        self,
        exception_class: Type[Exception],
        handler: Callable[[Request, Exception], JSONResponse],
    ) -> None:
        """
        Add an exception handler to the router.

        Args:
            exception_class: The exception class to handle.
            handler: The exception handler function.
        """
        self.router.add_exception_handler(exception_class, handler)

    def get_router(self) -> APIRouter:
        """
        Get the configured router.

        Returns:
            APIRouter: The configured router.
        """
        return self.router

    def get_model_type(self) -> Type[T]:
        """
        Get the Pydantic model type.

        Returns:
            Type[T]: The Pydantic model type.
        """
        return self.model_type
