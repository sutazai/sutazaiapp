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

    status: str = Field(
        ..., description="Overall system status", min_length=1, max_length=50
    )

    model_config = {
        "json_schema_extra": {"example": {"status": "ok"}},
        "extra": "forbid",  # Prevent additional fields
    }

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


def validate_system_status(status_data: Any) -> SystemStatus:
    """
    Validates the input dictionary against the SystemStatus model.

    Performs comprehensive validation with detailed error handling.

    Args:
        status_data: The raw status data.

    Returns:
        SystemStatus: A validated SystemStatus instance.

    Raises:
        HTTPException: If validation fails or input is invalid.
    """
    if not isinstance(status_data, (dict, str)):
        logger.warning("Invalid status format: %s", type(status_data))
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail="Invalid status format. Must be a dictionary or string.",
        )

    try:
        if isinstance(status_data, dict):
            return SystemStatus(**status_data)
        # Use model_validate_json for string input
        if hasattr(SystemStatus, "model_validate_json"):
            return SystemStatus.model_validate_json(status_data)
        # Fallback for older Pydantic versions
        return SystemStatus.parse_raw(status_data)
    except ValidationError as e:
        logger.error("Status validation error: %s", e)
        raise HTTPException(
            status_code=HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Validation failed: {str(e)}",
        ) from e
    except Exception as e:
        logger.exception("Unexpected error during status validation")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System status validation failed: {str(e)}",
        ) from e


# -----------------------------------------------------------------------------
# Route Definitions
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
    GET endpoint to retrieve system status.

    Simulates retrieving system status and validates the result.

    Args:
        _: Unused request parameter

    Returns:
        Dict[str, Any]: JSON serializable system status.

    Raises:
        HTTPException: If status retrieval or validation fails.
    """
    try:
        # Simulate retrieving system status
        status_data = {"status": "ok"}
        validated_status = validate_system_status(status_data)
        return validated_status.model_dump()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error in get_status")
        raise HTTPException(
            status_code=HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Unexpected error: {str(e)}",
        ) from e


async def custom_exception_handler(
    _: Request, exc: HTTPException
) -> JSONResponse:
    """
    Returns a custom JSON response for HTTP exceptions.

    Provides a standardized error response format.

    Args:
        _: Unused request parameter
        exc: The HTTP exception to handle

    Returns:
        JSONResponse: A formatted error response
    """
    logger.warning("HTTP Exception: %s - %s", exc.status_code, exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "Request processing error",
            "detail": str(exc.detail),
            "status_code": exc.status_code,
        },
    )


# Note: APIRouter does not support add_exception_handler.
# Register exception handlers on the FastAPI app instance.


def create_core_router() -> APIRouter:
    """
    Factory method to create and configure the core router.

    Returns:
        APIRouter: Configured core router with all routes and handlers.
    """
    return core_router


# Generic type handling
T = TypeVar("T", bound=BaseModel)


class RouterHandler(Generic[T]):
    """
    Generic router handler for type-safe request processing.

    Provides a flexible way to handle routing with type validation.
    """

    def __init__(self, router: APIRouter, model_type: Type[T]) -> None:
        """
        Initialize the router handler.

        Args:
            router: The FastAPI router to handle
            model_type: The Pydantic model type for validation
        """
        self.router = router
        self.model_type = model_type

    def add_exception_handler(
        self,
        exception_class: Type[Exception],
        handler: Callable[[Request, Exception], JSONResponse],
    ) -> None:
        """
        Stub method for adding exception handlers.

        APIRouter does not support setting an exception handler.

        Args:
            exception_class: The exception class to handle
            handler: The handler function
        """
        # This is a stub - APIRouter doesn't support exception handlers
        pass
