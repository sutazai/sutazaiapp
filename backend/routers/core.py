#!/usr/bin/env python3
"""
Core Router Module for SutazAI Backend

Provides core routing and system status endpoints with
comprehensive type annotations and error handling.
"""

from __future__ import annotations

import logging
from functools import lru_cache

# Standard Library Imports
from typing import Any, Callable, Dict, Generic, Type, TypeVar

# Third-Party Library Imports
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi_cache.decorator import cache
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
    ..., description="Overall system status", min_length=1, max_length=50,
    )

    model_config = {
    "json_schema_extra": {"example": {"status": "ok"}},
    "extra": "forbid",  # Prevent additional fields
    }

    def model_dump(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Return the dictionary representation of the model."""
        if hasattr(self, "dict"):
            return self.dict(*args, **kwargs)
        return super().model_dump(*args, **kwargs)

    @property
    def is_ok(self) -> bool:
        """Check if the status indicates an OK state."""
        return self.status.lower() == "ok"

    @property
    def status_level(self) -> str:
        """Get the severity level of the current status."""
        status_lower = self.status.lower()
        if status_lower == "ok":
            return "ok"
        if status_lower in ("warn", "warning"):
            return "warning"
        if status_lower in ("err", "error"):
            return "error"
        return "critical"


    @lru_cache(maxsize=128)
    def validate_system_status(status_data: Dict[str, Any]) -> SystemStatus:
        """
        Validates the input dictionary against the SystemStatus model.
        Uses LRU cache to avoid repeated validation of identical data.
        """
        try:
            return SystemStatus(**status_data)
        except ValidationError:
            raise
            except Exception as e:
                raise ValidationError(str(e), model=SystemStatus)


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
                @cache(expire=30)  # Cache status for 30 seconds
                async def get_status(_: Request) -> Dict[str, Any]:
                """Get the current system status."""
                try:
                    # Get status data (replace with actual implementation)
                    raw_status = {"status": "ok"}

                    # Validate and cache the status
                    validated_status = validate_system_status(raw_status)
                    return validated_status.model_dump()

                except ValidationError as e:
                    raise HTTPException(
                    status_code=HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=str(e),
                    )
                    except Exception as e:
                        logger.exception("Unexpected error: %s", e)
                        raise HTTPException(
                        status_code=HTTP_500_INTERNAL_SERVER_ERROR,
                        detail="An unexpected error occurred",
                        )


                        # -----------------------------------------------------------------------------
                        # Exception Handlers
                        # -----------------------------------------------------------------------------
                        @core_router.exception_handler(HTTPException)
                        async def custom_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
                        """Returns a custom JSON response for HTTP exceptions."""
                        return JSONResponse(
                    status_code=exc.status_code,
                    content={
                    "error": exc.detail,
                    "status_code": exc.status_code,
                    },
                    )


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
                            router: FastAPI router instance
                            model_type: Pydantic model type
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
                                    exception_class: Exception class to handle
                                        handler: Exception handler function
                                        """
                                        self.router.add_exception_handler(exception_class, handler)

                                        def get_router(self) -> APIRouter:
                                            """Get the FastAPI router instance."""
                                            return self.router

                                        def get_model_type(self) -> Type[T]:
                                            """Get the Pydantic model type."""
                                            return self.model_type
