#!/usr/bin/env python3
"""
Core Router Module for SutazAI Backend

Provides core routing and system status endpoints with
comprehensive type annotations and error handling.
"""

from __future__ import annotations

# Standard Library Imports
from typing import Any, Callable, Dict, Generic, TypeAdapter, TypeVar

# Third-Party Library Imports
from fastapi import APIRouter, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

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
    """

    status: str = Field(..., description="Overall system status")

    class Config:
        # For pydantic v1 use `schema_extra` to add metadata.
        # (For pydantic v2 use `model_config` if desired.)
        schema_extra = {"example": {"status": "ok"}}

    def model_dump(self) -> Dict[str, Any]:
        """
        Return the dictionary representation of the model.
        This method mirrors the pydantic v2 call; if using pydantic v1, replace with self.dict().
        """
        return self.dict()


def validate_system_status(status: Any) -> SystemStatus:
    """
    Validates the input dictionary against the SystemStatus model.

    Args:
        status (Any): The raw status data.

    Returns:
        SystemStatus: A validated SystemStatus instance.
    """
    adapter: TypeAdapter[SystemStatus] = TypeAdapter(SystemStatus)
    validated_status = adapter.validate_python(status)
    return validated_status


# -----------------------------------------------------------------------------
# Route Definitions
# -----------------------------------------------------------------------------
@core_router.get("/status", response_model=SystemStatus)
async def get_status(request: Request) -> Dict[str, Any]:
    """
    GET endpoint to retrieve system status.

    Args:
        request (Request): Incoming HTTP request.

    Returns:
        Dict[str, Any]: JSON serializable system status.
    """
    # Simulate retrieving system status (in real code you would check databases, caches, etc.)
    status = {"status": "ok"}
    try:
        validated_status = validate_system_status(status)
    except Exception as e:
        # If validation fails, raise an HTTPException
        raise HTTPException(status_code=500, detail="System status validation failed")

    return validated_status.model_dump()


async def custom_exception_handler(request: Request, exc: HTTPException) -> Response:
    """
    Returns a custom JSON response for HTTP exceptions.
    """
    return Response(
        content=f'{{"detail": "Custom Error: {exc.detail}"}}',
        status_code=exc.status_code,
        media_type="application/json",
    )


# Register the exception handler properly
core_router.add_exception_handler(HTTPException, custom_exception_handler)


def create_core_router() -> APIRouter:
    """
    Factory method to create and configure the core router.

    Returns:
        APIRouter: Configured core router.
    """
    return core_router


# Ensure TypeAdapter is used correctly
def some_function(obj: BaseModel):
    type_adapter = TypeAdapter(type(obj))
    # Rest of the function


# Generic type handling
T = TypeVar("T", bound=BaseModel)


class RouterHandler(Generic[T]):
    def __init__(self, router: APIRouter):
        self.router = router
        self.type_adapter = TypeAdapter(T)

    def add_exception_handler(
        self, exception_class: Type[Exception], handler: Callable
    ):
        self.router.add_exception_handler(exception_class, handler)
