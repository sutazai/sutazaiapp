"""
SutazAI Backend Main Module

This module provides the main FastAPI application instance and core middleware.
"""

import logging
from typing import Any, Awaitable, Callable, Dict

import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from backend.routers import core_router
from backend.routers.health import health_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)

app = FastAPI(
    title="SutazAI Backend",
    description="Autonomous AI Development Platform",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Define allowed origins
ALLOWED_ORIGINS = [
    "http://localhost:3000",  # Development frontend
    "http://localhost:8000",  # Development backend
    "https://sutazai.com",  # Production domain
    "https://api.sutazai.com",  # Production API domain
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],  # Can be restricted to specific methods if needed
    allow_headers=["*"],  # Can be restricted to specific headers if needed
)

# Include routers
app.include_router(core_router, prefix="/api", tags=["core"])
app.include_router(health_router, prefix="/api", tags=["health"])


@app.middleware("http")
async def log_requests(
    request: Request, call_next: Callable[[Request], Awaitable[Any]]
) -> Any:
    """
    Log all incoming requests for monitoring and debugging.

    Args:
        request: The incoming request
        call_next: The next middleware in the chain

    Returns:
        Any: The response from the next middleware
    """
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as e:
        logger.exception(f"Unhandled exception in middleware: {e}")
        raise


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """
    Basic health check endpoint.

    Returns:
        Dict[str, str]: A dictionary containing status and version information.
    """
    return {"status": "healthy", "version": "0.1.0"}


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for unhandled exceptions.

    Args:
        _: Unused request parameter
        exc: The exception that was raised

    Returns:
        JSONResponse: A JSON response containing error details
    """
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "message": str(exc),
            "type": type(exc).__name__,
        },
    )


if __name__ == "__main__":
    # Use 127.0.0.1 instead of 0.0.0.0 for development to avoid exposing
    # the server to all network interfaces (better security practice)
    # In production, this should be controlled by environment variables
    uvicorn.run(
        "backend.backend_main:app",
        host="127.0.0.1",  # Localhost only, more secure than 0.0.0.0
        port=8000,
        reload=True,
        log_level="info",
        workers=1,  # Single worker for development
    )
