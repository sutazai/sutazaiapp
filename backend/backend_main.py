#!/usr/bin/env python3.11
"""
SutazAI Backend Main Application

This module serves as the entry point for the SutazAI backend application.
It sets up the FastAPI app, includes routers, and defines middleware and exception handlers.
"""

# Standard Library Imports
import logging
from collections.abc import Awaitable
from typing import Any, Callable, Dict

# Third-Party Library Imports
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# Local Imports
from backend.config import Config
from backend.routers.core import core_router
from backend.routers.health import health_router

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load configuration
config = Config()

# Create FastAPI app
app = FastAPI(
    title="SutazAI Backend",
    description="Backend services for the SutazAI application",
    version="0.1.0",
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None,
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
async def log_requests(request: Request, call_next: Callable[[Request], Awaitable[Any]]) -> Any:
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
    except Exception as error:
        logger.exception(f"Unhandled exception in middleware: {error}")
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
            "detail": str(exc),
        },
    )


async def initialize_backend() -> None:
    """
    Initialize the backend application.

    This function is called when the application starts up.
    It can be used to perform any necessary initialization tasks.
    """
    logger.info("Initializing backend...")
    # Add initialization logic here
    logger.info("Backend initialized successfully")


@app.on_event("startup")
async def startup_event() -> None:
    """
    FastAPI startup event handler.

    This function is called when the application starts up.
    It triggers the backend initialization process.
    """
    await initialize_backend()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """
    FastAPI shutdown event handler.

    This function is called when the application shuts down.
    It can be used to perform any necessary cleanup tasks.
    """
    logger.info("Shutting down backend...")
    # Add shutdown logic here
    logger.info("Backend shut down successfully")


if __name__ == "__main__":
    # Use 127.0.0.1 instead of 0.0.0.0 for development to avoid exposing
    # In production, this should be controlled by environment variables
    asyncio.run(initialize_backend())
    uvicorn.run(
        "backend.backend_main:app",
        host="127.0.0.1",  # Localhost only, more secure than 0.0.0.0
        port=8000,
        reload=True,
        log_level="info",
        workers=1,  # Single worker for development
    )
