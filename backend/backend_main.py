#!/usr/bin/env python3.11
"""Backend Main Module

This is the entry point for the SutazAI backend application.
It sets up the FastAPI app, includes routers, and defines middleware and 
exception handlers.
"""

# Standard Library Imports
import asyncio
import logging
from collections.abc import Awaitable
from contextlib import asynccontextmanager
from typing import Any, Callable, Dict

# Third-Party Library Imports
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

# Local Imports
from backend.config import Config
from backend.routers.core import core_router
from backend.routers.health import health_router
from backend.utils import clear_cache

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Load configuration
config = Config()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application.
    
    Handles startup and shutdown events.
    """
    # Startup
    logger.info("Initializing backend...")
    await initialize_backend()
    logger.info("Backend initialized successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down backend...")
    clear_cache()
    logger.info("Backend shut down successfully")


# Create FastAPI app
app = FastAPI(
    title="SutazAI Backend",
    description="Backend services for the SutazAI application",
    version="0.1.0",
    docs_url="/docs" if config.debug else None,
    redoc_url="/redoc" if config.debug else None,
    lifespan=lifespan,
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
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(core_router, prefix="/api", tags=["core"])
app.include_router(health_router, prefix="/api", tags=["health"])


@app.middleware("http")
async def log_requests(
    request: Request,
    call_next: Callable[[Request], Awaitable[Any]],
) -> Any:
    """Log all incoming requests for monitoring and debugging."""
    logger.info(f"Request: {request.method} {request.url}")
    try:
        response = await call_next(request)
        logger.info(f"Response status: {response.status_code}")
        return response
    except Exception as error:
        logger.exception(f"Unhandled exception in middleware: {error}")
        raise error


@app.get("/health")
async def health_check() -> Dict[str, str]:
    """Basic health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.exception_handler(Exception)
async def global_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    """Global exception handler for unhandled exceptions."""
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal Server Error",
            "detail": str(exc),
        },
    )


async def initialize_backend() -> None:
    """Initialize the backend application."""
    logger.info("Initializing backend...")
    logger.info("Backend initialized successfully")


if __name__ == "__main__":
    asyncio.run(initialize_backend())
    uvicorn.run(
        "backend.backend_main:app",
        host=str(config.host),
        port=config.port,
        reload=config.debug,
        workers=config.server.workers,
        log_level="debug" if config.debug else "info",
    )
