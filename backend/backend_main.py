#!/usr/bin/env python3.11
"""Backend Main Module

This is the entry point for the SutazAI backend application.
It sets up the FastAPI app, includes routers, and defines middleware and exception handlers.
"""

# Standard Library Imports
import asyncio
import logging
from collections.abc import Awaitable
from typing import Any, Callable, Dict

# Third-Party Library Imports
import redis
import uvicorn
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi_cache import FastAPICache
from fastapi_cache.backends.redis import RedisBackend
from fastapi_cache.decorator import cache
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
@cache(expire=30)  # Cache health check for 30 seconds
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
    
    # Initialize Redis cache
    redis_client = redis.from_url(
        "redis://localhost",
        encoding="utf8",
        decode_responses=True,
    )
    FastAPICache.init(RedisBackend(redis_client), prefix="fastapi-cache")
    
    logger.info("Backend initialized successfully")


@app.on_event("startup")
async def startup_event() -> None:
    """FastAPI startup event handler."""
    await initialize_backend()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    """FastAPI shutdown event handler."""
    logger.info("Shutting down backend...")
    # Close Redis connection
    await FastAPICache.clear()
    logger.info("Backend shut down successfully")


if __name__ == "__main__":
    asyncio.run(initialize_backend())
    uvicorn.run(
        "backend.backend_main:app",
        host=str(config.host),
        port=config.port,
        reload=config.debug,
        workers=config.server.workers,
        timeout_keep_alive=config.server.keepalive,
        timeout=config.server.timeout,
        limit_max_requests=config.server.limit_max_requests,
        limit_max_requests_jitter=config.server.max_requests_jitter,
        limit_concurrency=config.server.limit_concurrency,
        backlog=config.server.backlog,
        log_level="debug" if config.debug else "info",
    )
