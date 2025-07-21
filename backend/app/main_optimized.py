"""
SutazAI Backend - Optimized FastAPI Application with Enterprise Performance
"""

import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict
import asyncio

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import JSONResponse, ORJSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
import uvicorn

# Performance optimizations
from app.core.middleware import (
    PerformanceMiddleware,
    RateLimitMiddleware,
    CacheMiddleware,
    CompressionMiddleware,
    SecurityHeadersMiddleware,
    ErrorHandlingMiddleware,
    MetricsMiddleware
)
from app.core.performance import performance_optimizer, load_balancer
from app.core.config import settings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import routers
try:
    from app.api.v1 import models, vectors, agents, brain, self_improvement
    routers_loaded = True
except Exception as e:
    logger.error(f"Error loading routers: {e}")
    routers_loaded = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifecycle with performance optimizations"""
    logger.info("🚀 Starting SutazAI AGI/ASI Backend with Performance Optimizations...")
    
    # Initialize connection pools
    try:
        # Database connection pool
        from app.core.database import init_db_pool
        await init_db_pool()
        
        # Redis connection pool (already in CacheManager)
        await performance_optimizer.cache_manager.get_redis()
        
        # Initialize services with optimizations
        from app.services.model_manager import model_manager
        from app.services.agent_orchestrator import agent_orchestrator
        from app.services.self_improvement import self_improvement_service
        
        # Pre-warm caches
        logger.info("Pre-warming caches...")
        await model_manager.list_models()  # Cache model list
        
        # Configure load balancer for multiple backend instances
        if settings.get("ENABLE_LOAD_BALANCING", False):
            for backend in settings.get("BACKEND_SERVERS", []):
                load_balancer.add_backend(backend["url"], backend.get("weight", 1))
        
        # Start background tasks
        background_tasks = []
        
        # Performance monitoring
        async def monitor_performance():
            while True:
                metrics = performance_optimizer.get_performance_metrics()
                logger.info(f"Performance metrics: {metrics}")
                await asyncio.sleep(60)  # Log every minute
        
        background_tasks.append(asyncio.create_task(monitor_performance()))
        
        # Health checks for load balancer
        if load_balancer.backends:
            async def health_check_loop():
                while True:
                    for backend in load_balancer.backends:
                        await load_balancer.health_check(backend)
                    await asyncio.sleep(30)  # Check every 30 seconds
            
            background_tasks.append(asyncio.create_task(health_check_loop()))
        
        logger.info("✅ All services initialized with performance optimizations")
        
    except Exception as e:
        logger.error(f"Startup error: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down SutazAI Backend...")
    
    # Cancel background tasks
    for task in background_tasks:
        task.cancel()
    
    # Close connections
    if performance_optimizer.cache_manager._redis_client:
        await performance_optimizer.cache_manager._redis_client.close()
    
    logger.info("Shutdown complete")


# Create optimized FastAPI app
app = FastAPI(
    title="SutazAI AGI/ASI System - Optimized",
    description="Enterprise-grade autonomous AI system with performance optimizations",
    version="9.0-optimized",
    lifespan=lifespan,
    default_response_class=ORJSONResponse,  # Faster JSON serialization
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware in correct order (bottom to top execution)
app.add_middleware(ErrorHandlingMiddleware)
app.add_middleware(PerformanceMiddleware)
app.add_middleware(MetricsMiddleware)
app.add_middleware(SecurityHeadersMiddleware)
app.add_middleware(CompressionMiddleware, minimum_size=500)
app.add_middleware(CacheMiddleware)
app.add_middleware(RateLimitMiddleware)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get("CORS_ORIGINS", ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=86400  # Cache preflight for 24 hours
)

# Trusted hosts
if settings.get("TRUSTED_HOSTS"):
    app.add_middleware(
        TrustedHostMiddleware,
        allowed_hosts=settings.get("TRUSTED_HOSTS", ["*"])
    )

# Prometheus metrics
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app, include_in_schema=False)

# Root endpoint
@app.get("/", response_class=JSONResponse)
async def root():
    """Root endpoint with system info"""
    return {
        "name": "SutazAI AGI/ASI System",
        "version": "9.0-optimized",
        "status": "operational",
        "features": [
            "Enterprise Performance Optimization",
            "Multi-level Caching",
            "Connection Pooling",
            "Rate Limiting",
            "Load Balancing",
            "Compression",
            "Security Headers",
            "Prometheus Metrics"
        ]
    }

# Health check with detailed status
@app.get("/health")
async def health_check():
    """Enhanced health check"""
    health_status = {
        "status": "healthy",
        "timestamp": asyncio.get_event_loop().time(),
        "services": {}
    }
    
    # Check database
    try:
        from app.core.database import check_db_health
        db_healthy = await check_db_health()
        health_status["services"]["database"] = "healthy" if db_healthy else "unhealthy"
    except Exception as e:
        health_status["services"]["database"] = f"error: {str(e)}"
    
    # Check Redis
    try:
        redis_client = await performance_optimizer.cache_manager.get_redis()
        await redis_client.ping()
        health_status["services"]["redis"] = "healthy"
    except Exception as e:
        health_status["services"]["redis"] = f"error: {str(e)}"
    
    # Check vector databases
    try:
        from app.services.vector_db_manager import vector_db_manager
        vector_health = await vector_db_manager.health_check()
        health_status["services"]["vector_db"] = vector_health
    except Exception as e:
        health_status["services"]["vector_db"] = f"error: {str(e)}"
    
    # Performance metrics
    health_status["performance"] = performance_optimizer.get_performance_metrics()
    
    # Overall status
    all_healthy = all(
        v == "healthy" or isinstance(v, dict) 
        for v in health_status["services"].values()
    )
    health_status["status"] = "healthy" if all_healthy else "degraded"
    
    return health_status

# Performance metrics endpoint
@app.get("/api/v1/metrics")
async def get_metrics():
    """Get detailed performance metrics"""
    return {
        "performance": performance_optimizer.get_performance_metrics(),
        "cache_stats": performance_optimizer.cache_manager.get_stats(),
        "memory": await performance_optimizer.profile_memory(),
        "request_timings": dict(performance_optimizer._request_timings)
    }

# Cache management endpoints
@app.post("/api/v1/cache/clear")
async def clear_cache(pattern: str = "*"):
    """Clear cache by pattern"""
    count = await performance_optimizer.cache_manager.clear_pattern(pattern)
    return {"cleared": count, "pattern": pattern}

@app.get("/api/v1/cache/stats")
async def cache_stats():
    """Get cache statistics"""
    return performance_optimizer.cache_manager.get_stats()

# Include API routers with caching
if routers_loaded:
    # Models API with caching
    app.include_router(
        models.router,
        prefix="/api/v1/models",
        tags=["models"]
    )
    
    # Vectors API
    app.include_router(
        vectors.router,
        prefix="/api/v1/vectors",
        tags=["vectors"]
    )
    
    # Agents API
    app.include_router(
        agents.router,
        prefix="/api/v1/agents",
        tags=["agents"]
    )
    
    # Brain API
    app.include_router(
        brain.router,
        prefix="/api/v1/brain",
        tags=["brain"]
    )
    
    # Self-improvement API
    app.include_router(
        self_improvement.router,
        prefix="/api/v1/self-improvement",
        tags=["self-improvement"]
    )

# WebSocket endpoint for real-time updates
@app.websocket("/ws")
async def websocket_endpoint(websocket):
    """WebSocket for real-time updates"""
    await websocket.accept()
    try:
        while True:
            # Send performance metrics every 5 seconds
            metrics = performance_optimizer.get_performance_metrics()
            await websocket.send_json({
                "type": "metrics",
                "data": metrics,
                "timestamp": asyncio.get_event_loop().time()
            })
            await asyncio.sleep(5)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    """Custom 404 handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Not found",
            "path": str(request.url.path),
            "message": "The requested resource was not found"
        }
    )

@app.exception_handler(500)
async def internal_error_handler(request: Request, exc):
    """Custom 500 handler"""
    logger.error(f"Internal error: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": "An unexpected error occurred",
            "request_id": getattr(request.state, "request_id", "unknown")
        }
    )

if __name__ == "__main__":
    # Run with optimized settings
    uvicorn.run(
        "app.main_optimized:app",
        host="0.0.0.0",
        port=8000,
        workers=4,  # Multiple workers for better performance
        loop="uvloop",  # Faster event loop
        access_log=False,  # Disable access logs for performance
        log_level="info"
    )