"""
Refactored API Router for v1 endpoints with proper organization and patterns
Consolidates duplicate endpoints and implements consistent naming conventions
"""
from fastapi import APIRouter
from app.api.v1.endpoints import (
    agents, models, documents, chat, system, hardware, 
    cache_optimized, circuit_breaker, mcp, features
)

# Import consolidated mesh module (combining v1 and v2)
from app.api.v1.endpoints import mesh_v2 as mesh_consolidated

# Create main API router
api_router = APIRouter()

# Core AI Operations
api_router.include_router(
    agents.router, 
    prefix="/agents", 
    tags=["agents"],
    responses={
        404: {"description": "Agent not found"},
        500: {"description": "Internal server error"}
    }
)

api_router.include_router(
    models.router, 
    prefix="/models", 
    tags=["models"],
    responses={
        502: {"description": "Ollama service unavailable"},
        500: {"description": "Internal server error"}
    }
)

api_router.include_router(
    chat.router, 
    prefix="/chat", 
    tags=["chat"],
    responses={
        429: {"description": "Rate limit exceeded"},
        500: {"description": "Internal server error"}
    }
)

# Document Management
api_router.include_router(
    documents.router, 
    prefix="/documents", 
    tags=["documents"],
    responses={
        404: {"description": "Document not found"},
        500: {"description": "Internal server error"}
    }
)

# Service Mesh Operations (Consolidated)
api_router.include_router(
    mesh_consolidated.router, 
    prefix="/mesh", 
    tags=["mesh"],
    responses={
        503: {"description": "Service unavailable"},
        500: {"description": "Internal server error"}
    }
)

# MCP Integration (Mesh-integrated)
api_router.include_router(
    mcp.router, 
    prefix="/mcp", 
    tags=["mcp"],
    responses={
        404: {"description": "MCP service not found"},
        500: {"description": "Internal server error"}
    }
)

# System Operations
api_router.include_router(
    system.router, 
    prefix="/system", 
    tags=["system"],
    responses={
        503: {"description": "System degraded"},
        500: {"description": "Internal server error"}
    }
)

api_router.include_router(
    hardware.router, 
    prefix="/hardware", 
    tags=["hardware"],
    responses={
        500: {"description": "Internal server error"}
    }
)

# Performance & Optimization
api_router.include_router(
    cache_optimized.router, 
    prefix="/cache", 
    tags=["cache"],
    responses={
        500: {"description": "Internal server error"}
    }
)

api_router.include_router(
    circuit_breaker.router, 
    prefix="/circuit-breaker", 
    tags=["circuit-breaker"],
    responses={
        503: {"description": "Circuit breaker open"},
        500: {"description": "Internal server error"}
    }
)

# Feature Management
api_router.include_router(
    features.router, 
    prefix="/features", 
    tags=["features"],
    responses={
        404: {"description": "Feature not found"},
        500: {"description": "Internal server error"}
    }
)

# API Versioning and Deprecation Headers
def add_api_version_headers(router: APIRouter):
    """Add versioning headers to all responses"""
    @router.middleware("http")
    async def add_version_headers(request, call_next):
        response = await call_next(request)
        response.headers["X-API-Version"] = "1.0.0"
        response.headers["X-API-Deprecation"] = "false"
        
        # Add deprecation warnings for legacy endpoints
        if "/mesh/enqueue" in str(request.url) or "/mesh/results" in str(request.url):
            response.headers["X-API-Deprecation"] = "true"
            response.headers["X-API-Deprecation-Date"] = "2025-09-01"
            response.headers["X-API-Alternative"] = "/api/v1/mesh/call"
        
        return response

# Apply versioning middleware
add_api_version_headers(api_router)