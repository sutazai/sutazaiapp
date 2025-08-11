"""
API Router for v1 endpoints
"""
from fastapi import APIRouter

from app.api.v1.endpoints import agents, models, documents, chat, system, hardware, cache, circuit_breaker, cache_optimized
from app.api.v1 import features

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(models.router, prefix="/models", tags=["models"])
api_router.include_router(documents.router, prefix="/documents", tags=["documents"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(system.router, prefix="/system", tags=["system"])
api_router.include_router(hardware.router, prefix="/hardware", tags=["hardware"])
api_router.include_router(cache.router, prefix="/cache", tags=["cache"])
api_router.include_router(cache_optimized.router, prefix="/cache-optimized", tags=["cache-optimized"])
api_router.include_router(circuit_breaker.router, prefix="/circuit-breaker", tags=["circuit-breaker"])
api_router.include_router(features.router, prefix="/features", tags=["features"])