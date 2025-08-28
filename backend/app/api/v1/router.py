"""
API v1 Router
Aggregates all endpoint routers
"""

from fastapi import APIRouter
from app.api.v1.endpoints import health, agents, vectors, chat, auth

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(vectors.router, prefix="/vectors", tags=["vectors"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])