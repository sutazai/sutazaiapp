"""
API v1 Router
Aggregates all endpoint routers including JARVIS endpoints
"""

from fastapi import APIRouter
from app.api.v1.endpoints import health, agents, vectors, chat, auth, voice, voice_demo, jarvis_websocket, jarvis_chat, simple_chat, models

api_router = APIRouter()

# Include endpoint routers
api_router.include_router(auth.router, prefix="/auth", tags=["authentication"])
api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(agents.router, prefix="/agents", tags=["agents"])
api_router.include_router(vectors.router, prefix="/vectors", tags=["vectors"])
api_router.include_router(chat.router, prefix="/chat", tags=["chat"])
api_router.include_router(voice.router, prefix="/voice", tags=["voice"])
api_router.include_router(voice_demo.router, prefix="/voice/demo", tags=["voice-demo"])
api_router.include_router(jarvis_websocket.router, prefix="/jarvis", tags=["jarvis"])
api_router.include_router(jarvis_chat.router, prefix="/jarvis", tags=["jarvis-chat"])
api_router.include_router(simple_chat.router, prefix="/simple_chat", tags=["simple-chat"])
api_router.include_router(models.router, prefix="/models", tags=["models"])