"""
API Router for v1 endpoints
"""
from fastapi import APIRouter

from app.api.v1.endpoints import agents, models, documents, chat, system, hardware, cache, circuit_breaker, cache_optimized, mesh, mesh_v2, mcp, mcp_stdio, mcp_emergency
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
api_router.include_router(mesh.router, prefix="/mesh", tags=["mesh"])  # Legacy Redis-based mesh
api_router.include_router(mesh_v2.router, prefix="/mesh/v2", tags=["mesh-v2"])  # Real service mesh
api_router.include_router(mcp.router, prefix="/mcp", tags=["mcp"])  # MCP server integration
api_router.include_router(mcp_stdio.router, prefix="/mcp-stdio", tags=["mcp-stdio"])  # STDIO MCP server integration (emergency fix)
api_router.include_router(mcp_emergency.router, prefix="/mcp-fix", tags=["mcp-emergency"])  # Emergency MCP fix with proper initialization
api_router.include_router(features.router, prefix="/features", tags=["features"])