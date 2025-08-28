"""
Health check endpoints for service monitoring
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.services.connections import service_connections
import logging

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/")
async def health_status() -> Dict[str, Any]:
    """Get overall system health status"""
    try:
        service_health = await service_connections.health_check()
        all_healthy = all(service_health.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "services": service_health,
            "healthy_count": sum(service_health.values()),
            "total_services": len(service_health)
        }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/services/{service_name}")
async def service_health(service_name: str) -> Dict[str, Any]:
    """Check health of a specific service"""
    valid_services = [
        "redis", "rabbitmq", "neo4j", "chromadb", 
        "qdrant", "faiss", "consul", "kong", "ollama"
    ]
    
    if service_name not in valid_services:
        raise HTTPException(
            status_code=404, 
            detail=f"Service '{service_name}' not found. Valid services: {valid_services}"
        )
    
    try:
        service_health = await service_connections.health_check()
        is_healthy = service_health.get(service_name, False)
        
        return {
            "service": service_name,
            "status": "healthy" if is_healthy else "unhealthy",
            "healthy": is_healthy
        }
    except Exception as e:
        logger.error(f"Service health check error for {service_name}: {e}")
        raise HTTPException(status_code=503, detail=str(e))