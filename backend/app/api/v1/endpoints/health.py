"""
Health check endpoints for service monitoring
Production-ready with comprehensive metrics
"""

from fastapi import APIRouter, HTTPException
from typing import Dict, Any
from app.services.connections import service_connections
from app.core.config import settings
from app.core.database import get_pool_status
import logging
import time

router = APIRouter()
logger = logging.getLogger(__name__)

# Track start time for uptime calculation
START_TIME = time.time()


@router.get("/")
async def health_status() -> Dict[str, Any]:
    """Get overall system health status with database pool metrics"""
    try:
        service_health = await service_connections.health_check()
        pool_status = await get_pool_status()
        all_healthy = all(service_health.values())
        uptime_seconds = int(time.time() - START_TIME)
        
        return {
            "version": settings.APP_VERSION,
            "status": "healthy" if all_healthy else "degraded",
            "uptime_seconds": uptime_seconds,
            "database_pool": pool_status,
            "services": service_health,
            "healthy_count": sum(service_health.values()),
            "total_services": len(service_health)
        }
    except Exception as e:
        logger.error(f"Health check error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/services")
async def all_services_health() -> Dict[str, Any]:
    """Get detailed health status of all services"""
    try:
        service_health = await service_connections.health_check()
        
        # Categorize services by status
        services_detail = []
        for service, is_healthy in service_health.items():
            services_detail.append({
                "name": service,
                "status": "healthy" if is_healthy else "unhealthy",
                "healthy": is_healthy
            })
        
        return {
            "total_services": len(service_health),
            "healthy_count": sum(service_health.values()),
            "unhealthy_count": len(service_health) - sum(service_health.values()),
            "status": "healthy" if all(service_health.values()) else "degraded",
            "services": services_detail
        }
    except Exception as e:
        logger.error(f"Services health check error: {e}")
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
        
        # Add more detail for specific services
        details = {}
        if service_name == "kong":
            details["admin_port"] = 10009
            details["proxy_port"] = 10008
            details["note"] = "Kong API Gateway for service routing"
        elif service_name == "redis":
            details["port"] = 10001
            details["note"] = "Cache and session storage"
        elif service_name == "postgres":
            details["port"] = 10000
            details["note"] = "Main database (not directly monitored)"
        
        return {
            "service": service_name,
            "status": "healthy" if is_healthy else "unhealthy",
            "healthy": is_healthy,
            **details
        }
    except Exception as e:
        logger.error(f"Service health check error for {service_name}: {e}")
        raise HTTPException(status_code=503, detail=str(e))


@router.get("/metrics")
async def get_metrics() -> Dict[str, Any]:
    """Get comprehensive system metrics for monitoring"""
    try:
        pool_status = await get_pool_status()
        service_health = await service_connections.health_check()
        redis_pool_stats = await service_connections.get_redis_pool_stats()
        uptime = int(time.time() - START_TIME)
        
        return {
            "timestamp": time.time(),
            "uptime_seconds": uptime,
            "database": {
                "pool_size": pool_status["size"],
                "active_connections": pool_status["checked_out"],
                "available_connections": pool_status["available"],
                "overflow_connections": pool_status["overflow"],
                "total_capacity": pool_status["total_connections"],
                "utilization_percent": (pool_status["checked_out"] / pool_status["total_connections"] * 100) if pool_status["total_connections"] > 0 else 0
            },
            "redis": {
                **redis_pool_stats,
                "utilization_percent": (redis_pool_stats.get("connections_in_use", 0) / redis_pool_stats.get("max_connections", 1) * 100) if "max_connections" in redis_pool_stats else 0
            },
            "services": {
                "total": len(service_health),
                "healthy": sum(service_health.values()),
                "unhealthy": len(service_health) - sum(service_health.values()),
                "health_percentage": (sum(service_health.values()) / len(service_health) * 100) if service_health else 0,
                "details": service_health
            },
            "application": {
                "version": settings.APP_VERSION,
                "environment": settings.ENVIRONMENT,
                "api_version": settings.API_V1_STR
            }
        }
    except Exception as e:
        logger.error(f"Metrics collection error: {e}", exc_info=True)
        raise HTTPException(status_code=503, detail=str(e))