#!/usr/bin/env python3
"""
SutazAI API Gateway - Centralized API Management and Routing
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json

import aioredis
from fastapi import FastAPI, HTTPException, Request, Response, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import httpx
import uvicorn
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Rate limiting
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="SutazAI API Gateway", version="1.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Security
security = HTTPBearer()

class ServiceRoute(BaseModel):
    service_name: str
    url: str
    health_check: str
    timeout: int = 30
    retry_count: int = 3

class APIGateway:
    def __init__(self):
        self.redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.redis = None
        self.services = self._initialize_services()
        self.health_status = {}
        
    def _initialize_services(self) -> Dict[str, ServiceRoute]:
        """Initialize service routing table"""
        return {
            "backend": ServiceRoute(
                service_name="backend-agi",
                url="http://backend-agi:8000",
                health_check="/health"
            ),
            "frontend": ServiceRoute(
                service_name="frontend-agi", 
                url="http://frontend-agi:8501",
                health_check="/"
            ),
            "ollama": ServiceRoute(
                service_name="ollama",
                url="http://ollama:11434",
                health_check="/api/tags"
            ),
            "chromadb": ServiceRoute(
                service_name="chromadb",
                url="http://chromadb:8000",
                health_check="/api/v1/heartbeat"
            ),
            "qdrant": ServiceRoute(
                service_name="qdrant",
                url="http://qdrant:6333",
                health_check="/health"
            ),
            "jarvis": ServiceRoute(
                service_name="jarvis-ai",
                url="http://jarvis-ai:8120",
                health_check="/health"
            ),
            "scheduler": ServiceRoute(
                service_name="task-scheduler",
                url="http://task-scheduler:8080",
                health_check="/health"
            ),
            "optimizer": ServiceRoute(
                service_name="model-optimizer",
                url="http://model-optimizer:8080",
                health_check="/health"
            ),
            "prometheus": ServiceRoute(
                service_name="prometheus",
                url="http://prometheus:9090",
                health_check="/-/healthy"
            ),
            "grafana": ServiceRoute(
                service_name="grafana",
                url="http://grafana:3000",
                health_check="/api/health"
            )
        }
    
    async def init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis = aioredis.from_url(self.redis_url)
            await self.redis.ping()
            logger.info("Redis connection established")
        except Exception as e:
            logger.error(f"Redis connection failed: {e}")
            
    async def get_service_health(self, service_name: str) -> bool:
        """Check service health"""
        if service_name not in self.services:
            return False
            
        service = self.services[service_name]
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{service.url}{service.health_check}")
                is_healthy = response.status_code == 200
                self.health_status[service_name] = {
                    "healthy": is_healthy,
                    "last_check": datetime.now().isoformat(),
                    "status_code": response.status_code
                }
                return is_healthy
        except Exception as e:
            logger.error(f"Health check failed for {service_name}: {e}")
            self.health_status[service_name] = {
                "healthy": False,
                "last_check": datetime.now().isoformat(),
                "error": str(e)
            }
            return False
    
    async def route_request(self, service_name: str, path: str, method: str, 
                          headers: dict, body: bytes = None, params: dict = None) -> Dict:
        """Route request to appropriate service"""
        if service_name not in self.services:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
            
        service = self.services[service_name]
        
        # Check if service is healthy
        if not await self.get_service_health(service_name):
            raise HTTPException(status_code=503, detail=f"Service {service_name} is unavailable")
        
        url = f"{service.url}{path}"
        
        try:
            async with httpx.AsyncClient(timeout=service.timeout) as client:
                # Prepare request
                request_kwargs = {
                    "headers": headers,
                    "params": params or {}
                }
                
                if body:
                    request_kwargs["content"] = body
                
                # Make request
                response = await client.request(method, url, **request_kwargs)
                
                # Log request
                await self._log_request(service_name, method, path, response.status_code)
                
                return {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "content": response.content,
                    "service": service_name
                }
                
        except httpx.TimeoutException:
            logger.error(f"Timeout routing to {service_name}")
            raise HTTPException(status_code=504, detail="Gateway timeout")
        except Exception as e:
            logger.error(f"Error routing to {service_name}: {e}")
            raise HTTPException(status_code=502, detail="Bad gateway")
    
    async def _log_request(self, service: str, method: str, path: str, status: int):
        """Log request to Redis for analytics"""
        if not self.redis:
            return
            
        try:
            log_entry = {
                "timestamp": datetime.now().isoformat(),
                "service": service,
                "method": method,
                "path": path,
                "status": status
            }
            
            await self.redis.lpush("gateway_logs", json.dumps(log_entry))
            await self.redis.ltrim("gateway_logs", 0, 9999)  # Keep last 10k logs
        except Exception as e:
            logger.error(f"Failed to log request: {e}")

# Initialize gateway
gateway = APIGateway()

@app.on_event("startup")
async def startup_event():
    """Initialize gateway on startup"""
    await gateway.init_redis()
    logger.info("API Gateway started successfully")

@app.get("/health")
async def health_check():
    """Gateway health check"""
    return {
        "status": "healthy",
        "service": "API Gateway",
        "timestamp": datetime.now().isoformat(),
        "services": len(gateway.services),
        "redis_connected": gateway.redis is not None
    }

@app.get("/services")
async def list_services():
    """List all registered services"""
    return {
        "services": {name: {
            "url": service.url,
            "health_check": service.health_check,
            "timeout": service.timeout
        } for name, service in gateway.services.items()}
    }

@app.get("/services/health")
async def services_health():
    """Check health of all services"""
    health_results = {}
    
    for service_name in gateway.services:
        health_results[service_name] = await gateway.get_service_health(service_name)
    
    return {
        "timestamp": datetime.now().isoformat(),
        "services": health_results,
        "detailed_status": gateway.health_status
    }

@app.get("/metrics")
async def get_metrics():
    """Get gateway metrics"""
    if not gateway.redis:
        return {"error": "Redis not available"}
    
    try:
        # Get recent logs
        logs = await gateway.redis.lrange("gateway_logs", 0, 99)
        
        # Process metrics
        total_requests = len(logs)
        services_used = set()
        status_codes = {}
        
        for log_entry in logs:
            try:
                entry = json.loads(log_entry)
                services_used.add(entry["service"])
                status = entry["status"]
                status_codes[status] = status_codes.get(status, 0) + 1
            except:
                continue
        
        return {
            "total_requests": total_requests,
            "services_used": len(services_used),
            "status_codes": status_codes,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        return {"error": str(e)}

@app.api_route("/{service_name}/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH"])
@limiter.limit("100/minute")
async def route_to_service(request: Request, service_name: str, path: str):
    """Route requests to specific services"""
    
    # Get request details
    method = request.method
    headers = dict(request.headers)
    params = dict(request.query_params)
    
    # Remove host header to avoid conflicts
    headers.pop("host", None)
    
    # Get body if present
    body = None
    if method in ["POST", "PUT", "PATCH"]:
        body = await request.body()
    
    # Route the request
    result = await gateway.route_request(
        service_name=service_name,
        path=f"/{path}",
        method=method,
        headers=headers,
        body=body,
        params=params
    )
    
    # Return response
    return Response(
        content=result["content"],
        status_code=result["status_code"],
        headers={k: v for k, v in result["headers"].items() 
                if k.lower() not in ["content-length", "content-encoding", "transfer-encoding"]}
    )

@app.get("/")
async def root():
    """Gateway root endpoint"""
    return {
        "service": "SutazAI API Gateway",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat(),
        "endpoints": {
            "health": "/health",
            "services": "/services",
            "services_health": "/services/health",
            "metrics": "/metrics",
            "route": "/{service_name}/{path}"
        },
        "available_services": list(gateway.services.keys())
    }

if __name__ == "__main__":
    port = int(os.environ.get("GATEWAY_PORT", 8080))
    host = os.environ.get("GATEWAY_HOST", "0.0.0.0")
    
    logger.info("Starting SutazAI API Gateway...")
    logger.info(f"Gateway URL: http://{host}:{port}")
    logger.info(f"Services registered: {len(gateway.services)}")
    
    uvicorn.run(app, host=host, port=port) 