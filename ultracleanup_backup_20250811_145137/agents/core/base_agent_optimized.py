"""
SutazAI - Optimized Base Agent Class
ULTRA ORGANIZATION - Eliminates code duplication across 100+ agents
Version: 2.0 - Production Ready
"""

import asyncio
import json
import logging
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from contextlib import asynccontextmanager

import aioredis
import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
import structlog

# Metrics
REQUEST_COUNT = Counter('agent_requests_total', 'Total agent requests', ['agent_type', 'endpoint'])
REQUEST_DURATION = Histogram('agent_request_duration_seconds', 'Request duration', ['agent_type'])
HEALTH_CHECK_COUNT = Counter('agent_health_checks_total', 'Health check count', ['agent_type', 'status'])


class AgentConfig(BaseModel):
    """Agent configuration model with validation"""
    agent_id: str = Field(..., description="Unique agent identifier")
    agent_type: str = Field(..., description="Agent type/category")
    agent_name: str = Field(..., description="Human-readable agent name")
    port: int = Field(8080, ge=1024, le=65535, description="Service port")
    log_level: str = Field("INFO", description="Logging level")
    redis_url: str = Field("redis://localhost:6379/0", description="Redis connection URL")
    api_endpoint: str = Field("http://localhost:8000", description="Backend API endpoint")
    enable_metrics: bool = Field(True, description="Enable Prometheus metrics")
    enable_health_check: bool = Field(True, description="Enable health checks")


class BaseAgent(ABC):
    """
    ULTRA OPTIMIZED Base Agent Class
    
    Consolidates common functionality for 100+ agent implementations:
    - Health checks and monitoring
    - Redis connectivity
    - HTTP client management  
    - Metrics collection
    - Structured logging
    - Configuration management
    """
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.logger = self._setup_logging()
        self.redis_client: Optional[aioredis.Redis] = None
        self.http_client: Optional[httpx.AsyncClient] = None
        self.app = self._create_app()
        self.startup_time = time.time()
        
        # Agent state
        self.is_healthy = True
        self.last_health_check = time.time()
        self.error_count = 0
        
    def _setup_logging(self) -> structlog.stdlib.BoundLogger:
        """Configure structured logging"""
        structlog.configure(
            processors=[
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.add_log_level,
                structlog.processors.JSONRenderer()
            ],
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
        
        logger = structlog.get_logger()
        logger = logger.bind(
            agent_id=self.config.agent_id,
            agent_type=self.config.agent_type
        )
        
        # Set logging level
        logging.basicConfig(level=getattr(logging, self.config.log_level.upper()))
        
        return logger
        
    def _create_app(self) -> FastAPI:
        """Create FastAPI application with standard endpoints"""
        
        @asynccontextmanager
        async def lifespan(app: FastAPI):
            # Startup
            await self.startup()
            yield
            # Shutdown
            await self.shutdown()
            
        app = FastAPI(
            title=f"SutazAI {self.config.agent_name}",
            description=f"Agent: {self.config.agent_type}",
            version="2.0",
            lifespan=lifespan
        )
        
        # Standard endpoints
        app.get("/health")(self.health_check)
        app.get("/status")(self.get_status)
        app.get("/metrics")(self.get_metrics)
        app.get("/info")(self.get_info)
        
        return app
        
    async def startup(self):
        """Initialize agent resources"""
        self.logger.info("Starting agent", agent_type=self.config.agent_type)
        
        try:
            # Initialize Redis
            self.redis_client = aioredis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            self.logger.info("Redis connected successfully")
            
            # Initialize HTTP client
            self.http_client = httpx.AsyncClient(timeout=30.0)
            
            # Custom agent initialization
            await self.initialize()
            
            self.logger.info("Agent startup completed")
            
        except Exception as e:
            self.logger.error("Agent startup failed", error=str(e))
            self.is_healthy = False
            raise
            
    async def shutdown(self):
        """Cleanup agent resources"""
        self.logger.info("Shutting down agent")
        
        try:
            # Custom agent cleanup
            await self.cleanup()
            
            # Close connections
            if self.redis_client:
                await self.redis_client.aclose()
                
            if self.http_client:
                await self.http_client.aclose()
                
            self.logger.info("Agent shutdown completed")
            
        except Exception as e:
            self.logger.error("Agent shutdown error", error=str(e))
            
    async def health_check(self) -> JSONResponse:
        """Standard health check endpoint"""
        
        start_time = time.time()
        status = "healthy"
        checks = {}
        
        try:
            # Redis health
            if self.redis_client:
                await self.redis_client.ping()
                checks["redis"] = "healthy"
            else:
                checks["redis"] = "not_configured"
                
            # Custom health checks
            agent_health = await self.check_health()
            checks["agent_specific"] = agent_health
            
            if not self.is_healthy or agent_health != "healthy":
                status = "unhealthy"
                
        except Exception as e:
            status = "unhealthy"
            checks["error"] = str(e)
            self.logger.error("Health check failed", error=str(e))
            
        finally:
            duration = time.time() - start_time
            REQUEST_DURATION.labels(agent_type=self.config.agent_type).observe(duration)
            HEALTH_CHECK_COUNT.labels(agent_type=self.config.agent_type, status=status).inc()
            
        self.last_health_check = time.time()
        
        response_data = {
            "status": status,
            "agent_type": self.config.agent_type,
            "agent_id": self.config.agent_id,
            "uptime_seconds": int(time.time() - self.startup_time),
            "checks": checks,
            "timestamp": time.time()
        }
        
        status_code = 200 if status == "healthy" else 503
        return JSONResponse(content=response_data, status_code=status_code)
        
    async def get_status(self) -> Dict[str, Any]:
        """Get detailed agent status"""
        return {
            "agent_id": self.config.agent_id,
            "agent_type": self.config.agent_type,
            "agent_name": self.config.agent_name,
            "is_healthy": self.is_healthy,
            "uptime_seconds": int(time.time() - self.startup_time),
            "error_count": self.error_count,
            "last_health_check": self.last_health_check,
            "config": self.config.dict()
        }
        
    async def get_metrics(self) -> str:
        """Prometheus metrics endpoint"""
        if not self.config.enable_metrics:
            raise HTTPException(status_code=404, detail="Metrics disabled")
            
        return generate_latest().decode('utf-8')
        
    async def get_info(self) -> Dict[str, Any]:
        """Agent information endpoint"""
        return {
            "agent_name": self.config.agent_name,
            "agent_type": self.config.agent_type,
            "version": "2.0",
            "description": f"SutazAI {self.config.agent_name} Agent",
            "endpoints": [
                "/health",
                "/status", 
                "/metrics",
                "/info"
            ],
            "capabilities": await self.get_capabilities()
        }
        
    def run(self, host: str = "0.0.0.0", port: Optional[int] = None):
        """Run the agent server"""
        import uvicorn
        
        port = port or self.config.port
        self.logger.info("Starting agent server", host=host, port=port)
        
        uvicorn.run(
            self.app,
            host=host,
            port=port,
            log_config=None  # Use our custom logging
        )
        
    # Abstract methods for custom agent implementation
    @abstractmethod
    async def initialize(self):
        """Custom agent initialization - implement in subclass"""
        pass
        
    @abstractmethod
    async def cleanup(self):
        """Custom agent cleanup - implement in subclass"""  
        pass
        
    @abstractmethod
    async def check_health(self) -> str:
        """Custom health check - return 'healthy' or error description"""
        pass
        
    @abstractmethod
    async def get_capabilities(self) -> List[str]:
        """Return list of agent capabilities"""
        pass
        
    # Utility methods for common operations
    async def redis_set(self, key: str, value: Union[str, dict], expire: Optional[int] = None):
        """Set Redis key with optional expiration"""
        if not self.redis_client:
            raise RuntimeError("Redis not connected")
            
        if isinstance(value, dict):
            value = json.dumps(value)
            
        await self.redis_client.set(key, value, ex=expire)
        
    async def redis_get(self, key: str) -> Optional[str]:
        """Get Redis value"""
        if not self.redis_client:
            raise RuntimeError("Redis not connected")
            
        value = await self.redis_client.get(key)
        return value.decode('utf-8') if value else None
        
    async def make_request(self, method: str, url: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request with error handling"""
        if not self.http_client:
            raise RuntimeError("HTTP client not initialized")
            
        try:
            response = await self.http_client.request(method, url, **kwargs)
            response.raise_for_status()
            return response.json()
            
        except Exception as e:
            self.error_count += 1
            self.logger.error("HTTP request failed", method=method, url=url, error=str(e))
            raise
            
    async def log_activity(self, activity: str, data: Optional[Dict] = None):
        """Log agent activity with structured data"""
        log_data = {
            "activity": activity,
            "agent_type": self.config.agent_type,
            "agent_id": self.config.agent_id,
            "timestamp": time.time()
        }
        
        if data:
            log_data["data"] = data
            
        self.logger.info("Agent activity", **log_data)