"""
SutazAI Platform Main Application
FastAPI backend with comprehensive service integrations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import sys
from app.core.config import settings
from app.core.database import init_db, close_db, Base
from app.services.connections import service_connections
from app.api.v1.router import api_router
# Import models to ensure they're registered
from app.models import User

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manage application lifecycle
    Initialize connections on startup, cleanup on shutdown
    """
    logger.info(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Connect to all services
        await service_connections.connect_all()
        logger.info("All service connections established")
        
        # Register with Consul
        await register_with_consul()
        
        yield
        
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down application")
        await service_connections.disconnect_all()
        await close_db()
        await deregister_from_consul()
        logger.info("Cleanup completed")


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(api_router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "docs": "/docs",
        "api": settings.API_V1_STR
    }


@app.get("/health")
async def health_check():
    """Basic health check"""
    return {"status": "healthy", "app": settings.APP_NAME}


@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check for all services"""
    try:
        service_health = await service_connections.health_check()
        all_healthy = all(service_health.values())
        
        return {
            "status": "healthy" if all_healthy else "degraded",
            "app": settings.APP_NAME,
            "version": settings.APP_VERSION,
            "services": service_health,
            "healthy_count": sum(service_health.values()),
            "total_services": len(service_health)
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))


async def register_with_consul():
    """Register service with Consul"""
    try:
        import socket
        hostname = socket.gethostname()
        
        service_definition = {
            "ID": f"sutazai-backend-{hostname}",
            "Name": "sutazai-backend",
            "Tags": ["api", "backend", "fastapi"],
            "Address": hostname,
            "Port": settings.PORT,
            "Check": {
                "HTTP": f"http://{hostname}:{settings.PORT}/health",
                "Interval": "30s",
                "Timeout": "10s"
            }
        }
        
        response = await service_connections.consul_client.put(
            "/v1/agent/service/register",
            json=service_definition
        )
        
        if response.status_code == 200:
            logger.info("Service registered with Consul")
        else:
            logger.warning(f"Consul registration failed: {response.status_code}")
    except Exception as e:
        logger.warning(f"Could not register with Consul: {e}")


async def deregister_from_consul():
    """Deregister service from Consul"""
    try:
        import socket
        hostname = socket.gethostname()
        service_id = f"sutazai-backend-{hostname}"
        
        response = await service_connections.consul_client.put(
            f"/v1/agent/service/deregister/{service_id}"
        )
        
        if response.status_code == 200:
            logger.info("Service deregistered from Consul")
    except Exception as e:
        logger.warning(f"Could not deregister from Consul: {e}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS if not settings.DEBUG else 1
    )