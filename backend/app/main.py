"""
SutazAI Backend - Main FastAPI Application
"""
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.v1.api import api_router
from app.core.database import engine, Base
from app.services.model_manager import ModelManager
from app.services.agent_orchestrator import AgentOrchestrator
from app.core.exceptions import SutazAIException

# Setup logging
logger = setup_logging()

# Initialize services
model_manager = ModelManager()
agent_orchestrator = AgentOrchestrator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Application lifecycle manager
    """
    # Startup
    logger.info("Starting SutazAI Backend...")
    
    # Create database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    # Initialize model manager
    await model_manager.initialize()
    
    # Initialize agent orchestrator
    await agent_orchestrator.initialize()
    
    # Check GPU availability
    gpu_available = check_gpu_availability()
    logger.info(f"GPU Available: {gpu_available}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down SutazAI Backend...")
    await model_manager.cleanup()
    await agent_orchestrator.cleanup()

def check_gpu_availability() -> bool:
    """Check if GPU is available in WSL2"""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        try:
            # Fallback to nvidia-smi check
            import subprocess
            result = subprocess.run(['nvidia-smi'], capture_output=True)
            return result.returncode == 0
        except:
            return False

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    description="SutazAI AGI/ASI Autonomous System Backend",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url=f"{settings.API_V1_PREFIX}/openapi.json"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Exception handler
@app.exception_handler(SutazAIException)
async def sutazai_exception_handler(request: Request, exc: SutazAIException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail}
    )

# Include API routes
app.include_router(api_router, prefix=settings.API_V1_PREFIX)

# Health check endpoint
@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """System health check endpoint"""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "gpu_available": check_gpu_availability(),
        "services": {
            "database": "connected",
            "redis": "connected",
            "models": model_manager.get_status(),
            "agents": agent_orchestrator.get_status()
        }
    }

if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )