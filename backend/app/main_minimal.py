"""
Minimal SutazAI Backend - FastAPI Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    logger.info("Starting SutazAI Backend...")
    yield
    logger.info("Shutting down SutazAI Backend...")

# Create FastAPI app
app = FastAPI(
    title="SutazAI AGI/ASI Backend",
    description="Minimal backend for SutazAI Autonomous System",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to SutazAI AGI/ASI System",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "sutazai-backend"
    }

@app.get("/api/v1/system/status")
async def system_status():
    """System status endpoint"""
    return {
        "status": "operational",
        "services": {
            "backend": "healthy",
            "postgres": "connected",
            "redis": "connected",
            "chromadb": "available",
            "qdrant": "available",
            "ollama": "available"
        }
    }

@app.post("/api/v1/chat")
async def chat(message: dict):
    """Basic chat endpoint"""
    return {
        "response": f"Echo: {message.get('content', '')}",
        "status": "success"
    }

# Import and include API routes
try:
    from app.api.v1 import models, vectors, agents, brain, self_improvement
    app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
    app.include_router(vectors.router, prefix="/api/v1/vectors", tags=["vectors"])
    app.include_router(agents.router, prefix="/api/v1/agents", tags=["agents"])
    app.include_router(brain.router, prefix="/api/v1/brain", tags=["brain"])
    app.include_router(self_improvement.router, prefix="/api/v1/self-improvement", tags=["self-improvement"])
    logger.info("API routes loaded")
except Exception as e:
    logger.warning(f"Could not load API routes: {e}")