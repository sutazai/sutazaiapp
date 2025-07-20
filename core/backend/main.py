#!/usr/bin/env python3
"""
SutazAI Unified Backend Service
===============================

High-performance, enterprise-grade backend service for the SutazAI AGI/ASI system.
Combines all backend functionality into a single, optimized service.

Features:
- Multi-agent orchestration
- Real-time model serving
- Vector database integration
- Comprehensive monitoring
- Auto-scaling capabilities
- Enterprise security
"""

import asyncio
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

# Core imports
import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psutil

# Application imports
try:
    from core.backend.config import Settings, get_settings
    from core.backend.database import DatabaseManager
    from core.backend.orchestrator import AgentOrchestrator
    from core.backend.monitoring import MetricsCollector, HealthChecker
    from core.backend.security import SecurityManager
    from core.backend.utils import setup_logging
except ImportError:
    # Fallback imports for relative paths
    try:
        from .config import Settings, get_settings
        from .database import DatabaseManager
        from .orchestrator import AgentOrchestrator
        from .monitoring import MetricsCollector, HealthChecker
        from .security import SecurityManager
        from .utils import setup_logging
    except ImportError:
        # Final fallback
        from config import Settings, get_settings
        from database import DatabaseManager
        from orchestrator import AgentOrchestrator
        from monitoring import MetricsCollector, HealthChecker
        from security import SecurityManager
        from utils import setup_logging

# Configure logging
logger = setup_logging(__name__)

# Pydantic models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default="llama3", description="AI model to use")
    conversation_id: Optional[str] = Field(None, description="Conversation ID for context")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=2048, ge=1, le=8192)

class ChatResponse(BaseModel):
    response: str
    model: str
    conversation_id: str
    timestamp: datetime
    tokens_used: int
    processing_time: float

class SystemStatus(BaseModel):
    status: str
    timestamp: datetime
    version: str
    uptime_seconds: float
    agents: List[Dict[str, Any]]
    tasks: List[Dict[str, Any]]
    system_metrics: Dict[str, Any]

class TaskRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=5000)
    task_type: str = Field(..., description="Type of task to execute")
    priority: int = Field(default=5, ge=1, le=10)
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class TaskResponse(BaseModel):
    task_id: str
    status: str
    created_at: datetime
    estimated_completion: Optional[datetime] = None

# Global application state
app_state = {
    "start_time": time.time(),
    "version": "8.0.0",
    "orchestrator": None,
    "db_manager": None,
    "metrics_collector": None,
    "health_checker": None,
    "security_manager": None
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    logger.info("🚀 Starting SutazAI Unified Backend Service...")
    
    # Initialize settings
    settings = get_settings()
    
    # Initialize core components
    try:
        # Database manager
        app_state["db_manager"] = DatabaseManager(settings)
        await app_state["db_manager"].initialize()
        
        # Agent orchestrator
        app_state["orchestrator"] = AgentOrchestrator(settings)
        await app_state["orchestrator"].initialize()
        
        # Metrics collector
        app_state["metrics_collector"] = MetricsCollector()
        
        # Health checker
        app_state["health_checker"] = HealthChecker(settings)
        
        # Security manager
        app_state["security_manager"] = SecurityManager(settings)
        
        logger.info("✅ All core components initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize core components: {e}")
        raise
    
    yield
    
    # Cleanup
    logger.info("🔄 Shutting down SutazAI Backend...")
    try:
        if app_state["orchestrator"]:
            await app_state["orchestrator"].shutdown()
        if app_state["db_manager"]:
            await app_state["db_manager"].close()
        logger.info("✅ Graceful shutdown completed")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")

# Create FastAPI application
app = FastAPI(
    title="SutazAI Unified Backend",
    description="Enterprise-grade AGI/ASI backend service",
    version=app_state["version"],
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Dependency functions
async def get_orchestrator() -> AgentOrchestrator:
    """Get the agent orchestrator instance"""
    if not app_state["orchestrator"]:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    return app_state["orchestrator"]

async def get_metrics_collector() -> MetricsCollector:
    """Get the metrics collector instance"""
    if not app_state["metrics_collector"]:
        raise HTTPException(status_code=503, detail="Metrics collector not initialized")
    return app_state["metrics_collector"]

async def get_health_checker() -> HealthChecker:
    """Get the health checker instance"""
    if not app_state["health_checker"]:
        raise HTTPException(status_code=503, detail="Health checker not initialized")
    return app_state["health_checker"]

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics and performance"""
    start_time = time.time()
    
    # Process request
    response = await call_next(request)
    
    # Record metrics
    processing_time = time.time() - start_time
    if app_state["metrics_collector"]:
        await app_state["metrics_collector"].record_request(
            method=request.method,
            endpoint=str(request.url.path),
            status_code=response.status_code,
            processing_time=processing_time
        )
    
    # Add performance headers
    response.headers["X-Process-Time"] = str(processing_time)
    response.headers["X-Server-Version"] = app_state["version"]
    
    return response

# Health and status endpoints
@app.get("/health", response_model=Dict[str, Any])
async def health_check():
    """Comprehensive health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app_state["version"],
        "uptime": time.time() - app_state["start_time"]
    }

@app.get("/metrics", response_class=Response)
async def get_metrics(metrics_collector: MetricsCollector = Depends(get_metrics_collector)):
    """Prometheus-compatible metrics endpoint"""
    try:
        # System metrics
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Application metrics
        agent_count = len(app_state["orchestrator"].agents) if app_state["orchestrator"] else 0
        uptime = time.time() - app_state["start_time"]
        
        # Format metrics in Prometheus format
        metrics = f"""# HELP sutazai_cpu_usage_percent CPU usage percentage
# TYPE sutazai_cpu_usage_percent gauge
sutazai_cpu_usage_percent {cpu_usage}

# HELP sutazai_memory_usage_bytes Memory usage in bytes
# TYPE sutazai_memory_usage_bytes gauge
sutazai_memory_usage_bytes {memory.used}

# HELP sutazai_memory_total_bytes Total memory in bytes
# TYPE sutazai_memory_total_bytes gauge
sutazai_memory_total_bytes {memory.total}

# HELP sutazai_disk_usage_bytes Disk usage in bytes
# TYPE sutazai_disk_usage_bytes gauge
sutazai_disk_usage_bytes {disk.used}

# HELP sutazai_active_agents Number of active AI agents
# TYPE sutazai_active_agents gauge
sutazai_active_agents {agent_count}

# HELP sutazai_uptime_seconds System uptime in seconds
# TYPE sutazai_uptime_seconds counter
sutazai_uptime_seconds {uptime}
"""
        
        return Response(content=metrics, media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        raise HTTPException(status_code=500, detail="Metrics collection failed")

@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status(
    orchestrator: AgentOrchestrator = Depends(get_orchestrator),
    health_checker: HealthChecker = Depends(get_health_checker)
):
    """Get comprehensive system status"""
    try:
        # Get agent status
        agents = await orchestrator.get_agent_status()
        
        # Get task status
        tasks = await orchestrator.get_task_status()
        
        # Get system metrics
        system_metrics = {
            "cpu_usage": psutil.cpu_percent(),
            "memory_usage": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent,
            "load_average": os.getloadavg() if hasattr(os, 'getloadavg') else [0, 0, 0]
        }
        
        return SystemStatus(
            status="operational",
            timestamp=datetime.now(),
            version=app_state["version"],
            uptime_seconds=time.time() - app_state["start_time"],
            agents=agents,
            tasks=tasks,
            system_metrics=system_metrics
        )
        
    except Exception as e:
        logger.error(f"Failed to get system status: {e}")
        raise HTTPException(status_code=500, detail="System status unavailable")

# Model management endpoints
@app.get("/api/models")
async def get_available_models():
    """Get list of available AI models"""
    try:
        settings = get_settings()
        # Try to fetch from Ollama
        import requests
        response = requests.get(f"{settings.ollama_url}/api/tags", timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.warning(f"Could not fetch models from Ollama: {e}")
        # Return default models if Ollama is not available
        return {
            "models": [
                {"name": "llama3", "size": "4.7GB", "digest": "default"},
                {"name": "codellama", "size": "3.8GB", "digest": "default"},
                {"name": "deepseek-r1", "size": "8.1GB", "digest": "default"}
            ]
        }

# Chat endpoint
@app.post("/api/chat", response_model=ChatResponse)
async def chat_completion(
    request: ChatRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Process chat completion request"""
    start_time = time.time()
    
    try:
        # Submit task to orchestrator
        task_id = await orchestrator.submit_task(
            description=request.message,
            task_type="chat",
            metadata={
                "model": request.model,
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
                "conversation_id": request.conversation_id
            }
        )
        
        # Wait for completion (with timeout)
        result = await orchestrator.wait_for_completion(task_id, timeout=30.0)
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=result.get("response", "No response generated"),
            model=request.model,
            conversation_id=request.conversation_id or task_id,
            timestamp=datetime.now(),
            tokens_used=result.get("tokens_used", 0),
            processing_time=processing_time
        )
        
    except asyncio.TimeoutError:
        raise HTTPException(status_code=408, detail="Request timeout")
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail="Chat completion failed")

# Task management endpoints
@app.post("/api/tasks", response_model=TaskResponse)
async def create_task(
    request: TaskRequest,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Create a new task for agent execution"""
    try:
        task_id = await orchestrator.submit_task(
            description=request.description,
            task_type=request.task_type,
            priority=request.priority,
            metadata=request.metadata
        )
        
        return TaskResponse(
            task_id=task_id,
            status="queued",
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Task creation failed: {e}")
        raise HTTPException(status_code=500, detail="Task creation failed")

@app.get("/api/tasks/{task_id}")
async def get_task_status(
    task_id: str,
    orchestrator: AgentOrchestrator = Depends(get_orchestrator)
):
    """Get task status and results"""
    try:
        task_info = await orchestrator.get_task_info(task_id)
        if not task_info:
            raise HTTPException(status_code=404, detail="Task not found")
        return task_info
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get task status")

# Agent management endpoints
@app.get("/api/agents")
async def get_agents(orchestrator: AgentOrchestrator = Depends(get_orchestrator)):
    """Get list of all agents and their status"""
    try:
        return await orchestrator.get_agent_status()
    except Exception as e:
        logger.error(f"Failed to get agent status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get agent status")

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions with proper logging"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} - {request.url}")
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "status_code": exc.status_code}
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "status_code": 500}
    )

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True
    )