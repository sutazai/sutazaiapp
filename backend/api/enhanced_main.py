#!/usr/bin/env python3
"""
SutazAI Enhanced Backend API
Enterprise-grade FastAPI application with AGI orchestration
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Any, Optional
from datetime import datetime

from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import uvicorn
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
import structlog

from backend.core.orchestrator import orchestrator, TaskRequest, TaskStatus
from backend.core.config import get_settings
from backend.utils.logging_setup import get_api_logger
from backend.models.base_models import Agent, Task, Workflow
from backend.security.auth import verify_jwt_token, get_current_user
from backend.middleware.security import SecurityMiddleware
from backend.middleware.monitoring import PrometheusMiddleware
from backend.routers import (
    agents_router,
    tasks_router, 
    models_router,
    documents_router,
    monitoring_router,
    workflows_router
)

logger = get_api_logger()
settings = get_settings()

# Prometheus metrics
REQUESTS = Counter('sutazai_http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_DURATION = Histogram('sutazai_http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('sutazai_active_connections', 'Active connections')
TASK_COUNTER = Counter('sutazai_tasks_total', 'Total tasks processed', ['task_type', 'status'])
AGENT_GAUGE = Gauge('sutazai_active_agents', 'Number of active agents')

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting SutazAI Enhanced Backend...")
    
    try:
        # Initialize orchestrator
        await orchestrator.initialize()
        
        # Start background tasks
        asyncio.create_task(health_check_background())
        asyncio.create_task(metrics_collection_background())
        
        logger.info("SutazAI Backend started successfully")
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down SutazAI Backend...")
        await orchestrator.shutdown()
        logger.info("SutazAI Backend shutdown complete")

# Create FastAPI application
app = FastAPI(
    title="SutazAI AGI/ASI System",
    description="Enterprise-grade Artificial General Intelligence platform",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan
)

# Security
security = HTTPBearer()

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(SecurityMiddleware)
app.add_middleware(PrometheusMiddleware)

# Include routers
app.include_router(agents_router.router, prefix="/api/v1/agents", tags=["agents"])
app.include_router(tasks_router.router, prefix="/api/v1/tasks", tags=["tasks"])
app.include_router(models_router.router, prefix="/api/v1/models", tags=["models"])
app.include_router(documents_router.router, prefix="/api/v1/documents", tags=["documents"])
app.include_router(workflows_router.router, prefix="/api/v1/workflows", tags=["workflows"])
app.include_router(monitoring_router.router, prefix="/api/v1/monitoring", tags=["monitoring"])

# Root endpoints

@app.get("/")
async def root():
    """Root endpoint with system information"""
    return {
        "name": "SutazAI AGI/ASI System",
        "version": "2.0.0",
        "status": "operational",
        "timestamp": datetime.now().isoformat(),
        "docs_url": "/docs",
        "health_url": "/health"
    }

@app.get("/health")
async def health_check():
    """Comprehensive health check endpoint"""
    try:
        # Check orchestrator health
        orchestrator_status = "healthy" if orchestrator.redis_client else "unhealthy"
        
        # Check database connectivity
        db_status = "healthy"  # TODO: Implement actual DB check
        
        # Get system metrics
        metrics = await orchestrator.get_system_metrics()
        
        # Check agent status
        agent_count = len([a for a in orchestrator.agents.values() if a["status"] == "idle" or a["status"] == "busy"])
        
        health_data = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "orchestrator": orchestrator_status,
                "database": db_status,
                "redis": "healthy" if orchestrator.redis_client else "unhealthy"
            },
            "metrics": {
                "active_agents": agent_count,
                "tasks_completed": metrics.get("tasks_completed", 0),
                "tasks_failed": metrics.get("tasks_failed", 0),
                "queue_length": metrics.get("queue_length", 0)
            },
            "uptime": time.time() - app.state.start_time if hasattr(app.state, 'start_time') else 0
        }
        
        # Determine overall health
        if orchestrator_status == "unhealthy" or db_status == "unhealthy":
            health_data["status"] = "unhealthy"
            return JSONResponse(content=health_data, status_code=503)
        
        return health_data
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            content={
                "status": "unhealthy",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            },
            status_code=503
        )

@app.get("/ready")
async def readiness_check():
    """Kubernetes-style readiness check"""
    try:
        # Check if orchestrator is ready
        if not orchestrator.redis_client:
            raise Exception("Orchestrator not ready")
        
        # Check if we have any agents
        if not orchestrator.agents:
            raise Exception("No agents available")
        
        return {"status": "ready", "timestamp": datetime.now().isoformat()}
        
    except Exception as e:
        return JSONResponse(
            content={"status": "not_ready", "error": str(e)},
            status_code=503
        )

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(content=generate_latest(), media_type="text/plain")

# Main API endpoints

@app.post("/api/v1/execute")
async def execute_task(
    task_data: Dict[str, Any],
    background_tasks: BackgroundTasks,
    current_user: Dict = Depends(get_current_user)
):
    """Execute a task using the AGI orchestrator"""
    try:
        # Create task request
        task_request = TaskRequest(
            type=task_data.get("type", "general"),
            payload=task_data.get("payload", {}),
            priority=task_data.get("priority", 5),
            timeout=task_data.get("timeout", 300),
            retries=task_data.get("retries", 3),
            dependencies=task_data.get("dependencies", []),
            agent_constraints=task_data.get("agent_constraints", {}),
            context={
                "user_id": current_user.get("id"),
                "request_id": task_data.get("request_id"),
                **task_data.get("context", {})
            }
        )
        
        # Submit task to orchestrator
        task_id = await orchestrator.submit_task(task_request)
        
        # Update metrics
        TASK_COUNTER.labels(task_type=task_request.type, status="submitted").inc()
        
        return {
            "task_id": task_id,
            "status": "submitted",
            "estimated_completion": task_request.timeout,
            "message": "Task submitted successfully"
        }
        
    except Exception as e:
        logger.error(f"Task execution failed: {e}")
        TASK_COUNTER.labels(task_type="unknown", status="failed").inc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/tasks/{task_id}/status")
async def get_task_status(
    task_id: str,
    current_user: Dict = Depends(get_current_user)
):
    """Get the status of a specific task"""
    try:
        task_result = await orchestrator.get_task_status(task_id)
        
        if not task_result:
            raise HTTPException(status_code=404, detail="Task not found")
        
        return {
            "task_id": task_id,
            "status": task_result.status.value,
            "result": task_result.result,
            "error": task_result.error,
            "execution_time": task_result.execution_time,
            "agent_id": task_result.agent_id,
            "completed_at": task_result.completed_at.isoformat() if task_result.completed_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get task status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/agents")
async def list_agents(current_user: Dict = Depends(get_current_user)):
    """List all available agents"""
    try:
        agents = []
        for agent_id, agent_data in orchestrator.agents.items():
            agents.append({
                "id": agent_id,
                "name": agent_data["name"],
                "type": agent_data["type"],
                "status": agent_data["status"].value if hasattr(agent_data["status"], 'value') else str(agent_data["status"]),
                "capabilities": agent_data.get("capabilities", []),
                "current_task": agent_data.get("current_task"),
                "performance_metrics": agent_data.get("performance_metrics", {}),
                "last_heartbeat": agent_data.get("last_heartbeat")
            })
        
        return {
            "agents": agents,
            "total_count": len(agents),
            "active_count": len([a for a in agents if a["status"] in ["idle", "busy"]])
        }
        
    except Exception as e:
        logger.error(f"Failed to list agents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/v1/system/metrics")
async def get_system_metrics(current_user: Dict = Depends(get_current_user)):
    """Get system-wide performance metrics"""
    try:
        metrics = await orchestrator.get_system_metrics()
        
        # Add additional system information
        metrics.update({
            "timestamp": datetime.now().isoformat(),
            "total_agents": len(orchestrator.agents),
            "active_tasks": len(orchestrator.active_tasks),
            "completed_tasks_count": len([
                r for r in orchestrator.task_results.values() 
                if r.status == TaskStatus.COMPLETED
            ])
        })
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/chat/completions")
async def chat_completion(
    request_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """OpenAI-compatible chat completion endpoint"""
    try:
        # Extract messages and model
        messages = request_data.get("messages", [])
        model = request_data.get("model", "deepseek-r1:8b")
        temperature = request_data.get("temperature", 0.7)
        max_tokens = request_data.get("max_tokens", 2048)
        stream = request_data.get("stream", False)
        
        # Create task for chat completion
        task_request = TaskRequest(
            type="chat_completion",
            payload={
                "messages": messages,
                "model": model,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "stream": stream
            },
            priority=1,  # High priority for chat
            timeout=60,  # 1 minute timeout for chat
            context={
                "user_id": current_user.get("id"),
                "endpoint": "chat_completion"
            }
        )
        
        # Submit and wait for quick completion
        task_id = await orchestrator.submit_task(task_request)
        
        # For non-streaming, wait for completion
        if not stream:
            # Poll for completion (max 60 seconds)
            for _ in range(60):
                task_result = await orchestrator.get_task_status(task_id)
                if task_result and task_result.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
                    break
                await asyncio.sleep(1)
            
            if task_result.status == TaskStatus.COMPLETED:
                return task_result.result
            else:
                raise HTTPException(status_code=500, detail=task_result.error or "Chat completion failed")
        
        # For streaming, return task ID for client to poll
        return {"task_id": task_id, "stream": True}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/code/generate")
async def generate_code(
    request_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Generate code using AI agents"""
    try:
        # Extract request parameters
        prompt = request_data.get("prompt", "")
        language = request_data.get("language", "python")
        style = request_data.get("style", "clean")
        include_tests = request_data.get("include_tests", False)
        include_docs = request_data.get("include_docs", False)
        
        if not prompt:
            raise HTTPException(status_code=400, detail="Prompt is required")
        
        # Create task for code generation
        task_request = TaskRequest(
            type="code_generation",
            payload={
                "prompt": prompt,
                "language": language,
                "style": style,
                "include_tests": include_tests,
                "include_docs": include_docs
            },
            priority=3,
            timeout=180,  # 3 minutes for code generation
            context={
                "user_id": current_user.get("id"),
                "endpoint": "code_generation"
            }
        )
        
        # Submit task
        task_id = await orchestrator.submit_task(task_request)
        
        return {
            "task_id": task_id,
            "status": "submitted",
            "message": "Code generation task submitted",
            "estimated_completion": 180
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Code generation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/documents/analyze")
async def analyze_document(
    request_data: Dict[str, Any],
    current_user: Dict = Depends(get_current_user)
):
    """Analyze uploaded document"""
    try:
        document_path = request_data.get("document_path")
        analysis_type = request_data.get("analysis_type", "comprehensive")
        
        if not document_path:
            raise HTTPException(status_code=400, detail="Document path is required")
        
        # Create task for document analysis
        task_request = TaskRequest(
            type="document_processing",
            payload={
                "document_path": document_path,
                "analysis_type": analysis_type
            },
            priority=4,
            timeout=300,  # 5 minutes for document processing
            context={
                "user_id": current_user.get("id"),
                "endpoint": "document_analysis"
            }
        )
        
        # Submit task
        task_id = await orchestrator.submit_task(task_request)
        
        return {
            "task_id": task_id,
            "status": "submitted", 
            "message": "Document analysis task submitted",
            "estimated_completion": 300
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Document analysis failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Background tasks

async def health_check_background():
    """Background health monitoring"""
    while True:
        try:
            await asyncio.sleep(30)
            
            # Update agent gauge
            active_agents = len([
                a for a in orchestrator.agents.values() 
                if a["status"] in ["idle", "busy"]
            ])
            AGENT_GAUGE.set(active_agents)
            
        except Exception as e:
            logger.error(f"Health check background task failed: {e}")

async def metrics_collection_background():
    """Background metrics collection"""
    while True:
        try:
            await asyncio.sleep(60)  # Collect metrics every minute
            
            # Collect and store metrics
            metrics = await orchestrator.get_system_metrics()
            
            # Log important metrics
            logger.info(
                "System metrics",
                active_agents=metrics.get("active_agents", 0),
                tasks_completed=metrics.get("tasks_completed", 0),
                tasks_failed=metrics.get("tasks_failed", 0),
                queue_length=metrics.get("queue_length", 0),
                average_execution_time=metrics.get("average_execution_time", 0)
            )
            
        except Exception as e:
            logger.error(f"Metrics collection background task failed: {e}")

# Error handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now().isoformat()
        }
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Set startup timestamp"""
    app.state.start_time = time.time()
    logger.info("SutazAI Enhanced Backend API started")

if __name__ == "__main__":
    uvicorn.run(
        "backend.api.enhanced_main:app",
        host=settings.SERVER_HOST,
        port=settings.SERVER_PORT,
        reload=settings.DEBUG,
        workers=1 if settings.DEBUG else settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower(),
        access_log=True
    )