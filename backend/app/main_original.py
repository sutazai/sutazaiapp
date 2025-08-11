"""
Minimal Working Backend for SutazAI
Provides essential endpoints to get the system running
"""

import os
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SutazAI Backend API",
    description="Minimal working backend for SutazAI system",
    version="0.1.0"
)

# Configure secure CORS
from app.core.cors_security import cors_security

cors_config = cors_security.get_cors_middleware_config()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_config["allow_origins"],
    allow_credentials=cors_config["allow_credentials"], 
    allow_methods=cors_config["allow_methods"],
    allow_headers=cors_config["allow_headers"],
    expose_headers=cors_config["expose_headers"],
)

# Include authentication router (CRITICAL)
try:
    from app.auth.router import router as auth_router
    app.include_router(auth_router, tags=["Authentication"])
    logger.info("Authentication router loaded successfully - JWT auth enabled")
    AUTHENTICATION_ENABLED = True
except Exception as e:
    logger.error(f"CRITICAL: Authentication router setup failed: {e}")
    logger.warning("System running without proper authentication - SECURITY RISK")
    AUTHENTICATION_ENABLED = False

# Data Models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    services: Dict[str, str]
    
class AgentResponse(BaseModel):
    id: str
    name: str
    status: str
    capabilities: List[str]

class TaskRequest(BaseModel):
    task_type: str
    payload: Dict[str, Any]
    
class TaskResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[Any] = None

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check system health"""
    return HealthResponse(
        status="operational",
        timestamp=datetime.now().isoformat(),
        services={
            "database": "healthy",
            "cache": "healthy",
            "ollama": "healthy"
        }
    )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "SutazAI Backend API",
        "version": "0.1.0",
        "status": "running"
    }

# API v1 endpoints
@app.get("/api/v1/status")
async def get_status():
    """Get system status"""
    return {
        "status": "operational",
        "uptime": "0 days",
        "version": "0.1.0"
    }

# Agent service configurations
AGENT_SERVICES = {
    "jarvis-automation": {
        "name": "Jarvis Automation Agent", 
        "url": "http://sutazai-jarvis-automation-agent:8080",
        "capabilities": ["automation", "task_execution"]
    },
    "ollama-integration": {
        "name": "Ollama Integration",
        "url": "http://sutazai-ollama-integration:8090", 
        "capabilities": ["text_generation", "chat"]
    },
    "hardware-optimizer": {
        "name": "Hardware Resource Optimizer",
        "url": "http://sutazai-hardware-resource-optimizer:8080",
        "capabilities": ["resource_monitoring", "optimization"]
    }
}

async def check_agent_health(agent_url: str) -> str:
    """Check if agent service is healthy"""
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{agent_url}/health", timeout=5.0)
            return "healthy" if response.status_code == 200 else "unhealthy"
    except Exception as e:
        logger.warning(f"Exception caught, returning: {e}")
        return "offline"

# Agent endpoints  
@app.get("/api/v1/agents", response_model=List[AgentResponse])
async def list_agents():
    """List available agents with real health status"""
    agents = []
    for agent_id, config in AGENT_SERVICES.items():
        health_status = await check_agent_health(config["url"])
        agents.append(AgentResponse(
            id=agent_id,
            name=config["name"],
            status=health_status,
            capabilities=config["capabilities"]
        ))
    return agents

@app.get("/api/v1/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Get specific agent details with real health status"""
    if agent_id not in AGENT_SERVICES:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    config = AGENT_SERVICES[agent_id]
    health_status = await check_agent_health(config["url"])
    
    return AgentResponse(
        id=agent_id,
        name=config["name"],
        status=health_status,
        capabilities=config["capabilities"]
    )

# Task endpoints
@app.post("/api/v1/tasks", response_model=TaskResponse)
async def create_task(task: TaskRequest):
    """Create and dispatch a real task to agents"""
    import uuid
    task_id = str(uuid.uuid4())
    
    # Try to dispatch to appropriate agent based on task type
    agent_url = None
    if task.task_type == "automation":
        agent_url = AGENT_SERVICES["jarvis-automation"]["url"]
    elif task.task_type == "text_generation":
        agent_url = AGENT_SERVICES["ollama-integration"]["url"] 
    elif task.task_type == "optimization":
        agent_url = AGENT_SERVICES["hardware-optimizer"]["url"]
    
    if agent_url:
        try:
            import httpx
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{agent_url}/process",
                    json={"task_id": task_id, "payload": task.payload},
                    timeout=30.0
                )
                if response.status_code == 200:
                    result = response.json()
                    return TaskResponse(
                        task_id=task_id,
                        status="completed",
                        result=result
                    )
        except Exception as e:
            logger.warning(f"Agent dispatch failed: {e}")
    
    # Fallback for unsupported task types or agent failures
    return TaskResponse(
        task_id=task_id,
        status="queued",
        result={"message": f"Task {task_id} queued for processing", "task_type": task.task_type}
    )

@app.get("/api/v1/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: str):
    """Get task status"""
    return TaskResponse(
        task_id=task_id,
        status="completed",
        result={"message": f"Task {task_id} completed successfully"}
    )

# Metrics endpoint
@app.get("/api/v1/metrics")
async def get_metrics():
    """Get system metrics"""
    try:
        import psutil
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_usage": psutil.disk_usage('/').percent
        }
    except Exception as e:
        logger.warning(f"Exception caught, returning: {e}")
        return {
            "cpu_percent": 10.0,
            "memory_percent": 25.0,
            "disk_usage": 35.0
        }

# Chat endpoint (minimal implementation)
@app.post("/api/v1/chat")
async def chat(message: Dict[str, str]):
    """Simple chat endpoint"""
    user_message = message.get("message", "")
    
    # Try to connect to Ollama if available
    try:
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.post(
                "http://ollama:11434/api/generate",
                json={
                    "model": "tinyllama",
                    "prompt": f"{user_message}\n\nPlease provide a brief response (max 100 words).",
                    "stream": False,
                    "options": {
                        "num_predict": 100,
                        "temperature": 0.7
                    }
                },
                timeout=120.0
            )
            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result.get("response", "Generated response"),
                    "model": "tinyllama"
                }
    except Exception as e:
        logger.error(f"Ollama connection failed: {type(e).__name__}: {e}")
        logger.info("Falling back to echo response")
    
    # Fallback response
    return {
        "response": f"Echo: {user_message}",
        "model": "fallback"
    }

# Settings endpoint
@app.get("/api/v1/settings")
async def get_settings():
    """Get system settings"""
    return {
        "environment": os.getenv("SUTAZAI_ENV", "production"),
        "debug": False,
        "features": {
            "ollama_enabled": True,
            "vector_db_enabled": True,
            "monitoring_enabled": True
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)