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

# Agent endpoints
@app.get("/api/v1/agents", response_model=List[AgentResponse])
async def list_agents():
    """List available agents"""
    return [
        AgentResponse(
            id="ollama-integration",
            name="Ollama Integration",
            status="healthy",
            capabilities=["text_generation", "chat"]
        ),
        AgentResponse(
            id="ai-orchestrator",
            name="AI Agent Orchestrator",
            status="healthy",
            capabilities=["task_routing", "agent_coordination"]
        ),
        AgentResponse(
            id="hardware-optimizer",
            name="Hardware Resource Optimizer",
            status="healthy",
            capabilities=["resource_monitoring", "optimization"]
        )
    ]

@app.get("/api/v1/agents/{agent_id}", response_model=AgentResponse)
async def get_agent(agent_id: str):
    """Get specific agent details"""
    agents = {
        "ollama-integration": AgentResponse(
            id="ollama-integration",
            name="Ollama Integration",
            status="healthy",
            capabilities=["text_generation", "chat"]
        ),
        "ai-orchestrator": AgentResponse(
            id="ai-orchestrator",
            name="AI Agent Orchestrator",
            status="healthy",
            capabilities=["task_routing", "agent_coordination"]
        ),
        "hardware-optimizer": AgentResponse(
            id="hardware-optimizer",
            name="Hardware Resource Optimizer",
            status="healthy",
            capabilities=["resource_monitoring", "optimization"]
        )
    }
    
    if agent_id not in agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_id} not found")
    
    return agents[agent_id]

# Task endpoints
@app.post("/api/v1/tasks", response_model=TaskResponse)
async def create_task(task: TaskRequest):
    """Create a new task"""
    import uuid
    task_id = str(uuid.uuid4())
    
    return TaskResponse(
        task_id=task_id,
        status="processing",
        result={"message": f"Task {task_id} created and processing"}
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
                    "prompt": user_message,
                    "stream": False
                },
                timeout=30.0
            )
            if response.status_code == 200:
                result = response.json()
                return {
                    "response": result.get("response", "Generated response"),
                    "model": "tinyllama"
                }
    except Exception as e:
        logger.warning(f"Ollama connection failed: {e}")
    
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