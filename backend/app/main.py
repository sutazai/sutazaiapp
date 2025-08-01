"""
SutazAI Task Automation System - Main Application
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SutazAI Task Automation API",
    description="Local AI-powered task automation system",
    version="1.0.0"
)

# Configure CORS
origins = [
    "http://localhost:8501",  # Streamlit frontend
    "http://localhost:8000",  # API docs
    "http://localhost:3000",  # Alternative frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Health check endpoint
@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "model": "tinyllama",
        "version": "1.0.0"
    }

# Root endpoint
@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "SutazAI Task Automation API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

# API v1 routes
# Basic API routes - minimal version
@app.get("/api/v1")
async def api_v1_root():
    """API v1 root endpoint"""
    return {
        "message": "SutazAI API v1",
        "version": "1.0.0",
        "endpoints": {
            "agents": "/api/v1/agents",
            "agents_status": "/api/v1/agents/status",
            "task_assignment": "/api/v1/agents/task",
            "health": "/api/v1/health"
        }
    }

@app.get("/api/v1/health")
async def api_health():
    return {"status": "healthy", "api_version": "v1"}

@app.get("/api/v1/agents")
async def get_agents():
    return {
        "agents": [
            {"name": "senior-ai-engineer", "status": "ready"},
            {"name": "infrastructure-devops-manager", "status": "ready"},
            {"name": "testing-qa-validator", "status": "ready"}
        ]
    }

@app.get("/api/v1/agents/status")
async def get_agents_status():
    """Get detailed agent status information"""
    return {
        "agents": [
            {
                "name": "senior-ai-engineer",
                "status": "active",
                "last_heartbeat": datetime.now().isoformat(),
                "tasks_completed": 0,
                "capabilities": ["code_analysis", "optimization", "ai_integration"]
            },
            {
                "name": "infrastructure-devops-manager",
                "status": "active",
                "last_heartbeat": datetime.now().isoformat(),
                "tasks_completed": 0,
                "capabilities": ["deployment", "infrastructure", "monitoring"]
            },
            {
                "name": "testing-qa-validator",
                "status": "active",
                "last_heartbeat": datetime.now().isoformat(),
                "tasks_completed": 0,
                "capabilities": ["testing", "quality_assurance", "validation"]
            }
        ],
        "total_agents": 3,
        "active_agents": 3,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/api/v1/agents/task")
async def assign_task_to_agent(request: dict):
    """Assign a task to an agent"""
    task = request.get("task", "")
    agent_type = request.get("agent_type", "senior-ai-engineer")
    priority = request.get("priority", "medium")
    
    if not task:
        raise HTTPException(status_code=400, detail="Task description is required")
    
    # Generate a task ID
    import uuid
    task_id = str(uuid.uuid4())
    
    return {
        "task_id": task_id,
        "task": task,
        "agent": agent_type,
        "priority": priority,
        "status": "queued",
        "created_at": datetime.now().isoformat(),
        "estimated_completion": "pending"
    }

# Agent management endpoints
@app.post("/api/agents/heartbeat")
async def agent_heartbeat(request: dict):
    """Agent heartbeat endpoint"""
    agent_name = request.get("agent_name", "unknown")
    logger.info(f"Heartbeat received from agent: {agent_name}")
    return {
        "status": "acknowledged",
        "agent_name": agent_name,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/api/tasks/next/{agent_name}")
async def get_next_task(agent_name: str):
    """Get next task for agent"""
    # For now, return no tasks available
    # This can be expanded with actual task queue logic
    return {
        "task": None,
        "message": f"No tasks available for {agent_name}",
        "timestamp": datetime.now().isoformat()
    }

# Error handlers
@app.exception_handler(404)
async def not_found(request, exc):
    return JSONResponse(
        status_code=404,
        content={"detail": "Not found"}
    )

@app.exception_handler(500)
async def internal_error(request, exc):
    logger.error(f"Internal error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("Starting SutazAI Task Automation System...")
    
    # Check Ollama connection
    import httpx
    try:
        ollama_url = os.getenv("OLLAMA_BASE_URL", "http://ollama:11434")
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_url}/api/tags")
            if response.status_code == 200:
                logger.info("✓ Ollama connection successful")
            else:
                logger.warning("✗ Ollama connection failed")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama: {e}")
    
    logger.info("SutazAI API started successfully")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("Shutting down SutazAI API...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )