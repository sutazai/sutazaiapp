#!/usr/bin/env python3
"""
FastAPI main application for ai-senior-engineer agent
Compatible with uvicorn: python -m uvicorn main:app --host 0.0.0.0 --port 8080
"""

import os
import logging
import asyncio
from contextlib import asynccontextmanager
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# Import the agent class
from app import Ai_Senior_EngineerAgent

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

# Global agent instance
agent_instance: Ai_Senior_EngineerAgent = None

# Pydantic models for API
class TaskRequest(BaseModel):
    type: str = "default"
    data: Dict[str, Any] = {}
    task_id: str = None

class TaskResponse(BaseModel):
    status: str
    result: Dict[str, Any] = None
    error: str = None
    agent: str

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan"""
    global agent_instance
    
    # Startup
    logger.info("Starting ai-senior-engineer agent...")
    agent_instance = Ai_Senior_EngineerAgent()
    logger.info(f"Agent {agent_instance.name} initialized on port {agent_instance.port}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down ai-senior-engineer agent...")
    if agent_instance:
        agent_instance.status = "shutdown"

# Create FastAPI app
app = FastAPI(
    title="AI Senior Engineer Agent",
    description="SutazAI AI Senior Engineer Agent API",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Senior Engineer Agent API",
        "agent": "ai-senior-engineer",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        health_data = await agent_instance.health_check()
        return JSONResponse(content=health_data)
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=500, detail=f"Health check failed: {str(e)}")

@app.post("/task", response_model=TaskResponse)
async def process_task(task: TaskRequest):
    """Process a task"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    try:
        # Convert to dict for processing
        task_data = {
            "type": task.type,
            "data": task.data,
            "id": task.task_id
        }
        
        result = await agent_instance.process_task(task_data)
        return TaskResponse(**result)
        
    except Exception as e:
        logger.error(f"Task processing failed: {e}")
        return TaskResponse(
            status="error",
            error=str(e),
            agent=agent_instance.agent_id if agent_instance else "ai-senior-engineer"
        )

@app.get("/status")
async def get_status():
    """Get agent status"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "agent_id": agent_instance.agent_id,
        "name": agent_instance.name,
        "status": agent_instance.status,
        "tasks_processed": getattr(agent_instance, 'tasks_processed', 0),
        "description": agent_instance.description,
        "port": agent_instance.port
    }

@app.get("/metrics")
async def get_metrics():
    """Get agent metrics"""
    if not agent_instance:
        raise HTTPException(status_code=503, detail="Agent not initialized")
    
    return {
        "agent_id": agent_instance.agent_id,
        "tasks_processed": getattr(agent_instance, 'tasks_processed', 0),
        "tasks_failed": getattr(agent_instance, 'tasks_failed', 0),
        "status": agent_instance.status,
        "uptime": "running"
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)