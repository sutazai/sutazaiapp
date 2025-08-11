#!/usr/bin/env python3
"""
Main entry point for hardware-resource-optimizer
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Hardware Resource Optimizer",
    description="Agent for Hardware Resource Optimizer operations",
    version="1.0.0"
)

# Request/Response models
class HealthResponse(BaseModel):
    status: str
    agent: str
    timestamp: str
    version: str = "1.0.0"

class TaskRequest(BaseModel):
    type: str = "process"
    data: dict = {}
    priority: str = "normal"

class TaskResponse(BaseModel):
    status: str
    agent: str
    result: dict = {}
    timestamp: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        agent="hardware-resource-optimizer",
        timestamp=datetime.utcnow().isoformat()
    )

# Main task processing endpoint
@app.post("/task", response_model=TaskResponse)
async def process_task(request: TaskRequest):
    """Process incoming tasks"""
    try:
        logger.info(f"Processing task of type: {request.type}")
        
        result = {
            "message": "Task processed successfully",
            "task_type": request.type,
            "data_keys": list(request.data.keys())
        }
        
        return TaskResponse(
            status="success",
            agent="hardware-resource-optimizer",
            result=result,
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Error processing task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": "hardware-resource-optimizer",
        "status": "running",
        "endpoints": ["/health", "/task", "/docs"]
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
