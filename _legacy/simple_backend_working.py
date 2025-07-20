#!/usr/bin/env python3
"""
SutazAI Simple Working Backend
==============================

A simplified, working backend service that eliminates import issues
and gets the system running quickly.
"""

import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional, Any
import json

# Core imports
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import psutil

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    system_metrics: Dict[str, Any]

# Global application state
app_state = {
    "start_time": time.time(),
    "version": "8.0.0-simplified",
    "agents": []
}

# Create FastAPI application
app = FastAPI(
    title="SutazAI Simplified Backend",
    description="Simplified working backend service",
    version=app_state["version"],
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    """Track request metrics and performance"""
    start_time = time.time()
    
    response = await call_next(request)
    
    processing_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(processing_time)
    response.headers["X-Server-Version"] = app_state["version"]
    
    return response

# Health and status endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": app_state["version"],
        "uptime": time.time() - app_state["start_time"]
    }

@app.get("/metrics", response_class=Response)
async def get_metrics():
    """Prometheus-compatible metrics endpoint"""
    try:
        cpu_usage = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        uptime = time.time() - app_state["start_time"]
        
        metrics = f"""# HELP sutazai_cpu_usage_percent CPU usage percentage
# TYPE sutazai_cpu_usage_percent gauge
sutazai_cpu_usage_percent {cpu_usage}

# HELP sutazai_memory_usage_bytes Memory usage in bytes
# TYPE sutazai_memory_usage_bytes gauge
sutazai_memory_usage_bytes {memory.used}

# HELP sutazai_memory_total_bytes Total memory in bytes
# TYPE sutazai_memory_total_bytes gauge
sutazai_memory_total_bytes {memory.total}

# HELP sutazai_uptime_seconds System uptime in seconds
# TYPE sutazai_uptime_seconds counter
sutazai_uptime_seconds {uptime}
"""
        
        return Response(content=metrics, media_type="text/plain")
        
    except Exception as e:
        logger.error(f"Failed to collect metrics: {e}")
        raise HTTPException(status_code=500, detail="Metrics collection failed")

@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get comprehensive system status"""
    try:
        # Mock agents for now
        agents = [
            {
                "id": "agent-1",
                "name": "Chat Agent",
                "type": "chat",
                "status": "idle",
                "capabilities": ["conversation", "qa"]
            }
        ]
        
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
        # Try to fetch from Ollama if available
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        import httpx
        async with httpx.AsyncClient() as client:
            response = await client.get(f"{ollama_url}/api/tags", timeout=5)
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
async def chat_completion(request: ChatRequest):
    """Process chat completion request"""
    start_time = time.time()
    
    try:
        # Simple mock response for now - replace with actual Ollama integration
        ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
        try:
            import httpx
            request_data = {
                "model": request.model,
                "prompt": request.message,
                "stream": False,
                "options": {
                    "temperature": request.temperature,
                    "num_ctx": request.max_tokens
                }
            }
            
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{ollama_url}/api/generate",
                    json=request_data,
                    timeout=30.0
                )
                
                if response.status_code == 200:
                    result = response.json()
                    ai_response = result.get("response", "I'm working on processing your request.")
                    tokens_used = result.get("eval_count", 0)
                else:
                    ai_response = "I'm currently setting up. Please try again in a moment."
                    tokens_used = 0
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            ai_response = f"I received your message: '{request.message}'. I'm currently initializing my AI capabilities. Please try again in a moment."
            tokens_used = 0
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            response=ai_response,
            model=request.model,
            conversation_id=request.conversation_id or f"conv-{int(time.time())}",
            timestamp=datetime.now(),
            tokens_used=tokens_used,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Chat completion failed: {e}")
        raise HTTPException(status_code=500, detail="Chat completion failed")

# Agent management endpoints
@app.get("/api/agents")
async def get_agents():
    """Get list of all agents and their status"""
    try:
        # Mock agent data for now
        agents = [
            {
                "id": "agent-1",
                "name": "Chat Agent",
                "type": "chat",
                "status": "idle",
                "capabilities": ["conversation", "qa", "general_chat"],
                "completed_tasks": 0,
                "last_activity": datetime.now().isoformat()
            },
            {
                "id": "agent-2", 
                "name": "Code Generator",
                "type": "code_generator",
                "status": "idle",
                "capabilities": ["code_generation", "code_review", "debugging"],
                "completed_tasks": 0,
                "last_activity": datetime.now().isoformat()
            }
        ]
        return agents
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

# Startup event
@app.on_event("startup")
async def startup_event():
    """Application startup tasks"""
    logger.info("🚀 SutazAI Simplified Backend starting up...")
    logger.info(f"✅ Backend ready on version {app_state['version']}")

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Application shutdown tasks"""
    logger.info("🔄 SutazAI Backend shutting down...")

# Main entry point
if __name__ == "__main__":
    uvicorn.run(
        "simple_backend_working:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
        access_log=True
    )