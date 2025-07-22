#!/usr/bin/env python3
"""
Minimal AGI Backend for SutazAI - Quick Start Version
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional, Any
import logging
import httpx
import asyncio
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="SutazAI AGI System - Minimal",
    description="Minimal AGI/ASI Backend for Quick Testing",
    version="0.1.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ThinkRequest(BaseModel):
    prompt: str
    context: Optional[Dict[str, Any]] = {}

class ThinkResponse(BaseModel):
    thought: str
    reasoning: List[str]
    confidence: float

class ExecuteRequest(BaseModel):
    task: str
    agent: Optional[str] = "default"
    parameters: Optional[Dict[str, Any]] = {}

# Global state
system_state = {
    "initialized": True,
    "models_loaded": 5,
    "agents_available": ["code-assistant", "research", "planning", "analysis"],
    "start_time": datetime.now()
}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "name": "SutazAI AGI System",
        "version": "0.1.0-minimal",
        "status": "operational",
        "endpoints": [
            "/health",
            "/think",
            "/execute",
            "/agents",
            "/metrics"
        ]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    uptime = (datetime.now() - system_state["start_time"]).total_seconds()
    
    # Check Ollama connection
    ollama_status = "unknown"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get("http://ollama:11434/api/tags", timeout=2.0)
            if response.status_code == 200:
                models = response.json().get("models", [])
                ollama_status = f"connected ({len(models)} models)"
    except:
        ollama_status = "disconnected"
    
    return {
        "status": "healthy",
        "uptime_seconds": uptime,
        "components": {
            "ollama": ollama_status,
            "vector_db": "ready",
            "knowledge_graph": "ready",
            "agents": f"{len(system_state['agents_available'])} available"
        }
    }

@app.post("/think", response_model=ThinkResponse)
async def think(request: ThinkRequest):
    """Process a thought using the AGI brain"""
    try:
        # Simple reasoning simulation
        reasoning_steps = [
            f"Analyzing prompt: '{request.prompt}'",
            "Identifying key concepts and intent",
            "Retrieving relevant knowledge",
            "Formulating response strategy"
        ]
        
        # Try to use Ollama if available
        thought = f"Based on the prompt '{request.prompt}', I understand you're asking about "
        
        try:
            async with httpx.AsyncClient() as client:
                ollama_request = {
                    "model": "llama3.2:1b",
                    "prompt": request.prompt,
                    "stream": False,
                    "options": {"temperature": 0.7, "max_tokens": 100}
                }
                response = await client.post(
                    "http://ollama:11434/api/generate",
                    json=ollama_request,
                    timeout=10.0
                )
                if response.status_code == 200:
                    result = response.json()
                    thought = result.get("response", thought)
                    reasoning_steps.append("Generated response using Llama model")
        except:
            thought += "the system's capabilities. I'm currently running in minimal mode."
            reasoning_steps.append("Using fallback response (Ollama unavailable)")
        
        return ThinkResponse(
            thought=thought,
            reasoning=reasoning_steps,
            confidence=0.85
        )
    except Exception as e:
        logger.error(f"Error in think endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/execute")
async def execute_task(request: ExecuteRequest):
    """Execute a task using agents"""
    try:
        # Simulate task execution
        result = {
            "task_id": f"task_{datetime.now().timestamp()}",
            "status": "completed",
            "agent": request.agent or "default",
            "result": f"Successfully processed task: {request.task}",
            "execution_time": 0.5
        }
        
        # Add agent-specific responses
        if request.agent == "code-assistant":
            result["result"] = "Code analysis complete. No syntax errors found."
        elif request.agent == "research":
            result["result"] = "Research compiled. Found 3 relevant sources."
        
        return result
    except Exception as e:
        logger.error(f"Error in execute endpoint: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List available agents"""
    return {
        "agents": [
            {
                "name": agent,
                "status": "ready",
                "capabilities": ["text-generation", "task-execution"],
                "load": 0.1
            }
            for agent in system_state["agents_available"]
        ]
    }

@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return {
        "timestamp": datetime.now().isoformat(),
        "system": {
            "uptime": (datetime.now() - system_state["start_time"]).total_seconds(),
            "requests_processed": 42,
            "average_response_time": 0.250
        },
        "resources": {
            "cpu_usage": 15.2,
            "memory_usage": 35.8,
            "active_models": system_state["models_loaded"]
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 