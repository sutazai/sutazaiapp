#!/usr/bin/env python3
"""Enhanced Task Coordinator Agent for SutazAI"""

import os
import logging
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import httpx
import json
from datetime import datetime

# Configure logging
logging.basicConfig(level=os.getenv('LOG_LEVEL', 'INFO'))
logger = logging.getLogger(__name__)

app = FastAPI(title="Task Assignment Coordinator")

# In-memory agent registry
registered_agents: Dict[str, Dict] = {}

class AgentConfig(BaseModel):
    name: str
    role: str
    capabilities: List[str]
    ollama_endpoint: str
    model: str
    max_tokens: int
    priority: str = "normal"
    agent_type: str
    description: str

class TaskRequest(BaseModel):
    task: str
    priority: str = "normal"
    required_capabilities: Optional[List[str]] = None

class HealthResponse(BaseModel):
    status: str
    agent: str
    ollama_connected: bool
    backend_connected: bool

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    ollama_url = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434')
    backend_url = os.getenv('BACKEND_URL', 'http://backend:8000')
    
    ollama_connected = False
    backend_connected = False
    
    try:
        async with httpx.AsyncClient() as client:
            # Check Ollama
            try:
                response = await client.get(f"{ollama_url}/api/tags", timeout=5.0)
                ollama_connected = response.status_code == 200
            except:
                pass
            
            # Check Backend
            try:
                response = await client.get(f"{backend_url}/health", timeout=5.0)
                backend_connected = response.status_code == 200
            except:
                pass
    except Exception as e:
        logger.error(f"Health check error: {e}")
    
    return HealthResponse(
        status="healthy" if ollama_connected and backend_connected else "degraded",
        agent="task-assignment-coordinator",
        ollama_connected=ollama_connected,
        backend_connected=backend_connected
    )

@app.post("/agents/register")
async def register_agent(agent_config: AgentConfig):
    """Register a new agent with the coordinator"""
    try:
        registered_agents[agent_config.name] = {
            "config": agent_config.dict(),
            "registered_at": datetime.now().isoformat(),
            "status": "active"
        }
        logger.info(f"Registered agent: {agent_config.name}")
        return {
            "status": "success",
            "message": f"Agent {agent_config.name} registered successfully",
            "agent_count": len(registered_agents)
        }
    except Exception as e:
        logger.error(f"Error registering agent {agent_config.name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to register agent: {str(e)}")

@app.get("/agents")
async def list_agents():
    """List all registered agents"""
    return {
        "agents": registered_agents,
        "count": len(registered_agents)
    }

@app.get("/agents/{agent_name}")
async def get_agent(agent_name: str):
    """Get specific agent details"""
    if agent_name not in registered_agents:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    return registered_agents[agent_name]

def find_best_agent(task: str, required_capabilities: Optional[List[str]] = None) -> Optional[str]:
    """Find the best agent for a given task"""
    if not registered_agents:
        return None
    
    # Simple task routing logic
    task_lower = task.lower()
    best_agent = None
    best_score = 0
    
    for agent_name, agent_data in registered_agents.items():
        config = agent_data["config"]
        score = 0
        
        # Check capabilities match
        if required_capabilities:
            matching_caps = sum(1 for cap in required_capabilities 
                              if cap in config["capabilities"])
            score += matching_caps * 10
        
        # Keyword matching for task routing
        if "ai" in task_lower or "ml" in task_lower or "model" in task_lower:
            if "senior-ai-engineer" in agent_name:
                score += 20
        elif "deploy" in task_lower or "ci" in task_lower or "cd" in task_lower:
            if "deployment-automation-master" in agent_name:
                score += 20
        elif "docker" in task_lower or "infrastructure" in task_lower or "devops" in task_lower:
            if "infrastructure-devops-manager" in agent_name:
                score += 20
        elif "ollama" in task_lower or "model" in task_lower:
            if "ollama-integration-specialist" in agent_name:
                score += 20
        elif "test" in task_lower or "qa" in task_lower or "quality" in task_lower:
            if "testing-qa-validator" in agent_name:
                score += 20
        
        # Priority bonus
        if config.get("priority") == "high":
            score += 5
        
        if score > best_score:
            best_score = score
            best_agent = agent_name
    
    return best_agent

@app.post("/assign")
async def assign_task(request: TaskRequest):
    """Assign a task to the appropriate agent"""
    logger.info(f"Received task: {request.task}")
    
    # Find best agent for the task
    assigned_agent = find_best_agent(request.task, request.required_capabilities)
    
    if not assigned_agent:
        return {
            "status": "queued",
            "task": request.task,
            "priority": request.priority,
            "message": "No suitable agent found, task queued",
            "available_agents": list(registered_agents.keys())
        }
    
    return {
        "status": "assigned",
        "task": request.task,
        "priority": request.priority,
        "assigned_agent": assigned_agent,
        "agent_details": registered_agents[assigned_agent]["config"],
        "assignment_time": datetime.now().isoformat()
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Enhanced Task Assignment Coordinator is running",
        "registered_agents": len(registered_agents),
        "agents": list(registered_agents.keys())
    }

if __name__ == "__main__":
    port = int(os.getenv('PORT', '8522'))
    uvicorn.run(app, host="0.0.0.0", port=port)