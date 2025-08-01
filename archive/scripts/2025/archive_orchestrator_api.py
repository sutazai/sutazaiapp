#!/usr/bin/env python3
"""
Orchestrator API Service for SutazAI v9 Enterprise
REST API interface for the Unified Agent Orchestrator
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from unified_agent_orchestrator import get_orchestrator, UnifiedAgentOrchestrator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Pydantic models
class TaskRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=2000)
    required_capabilities: List[str] = Field(..., min_items=1)
    priority: int = Field(5, ge=1, le=10)

class ChatRequest(BaseModel):
    agent_name: str = Field(..., min_length=1)
    message: Dict[str, Any]

class TaskResponse(BaseModel):
    task_id: str
    status: str
    message: str

class AgentStatusResponse(BaseModel):
    name: str
    status: str
    capabilities: List[str]
    port: int
    response_time: float
    success_count: int
    error_count: int

# FastAPI app
app = FastAPI(
    title="SutazAI Agent Orchestrator API",
    description="Unified orchestration API for containerized AI agents",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global orchestrator
orchestrator: Optional[UnifiedAgentOrchestrator] = None

@app.on_event("startup")
async def startup_event():
    """Initialize orchestrator on startup"""
    global orchestrator
    orchestrator = await get_orchestrator()
    
    # Start orchestrator in background
    asyncio.create_task(orchestrator.start())
    
    logger.info("Orchestrator API started successfully")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if orchestrator:
        await orchestrator.stop()
    logger.info("Orchestrator API shutdown complete")

# Health and status endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "service": "SutazAI Agent Orchestrator API",
        "version": "1.0.0"
    }

@app.get("/api/system/status")
async def get_system_status():
    """Get comprehensive system status"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    return orchestrator.get_system_stats()

@app.get("/api/agents", response_model=List[AgentStatusResponse])
async def get_agents():
    """Get all registered agents"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    agents_data = orchestrator.get_agents_status()
    
    # Convert to response models
    agents = []
    for agent_data in agents_data:
        agents.append(AgentStatusResponse(
            name=agent_data["name"],
            status=agent_data["status"],
            capabilities=[cap for cap in agent_data["capabilities"]],
            port=agent_data["port"],
            response_time=agent_data["response_time"],
            success_count=agent_data["success_count"],
            error_count=agent_data["error_count"]
        ))
    
    return agents

@app.get("/api/agents/{agent_name}")
async def get_agent_details(agent_name: str):
    """Get detailed information about a specific agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    agents = orchestrator.get_agents_status()
    agent = next((a for a in agents if a["name"] == agent_name), None)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    return agent

@app.get("/api/agents/by-capability/{capability}")
async def get_agents_by_capability(capability: str):
    """Get agents with specific capability"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    agents = orchestrator.get_agent_by_capability(capability)
    
    if not agents:
        return {"capability": capability, "agents": [], "message": "No agents found with this capability"}
    
    return {"capability": capability, "agents": agents, "count": len(agents)}

# Task management endpoints

@app.post("/api/tasks", response_model=TaskResponse)
async def submit_task(task_request: TaskRequest):
    """Submit a new task for execution"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        task_id = await orchestrator.submit_task(
            task_request.description,
            task_request.required_capabilities,
            task_request.priority
        )
        
        return TaskResponse(
            task_id=task_id,
            status="submitted",
            message="Task submitted successfully"
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    """Get status of a specific task"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    task = orchestrator.get_task_status(task_id)
    
    if not task:
        raise HTTPException(status_code=404, detail=f"Task {task_id} not found")
    
    return task

@app.get("/api/tasks")
async def get_all_tasks():
    """Get all tasks"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    tasks = [orchestrator.get_task_status(task_id) for task_id in orchestrator.tasks.keys()]
    return {"tasks": tasks, "count": len(tasks)}

# Communication endpoints

@app.post("/api/chat")
async def chat_with_agent(chat_request: ChatRequest):
    """Direct chat with a specific agent"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    result = await orchestrator.communicate_with_agent(
        chat_request.agent_name,
        chat_request.message
    )
    
    if not result:
        raise HTTPException(status_code=404, detail=f"Agent {chat_request.agent_name} not found")
    
    return result

@app.post("/api/broadcast")
async def broadcast_message(message: Dict[str, Any]):
    """Broadcast a message to all healthy agents"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    agents = orchestrator.get_agents_status()
    healthy_agents = [a for a in agents if a["status"] == "healthy"]
    
    results = {}
    
    for agent in healthy_agents:
        try:
            result = await orchestrator.communicate_with_agent(agent["name"], message)
            results[agent["name"]] = result
        except Exception as e:
            results[agent["name"]] = {"error": str(e)}
    
    return {
        "message": "Broadcast completed",
        "recipients": len(healthy_agents),
        "results": results
    }

# Multi-agent workflow endpoints

@app.post("/api/workflows/code-generation")
async def code_generation_workflow(request: Dict[str, Any]):
    """Execute a multi-agent code generation workflow"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        prompt = request.get("prompt", "")
        language = request.get("language", "python")
        
        # Step 1: Submit to code generation agent
        task_id = await orchestrator.submit_task(
            f"Generate {language} code: {prompt}",
            ["code_generation"],
            priority=8
        )
        
        # Wait for completion (simplified for demo)
        import asyncio
        for _ in range(30):  # Wait up to 30 seconds
            task_status = orchestrator.get_task_status(task_id)
            if task_status and task_status["status"] == "completed":
                return {
                    "workflow": "code-generation",
                    "status": "completed",
                    "task_id": task_id,
                    "result": task_status["result"]
                }
            await asyncio.sleep(1)
        
        return {
            "workflow": "code-generation",
            "status": "in_progress",
            "task_id": task_id,
            "message": "Task submitted but not yet completed"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/api/workflows/research-analysis")
async def research_analysis_workflow(request: Dict[str, Any]):
    """Execute a multi-agent research and analysis workflow"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    try:
        topic = request.get("topic", "")
        
        # Step 1: Research task
        research_task_id = await orchestrator.submit_task(
            f"Research topic: {topic}",
            ["research", "web_automation"],
            priority=7
        )
        
        # Step 2: Analysis task (will be assigned after research)
        analysis_task_id = await orchestrator.submit_task(
            f"Analyze research results for: {topic}",
            ["analysis", "reasoning"],
            priority=6
        )
        
        return {
            "workflow": "research-analysis",
            "status": "initiated",
            "research_task_id": research_task_id,
            "analysis_task_id": analysis_task_id,
            "message": "Multi-agent workflow initiated"
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# WebSocket endpoint for real-time updates

@app.websocket("/ws/updates")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time system updates"""
    await websocket.accept()
    
    try:
        while True:
            # Send system status every 5 seconds
            if orchestrator:
                stats = orchestrator.get_system_stats()
                await websocket.send_json({
                    "type": "system_status",
                    "data": stats,
                    "timestamp": time.time()
                })
            
            await asyncio.sleep(5)
            
    except WebSocketDisconnect:
        logger.info("WebSocket client disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")

# Administrative endpoints

@app.post("/api/admin/restart-agent/{agent_name}")
async def restart_agent(agent_name: str):
    """Restart a specific agent container"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    agent = next((a for a in orchestrator.agents.values() if a.name == agent_name), None)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    try:
        # Get container and restart it
        container = orchestrator.docker_client.containers.get(agent.container_id)
        container.restart()
        
        # Update agent status
        agent.status = "starting"
        
        return {
            "message": f"Agent {agent_name} restart initiated",
            "agent": agent_name,
            "container_id": agent.container_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to restart agent: {str(e)}")

@app.get("/api/admin/logs/{agent_name}")
async def get_agent_logs(agent_name: str, lines: int = 50):
    """Get logs from a specific agent container"""
    if not orchestrator:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized")
    
    agent = next((a for a in orchestrator.agents.values() if a.name == agent_name), None)
    
    if not agent:
        raise HTTPException(status_code=404, detail=f"Agent {agent_name} not found")
    
    try:
        container = orchestrator.docker_client.containers.get(agent.container_id)
        logs = container.logs(tail=lines).decode('utf-8')
        
        return {
            "agent": agent_name,
            "logs": logs,
            "lines_requested": lines
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get logs: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=9000,
        log_level="info"
    )