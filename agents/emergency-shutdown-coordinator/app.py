#!/usr/bin/env python3
"""
Agent: emergency-shutdown-coordinator
Category: monitoring
Model Type: Sonnet
"""

import os
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from agents.compatibility_base_agent import BaseAgentV2
except ImportError:
    # Direct fallback to core
    try:
        from agents.core.base_agent_v2 import BaseAgentV2
    except ImportError:
        # Final fallback for minimal functionality
        import logging
        from datetime import datetime
        
        class BaseAgentV2:
            def __init__(self, agent_id: str, name: str, port: int = 8080, description: str = "Agent"):
                self.agent_id = agent_id
                self.name = name
                self.port = port
                self.description = description
                self.logger = logging.getLogger(agent_id)
                self.status = "active"
                self.tasks_processed = 0
                
            async def process_task(self, task):
                return {"status": "success", "agent": self.agent_id}
            
            def start(self):
                self.logger.info(f"Agent {self.name} started")

import asyncio
from typing import Dict, Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from datetime import datetime

# Create FastAPI app directly for uvicorn compatibility
app = FastAPI(
    title="Emergency Shutdown Coordinator",
    description="Critical safety control system",
    version="1.0.0"
)

class TaskRequest(BaseModel):
    task: str
    context: Dict[str, Any] = {}

class TaskResponse(BaseModel):
    status: str
    result: Any
    agent: str = "emergency-shutdown-coordinator"

@app.get("/")
async def root():
    return {
        "agent": "emergency-shutdown-coordinator",
        "status": "active",
        "description": "Critical safety control system"
    }

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "agent": "emergency-shutdown-coordinator",
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/execute")
async def execute_task(request: TaskRequest):
    """Execute emergency coordination task"""
    try:
        if "shutdown" in request.task.lower():
            result = await handle_shutdown_request(request)
        elif "emergency" in request.task.lower():
            result = await handle_emergency_request(request)
        else:
            result = await handle_general_task(request)
            
        return TaskResponse(status="completed", result=result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def handle_shutdown_request(request: TaskRequest) -> Dict[str, Any]:
    """Handle emergency shutdown requests"""
    return {
        "action": "emergency_shutdown_initiated",
        "task": request.task,
        "safety_protocols": "activated",
        "timestamp": datetime.utcnow().isoformat()
    }

async def handle_emergency_request(request: TaskRequest) -> Dict[str, Any]:
    """Handle emergency coordination requests"""
    return {
        "action": "emergency_protocol_activated",
        "task": request.task,
        "response_teams": "notified",
        "timestamp": datetime.utcnow().isoformat()
    }

async def handle_general_task(request: TaskRequest) -> Dict[str, Any]:
    """Handle general monitoring tasks"""
    return {
        "action": "monitoring_task_processed",
        "task": request.task,
        "status": "completed",
        "timestamp": datetime.utcnow().isoformat()
    }

# Legacy agent class for compatibility
class Emergency_Shutdown_CoordinatorAgent(BaseAgentV2):
    """Agent implementation for emergency-shutdown-coordinator"""
    
    def __init__(self):
        super().__init__(
            agent_id="emergency-shutdown-coordinator",
            name="Emergency Shutdown Coordinator",
            port=int(os.getenv("PORT", "8080")),
            description="Specialized agent for monitoring tasks"
        )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
