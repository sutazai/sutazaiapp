#!/usr/bin/env python3
"""
Container startup script for SutazAI agents
Runs both HTTP server and agent logic
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

# Add paths for imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
from agents.core.utils import get_agent_name

## get_agent_name centralized in agents.core.utils

def load_agent_instance():
    """Load the agent instance from the local app.py"""
    agent_name = get_agent_name()
    
    try:
        # Import the local app.py file
        import importlib.util
        app_py_path = Path('/app/app.py')
        
        if app_py_path.exists():
            spec = importlib.util.spec_from_file_location("agent_app", app_py_path)
            if spec and spec.loader:
                agent_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(agent_module)
                
                # Find the agent class
                for attr_name in dir(agent_module):
                    attr = getattr(agent_module, attr_name)
                    if (isinstance(attr, type) and 
                        attr_name.endswith('Agent') and 
                        attr_name not in ['BaseAgentV2', 'BaseAgent']):
                        logger.info(f"Found agent class: {attr_name}")
                        return attr()
        
        # Fallback: create a simple agent
        logger.warning(f"No agent class found, creating simple agent for {agent_name}")
        
        class SimpleAgent:
            def __init__(self):
                self.agent_name = agent_name
                self.status = "active"
                self.tasks_processed = 0
                self.tasks_failed = 0
                
            async def process_task(self, task):
                return {"status": "success", "agent": self.agent_name}
            
            async def health_check(self):
                return {
                    "agent_name": self.agent_name,
                    "status": self.status,
                    "healthy": True,
                    "tasks_processed": self.tasks_processed,
                    "tasks_failed": self.tasks_failed
                }
        
        return SimpleAgent()
        
    except Exception as e:
        logger.error(f"Error loading agent: {e}")
        return None

# Global agent instance
agent_instance = None

# Create FastAPI app
app = FastAPI(
    title=f"SutazAI Agent - {get_agent_name()}",
    version="1.0.0"
)

@app.on_event("startup")
async def startup_event():
    """Initialize agent on startup"""
    global agent_instance
    agent_instance = load_agent_instance()
    logger.info(f"Agent {get_agent_name()} initialized")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        if agent_instance and hasattr(agent_instance, 'health_check'):
            return await agent_instance.health_check()
        else:
            return {
                "agent_name": get_agent_name(),
                "status": "active",
                "healthy": True,
                "message": "Simple health check"
            }
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return JSONResponse(
            status_code=500,
            content={"healthy": False, "error": str(e)}
        )

@app.get("/status")
async def get_status():
    """Get agent status"""
    agent_name = get_agent_name()
    if agent_instance:
        return {
            "agent_name": getattr(agent_instance, 'agent_name', agent_name),
            "status": getattr(agent_instance, 'status', 'active'),
            "tasks_processed": getattr(agent_instance, 'tasks_processed', 0),
            "tasks_failed": getattr(agent_instance, 'tasks_failed', 0)
        }
    else:
        return {
            "agent_name": agent_name,
            "status": "error",
            "message": "Agent not initialized"
        }

@app.post("/task")
async def process_task(task_data: dict):
    """Process a task"""
    try:
        if not agent_instance:
            return JSONResponse(
                status_code=503,
                content={"status": "error", "error": "Agent not initialized"}
            )
        
        if hasattr(agent_instance, 'process_task'):
            result = await agent_instance.process_task(task_data)
        else:
            result = {"status": "success", "message": "Task processed"}
        
        return {"status": "completed", "result": result}
        
    except Exception as e:
        logger.error(f"Task processing error: {e}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"SutazAI Agent - {get_agent_name()}",
        "version": "1.0.0",
        "agent": get_agent_name()
    }

if __name__ == "__main__":
    # Run with uvicorn
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    
    logger.info(f"Starting agent {get_agent_name()} on {host}:{port}")
    
    uvicorn.run(
        "container_startup:app",
        host=host,
        port=port,
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
