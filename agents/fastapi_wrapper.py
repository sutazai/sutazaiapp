#!/usr/bin/env python3
"""
FastAPI application wrapper for SutazAI agents
This provides HTTP endpoints for agent management and task processing
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

# Add paths for imports
current_dir = Path(__file__).parent.absolute()
sys.path.insert(0, str(current_dir))
sys.path.insert(0, str(current_dir.parent))

# Configure logging
logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)
from agents.core.utils import get_agent_name

# Global agent instance
agent_instance = None

class TaskRequest(BaseModel):
    task_id: str
    task_type: str
    data: Dict[str, Any]

class TaskResponse(BaseModel):
    status: str
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    processing_time: Optional[float] = None

## get_agent_name centralized in agents.core.utils

def import_agent_class(agent_name):
    """Dynamically import the agent class"""
    try:
        # Try to import from the specific agent directory
        agent_dir = current_dir / agent_name
        if agent_dir.exists() and (agent_dir / 'app.py').exists():
            # Add the agent directory to path
            agent_dir_str = str(agent_dir)
            if agent_dir_str not in sys.path:
                sys.path.insert(0, agent_dir_str)
            
            # Import dynamically with a unique module name
            import importlib.util
            spec = importlib.util.spec_from_file_location(f"agent_{agent_name}", agent_dir / 'app.py')
            if spec and spec.loader:
                agent_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(agent_module)
                
                # Find the agent class (should end with 'Agent')
                for attr_name in dir(agent_module):
                    attr = getattr(agent_module, attr_name)
                    if (isinstance(attr, type) and 
                        attr_name.endswith('Agent') and 
                        attr_name != 'BaseAgentV2' and
                        attr_name != 'BaseAgent'):
                        logger.info(f"Found agent class: {attr_name}")
                        return attr
        
        # Fallback to simple FastAPI agent for containers without proper agent implementation
        logger.warning(f"Could not find specific agent class for {agent_name}, creating simple FastAPI agent")
        return create_simple_agent_class(agent_name)
        
    except Exception as e:
        logger.error(f"Error importing agent class: {e}")
        return create_simple_agent_class(agent_name)

def create_simple_agent_class(agent_name):
    """Create a simple agent class for containers without proper implementation"""
    
    class SimpleAgent:
        def __init__(self):
            self.agent_name = agent_name
            self.status = "active"
            self.tasks_processed = 0
            self.tasks_failed = 0
            
        async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
            """Simple task processing"""
            try:
                task_type = task.get("type", "unknown")
                
                if task_type == "health":
                    return {"status": "healthy", "agent": self.agent_name}
                
                # Simple echo response
                self.tasks_processed += 1
                return {
                    "status": "success",
                    "message": f"Task processed by {self.agent_name}",
                    "task_id": task.get("id", "unknown"),
                    "agent": self.agent_name
                }
                
            except Exception as e:
                self.tasks_failed += 1
                logger.error(f"Error processing task: {e}")
                return {
                    "status": "error",
                    "error": str(e),
                    "agent": self.agent_name
                }
        
        async def health_check(self) -> Dict[str, Any]:
            """Health check"""
            return {
                "agent_name": self.agent_name,
                "status": self.status,
                "healthy": True,
                "tasks_processed": self.tasks_processed,
                "tasks_failed": self.tasks_failed
            }
    
    return SimpleAgent

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global agent_instance
    
    try:
        # Initialize agent
        agent_name = get_agent_name()
        logger.info(f"Initializing agent: {agent_name}")
        
        AgentClass = import_agent_class(agent_name)
        agent_instance = AgentClass()
        
        logger.info(f"Agent {agent_name} initialized successfully")
        yield
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        yield
    finally:
        logger.info("Shutting down agent")
        agent_instance = None

# Create FastAPI app
app = FastAPI(
    title="SutazAI Agent API",
    description="HTTP API for SutazAI agent management and task processing",
    version="1.0.0",
    lifespan=lifespan
)

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
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/status")
async def get_status():
    """Get agent status"""
    try:
        if agent_instance:
            status_info = {
                "agent_name": getattr(agent_instance, 'agent_name', get_agent_name()),
                "status": getattr(agent_instance, 'status', 'active'),
                "tasks_processed": getattr(agent_instance, 'tasks_processed', 0),
                "tasks_failed": getattr(agent_instance, 'tasks_failed', 0)
            }
            return status_info
        else:
            return {
                "agent_name": get_agent_name(),
                "status": "initializing",
                "message": "Agent not yet initialized"
            }
    except Exception as e:
        logger.error(f"Status check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/task")
async def process_task(task_request: TaskRequest, background_tasks: BackgroundTasks):
    """Process a task"""
    try:
        if not agent_instance:
            raise HTTPException(status_code=503, detail="Agent not initialized")
        
        task_data = {
            "id": task_request.task_id,
            "type": task_request.task_type,
            **task_request.data
        }
        
        if hasattr(agent_instance, 'process_task'):
            result = await agent_instance.process_task(task_data)
        else:
            # Fallback processing
            result = {
                "status": "success",
                "message": f"Task {task_request.task_id} processed",
                "agent": get_agent_name()
            }
        
        return TaskResponse(
            status="completed",
            result=result
        )
        
    except Exception as e:
        logger.error(f"Task processing error: {e}")
        return TaskResponse(
            status="failed",
            error=str(e)
        )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": f"SutazAI Agent API - {get_agent_name()}",
        "version": "1.0.0",
        "agent": get_agent_name()
    }

if __name__ == "__main__":
    # Run with uvicorn
    port = int(os.getenv("PORT", "8080"))
    host = os.getenv("HOST", "0.0.0.0")
    
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=False,
        log_level=os.getenv("LOG_LEVEL", "info").lower()
    )
