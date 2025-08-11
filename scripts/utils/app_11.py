#!/usr/bin/env python3
"""
Multi-Agent Coordinator - Orchestrates workflows between multiple agents
"""
import os
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import uvicorn
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentTask(BaseModel):
    task_id: str
    agent_id: str
    task_data: Dict[str, Any]
    priority: int = Field(default=1, description="Task priority (1-10)")
    timeout: int = Field(default=300, description="Task timeout in seconds")
    dependencies: List[str] = Field(default_factory=list, description="List of task IDs this task depends on")

class WorkflowRequest(BaseModel):
    workflow_id: str
    name: str
    description: str
    tasks: List[AgentTask]
    max_concurrent_tasks: int = Field(default=5, description="Maximum concurrent tasks")

class AgentStatus(BaseModel):
    agent_id: str
    status: str  # available, busy, offline
    load: float
    last_seen: datetime

class MultiAgentCoordinator:
    def __init__(self):
        self.redis_client = None
        self.agent_registry = {}
        self.active_workflows = {}
        self.task_queue = asyncio.Queue()
        self.agent_statuses = {}
        
    async def initialize(self):
        """Initialize Redis connection and agent registry"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis successfully")
            
            # Start background tasks
            asyncio.create_task(self.workflow_processor())
            asyncio.create_task(self.agent_health_monitor())
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def workflow_processor(self):
        """Background task to process workflows"""
        while True:
            try:
                # Process any pending workflows
                for workflow_id, workflow in list(self.active_workflows.items()):
                    await self.process_workflow(workflow_id, workflow)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                logger.error(f"Error in workflow processor: {e}")
                await asyncio.sleep(10)
    
    async def agent_health_monitor(self):
        """Monitor agent health and availability"""
        while True:
            try:
                # Check agent health
                for agent_id in list(self.agent_statuses.keys()):
                    await self.check_agent_health(agent_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in agent health monitor: {e}")
                await asyncio.sleep(60)
    
    async def register_agent(self, agent_id: str, agent_info: Dict[str, Any]):
        """Register an agent with the coordinator"""
        self.agent_registry[agent_id] = agent_info
        self.agent_statuses[agent_id] = AgentStatus(
            agent_id=agent_id,
            status="available",
            load=0.0,
            last_seen=datetime.utcnow()
        )
        
        # Store in Redis
        if self.redis_client:
            await self.redis_client.hset(
                "agent_registry", 
                agent_id, 
                json.dumps(agent_info)
            )
        
        logger.info(f"Registered agent: {agent_id}")
    
    async def submit_workflow(self, workflow: WorkflowRequest) -> Dict[str, Any]:
        """Submit a new workflow for processing"""
        try:
            # Validate workflow
            if not workflow.tasks:
                raise HTTPException(status_code=400, detail="Workflow must contain at least one task")
            
            # Store workflow
            workflow_data = {
                "workflow_id": workflow.workflow_id,
                "name": workflow.name,
                "description": workflow.description,
                "tasks": [task.dict() for task in workflow.tasks],
                "max_concurrent_tasks": workflow.max_concurrent_tasks,
                "status": "pending",
                "created_at": datetime.utcnow().isoformat(),
                "completed_tasks": [],
                "failed_tasks": []
            }
            
            self.active_workflows[workflow.workflow_id] = workflow_data
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.hset(
                    "active_workflows",
                    workflow.workflow_id,
                    json.dumps(workflow_data)
                )
            
            logger.info(f"Submitted workflow: {workflow.workflow_id}")
            
            return {
                "workflow_id": workflow.workflow_id,
                "status": "submitted",
                "message": "Workflow submitted for processing"
            }
            
        except Exception as e:
            logger.error(f"Error submitting workflow {workflow.workflow_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def process_workflow(self, workflow_id: str, workflow_data: Dict[str, Any]):
        """Process a specific workflow"""
        try:
            if workflow_data["status"] in ["completed", "failed"]:
                return
            
            # Find tasks ready to execute (dependencies satisfied)
            ready_tasks = []
            for task in workflow_data["tasks"]:
                if task["task_id"] not in workflow_data["completed_tasks"] and \
                   task["task_id"] not in workflow_data["failed_tasks"]:
                    # Check if all dependencies are satisfied
                    dependencies_met = all(
                        dep_id in workflow_data["completed_tasks"] 
                        for dep_id in task.get("dependencies", [])
                    )
                    if dependencies_met:
                        ready_tasks.append(task)
            
            # Execute ready tasks (respecting concurrency limit)
            currently_running = len([
                task for task in workflow_data["tasks"]
                if task.get("status") == "running"
            ])
            
            slots_available = workflow_data["max_concurrent_tasks"] - currently_running
            
            for i, task in enumerate(ready_tasks[:slots_available]):
                await self.execute_task(workflow_id, task)
            
        except Exception as e:
            logger.error(f"Error processing workflow {workflow_id}: {e}")
    
    async def execute_task(self, workflow_id: str, task: Dict[str, Any]):
        """Execute a single task"""
        try:
            task_id = task["task_id"]
            agent_id = task["agent_id"]
            
            # Find best available agent
            if agent_id not in self.agent_statuses:
                logger.error(f"Agent {agent_id} not found in registry")
                return
            
            agent_status = self.agent_statuses[agent_id]
            if agent_status.status != "available":
                logger.warning(f"Agent {agent_id} is not available (status: {agent_status.status})")
                return
            
            # Mark task as running
            task["status"] = "running"
            task["started_at"] = datetime.utcnow().isoformat()
            
            # Get agent endpoint
            agent_info = self.agent_registry.get(agent_id, {})
            agent_endpoint = agent_info.get("endpoint", f"http://{agent_id}:8080")
            
            # Send task to agent
            async with httpx.AsyncClient(timeout=task.get("timeout", 300)) as client:
                response = await client.post(
                    f"{agent_endpoint}/execute",
                    json=task["task_data"]
                )
                
                if response.status_code == 200:
                    result = response.json()
                    task["status"] = "completed"
                    task["completed_at"] = datetime.utcnow().isoformat()
                    task["result"] = result
                    
                    # Update workflow
                    workflow = self.active_workflows[workflow_id]
                    workflow["completed_tasks"].append(task_id)
                    
                    logger.info(f"Task {task_id} completed successfully")
                    
                else:
                    raise Exception(f"Agent returned status {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Error executing task {task.get('task_id', 'unknown')}: {e}")
            
            # Mark task as failed
            task["status"] = "failed"
            task["failed_at"] = datetime.utcnow().isoformat()
            task["error"] = str(e)
            
            # Update workflow
            workflow = self.active_workflows[workflow_id] 
            workflow["failed_tasks"].append(task["task_id"])
    
    async def check_agent_health(self, agent_id: str):
        """Check health of a specific agent"""
        try:
            agent_info = self.agent_registry.get(agent_id, {})
            agent_endpoint = agent_info.get("endpoint", f"http://{agent_id}:8080")
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{agent_endpoint}/health")
                
                if response.status_code == 200:
                    health_data = response.json()
                    self.agent_statuses[agent_id].status = "available"
                    self.agent_statuses[agent_id].load = health_data.get("load", 0.0)
                    self.agent_statuses[agent_id].last_seen = datetime.utcnow()
                else:
                    self.agent_statuses[agent_id].status = "offline"
                    
        except Exception as e:
            logger.warning(f"Agent {agent_id} health check failed: {e}")
            self.agent_statuses[agent_id].status = "offline"
    
    async def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """Get status of a specific workflow"""
        if workflow_id not in self.active_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow = self.active_workflows[workflow_id]
        
        # Calculate completion percentage
        total_tasks = len(workflow["tasks"])
        completed_tasks = len(workflow["completed_tasks"])
        failed_tasks = len(workflow["failed_tasks"])
        
        completion_percentage = (completed_tasks / total_tasks * 100) if total_tasks > 0 else 0
        
        # Determine overall status
        if failed_tasks > 0 and completed_tasks + failed_tasks == total_tasks:
            overall_status = "failed"
        elif completed_tasks == total_tasks:
            overall_status = "completed"
        elif completed_tasks > 0 or failed_tasks > 0:
            overall_status = "in_progress"
        else:
            overall_status = "pending"
        
        return {
            "workflow_id": workflow_id,
            "status": overall_status,
            "completion_percentage": completion_percentage,
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "created_at": workflow["created_at"]
        }

# Global coordinator instance
coordinator = MultiAgentCoordinator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await coordinator.initialize()
    yield
    # Shutdown
    if coordinator.redis_client:
        await coordinator.redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Multi-Agent Coordinator",
    description="Orchestrates workflows between multiple agents",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "multi-agent-coordinator",
        "timestamp": datetime.utcnow().isoformat(),
        "registered_agents": len(coordinator.agent_registry),
        "active_workflows": len(coordinator.active_workflows)
    }

@app.post("/register_agent")
async def register_agent(agent_id: str, agent_info: Dict[str, Any]):
    """Register a new agent"""
    await coordinator.register_agent(agent_id, agent_info)
    return {"message": f"Agent {agent_id} registered successfully"}

@app.post("/submit_workflow")
async def submit_workflow(workflow: WorkflowRequest):
    """Submit a new workflow"""
    return await coordinator.submit_workflow(workflow)

@app.get("/workflow/{workflow_id}/status")
async def get_workflow_status(workflow_id: str):
    """Get workflow status"""
    return await coordinator.get_workflow_status(workflow_id)

@app.get("/agents")
async def list_agents():
    """List all registered agents"""
    agents = []
    for agent_id, status in coordinator.agent_statuses.items():
        agent_info = coordinator.agent_registry.get(agent_id, {})
        agents.append({
            "agent_id": agent_id,
            "status": status.status,
            "load": status.load,
            "last_seen": status.last_seen.isoformat(),
            "capabilities": agent_info.get("capabilities", [])
        })
    return {"agents": agents}

@app.get("/workflows")
async def list_workflows():
    """List all workflows"""
    workflows = []
    for workflow_id in coordinator.active_workflows:
        status = await coordinator.get_workflow_status(workflow_id)
        workflows.append(status)
    return {"workflows": workflows}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": "multi-agent-coordinator",
        "status": "running",
        "description": "Multi-Agent Workflow Coordinator"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8587"))
    uvicorn.run(app, host="0.0.0.0", port=port)