#!/usr/bin/env python3
"""
Task Assignment Coordinator - Distributes tasks to agents based on capabilities and load
"""
import os
import asyncio
import json
import logging
import hashlib
from datetime import datetime, timedelta
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

class Task(BaseModel):
    task_id: str
    task_type: str
    description: str
    required_capabilities: List[str]
    priority: int = Field(default=5, description="Task priority (1-10)")
    estimated_duration: int = Field(default=300, description="Estimated duration in seconds")
    max_retries: int = Field(default=3, description="Maximum retry attempts")
    data: Dict[str, Any] = Field(default_factory=dict)
    deadline: Optional[datetime] = None

class AgentCapability(BaseModel):
    agent_id: str
    name: str
    capabilities: List[str]
    current_load: float = 0.0
    max_concurrent_tasks: int = 5
    performance_score: float = 1.0
    availability: str = "available"  # available, busy, offline
    last_seen: datetime = Field(default_factory=datetime.utcnow)

class TaskAssignment(BaseModel):
    assignment_id: str
    task_id: str
    agent_id: str
    assigned_at: datetime = Field(default_factory=datetime.utcnow)
    status: str = "assigned"  # assigned, in_progress, completed, failed
    estimated_completion: datetime
    actual_completion: Optional[datetime] = None
    retry_count: int = 0

class TaskAssignmentCoordinator:
    def __init__(self):
        self.redis_client = None
        self.task_queue = []
        self.agent_registry = {}
        self.active_assignments = {}
        self.completed_assignments = []
        self.failed_assignments = []
        
        # Assignment settings
        self.load_balancing_algorithm = os.getenv("LOAD_BALANCING_ALGORITHM", "weighted_round_robin")
        self.task_priority_enabled = os.getenv("TASK_PRIORITY_ENABLED", "true").lower() == "true"
        self.capability_matching = os.getenv("AGENT_CAPABILITY_MATCHING", "true").lower() == "true"
        self.performance_based = os.getenv("PERFORMANCE_BASED_ASSIGNMENT", "true").lower() == "true"
        self.max_queue_size = int(os.getenv("TASK_QUEUE_SIZE", "1000"))
        
    async def initialize(self):
        """Initialize Redis connection and start assignment processes"""
        try:
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            self.redis_client = redis.from_url(redis_url)
            await self.redis_client.ping()
            logger.info("Task Coordinator: Connected to Redis successfully")
            
            # Load existing agent registry from Redis
            await self.load_agent_registry()
            
            # Start background tasks
            asyncio.create_task(self.task_assignment_processor())
            asyncio.create_task(self.assignment_monitor())
            asyncio.create_task(self.agent_health_monitor())
            asyncio.create_task(self.performance_tracker())
            
            logger.info("Task Coordinator: All processes started")
            
        except Exception as e:
            logger.error(f"Task Coordinator: Failed to initialize: {e}")
            raise
    
    async def task_assignment_processor(self):
        """Main task assignment processing loop"""
        while True:
            try:
                if self.task_queue:
                    # Sort tasks by priority and deadline
                    self.task_queue.sort(key=lambda t: (-t.priority, t.deadline or datetime.max))
                    
                    # Assign tasks to available agents
                    for task in list(self.task_queue):
                        agent_id = await self.find_best_agent(task)
                        if agent_id:
                            await self.assign_task(task, agent_id)
                            self.task_queue.remove(task)
                
                await asyncio.sleep(5)  # Process every 5 seconds
                
            except Exception as e:
                logger.error(f"Task Coordinator: Error in assignment processor: {e}")
                await asyncio.sleep(15)
    
    async def assignment_monitor(self):
        """Monitor active task assignments"""
        while True:
            try:
                current_time = datetime.utcnow()
                overdue_assignments = []
                
                for assignment_id, assignment in self.active_assignments.items():
                    # Check for overdue assignments
                    if assignment.estimated_completion < current_time and assignment.status == "assigned":
                        overdue_assignments.append(assignment_id)
                    
                    # Check assignment status with agent
                    await self.check_assignment_status(assignment_id, assignment)
                
                # Handle overdue assignments
                for assignment_id in overdue_assignments:
                    await self.handle_overdue_assignment(assignment_id)
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Task Coordinator: Error in assignment monitor: {e}")
                await asyncio.sleep(60)
    
    async def agent_health_monitor(self):
        """Monitor agent health and availability"""
        while True:
            try:
                for agent_id, agent_info in self.agent_registry.items():
                    await self.check_agent_health(agent_id, agent_info)
                
                await asyncio.sleep(60)  # Health check every minute
                
            except Exception as e:
                logger.error(f"Task Coordinator: Error in health monitor: {e}")
                await asyncio.sleep(120)
    
    async def performance_tracker(self):
        """Track and update agent performance scores"""
        while True:
            try:
                await self.update_performance_scores()
                await self.optimize_assignment_strategies()
                
                await asyncio.sleep(300)  # Update every 5 minutes
                
            except Exception as e:
                logger.error(f"Task Coordinator: Error in performance tracker: {e}")
                await asyncio.sleep(600)
    
    async def load_agent_registry(self):
        """Load agent registry from Redis"""
        try:
            if self.redis_client:
                # Load from orchestrator's agent registry
                agent_data = await self.redis_client.hgetall("orchestrator:agents")
                for agent_id, agent_json in agent_data.items():
                    agent_info = json.loads(agent_json)
                    await self.register_agent(agent_info)
        except Exception as e:
            logger.warning(f"Task Coordinator: Could not load agent registry: {e}")
    
    async def register_agent(self, agent_data: Dict[str, Any]):
        """Register or update an agent in the registry"""
        try:
            agent_id = agent_data.get("agent_id")
            if not agent_id:
                return
            
            agent_capability = AgentCapability(
                agent_id=agent_id,
                name=agent_data.get("name", agent_id),
                capabilities=agent_data.get("capabilities", []),
                current_load=agent_data.get("load", 0.0),
                max_concurrent_tasks=agent_data.get("max_concurrent_tasks", 5),
                performance_score=agent_data.get("performance_score", 1.0),
                availability=agent_data.get("status", "available")
            )
            
            self.agent_registry[agent_id] = agent_capability
            
            logger.info(f"Task Coordinator: Registered agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Task Coordinator: Error registering agent: {e}")
    
    async def submit_task(self, task: Task) -> Dict[str, Any]:
        """Submit a new task for assignment"""
        try:
            # Validate task
            if not task.required_capabilities and self.capability_matching:
                raise HTTPException(status_code=400, detail="Task must specify required capabilities")
            
            # Check queue size
            if len(self.task_queue) >= self.max_queue_size:
                raise HTTPException(status_code=503, detail="Task queue is full")
            
            # Add to queue
            self.task_queue.append(task)
            
            # Store in Redis
            if self.redis_client:
                await self.redis_client.lpush(
                    "task_coordinator:queue",
                    json.dumps(task.dict(), default=str)
                )
            
            logger.info(f"Task Coordinator: Submitted task {task.task_id}")
            
            return {
                "task_id": task.task_id,
                "status": "queued",
                "queue_position": len(self.task_queue),
                "estimated_assignment_time": self.estimate_assignment_time(task)
            }
            
        except Exception as e:
            logger.error(f"Task Coordinator: Error submitting task {task.task_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def find_best_agent(self, task: Task) -> Optional[str]:
        """Find the best agent for a task"""
        try:
            available_agents = [
                agent for agent in self.agent_registry.values()
                if agent.availability == "available" and agent.current_load < 1.0
            ]
            
            if not available_agents:
                return None
            
            # Filter by capabilities if required
            if self.capability_matching and task.required_capabilities:
                capable_agents = []
                for agent in available_agents:
                    if all(cap in agent.capabilities for cap in task.required_capabilities):
                        capable_agents.append(agent)
                available_agents = capable_agents
            
            if not available_agents:
                return None
            
            # Apply assignment algorithm
            if self.load_balancing_algorithm == "weighted_round_robin":
                return self.weighted_round_robin_selection(available_agents, task)
            elif self.load_balancing_algorithm == "least_loaded":
                return self.least_loaded_selection(available_agents)
            elif self.load_balancing_algorithm == "performance_based":
                return self.performance_based_selection(available_agents, task)
            else:
                # Default to round robin
                return available_agents[0].agent_id
            
        except Exception as e:
            logger.error(f"Task Coordinator: Error finding best agent for task {task.task_id}: {e}")
            return None
    
    def weighted_round_robin_selection(self, agents: List[AgentCapability], task: Task) -> str:
        """Select agent using weighted round robin based on performance and load"""
        best_agent = None
        best_score = -1
        
        for agent in agents:
            # Calculate composite score
            load_factor = 1.0 - agent.current_load
            performance_factor = agent.performance_score
            availability_factor = 1.0 if agent.current_load < 0.8 else 0.5
            
            composite_score = (load_factor * 0.4) + (performance_factor * 0.4) + (availability_factor * 0.2)
            
            if composite_score > best_score:
                best_score = composite_score
                best_agent = agent
        
        return best_agent.agent_id if best_agent else agents[0].agent_id
    
    def least_loaded_selection(self, agents: List[AgentCapability]) -> str:
        """Select the least loaded agent"""
        return min(agents, key=lambda a: a.current_load).agent_id
    
    def performance_based_selection(self, agents: List[AgentCapability], task: Task) -> str:
        """Select agent based on performance for similar tasks"""
        if not self.performance_based:
            return self.least_loaded_selection(agents)
        
        # For now, use performance score
        best_agent = max(agents, key=lambda a: a.performance_score - a.current_load)
        return best_agent.agent_id
    
    async def assign_task(self, task: Task, agent_id: str):
        """Assign a task to a specific agent"""
        try:
            assignment_id = f"assign_{task.task_id}_{agent_id}_{int(datetime.utcnow().timestamp())}"
            
            estimated_completion = datetime.utcnow() + timedelta(seconds=task.estimated_duration)
            
            assignment = TaskAssignment(
                assignment_id=assignment_id,
                task_id=task.task_id,
                agent_id=agent_id,
                estimated_completion=estimated_completion
            )
            
            # Update agent load
            if agent_id in self.agent_registry:
                agent = self.agent_registry[agent_id]
                agent.current_load += 1.0 / agent.max_concurrent_tasks
                if agent.current_load >= 1.0:
                    agent.availability = "busy"
            
            # Send task to agent
            agent_info = self.agent_registry[agent_id]
            endpoint = f"http://{agent_id}:8080"
            
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{endpoint}/assign_task",
                    json={
                        "assignment_id": assignment_id,
                        "task": task.dict(default=str),
                        "coordinator_callback": f"http://task-assignment-coordinator:8551/assignment_callback"
                    }
                )
                
                if response.status_code == 200:
                    assignment.status = "in_progress"
                    self.active_assignments[assignment_id] = assignment
                    
                    logger.info(f"Task Coordinator: Assigned task {task.task_id} to agent {agent_id}")
                    
                    # Store in Redis
                    if self.redis_client:
                        await self.redis_client.hset(
                            "task_coordinator:assignments",
                            assignment_id,
                            json.dumps(assignment.dict(), default=str)
                        )
                else:
                    raise Exception(f"Agent returned status {response.status_code}")
                    
        except Exception as e:
            logger.error(f"Task Coordinator: Error assigning task {task.task_id} to {agent_id}: {e}")
            # Return task to queue
            self.task_queue.append(task)
    
    async def check_assignment_status(self, assignment_id: str, assignment: TaskAssignment):
        """Check the status of an assignment with the agent"""
        try:
            agent_id = assignment.agent_id
            endpoint = f"http://{agent_id}:8080"
            
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{endpoint}/assignment_status/{assignment_id}")
                
                if response.status_code == 200:
                    status_data = response.json()
                    assignment.status = status_data.get("status", assignment.status)
                    
                    if assignment.status == "completed":
                        await self.handle_completed_assignment(assignment_id, status_data)
                    elif assignment.status == "failed":
                        await self.handle_failed_assignment(assignment_id, status_data)
                        
        except Exception as e:
            logger.warning(f"Task Coordinator: Could not check status for assignment {assignment_id}: {e}")
    
    async def handle_completed_assignment(self, assignment_id: str, result_data: Dict[str, Any]):
        """Handle a completed assignment"""
        if assignment_id in self.active_assignments:
            assignment = self.active_assignments[assignment_id]
            assignment.status = "completed"
            assignment.actual_completion = datetime.utcnow()
            
            # Update agent load and performance
            agent = self.agent_registry.get(assignment.agent_id)
            if agent:
                agent.current_load = max(0, agent.current_load - 1.0 / agent.max_concurrent_tasks)
                agent.performance_score = min(10.0, agent.performance_score * 1.02)
                if agent.current_load < 1.0:
                    agent.availability = "available"
            
            # Move to completed
            self.completed_assignments.append(assignment)
            del self.active_assignments[assignment_id]
            
            logger.info(f"Task Coordinator: Assignment {assignment_id} completed successfully")
    
    async def handle_failed_assignment(self, assignment_id: str, error_data: Dict[str, Any]):
        """Handle a failed assignment"""
        if assignment_id in self.active_assignments:
            assignment = self.active_assignments[assignment_id]
            assignment.retry_count += 1
            
            # Update agent performance
            agent = self.agent_registry.get(assignment.agent_id)
            if agent:
                agent.current_load = max(0, agent.current_load - 1.0 / agent.max_concurrent_tasks)
                agent.performance_score = max(0.1, agent.performance_score * 0.98)
                if agent.current_load < 1.0:
                    agent.availability = "available"
            
            # Retry or fail permanently
            if assignment.retry_count < 3:  # Max retries from task
                # Find original task and re-queue
                task_data = error_data.get("task")
                if task_data:
                    task = Task(**task_data)
                    self.task_queue.append(task)
                    logger.info(f"Task Coordinator: Re-queuing task {task.task_id} for retry")
            else:
                assignment.status = "failed"
                self.failed_assignments.append(assignment)
                logger.error(f"Task Coordinator: Assignment {assignment_id} failed permanently")
            
            del self.active_assignments[assignment_id]
    
    async def handle_overdue_assignment(self, assignment_id: str):
        """Handle an overdue assignment"""
        assignment = self.active_assignments.get(assignment_id)
        if assignment:
            logger.warning(f"Task Coordinator: Assignment {assignment_id} is overdue")
            # Could implement timeout handling here
    
    async def check_agent_health(self, agent_id: str, agent_info: AgentCapability):
        """Check health of a specific agent"""
        try:
            endpoint = f"http://{agent_id}:8080"
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.get(f"{endpoint}/health")
                
                if response.status_code == 200:
                    health_data = response.json()
                    agent_info.availability = "available" if agent_info.current_load < 1.0 else "busy"
                    agent_info.last_seen = datetime.utcnow()
                else:
                    agent_info.availability = "offline"
                    
        except Exception as e:
            logger.warning(f"Task Coordinator: Agent {agent_id} health check failed: {e}")
            agent_info.availability = "offline"
    
    async def update_performance_scores(self):
        """Update agent performance scores based on completed assignments"""
        # This would analyze completion rates, accuracy, etc.
        pass
    
    async def optimize_assignment_strategies(self):
        """Optimize assignment strategies based on performance data"""
        pass
    
    def estimate_assignment_time(self, task: Task) -> str:
        """Estimate when a task will be assigned"""
        # Simple estimation based on queue position and agent availability
        available_agents = len([a for a in self.agent_registry.values() if a.availability == "available"])
        if available_agents == 0:
            return "waiting for available agents"
        
        queue_position = len(self.task_queue)
        estimated_minutes = (queue_position / available_agents) * 2  # Rough estimate
        
        return f"~{int(estimated_minutes)} minutes"
    
    async def get_assignment_statistics(self) -> Dict[str, Any]:
        """Get assignment statistics"""
        total_assignments = len(self.completed_assignments) + len(self.failed_assignments) + len(self.active_assignments)
        
        return {
            "queued_tasks": len(self.task_queue),
            "active_assignments": len(self.active_assignments),
            "completed_assignments": len(self.completed_assignments),
            "failed_assignments": len(self.failed_assignments),
            "total_assignments": total_assignments,
            "success_rate": len(self.completed_assignments) / max(1, total_assignments) * 100,
            "registered_agents": len(self.agent_registry),
            "available_agents": len([a for a in self.agent_registry.values() if a.availability == "available"])
        }

# Global coordinator instance
task_coordinator = TaskAssignmentCoordinator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await task_coordinator.initialize()
    yield
    # Shutdown
    if task_coordinator.redis_client:
        await task_coordinator.redis_client.close()

# Create FastAPI app
app = FastAPI(
    title="Task Assignment Coordinator",
    description="Distributes tasks to agents based on capabilities and load",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "agent": "task-assignment-coordinator",
        "timestamp": datetime.utcnow().isoformat(),
        "queued_tasks": len(task_coordinator.task_queue),
        "active_assignments": len(task_coordinator.active_assignments),
        "registered_agents": len(task_coordinator.agent_registry)
    }

@app.post("/submit_task")
async def submit_task(task: Task):
    """Submit a new task for assignment"""
    return await task_coordinator.submit_task(task)

@app.post("/register_agent")
async def register_agent(agent_data: Dict[str, Any]):
    """Register a new agent"""
    await task_coordinator.register_agent(agent_data)
    return {"message": f"Agent {agent_data.get('agent_id', 'unknown')} registered successfully"}

@app.post("/assignment_callback")
async def assignment_callback(callback_data: Dict[str, Any]):
    """Callback endpoint for assignment status updates"""
    assignment_id = callback_data.get("assignment_id")
    status = callback_data.get("status")
    
    if assignment_id in task_coordinator.active_assignments:
        if status == "completed":
            await task_coordinator.handle_completed_assignment(assignment_id, callback_data)
        elif status == "failed":
            await task_coordinator.handle_failed_assignment(assignment_id, callback_data)
    
    return {"message": "Callback processed"}

@app.get("/statistics")
async def get_statistics():
    """Get assignment statistics"""
    return await task_coordinator.get_assignment_statistics()

@app.get("/agents")
async def list_agents():
    """List all registered agents"""
    agents = []
    for agent_id, agent_info in task_coordinator.agent_registry.items():
        agents.append({
            "agent_id": agent_id,
            "name": agent_info.name,
            "capabilities": agent_info.capabilities,
            "current_load": agent_info.current_load,
            "availability": agent_info.availability,
            "performance_score": agent_info.performance_score,
            "last_seen": agent_info.last_seen.isoformat()
        })
    return {"agents": agents}

@app.get("/assignments/active")
async def list_active_assignments():
    """List active assignments"""
    assignments = []
    for assignment_id, assignment in task_coordinator.active_assignments.items():
        assignments.append({
            "assignment_id": assignment_id,
            "task_id": assignment.task_id,
            "agent_id": assignment.agent_id,
            "status": assignment.status,
            "assigned_at": assignment.assigned_at.isoformat(),
            "estimated_completion": assignment.estimated_completion.isoformat(),
            "retry_count": assignment.retry_count
        })
    return {"assignments": assignments}

@app.get("/queue")
async def list_queued_tasks():
    """List queued tasks"""
    tasks = []
    for task in task_coordinator.task_queue:
        tasks.append({
            "task_id": task.task_id,
            "task_type": task.task_type,
            "priority": task.priority,
            "required_capabilities": task.required_capabilities,
            "estimated_duration": task.estimated_duration
        })
    return {"queued_tasks": tasks}

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "agent": "task-assignment-coordinator",
        "status": "running",
        "description": "Task Distribution and Assignment Service"
    }

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8551"))
    uvicorn.run(app, host="0.0.0.0", port=port)