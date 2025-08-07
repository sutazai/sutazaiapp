#!/usr/bin/env python3
"""
AI Agent Orchestrator - Real Implementation with RabbitMQ
Manages agent coordination, task distribution, and conflict resolution
"""
import os
import sys
import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from contextlib import asynccontextmanager
from collections import defaultdict
import uuid

import uvicorn
import redis.asyncio as redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import httpx

# Add parent directory to path for imports
sys.path.append('/app')
sys.path.append('/opt/sutazaiapp/agents')

from core.messaging import (
    RabbitMQClient, MessageProcessor, 
    TaskMessage, StatusMessage, ResourceMessage, ErrorMessage,
    MessageType, Priority
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
AGENT_ID = "ai-agent-orchestrator"
REDIS_TASK_TTL = 3600  # 1 hour
MAX_TASK_RETRIES = 3
TASK_TIMEOUT_SECONDS = 300

# Data Models
class AgentCapability(BaseModel):
    """Agent capability definition"""
    capability_type: str
    proficiency: float = 1.0  # 0.0 to 1.0
    max_concurrent: int = 5
    average_duration_seconds: float = 60.0

class RegisteredAgent(BaseModel):
    """Registered agent information"""
    agent_id: str
    agent_type: str
    capabilities: List[AgentCapability]
    endpoint: str
    status: str = "online"
    current_load: float = 0.0
    max_load: float = 1.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    average_response_time: float = 0.0
    last_heartbeat: datetime = Field(default_factory=datetime.utcnow)
    
class TaskRequest(BaseModel):
    """Task request from clients"""
    task_type: str
    payload: Dict[str, Any]
    priority: int = 1
    timeout_seconds: int = 300
    required_capabilities: List[str] = []

class TaskAssignment(BaseModel):
    """Internal task assignment tracking"""
    task_id: str
    task_type: str
    assigned_agent: str
    status: str = "pending"
    priority: Priority
    created_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    error_message: Optional[str] = None


class OrchestratorMessageProcessor(MessageProcessor):
    """Message processor for the orchestrator"""
    
    def __init__(self, orchestrator):
        super().__init__(AGENT_ID)
        self.orchestrator = orchestrator
        
    async def handle_task_request(self, message: Dict[str, Any]):
        """Handle incoming task requests"""
        try:
            task_msg = TaskMessage(**message)
            
            # Find best agent for the task
            best_agent = await self.orchestrator.find_best_agent(
                task_msg.task_type,
                task_msg.payload.get("required_capabilities", [])
            )
            
            if not best_agent:
                # No suitable agent found - publish error
                await self.rabbitmq_client.publish_error(
                    error_code="NO_AGENT_AVAILABLE",
                    error_message=f"No agent available for task type: {task_msg.task_type}",
                    original_message_id=task_msg.message_id
                )
                return
            
            # Create assignment
            assignment = TaskAssignment(
                task_id=task_msg.task_id,
                task_type=task_msg.task_type,
                assigned_agent=best_agent.agent_id,
                priority=Priority(task_msg.priority),
                status="assigned"
            )
            
            # Store assignment
            await self.orchestrator.store_assignment(assignment)
            
            # Forward task to selected agent
            await self.rabbitmq_client.publish_task(
                task_id=task_msg.task_id,
                task_type=task_msg.task_type,
                payload=task_msg.payload,
                target_agent=best_agent.agent_id,
                priority=Priority(task_msg.priority)
            )
            
            logger.info(f"Assigned task {task_msg.task_id} to agent {best_agent.agent_id}")
            
        except Exception as e:
            logger.error(f"Error handling task request: {e}")
            await self.rabbitmq_client.publish_error(
                error_code="TASK_PROCESSING_ERROR",
                error_message=str(e),
                original_message_id=message.get("message_id")
            )
    
    async def handle_status_update(self, message: Dict[str, Any]):
        """Handle status updates from agents"""
        try:
            status_msg = StatusMessage(**message)
            
            # Update assignment status
            await self.orchestrator.update_task_status(
                status_msg.task_id,
                status_msg.status,
                status_msg.details
            )
            
            # If task failed and retries available, reassign
            if status_msg.status == "failed":
                assignment = await self.orchestrator.get_assignment(status_msg.task_id)
                if assignment and assignment.retry_count < MAX_TASK_RETRIES:
                    await self.orchestrator.retry_task(assignment)
            
        except Exception as e:
            logger.error(f"Error handling status update: {e}")
    
    async def handle_agent_heartbeat(self, message: Dict[str, Any]):
        """Handle agent heartbeat messages"""
        try:
            agent_id = message.get("source_agent")
            agent_data = message.get("payload", {})
            
            await self.orchestrator.update_agent_status(
                agent_id,
                agent_data.get("status", "online"),
                agent_data.get("current_load", 0.0),
                agent_data.get("capabilities", [])
            )
            
        except Exception as e:
            logger.error(f"Error handling agent heartbeat: {e}")


class AIAgentOrchestrator:
    """Main orchestrator implementation with real logic"""
    
    def __init__(self):
        self.redis_client = None
        self.message_processor = None
        self.registered_agents: Dict[str, RegisteredAgent] = {}
        self.task_assignments: Dict[str, TaskAssignment] = {}
        self.agent_capabilities: Dict[str, Set[str]] = defaultdict(set)
        self.task_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        
    async def initialize(self):
        """Initialize the orchestrator"""
        try:
            # Connect to Redis
            redis_url = os.getenv("REDIS_URL", "redis://redis:6379/0")
            self.redis_client = redis.from_url(redis_url, decode_responses=True)
            await self.redis_client.ping()
            logger.info("Connected to Redis")
            
            # Initialize message processor
            self.message_processor = OrchestratorMessageProcessor(self)
            await self.message_processor.start()
            logger.info("Message processor started")
            
            # Load registered agents from Redis
            await self.load_registered_agents()
            
            # Start background tasks
            self.running = True
            asyncio.create_task(self.task_scheduler())
            asyncio.create_task(self.health_monitor())
            asyncio.create_task(self.cleanup_old_tasks())
            
            logger.info("AI Agent Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown the orchestrator"""
        self.running = False
        
        if self.message_processor:
            await self.message_processor.stop()
        
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("AI Agent Orchestrator shutdown complete")
    
    async def load_registered_agents(self):
        """Load registered agents from Redis"""
        try:
            agent_keys = await self.redis_client.keys("agent:*:info")
            
            for key in agent_keys:
                agent_data = await self.redis_client.get(key)
                if agent_data:
                    agent_info = json.loads(agent_data)
                    agent = RegisteredAgent(**agent_info)
                    self.registered_agents[agent.agent_id] = agent
                    
                    # Index capabilities
                    for cap in agent.capabilities:
                        self.agent_capabilities[cap.capability_type].add(agent.agent_id)
            
            logger.info(f"Loaded {len(self.registered_agents)} registered agents")
            
        except Exception as e:
            logger.error(f"Error loading registered agents: {e}")
    
    async def register_agent(self, agent_data: Dict[str, Any]) -> RegisteredAgent:
        """Register a new agent"""
        try:
            # Create agent object
            agent = RegisteredAgent(
                agent_id=agent_data["agent_id"],
                agent_type=agent_data.get("agent_type", "generic"),
                capabilities=[AgentCapability(**cap) for cap in agent_data.get("capabilities", [])],
                endpoint=agent_data.get("endpoint", f"http://{agent_data['agent_id']}:8080")
            )
            
            # Store in memory
            self.registered_agents[agent.agent_id] = agent
            
            # Index capabilities
            for cap in agent.capabilities:
                self.agent_capabilities[cap.capability_type].add(agent.agent_id)
            
            # Store in Redis
            await self.redis_client.setex(
                f"agent:{agent.agent_id}:info",
                REDIS_TASK_TTL,
                agent.json()
            )
            
            logger.info(f"Registered agent: {agent.agent_id}")
            return agent
            
        except Exception as e:
            logger.error(f"Error registering agent: {e}")
            raise
    
    async def find_best_agent(
        self, 
        task_type: str, 
        required_capabilities: List[str]
    ) -> Optional[RegisteredAgent]:
        """Find the best available agent for a task"""
        try:
            candidates = []
            
            # Find agents with required capabilities
            for capability in required_capabilities:
                if capability in self.agent_capabilities:
                    for agent_id in self.agent_capabilities[capability]:
                        if agent_id in self.registered_agents:
                            agent = self.registered_agents[agent_id]
                            if agent.status == "online" and agent.current_load < agent.max_load:
                                candidates.append(agent)
            
            # If no specific capabilities required, consider all online agents
            if not required_capabilities:
                candidates = [
                    agent for agent in self.registered_agents.values()
                    if agent.status == "online" and agent.current_load < agent.max_load
                ]
            
            if not candidates:
                return None
            
            # Score and sort candidates
            def score_agent(agent: RegisteredAgent) -> float:
                load_score = 1.0 - (agent.current_load / agent.max_load)
                success_rate = (agent.tasks_completed / 
                              max(1, agent.tasks_completed + agent.tasks_failed))
                response_time_score = 1.0 / max(1.0, agent.average_response_time / 60.0)
                
                return (load_score * 0.4 + 
                       success_rate * 0.4 + 
                       response_time_score * 0.2)
            
            candidates.sort(key=score_agent, reverse=True)
            return candidates[0]
            
        except Exception as e:
            logger.error(f"Error finding best agent: {e}")
            return None
    
    async def store_assignment(self, assignment: TaskAssignment):
        """Store task assignment"""
        try:
            self.task_assignments[assignment.task_id] = assignment
            
            # Store in Redis
            await self.redis_client.setex(
                f"task:{assignment.task_id}:assignment",
                REDIS_TASK_TTL,
                assignment.json()
            )
            
            # Update agent load
            if assignment.assigned_agent in self.registered_agents:
                agent = self.registered_agents[assignment.assigned_agent]
                agent.current_load = min(agent.max_load, agent.current_load + 0.1)
            
        except Exception as e:
            logger.error(f"Error storing assignment: {e}")
    
    async def get_assignment(self, task_id: str) -> Optional[TaskAssignment]:
        """Get task assignment"""
        if task_id in self.task_assignments:
            return self.task_assignments[task_id]
        
        # Try to load from Redis
        data = await self.redis_client.get(f"task:{task_id}:assignment")
        if data:
            assignment = TaskAssignment(**json.loads(data))
            self.task_assignments[task_id] = assignment
            return assignment
        
        return None
    
    async def update_task_status(
        self, 
        task_id: str, 
        status: str, 
        details: Optional[Dict[str, Any]] = None
    ):
        """Update task status"""
        try:
            assignment = await self.get_assignment(task_id)
            if not assignment:
                logger.warning(f"Assignment not found for task {task_id}")
                return
            
            assignment.status = status
            
            if status == "processing":
                assignment.started_at = datetime.utcnow()
            elif status in ["completed", "failed"]:
                assignment.completed_at = datetime.utcnow()
                
                # Update agent metrics
                if assignment.assigned_agent in self.registered_agents:
                    agent = self.registered_agents[assignment.assigned_agent]
                    agent.current_load = max(0, agent.current_load - 0.1)
                    
                    if status == "completed":
                        agent.tasks_completed += 1
                    else:
                        agent.tasks_failed += 1
                    
                    # Update average response time
                    if assignment.started_at and assignment.completed_at:
                        duration = (assignment.completed_at - assignment.started_at).total_seconds()
                        agent.average_response_time = (
                            (agent.average_response_time * (agent.tasks_completed - 1) + duration) /
                            agent.tasks_completed
                        )
            
            # Store updated assignment
            await self.store_assignment(assignment)
            
            # Publish status update
            await self.message_processor.rabbitmq_client.publish_status(
                task_id=task_id,
                status=status,
                details=details
            )
            
        except Exception as e:
            logger.error(f"Error updating task status: {e}")
    
    async def retry_task(self, assignment: TaskAssignment):
        """Retry a failed task"""
        try:
            assignment.retry_count += 1
            assignment.status = "pending"
            assignment.error_message = None
            
            # Find new agent (might be same one)
            best_agent = await self.find_best_agent(
                assignment.task_type,
                []  # TODO: Extract required capabilities from original task
            )
            
            if best_agent:
                assignment.assigned_agent = best_agent.agent_id
                await self.store_assignment(assignment)
                
                # Republish task
                await self.message_processor.rabbitmq_client.publish_task(
                    task_id=assignment.task_id,
                    task_type=assignment.task_type,
                    payload={},  # TODO: Store and retrieve original payload
                    target_agent=best_agent.agent_id,
                    priority=assignment.priority
                )
                
                logger.info(f"Retrying task {assignment.task_id} (attempt {assignment.retry_count})")
            
        except Exception as e:
            logger.error(f"Error retrying task: {e}")
    
    async def update_agent_status(
        self,
        agent_id: str,
        status: str,
        current_load: float,
        capabilities: List[Dict[str, Any]]
    ):
        """Update agent status from heartbeat"""
        try:
            if agent_id not in self.registered_agents:
                # Auto-register new agent
                await self.register_agent({
                    "agent_id": agent_id,
                    "agent_type": "generic",
                    "capabilities": capabilities,
                    "endpoint": f"http://{agent_id}:8080"
                })
            
            agent = self.registered_agents[agent_id]
            agent.status = status
            agent.current_load = current_load
            agent.last_heartbeat = datetime.utcnow()
            
            # Update Redis
            await self.redis_client.setex(
                f"agent:{agent_id}:info",
                REDIS_TASK_TTL,
                agent.json()
            )
            
        except Exception as e:
            logger.error(f"Error updating agent status: {e}")
    
    async def task_scheduler(self):
        """Background task scheduler"""
        while self.running:
            try:
                # Process pending tasks from queue
                if not self.task_queue.empty():
                    task = await self.task_queue.get()
                    # Process task...
                
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error in task scheduler: {e}")
                await asyncio.sleep(5)
    
    async def health_monitor(self):
        """Monitor agent health"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                for agent_id, agent in self.registered_agents.items():
                    # Check for stale heartbeats
                    if (current_time - agent.last_heartbeat).total_seconds() > 120:
                        agent.status = "offline"
                        agent.current_load = 0.0
                        logger.warning(f"Agent {agent_id} marked offline (no heartbeat)")
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)
    
    async def cleanup_old_tasks(self):
        """Clean up old completed tasks"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                tasks_to_remove = []
                
                for task_id, assignment in self.task_assignments.items():
                    if assignment.completed_at:
                        age = (current_time - assignment.completed_at).total_seconds()
                        if age > 3600:  # 1 hour
                            tasks_to_remove.append(task_id)
                
                for task_id in tasks_to_remove:
                    del self.task_assignments[task_id]
                    await self.redis_client.delete(f"task:{task_id}:assignment")
                
                if tasks_to_remove:
                    logger.info(f"Cleaned up {len(tasks_to_remove)} old tasks")
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup: {e}")
                await asyncio.sleep(600)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get orchestrator status"""
        return {
            "status": "healthy",
            "registered_agents": len(self.registered_agents),
            "online_agents": len([a for a in self.registered_agents.values() if a.status == "online"]),
            "active_tasks": len([t for t in self.task_assignments.values() if t.status == "processing"]),
            "pending_tasks": len([t for t in self.task_assignments.values() if t.status == "pending"]),
            "completed_tasks": len([t for t in self.task_assignments.values() if t.status == "completed"]),
            "failed_tasks": len([t for t in self.task_assignments.values() if t.status == "failed"])
        }


# Global orchestrator instance
orchestrator = AIAgentOrchestrator()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await orchestrator.initialize()
    yield
    # Shutdown
    await orchestrator.shutdown()

# Create FastAPI app
app = FastAPI(
    title="AI Agent Orchestrator",
    description="Real implementation with RabbitMQ messaging",
    version="2.0.0",
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    status = await orchestrator.get_status()
    return {
        "status": "healthy",
        "agent": AGENT_ID,
        "timestamp": datetime.utcnow().isoformat(),
        **status
    }

@app.post("/register_agent")
async def register_agent(agent_data: Dict[str, Any]):
    """Register a new agent"""
    agent = await orchestrator.register_agent(agent_data)
    return {
        "status": "success",
        "agent_id": agent.agent_id,
        "message": f"Agent {agent.agent_id} registered successfully"
    }

@app.post("/submit_task")
async def submit_task(request: TaskRequest):
    """Submit a new task for orchestration"""
    task_id = str(uuid.uuid4())
    
    # Publish task to RabbitMQ
    message = await orchestrator.message_processor.rabbitmq_client.publish_task(
        task_id=task_id,
        task_type=request.task_type,
        payload=request.payload,
        priority=Priority(request.priority)
    )
    
    return {
        "task_id": task_id,
        "status": "submitted",
        "message_id": message.message_id
    }

@app.get("/task/{task_id}")
async def get_task_status(task_id: str):
    """Get task status"""
    assignment = await orchestrator.get_assignment(task_id)
    
    if not assignment:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return {
        "task_id": task_id,
        "status": assignment.status,
        "assigned_agent": assignment.assigned_agent,
        "created_at": assignment.created_at.isoformat(),
        "started_at": assignment.started_at.isoformat() if assignment.started_at else None,
        "completed_at": assignment.completed_at.isoformat() if assignment.completed_at else None,
        "retry_count": assignment.retry_count,
        "error_message": assignment.error_message
    }

@app.get("/agents")
async def list_agents():
    """List all registered agents"""
    agents = []
    for agent in orchestrator.registered_agents.values():
        agents.append({
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type,
            "status": agent.status,
            "current_load": agent.current_load,
            "tasks_completed": agent.tasks_completed,
            "tasks_failed": agent.tasks_failed,
            "average_response_time": agent.average_response_time,
            "last_heartbeat": agent.last_heartbeat.isoformat()
        })
    return {"agents": agents}

@app.get("/status")
async def get_orchestrator_status():
    """Get detailed orchestrator status"""
    return await orchestrator.get_status()

if __name__ == "__main__":
    port = int(os.getenv("PORT", "8589"))
    uvicorn.run(app, host="0.0.0.0", port=port)