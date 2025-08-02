"""
LocalAGI Orchestration Manager for SutazAI AGI/ASI System
Manages 38 AI agents (19 Opus, 19 Sonnet) with autonomous workflows
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor

import redis
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import psutil


class AgentType(Enum):
    """Agent types in the SutazAI system"""
    OPUS = "opus"
    SONNET = "sonnet"
    ORCHESTRATOR = "orchestrator"
    SPECIALIZED = "specialized"


class AgentStatus(Enum):
    """Current status of an agent"""
    IDLE = "idle"
    BUSY = "busy"
    UNAVAILABLE = "unavailable"
    MAINTENANCE = "maintenance"
    ERROR = "error"


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class AgentCapability:
    """Represents an agent's capability"""
    name: str
    description: str
    proficiency: float  # 0.0 to 1.0
    resource_cost: float  # Relative cost
    

@dataclass
class Agent:
    """Represents an AI agent in the system"""
    id: str
    name: str
    type: AgentType
    model: str  # Claude Opus or Sonnet
    status: AgentStatus
    capabilities: List[AgentCapability]
    current_load: float  # 0.0 to 1.0
    max_concurrent_tasks: int
    active_tasks: Set[str]
    last_heartbeat: datetime
    metadata: Dict[str, Any]
    
    def can_handle_task(self, required_capabilities: List[str]) -> bool:
        """Check if agent can handle a task with given capabilities"""
        agent_caps = {cap.name for cap in self.capabilities}
        return all(req_cap in agent_caps for req_cap in required_capabilities)
    
    def get_capability_score(self, required_capabilities: List[str]) -> float:
        """Calculate agent's capability score for given requirements"""
        if not self.can_handle_task(required_capabilities):
            return 0.0
        
        cap_dict = {cap.name: cap.proficiency for cap in self.capabilities}
        scores = [cap_dict.get(req_cap, 0.0) for req_cap in required_capabilities]
        return sum(scores) / len(scores) if scores else 0.0


@dataclass
class Task:
    """Represents a task in the orchestration system"""
    id: str
    type: str
    description: str
    required_capabilities: List[str]
    priority: TaskPriority
    dependencies: List[str]
    assigned_agent_id: Optional[str]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    result: Optional[Dict[str, Any]]
    error: Optional[str]
    metadata: Dict[str, Any]


@dataclass
class Workflow:
    """Represents a multi-agent workflow"""
    id: str
    name: str
    description: str
    tasks: List[Task]
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    metadata: Dict[str, Any]


class LocalAGIOrchestrator:
    """
    Main orchestration manager for LocalAGI system
    Handles agent registration, task distribution, and workflow execution
    """
    
    def __init__(self, redis_url: str = "redis://localhost:6379"):
        self.redis_url = redis_url
        self.redis_client = None
        self.agents: Dict[str, Agent] = {}
        self.tasks: Dict[str, Task] = {}
        self.workflows: Dict[str, Workflow] = {}
        self.task_queue = asyncio.Queue()
        self.workflow_templates: Dict[str, Callable] = {}
        self.running = False
        self.executor = ThreadPoolExecutor(max_workers=10)
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_task_time": 0.0,
            "agent_utilization": 0.0,
            "system_load": 0.0
        }
        
        self.logger = logging.getLogger(__name__)
        
    async def initialize(self):
        """Initialize the orchestration system"""
        self.redis_client = await aioredis.from_url(self.redis_url)
        
        # Initialize 38 agents (19 Opus, 19 Sonnet)
        await self._initialize_agents()
        
        # Start background tasks
        self.running = True
        await asyncio.gather(
            self._task_processor(),
            self._agent_monitor(),
            self._metrics_collector(),
            self._heartbeat_monitor()
        )
    
    async def _initialize_agents(self):
        """Initialize the 38 AI agents"""
        # Define common capabilities
        opus_capabilities = [
            AgentCapability("advanced_reasoning", "Complex logical reasoning", 0.95, 1.5),
            AgentCapability("creative_writing", "Creative content generation", 0.90, 1.2),
            AgentCapability("code_generation", "Software development", 0.85, 1.3),
            AgentCapability("data_analysis", "Data processing and analysis", 0.80, 1.1),
            AgentCapability("research", "Information gathering and synthesis", 0.88, 1.0),
            AgentCapability("problem_solving", "Complex problem resolution", 0.92, 1.4),
            AgentCapability("communication", "Human-like communication", 0.85, 0.8)
        ]
        
        sonnet_capabilities = [
            AgentCapability("reasoning", "Logical reasoning", 0.85, 1.0),
            AgentCapability("code_generation", "Software development", 0.90, 1.0),
            AgentCapability("data_analysis", "Data processing", 0.85, 0.9),
            AgentCapability("research", "Information gathering", 0.80, 0.8),
            AgentCapability("communication", "Communication", 0.82, 0.7),
            AgentCapability("efficiency", "Fast task completion", 0.95, 0.6)
        ]
        
        # Create 19 Opus agents
        for i in range(19):
            agent_id = f"opus-{i+1:03d}"
            agent = Agent(
                id=agent_id,
                name=f"Claude Opus Agent {i+1}",
                type=AgentType.OPUS,
                model="claude-3-opus-20240229",
                status=AgentStatus.IDLE,
                capabilities=opus_capabilities.copy(),
                current_load=0.0,
                max_concurrent_tasks=3,
                active_tasks=set(),
                last_heartbeat=datetime.now(),
                metadata={
                    "tier": "premium",
                    "specialization": self._get_agent_specialization(i, "opus")
                }
            )
            self.agents[agent_id] = agent
            await self._register_agent_redis(agent)
        
        # Create 19 Sonnet agents  
        for i in range(19):
            agent_id = f"sonnet-{i+1:03d}"
            agent = Agent(
                id=agent_id,
                name=f"Claude Sonnet Agent {i+1}",
                type=AgentType.SONNET,
                model="claude-3-5-sonnet-20241022",
                status=AgentStatus.IDLE,
                capabilities=sonnet_capabilities.copy(),
                current_load=0.0,
                max_concurrent_tasks=5,
                active_tasks=set(),
                last_heartbeat=datetime.now(),
                metadata={
                    "tier": "standard",
                    "specialization": self._get_agent_specialization(i, "sonnet")
                }
            )
            self.agents[agent_id] = agent
            await self._register_agent_redis(agent)
            
        self.logger.info(f"Initialized {len(self.agents)} agents")
    
    def _get_agent_specialization(self, index: int, agent_type: str) -> str:
        """Get agent specialization based on index"""
        specializations = [
            "code_generation", "data_analysis", "research", "creative_writing",
            "problem_solving", "communication", "reasoning", "optimization",
            "security", "testing", "documentation", "architecture",
            "deployment", "monitoring", "automation", "integration",
            "ai_training", "model_optimization", "system_admin"
        ]
        return specializations[index % len(specializations)]
    
    async def _register_agent_redis(self, agent: Agent):
        """Register agent in Redis for discovery"""
        agent_data = {
            "id": agent.id,
            "name": agent.name,
            "type": agent.type.value,
            "model": agent.model,
            "status": agent.status.value,
            "capabilities": [cap.name for cap in agent.capabilities],
            "max_concurrent_tasks": agent.max_concurrent_tasks,
            "metadata": agent.metadata,
            "last_updated": datetime.now().isoformat()
        }
        
        await self.redis_client.hset(
            "sutazai:agents",
            agent.id,
            json.dumps(agent_data)
        )
    
    async def discover_agents(self, required_capabilities: List[str] = None) -> List[Dict]:
        """Discover available agents with optional capability filtering"""
        agents_data = await self.redis_client.hgetall("sutazai:agents")
        agents = []
        
        for agent_id, agent_json in agents_data.items():
            agent_data = json.loads(agent_json)
            
            if required_capabilities:
                if not all(cap in agent_data["capabilities"] for cap in required_capabilities):
                    continue
                    
            agents.append(agent_data)
        
        return agents
    
    async def find_best_agent(self, required_capabilities: List[str], 
                            task_priority: TaskPriority) -> Optional[Agent]:
        """Find the best agent for a task"""
        available_agents = [
            agent for agent in self.agents.values()
            if (agent.status == AgentStatus.IDLE and 
                len(agent.active_tasks) < agent.max_concurrent_tasks and
                agent.can_handle_task(required_capabilities))
        ]
        
        if not available_agents:
            return None
        
        # Score agents based on capability match, load, and type preference
        def score_agent(agent: Agent) -> float:
            capability_score = agent.get_capability_score(required_capabilities)
            load_score = 1.0 - agent.current_load
            
            # Prefer Opus for high priority tasks
            type_score = 1.2 if (agent.type == AgentType.OPUS and 
                               task_priority in [TaskPriority.HIGH, TaskPriority.CRITICAL]) else 1.0
                               
            return capability_score * load_score * type_score
        
        best_agent = max(available_agents, key=score_agent)
        return best_agent
    
    async def submit_task(self, task_type: str, description: str, 
                         required_capabilities: List[str],
                         priority: TaskPriority = TaskPriority.MEDIUM,
                         dependencies: List[str] = None,
                         metadata: Dict[str, Any] = None) -> str:
        """Submit a new task to the orchestration system"""
        task_id = str(uuid.uuid4())
        
        task = Task(
            id=task_id,
            type=task_type,
            description=description,
            required_capabilities=required_capabilities,
            priority=priority,
            dependencies=dependencies or [],
            assigned_agent_id=None,
            status="pending",
            created_at=datetime.now(),
            started_at=None,
            completed_at=None,
            result=None,
            error=None,
            metadata=metadata or {}
        )
        
        self.tasks[task_id] = task
        await self.task_queue.put(task_id)
        
        # Store in Redis for persistence
        await self.redis_client.hset(
            "sutazai:tasks",
            task_id,
            json.dumps(asdict(task), default=str)
        )
        
        self.logger.info(f"Task {task_id} submitted: {description}")
        return task_id
    
    async def _task_processor(self):
        """Main task processing loop"""
        while self.running:
            try:
                # Get task from queue with timeout
                task_id = await asyncio.wait_for(
                    self.task_queue.get(), timeout=1.0
                )
                
                task = self.tasks.get(task_id)
                if not task:
                    continue
                
                # Check dependencies
                if not await self._check_dependencies(task):
                    # Re-queue if dependencies not met
                    await asyncio.sleep(5)
                    await self.task_queue.put(task_id)
                    continue
                
                # Find best agent
                agent = await self.find_best_agent(
                    task.required_capabilities, 
                    task.priority
                )
                
                if not agent:
                    # No available agent, re-queue
                    await asyncio.sleep(2)
                    await self.task_queue.put(task_id)
                    continue
                
                # Assign task to agent
                await self._assign_task(task, agent)
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(1)
    
    async def _check_dependencies(self, task: Task) -> bool:
        """Check if task dependencies are satisfied"""
        for dep_id in task.dependencies:
            dep_task = self.tasks.get(dep_id)
            if not dep_task or dep_task.status != "completed":
                return False
        return True
    
    async def _assign_task(self, task: Task, agent: Agent):
        """Assign a task to an agent"""
        task.assigned_agent_id = agent.id
        task.status = "assigned"
        task.started_at = datetime.now()
        
        agent.active_tasks.add(task.id)
        agent.status = AgentStatus.BUSY
        agent.current_load = len(agent.active_tasks) / agent.max_concurrent_tasks
        
        # Update Redis
        await self._update_task_redis(task)
        await self._register_agent_redis(agent)
        
        # Execute task asynchronously
        asyncio.create_task(self._execute_task(task, agent))
        
        self.logger.info(f"Task {task.id} assigned to agent {agent.id}")
    
    async def _execute_task(self, task: Task, agent: Agent):
        """Execute a task on an agent"""
        try:
            # Simulate task execution
            # In real implementation, this would call the actual agent
            execution_time = await self._simulate_task_execution(task, agent)
            
            # Mark task as completed
            task.status = "completed"
            task.completed_at = datetime.now()
            task.result = {
                "success": True,
                "execution_time": execution_time,
                "agent_id": agent.id
            }
            
            self.metrics["tasks_completed"] += 1
            
        except Exception as e:
            task.status = "failed"
            task.error = str(e)
            self.metrics["tasks_failed"] += 1
            self.logger.error(f"Task {task.id} failed: {e}")
        
        finally:
            # Clean up agent state
            agent.active_tasks.discard(task.id)
            agent.current_load = len(agent.active_tasks) / agent.max_concurrent_tasks
            if len(agent.active_tasks) == 0:
                agent.status = AgentStatus.IDLE
            
            # Update Redis
            await self._update_task_redis(task)
            await self._register_agent_redis(agent)
    
    async def _simulate_task_execution(self, task: Task, agent: Agent) -> float:
        """Simulate task execution (replace with actual agent calls)"""
        # Base execution time varies by agent type and task complexity
        base_time = 2.0 if agent.type == AgentType.OPUS else 1.5
        
        # Adjust for agent specialization
        specialization = agent.metadata.get("specialization", "")
        if specialization in task.required_capabilities:
            base_time *= 0.8  # Faster for specialized tasks
        
        # Add some randomness
        import random
        execution_time = base_time * (0.5 + random.random())
        
        await asyncio.sleep(execution_time)
        return execution_time
    
    async def _update_task_redis(self, task: Task):
        """Update task in Redis"""
        await self.redis_client.hset(
            "sutazai:tasks",
            task.id,
            json.dumps(asdict(task), default=str)
        )
    
    async def _agent_monitor(self):
        """Monitor agent health and status"""
        while self.running:
            try:
                for agent in self.agents.values():
                    # Update heartbeat
                    agent.last_heartbeat = datetime.now()
                    
                    # Check for stuck tasks
                    current_time = datetime.now()
                    for task_id in list(agent.active_tasks):
                        task = self.tasks.get(task_id)
                        if task and task.started_at:
                            elapsed = (current_time - task.started_at).total_seconds()
                            if elapsed > 300:  # 5 minutes timeout
                                self.logger.warning(f"Task {task_id} timeout on agent {agent.id}")
                                task.status = "timeout"
                                task.error = "Task execution timeout"
                                agent.active_tasks.discard(task_id)
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Error in agent monitor: {e}")
                await asyncio.sleep(5)
    
    async def _metrics_collector(self):
        """Collect system performance metrics"""
        while self.running:
            try:
                # Calculate agent utilization
                total_capacity = sum(agent.max_concurrent_tasks for agent in self.agents.values())
                active_tasks = sum(len(agent.active_tasks) for agent in self.agents.values())
                self.metrics["agent_utilization"] = active_tasks / total_capacity if total_capacity > 0 else 0
                
                # System load
                self.metrics["system_load"] = psutil.cpu_percent(interval=1) / 100.0
                
                # Store metrics in Redis
                await self.redis_client.hset(
                    "sutazai:metrics",
                    "current",
                    json.dumps(self.metrics)
                )
                
                await asyncio.sleep(60)  # Update every minute
                
            except Exception as e:
                self.logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(10)
    
    async def _heartbeat_monitor(self):
        """Monitor agent heartbeats"""
        while self.running:
            try:
                current_time = datetime.now()
                
                for agent in self.agents.values():
                    if (current_time - agent.last_heartbeat).total_seconds() > 120:
                        if agent.status != AgentStatus.UNAVAILABLE:
                            self.logger.warning(f"Agent {agent.id} missed heartbeat")
                            agent.status = AgentStatus.UNAVAILABLE
                            await self._register_agent_redis(agent)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat monitor: {e}")
                await asyncio.sleep(10)
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        agent_status = {}
        for status in AgentStatus:
            agent_status[status.value] = len([
                a for a in self.agents.values() if a.status == status
            ])
        
        task_status = {}
        for status in ["pending", "assigned", "completed", "failed", "timeout"]:
            task_status[status] = len([
                t for t in self.tasks.values() if t.status == status
            ])
        
        return {
            "agents": {
                "total": len(self.agents),
                "by_status": agent_status,
                "by_type": {
                    "opus": len([a for a in self.agents.values() if a.type == AgentType.OPUS]),
                    "sonnet": len([a for a in self.agents.values() if a.type == AgentType.SONNET])
                }
            },
            "tasks": {
                "total": len(self.tasks),
                "by_status": task_status
            },
            "metrics": self.metrics.copy(),
            "uptime": time.time() - self.start_time if hasattr(self, 'start_time') else 0
        }
    
    async def shutdown(self):
        """Gracefully shutdown the orchestration system"""
        self.running = False
        
        # Cancel all pending tasks
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        # Shutdown thread pool
        self.executor.shutdown(wait=True)
        
        self.logger.info("LocalAGI Orchestrator shutdown complete")


# FastAPI application for orchestrator API
app = FastAPI(title="LocalAGI Orchestrator", version="1.0.0")
orchestrator = LocalAGIOrchestrator()


@app.on_event("startup")
async def startup():
    orchestrator.start_time = time.time()
    await orchestrator.initialize()


@app.on_event("shutdown") 
async def shutdown():
    await orchestrator.shutdown()


# API Endpoints
class TaskRequest(BaseModel):
    type: str
    description: str
    required_capabilities: List[str]
    priority: str = "medium"
    dependencies: List[str] = []
    metadata: Dict[str, Any] = {}


@app.post("/tasks")
async def submit_task(task_request: TaskRequest):
    """Submit a new task"""
    priority_map = {
        "low": TaskPriority.LOW,
        "medium": TaskPriority.MEDIUM, 
        "high": TaskPriority.HIGH,
        "critical": TaskPriority.CRITICAL
    }
    
    task_id = await orchestrator.submit_task(
        task_type=task_request.type,
        description=task_request.description,
        required_capabilities=task_request.required_capabilities,
        priority=priority_map.get(task_request.priority, TaskPriority.MEDIUM),
        dependencies=task_request.dependencies,
        metadata=task_request.metadata
    )
    
    return {"task_id": task_id, "status": "submitted"}


@app.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """Get task status"""
    task = orchestrator.tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return asdict(task)


@app.get("/agents")
async def list_agents():
    """List all agents"""
    return await orchestrator.discover_agents()


@app.get("/agents/{agent_id}")
async def get_agent(agent_id: str):
    """Get agent details"""
    agent = orchestrator.agents.get(agent_id)
    if not agent:
        raise HTTPException(status_code=404, detail="Agent not found")
    
    return asdict(agent)


@app.get("/status")
async def get_status():
    """Get system status"""
    return await orchestrator.get_system_status()


@app.get("/metrics")
async def get_metrics():
    """Get system metrics"""
    return orchestrator.metrics


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8200)