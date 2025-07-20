#!/usr/bin/env python3
"""
SutazAI AGI/ASI Central Orchestrator
Advanced agent coordination and task management system
"""

import asyncio
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import json
from contextlib import asynccontextmanager

import redis.asyncio as redis
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import select, update, delete
import aiohttp
from celery import Celery
from kombu import Queue

from backend.core.config import get_settings
from backend.models.base_models import Agent, Task, Workflow, KnowledgeNode
from backend.utils.logging_setup import get_api_logger

logger = get_api_logger()
settings = get_settings()

class TaskStatus(Enum):
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"

class AgentStatus(Enum):
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"
    ERROR = "error"
    INITIALIZING = "initializing"

@dataclass
class TaskRequest:
    """Task request data structure"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    payload: Dict[str, Any] = field(default_factory=dict)
    priority: int = 5
    timeout: int = 300
    retries: int = 3
    dependencies: List[str] = field(default_factory=list)
    agent_constraints: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    context: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    agent_id: Optional[str] = None
    completed_at: Optional[datetime] = None

class AGIOrchestrator:
    """
    Advanced AGI/ASI Orchestrator for managing autonomous agents,
    task queues, and system-wide coordination
    """
    
    def __init__(self):
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.active_tasks: Dict[str, TaskRequest] = {}
        self.task_results: Dict[str, TaskResult] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}
        
        # Initialize connections
        self.redis_client: Optional[redis.Redis] = None
        self.db_engine = None
        self.session_factory = None
        self.celery_app = None
        
        # Agent management
        self.agent_discovery_interval = 30  # seconds
        self.health_check_interval = 60    # seconds
        self.task_timeout_check_interval = 30  # seconds
        
        # Performance metrics
        self.metrics = {
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_execution_time": 0.0,
            "active_agents": 0,
            "queue_length": 0
        }
        
    async def initialize(self):
        """Initialize all orchestrator components"""
        logger.info("Initializing AGI Orchestrator...")
        
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host=settings.REDIS_HOST,
                port=settings.REDIS_PORT,
                db=0,
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established")
            
            # Initialize database
            self.db_engine = create_async_engine(
                settings.DATABASE_URL,
                echo=False,
                pool_size=20,
                max_overflow=30
            )
            self.session_factory = sessionmaker(
                self.db_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            logger.info("Database connection established")
            
            # Initialize Celery for distributed task processing
            self.celery_app = Celery(
                'sutazai_orchestrator',
                broker=f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/1',
                backend=f'redis://{settings.REDIS_HOST}:{settings.REDIS_PORT}/1'
            )
            
            # Configure Celery task routing
            self.celery_app.conf.update(
                task_routes={
                    'sutazai.agents.code_generation': {'queue': 'code_generation'},
                    'sutazai.agents.document_processing': {'queue': 'documents'},
                    'sutazai.agents.web_automation': {'queue': 'web_automation'},
                    'sutazai.agents.security_analysis': {'queue': 'security'},
                    'sutazai.agents.general': {'queue': 'general'},
                },
                task_default_queue='general',
                worker_prefetch_multiplier=1,
                task_acks_late=True,
                worker_disable_rate_limits=False,
                task_reject_on_worker_lost=True,
            )
            
            # Start background tasks
            asyncio.create_task(self.agent_discovery_loop())
            asyncio.create_task(self.health_check_loop())
            asyncio.create_task(self.task_timeout_check_loop())
            asyncio.create_task(self.metrics_update_loop())
            
            # Discover existing agents
            await self.discover_agents()
            
            logger.info("AGI Orchestrator initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")
            raise
    
    async def discover_agents(self):
        """Discover and register available agents"""
        logger.info("Discovering available agents...")
        
        agent_services = [
            {"name": "autogpt", "type": "task_automation", "url": "http://sutazai-autogpt:8080"},
            {"name": "crewai", "type": "multi_agent", "url": "http://sutazai-crewai:8080"},
            {"name": "aider", "type": "code_generation", "url": "http://sutazai-aider:8080"},
            {"name": "gpt-engineer", "type": "code_generation", "url": "http://sutazai-gpt-engineer:8080"},
            {"name": "semgrep", "type": "security_analysis", "url": "http://sutazai-semgrep:8080"},
            {"name": "tabbyml", "type": "code_completion", "url": "http://tabbyml:8080"},
            {"name": "documind", "type": "document_processing", "url": "http://documind:8080"},
        ]
        
        for agent_config in agent_services:
            try:
                # Check if agent is responsive
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{agent_config['url']}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            await self.register_agent(agent_config)
                        else:
                            logger.warning(f"Agent {agent_config['name']} health check failed")
            except Exception as e:
                logger.warning(f"Could not reach agent {agent_config['name']}: {e}")
    
    async def register_agent(self, agent_config: Dict[str, Any]):
        """Register a new agent with the orchestrator"""
        agent_id = agent_config["name"]
        
        agent_data = {
            "id": agent_id,
            "name": agent_config["name"],
            "type": agent_config["type"],
            "url": agent_config["url"],
            "status": AgentStatus.IDLE,
            "capabilities": agent_config.get("capabilities", []),
            "current_task": None,
            "last_heartbeat": datetime.now(),
            "performance_metrics": {
                "tasks_completed": 0,
                "tasks_failed": 0,
                "average_execution_time": 0.0,
                "success_rate": 1.0
            },
            "resource_usage": {
                "cpu_percent": 0.0,
                "memory_percent": 0.0,
                "disk_usage": 0.0
            }
        }
        
        self.agents[agent_id] = agent_data
        
        # Store in Redis for distributed access
        await self.redis_client.hset(
            "sutazai:agents",
            agent_id,
            json.dumps(agent_data, default=str)
        )
        
        logger.info(f"Registered agent: {agent_id} ({agent_config['type']})")
    
    async def submit_task(self, task_request: TaskRequest) -> str:
        """Submit a task for execution"""
        logger.info(f"Submitting task {task_request.id} of type {task_request.type}")
        
        # Validate task request
        if not task_request.type:
            raise ValueError("Task type is required")
        
        # Store task in active tasks
        self.active_tasks[task_request.id] = task_request
        
        # Store in Redis for persistence
        await self.redis_client.hset(
            "sutazai:tasks:active",
            task_request.id,
            json.dumps(task_request.__dict__, default=str)
        )
        
        # Find suitable agent
        suitable_agent = await self.find_suitable_agent(task_request)
        
        if not suitable_agent:
            # Queue task for later execution
            await self.queue_task(task_request)
            logger.info(f"Task {task_request.id} queued - no suitable agent available")
        else:
            # Execute task immediately
            await self.execute_task(task_request, suitable_agent)
        
        return task_request.id
    
    async def find_suitable_agent(self, task_request: TaskRequest) -> Optional[str]:
        """Find the most suitable agent for a task"""
        best_agent = None
        best_score = 0.0
        
        for agent_id, agent_data in self.agents.items():
            if agent_data["status"] != AgentStatus.IDLE:
                continue
            
            # Calculate suitability score
            score = await self.calculate_agent_suitability(agent_data, task_request)
            
            if score > best_score:
                best_score = score
                best_agent = agent_id
        
        return best_agent if best_score > 0.5 else None
    
    async def calculate_agent_suitability(self, agent_data: Dict[str, Any], task_request: TaskRequest) -> float:
        """Calculate how suitable an agent is for a specific task"""
        score = 0.0
        
        # Type matching
        if agent_data["type"] == task_request.type:
            score += 0.5
        elif task_request.type in agent_data.get("capabilities", []):
            score += 0.3
        
        # Performance history
        performance = agent_data.get("performance_metrics", {})
        success_rate = performance.get("success_rate", 0.5)
        score += success_rate * 0.3
        
        # Current load (prefer less busy agents)
        if agent_data["status"] == AgentStatus.IDLE:
            score += 0.2
        
        # Agent constraints
        constraints = task_request.agent_constraints
        if constraints:
            if "preferred_agent" in constraints:
                if agent_data["id"] == constraints["preferred_agent"]:
                    score += 0.5
            if "excluded_agents" in constraints:
                if agent_data["id"] in constraints["excluded_agents"]:
                    score = 0.0
        
        return min(score, 1.0)
    
    async def execute_task(self, task_request: TaskRequest, agent_id: str):
        """Execute a task using the specified agent"""
        logger.info(f"Executing task {task_request.id} with agent {agent_id}")
        
        # Update agent status
        self.agents[agent_id]["status"] = AgentStatus.BUSY
        self.agents[agent_id]["current_task"] = task_request.id
        
        # Update task status
        task_result = TaskResult(
            task_id=task_request.id,
            status=TaskStatus.RUNNING,
            agent_id=agent_id
        )
        self.task_results[task_request.id] = task_result
        
        try:
            # Execute task via Celery
            celery_task = self.celery_app.send_task(
                f'sutazai.agents.{task_request.type}',
                args=[task_request.payload],
                kwargs={
                    'task_id': task_request.id,
                    'agent_id': agent_id,
                    'context': task_request.context
                },
                queue=self.get_queue_for_task_type(task_request.type),
                countdown=0,
                expires=task_request.timeout
            )
            
            # Monitor task execution
            asyncio.create_task(
                self.monitor_task_execution(task_request, agent_id, celery_task)
            )
            
        except Exception as e:
            logger.error(f"Failed to execute task {task_request.id}: {e}")
            await self.handle_task_failure(task_request.id, agent_id, str(e))
    
    async def monitor_task_execution(self, task_request: TaskRequest, agent_id: str, celery_task):
        """Monitor task execution and handle completion"""
        start_time = datetime.now()
        
        try:
            # Wait for task completion with timeout
            result = celery_task.get(timeout=task_request.timeout)
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update task result
            task_result = TaskResult(
                task_id=task_request.id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                agent_id=agent_id,
                completed_at=datetime.now()
            )
            
            await self.handle_task_completion(task_result)
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            await self.handle_task_failure(task_request.id, agent_id, str(e), execution_time)
    
    async def handle_task_completion(self, task_result: TaskResult):
        """Handle successful task completion"""
        logger.info(f"Task {task_result.task_id} completed successfully")
        
        # Update task results
        self.task_results[task_result.task_id] = task_result
        
        # Update agent status
        if task_result.agent_id and task_result.agent_id in self.agents:
            agent = self.agents[task_result.agent_id]
            agent["status"] = AgentStatus.IDLE
            agent["current_task"] = None
            
            # Update performance metrics
            metrics = agent["performance_metrics"]
            metrics["tasks_completed"] += 1
            
            # Update average execution time
            if task_result.execution_time:
                current_avg = metrics["average_execution_time"]
                completed_tasks = metrics["tasks_completed"]
                metrics["average_execution_time"] = (
                    (current_avg * (completed_tasks - 1) + task_result.execution_time) / completed_tasks
                )
            
            # Update success rate
            total_tasks = metrics["tasks_completed"] + metrics["tasks_failed"]
            metrics["success_rate"] = metrics["tasks_completed"] / total_tasks if total_tasks > 0 else 1.0
        
        # Remove from active tasks
        if task_result.task_id in self.active_tasks:
            del self.active_tasks[task_result.task_id]
        
        # Store result in Redis
        await self.redis_client.hset(
            "sutazai:tasks:completed",
            task_result.task_id,
            json.dumps(task_result.__dict__, default=str)
        )
        
        # Process any dependent tasks
        await self.process_dependent_tasks(task_result.task_id)
        
        # Update system metrics
        self.metrics["tasks_completed"] += 1
    
    async def handle_task_failure(self, task_id: str, agent_id: str, error: str, execution_time: Optional[float] = None):
        """Handle task failure"""
        logger.error(f"Task {task_id} failed: {error}")
        
        # Update task result
        task_result = TaskResult(
            task_id=task_id,
            status=TaskStatus.FAILED,
            error=error,
            execution_time=execution_time,
            agent_id=agent_id,
            completed_at=datetime.now()
        )
        self.task_results[task_id] = task_result
        
        # Update agent status and metrics
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            agent["status"] = AgentStatus.IDLE
            agent["current_task"] = None
            agent["performance_metrics"]["tasks_failed"] += 1
            
            # Update success rate
            metrics = agent["performance_metrics"]
            total_tasks = metrics["tasks_completed"] + metrics["tasks_failed"]
            metrics["success_rate"] = metrics["tasks_completed"] / total_tasks if total_tasks > 0 else 1.0
        
        # Check if task should be retried
        if task_id in self.active_tasks:
            task_request = self.active_tasks[task_id]
            if task_request.retries > 0:
                task_request.retries -= 1
                logger.info(f"Retrying task {task_id} ({task_request.retries} retries left)")
                await self.queue_task(task_request)
            else:
                # Remove from active tasks
                del self.active_tasks[task_id]
        
        # Update system metrics
        self.metrics["tasks_failed"] += 1
    
    async def queue_task(self, task_request: TaskRequest):
        """Add task to the execution queue"""
        queue_name = self.get_queue_for_task_type(task_request.type)
        
        await self.redis_client.lpush(
            f"sutazai:queue:{queue_name}",
            json.dumps(task_request.__dict__, default=str)
        )
        
        logger.info(f"Task {task_request.id} added to queue {queue_name}")
    
    def get_queue_for_task_type(self, task_type: str) -> str:
        """Get the appropriate queue for a task type"""
        queue_mapping = {
            "code_generation": "code_generation",
            "document_processing": "documents", 
            "web_automation": "web_automation",
            "security_analysis": "security",
            "task_automation": "general",
            "multi_agent": "general",
            "code_completion": "code_generation"
        }
        
        return queue_mapping.get(task_type, "general")
    
    async def process_queued_tasks(self):
        """Process tasks from queues when agents become available"""
        for queue_name in ["code_generation", "documents", "web_automation", "security", "general"]:
            # Check if we have available agents for this queue type
            available_agents = [
                agent_id for agent_id, agent_data in self.agents.items()
                if agent_data["status"] == AgentStatus.IDLE and 
                self.can_agent_handle_queue(agent_data, queue_name)
            ]
            
            if not available_agents:
                continue
            
            # Get tasks from queue
            task_data = await self.redis_client.rpop(f"sutazai:queue:{queue_name}")
            if task_data:
                try:
                    task_dict = json.loads(task_data)
                    task_request = TaskRequest(**task_dict)
                    
                    # Find best available agent
                    best_agent = None
                    best_score = 0.0
                    
                    for agent_id in available_agents:
                        score = await self.calculate_agent_suitability(
                            self.agents[agent_id], task_request
                        )
                        if score > best_score:
                            best_score = score
                            best_agent = agent_id
                    
                    if best_agent:
                        await self.execute_task(task_request, best_agent)
                    else:
                        # Put task back in queue
                        await self.redis_client.lpush(f"sutazai:queue:{queue_name}", task_data)
                        
                except Exception as e:
                    logger.error(f"Failed to process queued task: {e}")
    
    def can_agent_handle_queue(self, agent_data: Dict[str, Any], queue_name: str) -> bool:
        """Check if an agent can handle tasks from a specific queue"""
        agent_type = agent_data["type"]
        
        queue_type_mapping = {
            "code_generation": ["code_generation", "code_completion"],
            "documents": ["document_processing"],
            "web_automation": ["web_automation"],
            "security": ["security_analysis"],
            "general": ["task_automation", "multi_agent"]
        }
        
        return agent_type in queue_type_mapping.get(queue_name, [])
    
    async def process_dependent_tasks(self, completed_task_id: str):
        """Process tasks that depend on the completed task"""
        # Find tasks waiting for this dependency
        dependent_tasks = []
        
        for task_id, task_request in self.active_tasks.items():
            if completed_task_id in task_request.dependencies:
                task_request.dependencies.remove(completed_task_id)
                if not task_request.dependencies:  # All dependencies satisfied
                    dependent_tasks.append(task_request)
        
        # Execute dependent tasks
        for task_request in dependent_tasks:
            suitable_agent = await self.find_suitable_agent(task_request)
            if suitable_agent:
                await self.execute_task(task_request, suitable_agent)
            else:
                await self.queue_task(task_request)
    
    async def get_task_status(self, task_id: str) -> Optional[TaskResult]:
        """Get the status of a task"""
        if task_id in self.task_results:
            return self.task_results[task_id]
        
        if task_id in self.active_tasks:
            return TaskResult(
                task_id=task_id,
                status=TaskStatus.RUNNING
            )
        
        # Check Redis for completed tasks
        result_data = await self.redis_client.hget("sutazai:tasks:completed", task_id)
        if result_data:
            result_dict = json.loads(result_data)
            return TaskResult(**result_dict)
        
        return None
    
    async def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of an agent"""
        return self.agents.get(agent_id)
    
    async def get_system_metrics(self) -> Dict[str, Any]:
        """Get system-wide metrics"""
        # Update real-time metrics
        self.metrics["active_agents"] = len([
            agent for agent in self.agents.values()
            if agent["status"] in [AgentStatus.IDLE, AgentStatus.BUSY]
        ])
        
        self.metrics["queue_length"] = 0
        for queue_name in ["code_generation", "documents", "web_automation", "security", "general"]:
            queue_length = await self.redis_client.llen(f"sutazai:queue:{queue_name}")
            self.metrics["queue_length"] += queue_length
        
        return self.metrics
    
    # Background monitoring loops
    
    async def agent_discovery_loop(self):
        """Periodically discover new agents"""
        while True:
            try:
                await asyncio.sleep(self.agent_discovery_interval)
                await self.discover_agents()
            except Exception as e:
                logger.error(f"Agent discovery loop error: {e}")
    
    async def health_check_loop(self):
        """Periodically check agent health"""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self.check_agent_health()
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
    
    async def check_agent_health(self):
        """Check health of all registered agents"""
        for agent_id, agent_data in list(self.agents.items()):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{agent_data['url']}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        if response.status == 200:
                            agent_data["status"] = AgentStatus.IDLE if agent_data["current_task"] is None else AgentStatus.BUSY
                            agent_data["last_heartbeat"] = datetime.now()
                        else:
                            agent_data["status"] = AgentStatus.ERROR
            except Exception as e:
                logger.warning(f"Health check failed for agent {agent_id}: {e}")
                agent_data["status"] = AgentStatus.OFFLINE
    
    async def task_timeout_check_loop(self):
        """Check for timed out tasks"""
        while True:
            try:
                await asyncio.sleep(self.task_timeout_check_interval)
                await self.check_task_timeouts()
            except Exception as e:
                logger.error(f"Task timeout check loop error: {e}")
    
    async def check_task_timeouts(self):
        """Check for and handle timed out tasks"""
        current_time = datetime.now()
        
        for task_id, task_request in list(self.active_tasks.items()):
            if task_id in self.task_results and self.task_results[task_id].status == TaskStatus.RUNNING:
                elapsed = (current_time - task_request.created_at).total_seconds()
                if elapsed > task_request.timeout:
                    logger.warning(f"Task {task_id} timed out after {elapsed} seconds")
                    
                    # Find the agent executing this task
                    agent_id = None
                    for aid, agent_data in self.agents.items():
                        if agent_data.get("current_task") == task_id:
                            agent_id = aid
                            break
                    
                    await self.handle_task_failure(
                        task_id, 
                        agent_id, 
                        f"Task timed out after {elapsed} seconds"
                    )
    
    async def metrics_update_loop(self):
        """Periodically update system metrics"""
        while True:
            try:
                await asyncio.sleep(60)  # Update every minute
                await self.update_system_metrics()
                await self.process_queued_tasks()  # Also process queued tasks
            except Exception as e:
                logger.error(f"Metrics update loop error: {e}")
    
    async def update_system_metrics(self):
        """Update system-wide performance metrics"""
        # Calculate average execution time across all completed tasks
        if self.metrics["tasks_completed"] > 0:
            total_execution_time = 0.0
            completed_count = 0
            
            for task_result in self.task_results.values():
                if task_result.status == TaskStatus.COMPLETED and task_result.execution_time:
                    total_execution_time += task_result.execution_time
                    completed_count += 1
            
            if completed_count > 0:
                self.metrics["average_execution_time"] = total_execution_time / completed_count
        
        # Store metrics in Redis for monitoring
        await self.redis_client.hset(
            "sutazai:system:metrics",
            "current",
            json.dumps(self.metrics, default=str)
        )
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down AGI Orchestrator...")
        
        # Cancel all running tasks
        for task_id in list(self.active_tasks.keys()):
            await self.handle_task_failure(task_id, None, "System shutdown")
        
        # Close connections
        if self.redis_client:
            await self.redis_client.close()
        
        if self.db_engine:
            await self.db_engine.dispose()
        
        logger.info("AGI Orchestrator shutdown complete")

# Global orchestrator instance
orchestrator = AGIOrchestrator()

# Celery tasks for agent execution
@orchestrator.celery_app.task(bind=True)
def execute_agent_task(self, payload: Dict[str, Any], task_id: str, agent_id: str, context: Dict[str, Any] = None):
    """
    Execute a task using the specified agent
    This runs in a Celery worker process
    """
    import asyncio
    
    async def _execute():
        try:
            # Get agent configuration
            agent_data = orchestrator.agents.get(agent_id)
            if not agent_data:
                raise ValueError(f"Agent {agent_id} not found")
            
            # Make HTTP request to agent
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{agent_data['url']}/execute",
                    json={
                        "payload": payload,
                        "task_id": task_id,
                        "context": context or {}
                    },
                    timeout=aiohttp.ClientTimeout(total=300)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result
                    else:
                        error_text = await response.text()
                        raise Exception(f"Agent execution failed: {error_text}")
        
        except Exception as e:
            logger.error(f"Task execution failed: {e}")
            raise
    
    # Run the async function in the current event loop
    return asyncio.run(_execute())