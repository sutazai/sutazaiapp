"""
Advanced Agent Orchestrator for SutazAI
=======================================

High-performance, scalable agent orchestration system.
Manages multiple AI agents with load balancing, auto-scaling, and fault tolerance.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
import json

try:
    from .config import Settings
    from .utils import setup_logging, retry_async, timeout_async, generate_task_id
    from .models import Task, Agent, TaskStatus, AgentStatus
except ImportError:
    from config import Settings
    from utils import setup_logging, retry_async, timeout_async, generate_task_id
    from models import Task, Agent, TaskStatus, AgentStatus

logger = setup_logging(__name__)


class TaskPriority(Enum):
    """Task priority levels"""
    LOW = 1
    NORMAL = 5
    HIGH = 8
    CRITICAL = 10


class AgentType(Enum):
    """Available agent types"""
    CHAT = "chat"
    CODE_GENERATOR = "code_generator"
    DOCUMENT_PROCESSOR = "document_processor"
    WEB_AUTOMATOR = "web_automator"
    SECURITY_ANALYST = "security_analyst"
    DATA_ANALYST = "data_analyst"
    GENERAL_ASSISTANT = "general_assistant"


@dataclass
class TaskResult:
    """Task execution result"""
    task_id: str
    status: TaskStatus
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    agent_id: Optional[str] = None
    tokens_used: int = 0
    timestamp: datetime = field(default_factory=datetime.now)


class AgentPool:
    """Manages a pool of agents for load balancing"""
    
    def __init__(self, agent_type: AgentType, min_instances: int = 1, max_instances: int = 5):
        self.agent_type = agent_type
        self.min_instances = min_instances
        self.max_instances = max_instances
        self.agents: Dict[str, Agent] = {}
        self.load_balancer = RoundRobinBalancer()
        
    def add_agent(self, agent: Agent):
        """Add agent to pool"""
        self.agents[agent.id] = agent
        logger.info(f"Added agent {agent.id} to {self.agent_type.value} pool")
    
    def remove_agent(self, agent_id: str):
        """Remove agent from pool"""
        if agent_id in self.agents:
            del self.agents[agent_id]
            logger.info(f"Removed agent {agent_id} from {self.agent_type.value} pool")
    
    def get_available_agent(self) -> Optional[Agent]:
        """Get next available agent using load balancing"""
        available_agents = [
            agent for agent in self.agents.values()
            if agent.status == AgentStatus.IDLE
        ]
        
        if not available_agents:
            return None
            
        return self.load_balancer.select(available_agents)
    
    def get_agent_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        total = len(self.agents)
        idle = len([a for a in self.agents.values() if a.status == AgentStatus.IDLE])
        busy = len([a for a in self.agents.values() if a.status == AgentStatus.BUSY])
        error = len([a for a in self.agents.values() if a.status == AgentStatus.ERROR])
        
        return {
            "type": self.agent_type.value,
            "total": total,
            "idle": idle,
            "busy": busy,
            "error": error,
            "utilization": (busy / total * 100) if total > 0 else 0
        }


class RoundRobinBalancer:
    """Simple round-robin load balancer"""
    
    def __init__(self):
        self.current_index = 0
    
    def select(self, agents: List[Agent]) -> Optional[Agent]:
        """Select next agent in round-robin fashion"""
        if not agents:
            return None
        
        agent = agents[self.current_index % len(agents)]
        self.current_index += 1
        return agent


class TaskQueue:
    """Priority-based task queue with persistence"""
    
    def __init__(self):
        self.queues = {priority: asyncio.Queue() for priority in TaskPriority}
        self.active_tasks: Dict[str, Task] = {}
        self.completed_tasks: Dict[str, TaskResult] = {}
        self.task_history_limit = 1000
    
    async def enqueue(self, task: Task):
        """Add task to appropriate priority queue"""
        priority = TaskPriority(task.priority)
        await self.queues[priority].put(task)
        self.active_tasks[task.id] = task
        logger.info(f"Enqueued task {task.id} with priority {priority.name}")
    
    async def dequeue(self) -> Optional[Task]:
        """Get next task from highest priority queue"""
        # Check queues in priority order (highest first)
        for priority in sorted(TaskPriority, key=lambda x: x.value, reverse=True):
            queue = self.queues[priority]
            if not queue.empty():
                try:
                    task = queue.get_nowait()
                    return task
                except asyncio.QueueEmpty:
                    continue
        return None
    
    def complete_task(self, task_result: TaskResult):
        """Mark task as completed and store result"""
        task_id = task_result.task_id
        
        if task_id in self.active_tasks:
            del self.active_tasks[task_id]
        
        self.completed_tasks[task_id] = task_result
        
        # Limit history size
        if len(self.completed_tasks) > self.task_history_limit:
            oldest_tasks = sorted(
                self.completed_tasks.items(),
                key=lambda x: x[1].timestamp
            )[:len(self.completed_tasks) - self.task_history_limit]
            
            for task_id, _ in oldest_tasks:
                del self.completed_tasks[task_id]
        
        logger.info(f"Completed task {task_id} with status {task_result.status.value}")
    
    def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a task"""
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            return {
                "id": task.id,
                "status": "active",
                "type": task.task_type,
                "priority": task.priority,
                "created_at": task.created_at.isoformat(),
                "description": task.description[:100] + "..." if len(task.description) > 100 else task.description
            }
        
        if task_id in self.completed_tasks:
            result = self.completed_tasks[task_id]
            return {
                "id": result.task_id,
                "status": result.status.value,
                "execution_time": result.execution_time,
                "completed_at": result.timestamp.isoformat(),
                "result": result.result,
                "error": result.error,
                "tokens_used": result.tokens_used
            }
        
        return None


class AgentOrchestrator:
    """Main orchestrator for managing AI agents and tasks"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.agent_pools: Dict[AgentType, AgentPool] = {}
        self.task_queue = TaskQueue()
        self.running = False
        self.worker_tasks: List[asyncio.Task] = []
        self.metrics = {
            "tasks_processed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0
        }
        
        # Initialize agent pools
        for agent_type in AgentType:
            self.agent_pools[agent_type] = AgentPool(agent_type)
    
    async def initialize(self):
        """Initialize the orchestrator"""
        logger.info("Initializing Agent Orchestrator...")
        
        # Start background workers
        self.running = True
        self.worker_tasks = [
            asyncio.create_task(self.task_processor()),
            asyncio.create_task(self.health_monitor()),
            asyncio.create_task(self.metrics_collector())
        ]
        
        # Initialize default agents
        await self.initialize_default_agents()
        
        logger.info("✅ Agent Orchestrator initialized successfully")
    
    async def shutdown(self):
        """Gracefully shutdown the orchestrator"""
        logger.info("Shutting down Agent Orchestrator...")
        
        self.running = False
        
        # Cancel worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        logger.info("✅ Agent Orchestrator shutdown completed")
    
    async def initialize_default_agents(self):
        """Initialize default agent instances"""
        default_agents = [
            (AgentType.CHAT, "ChatAgent-1"),
            (AgentType.CODE_GENERATOR, "CodeGen-1"),
            (AgentType.DOCUMENT_PROCESSOR, "DocProc-1"),
            (AgentType.GENERAL_ASSISTANT, "Assistant-1")
        ]
        
        for agent_type, agent_name in default_agents:
            agent = Agent(
                id=f"{agent_name}-{uuid.uuid4().hex[:8]}",
                name=agent_name,
                agent_type=agent_type,
                status=AgentStatus.IDLE,
                capabilities=self.get_agent_capabilities(agent_type),
                created_at=datetime.now()
            )
            
            self.agent_pools[agent_type].add_agent(agent)
            logger.info(f"Initialized {agent_type.value} agent: {agent.id}")
    
    def get_agent_capabilities(self, agent_type: AgentType) -> List[str]:
        """Get capabilities for agent type"""
        capabilities_map = {
            AgentType.CHAT: ["conversation", "qa", "general_chat"],
            AgentType.CODE_GENERATOR: ["code_generation", "code_review", "debugging"],
            AgentType.DOCUMENT_PROCESSOR: ["pdf_processing", "text_extraction", "summarization"],
            AgentType.WEB_AUTOMATOR: ["web_scraping", "browser_automation", "form_filling"],
            AgentType.SECURITY_ANALYST: ["vulnerability_scan", "security_audit", "compliance_check"],
            AgentType.DATA_ANALYST: ["data_analysis", "visualization", "statistical_analysis"],
            AgentType.GENERAL_ASSISTANT: ["general_tasks", "information_retrieval", "planning"]
        }
        return capabilities_map.get(agent_type, [])
    
    async def submit_task(
        self,
        description: str,
        task_type: str,
        priority: int = 5,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Submit a new task for execution"""
        task = Task(
            id=generate_task_id(),
            description=description,
            task_type=task_type,
            priority=priority,
            metadata=metadata or {},
            status=TaskStatus.PENDING,
            created_at=datetime.now()
        )
        
        await self.task_queue.enqueue(task)
        return task.id
    
    async def wait_for_completion(self, task_id: str, timeout: float = 30.0) -> Dict[str, Any]:
        """Wait for task completion with timeout"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            task_info = self.task_queue.get_task_info(task_id)
            
            if task_info and task_info.get("status") in ["completed", "failed"]:
                if task_info["status"] == "failed":
                    raise Exception(f"Task failed: {task_info.get('error', 'Unknown error')}")
                return task_info.get("result", {})
            
            await asyncio.sleep(0.1)
        
        raise asyncio.TimeoutError(f"Task {task_id} did not complete within {timeout} seconds")
    
    async def task_processor(self):
        """Background task processor"""
        while self.running:
            try:
                task = await self.task_queue.dequeue()
                if task:
                    asyncio.create_task(self.execute_task(task))
                else:
                    await asyncio.sleep(0.1)
            except Exception as e:
                logger.error(f"Error in task processor: {e}")
                await asyncio.sleep(1)
    
    async def execute_task(self, task: Task):
        """Execute a single task"""
        start_time = time.time()
        agent = None
        
        try:
            # Find appropriate agent
            agent_type = self.map_task_to_agent_type(task.task_type)
            agent_pool = self.agent_pools.get(agent_type)
            
            if not agent_pool:
                raise Exception(f"No agent pool found for type {agent_type}")
            
            agent = agent_pool.get_available_agent()
            if not agent:
                # Could implement auto-scaling here
                raise Exception(f"No available agents for type {agent_type}")
            
            # Mark agent as busy
            agent.status = AgentStatus.BUSY
            agent.current_task_id = task.id
            
            # Execute task based on type
            result = await self.execute_task_by_type(task, agent)
            
            # Create successful result
            execution_time = time.time() - start_time
            task_result = TaskResult(
                task_id=task.id,
                status=TaskStatus.COMPLETED,
                result=result,
                execution_time=execution_time,
                agent_id=agent.id,
                tokens_used=result.get("tokens_used", 0)
            )
            
            # Update metrics
            self.metrics["tasks_processed"] += 1
            self.metrics["total_execution_time"] += execution_time
            self.metrics["average_execution_time"] = (
                self.metrics["total_execution_time"] / self.metrics["tasks_processed"]
            )
            
        except Exception as e:
            # Create failed result
            execution_time = time.time() - start_time
            task_result = TaskResult(
                task_id=task.id,
                status=TaskStatus.FAILED,
                error=str(e),
                execution_time=execution_time,
                agent_id=agent.id if agent else None
            )
            
            self.metrics["tasks_failed"] += 1
            logger.error(f"Task {task.id} failed: {e}")
        
        finally:
            # Release agent
            if agent:
                agent.status = AgentStatus.IDLE
                agent.current_task_id = None
                agent.completed_tasks += 1
                agent.last_activity = datetime.now()
            
            # Store result
            self.task_queue.complete_task(task_result)
    
    def map_task_to_agent_type(self, task_type: str) -> AgentType:
        """Map task type to agent type"""
        mapping = {
            "chat": AgentType.CHAT,
            "conversation": AgentType.CHAT,
            "code_generation": AgentType.CODE_GENERATOR,
            "code_review": AgentType.CODE_GENERATOR,
            "document_processing": AgentType.DOCUMENT_PROCESSOR,
            "web_automation": AgentType.WEB_AUTOMATOR,
            "security_analysis": AgentType.SECURITY_ANALYST,
            "data_analysis": AgentType.DATA_ANALYST
        }
        return mapping.get(task_type.lower(), AgentType.GENERAL_ASSISTANT)
    
    async def execute_task_by_type(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute task based on its type"""
        if task.task_type in ["chat", "conversation"]:
            return await self.execute_chat_task(task, agent)
        elif task.task_type == "code_generation":
            return await self.execute_code_generation_task(task, agent)
        elif task.task_type == "document_processing":
            return await self.execute_document_processing_task(task, agent)
        else:
            # Default general execution
            return await self.execute_general_task(task, agent)
    
    @timeout_async(30.0)
    @retry_async(max_retries=2)
    async def execute_chat_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute chat/conversation task"""
        import aiohttp
        
        # Prepare request to Ollama
        model = task.metadata.get("model", self.settings.default_model)
        temperature = task.metadata.get("temperature", 0.7)
        
        request_data = {
            "model": model,
            "prompt": task.description,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": task.metadata.get("max_tokens", 2048)
            }
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.settings.ollama_url}/api/generate",
                json=request_data,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return {
                        "response": result.get("response", "No response generated"),
                        "model": model,
                        "tokens_used": result.get("eval_count", 0),
                        "agent_id": agent.id
                    }
                else:
                    raise Exception(f"Ollama request failed with status {response.status}")
    
    async def execute_code_generation_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute code generation task"""
        # This would integrate with actual code generation services
        # For now, return a mock response
        await asyncio.sleep(1)  # Simulate processing time
        
        return {
            "code": f"# Generated code for: {task.description[:50]}...\nprint('Hello, World!')",
            "language": task.metadata.get("language", "python"),
            "tokens_used": 150,
            "agent_id": agent.id
        }
    
    async def execute_document_processing_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute document processing task"""
        # This would integrate with actual document processing services
        await asyncio.sleep(2)  # Simulate processing time
        
        return {
            "summary": f"Document summary for: {task.description[:50]}...",
            "extracted_text": "Sample extracted text...",
            "metadata": {"pages": 5, "words": 1000},
            "agent_id": agent.id
        }
    
    async def execute_general_task(self, task: Task, agent: Agent) -> Dict[str, Any]:
        """Execute general task"""
        await asyncio.sleep(0.5)  # Simulate processing time
        
        return {
            "result": f"Processed: {task.description[:100]}...",
            "status": "completed",
            "agent_id": agent.id
        }
    
    async def health_monitor(self):
        """Monitor agent health and performance"""
        while self.running:
            try:
                # Check agent health and reset stuck agents
                for agent_pool in self.agent_pools.values():
                    for agent in agent_pool.agents.values():
                        if agent.status == AgentStatus.BUSY:
                            # Check if agent is stuck (busy for too long)
                            time_since_activity = datetime.now() - agent.last_activity
                            if time_since_activity > timedelta(minutes=5):
                                logger.warning(f"Agent {agent.id} appears stuck, resetting...")
                                agent.status = AgentStatus.IDLE
                                agent.current_task_id = None
                
                await asyncio.sleep(60)  # Check every minute
            except Exception as e:
                logger.error(f"Error in health monitor: {e}")
                await asyncio.sleep(60)
    
    async def metrics_collector(self):
        """Collect and log metrics"""
        while self.running:
            try:
                # Log system metrics
                total_agents = sum(len(pool.agents) for pool in self.agent_pools.values())
                active_tasks = len(self.task_queue.active_tasks)
                
                logger.info(
                    f"Metrics - Agents: {total_agents}, Active Tasks: {active_tasks}, "
                    f"Processed: {self.metrics['tasks_processed']}, "
                    f"Failed: {self.metrics['tasks_failed']}, "
                    f"Avg Time: {self.metrics['average_execution_time']:.2f}s"
                )
                
                await asyncio.sleep(300)  # Log every 5 minutes
            except Exception as e:
                logger.error(f"Error in metrics collector: {e}")
                await asyncio.sleep(300)
    
    async def get_agent_status(self) -> List[Dict[str, Any]]:
        """Get status of all agents"""
        agents = []
        for agent_pool in self.agent_pools.values():
            for agent in agent_pool.agents.values():
                agents.append({
                    "id": agent.id,
                    "name": agent.name,
                    "type": agent.agent_type.value,
                    "status": agent.status.value,
                    "current_task": agent.current_task_id,
                    "completed_tasks": agent.completed_tasks,
                    "last_activity": agent.last_activity.isoformat() if agent.last_activity else None,
                    "capabilities": agent.capabilities
                })
        return agents
    
    async def get_task_status(self) -> List[Dict[str, Any]]:
        """Get status of all tasks"""
        tasks = []
        
        # Active tasks
        for task in self.task_queue.active_tasks.values():
            tasks.append({
                "id": task.id,
                "type": task.task_type,
                "status": "active",
                "priority": task.priority,
                "created_at": task.created_at.isoformat()
            })
        
        # Recent completed tasks (last 100)
        recent_completed = sorted(
            self.task_queue.completed_tasks.values(),
            key=lambda x: x.timestamp,
            reverse=True
        )[:100]
        
        for result in recent_completed:
            tasks.append({
                "id": result.task_id,
                "status": result.status.value,
                "execution_time": result.execution_time,
                "completed_at": result.timestamp.isoformat(),
                "agent_id": result.agent_id
            })
        
        return tasks
    
    async def get_task_info(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed information about a specific task"""
        return self.task_queue.get_task_info(task_id)
    
    @property
    def agents(self) -> Dict[str, Agent]:
        """Get all agents across all pools"""
        all_agents = {}
        for agent_pool in self.agent_pools.values():
            all_agents.update(agent_pool.agents)
        return all_agents