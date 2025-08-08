"""
Orchestration Controller - Advanced Agent Workflow Management
============================================================

This controller manages complex multi-agent workflows, task distribution,
and autonomous agent coordination. It operates independently using local
models and provides sophisticated orchestration capabilities.

Features:
- Multi-agent workflow orchestration
- Dynamic task decomposition and distribution
- Autonomous agent coordination
- Workflow state management
- Error handling and recovery
- Performance optimization
- Resource allocation
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import datetime, timedelta
from enum import Enum
from app.schemas.message_types import TaskPriority
from typing import Any, Dict, List, Optional, Set, Union, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque

from .base_agent import BaseAgent, AgentMessage, AgentStatus, AgentCapability
from .agent_message_bus import AgentMessageBus, MessageRoute, RoutingStrategy
from .universal_agent_factory import UniversalAgentFactory


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Individual task status"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


# Use canonical TaskPriority from app.schemas.message_types


@dataclass
class Task:
    """Individual task within a workflow"""
    id: str
    name: str
    description: str
    task_type: str
    priority: TaskPriority
    required_capabilities: List[AgentCapability]
    input_data: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0
    max_retries: int = 3
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "task_type": self.task_type,
            "priority": TaskPriority.from_value(self.priority).rank,
            "required_capabilities": [cap.value for cap in self.required_capabilities],
            "input_data": self.input_data,
            "dependencies": self.dependencies,
            "assigned_agent": self.assigned_agent,
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "timeout_seconds": self.timeout_seconds,
            "metadata": self.metadata
        }


@dataclass
class Workflow:
    """Multi-agent workflow definition"""
    id: str
    name: str
    description: str
    tasks: Dict[str, Task]
    status: WorkflowStatus = WorkflowStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error: Optional[str] = None
    progress: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_ready_tasks(self) -> List[Task]:
        """Get tasks that are ready to execute (dependencies met)"""
        ready_tasks = []
        
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
                
            # Check if all dependencies are completed
            dependencies_met = True
            for dep_id in task.dependencies:
                dep_task = self.tasks.get(dep_id)
                if not dep_task or dep_task.status != TaskStatus.COMPLETED:
                    dependencies_met = False
                    break
            
            if dependencies_met:
                ready_tasks.append(task)
        
        return sorted(ready_tasks, key=lambda t: t.priority.value)
    
    def get_completion_percentage(self) -> float:
        """Calculate workflow completion percentage"""
        if not self.tasks:
            return 0.0
        
        completed_tasks = sum(1 for task in self.tasks.values() 
                            if task.status == TaskStatus.COMPLETED)
        return (completed_tasks / len(self.tasks)) * 100.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error": self.error,
            "progress": self.progress,
            "metadata": self.metadata
        }


class OrchestrationController:
    """
    Advanced Agent Orchestration Controller
    
    Manages complex multi-agent workflows, coordinates task execution,
    and provides autonomous agent management capabilities.
    """
    
    def __init__(self, agent_factory: UniversalAgentFactory,
                 message_bus: AgentMessageBus,
                 max_concurrent_workflows: int = 10,
                 max_concurrent_tasks: int = 50):
        
        self.agent_factory = agent_factory
        self.message_bus = message_bus
        self.max_concurrent_workflows = max_concurrent_workflows
        self.max_concurrent_tasks = max_concurrent_tasks
        
        # Workflow management
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Set[str] = set()
        self.task_queue: deque = deque()
        self.assigned_tasks: Dict[str, str] = {}  # task_id -> agent_id
        
        # Agent management
        self.available_agents: Dict[str, BaseAgent] = {}
        self.agent_workloads: Dict[str, int] = defaultdict(int)
        self.agent_capabilities_cache: Dict[str, Set[AgentCapability]] = {}
        
        # Performance tracking
        self.execution_stats = {
            "workflows_completed": 0,
            "workflows_failed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "average_task_duration": 0.0,
            "agent_utilization": {}
        }
        
        # Background processing
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown_event = asyncio.Event()
        
        self.logger = logging.getLogger("orchestration_controller")
    
    async def initialize(self) -> bool:
        """Initialize the orchestration controller"""
        try:
            self.logger.info("Initializing Orchestration Controller")
            
            # Register message handlers
            self.message_bus.register_message_handler("task_result", self._handle_task_result)
            self.message_bus.register_message_handler("task_error", self._handle_task_error)
            self.message_bus.register_message_handler("agent_status", self._handle_agent_status)
            
            # Start background tasks
            self._start_background_tasks()
            
            # Load available agents
            await self._refresh_agent_registry()
            
            self.logger.info("Orchestration Controller initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize orchestration controller: {e}")
            return False
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        tasks = [
            self._workflow_processor(),
            self._task_scheduler(),
            self._agent_monitor(),
            self._performance_tracker(),
            self._cleanup_completed_workflows()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
    
    async def _refresh_agent_registry(self):
        """Refresh available agents from factory"""
        self.available_agents = self.agent_factory.get_active_agents()
        
        # Update capability cache
        for agent_id, agent in self.available_agents.items():
            self.agent_capabilities_cache[agent_id] = agent.capabilities
            if agent_id not in self.agent_workloads:
                self.agent_workloads[agent_id] = 0
    
    async def create_workflow(self, workflow_spec: Dict[str, Any]) -> str:
        """Create a new workflow from specification"""
        try:
            workflow_id = workflow_spec.get("id", str(uuid.uuid4()))
            
            # Create tasks
            tasks = {}
            for task_spec in workflow_spec.get("tasks", []):
                task = Task(
                    id=task_spec["id"],
                    name=task_spec["name"],
                    description=task_spec.get("description", ""),
                    task_type=task_spec["task_type"],
                    priority=TaskPriority.from_value(task_spec.get("priority", 3)),
                    required_capabilities=[
                        AgentCapability(cap) for cap in task_spec.get("required_capabilities", [])
                    ],
                    input_data=task_spec.get("input_data", {}),
                    dependencies=task_spec.get("dependencies", []),
                    max_retries=task_spec.get("max_retries", 3),
                    timeout_seconds=task_spec.get("timeout_seconds", 300),
                    metadata=task_spec.get("metadata", {})
                )
                tasks[task.id] = task
            
            # Create workflow
            workflow = Workflow(
                id=workflow_id,
                name=workflow_spec["name"],
                description=workflow_spec.get("description", ""),
                tasks=tasks,
                metadata=workflow_spec.get("metadata", {})
            )
            
            self.workflows[workflow_id] = workflow
            
            self.logger.info(f"Created workflow {workflow_id} with {len(tasks)} tasks")
            return workflow_id
            
        except Exception as e:
            self.logger.error(f"Failed to create workflow: {e}")
            raise
    
    async def start_workflow(self, workflow_id: str) -> bool:
        """Start executing a workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if len(self.active_workflows) >= self.max_concurrent_workflows:
            self.logger.warning(f"Maximum concurrent workflows reached")
            return False
        
        workflow = self.workflows[workflow_id]
        
        if workflow.status != WorkflowStatus.PENDING:
            raise ValueError(f"Workflow {workflow_id} is not in pending state")
        
        workflow.status = WorkflowStatus.RUNNING
        workflow.started_at = datetime.utcnow()
        self.active_workflows.add(workflow_id)
        
        # Queue initial ready tasks
        ready_tasks = workflow.get_ready_tasks()
        for task in ready_tasks:
            self.task_queue.append((workflow_id, task.id))
        
        self.logger.info(f"Started workflow {workflow_id}")
        return True
    
    async def pause_workflow(self, workflow_id: str):
        """Pause workflow execution"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        if workflow.status == WorkflowStatus.RUNNING:
            workflow.status = WorkflowStatus.PAUSED
            self.logger.info(f"Paused workflow {workflow_id}")
    
    async def resume_workflow(self, workflow_id: str):
        """Resume paused workflow"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        if workflow.status == WorkflowStatus.PAUSED:
            workflow.status = WorkflowStatus.RUNNING
            
            # Re-queue ready tasks
            ready_tasks = workflow.get_ready_tasks()
            for task in ready_tasks:
                self.task_queue.append((workflow_id, task.id))
                
            self.logger.info(f"Resumed workflow {workflow_id}")
    
    async def cancel_workflow(self, workflow_id: str):
        """Cancel workflow execution"""
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow = self.workflows[workflow_id]
        workflow.status = WorkflowStatus.CANCELLED
        workflow.completed_at = datetime.utcnow()
        
        # Cancel running tasks
        for task in workflow.tasks.values():
            if task.status == TaskStatus.IN_PROGRESS:
                await self._cancel_task(task)
        
        self.active_workflows.discard(workflow_id)
        self.logger.info(f"Cancelled workflow {workflow_id}")
    
    async def _workflow_processor(self):
        """Background task to process workflow state changes"""
        while not self._shutdown_event.is_set():
            try:
                for workflow_id in list(self.active_workflows):
                    workflow = self.workflows[workflow_id]
                    
                    if workflow.status != WorkflowStatus.RUNNING:
                        continue
                    
                    # Update progress
                    workflow.progress = workflow.get_completion_percentage()
                    
                    # Check if workflow is complete
                    if all(task.status in [TaskStatus.COMPLETED, TaskStatus.CANCELLED] 
                          for task in workflow.tasks.values()):
                        
                        # Check if any tasks failed
                        failed_tasks = [task for task in workflow.tasks.values() 
                                      if task.status == TaskStatus.FAILED]
                        
                        if failed_tasks:
                            workflow.status = WorkflowStatus.FAILED
                            workflow.error = f"{len(failed_tasks)} tasks failed"
                        else:
                            workflow.status = WorkflowStatus.COMPLETED
                        
                        workflow.completed_at = datetime.utcnow()
                        self.active_workflows.discard(workflow_id)
                        
                        if workflow.status == WorkflowStatus.COMPLETED:
                            self.execution_stats["workflows_completed"] += 1
                            self.logger.info(f"Workflow {workflow_id} completed successfully")
                        else:
                            self.execution_stats["workflows_failed"] += 1
                            self.logger.error(f"Workflow {workflow_id} failed: {workflow.error}")
                    
                    # Queue newly ready tasks
                    ready_tasks = workflow.get_ready_tasks()
                    for task in ready_tasks:
                        if (workflow_id, task.id) not in [(w, t) for w, t in self.task_queue]:
                            self.task_queue.append((workflow_id, task.id))
                
                await asyncio.sleep(1)  # Process every second
                
            except Exception as e:
                self.logger.error(f"Workflow processor error: {e}")
                await asyncio.sleep(5)
    
    async def _task_scheduler(self):
        """Background task to schedule tasks to agents"""
        while not self._shutdown_event.is_set():
            try:
                if not self.task_queue:
                    await asyncio.sleep(0.5)
                    continue
                
                if len(self.assigned_tasks) >= self.max_concurrent_tasks:
                    await asyncio.sleep(1)
                    continue
                
                # Get next task
                workflow_id, task_id = self.task_queue.popleft()
                workflow = self.workflows.get(workflow_id)
                
                if not workflow or workflow.status != WorkflowStatus.RUNNING:
                    continue
                
                task = workflow.tasks.get(task_id)
                if not task or task.status != TaskStatus.PENDING:
                    continue
                
                # Find suitable agent
                agent_id = await self._find_best_agent(task)
                
                if agent_id:
                    await self._assign_task(workflow_id, task, agent_id)
                else:
                    # No agent available, put task back in queue
                    self.task_queue.append((workflow_id, task_id))
                    await asyncio.sleep(2)  # Wait before retrying
                
            except Exception as e:
                self.logger.error(f"Task scheduler error: {e}")
                await asyncio.sleep(1)
    
    async def _find_best_agent(self, task: Task) -> Optional[str]:
        """Find the best agent for a task"""
        await self._refresh_agent_registry()
        
        suitable_agents = []
        
        for agent_id, agent in self.available_agents.items():
            # Check if agent has required capabilities
            if not all(cap in agent.capabilities for cap in task.required_capabilities):
                continue
            
            # Check agent status
            if agent.status not in [AgentStatus.IDLE, AgentStatus.ACTIVE]:
                continue
            
            # Check agent capacity
            current_workload = self.agent_workloads.get(agent_id, 0)
            if current_workload >= agent.config.max_concurrent_tasks:
                continue
            
            suitable_agents.append((agent_id, current_workload))
        
        if not suitable_agents:
            return None
        
        # Select agent with lowest workload
        return min(suitable_agents, key=lambda x: x[1])[0]
    
    async def _assign_task(self, workflow_id: str, task: Task, agent_id: str):
        """Assign task to an agent"""
        try:
            task.assigned_agent = agent_id
            task.status = TaskStatus.ASSIGNED
            task.started_at = datetime.utcnow()
            
            self.assigned_tasks[task.id] = agent_id
            self.agent_workloads[agent_id] += 1
            
            # Send task to agent
            task_message = AgentMessage(
                sender_id="orchestrator",
                receiver_id=agent_id,
                message_type="execute_task",
                content={
                    "workflow_id": workflow_id,
                    "task_id": task.id,
                    "task_type": task.task_type,
                    "input_data": task.input_data,
                    "timeout_seconds": task.timeout_seconds,
                    "metadata": task.metadata
                },
                priority=task.priority.value
            )
            
            await self.message_bus.send_message(task_message)
            
            task.status = TaskStatus.IN_PROGRESS
            
            self.logger.info(f"Assigned task {task.id} to agent {agent_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to assign task {task.id}: {e}")
            task.status = TaskStatus.PENDING
            self.assigned_tasks.pop(task.id, None)
            self.agent_workloads[agent_id] = max(0, self.agent_workloads[agent_id] - 1)
    
    async def _handle_task_result(self, message: AgentMessage):
        """Handle task completion result"""
        content = message.content
        task_id = content.get("task_id")
        workflow_id = content.get("workflow_id")
        
        if not task_id or not workflow_id:
            return
        
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return
        
        task = workflow.tasks.get(task_id)
        if not task:
            return
        
        # Update task
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.utcnow()
        task.result = content.get("result")
        
        # Update agent workload
        agent_id = self.assigned_tasks.pop(task_id, None)
        if agent_id:
            self.agent_workloads[agent_id] = max(0, self.agent_workloads[agent_id] - 1)
        
        self.execution_stats["tasks_completed"] += 1
        
        # Update average task duration
        if task.started_at:
            duration = (task.completed_at - task.started_at).total_seconds()
            current_avg = self.execution_stats["average_task_duration"]
            total_tasks = self.execution_stats["tasks_completed"]
            self.execution_stats["average_task_duration"] = (
                (current_avg * (total_tasks - 1) + duration) / total_tasks
            )
        
        self.logger.info(f"Task {task_id} completed by agent {agent_id}")
    
    async def _handle_task_error(self, message: AgentMessage):
        """Handle task execution error"""
        content = message.content
        task_id = content.get("task_id")
        workflow_id = content.get("workflow_id")
        error = content.get("error", "Unknown error")
        
        if not task_id or not workflow_id:
            return
        
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return
        
        task = workflow.tasks.get(task_id)
        if not task:
            return
        
        # Update agent workload
        agent_id = self.assigned_tasks.pop(task_id, None)
        if agent_id:
            self.agent_workloads[agent_id] = max(0, self.agent_workloads[agent_id] - 1)
        
        # Handle retry logic
        if task.retry_count < task.max_retries:
            task.retry_count += 1
            task.status = TaskStatus.RETRYING
            task.assigned_agent = None
            
            # Re-queue task
            self.task_queue.append((workflow_id, task_id))
            
            self.logger.warning(f"Task {task_id} failed, retrying ({task.retry_count}/{task.max_retries})")
        else:
            task.status = TaskStatus.FAILED
            task.error = error
            task.completed_at = datetime.utcnow()
            
            self.execution_stats["tasks_failed"] += 1
            
            self.logger.error(f"Task {task_id} failed permanently: {error}")
    
    async def _handle_agent_status(self, message: AgentMessage):
        """Handle agent status updates"""
        agent_id = message.sender_id
        status_info = message.content
        
        # Update agent utilization stats
        active_tasks = status_info.get("active_tasks", 0)
        self.execution_stats["agent_utilization"][agent_id] = active_tasks
    
    async def _cancel_task(self, task: Task):
        """Cancel a running task"""
        if task.assigned_agent:
            cancel_message = AgentMessage(
                sender_id="orchestrator",
                receiver_id=task.assigned_agent,
                message_type="cancel_task",
                content={"task_id": task.id}
            )
            
            await self.message_bus.send_message(cancel_message)
            
            # Update workload
            self.agent_workloads[task.assigned_agent] = max(
                0, self.agent_workloads[task.assigned_agent] - 1
            )
        
        task.status = TaskStatus.CANCELLED
        task.completed_at = datetime.utcnow()
        self.assigned_tasks.pop(task.id, None)
    
    async def _agent_monitor(self):
        """Monitor agent health and availability"""
        while not self._shutdown_event.is_set():
            try:
                await self._refresh_agent_registry()
                
                # Remove dead agents from workload tracking
                dead_agents = set(self.agent_workloads.keys()) - set(self.available_agents.keys())
                for agent_id in dead_agents:
                    del self.agent_workloads[agent_id]
                    self.agent_capabilities_cache.pop(agent_id, None)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Agent monitor error: {e}")
                await asyncio.sleep(10)
    
    async def _performance_tracker(self):
        """Track performance metrics"""
        while not self._shutdown_event.is_set():
            try:
                # Log performance stats periodically
                if self.execution_stats["workflows_completed"] > 0:
                    success_rate = (
                        self.execution_stats["workflows_completed"] / 
                        (self.execution_stats["workflows_completed"] + 
                         self.execution_stats["workflows_failed"])
                    ) * 100
                    
                    self.logger.info(
                        f"Performance: {success_rate:.1f}% workflow success rate, "
                        f"avg task duration: {self.execution_stats['average_task_duration']:.2f}s"
                    )
                
                await asyncio.sleep(60)  # Log every minute
                
            except Exception as e:
                self.logger.error(f"Performance tracker error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_completed_workflows(self):
        """Clean up old completed workflows"""
        while not self._shutdown_event.is_set():
            try:
                cutoff_time = datetime.utcnow() - timedelta(hours=24)  # Keep for 24 hours
                
                workflows_to_remove = []
                for workflow_id, workflow in self.workflows.items():
                    if (workflow.status in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED, 
                                          WorkflowStatus.CANCELLED] and
                        workflow.completed_at and workflow.completed_at < cutoff_time):
                        workflows_to_remove.append(workflow_id)
                
                for workflow_id in workflows_to_remove:
                    del self.workflows[workflow_id]
                    self.logger.info(f"Cleaned up old workflow {workflow_id}")
                
                await asyncio.sleep(3600)  # Clean up every hour
                
            except Exception as e:
                self.logger.error(f"Cleanup error: {e}")
                await asyncio.sleep(3600)
    
    # Public query methods
    def get_workflow(self, workflow_id: str) -> Optional[Workflow]:
        """Get workflow by ID"""
        return self.workflows.get(workflow_id)
    
    def get_all_workflows(self) -> Dict[str, Workflow]:
        """Get all workflows"""
        return self.workflows.copy()
    
    def get_active_workflows(self) -> List[Workflow]:
        """Get all active workflows"""
        return [workflow for workflow in self.workflows.values() 
                if workflow.status == WorkflowStatus.RUNNING]
    
    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed workflow status"""
        workflow = self.workflows.get(workflow_id)
        if not workflow:
            return None
        
        return {
            "id": workflow.id,
            "name": workflow.name,
            "status": workflow.status.value,
            "progress": workflow.progress,
            "total_tasks": len(workflow.tasks),
            "completed_tasks": sum(1 for task in workflow.tasks.values() 
                                 if task.status == TaskStatus.COMPLETED),
            "failed_tasks": sum(1 for task in workflow.tasks.values() 
                               if task.status == TaskStatus.FAILED),
            "running_tasks": sum(1 for task in workflow.tasks.values() 
                                if task.status == TaskStatus.IN_PROGRESS),
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat() if workflow.started_at else None,
            "completed_at": workflow.completed_at.isoformat() if workflow.completed_at else None,
            "error": workflow.error
        }
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        return self.execution_stats.copy()
    
    def get_agent_workloads(self) -> Dict[str, int]:
        """Get current agent workloads"""
        return self.agent_workloads.copy()
    
    async def shutdown(self):
        """Shutdown the orchestration controller"""
        self.logger.info("Shutting down Orchestration Controller")
        
        # Cancel all active workflows
        for workflow_id in list(self.active_workflows):
            await self.cancel_workflow(workflow_id)
        
        # Signal shutdown to background tasks
        self._shutdown_event.set()
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        self.logger.info("Orchestration Controller shutdown complete")


# Global orchestration controller
_orchestration_controller: Optional[OrchestrationController] = None


def get_orchestration_controller() -> Optional[OrchestrationController]:
    """Get the global orchestration controller"""
    return _orchestration_controller


def set_orchestration_controller(controller: OrchestrationController):
    """Set the global orchestration controller"""
    global _orchestration_controller
    _orchestration_controller = controller
