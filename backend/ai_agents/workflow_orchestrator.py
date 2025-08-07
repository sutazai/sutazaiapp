"""
Multi-Agent Workflow Orchestration Framework
Provides advanced workflow orchestration, task dependencies, and complex multi-agent coordination.
"""

import asyncio
import json
import logging
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Union, Callable, Awaitable
import networkx as nx
from universal_client import (
    UniversalAgentClient, AgentType, TaskRequest, TaskResponse, Priority
)
from discovery_service import DiscoveryService, AgentMatch

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status."""
    PENDING = "pending"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class TaskStatus(Enum):
    """Individual task status."""
    WAITING = "waiting"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    RETRYING = "retrying"


class ExecutionStrategy(Enum):
    """Workflow execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class RetryStrategy(Enum):
    """Task retry strategies."""
    NONE = "none"
    FIXED_DELAY = "fixed_delay"
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"


@dataclass
class WorkflowTask:
    """Represents a single task in a workflow."""
    id: str
    name: str
    description: str
    agent_type: Optional[Union[AgentType, str]] = None
    agent_requirements: List[str] = field(default_factory=list)
    parameters: Dict[str, Any] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)
    timeout: int = 300
    retry_count: int = 0
    max_retries: int = 3
    retry_strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    priority: Priority = Priority.MEDIUM
    status: TaskStatus = TaskStatus.WAITING
    result: Any = None
    error: Optional[str] = None
    assigned_agent: Optional[str] = None
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Conditional execution
    condition: Optional[Callable[[Dict[str, Any]], bool]] = None
    skip_on_failure: bool = False
    
    # Task transformation
    input_transform: Optional[Callable[[Dict[str, Any]], Dict[str, Any]]] = None
    output_transform: Optional[Callable[[Any], Any]] = None


@dataclass
class WorkflowDefinition:
    """Complete workflow definition."""
    id: str
    name: str
    description: str
    tasks: List[WorkflowTask]
    execution_strategy: ExecutionStrategy = ExecutionStrategy.HYBRID
    max_parallel_tasks: int = 5
    timeout: int = 3600  # 1 hour default
    retry_failed_workflow: bool = False
    max_workflow_retries: int = 1
    
    # Workflow-level parameters
    global_parameters: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, Any] = field(default_factory=dict)
    
    # Event handlers
    on_start: Optional[Callable[[], Awaitable[None]]] = None
    on_complete: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None
    on_failure: Optional[Callable[[Exception], Awaitable[None]]] = None
    on_task_complete: Optional[Callable[[WorkflowTask], Awaitable[None]]] = None
    on_task_failure: Optional[Callable[[WorkflowTask, Exception], Awaitable[None]]] = None


@dataclass
class WorkflowExecution:
    """Runtime workflow execution state."""
    workflow_id: str
    execution_id: str
    definition: WorkflowDefinition
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    current_context: Dict[str, Any] = field(default_factory=dict)
    task_results: Dict[str, Any] = field(default_factory=dict)
    execution_graph: Optional[nx.DiGraph] = None
    active_tasks: Set[str] = field(default_factory=set)
    completed_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    retry_count: int = 0
    error: Optional[str] = None
    metrics: Dict[str, Any] = field(default_factory=dict)


class TaskScheduler:
    """Intelligent task scheduler for workflow orchestration."""
    
    def __init__(self, discovery_service: DiscoveryService):
        self.discovery_service = discovery_service
        self.task_queue: deque = deque()
        self.running_tasks: Dict[str, asyncio.Task] = {}
        self.task_history: Dict[str, List[TaskResponse]] = defaultdict(list)
    
    def can_execute_task(self, task: WorkflowTask, completed_tasks: Set[str]) -> bool:
        """Check if a task can be executed based on dependencies."""
        if task.status != TaskStatus.WAITING:
            return False
        
        # Check if all dependencies are completed
        for dep_id in task.dependencies:
            if dep_id not in completed_tasks:
                return False
        
        return True
    
    def get_ready_tasks(
        self,
        workflow_execution: WorkflowExecution
    ) -> List[WorkflowTask]:
        """Get tasks that are ready to execute."""
        ready_tasks = []
        
        for task in workflow_execution.definition.tasks:
            if (task.status == TaskStatus.WAITING and 
                self.can_execute_task(task, workflow_execution.completed_tasks)):
                
                # Check condition if present
                if task.condition:
                    try:
                        if not task.condition(workflow_execution.current_context):
                            task.status = TaskStatus.SKIPPED
                            continue
                    except Exception as e:
                        logger.warning(f"Task condition evaluation failed for {task.id}: {str(e)}")
                        continue
                
                task.status = TaskStatus.READY
                ready_tasks.append(task)
        
        return ready_tasks
    
    def select_optimal_agent(self, task: WorkflowTask) -> Optional[AgentMatch]:
        """Select the optimal agent for a task."""
        if task.agent_type:
            # Specific agent type requested
            if isinstance(task.agent_type, AgentType):
                agent_id = task.agent_type.value
            else:
                agent_id = task.agent_type
            
            agent_info = self.discovery_service.registry.agents.get(agent_id)
            if agent_info:
                return AgentMatch(
                    agent_info=agent_info,
                    match_score=1.0,
                    capability_scores={},
                    load_score=1.0,
                    priority_score=1.0,
                    response_time_score=1.0,
                    total_score=1.0
                )
        
        # Find best agent based on requirements
        if task.agent_requirements:
            return self.discovery_service.find_best_agent(
                task_description=task.description,
                capabilities=task.agent_requirements,
                priority=task.priority
            )
        
        return None
    
    async def execute_task(
        self,
        task: WorkflowTask,
        workflow_execution: WorkflowExecution,
        universal_client: UniversalAgentClient
    ) -> TaskResponse:
        """Execute a single task."""
        task.status = TaskStatus.RUNNING
        task.start_time = datetime.now()
        
        try:
            # Select agent
            agent_match = self.select_optimal_agent(task)
            if not agent_match:
                raise Exception(f"No suitable agent found for task {task.id}")
            
            task.assigned_agent = agent_match.agent_info.id
            
            # Prepare parameters
            parameters = task.parameters.copy()
            
            # Apply input transformation
            if task.input_transform:
                try:
                    parameters = task.input_transform(workflow_execution.current_context)
                except Exception as e:
                    logger.warning(f"Input transformation failed for task {task.id}: {str(e)}")
            
            # Add workflow context
            parameters.update({
                "workflow_id": workflow_execution.workflow_id,
                "execution_id": workflow_execution.execution_id,
                "task_id": task.id,
                "context": workflow_execution.current_context
            })
            
            # Execute task
            response = await universal_client.execute_task(
                agent_type=agent_match.agent_info.id,
                task_description=task.description,
                parameters=parameters,
                priority=task.priority,
                timeout=task.timeout
            )
            
            # Process response
            task.end_time = datetime.now()
            task.execution_time = (task.end_time - task.start_time).total_seconds()
            
            if response.status == "completed" or response.status == "success":
                task.status = TaskStatus.COMPLETED
                task.result = response.result
                
                # Apply output transformation
                if task.output_transform:
                    try:
                        task.result = task.output_transform(task.result)
                    except Exception as e:
                        logger.warning(f"Output transformation failed for task {task.id}: {str(e)}")
                
                # Update workflow context
                workflow_execution.current_context[task.id] = task.result
                workflow_execution.task_results[task.id] = task.result
                
            else:
                task.status = TaskStatus.FAILED
                task.error = response.error or "Task execution failed"
                raise Exception(task.error)
            
            # Record history
            self.task_history[task.id].append(response)
            
            return response
            
        except Exception as e:
            task.end_time = datetime.now()
            task.execution_time = (task.end_time - task.start_time).total_seconds()
            task.status = TaskStatus.FAILED
            task.error = str(e)
            
            logger.error(f"Task {task.id} failed: {str(e)}")
            raise
    
    async def retry_task(
        self,
        task: WorkflowTask,
        workflow_execution: WorkflowExecution,
        universal_client: UniversalAgentClient
    ) -> TaskResponse:
        """Retry a failed task with backoff strategy."""
        if task.retry_count >= task.max_retries:
            raise Exception(f"Task {task.id} exceeded maximum retries ({task.max_retries})")
        
        task.retry_count += 1
        task.status = TaskStatus.RETRYING
        
        # Calculate delay based on retry strategy
        delay = self._calculate_retry_delay(task)
        if delay > 0:
            logger.info(f"Retrying task {task.id} in {delay} seconds (attempt {task.retry_count})")
            await asyncio.sleep(delay)
        
        return await self.execute_task(task, workflow_execution, universal_client)
    
    def _calculate_retry_delay(self, task: WorkflowTask) -> float:
        """Calculate retry delay based on strategy."""
        if task.retry_strategy == RetryStrategy.NONE:
            return 0.0
        elif task.retry_strategy == RetryStrategy.FIXED_DELAY:
            return 5.0  # 5 seconds
        elif task.retry_strategy == RetryStrategy.LINEAR_BACKOFF:
            return task.retry_count * 2.0
        elif task.retry_strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            return min(2 ** task.retry_count, 60.0)  # Cap at 60 seconds
        else:
            return 1.0


class WorkflowEngine:
    """Main workflow orchestration engine."""
    
    def __init__(
        self,
        universal_client: UniversalAgentClient,
        discovery_service: DiscoveryService
    ):
        self.universal_client = universal_client
        self.discovery_service = discovery_service
        self.scheduler = TaskScheduler(discovery_service)
        
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.completed_executions: Dict[str, WorkflowExecution] = {}
        self.workflow_definitions: Dict[str, WorkflowDefinition] = {}
        
        # Event system
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Metrics
        self.metrics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "total_tasks": 0,
            "successful_tasks": 0,
            "failed_tasks": 0,
            "avg_workflow_duration": 0.0,
            "avg_task_duration": 0.0
        }
    
    def register_workflow(self, workflow_definition: WorkflowDefinition):
        """Register a workflow definition."""
        self.workflow_definitions[workflow_definition.id] = workflow_definition
        logger.info(f"Registered workflow: {workflow_definition.name}")
    
    def create_execution_graph(self, workflow_definition: WorkflowDefinition) -> nx.DiGraph:
        """Create execution graph from workflow definition."""
        graph = nx.DiGraph()
        
        # Add all tasks as nodes
        for task in workflow_definition.tasks:
            graph.add_node(task.id, task=task)
        
        # Add dependency edges
        for task in workflow_definition.tasks:
            for dep_id in task.dependencies:
                if graph.has_node(dep_id):
                    graph.add_edge(dep_id, task.id)
                else:
                    logger.warning(f"Dependency {dep_id} not found for task {task.id}")
        
        # Validate graph (check for cycles)
        if not nx.is_directed_acyclic_graph(graph):
            raise ValueError("Workflow contains circular dependencies")
        
        return graph
    
    async def start_workflow(
        self,
        workflow_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> str:
        """Start workflow execution."""
        if workflow_id not in self.workflow_definitions:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        workflow_def = self.workflow_definitions[workflow_id]
        execution_id = f"{workflow_id}_{uuid.uuid4().hex[:8]}"
        
        # Create execution instance
        execution = WorkflowExecution(
            workflow_id=workflow_id,
            execution_id=execution_id,
            definition=workflow_def,
            status=WorkflowStatus.PENDING,
            start_time=datetime.now(),
            current_context=parameters or {},
            execution_graph=self.create_execution_graph(workflow_def)
        )
        
        # Initialize global parameters
        execution.current_context.update(workflow_def.global_parameters)
        execution.current_context.update(workflow_def.environment)
        
        # Reset all task statuses
        for task in execution.definition.tasks:
            task.status = TaskStatus.WAITING
            task.retry_count = 0
            task.result = None
            task.error = None
            task.assigned_agent = None
            task.start_time = None
            task.end_time = None
            task.execution_time = 0.0
        
        self.active_executions[execution_id] = execution
        
        # Start execution in background
        asyncio.create_task(self._execute_workflow(execution))
        
        self.metrics["total_workflows"] += 1
        logger.info(f"Started workflow execution: {execution_id}")
        
        return execution_id
    
    async def _execute_workflow(self, execution: WorkflowExecution):
        """Execute a workflow."""
        try:
            execution.status = WorkflowStatus.RUNNING
            
            # Call start handler
            if execution.definition.on_start:
                await execution.definition.on_start()
            
            # Execute based on strategy
            if execution.definition.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                await self._execute_sequential(execution)
            elif execution.definition.execution_strategy == ExecutionStrategy.PARALLEL:
                await self._execute_parallel(execution)
            elif execution.definition.execution_strategy == ExecutionStrategy.HYBRID:
                await self._execute_hybrid(execution)
            else:  # ADAPTIVE
                await self._execute_adaptive(execution)
            
            # Check final status
            if execution.failed_tasks and not execution.definition.retry_failed_workflow:
                execution.status = WorkflowStatus.FAILED
                execution.error = f"Tasks failed: {list(execution.failed_tasks)}"
            else:
                execution.status = WorkflowStatus.COMPLETED
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.error = str(e)
            logger.error(f"Workflow {execution.execution_id} failed: {str(e)}")
        
        finally:
            execution.end_time = datetime.now()
            
            # Update metrics
            self._update_workflow_metrics(execution)
            
            # Move to completed
            if execution.execution_id in self.active_executions:
                del self.active_executions[execution.execution_id]
            self.completed_executions[execution.execution_id] = execution
            
            # Call completion handlers
            try:
                if execution.status == WorkflowStatus.COMPLETED and execution.definition.on_complete:
                    await execution.definition.on_complete(execution.task_results)
                elif execution.status == WorkflowStatus.FAILED and execution.definition.on_failure:
                    await execution.definition.on_failure(Exception(execution.error))
            except Exception as e:
                logger.error(f"Error in workflow handler: {str(e)}")
            
            logger.info(f"Workflow {execution.execution_id} completed with status: {execution.status}")
    
    async def _execute_sequential(self, execution: WorkflowExecution):
        """Execute workflow sequentially following dependency order."""
        # Get topological order
        try:
            task_order = list(nx.topological_sort(execution.execution_graph))
        except nx.NetworkXError:
            raise Exception("Cannot determine execution order - graph may have cycles")
        
        for task_id in task_order:
            task = self._get_task_by_id(execution, task_id)
            if not task:
                continue
            
            if task.status == TaskStatus.SKIPPED:
                continue
            
            await self._execute_single_task(task, execution)
            
            if task.status == TaskStatus.FAILED and not task.skip_on_failure:
                execution.failed_tasks.add(task_id)
                if not execution.definition.retry_failed_workflow:
                    break
    
    async def _execute_parallel(self, execution: WorkflowExecution):
        """Execute all independent tasks in parallel."""
        while True:
            ready_tasks = self.scheduler.get_ready_tasks(execution)
            if not ready_tasks:
                break
            
            # Limit parallel execution
            batch_size = min(
                len(ready_tasks),
                execution.definition.max_parallel_tasks
            )
            
            batch = ready_tasks[:batch_size]
            
            # Execute batch in parallel
            tasks = []
            for task in batch:
                tasks.append(self._execute_single_task(task, execution))
            
            await asyncio.gather(*tasks, return_exceptions=True)
            
            # Update completed tasks
            for task in batch:
                if task.status == TaskStatus.COMPLETED:
                    execution.completed_tasks.add(task.id)
                elif task.status == TaskStatus.FAILED:
                    execution.failed_tasks.add(task.id)
    
    async def _execute_hybrid(self, execution: WorkflowExecution):
        """Execute workflow using hybrid strategy (parallel where possible)."""
        while True:
            ready_tasks = self.scheduler.get_ready_tasks(execution)
            if not ready_tasks:
                break
            
            # Group tasks by dependency level
            levels = self._group_tasks_by_level(ready_tasks, execution)
            
            for level_tasks in levels:
                if not level_tasks:
                    continue
                
                # Execute level in parallel
                batch_size = min(
                    len(level_tasks),
                    execution.definition.max_parallel_tasks
                )
                
                for i in range(0, len(level_tasks), batch_size):
                    batch = level_tasks[i:i + batch_size]
                    
                    tasks = []
                    for task in batch:
                        tasks.append(self._execute_single_task(task, execution))
                    
                    await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Update completed tasks
                    for task in batch:
                        if task.status == TaskStatus.COMPLETED:
                            execution.completed_tasks.add(task.id)
                        elif task.status == TaskStatus.FAILED:
                            execution.failed_tasks.add(task.id)
                            
                            # Stop if critical task failed
                            if not task.skip_on_failure:
                                return
    
    async def _execute_adaptive(self, execution: WorkflowExecution):
        """Execute workflow with adaptive strategy based on runtime conditions."""
        # Start with hybrid approach but adapt based on performance
        await self._execute_hybrid(execution)
    
    async def _execute_single_task(
        self,
        task: WorkflowTask,
        execution: WorkflowExecution
    ):
        """Execute a single task with retry logic."""
        for attempt in range(task.max_retries + 1):
            try:
                response = await self.scheduler.execute_task(
                    task, execution, self.universal_client
                )
                
                # Update metrics
                self.metrics["total_tasks"] += 1
                if task.status == TaskStatus.COMPLETED:
                    self.metrics["successful_tasks"] += 1
                
                # Call task completion handler
                if (task.status == TaskStatus.COMPLETED and 
                    execution.definition.on_task_complete):
                    await execution.definition.on_task_complete(task)
                
                return response
                
            except Exception as e:
                if attempt < task.max_retries:
                    try:
                        await self.scheduler.retry_task(task, execution, self.universal_client)
                    except Exception:
                        pass  # Will be handled in final attempt
                else:
                    # Final failure
                    self.metrics["failed_tasks"] += 1
                    
                    if execution.definition.on_task_failure:
                        await execution.definition.on_task_failure(task, e)
                    
                    logger.error(f"Task {task.id} failed after {task.max_retries} retries: {str(e)}")
                    break
    
    def _get_task_by_id(self, execution: WorkflowExecution, task_id: str) -> Optional[WorkflowTask]:
        """Get task by ID from execution."""
        for task in execution.definition.tasks:
            if task.id == task_id:
                return task
        return None
    
    def _group_tasks_by_level(
        self,
        tasks: List[WorkflowTask],
        execution: WorkflowExecution
    ) -> List[List[WorkflowTask]]:
        """Group tasks by dependency level."""
        levels = []
        remaining_tasks = tasks.copy()
        
        while remaining_tasks:
            current_level = []
            next_remaining = []
            
            for task in remaining_tasks:
                # Check if all dependencies are satisfied
                can_execute = all(
                    dep_id in execution.completed_tasks
                    for dep_id in task.dependencies
                )
                
                if can_execute:
                    current_level.append(task)
                else:
                    next_remaining.append(task)
            
            if current_level:
                levels.append(current_level)
                remaining_tasks = next_remaining
            else:
                # No progress - break to avoid infinite loop
                break
        
        return levels
    
    def _update_workflow_metrics(self, execution: WorkflowExecution):
        """Update workflow execution metrics."""
        if execution.status == WorkflowStatus.COMPLETED:
            self.metrics["successful_workflows"] += 1
        elif execution.status == WorkflowStatus.FAILED:
            self.metrics["failed_workflows"] += 1
        
        # Update duration metrics
        if execution.end_time:
            duration = (execution.end_time - execution.start_time).total_seconds()
            current_avg = self.metrics["avg_workflow_duration"]
            total_workflows = self.metrics["successful_workflows"] + self.metrics["failed_workflows"]
            
            if total_workflows > 0:
                self.metrics["avg_workflow_duration"] = (
                    (current_avg * (total_workflows - 1) + duration) / total_workflows
                )
        
        # Update task duration metrics
        total_task_duration = 0.0
        task_count = 0
        
        for task in execution.definition.tasks:
            if task.execution_time > 0:
                total_task_duration += task.execution_time
                task_count += 1
        
        if task_count > 0:
            avg_task_duration = total_task_duration / task_count
            current_avg = self.metrics["avg_task_duration"]
            total_tasks = self.metrics["successful_tasks"] + self.metrics["failed_tasks"]
            
            if total_tasks > 0:
                self.metrics["avg_task_duration"] = (
                    (current_avg * (total_tasks - task_count) + avg_task_duration * task_count) / total_tasks
                )
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get workflow execution status."""
        execution = (
            self.active_executions.get(execution_id) or
            self.completed_executions.get(execution_id)
        )
        
        if not execution:
            return None
        
        task_statuses = {}
        for task in execution.definition.tasks:
            task_statuses[task.id] = {
                "name": task.name,
                "status": task.status.value,
                "assigned_agent": task.assigned_agent,
                "execution_time": task.execution_time,
                "error": task.error
            }
        
        duration = None
        if execution.end_time:
            duration = (execution.end_time - execution.start_time).total_seconds()
        elif execution.status == WorkflowStatus.RUNNING:
            duration = (datetime.now() - execution.start_time).total_seconds()
        
        return {
            "execution_id": execution_id,
            "workflow_id": execution.workflow_id,
            "status": execution.status.value,
            "start_time": execution.start_time.isoformat(),
            "end_time": execution.end_time.isoformat() if execution.end_time else None,
            "duration": duration,
            "tasks": task_statuses,
            "completed_tasks": len(execution.completed_tasks),
            "failed_tasks": len(execution.failed_tasks),
            "total_tasks": len(execution.definition.tasks),
            "error": execution.error
        }
    
    def cancel_workflow(self, execution_id: str) -> bool:
        """Cancel a running workflow."""
        if execution_id in self.active_executions:
            execution = self.active_executions[execution_id]
            execution.status = WorkflowStatus.CANCELLED
            
            # Cancel running tasks
            for task in execution.definition.tasks:
                if task.status == TaskStatus.RUNNING:
                    task.status = TaskStatus.FAILED
                    task.error = "Workflow cancelled"
            
            execution.end_time = datetime.now()
            
            # Move to completed
            del self.active_executions[execution_id]
            self.completed_executions[execution_id] = execution
            
            logger.info(f"Cancelled workflow execution: {execution_id}")
            return True
        
        return False
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get orchestration metrics."""
        return {
            **self.metrics,
            "active_workflows": len(self.active_executions),
            "completed_workflows": len(self.completed_executions),
            "registered_workflows": len(self.workflow_definitions)
        }


# Workflow Builder Helper
class WorkflowBuilder:
    """Helper class for building workflows programmatically."""
    
    def __init__(self, workflow_id: str, name: str, description: str = ""):
        self.workflow_def = WorkflowDefinition(
            id=workflow_id,
            name=name,
            description=description,
            tasks=[]
        )
    
    def add_task(
        self,
        task_id: str,
        name: str,
        description: str,
        agent_type: Optional[Union[AgentType, str]] = None,
        agent_requirements: Optional[List[str]] = None,
        parameters: Optional[Dict[str, Any]] = None,
        dependencies: Optional[List[str]] = None,
        **kwargs
    ) -> 'WorkflowBuilder':
        """Add a task to the workflow."""
        task = WorkflowTask(
            id=task_id,
            name=name,
            description=description,
            agent_type=agent_type,
            agent_requirements=agent_requirements or [],
            parameters=parameters or {},
            dependencies=dependencies or [],
            **kwargs
        )
        
        self.workflow_def.tasks.append(task)
        return self
    
    def set_execution_strategy(self, strategy: ExecutionStrategy) -> 'WorkflowBuilder':
        """Set workflow execution strategy."""
        self.workflow_def.execution_strategy = strategy
        return self
    
    def set_max_parallel_tasks(self, max_tasks: int) -> 'WorkflowBuilder':
        """Set maximum parallel tasks."""
        self.workflow_def.max_parallel_tasks = max_tasks
        return self
    
    def set_timeout(self, timeout: int) -> 'WorkflowBuilder':
        """Set workflow timeout."""
        self.workflow_def.timeout = timeout
        return self
    
    def add_global_parameter(self, key: str, value: Any) -> 'WorkflowBuilder':
        """Add global parameter."""
        self.workflow_def.global_parameters[key] = value
        return self
    
    def build(self) -> WorkflowDefinition:
        """Build the workflow definition."""
        return self.workflow_def


# Example workflow definitions
def create_code_analysis_workflow() -> WorkflowDefinition:
    """Create a comprehensive code analysis workflow."""
    return (
        WorkflowBuilder("code_analysis", "Comprehensive Code Analysis")
        .add_task(
            "fetch_code",
            "Fetch Code Repository",
            "Retrieve code from repository",
            agent_requirements=["code_management", "git_operations"],
            parameters={"repository_url": "{{ repo_url }}"}
        )
        .add_task(
            "static_analysis",
            "Static Code Analysis",
            "Perform static analysis using Semgrep",
            agent_type="semgrep-security-analyzer",
            dependencies=["fetch_code"],
            parameters={"code_path": "{{ fetch_code.result.path }}"}
        )
        .add_task(
            "quality_check",
            "Code Quality Assessment",
            "Analyze code quality and style",
            agent_type="code-generation-improver",
            dependencies=["fetch_code"],
            parameters={"code_path": "{{ fetch_code.result.path }}"}
        )
        .add_task(
            "security_scan",
            "Security Vulnerability Scan",
            "Scan for security vulnerabilities",
            agent_type="security-pentesting-specialist",
            dependencies=["static_analysis"],
            parameters={"analysis_report": "{{ static_analysis.result }}"}
        )
        .add_task(
            "generate_report",
            "Generate Analysis Report",
            "Compile comprehensive analysis report",
            agent_type="document-knowledge-manager",
            dependencies=["static_analysis", "quality_check", "security_scan"],
            parameters={
                "static_analysis": "{{ static_analysis.result }}",
                "quality_check": "{{ quality_check.result }}",
                "security_scan": "{{ security_scan.result }}"
            }
        )
        .set_execution_strategy(ExecutionStrategy.HYBRID)
        .set_max_parallel_tasks(3)
        .build()
    )


# Example usage and testing
if __name__ == "__main__":
    async def main():
        """Example usage of the Workflow Orchestrator."""
        
        # Initialize components
        async with UniversalAgentClient() as client:
            discovery = DiscoveryService(client)
            await discovery.start()
            
            try:
                # Initialize workflow engine
                engine = WorkflowEngine(client, discovery)
                
                # Register workflow
                workflow_def = create_code_analysis_workflow()
                engine.register_workflow(workflow_def)
                
                # Start workflow
                execution_id = await engine.start_workflow(
                    "code_analysis",
                    parameters={"repo_url": "https://github.com/example/repo"}
                )
                
                print(f"Started workflow execution: {execution_id}")
                
                # Monitor execution
                while True:
                    status = engine.get_execution_status(execution_id)
                    if not status:
                        break
                    
                    print(f"Status: {status['status']}")
                    print(f"Completed tasks: {status['completed_tasks']}/{status['total_tasks']}")
                    
                    if status['status'] in ['completed', 'failed', 'cancelled']:
                        break
                    
                    await asyncio.sleep(5)
                
                # Show final results
                final_status = engine.get_execution_status(execution_id)
                print(f"Final status: {final_status}")
                
                # Show metrics
                metrics = engine.get_metrics()
                print(f"Engine metrics: {metrics}")
                
            finally:
                await discovery.stop()
    
    # Run the example
    asyncio.run(main())