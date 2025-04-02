"""
Non-Linear Workflow Engine

This module provides functionality for defining and executing complex workflows
with parallel execution, branching logic, and dependency tracking.
"""

import uuid
import time
import logging
import threading
import queue
from enum import Enum
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime

from ..protocols.message_protocol import MessageType, MessageProtocol
from ..protocols.agent_communication import AgentCommunication

# Configure logging
logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a workflow task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    BLOCKED = "blocked"  # Waiting for dependencies


@dataclass
class WorkflowTask:
    """
    Represents a task in a workflow.
    """

    task_id: str
    agent_id: Optional[str]  # Agent ID or None for system tasks
    task_type: str
    parameters: Dict[str, Any]
    dependencies: Set[str] = field(default_factory=set)  # Task IDs this task depends on
    dependents: Set[str] = field(
        default_factory=set
    )  # Task IDs that depend on this task
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    timeout: Optional[int] = None  # Timeout in seconds
    retry_count: int = 0
    max_retries: int = 3
    priority: int = 3  # 1 (highest) to 5 (lowest)
    is_conditional: bool = False
    condition: Optional[Dict[str, Any]] = None

    def can_execute(self, completed_tasks: Set[str]) -> bool:
        """
        Check if this task can be executed.

        Args:
            completed_tasks: Set of completed task IDs

        Returns:
            bool: True if task can be executed, False otherwise
        """
        # Check if task is already running, completed, failed, or cancelled
        if self.status in (
            TaskStatus.RUNNING,
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            return False

        # Check if all dependencies are completed
        return all(dep in completed_tasks for dep in self.dependencies)

    def to_dict(self) -> Dict[str, Any]:
        """Convert task to dictionary."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "task_type": self.task_type,
            "parameters": self.parameters,
            "dependencies": list(self.dependencies),
            "dependents": list(self.dependents),
            "status": self.status.value,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "timeout": self.timeout,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "priority": self.priority,
            "is_conditional": self.is_conditional,
            "condition": self.condition,
        }


class Workflow:
    """
    Represents a complete workflow with multiple tasks.
    """

    def __init__(
        self,
        workflow_id: Optional[str] = None,
        name: str = "Unnamed Workflow",
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize a workflow.

        Args:
            workflow_id: Optional workflow ID (generated if not provided)
            name: Workflow name
            description: Workflow description
            metadata: Additional workflow metadata
        """
        self.workflow_id = workflow_id or str(uuid.uuid4())
        self.name = name
        self.description = description
        self.metadata = metadata or {}
        self.tasks: Dict[str, WorkflowTask] = {}
        self.execution_order: List[List[str]] = []  # List of lists of parallel task IDs
        self.created_at = datetime.utcnow()
        self.started_at: Optional[datetime] = None
        self.completed_at: Optional[datetime] = None
        self.status = TaskStatus.PENDING
        self.result: Dict[str, Any] = {}

    def add_task(self, task: WorkflowTask) -> str:
        """
        Add a task to the workflow.

        Args:
            task: Task to add

        Returns:
            str: Task ID
        """
        self.tasks[task.task_id] = task
        return task.task_id

    def add_dependency(self, dependent_task_id: str, dependency_task_id: str) -> None:
        """
        Add a dependency between tasks.

        Args:
            dependent_task_id: ID of the dependent task
            dependency_task_id: ID of the dependency task

        Raises:
            ValueError: If task IDs are invalid or would create a cycle
        """
        if dependent_task_id not in self.tasks:
            raise ValueError(f"Dependent task {dependent_task_id} not found")

        if dependency_task_id not in self.tasks:
            raise ValueError(f"Dependency task {dependency_task_id} not found")

        # Add dependency
        self.tasks[dependent_task_id].dependencies.add(dependency_task_id)
        self.tasks[dependency_task_id].dependents.add(dependent_task_id)

        # Check for cycles
        if self._has_cycle():
            # Revert change
            self.tasks[dependent_task_id].dependencies.remove(dependency_task_id)
            self.tasks[dependency_task_id].dependents.remove(dependent_task_id)
            raise ValueError("Adding this dependency would create a cycle")

    def _has_cycle(self) -> bool:
        """
        Check if the workflow has a cycle.

        Returns:
            bool: True if workflow has a cycle, False otherwise
        """
        visited = set()
        path = set()

        def dfs(task_id: str) -> bool:
            if task_id in path:
                return True

            if task_id in visited:
                return False

            visited.add(task_id)
            path.add(task_id)

            for dep_id in self.tasks[task_id].dependents:
                if dfs(dep_id):
                    return True

            path.remove(task_id)
            return False

        for task_id in self.tasks:
            if dfs(task_id):
                return True

        return False

    def compute_execution_order(self) -> List[List[str]]:
        """
        Compute the execution order of tasks based on dependencies.

        Returns:
            List[List[str]]: List of lists of parallel task IDs
        """
        # Reset execution order
        self.execution_order = []

        # Copy tasks to avoid modifying the original
        remaining_tasks = {
            task_id: task.dependencies.copy() for task_id, task in self.tasks.items()
        }

        while remaining_tasks:
            # Find tasks with no dependencies
            ready_tasks = [
                task_id for task_id, deps in remaining_tasks.items() if not deps
            ]

            if not ready_tasks:
                # Cycle detected
                logger.error("Cycle detected in workflow")
                break

            # Add to execution order
            self.execution_order.append(ready_tasks)

            # Remove tasks from remaining
            for task_id in ready_tasks:
                del remaining_tasks[task_id]

            # Update dependencies
            for task_id, deps in remaining_tasks.items():
                deps.difference_update(ready_tasks)

        return self.execution_order

    def to_dict(self) -> Dict[str, Any]:
        """Convert workflow to dictionary."""
        return {
            "workflow_id": self.workflow_id,
            "name": self.name,
            "description": self.description,
            "metadata": self.metadata,
            "tasks": {task_id: task.to_dict() for task_id, task in self.tasks.items()},
            "execution_order": self.execution_order,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
            "status": self.status.value,
            "result": self.result,
        }


class WorkflowEngine:
    """
    Engine for executing and managing workflows.

    This class provides functionality for executing workflows with parallel
    task execution, dependency tracking, and conditional branching.
    """

    def __init__(
        self, agent_communication: AgentCommunication, max_concurrent_tasks: int = 10
    ):
        """
        Initialize the workflow engine.

        Args:
            agent_communication: Agent communication system
            max_concurrent_tasks: Maximum number of concurrent tasks
        """
        self.agent_communication = agent_communication
        self.max_concurrent_tasks = max_concurrent_tasks
        self.workflows: Dict[str, Workflow] = {}
        self.active_workflows: Dict[str, threading.Thread] = {}
        self.task_queue = queue.PriorityQueue()
        self.task_executors: Dict[str, threading.Thread] = {}
        self.running = False
        self.task_semaphore = threading.Semaphore(max_concurrent_tasks)
        self.result_lock = threading.Lock()

    def start(self) -> None:
        """Start the workflow engine."""
        if self.running:
            return

        self.running = True
        self.task_executor_thread = threading.Thread(
            target=self._task_executor_loop, daemon=True
        )
        self.task_executor_thread.start()
        logger.info("Workflow engine started")

    def stop(self) -> None:
        """Stop the workflow engine."""
        self.running = False

        # Wait for executor thread to finish
        if (
            hasattr(self, "task_executor_thread")
            and self.task_executor_thread.is_alive()
        ):
            self.task_executor_thread.join(timeout=5.0)

        # Cancel active workflows
        for workflow_id in list(self.active_workflows.keys()):
            self.cancel_workflow(workflow_id)

        logger.info("Workflow engine stopped")

    def create_workflow(
        self,
        name: str,
        description: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Create a new workflow.

        Args:
            name: Workflow name
            description: Workflow description
            metadata: Additional workflow metadata

        Returns:
            str: Workflow ID
        """
        workflow = Workflow(name=name, description=description, metadata=metadata)
        self.workflows[workflow.workflow_id] = workflow
        return workflow.workflow_id

    def add_task(
        self,
        workflow_id: str,
        agent_id: Optional[str],
        task_type: str,
        parameters: Dict[str, Any],
        dependencies: Optional[List[str]] = None,
        timeout: Optional[int] = None,
        max_retries: int = 3,
        priority: int = 3,
        is_conditional: bool = False,
        condition: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add a task to a workflow.

        Args:
            workflow_id: Workflow ID
            agent_id: Agent ID or None for system tasks
            task_type: Type of task
            parameters: Task parameters
            dependencies: List of task IDs this task depends on
            timeout: Timeout in seconds
            max_retries: Maximum number of retries
            priority: Task priority (1-5, 1 is highest)
            is_conditional: Whether this task is conditional
            condition: Condition for executing this task

        Returns:
            str: Task ID

        Raises:
            ValueError: If workflow ID is invalid
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        task_id = str(uuid.uuid4())
        task = WorkflowTask(
            task_id=task_id,
            agent_id=agent_id,
            task_type=task_type,
            parameters=parameters,
            dependencies=set(dependencies or []),
            status=TaskStatus.PENDING,
            timeout=timeout,
            max_retries=max_retries,
            priority=priority,
            is_conditional=is_conditional,
            condition=condition,
        )

        # Add task to workflow
        self.workflows[workflow_id].add_task(task)

        # Add dependencies
        if dependencies:
            for dep_id in dependencies:
                if dep_id not in self.workflows[workflow_id].tasks:
                    raise ValueError(f"Dependency task {dep_id} not found")

                self.workflows[workflow_id].tasks[dep_id].dependents.add(task_id)

        return task_id

    def execute_workflow(
        self, workflow_id: str, async_execution: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Execute a workflow.

        Args:
            workflow_id: Workflow ID
            async_execution: Whether to execute asynchronously

        Returns:
            Optional[Dict[str, Any]]: Workflow result if synchronous execution

        Raises:
            ValueError: If workflow ID is invalid
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]

        # Compute execution order
        workflow.compute_execution_order()

        # Mark workflow as running
        workflow.status = TaskStatus.RUNNING
        workflow.started_at = datetime.utcnow()

        if async_execution:
            # Start workflow in a separate thread
            thread = threading.Thread(
                target=self._execute_workflow_async, args=(workflow_id,), daemon=True
            )
            self.active_workflows[workflow_id] = thread
            thread.start()
            return None
        else:
            # Execute workflow synchronously
            return self._execute_workflow_sync(workflow_id)

    def _execute_workflow_sync(self, workflow_id: str) -> Dict[str, Any]:
        """
        Execute a workflow synchronously.

        Args:
            workflow_id: Workflow ID

        Returns:
            Dict[str, Any]: Workflow result
        """
        workflow = self.workflows[workflow_id]
        completed_tasks: Set[str] = set()

        # Execute each level in the execution order
        for level in workflow.execution_order:
            # Create threads for each task in the level
            threads = []
            for task_id in level:
                task = workflow.tasks[task_id]

                # Check if task is conditional
                if task.is_conditional and task.condition:
                    # Evaluate condition using task results
                    if not self._evaluate_condition(task.condition, workflow):
                        # Skip task
                        task.status = TaskStatus.CANCELLED
                        completed_tasks.add(task_id)
                        continue

                thread = threading.Thread(
                    target=self._execute_task, args=(workflow_id, task_id)
                )
                threads.append(thread)
                thread.start()

            # Wait for all tasks in this level to complete
            for thread in threads:
                thread.join()

            # Update completed tasks
            for task_id in level:
                task = workflow.tasks[task_id]
                if task.status == TaskStatus.COMPLETED:
                    completed_tasks.add(task_id)

        # Update workflow status
        failed_tasks = [
            task_id
            for task_id, task in workflow.tasks.items()
            if task.status == TaskStatus.FAILED
        ]

        if failed_tasks:
            workflow.status = TaskStatus.FAILED
        else:
            workflow.status = TaskStatus.COMPLETED

        workflow.completed_at = datetime.utcnow()

        # Aggregate results
        for task_id, task in workflow.tasks.items():
            if task.status == TaskStatus.COMPLETED and task.result:
                workflow.result[task_id] = task.result

        return workflow.result

    def _execute_workflow_async(self, workflow_id: str) -> None:
        """
        Execute a workflow asynchronously.

        Args:
            workflow_id: Workflow ID
        """
        try:
            self._execute_workflow_sync(workflow_id)
        except Exception as e:
            logger.error(f"Error executing workflow {workflow_id}: {e}")
            self.workflows[workflow_id].status = TaskStatus.FAILED
        finally:
            # Remove from active workflows
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]

    def _execute_task(self, workflow_id: str, task_id: str) -> None:
        """
        Execute a task.

        Args:
            workflow_id: Workflow ID
            task_id: Task ID
        """
        workflow = self.workflows[workflow_id]
        task = workflow.tasks[task_id]

        # Update task status
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.utcnow()

        # Execute task
        try:
            # Acquire semaphore
            self.task_semaphore.acquire()

            # Check if agent-based task
            if task.agent_id:
                # Send task request to agent
                content = {
                    "task_type": task.task_type,
                    "parameters": task.parameters,
                    "workflow_id": workflow_id,
                    "task_id": task_id,
                }

                # Create message for agent
                message = MessageProtocol.create_message(
                    message_type=MessageType.TASK_REQUEST,
                    sender_id="workflow_engine",
                    content=content,
                    recipient_id=task.agent_id,
                    ttl=task.timeout,
                    priority=task.priority,
                )

                # Send message
                if not self.agent_communication.send_message(message):
                    raise Exception(
                        f"Failed to send task request to agent {task.agent_id}"
                    )

                # Wait for response
                result = self._wait_for_response(message.message_id, task.timeout)

                if result:
                    task.result = result
                    task.status = TaskStatus.COMPLETED
                else:
                    # Retry
                    if task.retry_count < task.max_retries:
                        task.retry_count += 1
                        self._execute_task(workflow_id, task_id)
                    else:
                        task.status = TaskStatus.FAILED
                        task.error = "Task execution timed out or failed"
            else:
                # System task
                # TODO: Implement system task execution
                task.status = TaskStatus.COMPLETED
                task.result = {"status": "success"}

        except Exception as e:
            logger.error(f"Error executing task {task_id}: {e}")
            task.status = TaskStatus.FAILED
            task.error = str(e)
        finally:
            # Update task status
            task.completed_at = datetime.utcnow()

            # Release semaphore
            self.task_semaphore.release()

    def _wait_for_response(
        self, request_id: str, timeout: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Wait for a response to a task request.

        Args:
            request_id: Request message ID
            timeout: Timeout in seconds

        Returns:
            Optional[Dict[str, Any]]: Response content or None if timeout
        """
        # TODO: Implement response waiting with timeout
        # This is a placeholder implementation
        return {"status": "success", "result": {}}

    def _evaluate_condition(
        self, condition: Dict[str, Any], workflow: Workflow
    ) -> bool:
        """
        Evaluate a condition for a conditional task.

        Args:
            condition: Condition to evaluate
            workflow: Workflow containing the task

        Returns:
            bool: True if condition is satisfied, False otherwise
        """
        # TODO: Implement condition evaluation
        # This is a placeholder implementation
        return True

    def get_workflow_status(self, workflow_id: str) -> Dict[str, Any]:
        """
        Get the status of a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            Dict[str, Any]: Workflow status

        Raises:
            ValueError: If workflow ID is invalid
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]

        return {
            "workflow_id": workflow.workflow_id,
            "name": workflow.name,
            "status": workflow.status.value,
            "created_at": workflow.created_at.isoformat(),
            "started_at": workflow.started_at.isoformat()
            if workflow.started_at
            else None,
            "completed_at": workflow.completed_at.isoformat()
            if workflow.completed_at
            else None,
            "tasks": {
                task_id: {"status": task.status.value, "error": task.error}
                for task_id, task in workflow.tasks.items()
            },
        }

    def cancel_workflow(self, workflow_id: str) -> bool:
        """
        Cancel a workflow.

        Args:
            workflow_id: Workflow ID

        Returns:
            bool: True if workflow was cancelled, False otherwise

        Raises:
            ValueError: If workflow ID is invalid
        """
        if workflow_id not in self.workflows:
            raise ValueError(f"Workflow {workflow_id} not found")

        workflow = self.workflows[workflow_id]

        # Check if workflow is already completed
        if workflow.status in (
            TaskStatus.COMPLETED,
            TaskStatus.FAILED,
            TaskStatus.CANCELLED,
        ):
            return False

        # Cancel all running tasks
        for task_id, task in workflow.tasks.items():
            if task.status == TaskStatus.RUNNING:
                task.status = TaskStatus.CANCELLED

        # Update workflow status
        workflow.status = TaskStatus.CANCELLED
        workflow.completed_at = datetime.utcnow()

        # Remove from active workflows
        if workflow_id in self.active_workflows:
            # Interrupt thread (will exit gracefully when tasks are cancelled)
            self.active_workflows[workflow_id].join(timeout=5.0)
            del self.active_workflows[workflow_id]

        return True

    def _task_executor_loop(self) -> None:
        """Task executor loop."""
        while self.running:
            try:
                time.sleep(0.1)  # Prevent CPU spinning
                # TODO: Implement task queue processing
            except Exception as e:
                logger.error(f"Error in task executor loop: {e}")
