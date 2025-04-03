"""
Parallel Processing Module

This module provides functionality for parallel processing of tasks by multiple agents,
enabling efficient distribution of workloads and coordination of results.
"""

import uuid
import time
import logging
import threading
import concurrent.futures
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, field
from datetime import datetime

from .message_protocol import MessageType, MessageProtocol
from .agent_communication import AgentCommunication

# Configure logging
logger = logging.getLogger(__name__)


class TaskDistributionStrategy(Enum):
    """Strategy for distributing tasks to agents."""

    ROUND_ROBIN = "round_robin"
    LOAD_BALANCED = "load_balanced"
    CAPABILITY_BASED = "capability_based"
    RANDOM = "random"


@dataclass
class SubTask:
    """
    A sub-task to be processed by an agent.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: Optional[str] = None
    parameters: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert sub-task to dictionary."""
        return {
            "task_id": self.task_id,
            "agent_id": self.agent_id,
            "parameters": self.parameters,
            "result": self.result,
            "error": self.error,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }


@dataclass
class ParallelTask:
    """
    A task that can be processed in parallel by multiple agents.
    """

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = "parallel_task"
    description: str = ""
    sub_tasks: List[SubTask] = field(default_factory=list)
    result_aggregation: Optional[str] = (
        None  # How to aggregate results: sum, average, concat, etc.
    )
    combined_result: Optional[Dict[str, Any]] = None
    status: str = "pending"
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert parallel task to dictionary."""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "description": self.description,
            "sub_tasks": [st.to_dict() for st in self.sub_tasks],
            "result_aggregation": self.result_aggregation,
            "combined_result": self.combined_result,
            "status": self.status,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat()
            if self.completed_at
            else None,
        }

    def is_complete(self) -> bool:
        """Check if all sub-tasks are complete."""
        return all(st.status in ("completed", "failed") for st in self.sub_tasks)

    def pending_count(self) -> int:
        """Get count of pending sub-tasks."""
        return sum(1 for st in self.sub_tasks if st.status == "pending")

    def running_count(self) -> int:
        """Get count of running sub-tasks."""
        return sum(1 for st in self.sub_tasks if st.status == "running")

    def completed_count(self) -> int:
        """Get count of completed sub-tasks."""
        return sum(1 for st in self.sub_tasks if st.status == "completed")

    def failed_count(self) -> int:
        """Get count of failed sub-tasks."""
        return sum(1 for st in self.sub_tasks if st.status == "failed")

    def progress(self) -> float:
        """Get progress percentage."""
        if not self.sub_tasks:
            return 0.0
        return (
            (self.completed_count() + self.failed_count()) / len(self.sub_tasks) * 100.0
        )


class ParallelTaskProcessor:
    """
    Processes tasks in parallel using multiple agents.

    This class distributes sub-tasks to agents, monitors progress,
    and aggregates results when all sub-tasks are complete.
    """

    def __init__(self, agent_communication: AgentCommunication, max_workers: int = 10):
        """
        Initialize the parallel task processor.

        Args:
            agent_communication: Agent communication system
            max_workers: Maximum number of worker threads
        """
        self.agent_communication = agent_communication
        self.max_workers = max_workers
        self.tasks: Dict[str, ParallelTask] = {}
        self.lock = threading.RLock()
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=max_workers)
        self.running = False
        self.processing_thread = None
        self.callbacks: Dict[str, Callable[[ParallelTask], None]] = {}
        self.agent_load: Dict[
            str, int
        ] = {}  # Track number of tasks assigned to each agent

    def start(self) -> None:
        """Start the task processor."""
        if self.running:
            return

        self.running = True
        self.processing_thread = threading.Thread( # type: ignore[assignment]
            target=self._process_tasks, daemon=True
        )
        self.processing_thread.start() # type: ignore[union-attr]
        logger.info("Parallel task processor started")

    def stop(self) -> None:
        """Stop the task processor."""
        self.running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5.0)

        self.executor.shutdown(wait=False)
        logger.info("Parallel task processor stopped")

    def create_task(
        self, task_type: str, description: str, result_aggregation: Optional[str] = None
    ) -> str:
        """
        Create a new parallel task.

        Args:
            task_type: Type of task
            description: Task description
            result_aggregation: How to aggregate results

        Returns:
            str: Task ID
        """
        with self.lock:
            task = ParallelTask(
                task_type=task_type,
                description=description,
                result_aggregation=result_aggregation,
            )

            self.tasks[task.task_id] = task
            logger.info(f"Created parallel task {task.task_id} of type {task_type}")

            return task.task_id

    def add_sub_task(
        self, task_id: str, parameters: Dict[str, Any], agent_id: Optional[str] = None
    ) -> str:
        """
        Add a sub-task to a parallel task.

        Args:
            task_id: ID of the parallel task
            parameters: Parameters for the sub-task
            agent_id: ID of the agent to assign (optional)

        Returns:
            str: Sub-task ID
        """
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Parallel task {task_id} not found")

            # If task is already running or complete, can't add sub-tasks
            if self.tasks[task_id].status not in ("pending"):
                raise ValueError(
                    f"Cannot add sub-tasks to {self.tasks[task_id].status} task"
                )

            sub_task = SubTask(agent_id=agent_id, parameters=parameters)

            self.tasks[task_id].sub_tasks.append(sub_task)
            logger.debug(
                f"Added sub-task {sub_task.task_id} to parallel task {task_id}"
            )

            return sub_task.task_id

    def execute_task(
        self,
        task_id: str,
        strategy: TaskDistributionStrategy = TaskDistributionStrategy.ROUND_ROBIN,
        available_agents: Optional[List[str]] = None,
        callback: Optional[Callable[[ParallelTask], None]] = None,
    ) -> None:
        """
        Execute a parallel task.

        Args:
            task_id: ID of the parallel task
            strategy: Strategy for distributing sub-tasks
            available_agents: List of available agent IDs
            callback: Function to call when task is complete
        """
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Parallel task {task_id} not found")

            task = self.tasks[task_id]

            # If task is already running or complete, can't execute again
            if task.status != "pending":
                raise ValueError(f"Cannot execute {task.status} task")

            # Record start time
            task.status = "running"
            task.started_at = datetime.utcnow()

            # Store callback if provided
            if callback:
                self.callbacks[task_id] = callback

            # Distribute sub-tasks based on strategy
            self._distribute_sub_tasks(task, strategy, available_agents)

            logger.info(
                f"Started execution of parallel task {task_id} with {len(task.sub_tasks)} sub-tasks"
            )

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """
        Get status of a parallel task.

        Args:
            task_id: ID of the parallel task

        Returns:
            Dict[str, Any]: Task status
        """
        with self.lock:
            if task_id not in self.tasks:
                raise ValueError(f"Parallel task {task_id} not found")

            task = self.tasks[task_id]

            status = {
                "task_id": task.task_id,
                "status": task.status,
                "total_sub_tasks": len(task.sub_tasks),
                "pending": task.pending_count(),
                "running": task.running_count(),
                "completed": task.completed_count(),
                "failed": task.failed_count(),
                "progress": task.progress(),
                "created_at": task.created_at.isoformat(),
                "started_at": task.started_at.isoformat() if task.started_at else None,
                "completed_at": task.completed_at.isoformat()
                if task.completed_at
                else None,
            }

            if task.status == "completed":
                status["result"] = task.combined_result

            return status

    def cancel_task(self, task_id: str) -> bool:
        """
        Cancel a parallel task.

        Args:
            task_id: ID of the parallel task

        Returns:
            bool: True if task was cancelled, False otherwise
        """
        with self.lock:
            if task_id not in self.tasks:
                return False

            task = self.tasks[task_id]

            if task.status in ("completed", "failed", "cancelled"):
                return False

            task.status = "cancelled"

            # Cancel any pending sub-tasks
            for sub_task in task.sub_tasks:
                if sub_task.status == "pending":
                    sub_task.status = "cancelled"

            logger.info(f"Cancelled parallel task {task_id}")

            return True

    def _distribute_sub_tasks(
        self,
        task: ParallelTask,
        strategy: TaskDistributionStrategy,
        available_agents: Optional[List[str]] = None,
    ) -> None:
        """
        Distribute sub-tasks to agents based on strategy.

        Args:
            task: Parallel task
            strategy: Distribution strategy
            available_agents: List of available agent IDs
        """
        # If no available agents, can't distribute
        if not available_agents:
            logger.warning(f"No available agents for task {task.task_id}")
            return

        # Initialize agent load if not already tracking these agents
        for agent_id in available_agents:
            if agent_id not in self.agent_load:
                self.agent_load[agent_id] = 0

        # Distribute based on strategy
        if strategy == TaskDistributionStrategy.ROUND_ROBIN:
            self._distribute_round_robin(task, available_agents)
        elif strategy == TaskDistributionStrategy.LOAD_BALANCED:
            self._distribute_load_balanced(task, available_agents)
        elif strategy == TaskDistributionStrategy.CAPABILITY_BASED:
            self._distribute_capability_based(task, available_agents)
        else:  # Default to random
            self._distribute_random(task, available_agents)

    def _distribute_round_robin(
        self, task: ParallelTask, available_agents: List[str]
    ) -> None:
        """
        Distribute sub-tasks using round-robin strategy.

        Args:
            task: Parallel task
            available_agents: List of available agent IDs
        """
        agent_count = len(available_agents)

        for i, sub_task in enumerate([st for st in task.sub_tasks if not st.agent_id]):
            agent_index = i % agent_count
            sub_task.agent_id = available_agents[agent_index]
            self.agent_load[sub_task.agent_id] += 1

    def _distribute_load_balanced(
        self, task: ParallelTask, available_agents: List[str]
    ) -> None:
        """
        Distribute sub-tasks based on current agent load.

        Args:
            task: Parallel task
            available_agents: List of available agent IDs
        """
        for sub_task in [st for st in task.sub_tasks if not st.agent_id]:
            # Find agent with lowest load
            agent_id = min(available_agents, key=lambda a: self.agent_load.get(a, 0))

            sub_task.agent_id = agent_id
            self.agent_load[agent_id] += 1

    def _distribute_capability_based(
        self, task: ParallelTask, available_agents: List[str]
    ) -> None:
        """
        Distribute sub-tasks based on agent capabilities.

        Args:
            task: Parallel task
            available_agents: List of available agent IDs
        """
        # This is a placeholder - in a real implementation, you would need to match
        # sub-task requirements with agent capabilities

        # For now, just use load balancing
        self._distribute_load_balanced(task, available_agents)

    def _distribute_random(
        self, task: ParallelTask, available_agents: List[str]
    ) -> None:
        """
        Distribute sub-tasks randomly.

        Args:
            task: Parallel task
            available_agents: List of available agent IDs
        """
        import random

        for sub_task in [st for st in task.sub_tasks if not st.agent_id]:
            agent_id = random.choice(available_agents)
            sub_task.agent_id = agent_id
            self.agent_load[agent_id] += 1

    def _process_tasks(self) -> None:
        """Process parallel tasks and their sub-tasks."""
        while self.running:
            try:
                active_tasks = []

                # Get list of active tasks
                with self.lock:
                    active_tasks = [
                        task_id
                        for task_id, task in self.tasks.items()
                        if task.status == "running"
                    ]

                # Process each active task
                for task_id in active_tasks:
                    self._process_task(task_id)

                # Sleep briefly to avoid tight loop
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error in parallel task processor: {e}")

    def _process_task(self, task_id: str) -> None:
        """
        Process a single parallel task.

        Args:
            task_id: ID of the task to process
        """
        with self.lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]

            if task.status != "running":
                return

            # Check for pending sub-tasks that need to be started
            pending_tasks = [
                st for st in task.sub_tasks if st.status == "pending" and st.agent_id
            ]

            # Submit pending sub-tasks to executor
            for sub_task in pending_tasks:
                self.executor.submit(self._execute_sub_task, task_id, sub_task.task_id)
                sub_task.status = "running"
                sub_task.started_at = datetime.utcnow()

            # Check if all sub-tasks are complete
            if task.is_complete():
                self._finalize_task(task)

    def _execute_sub_task(self, task_id: str, sub_task_id: str) -> None:
        """
        Execute a single sub-task.

        Args:
            task_id: ID of the parallel task
            sub_task_id: ID of the sub-task
        """
        sub_task = None

        # Get the sub-task
        with self.lock:
            if task_id not in self.tasks:
                return

            task = self.tasks[task_id]

            # Find the sub-task
            for st in task.sub_tasks:
                if st.task_id == sub_task_id:
                    sub_task = st
                    break

            if not sub_task or sub_task.status != "running":
                return

        try:
            # Create a request message
            message = MessageProtocol.create_message(
                message_type=MessageType.TASK_REQUEST,
                sender_id="parallel_processor",
                recipient_id=sub_task.agent_id,
                content={
                    "task_id": sub_task.task_id,
                    "parent_task_id": task_id,
                    "task_type": task.task_type,
                    "parameters": sub_task.parameters,
                },
            )

            # Send the message
            self.agent_communication.send_message(message)

            # Wait for response (with timeout)
            response = self._wait_for_response(sub_task_id, timeout=300)

            with self.lock:
                if task_id not in self.tasks:
                    return

                # Update the sub-task with the response
                for st in self.tasks[task_id].sub_tasks:
                    if st.task_id == sub_task_id:
                        st.status = "completed"
                        st.completed_at = datetime.utcnow()
                        st.result = response

                        # Update agent load
                        if st.agent_id in self.agent_load:
                            self.agent_load[st.agent_id] = max(
                                0, self.agent_load[st.agent_id] - 1
                            )

                        break

        except Exception as e:
            logger.error(f"Error executing sub-task {sub_task_id}: {e}")

            with self.lock:
                if task_id not in self.tasks:
                    return

                # Update the sub-task with the error
                for st in self.tasks[task_id].sub_tasks:
                    if st.task_id == sub_task_id:
                        st.status = "failed"
                        st.completed_at = datetime.utcnow()
                        st.error = str(e)

                        # Update agent load
                        if st.agent_id in self.agent_load:
                            self.agent_load[st.agent_id] = max(
                                0, self.agent_load[st.agent_id] - 1
                            )

                        break

    def _wait_for_response(
        self, sub_task_id: str, timeout: int = 300
    ) -> Dict[str, Any]:
        """
        Wait for a response to a sub-task request.

        Args:
            sub_task_id: ID of the sub-task
            timeout: Timeout in seconds

        Returns:
            Dict[str, Any]: Response content

        Raises:
            TimeoutError: If response times out
        """
        # In a real implementation, you would set up a callback for the response
        # and wait for it using an event or similar mechanism

        # This is a placeholder implementation
        time.sleep(0.5)  # Simulate response time

        # Return a dummy response
        return {"status": "success", "result": "Demo result"}

    def _finalize_task(self, task: ParallelTask) -> None:
        """
        Finalize a parallel task by aggregating results.

        Args:
            task: Parallel task to finalize
        """
        # Mark as completed
        task.status = "completed"
        task.completed_at = datetime.utcnow()

        # Aggregate results based on specified method
        task.combined_result = self._aggregate_results(task)

        logger.info(f"Completed parallel task {task.task_id}")

        # Call the callback if registered
        if task.task_id in self.callbacks:
            try:
                self.callbacks[task.task_id](task)
            except Exception as e:
                logger.error(f"Error in task callback: {e}")
            finally:
                del self.callbacks[task.task_id]

    def _aggregate_results(self, task: ParallelTask) -> Dict[str, Any]:
        """
        Aggregate results from sub-tasks.

        Args:
            task: Parallel task

        Returns:
            Dict[str, Any]: Aggregated result
        """
        # Get completed sub-tasks
        completed_tasks = [st for st in task.sub_tasks if st.status == "completed"]

        # If no completed tasks, return empty result
        if not completed_tasks:
            return {"status": "no_results"}

        # Get results
        results = [st.result for st in completed_tasks if st.result]

        # If no results, return empty result
        if not results:
            return {"status": "no_results"}

        # Aggregate based on specified method
        if task.result_aggregation == "sum":
            # Sum numeric values with the same keys
            combined: Dict[str, Any] = {}
            for result in results:
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        combined[key] = combined.get(key, 0) + value
            return combined

        elif task.result_aggregation == "average":
            # Average numeric values with the same keys
            combined: Dict[str, Any] = {}
            counts: Dict[str, Any] = {}

            for result in results:
                for key, value in result.items():
                    if isinstance(value, (int, float)):
                        combined[key] = combined.get(key, 0) + value
                        counts[key] = counts.get(key, 0) + 1

            for key in combined:
                if counts[key] > 0:
                    combined[key] = combined[key] / counts[key]

            return combined

        elif task.result_aggregation == "concatenate_list":
            # Use a distinct variable name for this block
            combined_list: List[Any] = []
            for r in results:
                if isinstance(r, list):
                    combined_list.extend(r)
                elif r is not None:
                    combined_list.append(r)
            combined = combined_list # type: ignore[assignment]

        elif task.result_aggregation == "concat":
            # Concatenate list values with the same keys
            combined = {}

            for result in results:
                for key, value in result.items():
                    if key not in combined:
                        combined[key] = []

                    if isinstance(value, list):
                        combined[key].extend(value)
                    else:
                        combined[key].append(value)

            return combined

        else:
            # Default: return list of all results
            return {
                "results": results,
                "successful_count": len(completed_tasks),
                "failed_count": task.failed_count(),
            }
