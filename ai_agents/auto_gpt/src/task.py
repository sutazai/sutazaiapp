from typing import Dict, List, Optional

#!/usr/bin/env python3.11
"""
Task management module for AutoGPT agent.

This module provides classes and utilities for managing tasks,
including task planning, execution, and status tracking.
"""

from typing import dict, list, Optional
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import json
import os


class TaskStatus(Enum):
    """Enumeration of possible task statuses."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"


    @dataclass
    class TaskStep:
        """Represents a single step in a task's execution plan."""

        description: str
        status: TaskStatus = TaskStatus.PENDING
        result: Optional[str] = None
        started_at: Optional[datetime] = None
        completed_at: Optional[datetime] = None

        def to_dict(self) -> Dict:
            """Convert task step to dictionary format."""
        return {
        "description": self.description,
        "status": self.status.value,
        "result": self.result,
        "started_at": self.started_at.isoformat() if self.started_at else None,
        "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

        @classmethod
        def from_dict(cls, data: Dict) -> "TaskStep":
            """Create task step from dictionary format."""
        return cls(
        description=data["description"],
        status=TaskStatus(data["status"]),
        result=data["result"],
        started_at=datetime.fromisoformat(
            data["started_at"]) if data["started_at"] else None,
        completed_at=datetime.fromisoformat(
            data["completed_at"]) if data["completed_at"] else None,
        )


        class Task:
            """Represents a task to be executed by the agent."""

            def __init__(
                self, objective: str, context: Optional[Dict] = None, max_steps: int = 5, persist_path: Optional[str] = None
                ):
                """
                Initialize a new task.

                Args:
                objective: The main objective of the task
                context: Additional context for the task (optional)
                max_steps: Maximum number of steps allowed for the task
                persist_path: Path to persist task state to disk (optional)
                """
                self.objective = objective
                self.context = context or {}
                self.max_steps = max_steps
                self.persist_path = persist_path
                self.steps: List[TaskStep] = []
                self.created_at = datetime.now()
                self.started_at: Optional[datetime] = None
                self.completed_at: Optional[datetime] = None

                if persist_path and os.path.exists(persist_path):
                    self.load()

                    @property
                    def status(self) -> TaskStatus:
                        """Get the current status of the task."""
                        if not self.steps:
                        return TaskStatus.PENDING

                        if any(
                            step.status == TaskStatus.FAILED for step in self.steps):
                        return TaskStatus.FAILED

                        if all(
                            step.status == TaskStatus.COMPLETED for step in self.steps):
                        return TaskStatus.COMPLETED

                        if any(
                            step.status == TaskStatus.IN_PROGRESS for step in self.steps):
                        return TaskStatus.IN_PROGRESS

                    return TaskStatus.PENDING

                    def add_step(self, description: str) -> TaskStep:
                        """
                        Add a new step to the task.

                        Args:
                        description: Description of the step

                        Returns:
                        TaskStep: The created step

                        Raises:
                        ValueError: If maximum number of steps would be exceeded
                        """
                        if len(self.steps) >= self.max_steps:
                        raise ValueError(
                            f"Cannot add step: maximum of {self.max_steps} steps allowed")

                        step = TaskStep(description=description)
                        self.steps.append(step)

                        if self.persist_path:
                            self.save()

                        return step

                        def start_step(self, index: int) -> None:
                            """
                            Mark a step as started.

                            Args:
                            index: Index of the step to start

                            Raises:
                            IndexError: If index is out of range
                            ValueError: If step is not in PENDING status
                            """
                            if not 0 <= index < len(self.steps):
                            raise IndexError(
                                f"Step index {index} out of range")

                            step = self.steps[index]
                            if step.status != TaskStatus.PENDING:
                            raise ValueError(
                                f"Cannot start step: status is {step.status}")

                            step.status = TaskStatus.IN_PROGRESS
                            step.started_at = datetime.now()

                            if self.started_at is None:
                                self.started_at = datetime.now()

                                if self.persist_path:
                                    self.save()

                                    def complete_step(
                                        self,
                                        index: int,
                                        result: Optional[str] = None) -> None:
                                        """
                                        Mark a step as completed.

                                        Args:
                                        index: Index of the step to complete
                                        result: Result of the step execution (
                                            optional)

                                        Raises:
                                        IndexError: If index is out of range
                                                                                ValueError: If step is not in \
                                            IN_PROGRESS status
                                        """
                                        if not 0 <= index < len(self.steps):
                                        raise IndexError(
                                            f"Step index {index} out of range")

                                        step = self.steps[index]
                                        if step.status != TaskStatus.IN_PROGRESS:
                                        raise ValueError(
                                            f"Cannot complete step: status is {step.status}")

                                        step.status = TaskStatus.COMPLETED
                                        step.result = result
                                        step.completed_at = datetime.now()

                                        if all(
                                            s.status == TaskStatus.COMPLETED for s in self.steps):
                                            self.completed_at = datetime.now()

                                            if self.persist_path:
                                                self.save()

                                                def fail_step(
                                                    self,
                                                    index: int,
                                                    error: str) -> None:
                                                    """
                                                    Mark a step as failed.

                                                    Args:
                                                    index: Index of the step to fail
                                                    error: Error message explaining the failure

                                                    Raises:
                                                                                                        IndexError: If index is \
                                                        out of range
                                                                                                        ValueError: If step is not in \
                                                        IN_PROGRESS status
                                                    """
                                                    if not 0 <= index < len(
                                                        self.steps):
                                                    raise IndexError(
                                                        f"Step index {index} out of range")

                                                    step = self.steps[index]
                                                    if step.status != TaskStatus.IN_PROGRESS:
                                                    raise ValueError(
                                                        f"Cannot fail step: status is {step.status}")

                                                    step.status = TaskStatus.FAILED
                                                    step.result = error
                                                    step.completed_at = datetime.now()

                                                    if self.persist_path:
                                                        self.save()

                                                        def to_dict(
                                                            self) -> Dict:
                                                            """Convert task to dictionary format."""
                                                        return {
                                                        "objective": self.objective,
                                                        "context": self.context,
                                                        "max_steps": self.max_steps,
                                                        "steps": [step.to_dict() for step in self.steps],
                                                        "created_at": self.created_at.isoformat(),
                                                        "started_at": self.started_at.isoformat() if self.started_at else None,
                                                        "completed_at": self.completed_at.isoformat() if self.completed_at else None,
                                                        }

                                                        def save(self) -> None:
                                                                                                                        """Save task state to disk if persist_path is \
                                                                set."""
                                                            if not self.persist_path:
                                                            return

                                                            os.makedirs(
                                                                os.path.dirname(self.persist_path),
                                                                exist_ok=True)
                                                            with open(
                                                                self.persist_path,
                                                                "w") as f:
                                                            json.dump(
                                                                self.to_dict(),
                                                                f,
                                                                indent=2)

                                                            def load(
                                                                self) -> None:
                                                                                                                                """Load task state from disk if persist_path is \
                                                                    set."""
                                                                if not self.persist_path or not os.path.exists(
                                                                    self.persist_path):
                                                                return

                                                                with open(
                                                                    self.persist_path,
                                                                    "r") as f:
                                                                data = json.load(
                                                                    f)

                                                                self.objective = data["objective"]
                                                                self.context = data["context"]
                                                                self.max_steps = data["max_steps"]
                                                                self.steps = [TaskStep.from_dict(
                                                                    step) for step in data["steps"]]
                                                                self.created_at = datetime.fromisoformat(
                                                                    data["created_at"])
                                                                self.started_at = datetime.fromisoformat(
                                                                    data["started_at"]) if data["started_at"] else None
                                                                self.completed_at = datetime.fromisoformat(
                                                                    data["completed_at"]) if data["completed_at"] else None
