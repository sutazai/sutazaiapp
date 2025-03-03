from typing import Self
from typing import Self
from typing import Self
from typing import Self
from typing import Self
from typing import Self
from typing import Self
from typing import Dict, List, Optional#!/usr/bin/env python3.11"""Task management module for AutoGPT agent.This module provides classes and utilities for managing tasks,including task planning, execution, and status tracking."""from typing import dict, list, Optionalfrom dataclasses import dataclassfrom datetime import datetimefrom enum import Enumimport jsonimport osclass TaskStatus(Enum):    """Enumeration of possible task statuses."""    PENDING = "pending"    IN_PROGRESS = "in_progress"    COMPLETED = "completed"    FAILED = "failed"    @dataclass    class TaskStep:        """Represents a single step in a task's execution plan."""        description: str        status: TaskStatus = TaskStatus.PENDING        result: str | None = None        started_at: datetime | None = None        completed_at: datetime | None = None    def to_dict(self) -> Self else None,        }    @classmethod
from typing import Union
from typing import Optional
def from_dict(        cls,        data: Dict) -> "TaskStep":    """Create task step from dictionary format."""        return cls(        description=data["description"],        status=TaskStatus(data["status"]),        result=data["result"],        started_at=datetime.fromisoformat(        data["started_at"]) if data["started_at"] else None,
completed_at=datetime.fromisoformat(
data["completed_at"]) if data["completed_at"] else None,
)
class Task:        """Represents a task to be executed by the agent."""    def __init__(        self,        objective: str,        context: Dict | None = None,        max_steps: int = 5,        persist_path: str | None = None):    """        Initialize a new task.        Args:    objective: The main objective of the task            context: Additional context for the task (optional)
max_steps: Maximum number of steps allowed for the task
persist_path: Path to persist task state to disk (optional)
"""
self.objective = objective
self.context = context or {}
self.max_steps = max_steps
self.persist_path = persist_path
self.steps: List[TaskStep] = []
self.created_at = datetime.now()
self.started_at: datetime | None = None
self.completed_at: datetime | None = None
if persist_path and os.path.exists(persist_path):        self.load()
@property
def status(self) -> Self):                return TaskStatus.IN_PROGRESS                    return TaskStatus.PENDING
def add_step(self, description: str) -> Self} steps allowed")                    step = TaskStep(description=description)
self.steps.append(step)
if self.persist_path:        self.save()
return step
def start_step(self, index: int) -> Self[index]
if step.status != TaskStatus.PENDING:        raise ValueError(
f"Cannot start step: status is {step.status}")
step.status = TaskStatus.IN_PROGRESS
step.started_at = datetime.now()
if self.started_at is None:        self.started_at = datetime.now()
if self.persist_path:        self.save()
def complete_step(        self, index: int, result: str | None = None) -> None:    """        Mark a step as completed.        Args:    index: Index of the step to complete        result: Result of the step execution (        optional)        Raises:    IndexError: If index is out of range        ValueError: If step is not in \                                        IN_PROGRESS status
"""
if not 0 <= index < len(
self.steps):                raise IndexError(
f"Step index {index} out of range")
step = self.steps[index]
if step.status != TaskStatus.IN_PROGRESS:        raise ValueError(
f"Cannot complete step: status is {step.status}")
step.status = TaskStatus.COMPLETED
step.result = result
step.completed_at = datetime.now()
if all(
s.status == TaskStatus.COMPLETED for s in self.steps):                self.completed_at = datetime.now()
if self.persist_path:        self.save()
def fail_step(        self,        index: int,        error: str) -> None:    """        Mark a step as failed.        Args:    index: Index of the step to fail        error: Error message explaining the failure        Raises:    IndexError: If index is \                                                            out of range
ValueError: If step is not in \
IN_PROGRESS status
"""
if not 0 <= index < len(
self.steps):                raise IndexError(
f"Step index {index} out of range")
step = self.steps[index]
if step.status != TaskStatus.IN_PROGRESS:        raise ValueError(
f"Cannot fail step: status is {step.status}")
step.status = TaskStatus.FAILED
step.result = error
step.completed_at = datetime.now()
if self.persist_path:        self.save()
def to_dict(        self) -> Self.isoformat(),
"started_at": self.started_at.isoformat() if self.started_at else None,
"completed_at": self.completed_at.isoformat() if self.completed_at else None,
}
def save(        self) -> Self,
"w") as f:                json.dump(
self.to_dict(),
f,
indent=2)
def load(        self) -> Self,            "r") as f:                data = json.load(                                                                                    f)"""
self.objective = data[
"objective"]
self.context = data["context"]
self.max_steps = data[
"max_steps"]
self.steps = [TaskStep.from_dict(
step) for step in data["steps"]]
self.created_at = datetime.fromisoformat(
data["created_at"])
self.started_at = datetime.fromisoformat(
data["started_at"]) if data["started_at"] else None
self.completed_at = datetime.fromisoformat(
data["completed_at"]) if data["completed_at"] else None

"""""""""