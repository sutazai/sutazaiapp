#!/usr/bin/env python3.11
"""Tests for the task management module of the AutoGPT agent."""

import pytest
from datetime import datetime

from ai_agents.auto_gpt.src.task import Task, TaskStatus, TaskStep


@pytest.fixture
def test_task(tmp_path) -> Task:
    """Create a test task instance with a temporary persist path."""
    persist_path = tmp_path / "task.json"
    return Task(
        objective="Test objective",
        context={"test_key": "test_value"},
        max_steps=3,
        persist_path=str(persist_path),
    )


def test_task_step_creation():
    """Test creating a task step."""
    step = TaskStep(description="Test step")
    assert step.description == "Test step"
    assert step.status == TaskStatus.PENDING
    assert step.result is None
    assert step.started_at is None
    assert step.completed_at is None


def test_task_step_to_dict():
    """Test converting a task step to dictionary format."""
    step = TaskStep(description="Test step")
    data = step.to_dict()
    assert data["description"] == "Test step"
    assert data["status"] == TaskStatus.PENDING.value
    assert data["result"] is None
    assert data["started_at"] is None
    assert data["completed_at"] is None


def test_task_step_from_dict():
    """Test creating a task step from dictionary format."""
    now = datetime.now()
    data = {
        "description": "Test step",
        "status": TaskStatus.COMPLETED.value,
        "result": "Success",
        "started_at": now.isoformat(),
        "completed_at": now.isoformat(),
    }
    step = TaskStep.from_dict(data)
    assert step.description == "Test step"
    assert step.status == TaskStatus.COMPLETED
    assert step.result == "Success"
    assert isinstance(step.started_at, datetime)
    assert isinstance(step.completed_at, datetime)


def test_task_initialization(test_task):
    """Test initializing task."""
    assert test_task.objective == "Test objective"
    assert test_task.context == {"test_key": "test_value"}
    assert test_task.max_steps == 3
    assert test_task.persist_path is not None
    assert isinstance(test_task.steps, list)
    assert len(test_task.steps) == 0
    assert test_task.created_at is not None
    assert test_task.started_at is None
    assert test_task.completed_at is None


def test_task_status(test_task):
    """Test task status calculation."""
    # Initial status
    assert test_task.status == TaskStatus.PENDING
    
    # Add and start a step
    step = test_task.add_step("Test step")
    test_task.start_step(0)
    assert test_task.status == TaskStatus.IN_PROGRESS
    
    # Complete the step
    test_task.complete_step(0, "Success")
    assert test_task.status == TaskStatus.COMPLETED
    
    # Add another step and fail it
    step = test_task.add_step("Failed step")
    test_task.start_step(1)
    test_task.fail_step(1, "Error")
    assert test_task.status == TaskStatus.FAILED


def test_add_step(test_task):
    """Test adding steps to task."""
    # Add steps up to limit
    for i in range(3):
        step = test_task.add_step(f"Step {i}")
        assert step.description == f"Step {i}"
    
    # Try to add one more step
    with pytest.raises(ValueError):
        test_task.add_step("Extra step")


def test_start_step(test_task):
    """Test starting a task step."""
    test_task.add_step("Test step")
    test_task.start_step(0)
    step = test_task.steps[0]
    assert step.status == TaskStatus.IN_PROGRESS
    assert step.started_at is not None
    assert test_task.started_at is not None
    
    # Try to start same step again
    with pytest.raises(ValueError):
        test_task.start_step(0)
    
    # Try to start non-existent step
    with pytest.raises(IndexError):
        test_task.start_step(1)


def test_complete_step(test_task):
    """Test completing a task step."""
    test_task.add_step("Test step")
    test_task.start_step(0)
    test_task.complete_step(0, "Success")
    step = test_task.steps[0]
    assert step.status == TaskStatus.COMPLETED
    assert step.result == "Success"
    assert step.completed_at is not None
    
    # All steps completed, task should be marked as completed
    assert test_task.completed_at is not None
    
    # Try to complete step that's not in progress
    with pytest.raises(ValueError):
        test_task.complete_step(0, "Already completed")


def test_fail_step(test_task):
    """Test failing a task step."""
    test_task.add_step("Test step")
    test_task.start_step(0)
    test_task.fail_step(0, "Error message")
    step = test_task.steps[0]
    assert step.status == TaskStatus.FAILED
    assert step.result == "Error message"
    assert step.completed_at is not None
    
    # Try to fail step that's not in progress
    with pytest.raises(ValueError):
        test_task.fail_step(0, "Already failed")


def test_task_persistence(test_task):
    """Test saving and loading task state."""
    # Add and complete a step
    test_task.add_step("Test step")
    test_task.start_step(0)
    test_task.complete_step(0, "Success")
    test_task.save()
    
    # Create new task instance with same persist path
    new_task = Task(
        objective="Test objective",
        persist_path=test_task.persist_path,
    )
    
    # Should load the saved state
    assert new_task.objective == test_task.objective
    assert len(new_task.steps) == 1
    assert new_task.steps[0].status == TaskStatus.COMPLETED
    assert new_task.steps[0].result == "Success"


def test_invalid_persist_path():
    """Test handling invalid persistence path."""
    # Try to create task with invalid path
    task = Task(
        objective="Test objective",
        persist_path="/invalid/path/task.json",
    )
    
    # Should not raise error, but save() should fail silently
    task.add_step("Test step")
    task.save()  # Should not raise error


def test_task_to_dict(test_task):
    """Test converting task to dictionary format."""
    test_task.add_step("Test step")
    test_task.start_step(0)
    test_task.complete_step(0, "Success")
    data = test_task.to_dict()
    
    assert data["objective"] == "Test objective"
    assert data["context"] == {"test_key": "test_value"}
    assert data["max_steps"] == 3
    assert len(data["steps"]) == 1
    assert isinstance(data["created_at"], str)
    assert isinstance(data["started_at"], str)
    assert isinstance(data["completed_at"], str)

"""""""""