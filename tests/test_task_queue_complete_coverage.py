#!/usr/bin/env python3.11
"""
Complete coverage tests for the task_queue module.
These tests are specifically designed to cover lines that aren't covered by existing tests.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime, timedelta

from core_system.orchestrator.models import (
    OrchestratorConfig, Task, TaskStatus, AgentStatus
)
from core_system.orchestrator.task_queue import TaskQueue
from core_system.orchestrator.exceptions import TaskError

# Import QueueFullError class
class QueueFullError(Exception):
    """Exception raised when the task queue is full."""
    pass

@pytest.fixture
def config():
    """Create a test configuration."""
    return OrchestratorConfig(
        primary_server="primary",
        secondary_server="secondary",
        sync_interval=60,
        max_agents=10,
        task_timeout=300
    )

@pytest.fixture
def task_queue(config):
    """Create a TaskQueue instance for testing."""
    return TaskQueue(config=config)

@pytest.fixture
def task():
    """Create a test task."""
    return Task(
        id="test-task-1",
        type="test",
        parameters={"param1": "value1"},
        priority=0
    )

@pytest.fixture
def task_dict():
    """Create a test task dictionary."""
    return {
        "id": "test-task-1",
        "type": "test",
        "parameters": {"param1": "value1"},
        "priority": 0
    }

class TestTaskQueueCompleteCoverage:
    """Tests to ensure complete coverage of the TaskQueue."""

    async def test_process_loop_exception(self, task_queue):
        """Test the _process_loop with an exception."""
        # Mock process to raise an exception
        task_queue.process = MagicMock(side_effect=Exception("Test exception"))
        
        # Override asyncio.sleep to make the test faster
        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            # Set is_running to True, then False after one iteration
            task_queue.is_running = True
            
            async def stop_after_one_iteration():
                await asyncio.sleep(0.05)
                task_queue.is_running = False
            
            stop_task = asyncio.create_task(stop_after_one_iteration())
            
            # Run the process loop
            await task_queue._process_loop()
            
            # Verify
            task_queue.process.assert_called_once()
            # Should call sleep with 1 after an error
            assert mock_sleep.call_args_list[-1][0][0] == 1
            
            # Clean up
            stop_task.cancel()
            try:
                await stop_task
            except asyncio.CancelledError:
                pass

    async def test_stop_with_mock_task(self, task_queue):
        """Test stopping the task queue when process_task is a mock."""
        # Set up a mock process task
        mock_task = MagicMock()
        mock_task.__class__.__name__ = 'AsyncMock'
        task_queue.process_task = mock_task
        task_queue.is_running = True
        
        # Call stop method
        await task_queue.stop()
        
        # Verify
        assert not task_queue.is_running
        mock_task.cancel.assert_called_once()

    async def test_stop_with_real_task(self, task_queue):
        """Test stopping the task queue with a real asyncio task."""
        # Create a real asyncio task
        async def dummy_task():
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
        
        task_queue.is_running = True
        task_queue.process_task = asyncio.create_task(dummy_task())
        
        # Call stop method
        await task_queue.stop()
        
        # Verify
        assert not task_queue.is_running
        assert task_queue.process_task.cancelled()

    async def test_submit_duplicate_id(self, task_queue, task_dict):
        """Test submitting a task with a duplicate ID."""
        # First submit a task
        await task_queue.submit(task_dict)
        
        # Try to submit another task with the same ID
        with pytest.raises(TaskError):
            await task_queue.submit(task_dict)

    async def test_submit_queue_full(self, task_queue, task_dict):
        """Test submitting a task when the queue is full."""
        # Set max_pending_tasks to 0 to simulate a full queue
        task_queue.max_pending_tasks = 0
        
        # Try to submit a task
        with pytest.raises(QueueFullError):
            await task_queue.submit(task_dict)

    async def test_submit_without_id(self, task_queue):
        """Test submitting a task without an ID."""
        # Create a task dictionary without an ID
        task_dict = {
            "type": "test",
            "parameters": {"param1": "value1"}
        }
        
        # Submit the task
        result = await task_queue.submit(task_dict)
        
        # Verify an ID was generated
        assert "id" in result
        assert task_queue.tasks[result["id"]].id == result["id"]

    async def test_submit_without_priority(self, task_queue):
        """Test submitting a task without a priority."""
        # Create a task dictionary without a priority
        task_dict = {
            "id": "test-task-no-priority",
            "type": "test",
            "parameters": {"param1": "value1"}
        }
        
        # Submit the task
        result = await task_queue.submit(task_dict)
        
        # Verify default priority was set
        assert task_queue.tasks[result["id"]].priority == 0

    def test_process_no_tasks(self, task_queue):
        """Test process with no tasks in the queue."""
        # Call process
        task_queue.process()
        
        # Nothing to verify, just ensure it doesn't fail

    def test_process_no_available_agents(self, task_queue, task):
        """Test process with no available agents."""
        # Add a task to the queue
        task_queue.tasks[task.id] = task
        task_queue.task_queue.put_nowait((task.priority, task.id))
        
        # Mock agent_manager.get_available_agent to return None
        task_queue.agent_manager = MagicMock()
        task_queue.agent_manager.get_available_agent = AsyncMock(return_value=None)
        
        # Create a fake event loop and run the process method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(task_queue._process_next())
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        
        # Verify
        task_queue.agent_manager.get_available_agent.assert_called_once()

    def test_process_successfully_assigned(self, task_queue, task):
        """Test successful task assignment in process."""
        # Add a task to the queue
        task_queue.tasks[task.id] = task
        task_queue.task_queue.put_nowait((task.priority, task.id))
        
        # Mock agent_manager methods
        agent = MagicMock()
        agent.id = "test-agent"
        agent.status = AgentStatus.IDLE
        
        task_queue.agent_manager = MagicMock()
        task_queue.agent_manager.get_available_agent = AsyncMock(return_value=agent)
        task_queue.agent_manager.assign_task = AsyncMock(return_value=True)
        
        # Create a fake event loop and run the process method
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(task_queue._process_next())
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        
        # Verify
        task_queue.agent_manager.get_available_agent.assert_called_once()
        task_queue.agent_manager.assign_task.assert_called_once_with(agent.id, task)
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None

    async def test_process_assignment_failed(self, task_queue, task):
        """Test process when task assignment fails."""
        # Add a task to the queue
        task_queue.tasks[task.id] = task
        task_queue.task_queue.put_nowait((task.priority, task.id))
        
        # Mock agent_manager methods
        agent = MagicMock()
        agent.id = "test-agent"
        agent.status = AgentStatus.IDLE
        
        task_queue.agent_manager = MagicMock()
        task_queue.agent_manager.get_available_agent = AsyncMock(return_value=agent)
        task_queue.agent_manager.assign_task = AsyncMock(return_value=False)
        
        # Call _process_next
        await task_queue._process_next()
        
        # Verify
        task_queue.agent_manager.get_available_agent.assert_called_once()
        task_queue.agent_manager.assign_task.assert_called_once_with(agent.id, task)
        # Task should be put back in the queue
        assert not task_queue.task_queue.empty()

    async def test_peek_next_no_tasks(self, task_queue):
        """Test peek_next with no tasks."""
        result = await task_queue.peek_next()
        assert result is None

    async def test_peek_next_with_tasks(self, task_queue, task):
        """Test peek_next with tasks in the queue."""
        # Add a task to the queue
        task_queue.tasks[task.id] = task
        task_queue.task_queue.put_nowait((task.priority, task.id))
        
        # Call peek_next
        result = await task_queue.peek_next()
        
        # Verify
        assert result.id == task.id
        # Task should still be in the queue
        assert not task_queue.task_queue.empty()

    async def test_get_tasks_by_status(self, task_queue):
        """Test get_tasks_by_status."""
        # Create tasks with different statuses
        pending_task = Task(id="pending-task", type="test", parameters={}, status=TaskStatus.PENDING)
        in_progress_task = Task(id="in-progress-task", type="test", parameters={}, status=TaskStatus.IN_PROGRESS)
        completed_task = Task(id="completed-task", type="test", parameters={}, status=TaskStatus.COMPLETED)
        
        # Add tasks to the queue
        task_queue.tasks[pending_task.id] = pending_task
        task_queue.tasks[in_progress_task.id] = in_progress_task
        task_queue.tasks[completed_task.id] = completed_task
        
        # Call get_tasks_by_status
        pending_tasks = task_queue.get_tasks_by_status(TaskStatus.PENDING)
        in_progress_tasks = task_queue.get_tasks_by_status(TaskStatus.IN_PROGRESS)
        completed_tasks = task_queue.get_tasks_by_status(TaskStatus.COMPLETED)
        
        # Verify
        assert len(pending_tasks) == 1
        assert pending_tasks[0].id == pending_task.id
        
        assert len(in_progress_tasks) == 1
        assert in_progress_tasks[0].id == in_progress_task.id
        
        assert len(completed_tasks) == 1
        assert completed_tasks[0].id == completed_task.id

    async def test_update_task_priority(self, task_queue, task):
        """Test update_task_priority."""
        # Add a task to the queue
        task_queue.tasks[task.id] = task
        task_queue.task_queue.put_nowait((task.priority, task.id))
        
        # Call update_task_priority
        await task_queue.update_task_priority(task.id, 5)
        
        # Verify
        assert task_queue.tasks[task.id].priority == 5
        # Task should be requeued with new priority
        assert not task_queue.task_queue.empty()
        # Get the task from the queue
        priority, task_id = task_queue.task_queue.get_nowait()
        assert priority == 5
        assert task_id == task.id

    async def test_update_task_priority_task_not_found(self, task_queue):
        """Test update_task_priority with a non-existent task."""
        with pytest.raises(TaskError):
            await task_queue.update_task_priority("non-existent-task", 5)

    async def test_update_task_priority_not_pending(self, task_queue, task):
        """Test update_task_priority with a non-pending task."""
        # Add a task to the queue but mark it as in progress
        task.status = TaskStatus.IN_PROGRESS
        task_queue.tasks[task.id] = task
        
        # Call update_task_priority
        with pytest.raises(TaskError):
            await task_queue.update_task_priority(task.id, 5)

    async def test_clear(self, task_queue, task):
        """Test clear."""
        # Add a task to the queue
        task_queue.tasks[task.id] = task
        task_queue.task_queue.put_nowait((task.priority, task.id))
        
        # Call clear
        task_queue.clear()
        
        # Verify
        assert len(task_queue.tasks) == 0
        assert task_queue.task_queue.empty()

    async def test_size(self, task_queue, task):
        """Test size."""
        # Add a task to the queue
        task_queue.tasks[task.id] = task
        task_queue.task_queue.put_nowait((task.priority, task.id))
        
        # Add another task with a different status
        completed_task = Task(id="completed-task", type="test", parameters={}, status=TaskStatus.COMPLETED)
        task_queue.tasks[completed_task.id] = completed_task
        
        # Call size
        result = task_queue.size()
        
        # Verify - only the pending task should be counted
        assert result == 1

    async def test_remove_task_not_found(self, task_queue):
        """Test remove with a non-existent task."""
        with pytest.raises(TaskError):
            await task_queue.remove("non-existent-task")

    async def test_remove_in_progress_task(self, task_queue, task):
        """Test remove with an in-progress task."""
        # Add a task to the queue but mark it as in progress
        task.status = TaskStatus.IN_PROGRESS
        task_queue.tasks[task.id] = task
        
        # Call remove
        await task_queue.remove(task.id)
        
        # Verify
        assert task.id not in task_queue.tasks 