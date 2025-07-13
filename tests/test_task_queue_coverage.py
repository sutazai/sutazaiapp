#!/usr/bin/env python3
"""
Additional tests to achieve 100% coverage for task_queue
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core_system.orchestrator import task_queue
from core_system.orchestrator.models import OrchestratorConfig, Task


@pytest.fixture
def config():
    """Fixture for OrchestratorConfig."""
    return OrchestratorConfig(
        primary_server="192.168.100.28",
        secondary_server="192.168.100.100",
        sync_interval=60,
        max_agents=10,
        task_timeout=3600,
    )


@pytest.fixture
def task_queue_fixture(config):
    """Fixture for task_queue module."""
    return task_queue.TaskQueue(config)


@pytest.fixture
def taskqueue_fixture(config):
    """Fixture for TaskQueue class."""
    return task_queue.TaskQueue(config)


@pytest.mark.asyncio
async def test_task_queue_start(task_queue_fixture):
    """Test for start function in task_queue."""
    # Set initial state
    task_queue_fixture.is_running = False

    # Start task queue
    await task_queue_fixture.start()

    # Verify it's running
    assert task_queue_fixture.is_running is True


@pytest.mark.asyncio
async def test_task_queue_stop(task_queue_fixture):
    """Test for stop function in task_queue."""
    # Set up the fixture
    task_queue_fixture.is_running = True
    task_queue_fixture.processing_task = AsyncMock()

    # Call stop
    await task_queue_fixture.stop()

    # Verify it's stopped
    assert task_queue_fixture.is_running is False


@pytest.mark.asyncio
async def test_task_queue_process_loop(task_queue_fixture):
    """Test for _process_loop function in task_queue."""
    # Set initial state
    task_queue_fixture.is_running = True

    # Mock methods to control execution flow
    task_queue_fixture.process_next_task = AsyncMock()
    task_queue_fixture.process_next_task.side_effect = [None, Exception("Test exception")]

    # Run the process loop with a controlled exit
    try:
        await task_queue_fixture._process_loop()
    except Exception:
        pass

    # Verify process_next_task was called
    assert task_queue_fixture.process_next_task.call_count > 0


@pytest.mark.asyncio
async def test_task_queue_put(task_queue_fixture):
    """Test for put function in task_queue."""
    # Create a test task
    test_task = Task(
        id="test-task",
        type="test",
        parameters={},
        priority=1
    )

    # Add to queue
    await task_queue_fixture.put(test_task)

    # Verify task was added
    assert task_queue_fixture.size() > 0


@pytest.mark.asyncio
async def test_task_queue_get(task_queue_fixture):
    """Test for get function in task_queue."""
    # Create and add a test task
    test_task = Task(
        id="test-task",
        type="test",
        parameters={},
        priority=1
    )
    await task_queue_fixture.put(test_task)

    # Get task
    result = await task_queue_fixture.get()

    # Verify we got a task
    assert result is not None
    assert result.id == "test-task"


@pytest.mark.asyncio
def test_task_queue_size(task_queue_fixture):
    """Test for size function in task_queue."""
    # Initially queue should be empty
    initial_size = task_queue_fixture.size()
    assert initial_size == 0


@pytest.mark.asyncio
async def test_task_queue_get_by_type(task_queue_fixture):
    """Test for get_tasks_by_type function in task_queue."""
    # Create and add test tasks of different types
    test_task1 = Task(
        id="test-task-1",
        type="type1",
        parameters={},
        priority=1
    )
    test_task2 = Task(
        id="test-task-2",
        type="type2",
        parameters={},
        priority=1
    )

    await task_queue_fixture.put(test_task1)
    await task_queue_fixture.put(test_task2)

    # Get tasks by type
    result = await task_queue_fixture.get_tasks_by_type("type1")

    # Verify we got the right tasks
    assert len(result) > 0
    assert all(task.type == "type1" for task in result)


@pytest.mark.asyncio
async def test_task_queue_requeue(task_queue_fixture):
    """Test for requeue_failed_tasks function in task_queue."""
    # This test depends on the implementation details
    # Add implementation-specific test logic here
    result = await task_queue_fixture.requeue_failed_tasks()
    assert result is not None  # Replace with appropriate assertion


# Skip duplicate tests for the same class
# The taskqueue_fixture is the same as task_queue_fixture
