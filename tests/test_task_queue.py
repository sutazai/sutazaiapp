"""Tests for the TaskQueue module."""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from core_system.orchestrator.models import OrchestratorConfig, Task, TaskStatus
from core_system.orchestrator.task_queue import TaskQueue

@pytest.fixture
def config():
    return OrchestratorConfig(
        primary_server="192.168.100.28",
        secondary_server="192.168.100.100",
        sync_interval=60,
        max_agents=10,
        task_timeout=3600,
    )

@pytest.fixture
def task_queue(config):
    return TaskQueue(config)

@pytest.fixture
def sample_task():
    return {
        "id": "task1",
        "type": "test",
        "parameters": {"key": "value"},
        "priority": 5
    }

@pytest.mark.asyncio
async def test_init(config):
    """Test TaskQueue initialization."""
    queue = TaskQueue(config)
    assert queue.config == config
    assert queue.is_running == False
    assert queue.task_timeout == 3600
    assert queue.tasks == []
    assert queue._queue == []

@pytest.mark.asyncio
async def test_submit(task_queue, sample_task):
    """Test task submission."""
    task_queue.submit(sample_task)
    assert len(task_queue.tasks) == 1
    assert task_queue.tasks[0].id == sample_task["id"]
    assert task_queue.tasks[0].type == sample_task["type"]
    assert task_queue.tasks[0].parameters == sample_task["parameters"]
    assert task_queue.tasks[0].priority == sample_task["priority"]

@pytest.mark.asyncio
async def test_process(task_queue):
    """Test starting task processing."""
    mock_create_task = AsyncMock()
    mock_task = AsyncMock()
    mock_create_task.return_value = mock_task

    with patch("asyncio.create_task", mock_create_task):
        task_queue.process()
        assert task_queue.is_running == True
        mock_create_task.assert_called_once()

@pytest.mark.asyncio
async def test_start_stop(task_queue):
    """Test starting and stopping the task queue."""
    # Create a mock for asyncio.create_task
    mock_create_task = AsyncMock()
    mock_task = AsyncMock()
    mock_create_task.return_value = mock_task

    with patch("asyncio.create_task", mock_create_task):
        # Test start
        await task_queue.start()
        assert task_queue.is_running == True
        mock_create_task.assert_called_once()

        # Set the process task
        task_queue.process_task = mock_task

        # Test stop
        await task_queue.stop()
        assert task_queue.is_running == False
        mock_task.cancel.assert_called_once()

@pytest.mark.asyncio
async def test_process_loop(task_queue, sample_task):
    """Test task processing loop."""
    task_queue.submit(sample_task)

    # Start the processing loop but run it for only a short time
    task_queue.is_running = True
    process_task = asyncio.create_task(task_queue._process_loop())

    # Wait a short time to allow some processing
    await asyncio.sleep(0.3)

    # Stop the loop
    task_queue.is_running = False
    await asyncio.sleep(0.1)  # Give it time to complete
    process_task.cancel()  # Cancel the task

    # Tasks should have been processed
    assert len(task_queue.tasks) == 0

@pytest.mark.asyncio
async def test_put_get(task_queue):
    """Test putting tasks into and getting tasks from the queue."""
    task1 = Task(id="task1", type="test", parameters={}, priority=1)
    task2 = Task(id="task2", type="test", parameters={}, priority=2)

    # Put tasks in queue
    result1 = await task_queue.put(task1)
    result2 = await task_queue.put(task2)

    assert result1 == True
    assert result2 == True
    assert task_queue.size() == 2

    # Get tasks in priority order
    retrieved_task = await task_queue.get()
    assert retrieved_task.id == "task2"  # Higher priority task first

    retrieved_task = await task_queue.get()
    assert retrieved_task.id == "task1"

    # Queue should be empty now
    assert task_queue.size() == 0
    assert await task_queue.get() is None

@pytest.mark.asyncio
async def test_queue_full(task_queue):
    """Test behavior when queue is full."""
    task_queue.max_size = 2

    task1 = Task(id="task1", type="test", parameters={}, priority=1)
    task2 = Task(id="task2", type="test", parameters={}, priority=2)
    task3 = Task(id="task3", type="test", parameters={}, priority=3)

    await task_queue.put(task1)
    await task_queue.put(task2)
    result = await task_queue.put(task3)

    assert result == False
    assert task_queue.size() == 2

@pytest.mark.asyncio
async def test_peek(task_queue):
    """Test peeking at the next task without removing it."""
    task1 = Task(id="task1", type="test", parameters={}, priority=1)
    task2 = Task(id="task2", type="test", parameters={}, priority=2)

    await task_queue.put(task1)
    await task_queue.put(task2)

    # Peek should return the highest priority task without removing it
    peeked_task = await task_queue.peek()
    assert peeked_task.id == "task2"
    assert task_queue.size() == 2

    # Empty queue peek
    await task_queue.clear()
    assert await task_queue.peek() is None

@pytest.mark.asyncio
async def test_remove(task_queue):
    """Test removing a specific task from the queue."""
    task1 = Task(id="task1", type="test", parameters={}, priority=1)
    task2 = Task(id="task2", type="test", parameters={}, priority=2)

    await task_queue.put(task1)
    await task_queue.put(task2)

    # Remove the first task
    result = await task_queue.remove("task1")
    assert result == True
    assert task_queue.size() == 1

    # Attempt to remove a non-existent task
    result = await task_queue.remove("nonexistent")
    assert result == False

    # Remove the remaining task
    result = await task_queue.remove("task2")
    assert result == True
    assert task_queue.size() == 0

@pytest.mark.asyncio
async def test_clear(task_queue):
    """Test clearing all tasks from the queue."""
    task1 = Task(id="task1", type="test", parameters={}, priority=1)
    task2 = Task(id="task2", type="test", parameters={}, priority=2)

    await task_queue.put(task1)
    await task_queue.put(task2)
    assert task_queue.size() == 2

    await task_queue.clear()
    assert task_queue.size() == 0

@pytest.mark.asyncio
async def test_get_all_tasks(task_queue):
    """Test getting all tasks in the queue."""
    task1 = Task(id="task1", type="test", parameters={}, priority=1)
    task2 = Task(id="task2", type="test", parameters={}, priority=2)

    await task_queue.put(task1)
    await task_queue.put(task2)

    tasks = await task_queue.get_all_tasks()
    assert len(tasks) == 2

    # Empty queue
    await task_queue.clear()
    tasks = await task_queue.get_all_tasks()
    assert len(tasks) == 0

@pytest.mark.asyncio
async def test_update_task_priority(task_queue):
    """Test updating the priority of a task in the queue."""
    task1 = Task(id="task1", type="test", parameters={}, priority=1)
    task2 = Task(id="task2", type="test", parameters={}, priority=2)

    await task_queue.put(task1)
    await task_queue.put(task2)

    # Update priority of task1
    result = await task_queue.update_task_priority("task1", 3)
    assert result == True

    # Now task1 should be retrieved first due to higher priority
    retrieved_task = await task_queue.get()
    assert retrieved_task.id == "task1"
    assert retrieved_task.priority == 3

    # Attempt to update a non-existent task
    result = await task_queue.update_task_priority("nonexistent", 5)
    assert result == False

@pytest.mark.asyncio
async def test_get_tasks_by_status(task_queue):
    """Test filtering tasks by status."""
    task1 = Task(id="task1", type="test", parameters={}, priority=1)
    task2 = Task(id="task2", type="test", parameters={}, priority=2)
    task1.status = TaskStatus.PENDING
    task2.status = TaskStatus.IN_PROGRESS

    await task_queue.put(task1)
    await task_queue.put(task2)

    pending_tasks = await task_queue.get_tasks_by_status(TaskStatus.PENDING)
    assert len(pending_tasks) == 1
    assert pending_tasks[0].id == "task1"

    in_progress_tasks = await task_queue.get_tasks_by_status(TaskStatus.IN_PROGRESS)
    assert len(in_progress_tasks) == 1
    assert in_progress_tasks[0].id == "task2"
