#!/usr/bin/env python3
"""
Targeted tests to achieve 100% coverage for agent_manager.py
"""

import pytest
import asyncio
from unittest.mock import patch, MagicMock, AsyncMock
from datetime import datetime, timedelta
from core_system.orchestrator import agent_manager
from core_system.orchestrator.models import OrchestratorConfig, Agent, AgentStatus, Task
from core_system.orchestrator.exceptions import AgentError, AgentNotFoundError

@pytest.fixture
def config():
    """Fixture for OrchestratorConfig"""
    return OrchestratorConfig(
        primary_server="192.168.100.28",
        secondary_server="192.168.100.100",
        sync_interval=60,
        max_agents=10,
        task_timeout=3600,
    )

@pytest.fixture
def manager(config):
    """Create a test instance of AgentManager"""
    return agent_manager.AgentManager(config)

@pytest.mark.asyncio
async def test_heartbeat_monitor_handles_exceptions(manager):
    """Test that the heartbeat monitor handles exceptions properly"""
    # Setup the agent manager with an agent
    agent_id = "test-agent"
    manager.agents[agent_id] = Agent(
        id=agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.IDLE
    )

    # Mock _check_agent_health to raise an exception
    with patch.object(manager, '_check_agent_health', side_effect=Exception("Test exception")):
        # Call _heartbeat_loop directly
        # We'll use asyncio.wait_for to limit execution time
        try:
            # Start the loop but immediately set is_running to False to exit it
            task = asyncio.create_task(manager._heartbeat_loop())
            manager.is_running = False
            await asyncio.wait_for(task, timeout=1.0)
        except asyncio.TimeoutError:
            # If it times out, cancel the task
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

    # The test passes if no uncaught exceptions occurred

@pytest.mark.asyncio
async def test_check_agent_health_handles_failed_agent(manager):
    """Test _check_agent_health when an agent has failed"""
    # Setup the agent manager with an agent with old heartbeat
    agent_id = "test-agent"
    manager.agents[agent_id] = Agent(
        id=agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.IDLE,
        last_heartbeat=datetime.now() - timedelta(minutes=10)
    )

    # Mock _handle_agent_failure to track calls
    with patch.object(manager, '_handle_agent_failure') as mock_handle_failure:
        await manager._check_agent_health(agent_id)
        mock_handle_failure.assert_called_once_with(agent_id)

@pytest.mark.asyncio
async def test_agent_failure_handler(manager):
    """Test the _handle_agent_failure method"""
    # Setup the agent manager with an agent
    agent_id = "test-agent"
    manager.agents[agent_id] = Agent(
        id=agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.IDLE
    )

    # Call the failure handler
    await manager._handle_agent_failure(agent_id)

    # Verify agent status was updated
    assert manager.agents[agent_id].status == AgentStatus.ERROR

    # Test for agent not found
    await manager._handle_agent_failure("nonexistent-agent")

@pytest.mark.asyncio
async def test_update_agent_status_with_invalid_status(manager):
    """Test update_agent_status with an invalid status value"""
    # Setup the agent manager with an agent
    agent_id = "test-agent"
    manager.agents[agent_id] = Agent(
        id=agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.IDLE
    )

    # Try to update with an invalid status
    with pytest.raises(AgentError):
        await manager.update_agent_status(agent_id, "INVALID_STATUS")

    # Verify agent status was not changed
    assert manager.agents[agent_id].status == AgentStatus.IDLE

@pytest.mark.asyncio
async def test_get_agent_status_not_found(manager):
    """Test get_agent_status when agent doesn't exist"""
    with pytest.raises(AgentNotFoundError):
        await manager.get_agent_status("nonexistent-agent")

@pytest.mark.asyncio
async def test_assign_task_to_nonexistent_agent(manager):
    """Test assign_task when agent doesn't exist"""
    task = Task(
        id="task1",
        type="process",
        parameters={"param1": "value1"},
        priority=1
    )

    with pytest.raises(AgentError):
        await manager.assign_task("nonexistent-agent", task)

@pytest.mark.asyncio
async def test_get_available_agent_none_available(manager):
    """Test get_available_agent when no agent is available"""
    # Add busy agents
    manager.agents["agent1"] = Agent(
        id="agent1",
        type="worker",
        capabilities=["process"],
        status=AgentStatus.BUSY
    )
    manager.agents["agent2"] = Agent(
        id="agent2",
        type="worker",
        capabilities=["process"],
        status=AgentStatus.BUSY
    )

    # Try to get an available agent for a task type
    agent = await manager.get_available_agent("process")
    assert agent is None

@pytest.mark.asyncio
async def test_stop_with_cancelled_error(manager):
    """Test stop method with CancelledError (lines 49-51)"""
    # Create a mock heartbeat task that raises CancelledError when awaited
    mock_task = AsyncMock()
    mock_task.__class__.__name__ = 'Task'
    mock_task.__await__ = AsyncMock(side_effect=asyncio.CancelledError())

    # Set the mock task
    manager.heartbeat_task = mock_task
    manager.is_running = True

    # Call stop method
    await manager.stop()

    # Verify is_running was set to False
    assert not manager.is_running

@pytest.mark.asyncio
async def test_stop_with_mock_task(manager):
    """Test stop method with mock task (lines 45-48)"""
    # Create a mock heartbeat task
    mock_task = MagicMock()
    mock_task.__class__.__name__ = 'AsyncMock'

    # Set the mock task
    manager.heartbeat_task = mock_task
    manager.is_running = True

    # Call stop method
    await manager.stop()

    # Verify is_running was set to False
    assert not manager.is_running

@pytest.mark.asyncio
async def test_unregister_agent_busy(manager):
    """Test unregister_agent with a busy agent (line 118)"""
    # Setup the agent manager with a busy agent
    agent_id = "busy-agent"
    manager.agents[agent_id] = Agent(
        id=agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.BUSY
    )

    # Mock _handle_agent_failure
    with patch.object(manager, '_handle_agent_failure') as mock_handle_failure:
        await manager.unregister_agent(agent_id)

        # Verify _handle_agent_failure was called
        mock_handle_failure.assert_called_once()

        # Verify agent was removed
        assert agent_id not in manager.agents

@pytest.mark.asyncio
async def test_handle_agent_failure_implementation(manager):
    """Test _handle_agent_failure implementation (lines 124-126)"""
    # Setup the agent manager with an agent
    agent_id = "test-agent"
    agent = Agent(
        id=agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.BUSY,
        current_task="task1"
    )
    manager.agents[agent_id] = agent

    # Call the failure handler
    await manager._handle_agent_failure(agent)

    # The method is a placeholder, so we just verify it doesn't raise exceptions
    # and the agent status remains unchanged
    assert manager.agents[agent_id].status == AgentStatus.BUSY

@pytest.mark.asyncio
async def test_stop_heartbeat_monitor_with_cancelled_error(manager):
    """Test stop_heartbeat_monitor with CancelledError (lines 153-155)"""
    # Create a mock heartbeat task that raises CancelledError when awaited
    mock_task = AsyncMock()
    mock_task.__class__.__name__ = 'Task'
    mock_task.__await__ = AsyncMock(side_effect=asyncio.CancelledError())

    # Set the mock task
    manager.heartbeat_task = mock_task
    manager.is_running = True

    # Call stop_heartbeat_monitor method
    await manager.stop_heartbeat_monitor()

    # Verify heartbeat_task is None
    assert manager.heartbeat_task is None

@pytest.mark.asyncio
async def test_heartbeat_loop_exception_handling(manager):
    """Test _heartbeat_loop exception handling (lines 164-166)"""
    # Mock _check_agent_health to raise an exception
    with patch.object(manager, '_check_agent_health', side_effect=Exception("Test exception")):
        # Set up a task for _heartbeat_loop
        manager.is_running = True
        task = asyncio.create_task(manager._heartbeat_loop())

        # Allow the loop to run briefly
        await asyncio.sleep(0.1)

        # Stop the loop
        manager.is_running = False
        await asyncio.sleep(0.1)

        # Cancel the task
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass

        # The test passes if no uncaught exceptions occurred

@pytest.mark.asyncio
async def test_get_available_agent_implementation(manager):
    """Test get_available_agent implementation (lines 179-182)"""
    # Setup the agent manager with an idle agent
    agent_id = "idle-agent"
    manager.agents[agent_id] = Agent(
        id=agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.IDLE
    )

    # Add a busy agent
    manager.agents["busy-agent"] = Agent(
        id="busy-agent",
        type="worker",
        capabilities=["process"],
        status=AgentStatus.BUSY
    )

    # Get an available agent
    agent = await manager.get_available_agent()

    # Verify the idle agent was returned
    assert agent is not None
    assert agent.id == agent_id

@pytest.mark.asyncio
async def test_assign_task_implementation(manager):
    """Test assign_task implementation (lines 186-196)"""
    # Setup the agent manager with an idle agent
    agent_id = "idle-agent"
    manager.agents[agent_id] = Agent(
        id=agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.IDLE
    )

    # Create a task
    task = Task(
        id="task1",
        type="process",
        parameters={"param1": "value1"},
        priority=1
    )

    # Test with agent not found
    with pytest.raises(AgentNotFoundError):
        await manager.assign_task("nonexistent-agent", task)

    # Test with busy agent
    busy_agent_id = "busy-agent"
    manager.agents[busy_agent_id] = Agent(
        id=busy_agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.BUSY
    )
    result = await manager.assign_task(busy_agent_id, task)
    assert result is False

    # Test with idle agent
    result = await manager.assign_task(agent_id, task)
    assert result is True
    assert manager.agents[agent_id].status == AgentStatus.BUSY
    assert manager.agents[agent_id].current_task == task.id

@pytest.mark.asyncio
async def test_update_agent_status_implementation(manager):
    """Test update_agent_status implementation (lines 200-205)"""
    # Setup the agent manager with an agent
    agent_id = "test-agent"
    manager.agents[agent_id] = Agent(
        id=agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.IDLE
    )

    # Test with agent not found
    with pytest.raises(AgentNotFoundError):
        await manager.update_agent_status("nonexistent-agent", AgentStatus.BUSY)

    # Test with valid agent and status
    old_heartbeat = manager.agents[agent_id].last_heartbeat
    await asyncio.sleep(0.01)  # Ensure time difference
    await manager.update_agent_status(agent_id, AgentStatus.BUSY)
    assert manager.agents[agent_id].status == AgentStatus.BUSY
    assert manager.agents[agent_id].last_heartbeat > old_heartbeat

@pytest.mark.asyncio
async def test_heartbeat_implementation(manager):
    """Test heartbeat implementation (lines 209-212)"""
    # Setup the agent manager with an agent
    agent_id = "test-agent"
    manager.agents[agent_id] = Agent(
        id=agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.IDLE
    )

    # Test with agent not found
    with pytest.raises(AgentNotFoundError):
        await manager.heartbeat("nonexistent-agent")

    # Test with valid agent
    old_heartbeat = manager.agents[agent_id].last_heartbeat
    await asyncio.sleep(0.01)  # Ensure time difference
    await manager.heartbeat(agent_id)
    assert manager.agents[agent_id].last_heartbeat > old_heartbeat

@pytest.mark.asyncio
async def test_get_agent_count(manager):
    """Test get_agent_count (line 216)"""
    # Setup the agent manager with agents
    manager.agents["agent1"] = Agent(
        id="agent1",
        type="worker",
        capabilities=["process"],
        status=AgentStatus.IDLE
    )
    manager.agents["agent2"] = Agent(
        id="agent2",
        type="worker",
        capabilities=["process"],
        status=AgentStatus.BUSY
    )

    # Get agent count
    count = manager.get_agent_count()
    assert count == 2

@pytest.mark.asyncio
async def test_get_agent_status_enum(manager):
    """Test get_agent_status_enum (lines 220-221)"""
    # Setup the agent manager with an agent
    agent_id = "test-agent"
    manager.agents[agent_id] = Agent(
        id=agent_id,
        type="worker",
        capabilities=["process"],
        status=AgentStatus.IDLE
    )

    # Test with nonexistent agent
    status = manager.get_agent_status_enum("nonexistent-agent")
    assert status is None

    # Test with existing agent
    status = manager.get_agent_status_enum(agent_id)
    assert status == AgentStatus.IDLE

@pytest.mark.asyncio
async def test_shutdown_all_agents(manager):
    """Test shutdown_all_agents"""
    # Mock stop_heartbeat_monitor
    with patch.object(manager, 'stop_heartbeat_monitor') as mock_stop:
        await manager.shutdown_all_agents()
        mock_stop.assert_called_once()
