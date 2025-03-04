#!/usr/bin/env python3.11
"""
Complete coverage tests for the agent manager module.
These tests are specifically designed to cover lines that aren't covered by existing tests.
"""

import asyncio
import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import pytest
from datetime import datetime, timedelta

from core_system.orchestrator.models import (
    OrchestratorConfig, Agent, AgentStatus, Task, TaskStatus
)
from core_system.orchestrator.agent_manager import AgentManager
from core_system.orchestrator.exceptions import AgentError, AgentNotFoundError

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
def agent_manager(config):
    """Create an agent manager instance for testing."""
    return AgentManager(config=config)

@pytest.fixture
def agent_dict():
    """Create a test agent dictionary."""
    return {
        "id": "test-agent-1",
        "type": "test",
        "capabilities": ["capability1", "capability2"]
    }

@pytest.fixture
def registered_agent(agent_manager, agent_dict):
    """Create and register a test agent."""
    agent_obj = Agent(
        id=agent_dict["id"],
        type=agent_dict["type"],
        capabilities=agent_dict.get("capabilities", []),
        status=AgentStatus.IDLE
    )
    agent_manager.agents[agent_obj.id] = agent_obj
    return agent_obj

@pytest.fixture
def task():
    """Create a test task."""
    return Task(
        id="test-task-1",
        type="test",
        parameters={"param1": "value1"}
    )

class TestAgentManagerCompleteCoverage:
    """Tests to ensure complete coverage of the agent manager."""

    async def test_stop_with_mock_task(self, agent_manager):
        """Test stopping the agent manager when heartbeat_task is a mock."""
        # Set up a mock heartbeat task
        mock_task = MagicMock()
        mock_task.__class__.__name__ = 'AsyncMock'
        agent_manager.heartbeat_task = mock_task
        agent_manager.is_running = True
        
        # Call stop method
        await agent_manager.stop()
        
        # Verify
        assert not agent_manager.is_running
        mock_task.cancel.assert_called_once()

    async def test_stop_with_real_task(self, agent_manager):
        """Test stopping the agent manager with a real asyncio task."""
        # Create a real asyncio task
        async def dummy_task():
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
        
        agent_manager.is_running = True
        agent_manager.heartbeat_task = asyncio.create_task(dummy_task())
        
        # Call stop method
        await agent_manager.stop()
        
        # Verify
        assert not agent_manager.is_running
        assert agent_manager.heartbeat_task.cancelled()

    async def test_start_agent_not_found(self, agent_manager):
        """Test starting a non-existent agent."""
        with pytest.raises(AgentNotFoundError):
            await agent_manager.start_agent("non-existent-agent")

    async def test_stop_agent_not_found(self, agent_manager):
        """Test stopping a non-existent agent."""
        with pytest.raises(AgentNotFoundError):
            await agent_manager.stop_agent("non-existent-agent")

    async def test_register_agent_max_exceeded(self, agent_manager, agent_dict):
        """Test registering an agent when max agents is reached."""
        # Set max_agents to 0 to trigger the error
        agent_manager.max_agents = 0
        
        with pytest.raises(AgentError):
            await agent_manager.register_agent(agent_dict)

    async def test_unregister_busy_agent(self, agent_manager, registered_agent):
        """Test unregistering an agent that is busy."""
        # Mark the agent as busy
        registered_agent.status = AgentStatus.BUSY
        
        # Mock the _handle_agent_failure method
        agent_manager._handle_agent_failure = AsyncMock()
        
        # Call unregister_agent
        await agent_manager.unregister_agent(registered_agent.id)
        
        # Verify
        agent_manager._handle_agent_failure.assert_called_once_with(registered_agent)
        assert registered_agent.id not in agent_manager.agents

    async def test_handle_agent_failure(self, agent_manager, registered_agent):
        """Test handling agent failure."""
        # Mark the agent as busy
        registered_agent.status = AgentStatus.BUSY
        registered_agent.current_task = "test-task"
        
        # Call _handle_agent_failure
        await agent_manager._handle_agent_failure(registered_agent)
        
        # Verify
        assert registered_agent.status == AgentStatus.ERROR
        assert registered_agent.current_task is None

    async def test_update_heartbeat(self, agent_manager, registered_agent):
        """Test update_heartbeat with a valid agent."""
        original_heartbeat = registered_agent.last_heartbeat
        
        # Wait a brief moment to ensure time difference
        await asyncio.sleep(0.01)
        
        # Call update_heartbeat
        agent_manager.update_heartbeat(registered_agent.id)
        
        # Verify
        assert registered_agent.last_heartbeat > original_heartbeat

    async def test_stop_heartbeat_monitor_with_mock_task(self, agent_manager):
        """Test stop_heartbeat_monitor with a mock task."""
        # Set up a mock heartbeat task
        mock_task = MagicMock()
        mock_task.__class__.__name__ = 'AsyncMock'
        agent_manager.heartbeat_task = mock_task
        agent_manager.is_running = True
        
        # Call stop_heartbeat_monitor
        await agent_manager.stop_heartbeat_monitor()
        
        # Verify
        assert not agent_manager.is_running
        mock_task.cancel.assert_called_once()

    async def test_stop_heartbeat_monitor_with_real_task(self, agent_manager):
        """Test stop_heartbeat_monitor with a real asyncio task."""
        # Create a real asyncio task
        async def dummy_task():
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
        
        agent_manager.is_running = True
        agent_manager.heartbeat_task = asyncio.create_task(dummy_task())
        
        # Call stop_heartbeat_monitor
        await agent_manager.stop_heartbeat_monitor()
        
        # Verify
        assert not agent_manager.is_running
        assert agent_manager.heartbeat_task.cancelled()

    async def test_heartbeat_loop_with_exception(self, agent_manager):
        """Test the _heartbeat_loop with an exception in _check_agent_health."""
        # Mock _check_agent_health to raise an exception
        agent_manager._check_agent_health = AsyncMock(side_effect=Exception("Test exception"))
        
        # Override asyncio.sleep to make the test faster
        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            # Set is_running to True, then False after one iteration
            agent_manager.is_running = True
            
            async def stop_after_one_iteration():
                await asyncio.sleep(0.05)
                agent_manager.is_running = False
            
            stop_task = asyncio.create_task(stop_after_one_iteration())
            
            # Run the heartbeat loop
            await agent_manager._heartbeat_loop()
            
            # Verify
            agent_manager._check_agent_health.assert_called_once()
            # Should call sleep with 1 after an error
            assert mock_sleep.call_args_list[-1][0][0] == 1
            
            # Clean up
            stop_task.cancel()
            try:
                await stop_task
            except asyncio.CancelledError:
                pass

    async def test_check_agent_health(self, agent_manager, registered_agent):
        """Test _check_agent_health marking an agent as offline."""
        # Set the agent's last_heartbeat to be older than 1 minute
        registered_agent.last_heartbeat = datetime.now() - timedelta(minutes=2)
        
        # Call _check_agent_health
        await agent_manager._check_agent_health()
        
        # Verify
        assert registered_agent.status == AgentStatus.OFFLINE

    async def test_get_available_agent_none_available(self, agent_manager, registered_agent):
        """Test get_available_agent when no agents are available."""
        # Mark the agent as busy
        registered_agent.status = AgentStatus.BUSY
        
        # Call get_available_agent
        result = await agent_manager.get_available_agent()
        
        # Verify
        assert result is None

    async def test_assign_task_agent_not_found(self, agent_manager, task):
        """Test assign_task with a non-existent agent."""
        with pytest.raises(AgentNotFoundError):
            await agent_manager.assign_task("non-existent-agent", task)

    async def test_assign_task_agent_not_idle(self, agent_manager, registered_agent, task):
        """Test assign_task with an agent that is not idle."""
        # Mark the agent as busy
        registered_agent.status = AgentStatus.BUSY
        
        # Call assign_task
        result = await agent_manager.assign_task(registered_agent.id, task)
        
        # Verify
        assert not result

    async def test_update_agent_status_agent_not_found(self, agent_manager):
        """Test update_agent_status with a non-existent agent."""
        with pytest.raises(AgentNotFoundError):
            await agent_manager.update_agent_status("non-existent-agent", AgentStatus.IDLE)

    async def test_heartbeat_agent_not_found(self, agent_manager):
        """Test heartbeat with a non-existent agent."""
        with pytest.raises(AgentNotFoundError):
            await agent_manager.heartbeat("non-existent-agent")

    def test_get_agent_status_enum_agent_not_found(self, agent_manager):
        """Test get_agent_status_enum with a non-existent agent."""
        result = agent_manager.get_agent_status_enum("non-existent-agent")
        assert result is None

    async def test_shutdown_all_agents(self, agent_manager):
        """Test shutdown_all_agents."""
        # Register a few agents
        for i in range(3):
            agent = Agent(
                id=f"test-agent-{i}",
                type="test",
                capabilities=["test"],
                status=AgentStatus.IDLE
            )
            agent_manager.agents[agent.id] = agent
        
        # Mock stop_heartbeat_monitor
        agent_manager.stop_heartbeat_monitor = AsyncMock()
        
        # Call shutdown_all_agents
        await agent_manager.shutdown_all_agents()
        
        # Verify
        agent_manager.stop_heartbeat_monitor.assert_called_once()
        assert len(agent_manager.agents) == 0 