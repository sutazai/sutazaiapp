#!/usr/bin/env python3.11
"""
Complete coverage tests for the sync_manager module.
These tests are specifically designed to cover lines that aren't covered by existing tests.
"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from datetime import datetime

from core_system.orchestrator.models import (
    OrchestratorConfig, SyncData, ServerConfig, SyncStatus,
    Agent, Task, AgentStatus, TaskStatus
)
from core_system.orchestrator.sync_manager import SyncManager
from core_system.orchestrator.exceptions import SyncError

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
def sync_manager(config):
    """Create a SyncManager instance for testing."""
    return SyncManager(config=config)

@pytest.fixture
def server_config():
    """Create a test server configuration."""
    return ServerConfig(
        id="test-server",
        host="localhost",
        port=8000,
        is_primary=True,
        sync_port=8001,
        api_key="test-api-key"
    )

@pytest.fixture
def sync_data():
    """Create test sync data."""
    return SyncData(
        timestamp=datetime.now(),
        server_id="test-server",
        tasks={},
        agents={},
        metadata={}
    )

class TestSyncManagerCompleteCoverage:
    """Tests to ensure complete coverage of the SyncManager."""

    async def test_sync_loop_exception(self, sync_manager):
        """Test the _sync_loop with an exception."""
        # Mock sync to raise an exception
        sync_manager.sync = MagicMock(side_effect=Exception("Test exception"))
        
        # Override asyncio.sleep to make the test faster
        with patch("asyncio.sleep", new=AsyncMock()) as mock_sleep:
            # Set is_running to True, then False after one iteration
            sync_manager.is_running = True
            
            async def stop_after_one_iteration():
                await asyncio.sleep(0.05)
                sync_manager.is_running = False
            
            stop_task = asyncio.create_task(stop_after_one_iteration())
            
            # Run the sync loop
            await sync_manager._sync_loop()
            
            # Verify
            sync_manager.sync.assert_called_once()
            # Should call sleep with the sync_interval
            mock_sleep.assert_called_with(sync_manager.sync_interval)
            
            # Clean up
            stop_task.cancel()
            try:
                await stop_task
            except asyncio.CancelledError:
                pass

    async def test_stop_with_mock_task(self, sync_manager):
        """Test stopping the sync manager when sync_task is a mock."""
        # Set up a mock sync task
        mock_task = MagicMock()
        mock_task.__class__.__name__ = 'AsyncMock'
        sync_manager.sync_task = mock_task
        sync_manager.is_running = True
        
        # Call stop method
        await sync_manager.stop()
        
        # Verify
        assert not sync_manager.is_running
        mock_task.cancel.assert_called_once()

    async def test_stop_with_real_task(self, sync_manager):
        """Test stopping the sync manager with a real asyncio task."""
        # Create a real asyncio task
        async def dummy_task():
            try:
                while True:
                    await asyncio.sleep(0.1)
            except asyncio.CancelledError:
                pass
        
        sync_manager.is_running = True
        sync_manager.sync_task = asyncio.create_task(dummy_task())
        
        # Call stop method
        await sync_manager.stop()
        
        # Verify
        assert not sync_manager.is_running
        assert sync_manager.sync_task.cancelled()

    async def test_deploy_exception(self, sync_manager):
        """Test handling exceptions during deploy."""
        # Mock dependency
        sync_manager._sync_with_server = AsyncMock(side_effect=Exception("Test exception"))
        
        # Call deploy
        with pytest.raises(SyncError):
            await sync_manager.deploy("test-server")
        
        # Verify
        sync_manager._sync_with_server.assert_called_once_with("test-server")

    async def test_rollback_exception(self, sync_manager):
        """Test handling exceptions during rollback."""
        # Mock dependency
        sync_manager._sync_with_server = AsyncMock(side_effect=Exception("Test exception"))
        
        # Call rollback
        with pytest.raises(SyncError):
            await sync_manager.rollback("test-server")
        
        # Verify
        sync_manager._sync_with_server.assert_called_once_with("test-server")

    def test_sync_exception(self, sync_manager):
        """Test handling exceptions during sync."""
        # Mock dependency
        sync_manager._prepare_sync_data = MagicMock(side_effect=Exception("Test exception"))
        
        # Call sync
        # Should not raise an exception, but log it
        sync_manager.sync()
        
        # Verify
        sync_manager._prepare_sync_data.assert_called_once()

    async def test_get_status(self, sync_manager):
        """Test get_status method."""
        # Set some data for testing
        sync_manager.last_sync_time = datetime.now()
        sync_manager.last_sync_status = SyncStatus.SUCCESS
        
        # Call get_status
        result = await sync_manager.get_status()
        
        # Verify
        assert "last_sync_time" in result
        assert "status" in result
        assert result["status"] == SyncStatus.SUCCESS.name

    def test_prepare_sync_data(self, sync_manager):
        """Test _prepare_sync_data method."""
        # Mock dependencies
        task_queue = MagicMock()
        agent_manager = MagicMock()
        
        # Create some mock tasks and agents
        tasks = {
            "task1": Task(id="task1", type="test", parameters={}),
            "task2": Task(id="task2", type="test", parameters={})
        }
        
        agents = {
            "agent1": Agent(id="agent1", type="test", capabilities=[]),
            "agent2": Agent(id="agent2", type="test", capabilities=[])
        }
        
        task_queue.get_all_tasks.return_value = list(tasks.values())
        sync_manager.task_queue = task_queue
        
        sync_manager.agent_manager = agent_manager
        sync_manager.agent_manager.agents = agents
        
        # Call _prepare_sync_data
        result = sync_manager._prepare_sync_data()
        
        # Verify
        assert isinstance(result, SyncData)
        assert len(result.tasks) == 2
        assert len(result.agents) == 2
        assert "task1" in result.tasks
        assert "agent1" in result.agents

    async def test_sync_with_server(self, sync_manager, sync_data):
        """Test _sync_with_server method."""
        # Mock aiohttp ClientSession
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 200
        mock_response.json = AsyncMock(return_value={"status": "success"})
        mock_session.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Call _sync_with_server
            await sync_manager._sync_with_server("test-server", sync_data)
            
            # Verify
            mock_session.__aenter__.return_value.post.assert_called_once()

    async def test_sync_with_server_error_response(self, sync_manager, sync_data):
        """Test _sync_with_server method with an error response."""
        # Mock aiohttp ClientSession
        mock_session = AsyncMock()
        mock_response = AsyncMock()
        mock_response.status = 500
        mock_response.text = AsyncMock(return_value="Internal Server Error")
        mock_session.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Call _sync_with_server
            with pytest.raises(SyncError):
                await sync_manager._sync_with_server("test-server", sync_data)
            
            # Verify
            mock_session.__aenter__.return_value.post.assert_called_once()

    async def test_sync_with_server_exception(self, sync_manager, sync_data):
        """Test _sync_with_server method with an exception."""
        # Mock aiohttp ClientSession to raise an exception
        mock_session = AsyncMock()
        mock_session.__aenter__.return_value.post = AsyncMock(side_effect=Exception("Test exception"))
        
        with patch("aiohttp.ClientSession", return_value=mock_session):
            # Call _sync_with_server
            with pytest.raises(SyncError):
                await sync_manager._sync_with_server("test-server", sync_data)
            
            # Verify
            mock_session.__aenter__.return_value.post.assert_called_once() 