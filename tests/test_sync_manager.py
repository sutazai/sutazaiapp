"""Tests for the SyncManager module."""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from datetime import datetime, timedelta

from core_system.orchestrator.models import OrchestratorConfig, SyncStatus
from core_system.orchestrator.sync_manager import SyncManager

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
def sync_manager(config):
    return SyncManager(config)

@pytest.mark.asyncio
async def test_init(config):
    """Test SyncManager initialization."""
    manager = SyncManager(config)
    assert manager.config == config
    assert manager.primary_server == config.primary_server
    assert manager.secondary_server == config.secondary_server
    assert manager.sync_interval == config.sync_interval
    assert manager.is_running == False
    assert manager.last_sync_time is None
    assert manager.sync_task is None

@pytest.mark.asyncio
async def test_start_stop(sync_manager):
    """Test starting and stopping the synchronization manager."""
    # Create a mock for asyncio.create_task
    mock_create_task = AsyncMock()
    mock_task = AsyncMock()
    mock_create_task.return_value = mock_task

    with patch("asyncio.create_task", mock_create_task):
        # Test start
        await sync_manager.start()
        assert sync_manager.is_running == True
        mock_create_task.assert_called_once()

        # Set the sync task
        sync_manager.sync_task = mock_task

        # Test stop
        await sync_manager.stop()
        assert sync_manager.is_running == False
        mock_task.cancel.assert_called_once()

@pytest.mark.asyncio
async def test_deploy(sync_manager):
    """Test deploying changes to a target server."""
    target_server = "192.168.100.100"
    await sync_manager.deploy(target_server)
    # This test verifies the method runs without errors

@pytest.mark.asyncio
async def test_rollback(sync_manager):
    """Test rolling back changes on a target server."""
    target_server = "192.168.100.100"
    # This test verifies the method runs without errors
    await sync_manager.rollback(target_server)

@pytest.mark.asyncio
async def test_sync(sync_manager):
    """Test synchronization process."""
    # Test that sync runs without errors and updates last_sync_time
    before_sync = datetime.now()
    sync_manager.sync()
    after_sync = datetime.now()

    assert sync_manager.last_sync_time is not None
    assert before_sync <= sync_manager.last_sync_time <= after_sync

@pytest.mark.asyncio
async def test_sync_loop(sync_manager):
    """Test the sync loop."""
    # Patch the sync method to track calls
    with patch.object(sync_manager, "sync") as mock_sync:
        # Start the sync loop
        sync_manager.is_running = True
        sync_task = asyncio.create_task(sync_manager._sync_loop())

        # Wait for the loop to run a few times
        await asyncio.sleep(0.3)

        # Stop the loop
        sync_manager.is_running = False
        await asyncio.sleep(0.1)  # Give it time to complete
        sync_task.cancel()  # Cancel the task

        # Verify that sync was called at least once
        assert mock_sync.called

@pytest.mark.asyncio
async def test_sync_loop_error_handling(sync_manager):
    """Test error handling in the sync loop."""
    # Make sync raise an exception
    with patch.object(sync_manager, "sync") as mock_sync:
        mock_sync.side_effect = Exception("Test error")

        # Start the sync loop
        sync_manager.is_running = True
        sync_task = asyncio.create_task(sync_manager._sync_loop())

        # Wait for the loop to run a few times
        await asyncio.sleep(0.3)

        # Stop the loop
        sync_manager.is_running = False
        await asyncio.sleep(0.1)  # Give it time to complete
        sync_task.cancel()  # Cancel the task

        # Verify that sync was called at least once despite the error
        assert mock_sync.called

@pytest.mark.asyncio
async def test_sync_with_server(sync_manager):
    """Test synchronization with a specific server."""
    # This test would ideally mock network requests
    # and verify the sync process with another server
    pass
