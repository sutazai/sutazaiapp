#!/usr/bin/env python3
"""
Additional tests to achieve 100% coverage for sync_manager
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from core_system.orchestrator import sync_manager
from core_system.orchestrator.models import OrchestratorConfig, SyncData


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
def sync_manager_fixture(config):
    """Fixture for SyncManager class."""
    return sync_manager.SyncManager(config)


@pytest.mark.asyncio
async def test_func_start(sync_manager_fixture):
    """Test for start function in sync_manager."""
    # Set initial state
    sync_manager_fixture.is_running = False

    # Start the sync manager
    await sync_manager_fixture.start()

    # Verify it's running
    assert sync_manager_fixture.is_running is True
    assert sync_manager_fixture.sync_task is not None


@pytest.mark.asyncio
async def test_func_stop(sync_manager_fixture):
    """Test for stop function in sync_manager."""
    # Set initial state
    sync_manager_fixture.is_running = True
    sync_manager_fixture.sync_task = AsyncMock()

    # Stop the sync manager
    await sync_manager_fixture.stop()

    # Verify it's stopped
    assert sync_manager_fixture.is_running is False


@pytest.mark.asyncio
async def test_func__sync_loop(sync_manager_fixture):
    """Test for _sync_loop function in sync_manager."""
    # Set initial state
    sync_manager_fixture.is_running = True

    # Mock methods to control execution flow
    sync_manager_fixture.sync = AsyncMock()
    sync_manager_fixture.sync.side_effect = [None, Exception("Test exception")]

    # Run the sync loop with a controlled exit
    try:
        await sync_manager_fixture._sync_loop()
    except Exception:
        pass

    # Verify sync was called
    assert sync_manager_fixture.sync.call_count > 0


@pytest.mark.asyncio
async def test_sync_method(sync_manager_fixture):
    """Test the sync method"""
    # Mock _get_sync_data and _send_sync_data
    sync_manager_fixture._get_sync_data = AsyncMock(return_value={"test": "data"})
    sync_manager_fixture._send_sync_data = AsyncMock()

    # Call sync
    await sync_manager_fixture.sync()

    # Verify methods were called
    sync_manager_fixture._get_sync_data.assert_called_once()
    sync_manager_fixture._send_sync_data.assert_called_once()


@pytest.mark.asyncio
async def test_get_sync_data_method(sync_manager_fixture):
    """Test the _get_sync_data method"""
    # Mock agent_manager and task_queue
    sync_manager_fixture.agent_manager = MagicMock()
    sync_manager_fixture.task_queue = MagicMock()

    # Set return values for mocks
    sync_manager_fixture.agent_manager.get_all_agents = AsyncMock(return_value={"agent1": "data"})
    sync_manager_fixture.task_queue.get_all_tasks = AsyncMock(return_value={"task1": "data"})

    # Call _get_sync_data
    result = await sync_manager_fixture._get_sync_data()

    # Verify result
    assert "timestamp" in result
    assert "server_id" in result
    assert "tasks" in result
    assert "agents" in result


@pytest.mark.asyncio
async def test_send_sync_data_method(sync_manager_fixture):
    """Test the _send_sync_data method"""
    # Create test data
    test_data = {"test": "data"}

    # Mock _send_to_secondary and _send_to_primary
    sync_manager_fixture._send_to_secondary = AsyncMock()
    sync_manager_fixture._send_to_primary = AsyncMock()

    # Call _send_sync_data
    await sync_manager_fixture._send_sync_data(test_data)

    # Verify methods were called
    assert sync_manager_fixture._send_to_secondary.called or sync_manager_fixture._send_to_primary.called


@pytest.mark.asyncio
async def test_handle_sync_data_method(sync_manager_fixture):
    """Test the handle_sync_data method"""
    # Create test sync data
    test_data = {
        "tasks": {"task1": "data"},
        "agents": {"agent1": "data"}
    }

    # Mock agent_manager and task_queue
    sync_manager_fixture.agent_manager = MagicMock()
    sync_manager_fixture.task_queue = MagicMock()
    sync_manager_fixture.agent_manager.update_from_sync = AsyncMock()
    sync_manager_fixture.task_queue.update_from_sync = AsyncMock()

    # Call handle_sync_data
    await sync_manager_fixture.handle_sync_data(test_data)

    # Verify methods were called
    sync_manager_fixture.agent_manager.update_from_sync.assert_called_once_with(test_data["agents"])
    sync_manager_fixture.task_queue.update_from_sync.assert_called_once_with(test_data["tasks"])
