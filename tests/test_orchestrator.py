#!/usr/bin/env python3.11
"""
Tests for Supreme AI Orchestrator

This module contains comprehensive tests for the orchestrator system.
"""

import asyncio
import os
import pytest
from datetime import datetime, timedelta
from unittest.mock import Mock, AsyncMock, patch

from core_system.orchestrator.supreme_ai import SupremeAIOrchestrator, OrchestratorConfig
from core_system.orchestrator.models import Task, Agent, TaskStatus, AgentStatus, SyncStatus
from core_system.orchestrator.exceptions import (
    OrchestratorError,
    AgentError,
    SyncError,
    QueueError
)

# Test configuration
TEST_CONFIG = OrchestratorConfig(
    primary_server="http://localhost:8000",
    secondary_server="http://localhost:8010",
    sync_interval=1,
    max_agents=2,
    task_timeout=5
)

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
async def orchestrator(config):
    with patch("core_system.orchestrator.task_queue.TaskQueue.start", new_callable=AsyncMock) as mock_task_start, \
         patch("core_system.orchestrator.task_queue.TaskQueue.stop", new_callable=AsyncMock) as mock_task_stop, \
         patch("core_system.orchestrator.agent_manager.AgentManager.start", new_callable=AsyncMock) as mock_agent_start, \
         patch("core_system.orchestrator.agent_manager.AgentManager.stop", new_callable=AsyncMock) as mock_agent_stop, \
         patch("core_system.orchestrator.sync_manager.SyncManager.start", new_callable=AsyncMock) as mock_sync_start, \
         patch("core_system.orchestrator.sync_manager.SyncManager.stop", new_callable=AsyncMock) as mock_sync_stop:
        orchestrator = SupremeAIOrchestrator(config)
        orchestrator.is_running = True  # Set to running state
        yield orchestrator

@pytest.fixture
async def orchestrator_instance():
    """Create a test orchestrator instance"""
    orchestrator = SupremeAIOrchestrator(TEST_CONFIG)
    yield orchestrator
    await orchestrator.stop()

@pytest.fixture
def test_task():
    """Create a test task"""
    return Task(
        id="test-task-001",
        type="test",
        parameters={"key": "value"},
        priority=1
    )

@pytest.fixture
def test_agent():
    """Create a test agent"""
    return Agent(
        id="test-agent-001",
        type="test",
        capabilities=["test"],
        status=AgentStatus.IDLE
    )

@pytest.mark.asyncio
async def test_orchestrator_initialization(orchestrator):
    assert orchestrator.config is not None
    assert orchestrator.sync_manager is not None
    assert orchestrator.agent_manager is not None
    assert orchestrator.is_running is True

@pytest.mark.asyncio
async def test_orchestrator_start_stop(orchestrator):
    with patch("core_system.orchestrator.sync_manager.SyncManager.start", new_callable=AsyncMock) as mock_start, \
         patch("core_system.orchestrator.sync_manager.SyncManager.stop", new_callable=AsyncMock) as mock_stop:
        await orchestrator.start_sync()
        mock_start.assert_called_once()
        await orchestrator.stop_sync()
        mock_stop.assert_called_once()

@pytest.mark.asyncio
async def test_task_submission(orchestrator):
    with patch("core_system.orchestrator.task_queue.TaskQueue.submit", new_callable=AsyncMock) as mock_submit:
        task = {"id": "task1", "type": "test"}
        await orchestrator.submit_task(task)
        mock_submit.assert_called_once_with(task)

@pytest.mark.asyncio
async def test_agent_registration(orchestrator):
    with patch("core_system.orchestrator.agent_manager.AgentManager.register_agent", new_callable=AsyncMock) as mock_register:
        agent = {"id": "agent1", "type": "test"}
        await orchestrator.register_agent(agent)
        mock_register.assert_called_once_with(agent)

@pytest.mark.asyncio
async def test_task_processing(orchestrator):
    with patch("core_system.orchestrator.task_queue.TaskQueue.process") as mock_process:
        orchestrator.process_tasks()
        mock_process.assert_called_once()

@pytest.mark.asyncio
async def test_synchronization(orchestrator):
    with patch("core_system.orchestrator.sync_manager.SyncManager.sync") as mock_sync:
        orchestrator.sync()
        mock_sync.assert_called_once()

@pytest.mark.asyncio
async def test_error_handling(orchestrator):
    with patch("core_system.orchestrator.task_queue.TaskQueue.submit", new_callable=AsyncMock) as mock_submit:
        mock_submit.side_effect = Exception("Test error")
        with pytest.raises(Exception):
            await orchestrator.submit_task({"id": "task1"})

@pytest.mark.asyncio
async def test_task_priority(orchestrator):
    with patch("core_system.orchestrator.task_queue.TaskQueue.submit", new_callable=AsyncMock) as mock_submit:
        task1 = {"id": "task1", "priority": 1}
        task2 = {"id": "task2", "priority": 2}
        await orchestrator.submit_task(task1)
        await orchestrator.submit_task(task2)
        assert mock_submit.call_args_list[0][0][0]["priority"] < mock_submit.call_args_list[1][0][0]["priority"]

@pytest.mark.asyncio
async def test_agent_heartbeat(orchestrator):
    with patch("core_system.orchestrator.agent_manager.AgentManager.update_heartbeat") as mock_heartbeat:
        agent_id = "agent1"
        orchestrator.update_agent_heartbeat(agent_id)
        mock_heartbeat.assert_called_once_with(agent_id)

@pytest.mark.asyncio
async def test_start_sync(orchestrator):
    with patch("core_system.orchestrator.sync_manager.SyncManager.start", new_callable=AsyncMock) as mock_start:
        await orchestrator.start_sync()
        mock_start.assert_called_once()

@pytest.mark.asyncio
async def test_stop_sync(orchestrator):
    with patch("core_system.orchestrator.sync_manager.SyncManager.stop", new_callable=AsyncMock) as mock_stop:
        await orchestrator.stop_sync()
        mock_stop.assert_called_once()

@pytest.mark.asyncio
async def test_deploy(orchestrator):
    with patch("core_system.orchestrator.sync_manager.SyncManager.deploy", new_callable=AsyncMock) as mock_deploy:
        target_server = "192.168.100.100"
        await orchestrator.deploy(target_server)
        mock_deploy.assert_called_once_with(target_server)

@pytest.mark.asyncio
async def test_rollback(orchestrator):
    with patch("core_system.orchestrator.sync_manager.SyncManager.rollback", new_callable=AsyncMock) as mock_rollback:
        target_server = "192.168.100.100"
        await orchestrator.rollback(target_server)
        mock_rollback.assert_called_once_with(target_server)

@pytest.mark.asyncio
async def test_start_agent(orchestrator):
    with patch("core_system.orchestrator.agent_manager.AgentManager.start_agent", new_callable=AsyncMock) as mock_start:
        agent_id = "test_agent"
        await orchestrator.start_agent(agent_id)
        mock_start.assert_called_once_with(agent_id)

@pytest.mark.asyncio
async def test_stop_agent(orchestrator):
    with patch("core_system.orchestrator.agent_manager.AgentManager.stop_agent", new_callable=AsyncMock) as mock_stop:
        agent_id = "test_agent"
        await orchestrator.stop_agent(agent_id)
        mock_stop.assert_called_once_with(agent_id)

@pytest.mark.asyncio
async def test_list_agents(orchestrator):
    with patch("core_system.orchestrator.agent_manager.AgentManager.list_agents", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = ["agent1", "agent2"]
        agents = await orchestrator.list_agents()
        assert agents == ["agent1", "agent2"]
        mock_list.assert_called_once()

@pytest.mark.asyncio
async def test_get_agent_status(orchestrator):
    with patch("core_system.orchestrator.agent_manager.AgentManager.get_agent_status", new_callable=AsyncMock) as mock_status:
        agent_id = "test_agent"
        mock_status.return_value = {"status": "running"}
        status = await orchestrator.get_agent_status(agent_id)
        assert status == {"status": "running"}
        mock_status.assert_called_once_with(agent_id)

if __name__ == "__main__":
    pytest.main([__file__]) 