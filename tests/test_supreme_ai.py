"""Tests for the SupremeAIOrchestrator class."""
import pytest
from unittest.mock import Mock, AsyncMock, patch

from core_system.orchestrator.models import OrchestratorConfig
from core_system.orchestrator.exceptions import OrchestratorError
from core_system.orchestrator.supreme_ai import SupremeAIOrchestrator

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
    with patch("core_system.orchestrator.task_queue.TaskQueue.start", new_callable=AsyncMock), \
         patch("core_system.orchestrator.agent_manager.AgentManager.start", new_callable=AsyncMock), \
         patch("core_system.orchestrator.sync_manager.SyncManager.start", new_callable=AsyncMock):
        orchestrator = SupremeAIOrchestrator(config)
        orchestrator.is_running = True  # Set to running state
        yield orchestrator

@pytest.mark.asyncio
async def test_init(config):
    """Test orchestrator initialization."""
    orchestrator = SupremeAIOrchestrator(config)
    assert orchestrator.config == config
    assert orchestrator.task_queue is not None
    assert orchestrator.agent_manager is not None
    assert orchestrator.sync_manager is not None
    assert orchestrator.is_running == False

@pytest.mark.asyncio
async def test_start_stop():
    """Test starting and stopping the orchestrator."""
    config = OrchestratorConfig(
        primary_server="192.168.100.28",
        secondary_server="192.168.100.100",
        sync_interval=60,
        max_agents=10,
        task_timeout=3600,
    )

    # Create mocks for the components
    with patch("core_system.orchestrator.task_queue.TaskQueue.start", new_callable=AsyncMock) as mock_task_start, \
         patch("core_system.orchestrator.task_queue.TaskQueue.stop", new_callable=AsyncMock) as mock_task_stop, \
         patch("core_system.orchestrator.agent_manager.AgentManager.start", new_callable=AsyncMock) as mock_agent_start, \
         patch("core_system.orchestrator.agent_manager.AgentManager.stop", new_callable=AsyncMock) as mock_agent_stop, \
         patch("core_system.orchestrator.sync_manager.SyncManager.start", new_callable=AsyncMock) as mock_sync_start, \
         patch("core_system.orchestrator.sync_manager.SyncManager.stop", new_callable=AsyncMock) as mock_sync_stop:

        # Create the orchestrator
        orchestrator = SupremeAIOrchestrator(config)

        # Test start
        await orchestrator.start()
        assert orchestrator.is_running == True
        mock_task_start.assert_called_once()
        mock_agent_start.assert_called_once()
        mock_sync_start.assert_called_once()

        # Test stop
        await orchestrator.stop()
        assert orchestrator.is_running == False
        mock_task_stop.assert_called_once()
        mock_agent_stop.assert_called_once()
        mock_sync_stop.assert_called_once()

@pytest.mark.asyncio
async def test_submit_task(orchestrator):
    """Test task submission."""
    with patch("core_system.orchestrator.task_queue.TaskQueue.submit", new_callable=AsyncMock) as mock_submit:
        task = {"id": "task1", "type": "test"}
        await orchestrator.submit_task(task)
        mock_submit.assert_called_once_with(task)

@pytest.mark.asyncio
async def test_submit_task_not_running():
    """Test task submission when orchestrator is not running."""
    config = OrchestratorConfig(
        primary_server="192.168.100.28",
        secondary_server="192.168.100.100",
        sync_interval=60,
        max_agents=10,
        task_timeout=3600,
    )

    orchestrator = SupremeAIOrchestrator(config)
    orchestrator.is_running = False

    with pytest.raises(OrchestratorError):
        await orchestrator.submit_task({"id": "task1", "type": "test"})

@pytest.mark.asyncio
async def test_register_agent(orchestrator):
    """Test agent registration."""
    with patch("core_system.orchestrator.agent_manager.AgentManager.register_agent", new_callable=AsyncMock) as mock_register:
        agent = {"id": "agent1", "type": "test"}
        await orchestrator.register_agent(agent)
        mock_register.assert_called_once_with(agent)

@pytest.mark.asyncio
async def test_register_agent_not_running():
    """Test agent registration when orchestrator is not running."""
    config = OrchestratorConfig(
        primary_server="192.168.100.28",
        secondary_server="192.168.100.100",
        sync_interval=60,
        max_agents=10,
        task_timeout=3600,
    )

    orchestrator = SupremeAIOrchestrator(config)
    orchestrator.is_running = False

    with pytest.raises(OrchestratorError):
        await orchestrator.register_agent({"id": "agent1", "type": "test"})

@pytest.mark.asyncio
async def test_get_agent_status(orchestrator):
    """Test getting agent status."""
    with patch("core_system.orchestrator.agent_manager.AgentManager.get_agent_status", new_callable=AsyncMock) as mock_status:
        mock_status.return_value = {"status": "running"}
        status = await orchestrator.get_agent_status("agent1")
        assert status == {"status": "running"}
        mock_status.assert_called_once_with("agent1")

@pytest.mark.asyncio
async def test_list_agents(orchestrator):
    """Test listing agents."""
    with patch("core_system.orchestrator.agent_manager.AgentManager.list_agents", new_callable=AsyncMock) as mock_list:
        mock_list.return_value = [{"id": "agent1"}, {"id": "agent2"}]
        agents = await orchestrator.list_agents()
        assert agents == [{"id": "agent1"}, {"id": "agent2"}]
        mock_list.assert_called_once()

@pytest.mark.asyncio
async def test_start_agent(orchestrator):
    """Test starting an agent."""
    with patch("core_system.orchestrator.agent_manager.AgentManager.start_agent", new_callable=AsyncMock) as mock_start:
        await orchestrator.start_agent("agent1")
        mock_start.assert_called_once_with("agent1")

@pytest.mark.asyncio
async def test_stop_agent(orchestrator):
    """Test stopping an agent."""
    with patch("core_system.orchestrator.agent_manager.AgentManager.stop_agent", new_callable=AsyncMock) as mock_stop:
        await orchestrator.stop_agent("agent1")
        mock_stop.assert_called_once_with("agent1")

@pytest.mark.asyncio
async def test_start_sync(orchestrator):
    """Test starting synchronization."""
    with patch("core_system.orchestrator.sync_manager.SyncManager.start", new_callable=AsyncMock) as mock_start:
        await orchestrator.start_sync()
        mock_start.assert_called_once()

@pytest.mark.asyncio
async def test_stop_sync(orchestrator):
    """Test stopping synchronization."""
    with patch("core_system.orchestrator.sync_manager.SyncManager.stop", new_callable=AsyncMock) as mock_stop:
        await orchestrator.stop_sync()
        mock_stop.assert_called_once()

@pytest.mark.asyncio
async def test_deploy(orchestrator):
    """Test deploying changes."""
    with patch("core_system.orchestrator.sync_manager.SyncManager.deploy", new_callable=AsyncMock) as mock_deploy:
        target_server = "192.168.100.100"
        await orchestrator.deploy(target_server)
        mock_deploy.assert_called_once_with(target_server)

@pytest.mark.asyncio
async def test_rollback(orchestrator):
    """Test rolling back changes."""
    with patch("core_system.orchestrator.sync_manager.SyncManager.rollback", new_callable=AsyncMock) as mock_rollback:
        target_server = "192.168.100.100"
        await orchestrator.rollback(target_server)
        mock_rollback.assert_called_once_with(target_server)

@pytest.mark.asyncio
async def test_error_handling(orchestrator):
    """Test error handling in task submission."""
    with patch("core_system.orchestrator.task_queue.TaskQueue.submit", new_callable=AsyncMock) as mock_submit:
        mock_submit.side_effect = Exception("Test error")
        with pytest.raises(OrchestratorError):
            await orchestrator.submit_task({"id": "task1"})
