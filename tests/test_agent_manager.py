"""Tests for the AgentManager module."""
import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch, MagicMock
from datetime import datetime, timedelta

from core_system.orchestrator.models import OrchestratorConfig, Agent, AgentStatus
from core_system.orchestrator.exceptions import AgentError, AgentNotFoundError
from core_system.orchestrator.agent_manager import AgentManager

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
def agent_manager(config):
    return AgentManager(config)

@pytest.fixture
def sample_agent():
    return {
        "id": "agent1",
        "type": "worker",
        "capabilities": ["task1", "task2"],
        "status": "idle"
    }

@pytest.mark.asyncio
async def test_init(config):
    """Test AgentManager initialization."""
    manager = AgentManager(config)
    assert manager.config == config
    assert manager.agents == {}
    assert manager.max_agents == 10
    assert manager.is_running == False
    assert manager.heartbeat_task is None

@pytest.mark.asyncio
async def test_start_stop(agent_manager):
    """Test starting and stopping the agent manager."""
    with patch.object(asyncio, "create_task") as mock_create_task:
        # Test start
        await agent_manager.start()
        assert agent_manager.is_running == True
        mock_create_task.assert_called_once()
        
        # Create a mock task for stopping
        mock_task = MagicMock()
        mock_task.cancel = MagicMock()
        # Make the mock awaitable by patching the agent_manager.stop method
        with patch.object(agent_manager, 'heartbeat_task', mock_task):
            # Test stop
            await agent_manager.stop()
            assert agent_manager.is_running == False
            mock_task.cancel.assert_called_once()

@pytest.mark.asyncio
async def test_register_agent(agent_manager, sample_agent):
    """Test registering a new agent."""
    # Register an agent
    agent_obj = await agent_manager.register_agent(sample_agent)
    
    # The agent should be in the agents dictionary
    assert sample_agent["id"] in agent_manager.agents
    registered_agent = agent_manager.agents[sample_agent["id"]]
    assert registered_agent.id == sample_agent["id"]
    assert registered_agent.type == sample_agent["type"]
    assert registered_agent.capabilities == sample_agent["capabilities"]
    
    # Test max agents limit
    agent_manager.max_agents = 1
    
    # Try to register another agent
    second_agent = {
        "id": "agent2",
        "type": "worker",
        "capabilities": ["task1"],
        "status": "idle"
    }
    
    # Should fail because max_agents is reached
    with pytest.raises(AgentError):
        await agent_manager.register_agent(second_agent)

@pytest.mark.asyncio
async def test_unregister_agent(agent_manager, sample_agent):
    """Test unregistering an agent."""
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    assert sample_agent["id"] in agent_manager.agents
    
    # Add a mock for _handle_agent_failure in case it's needed
    with patch.object(agent_manager, "_handle_agent_failure", new_callable=AsyncMock) as mock_handle:
        # Unregister the agent
        await agent_manager.unregister_agent(sample_agent["id"])
        assert sample_agent["id"] not in agent_manager.agents
        
        # Unregistering a non-existent agent should return gracefully
        await agent_manager.unregister_agent("nonexistent")

@pytest.mark.asyncio
async def test_start_agent(agent_manager, sample_agent):
    """Test starting an agent."""
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    
    # Start the agent
    await agent_manager.start_agent(sample_agent["id"])
    assert agent_manager.agents[sample_agent["id"]].status == AgentStatus.BUSY
    
    # Starting a non-existent agent should raise an exception
    with pytest.raises(AgentNotFoundError):
        await agent_manager.start_agent("nonexistent")

@pytest.mark.asyncio
async def test_stop_agent(agent_manager, sample_agent):
    """Test stopping an agent."""
    # Register and start an agent
    await agent_manager.register_agent(sample_agent)
    await agent_manager.start_agent(sample_agent["id"])
    
    # Stop the agent
    await agent_manager.stop_agent(sample_agent["id"])
    assert agent_manager.agents[sample_agent["id"]].status == AgentStatus.IDLE
    
    # Stopping a non-existent agent should raise an exception
    with pytest.raises(AgentNotFoundError):
        await agent_manager.stop_agent("nonexistent")

@pytest.mark.asyncio
async def test_update_heartbeat(agent_manager, sample_agent):
    """Test updating agent heartbeat."""
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    original_time = agent_manager.agents[sample_agent["id"]].last_heartbeat
    
    # Wait a bit
    await asyncio.sleep(0.1)
    
    # Update heartbeat
    agent_manager.update_heartbeat(sample_agent["id"])
    new_time = agent_manager.agents[sample_agent["id"]].last_heartbeat
    
    # New time should be later than original time
    assert new_time > original_time
    
    # Updating heartbeat for a non-existent agent should be handled gracefully
    agent_manager.update_heartbeat("nonexistent")

@pytest.mark.asyncio
async def test_get_agent_status(agent_manager, sample_agent):
    """Test getting agent status."""
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    
    # Get status
    status = await agent_manager.get_agent_status(sample_agent["id"])
    assert status["status"] == "IDLE"
    assert status["id"] == sample_agent["id"]
    
    # Getting status for a non-existent agent should raise an exception
    with pytest.raises(AgentNotFoundError):
        await agent_manager.get_agent_status("nonexistent")

@pytest.mark.asyncio
async def test_list_agents(agent_manager, sample_agent):
    """Test listing all agents."""
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    
    # List agents
    agents = await agent_manager.list_agents()
    assert len(agents) == 1
    assert agents[0]["id"] == sample_agent["id"]
    
    # Register another agent
    second_agent = {
        "id": "agent2",
        "type": "worker",
        "capabilities": ["task1"],
        "status": "idle"
    }
    await agent_manager.register_agent(second_agent)
    
    # List agents again
    agents = await agent_manager.list_agents()
    assert len(agents) == 2
    agent_ids = [agent["id"] for agent in agents]
    assert sample_agent["id"] in agent_ids
    assert second_agent["id"] in agent_ids

@pytest.mark.asyncio
async def test_heartbeat_loop(agent_manager, sample_agent):
    """Test the heartbeat monitoring loop."""
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    
    # Patch _check_agent_health for testing
    with patch.object(agent_manager, "_check_agent_health", new_callable=AsyncMock) as mock_check:
        # Set up the heartbeat task
        agent_manager.is_running = True
        heartbeat_task = asyncio.create_task(agent_manager._heartbeat_loop())
        
        # Wait a short time for the loop to run
        await asyncio.sleep(0.3)
        
        # Stop the loop
        agent_manager.is_running = False
        await heartbeat_task
        
        # Verify that _check_agent_health was called
        assert mock_check.called

@pytest.mark.asyncio
async def test_check_agent_health(agent_manager, sample_agent):
    """Test checking agent health."""
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    
    # Check health with recent heartbeat - should remain IDLE
    await agent_manager._check_agent_health()
    assert agent_manager.agents[sample_agent["id"]].status == AgentStatus.IDLE
    
    # Set the heartbeat to a time far in the past
    agent_manager.agents[sample_agent["id"]].last_heartbeat = datetime.now() - timedelta(minutes=2)
    
    # Check health with old heartbeat - should change to OFFLINE
    await agent_manager._check_agent_health()
    assert agent_manager.agents[sample_agent["id"]].status == AgentStatus.OFFLINE

@pytest.mark.asyncio
async def test_shutdown_all_agents(agent_manager):
    """Test shutting down all agents."""
    # Patch stop_heartbeat_monitor and unregister_agent for testing
    with patch.object(agent_manager, "stop_heartbeat_monitor", new_callable=AsyncMock) as mock_stop, \
         patch.object(agent_manager, "unregister_agent", new_callable=AsyncMock) as mock_unreg:
         
        # Register multiple agents
        for i in range(3):
            agent = {
                "id": f"agent{i}",
                "type": "worker",
                "capabilities": ["task1"],
                "status": "idle"
            }
            await agent_manager.register_agent(agent)
        
        # Shutdown all agents
        await agent_manager.shutdown_all_agents()
        
        # Verify stop_heartbeat_monitor was called
        mock_stop.assert_called_once()
        
        # Verify unregister_agent was called for each agent
        assert mock_unreg.call_count == 3

@pytest.mark.asyncio
async def test_start_heartbeat_monitor(agent_manager):
    """Test starting the heartbeat monitor."""
    with patch.object(asyncio, "create_task") as mock_create_task:
        # Start the heartbeat monitor
        await agent_manager.start_heartbeat_monitor()
        
        # Verify it started correctly
        assert agent_manager.is_running == True
        mock_create_task.assert_called_once()

@pytest.mark.asyncio
async def test_stop_heartbeat_monitor(agent_manager):
    """Test stopping the heartbeat monitor."""
    # Set up a mock heartbeat task
    mock_task = AsyncMock()
    agent_manager.heartbeat_task = mock_task
    agent_manager.is_running = True
    
    # Stop the heartbeat monitor
    await agent_manager.stop_heartbeat_monitor()
    
    # Verify it stopped correctly
    assert agent_manager.is_running == False
    mock_task.cancel.assert_called_once() 