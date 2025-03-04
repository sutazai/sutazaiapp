#!/bin/bash

# Script to fix asynchronous test issues in SutazaiApp
# Focuses on the mock objects in async contexts that are causing test failures

echo "Starting async test fix process..."

# SSH key to use
SSH_KEY="/root/.ssh/sutazaiapp_sync_key"
REMOTE_SERVER="root@192.168.100.100"

# Fix tests with AsyncMock issues
echo "Fixing tests with AsyncMock issues..."
ssh -i $SSH_KEY $REMOTE_SERVER "cd /opt/sutazaiapp && cat > tests/async_utils.py << 'EOF'
\"\"\"Utilities for async testing.\"\"\"
import asyncio
from unittest.mock import AsyncMock as BaseAsyncMock, MagicMock, patch


class AsyncMock(BaseAsyncMock):
    \"\"\"Enhanced AsyncMock that can be awaited in tests.\"\"\"
    
    def __await__(self):
        future = asyncio.Future()
        future.set_result(self.return_value)
        return future.__await__()

    async def __aenter__(self):
        return self.return_value

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


def async_mock_task():
    \"\"\"Create a mock task that can be awaited.\"\"\"
    mock = AsyncMock()
    # Set up the mock task so it can be awaited after cancel
    mock.cancel.return_value = None
    return mock


def async_manager_patch():
    \"\"\"Patch asyncio.create_task for testing async managers.\"\"\"
    def patch_decorator(func):
        async def wrapper(*args, **kwargs):
            with patch('asyncio.create_task', side_effect=async_mock_task):
                return await func(*args, **kwargs)
        return wrapper
    return patch_decorator
EOF"

# Update agent_manager test
echo "Updating agent manager test..."
ssh -i $SSH_KEY $REMOTE_SERVER "cd /opt/sutazaiapp && cat > tests/test_agent_manager_fixed.py << 'EOF'
\"\"\"Tests for the agent manager module.\"\"\"
import pytest
from unittest.mock import patch, MagicMock
import asyncio
import json
from datetime import datetime, timedelta

from core_system.orchestrator.agent_manager import AgentManager
from core_system.orchestrator.exceptions import AgentNotFoundError
from core_system.orchestrator.models import AgentStatus, AgentType

from tests.async_utils import AsyncMock, async_mock_task, async_manager_patch


@pytest.fixture
def agent_manager():
    \"\"\"Create an agent manager for testing.\"\"\"
    return AgentManager()


@pytest.fixture
def sample_agent():
    \"\"\"Create a sample agent for testing.\"\"\"
    return {
        \"id\": \"agent1\",
        \"type\": \"worker\",
        \"status\": \"idle\",
        \"capabilities\": [\"task1\", \"task2\"]
    }


@pytest.mark.asyncio
async def test_register_agent(agent_manager, sample_agent):
    \"\"\"Test registering an agent.\"\"\"
    # Register the agent
    result = await agent_manager.register_agent(sample_agent)
    
    # Check the result
    assert result == True
    
    # Check that the agent was added to the registry
    assert sample_agent[\"id\"] in agent_manager.agents
    assert agent_manager.agents[sample_agent[\"id\"]] == sample_agent


@pytest.mark.asyncio
async def test_register_duplicate_agent(agent_manager, sample_agent):
    \"\"\"Test registering a duplicate agent.\"\"\"
    # Register the agent
    await agent_manager.register_agent(sample_agent)
    
    # Register the same agent again
    result = await agent_manager.register_agent(sample_agent)
    
    # Check the result (should be False for duplicate)
    assert result == False


@pytest.mark.asyncio
async def test_unregister_agent(agent_manager, sample_agent):
    \"\"\"Test unregistering an agent.\"\"\"
    # Register the agent
    await agent_manager.register_agent(sample_agent)
    
    # Unregister the agent
    result = await agent_manager.unregister_agent(sample_agent[\"id\"])
    
    # Check the result
    assert result == True
    
    # Check that the agent was removed from the registry
    assert sample_agent[\"id\"] not in agent_manager.agents


@pytest.mark.asyncio
async def test_unregister_unknown_agent(agent_manager):
    \"\"\"Test unregistering an unknown agent.\"\"\"
    # Unregister an unknown agent
    result = await agent_manager.unregister_agent(\"unknown_agent\")
    
    # Check the result (should be False for unknown agent)
    assert result == False


@pytest.mark.asyncio
async def test_get_agent(agent_manager, sample_agent):
    \"\"\"Test getting an agent.\"\"\"
    # Register the agent
    await agent_manager.register_agent(sample_agent)
    
    # Get the agent
    agent = await agent_manager.get_agent(sample_agent[\"id\"])
    
    # Check the agent
    assert agent == sample_agent


@pytest.mark.asyncio
async def test_get_unknown_agent(agent_manager):
    \"\"\"Test getting an unknown agent.\"\"\"
    # Try to get an unknown agent
    with pytest.raises(AgentNotFoundError):
        await agent_manager.get_agent(\"unknown_agent\")


@pytest.mark.asyncio
async def test_get_agents(agent_manager, sample_agent):
    \"\"\"Test getting all agents.\"\"\"
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    
    # Create another agent
    another_agent = {
        \"id\": \"agent2\",
        \"type\": \"processor\",
        \"status\": \"idle\",
        \"capabilities\": [\"task3\", \"task4\"]
    }
    
    # Register the second agent
    await agent_manager.register_agent(another_agent)
    
    # Get all agents
    agents = await agent_manager.get_agents()
    
    # Check the agents
    assert len(agents) == 2
    assert sample_agent[\"id\"] in [a[\"id\"] for a in agents]
    assert another_agent[\"id\"] in [a[\"id\"] for a in agents]


@pytest.mark.asyncio
async def test_get_agents_by_type(agent_manager, sample_agent):
    \"\"\"Test getting agents by type.\"\"\"
    # Register a worker agent
    await agent_manager.register_agent(sample_agent)
    
    # Create a processor agent
    processor_agent = {
        \"id\": \"agent2\",
        \"type\": \"processor\",
        \"status\": \"idle\",
        \"capabilities\": [\"task3\", \"task4\"]
    }
    
    # Register the processor agent
    await agent_manager.register_agent(processor_agent)
    
    # Get worker agents
    workers = await agent_manager.get_agents_by_type(\"worker\")
    
    # Check the workers
    assert len(workers) == 1
    assert workers[0][\"id\"] == sample_agent[\"id\"]
    
    # Get processor agents
    processors = await agent_manager.get_agents_by_type(\"processor\")
    
    # Check the processors
    assert len(processors) == 1
    assert processors[0][\"id\"] == processor_agent[\"id\"]


@pytest.mark.asyncio
async def test_get_agents_by_capability(agent_manager, sample_agent):
    \"\"\"Test getting agents by capability.\"\"\"
    # Register an agent with capabilities
    await agent_manager.register_agent(sample_agent)
    
    # Create another agent with different capabilities
    another_agent = {
        \"id\": \"agent2\",
        \"type\": \"processor\",
        \"status\": \"idle\",
        \"capabilities\": [\"task2\", \"task3\"]
    }
    
    # Register the second agent
    await agent_manager.register_agent(another_agent)
    
    # Get agents with capability "task1"
    task1_agents = await agent_manager.get_agents_by_capability(\"task1\")
    
    # Check the agents with capability "task1"
    assert len(task1_agents) == 1
    assert task1_agents[0][\"id\"] == sample_agent[\"id\"]
    
    # Get agents with capability "task2"
    task2_agents = await agent_manager.get_agents_by_capability(\"task2\")
    
    # Check the agents with capability "task2"
    assert len(task2_agents) == 2
    assert sample_agent[\"id\"] in [a[\"id\"] for a in task2_agents]
    assert another_agent[\"id\"] in [a[\"id\"] for a in task2_agents]


@pytest.mark.asyncio
async def test_update_agent_status(agent_manager, sample_agent):
    \"\"\"Test updating agent status.\"\"\"
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    
    # Update the agent status
    status = AgentStatus.BUSY.value
    await agent_manager.update_agent_status(sample_agent[\"id\"], status)
    
    # Get the agent and check the status
    agent = await agent_manager.get_agent(sample_agent[\"id\"])
    assert agent[\"status\"] == status


@pytest.mark.asyncio
async def test_update_unknown_agent_status(agent_manager):
    \"\"\"Test updating status of an unknown agent.\"\"\"
    # Try to update the status of an unknown agent
    with pytest.raises(AgentNotFoundError):
        await agent_manager.update_agent_status(\"unknown_agent\", AgentStatus.BUSY.value)


@pytest.mark.asyncio
async def test_get_agent_status(agent_manager, sample_agent):
    \"\"\"Test getting agent status.\"\"\"
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    
    # Get status
    status = await agent_manager.get_agent_status(sample_agent[\"id\"])
    
    # Check the status
    assert status == sample_agent[\"status\"]


@pytest.mark.asyncio
async def test_get_unknown_agent_status(agent_manager):
    \"\"\"Test getting status of an unknown agent.\"\"\"
    # Try to get the status of an unknown agent
    with pytest.raises(AgentNotFoundError):
        await agent_manager.get_agent_status(\"unknown_agent\")


@pytest.mark.asyncio
@async_manager_patch()
async def test_start_stop(agent_manager):
    \"\"\"Test starting and stopping the agent manager.\"\"\"
    # Test start
    await agent_manager.start()
    assert agent_manager.is_running == True
    assert agent_manager.heartbeat_task is not None
    
    # Test stop
    await agent_manager.stop()
    assert agent_manager.is_running == False


@pytest.mark.asyncio
async def test_heartbeat_check(agent_manager, sample_agent):
    \"\"\"Test heartbeat check.\"\"\"
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    
    # Update the last heartbeat time to a recent time
    agent_manager.last_heartbeat[sample_agent[\"id\"]] = datetime.now()
    
    # Run heartbeat check
    await agent_manager._heartbeat_check()
    
    # Check that the agent is still registered
    assert sample_agent[\"id\"] in agent_manager.agents
    
    # Update the last heartbeat time to an old time
    agent_manager.last_heartbeat[sample_agent[\"id\"]] = datetime.now() - timedelta(minutes=10)
    
    # Run heartbeat check
    await agent_manager._heartbeat_check()
    
    # Check that the agent was unregistered
    assert sample_agent[\"id\"] not in agent_manager.agents


@pytest.mark.asyncio
async def test_update_heartbeat(agent_manager, sample_agent):
    \"\"\"Test updating agent heartbeat.\"\"\"
    # Register an agent
    await agent_manager.register_agent(sample_agent)
    
    # Update heartbeat
    await agent_manager.update_heartbeat(sample_agent[\"id\"])
    
    # Check that the last heartbeat was updated
    assert sample_agent[\"id\"] in agent_manager.last_heartbeat
    # Check that the timestamp is recent
    heartbeat_time = agent_manager.last_heartbeat[sample_agent[\"id\"]]
    now = datetime.now()
    assert (now - heartbeat_time).total_seconds() < 1  # Within a second


@pytest.mark.asyncio
async def test_update_unknown_agent_heartbeat(agent_manager):
    \"\"\"Test updating heartbeat of an unknown agent.\"\"\"
    # Try to update the heartbeat of an unknown agent
    with pytest.raises(AgentNotFoundError):
        await agent_manager.update_heartbeat(\"unknown_agent\")


@pytest.mark.asyncio
async def test_stop_heartbeat_monitor(agent_manager):
    \"\"\"Test stopping the heartbeat monitor.\"\"\"
    # Create a mock for the heartbeat task
    agent_manager.heartbeat_task = async_mock_task()
    agent_manager.is_running = True
    
    # Stop the heartbeat monitor
    await agent_manager.stop_heartbeat_monitor()
    
    # Check that is_running is False
    assert agent_manager.is_running == False
    # Check that cancel was called
    agent_manager.heartbeat_task.cancel.assert_called_once()
EOF"

# Copy the fixed test file over the original
echo "Replacing the original test file with the fixed version..."
ssh -i $SSH_KEY $REMOTE_SERVER "cd /opt/sutazaiapp && mv tests/test_agent_manager_fixed.py tests/test_agent_manager.py"

# Fix sync_manager and task_queue tests similarly
echo "Fixing sync_manager and task_queue tests..."
ssh -i $SSH_KEY $REMOTE_SERVER "cd /opt/sutazaiapp && sed -i 's/from unittest.mock import AsyncMock, Mock, patch/from tests.async_utils import AsyncMock, async_mock_task\\nfrom unittest.mock import Mock, patch/g' tests/test_sync_manager.py tests/test_task_queue.py"

# Fix the mock tasks in sync_manager test
ssh -i $SSH_KEY $REMOTE_SERVER "cd /opt/sutazaiapp && sed -i 's/mock_task = AsyncMock()/mock_task = async_mock_task()/g' tests/test_sync_manager.py tests/test_task_queue.py"

# Run the tests again to verify the fixes
echo "Running tests to verify fixes..."
ssh -i $SSH_KEY $REMOTE_SERVER "cd /opt/sutazaiapp && source venv/bin/activate && ./scripts/run_tests.sh"

echo "Async test fix process complete!" 