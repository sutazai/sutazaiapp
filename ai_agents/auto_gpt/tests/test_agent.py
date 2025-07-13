#!/usr/bin/env python3.11
"""Tests for the AutoGPT agent module."""

import pytest
from datetime import datetime
from unittest.mock import Mock, patch

    AutoGPTAgent,
    AgentConfig,
    AgentState,
    AgentError,
)
from ai_agents.auto_gpt.src.memory import Memory, Message
from ai_agents.auto_gpt.src.task import Task, TaskStatus
from ai_agents.auto_gpt.src.model import ModelManager, ModelConfig


@pytest.fixture
def mock_model():
    """Create a mock model manager."""
    model = Mock(spec=ModelManager)
    model.generate_response.return_value = "Test response"
    return model


@pytest.fixture
def mock_memory():
    """Create a mock memory instance."""
    memory = Mock(spec=Memory)
    memory.add_message.return_value = None
    memory.get_recent_messages.return_value = []
    return memory


@pytest.fixture
def agent_config():
    """Create a test agent configuration."""
    return AgentConfig(
        name="TestAgent",
        description="Test agent for unit tests",
        max_steps=5,
        max_memory=100,
        temperature=0.7,
        model_config=ModelConfig(
            model_name="test-model",
            max_tokens=1000,
            temperature=0.7,
        ),
    )


@pytest.fixture
def test_agent(agent_config, mock_model, mock_memory):
    """Create a test agent instance."""
    return AutoGPTAgent(
        config=agent_config,
        model=mock_model,
        memory=mock_memory,
    )


def test_agent_initialization(test_agent, agent_config):
    """Test agent initialization."""
    assert test_agent.name == agent_config.name
    assert test_agent.description == agent_config.description
    assert test_agent.max_steps == agent_config.max_steps
    assert test_agent.max_memory == agent_config.max_memory
    assert test_agent.temperature == agent_config.temperature
    assert test_agent.state == AgentState.IDLE
    assert test_agent.current_task is None
    assert test_agent.step_count == 0


def test_agent_start_task(test_agent, mock_memory):
    """Test starting a new task."""
    task = Task(
        objective="Test objective",
        context={"test_key": "test_value"},
    )

    test_agent.start_task(task)
    assert test_agent.current_task == task
    assert test_agent.state == AgentState.WORKING
    assert test_agent.step_count == 0

    # Verify memory was updated
    mock_memory.add_message.assert_called_once()


def test_agent_execute_step(test_agent, mock_model, mock_memory):
    """Test executing a single step."""
    task = Task(
        objective="Test objective",
        context={"test_key": "test_value"},
    )
    test_agent.start_task(task)

    # Execute a step
    result = test_agent.execute_step()
    assert result == "Test response"
    assert test_agent.step_count == 1

    # Verify model was called
    mock_model.generate_response.assert_called_once()

    # Verify memory was updated
    mock_memory.add_message.assert_called()


def test_agent_max_steps(test_agent, mock_model):
    """Test maximum steps limit."""
    task = Task(
        objective="Test objective",
        context={"test_key": "test_value"},
    )
    test_agent.start_task(task)

    # Execute steps up to limit
    for _ in range(test_agent.max_steps):
        test_agent.execute_step()

    # Try to execute one more step
    with pytest.raises(AgentError):
        test_agent.execute_step()


def test_agent_task_completion(test_agent, mock_model, mock_memory):
    """Test task completion."""
    task = Task(
        objective="Test objective",
        context={"test_key": "test_value"},
    )
    test_agent.start_task(task)

    # Mock model to indicate task completion
    mock_model.generate_response.return_value = "Task completed successfully"

    # Execute step
    result = test_agent.execute_step()
    assert result == "Task completed successfully"
    assert test_agent.state == AgentState.IDLE
    assert test_agent.current_task is None
    assert test_agent.current_task.status == TaskStatus.COMPLETED


def test_agent_task_failure(test_agent, mock_model):
    """Test task failure handling."""
    task = Task(
        objective="Test objective",
        context={"test_key": "test_value"},
    )
    test_agent.start_task(task)

    # Mock model to simulate error
    mock_model.generate_response.side_effect = Exception("Test error")

    # Execute step
    with pytest.raises(AgentError):
        test_agent.execute_step()

    assert test_agent.state == AgentState.ERROR
    assert test_agent.current_task.status == TaskStatus.FAILED


def test_agent_memory_management(test_agent, mock_memory):
    """Test memory management."""
    task = Task(
        objective="Test objective",
        context={"test_key": "test_value"},
    )
    test_agent.start_task(task)

    # Add messages to memory
    messages = [
        Message(
            role="user",
            content="Test message 1",
            timestamp=datetime.now(),
        ),
        Message(
            role="assistant",
            content="Test message 2",
            timestamp=datetime.now(),
        ),
    ]
    mock_memory.get_recent_messages.return_value = messages

    # Verify memory access
    recent_messages = test_agent.get_recent_messages()
    assert recent_messages == messages
    mock_memory.get_recent_messages.assert_called_once()


def test_agent_state_transitions(test_agent):
    """Test agent state transitions."""
    assert test_agent.state == AgentState.IDLE

    # Start task
    task = Task(
        objective="Test objective",
        context={"test_key": "test_value"},
    )
    test_agent.start_task(task)
    assert test_agent.state == AgentState.WORKING

    # Complete task
    test_agent.current_task.status = TaskStatus.COMPLETED
    test_agent._update_state()
    assert test_agent.state == AgentState.IDLE

    # Fail task
    test_agent.start_task(task)
    test_agent.current_task.status = TaskStatus.FAILED
    test_agent._update_state()
    assert test_agent.state == AgentState.ERROR
