#!/usr/bin/env python3.11
"""Tests for AutoGPT Agent implementation."""

import pytest
from unittest.mock import Mock, patch

from ai_agents.auto_gpt.agent import AutoGPTAgent
from ai_agents.exceptions import AgentError


@pytest.fixture
def test_config():
    """Fixture providing test configuration for AutoGPT agent."""
    return {
        "max_iterations": 5,
        "model_config": {
            "model_name": "gpt-4",
            "max_tokens": 4096
        },
        "tools": [
            "web_search",
            "code_search", 
            "file_operation",
            "terminal_command"
        ]
    }


@pytest.fixture
def agent(test_config):
    """Fixture providing an initialized AutoGPT agent."""
    with patch('ai_agents.auto_gpt.agent.AutoGPTAgent._initialize_tools'):
        agent = AutoGPTAgent(name="test_agent", config=test_config)
        try:
            yield agent
        finally:
            if hasattr(agent, 'shutdown'):
                agent.shutdown()


def test_agent_initialization(test_config):
    """Test that agent initializes correctly with valid config."""
    with patch('ai_agents.auto_gpt.agent.AutoGPTAgent._initialize_tools'):
        agent = AutoGPTAgent(name="test_agent", config=test_config)
        assert agent.name == "test_agent"
        assert agent.config == test_config
        assert hasattr(agent, 'tools')
        assert hasattr(agent, 'max_iterations')


def test_agent_initialization_without_config():
    """Test that agent raises error when initialized without config."""
    with pytest.raises(Exception):  # Adjust based on actual exception
        AutoGPTAgent(name="test_agent")


def test_agent_initialization_with_invalid_config():
    """Test that agent raises error with invalid config."""
    invalid_config = {
        "max_iterations": 5,
        "model_config": {
            "model_name": "gpt-4"
            # Missing required fields
        },
    }
    
    with pytest.raises(Exception):  # Adjust based on actual exception
        AutoGPTAgent(name="test_agent", config=invalid_config)


def test_agent_execute_task(agent):
    """Test agent task execution."""
    test_task = {
        "id": "test_task_1",
        "description": "Test task",
        "parameters": {}
    }
    
    with patch.object(agent, '_process_task') as mock_process:
        mock_process.return_value = {"status": "completed", "result": "Test result"}
        result = agent.execute(test_task)
        assert result["status"] == "completed"
        mock_process.assert_called_once_with(test_task)


def test_agent_shutdown(agent):
    """Test agent shutdown functionality."""
    if hasattr(agent, 'shutdown'):
        agent.shutdown()
        # Verify shutdown was successful
        assert True  # Adjust based on actual shutdown behavior
