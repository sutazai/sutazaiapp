"""Tests for AutoGPT Agent"""

import os
import pytest
from typing import Dict, Any

from ai_agents.auto_gpt.src import AutoGPTAgent
from ai_agents.base_agent import AgentError


@pytest.fixture
def test_config() -> Dict[str, Any]:
    """Fixture providing a test configuration."""
    return {
        "max_iterations": 5,
        "verbose_mode": True,
        "log_dir": "logs/test_auto_gpt",
        "model_config": {"model_name": "gpt-4", "temperature": 0.7, "max_tokens": 4096},
        "tools": ["web_search", "code_search", "file_operation", "terminal_command"],
    }


@pytest.fixture
def agent(test_config):
    """Fixture providing an initialized AutoGPT agent."""
    agent = AutoGPTAgent(name="test_agent", config=test_config)
    try:
        yield agent
    finally:
        agent.shutdown()


def test_agent_initialization(test_config):
    """Test that agent initializes correctly with valid config."""
    agent = AutoGPTAgent(name="test_agent", config=test_config)
    assert agent.name == "test_agent"
    assert agent.config == test_config
    assert agent.tools == test_config["tools"]
    assert agent.max_iterations == test_config["max_iterations"]


def test_agent_initialization_without_config():
    """Test that agent raises error when initialized without config."""
    with pytest.raises(AgentError, match="Missing required configuration field"):
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
    with pytest.raises(AgentError, match="Invalid model configuration"):
        AutoGPTAgent(name="test_agent", config=invalid_config)


def test_agent_execute_without_objective(agent):
    """Test that agent raises error when executing task without objective."""
    with pytest.raises(AgentError, match="Task must include an objective"):
        agent.execute({})


def test_agent_execute_simple_task(agent):
    """Test that agent can execute a simple task."""
    task = {"objective": "Test task execution", "parameters": {"test_param": "test_value"}}
    result = agent.execute(task)
    assert result["status"] == "success"
    assert "iterations" in result
    assert result["iterations"] == 1  # Current implementation returns after 1 iteration


def test_agent_log_directory_creation(test_config):
    """Test that agent creates log directory if it doesn't exist."""
    test_config["log_dir"] = "logs/test_auto_gpt_new"
    agent = AutoGPTAgent(name="test_agent", config=test_config)
    assert os.path.exists(test_config["log_dir"])
    agent.shutdown()
