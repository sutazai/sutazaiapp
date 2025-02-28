"""
SutazAI Agents Package

This package provides a framework for creating and managing AI agents
that can autonomously execute tasks using language models and various tools.
"""

from ai_agents.base_agent import BaseAgent, AgentError, BaseAgentImplementation
from ai_agents.agent_factory import AgentFactory
from ai_agents.agent_config_manager import AgentConfigManager

__version__ = "0.1.0"
__all__ = [
    "BaseAgent",
    "AgentError",
    "BaseAgentImplementation",
    "AgentFactory",
    "AgentConfigManager",
]
