#!/usr/bin/env python3.11
"""
Centralized Agent Creation and Management System
"""

import importlib
import os
from typing import Any, Dict, Optional, Type

from loguru import logger

from ai_agents.agent_config_manager import AgentConfigManager
from ai_agents.base_agent import BaseAgent


class AgentFactory:
    """
    A factory class for creating and managing AI agents dynamically.
    """

    def __init__(
        self,
        agents_dir: str = "/opt/sutazaiapp/ai_agents",
        config_manager: Optional[AgentConfigManager] = None,
    ):
        """
        Initialize Agent Factory

        Args:
            agents_dir (str): Base directory for AI agents
            config_manager (AgentConfigManager, optional): Configuration management system
        """
        self.agents_dir = agents_dir
        self.config_manager = config_manager or AgentConfigManager()

        # Agent registry
        self._agent_registry: Dict[str, Type[BaseAgent]] = {}

        # Logging configuration
        logger.add(
            os.path.join(agents_dir, "agent_factory.log"),
            rotation="10 MB",
            level="INFO",
        )

        # Discover and register agents
        self._discover_agents()

    def _discover_agents(self):
        """
        Discover and register available agent classes by scanning the agents directory.
        """
        logger.info("🔍 Discovering AI Agents")

        for agent_type in os.listdir(self.agents_dir):
            agent_path = os.path.join(self.agents_dir, agent_type)

            # Skip if not a directory or if it is the base_agent which is abstract
            if not os.path.isdir(agent_path) or agent_type == "base_agent":
                continue

            try:
                module = importlib.import_module(f"ai_agents.{agent_type}.src")
                agent_class = getattr(module, f"{agent_type.capitalize()}Agent", None)

                if agent_class and issubclass(agent_class, BaseAgent):
                    self._agent_registry[agent_type] = agent_class
                    logger.info("✅ Registered Agent: %s", agent_type)
            except Exception as e:
                logger.exception(f"❌ Failed to register agent {agent_type}: {e}")

    def create_agent(
        self,
        agent_type: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        """
        Create an agent instance by type

        Args:
            agent_type (str): Type of agent to create
            config (Optional[Dict]): Agent configuration

        Returns:
            BaseAgent: Instantiated agent

        Raises:
            ValueError: If agent type is not registered
        """
        if agent_type not in self._agent_registry:
            raise ValueError(
                f"Unknown agent type: {agent_type}. Available types: {', '.join(self._agent_registry.keys())}"
            )

        # Load configuration if not provided
        if config is None:
            try:
                config = self.config_manager.load_config(agent_type)
            except Exception as e:
                logger.warning(f"Using default configuration for {agent_type}: {e}")
                config = {}

        agent_class = self._agent_registry[agent_type]

        # Extract BaseAgent parameters from config
        agent_name = agent_type
        log_dir = config.pop("log_dir", "logs") if config else "logs"

        # Create the agent instance with proper parameters
        agent_instance = agent_class(
            agent_name=agent_name,
            log_dir=log_dir,
            **(config or {})
        )

        return agent_instance

    def list_available_agents(self) -> Dict[str, str]:
        """
        List all registered agents

        Returns:
            Dict: Mapping of agent types to their class names
        """
        return {
            agent_type: agent_class.__name__
            for agent_type, agent_class in self._agent_registry.items()
        }

    @staticmethod
    def main():
        """Demonstration of Agent Factory capabilities"""
        factory = AgentFactory()

        # List available agents
        print("Available Agents:", factory.list_available_agents())

        # Create an agent
        try:
            auto_gpt_agent = factory.create_agent("auto_gpt")
            print(f"Agent Performance: {auto_gpt_agent.get_performance_summary()}")
        except Exception as e:
            logger.exception(f"Agent creation failed: {e}")


if __name__ == "__main__":
    AgentFactory.main()
