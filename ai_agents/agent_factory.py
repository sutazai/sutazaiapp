#!/usr/bin/env python3.11
"""Agent Factory Module

This module provides the AgentFactory class for creating and managing AI agents.
"""

import importlib
import os
from typing import Any, Dict, Optional, Type

from loguru import logger

from ai_agents.base_agent import BaseAgent
from typing import Union
from typing import Optional


class AgentFactory:
    """Factory class for creating and managing AI agents.
    
    This class handles:
    - Agent registration
    - Agent instantiation
    - Agent configuration management
    """
    
    def __init__(self, agents_dir: str = "/opt/sutazaiapp/ai_agents"):
        """Initialize the agent factory.
        
        Args:
            agents_dir: Directory containing agent modules
        """
        self.agents_dir = agents_dir
        self._agent_registry: Dict[str, Type[BaseAgent]] = {}
        self._load_agents()
        
    def _load_agents(self) -> None:
        """Load available agent modules from the agents directory."""
        for agent_type in os.listdir(self.agents_dir):
            agent_path = os.path.join(self.agents_dir, agent_type)
            if os.path.isdir(agent_path) and not agent_type.startswith('_'):
                try:
                    module = importlib.import_module(f"ai_agents.{agent_type}")
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            isinstance(attr, type) 
                            and issubclass(attr, BaseAgent) 
                            and attr != BaseAgent
                        ):
                            self._agent_registry[agent_type] = attr
                            logger.info(f"Registered agent type: {agent_type}")
                except ImportError as e:
                    logger.error(f"Failed to load agent module {agent_type}: {e}")
                    
    def create_agent(
        self,
        agent_type: str,
        name: str | None = None,
        config: Optional[Dict[str, Any]] = None,
    ) -> BaseAgent:
        """Create a new agent instance.
        
        Args:
            agent_type: Type of agent to create
            name: Optional name for the agent
            config: Optional configuration for the agent
            
        Returns:
            An instance of the requested agent type
            
        Raises:
            ValueError: If the agent type is not registered
        """
        if agent_type not in self._agent_registry:
            raise ValueError(f"Unknown agent type: {agent_type}")
            
        agent_class = self._agent_registry[agent_type]
        agent = agent_class(name=name, config=config)
        logger.info(f"Created agent: {agent}")
        return agent
        
    def get_available_agents(self) -> Dict[str, str]:
        """Get a mapping of available agent types to their class names.
        
        Returns:
            Dict mapping agent types to their class names
        """
        return {
            agent_type: agent_class.__name__
            for agent_type, agent_class in self._agent_registry.items()
        }
        
    @staticmethod
    def validate_config(config: Dict[str, Any]) -> bool:
        """Validate an agent configuration.
        
        Args:
            config: Configuration dictionary to validate
            
        Returns:
            bool: Whether the configuration is valid
        """
        # Basic validation - can be extended based on requirements
        required_fields = ["model", "parameters"]
        return all(field in config for field in required_fields)


def main():
    """Demonstration of Agent Factory capabilities."""
    factory = AgentFactory()
    available_agents = factory.get_available_agents()
    logger.info(f"Available agents: {available_agents}")
    
    # Example agent creation
    try:
        config = {
            "model": "gpt-4",
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 2000,
            }
        }
        agent = factory.create_agent("document_processor", config=config)
        logger.info(f"Created agent: {agent}")
    except ValueError as e:
        logger.error(f"Failed to create agent: {e}")


if __name__ == "__main__":
    main()