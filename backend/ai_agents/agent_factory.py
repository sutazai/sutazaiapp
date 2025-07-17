"""
Agent Factory Module

This module provides a factory for creating and managing AI agents.
It includes improved security, error handling, and dependency management.
"""

import os
import json
import logging
import importlib
from typing import Dict, Any, Optional, List, Type
from pathlib import Path
from datetime import datetime
from pydantic import ValidationError

from .base_agent import BaseAgent, AgentError
from .agent_framework import AgentConfig

# Import other agent types if AgentFactory creates them

logger = logging.getLogger(__name__)


class AgentFactory:
    """Factory class for creating and managing AI agents."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        agents_dir: str = "/opt/sutazaiapp/ai_agents",
    ):
        """
        Initialize the agent factory.

        Args:
            config_path: Optional path to agent configuration file
            agents_dir: Directory containing agent modules
        """
        self.agents: Dict[str, Dict[str, Any]] = {}
        self.agent_classes: Dict[str, Type[BaseAgent]] = {}
        self.config_path = config_path if config_path is not None else os.getenv(
            "AGENT_CONFIG_PATH", "config/agents.json"
        )
        self.agents_dir = agents_dir
        self._load_agents()
        self._load_agent_classes()

    def _load_agents(self) -> None:
        """Load agent configurations from file."""
        if not self.config_path:
             logger.warning("No agent config path provided or found in environment.")
             return
        try:
            config_path_obj = Path(self.config_path)
            if not config_path_obj.exists():
                logger.warning(f"Agent config file not found at {self.config_path}")
                return

            with open(config_path_obj, "r") as f:
                config = json.load(f)

            # Iterate over the agent configurations within the "agents" key
            for agent_id, agent_config in config.get("agents", {}).items():
                try:
                    self._validate_config_dict(agent_config)
                    self.agents[agent_id] = agent_config
                except AgentError as e:
                    logger.error(f"Invalid config for agent {agent_id}: {str(e)}")

        except Exception as e:
            logger.error(f"Error loading agent configs: {str(e)}")
            raise AgentError(f"Failed to load agent configurations: {str(e)}")

    def _load_agent_classes(self) -> None:
        """Dynamically load agent classes from the agents directory."""
        try:
            for item_name in os.listdir(self.agents_dir):
                item_path = os.path.join(self.agents_dir, item_name)
                
                # Handle subdirectories (potential agent packages)
                if os.path.isdir(item_path) and not item_name.startswith("_"):
                    try:
                        module = importlib.import_module(f"ai_agents.{item_name}")
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (
                                isinstance(attr, type)
                                and issubclass(attr, BaseAgent)
                                and attr != BaseAgent
                            ):
                                # Use directory name as type key
                                self.agent_classes[item_name] = attr 
                                logger.info(f"Registered agent class from dir: {item_name} -> {attr.__name__}")
                    except ImportError as e:
                        logger.warning(f"Could not import agent module from directory {item_name}: {e}")
                    except Exception as e:
                        logger.error(f"Error processing directory {item_name}: {e}")

                # Handle .py files directly in agents_dir (individual agent modules)
                elif item_name.endswith('.py') and not item_name.startswith('_') and item_name != '__init__.py':
                    module_name = item_name[:-3] # remove .py suffix
                    try:
                        module = importlib.import_module(f"ai_agents.{module_name}")
                        for attr_name in dir(module):
                            attr = getattr(module, attr_name)
                            if (
                                isinstance(attr, type)
                                and issubclass(attr, BaseAgent)
                                and attr != BaseAgent
                            ):
                                # Use module name (without .py) as type key
                                self.agent_classes[module_name] = attr
                                logger.info(f"Registered agent class from file: {module_name} -> {attr.__name__}")
                    except ImportError as e:
                        logger.warning(f"Could not import agent module from file {item_name}: {e}")
                    except Exception as e:
                         logger.error(f"Error processing file {item_name}: {e}")

        except Exception as e:
            logger.error(f"Error scanning agent classes directory {self.agents_dir}: {str(e)}")
            raise AgentError(f"Failed to load agent classes: {str(e)}")

    def _validate_config_dict(self, config: Dict[str, Any]) -> None:
        """
        Validate agent configuration dictionary format.

        Args:
            config: Agent configuration dictionary

        Raises:
            AgentError: If configuration is invalid
        """
        required_fields = ["type", "model_id", "capabilities"]

        for field in required_fields:
            if field not in config:
                raise AgentError(f"Missing required field: {field}")

        if not isinstance(config["capabilities"], list):
            raise AgentError("Capabilities must be a list")

        # No longer validating model as a dictionary
        # Validate that model_id is a string
        if not isinstance(config.get("model_id"), str):
            raise AgentError("model_id must be a string")

    def _validate_config_object(self, config: AgentConfig) -> None:
        """Validate the populated AgentConfig object."""
        if not config.name:
             raise AgentError("AgentConfig must have a non-empty name.")
        if not config.model_id:
             raise AgentError("AgentConfig must have a non-empty model_id.")
        if not config.capabilities:
             logger.warning(f"AgentConfig for '{config.name}' has no capabilities listed.")
        # Add more checks as needed based on AgentConfig fields

    def create_agent(self, agent_id: str, **kwargs) -> BaseAgent:
        """
        Create a new agent instance.

        Args:
            agent_id: Unique identifier for the agent
            **kwargs: Additional configuration parameters

        Returns:
            BaseAgent: New agent instance

        Raises:
            AgentError: If agent creation fails
        """
        if agent_id not in self.agents:
            raise AgentError(f"Unknown agent type: {agent_id}")

        try:
            config = self.agents[agent_id].copy()
            config.update(kwargs)

            # Add metadata
            config["metadata"] = {
                "created_at": datetime.utcnow().isoformat(),
                "agent_id": agent_id,
            }

            # Get agent class
            agent_type = config["type"]
            if agent_type not in self.agent_classes:
                raise AgentError(f"Agent class not found for type: {agent_type}")

            agent_class = self.agent_classes[agent_type]

            # Convert dictionary to AgentConfig object before instantiation
            try:
                agent_config_obj = AgentConfig.from_dict(config)
            except ValueError as e:
                 raise AgentError(f"Invalid configuration dictionary for agent {agent_id}: {e}")

            # Validate the created AgentConfig object
            self._validate_config_object(agent_config_obj)

            agent = agent_class(agent_config_obj) # Pass the object
            return agent

        except Exception as e:
            logger.error(f"Error creating agent {agent_id}: {str(e)}")
            raise AgentError(f"Failed to create agent: {str(e)}")

    def get_available_agents(self) -> List[str]:
        """
        Get list of available agent types.

        Returns:
            List[str]: List of agent type identifiers
        """
        return list(self.agents.keys())

    def get_agent_config(self, agent_id: str) -> Dict[str, Any]:
        """
        Get configuration for a specific agent type.

        Args:
            agent_id: Agent type identifier

        Returns:
            Dict[str, Any]: Agent configuration

        Raises:
            AgentError: If agent type is unknown
        """
        if agent_id not in self.agents:
            raise AgentError(f"Unknown agent type: {agent_id}")

        return self.agents[agent_id].copy()

    def register_agent(self, agent_id: str, config: Dict[str, Any]) -> None:
        """
        Register a new agent type.

        Args:
            agent_id: Unique identifier for the agent type
            config: Agent configuration dictionary

        Raises:
            AgentError: If registration fails
        """
        try:
            # Check if agent already exists
            if agent_id in self.agents:
                raise AgentError(f"Agent with ID {agent_id} already exists")

            # Validate the configuration dictionary format
            self._validate_config_dict(config)
            self.agents[agent_id] = config

            # Update config file
            self._save_config()

        except Exception as e:
            logger.error(f"Error registering agent {agent_id}: {str(e)}")
            raise AgentError(f"Failed to register agent: {str(e)}")

    def _save_config(self) -> None:
        """Save current agent configurations to file."""
        if not self.config_path:
            logger.error("Cannot save agent config: No config path specified.")
            raise AgentError("Agent configuration path is not set.")
        try:
            config_path_obj = Path(self.config_path)
            config_path_obj.parent.mkdir(parents=True, exist_ok=True)

            # Save the 'agents' dictionary, not self.agents directly if it contains more
            save_data = {"agents": self.agents} 
            with open(config_path_obj, "w") as f:
                json.dump(save_data, f, indent=2)

        except Exception as e:
            logger.error(f"Error saving agent configs: {str(e)}")
            raise AgentError(f"Failed to save agent configurations: {str(e)}")

    def validate_agent(self, agent: BaseAgent) -> bool:
        """
        Validate an existing agent instance against its configuration.
        (This might be better placed elsewhere or refactored)
        """
        if not isinstance(agent, BaseAgent):
             raise TypeError("Expected a BaseAgent instance.")

        config_dict = self.agents.get(agent.config.name) # Get original dict config by name
        if not config_dict:
             logger.warning(f"No original config found for agent {agent.config.name} during validation.")
             # Optionally perform basic validation on the agent's current config object
             try:
                 self._validate_config_object(agent.config) # Use object validation
                 return True # Assume valid if object passes basic checks
             except AgentError as e:
                 logger.error(f"Agent {agent.config.name} internal config validation failed: {e}")
                 return False

        # Validate the dictionary config (redundant if loaded correctly)
        try:
            self._validate_config_dict(config_dict)
        except AgentError as e:
            logger.error(f"Stored config dict for {agent.config.name} is invalid: {e}")
            return False

        # Validate the agent's current config object
        try:
             self._validate_config_object(agent.config) # Use object validation
        except AgentError as e:
             logger.error(f"Agent {agent.config.name} internal config validation failed: {e}")
             return False

        # Optional: Compare agent.config attributes against config_dict if needed

        return True

    def _create_agent_instance(self, agent_class: type[BaseAgent], config: AgentConfig) -> BaseAgent:
        """Create an instance of the agent."""
        try:
            # Pass AgentConfig object directly
            agent = agent_class(config=config, agent_manager=self) # Pass self as agent_manager
            logger.info(f"Successfully created agent instance: {config.agent_id}")
            return agent
        except Exception as e:
            logger.error(f"Error creating agent instance: {str(e)}")
            raise AgentError(f"Failed to create agent instance: {str(e)}")

    def load_config_from_file(self, config_path: str) -> AgentConfig:
        """Load agent configuration from a file path."""
        if not isinstance(config_path, str):
            raise AgentError(f"Invalid config format: {type(config_path)}")

        config_path_obj = Path(config_path)
        if not config_path_obj.exists():
            raise FileNotFoundError(f"Agent configuration file not found: {config_path}")

        try:
            with open(config_path_obj, 'r') as f:
                config_dict = json.load(f)

            # Validate and create AgentConfig object
            agent_config = AgentConfig(**config_dict)
            self._validate_config_object(agent_config) # Validate loaded config
            return agent_config

        except json.JSONDecodeError as e:
            raise AgentError(f"Error decoding agent configuration file: {e}") from e
        except ValidationError as e:
            raise AgentError(f"Invalid agent configuration format: {e}") from e
        except Exception as e:
            logger.error(f"Failed to load agent config from {config_path}: {e}", exc_info=True)
            raise AgentError(f"Could not load agent configuration: {e}") from e
