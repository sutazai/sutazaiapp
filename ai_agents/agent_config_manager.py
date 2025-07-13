#!/usr/bin/env python3.11
"""Agent Configuration Manager Module

This module provides the AgentConfigManager class for managing agent configurations.
"""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

from loguru import logger


class AgentConfigManager:
    """Manages configuration for AI agents.

    This class handles:
    - Loading agent configurations
    - Validating configurations
    - Persisting configuration changes
    """

    def __init__(self, config_dir: str = "/opt/sutazaiapp/config/agents"):
        """Initialize the configuration manager.

        Args:
            config_dir: Directory containing agent configurations
        """
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Set up logging
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        logger.add(
            log_dir / "agent_config.log",
            rotation="10 MB",
            level="INFO"
        )

    def load_config(self, agent_type: str) -> Dict[str, Any]:
        """Load configuration for a specific agent type.

        Args:
            agent_type: Type of agent to load configuration for

        Returns:
            Dict containing agent configuration

        Raises:
            FileNotFoundError: If configuration file doesn't exist
        """
        config_file = self.config_dir / f"{agent_type}.json"
        if not config_file.exists():
            raise FileNotFoundError(f"No configuration found for agent type: {agent_type}")

        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration for {agent_type}")
            return config
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in configuration file for {agent_type}: {e}")
            raise

    def save_config(self, agent_type: str, config: Dict[str, Any]) -> None:
        """Save configuration for a specific agent type.

        Args:
            agent_type: Type of agent to save configuration for
            config: Configuration to save
        """
        config_file = self.config_dir / f"{agent_type}.json"
        try:
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=4)
            logger.info(f"Saved configuration for {agent_type}")
        except Exception as e:
            logger.error(f"Failed to save configuration for {agent_type}: {e}")
            raise

    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate an agent configuration.

        Args:
            config: Configuration to validate

        Returns:
            bool: Whether the configuration is valid
        """
        required_fields = [
            "model",
            "parameters",
            "max_retries",
            "timeout"
        ]

        return all(field in config for field in required_fields)

    def get_default_config(self, agent_type: str) -> Dict[str, Any]:
        """Get default configuration for an agent type.

        Args:
            agent_type: Type of agent to get default configuration for

        Returns:
            Dict containing default configuration
        """
        return {
            "model": "gpt-4",
            "parameters": {
                "temperature": 0.7,
                "max_tokens": 2000,
                "top_p": 1.0
            },
            "max_retries": 3,
            "timeout": 30
        }

    def list_configurations(self) -> Dict[str, Dict[str, Any]]:
        """List all available agent configurations.

        Returns:
            Dict mapping agent types to their configurations
        """
        configs = {}
        for config_file in self.config_dir.glob("*.json"):
            agent_type = config_file.stem
            try:
                configs[agent_type] = self.load_config(agent_type)
            except Exception as e:
                logger.error(f"Failed to load configuration for {agent_type}: {e}")

        return configs


def main():
    """Demonstration of configuration management."""
    config_manager = AgentConfigManager()

    # Example configuration
    test_config = {
        "model": "gpt-4",
        "parameters": {
            "temperature": 0.7,
            "max_tokens": 2000,
            "top_p": 1.0
        },
        "max_retries": 3,
        "timeout": 30
    }

    try:
        # Save test configuration
        config_manager.save_config("test_agent", test_config)

        # Load and validate configuration
        loaded_config = config_manager.load_config("test_agent")
        is_valid = config_manager.validate_config(loaded_config)

        logger.info(f"Loaded configuration is valid: {is_valid}")
        logger.info(f"Available configurations: {config_manager.list_configurations()}")

    except Exception as e:
        logger.error(f"Configuration management failed: {e}")


if __name__ == "__main__":
    main()
