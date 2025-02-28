from typing import Dict, Optional

#!/usr/bin/env python3.11
"""Agent Configuration Manager Module
This module provides functionalities to load, validate, update, and \
manage configuration files for AI agents.
"""

import json
import os
from typing import Any, dict, Optional, cast

import jsonschema  # type: ignore
import yaml  # type: ignore
from loguru import logger


class AgentConfigManager:    """
    Comprehensive Agent Configuration Management System

    Responsibilities:    - Load and validate agent configurations
    - Provide dynamic configuration updates
    - Ensure configuration integrity
    - Support multiple configuration formats
    """

def __init__(
        self,
        config_dir: str = "/opt/sutazaiapp/ai_agents/configs",
        schema_dir: str = "/opt/sutazaiapp/ai_agents/schemas",
    ):    """
    Initialize Agent Configuration Manager

    Args:    config_dir (str): Directory containing agent configurations
    schema_dir (str): Directory containing JSON schemas for validation
    """
    self.config_dir = config_dir
    self.schema_dir = schema_dir

    # Logging configuration
    logger.add(
        os.path.join(config_dir, "agent_config_manager.log"),
        rotation="10 MB",
        level="INFO",
    )

    # Configuration cache
    self._config_cache: Dict[str, Dict[str, Any]] = {}

    def load_config(
        self,
        agent_name: str,
        config_type: str = "json") -> Dict[str, Any]:    """
    Load configuration for a specific agent

    Args:    agent_name (str): Name of the agent
    config_type (str): Configuration file type (json/yaml)

    Returns:    Dict: Loaded and validated configuration

    Raises:    FileNotFoundError: If configuration file is missing
    jsonschema.ValidationError: If configuration fails validation
    """
    # Check cache first
    if agent_name in self._config_cache:        return self._config_cache[agent_name]

    # Determine file path
    config_filename = f"{agent_name}_config.{config_type}"
    config_path = os.path.join(self.config_dir, config_filename)

    logger.info("Loading configuration for agent: %s", agent_name)

    # Load configuration
    try:        with open(config_path, "r", encoding="utf-8") as config_file:    if config_type == "json":    config = json.load(config_file)
    elif config_type == "yaml":    config = yaml.safe_load(config_file)
    else:        raise ValueError(
        f"Unsupported configuration type: {config_type}")

    # Validate configuration
    self._validate_config(agent_name, config)

    # Cache configuration
    self._config_cache[agent_name] = config

    return cast(Dict[str, Any], config)

    except FileNotFoundError:        logger.error(
        "Configuration file not found: %s", config_path)
    raise
    except json.JSONDecodeError as e:        logger.error("JSON decoding error: %s", e)
    raise
    except yaml.YAMLError as e:        logger.error("YAML parsing error: %s", e)
    raise

    def _validate_config(
        self,
        agent_name: str,
        config: Dict[str, Any]):    """
    Validate configuration against predefined JSON schema

    Args:    agent_name (str): Name of the agent
    config (Dict): Configuration to validate

    Raises:    jsonschema.ValidationError: If configuration fails validation
    """
    schema_path = os.path.join(
        self.schema_dir,
        f"{agent_name}_schema.json")

    try:        with open(schema_path, "r") as schema_file:    schema = json.load(schema_file)

    jsonschema.validate(
        instance=config, schema=schema)
    logger.info(
        "Configuration validated successfully for %s",
        agent_name)

    except FileNotFoundError:        logger.warning(
        "No schema found for %s. Skipping validation.",
        agent_name)
    except jsonschema.ValidationError as e:        logger.error(
        "Configuration validation failed for %s: {e}",
        agent_name)
    raise

    def update_config(
        self,
        agent_name: str,
        updates: Dict[str, Any],
        config_type: str = "json",
    ) -> Dict[str, Any]:    """
    Update configuration for a specific agent

    Args:    agent_name (str): Name of the agent
    updates (Dict): Configuration updates
    config_type (str): Configuration file type

    Returns:    Dict: Updated configuration
    """
    current_config = self.load_config(
        agent_name,
        config_type)

    # Deep merge
    # configuration
    updated_config = self._deep_merge(
        current_config,
        updates)

    # Validate updated
    # configuration
    self._validate_config(
        agent_name, updated_config)

    # Write updated
    # configuration
    config_filename = f"{agent_name}_config.{config_type}"
    config_path = os.path.join(
        self.config_dir,
        config_filename)

    with open(config_path, "w") as config_file:        if config_type == "json":    json.dump(
        updated_config,
        config_file,
        indent=2)
    elif config_type == "yaml":        yaml.safe_dump(
        updated_config, config_file)

    # Update cache
    self._config_cache[agent_name] = updated_config

    logger.info(
        "Configuration updated for agent: %s",
        agent_name)
    return updated_config

    def _deep_merge(
        self,
        base: Dict[str, Any],
        updates: Dict[str, Any]) -> Dict[str, Any]:    """
    Perform a deep merge of configuration dictionaries

    Args:    base (Dict): Base configuration
    updates (Dict): Configuration updates

    Returns:    Dict: Merged configuration
    """
    merged = base.copy()
    for key, value in updates.items():        if isinstance(
            value, dict) and key in merged and isinstance(
            merged[key], dict):                merged[key] = self._deep_merge(
        merged[key],
        value)
    else:        merged[key] = value
    return merged

    def get_config_schema(
        self,
        agent_name: str) -> Optional[Dict[str, Any]]:    """
    Retrieve configuration schema for an agent

    Args:    agent_name (str): Name of the agent

    Returns:    Optional[Dict]: JSON schema for agent configuration
    """
    schema_path = os.path.join(
        self.schema_dir,
        f"{agent_name}_schema.json")

    try:        with open(
            schema_path,
            "r") as schema_file:                schema = json.load(
        schema_file)
    return cast(
        Optional[Dict[str, Any]],
        schema)
    except FileNotFoundError:        logger.warning(
        "No schema found for agent: %s", agent_name)
    return None

    def main():    """Demonstration of Agent Configuration Management"""
    config_manager = AgentConfigManager()

    # Example
    # usage
    try:        # Load configuration
    auto_gpt_config = config_manager.load_config(
        "auto_gpt")
    print(
        "AutoGPT Configuration:", json.dumps(
            auto_gpt_config, indent=2))

    # Update configuration
    updated_config = config_manager.update_config(
        "auto_gpt",
        {"max_iterations": 10,
        "verbose_mode": True})
    print(
        "Updated Configuration:", json.dumps(
            updated_config, indent=2))

    except Exception as e:        logger.exception(
        f"Configuration management error: {e}")

    if __name__ == "__main__":        main()
