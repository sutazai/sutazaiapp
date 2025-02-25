"""
Advanced Configuration Management System

This module provides a comprehensive configuration management system
using Pydantic for type safety and validation.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from pydantic import BaseSettings, Field

logger = logging.getLogger(__name__)


class ConfigSettings(BaseSettings):
    """
    Core configuration settings with type validation.
    """

    # Server settings
    host: str = Field(default="127.0.0.1", env="SERVER_HOST")
    port: int = Field(default=8000, env="SERVER_PORT")
    debug: bool = Field(default=False, env="DEBUG")

    # Security settings
    secret_key: str = Field(default="", env="SECRET_KEY")
    api_key: Optional[str] = Field(default=None, env="API_KEY")

    # Database settings
    db_url: str = Field(default="sqlite:///./app.db", env="DATABASE_URL")

    # Logging settings
    log_level: str = Field(default="INFO", env="LOG_LEVEL")

    class Config:
        env_file = ".env"
        case_sensitive = True


def get_config_settings() -> ConfigSettings:
    """
    Get configuration settings with comprehensive error handling.

    Returns:
        ConfigSettings: Configuration settings object

    Raises:
        ValueError: If required configuration values are missing
    """
    try:
        settings = ConfigSettings()
        logger.info("Successfully loaded configuration settings")
        return settings
    except Exception as e:
        logger.error(f"Failed to load configuration settings: {e}")
        raise ValueError(f"Configuration error: {e}") from e


def save_config_settings(settings: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration settings to a file.

    Args:
        settings (Dict[str, Any]): Settings to save
        config_path (str): Path to save configuration
    """
    try:
        config_dir = os.path.dirname(config_path)
        os.makedirs(config_dir, exist_ok=True)

        with open(config_path, "w") as f:
            json.dump(settings, f, indent=2)

        logger.info(f"Successfully saved configuration to {config_path}")
    except Exception as e:
        logger.error(f"Failed to save configuration: {e}")
        raise


def load_config_file(config_path: str) -> Dict[str, Any]:
    """
    Load configuration settings from a file.

    Args:
        config_path (str): Path to configuration file

    Returns:
        Dict[str, Any]: Loaded configuration settings
    """
    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        logger.info(f"Successfully loaded configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Failed to load configuration file: {e}")
        raise


def update_config_settings(
    settings: ConfigSettings, updates: Dict[str, Any]
) -> ConfigSettings:
    """
    Update configuration settings with new values.

    Args:
        settings (ConfigSettings): Current settings
        updates (Dict[str, Any]): New values to apply

    Returns:
        ConfigSettings: Updated settings
    """
    try:
        current_dict = settings.dict()
        current_dict.update(updates)
        return ConfigSettings(**current_dict)
    except Exception as e:
        logger.error(f"Failed to update configuration: {e}")
        raise


def validate_config_settings(settings: Dict[str, Any]) -> bool:
    """
    Validate configuration settings.

    Args:
        settings (Dict[str, Any]): Settings to validate

    Returns:
        bool: Whether settings are valid
    """
    try:
        ConfigSettings(**settings)
        return True
    except Exception as e:
        logger.error(f"Invalid configuration: {e}")
        return False


# Export configuration utilities
__all__ = [
    "ConfigSettings",
    "get_config_settings",
    "save_config_settings",
    "load_config_file",
    "update_config_settings",
    "validate_config_settings",
]
