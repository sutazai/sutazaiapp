#!/usr/bin/env python3
"""
Configuration module for the SutazAI backend.

This module defines a configuration class using Pydantic to ensure that
all required parameters are specified with appropriate defaults.
"""

from typing import List, Dict, Any
import os

from pydantic import BaseModel, Field


class Config(BaseModel):
    """
    Main configuration class for the SutazAI backend.

    Attributes:
        host: The host address to bind to
        port: The port to listen on
        debug: Whether to enable debug mode
        trusted_hosts: List of trusted host addresses
    """

    host: str = Field(
        default="127.0.0.1", description="Host address to bind to"
    )
    port: int = Field(default=8000, description="Port to listen on")
    debug: bool = Field(default=False, description="Enable debug mode")
    trusted_hosts: List[str] = Field(
        default_factory=list, description="List of trusted host addresses"
    )

    # Model configuration that works with both Pydantic v1 and v2
    if hasattr(BaseModel, "model_config"):
        # Pydantic v2 approach
        model_config = {"extra": "allow"}
    else:
        # Pydantic v1 approach
        class Config:
            extra = "allow"


# For convenience, create a config instance with environment variables
def load_config_from_env() -> Config:
    """
    Load configuration from environment variables.

    Returns:
        Config: Configuration object initialized from environment variables
    """
    env_vars: Dict[str, str] = {}
    if os.path.exists(".env"):
        # Simple env file parsing (could be replaced with python-dotenv)
        with open(".env", "r", encoding="utf-8") as env_file:
            for line in env_file:
                line = line.strip()
                if line and not line.startswith("#"):
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()

    # Override with actual environment variables
    for key, value in os.environ.items():
        env_vars[key] = value

    # Extract relevant config items
    config_data: Dict[str, Any] = {}
    if "HOST" in env_vars:
        config_data["host"] = env_vars["HOST"]
    if "PORT" in env_vars:
        try:
            config_data["port"] = int(env_vars["PORT"])
        except ValueError:
            pass
    if "DEBUG" in env_vars:
        config_data["debug"] = env_vars["DEBUG"].lower() in (
            "true",
            "1",
            "yes",
        )
    if "TRUSTED_HOSTS" in env_vars:
        hosts_str = env_vars["TRUSTED_HOSTS"]
        config_data["trusted_hosts"] = [
            h.strip() for h in hosts_str.split(",")
        ]

    return Config(**config_data)


# Create a config instance
config = load_config_from_env()


if __name__ == "__main__":
    print("Backend configuration:")
    print(f"  HOST: {config.host}")
    print(f"  PORT: {config.port}")
    print(f"  DEBUG: {config.debug}")
    print(f"  TRUSTED_HOSTS: {config.trusted_hosts}")
