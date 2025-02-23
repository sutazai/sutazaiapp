#!/usr/bin/env python3
"""
Configuration module for the SutazAI backend.

This module defines a configuration class using Pydantic to ensure that
all required parameters are specified with appropriate defaults.
"""

from typing import List

from pydantic import BaseModel, Field


# pylint: disable=too-few-public-methods
class Config(BaseModel):
    """
    Main configuration class for the SutazAI backend.

    Attributes:
        host: The host address to bind to
        port: The port to listen on
        debug: Whether to enable debug mode
        trusted_hosts: List of trusted host addresses
    """

    host: str = Field(default="127.0.0.1", description="Host address to bind to")
    port: int = Field(default=8000, description="Port to listen on")
    debug: bool = Field(default=False, description="Enable debug mode")
    trusted_hosts: List[str] = Field(
        default_factory=list, description="List of trusted host addresses"
    )

    model_config = {"env_file": ".env", "extra": "allow"}


# For convenience, create a config instance
config = Config()


if __name__ == "__main__":
    print("Backend configuration:")
    print(f"  HOST: {config.host}")
    print(f"  PORT: {config.port}")
    print(f"  DEBUG: {config.debug}")
    print(f"  TRUSTED_HOSTS: {config.trusted_hosts}")
