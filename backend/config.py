#!/usr/bin/env python3.11
"""
Configuration module for the SutazAI backend.

This module defines a configuration class using Pydantic to ensure that
all required parameters are specified with appropriate defaults.
"""

import os
from typing import Any, Dict, Generic, List, Optional, TypeVar

from pydantic import BaseModel as PydanticBaseModel
from pydantic import Field, IPvAnyAddress, ValidationError, validator
from pydantic.generics import GenericModel


class BaseModel(PydanticBaseModel):    """Base model class with common configuration."""
    class Config:        """Pydantic configuration settings."""
        arbitrary_types_allowed = True
        extra = "allow"

        class Config(BaseModel):            """
            Main configuration class for the SutazAI backend.

            Attributes:            host: The host address to bind to
            port: The port to listen on
            debug: Whether to enable debug mode
            trusted_hosts: List of trusted host addresses
            """

            host: IPvAnyAddress = Field(
                default="127.0.0.1",
                description="Host address to bind to")
            port: int = Field(default=8000, ge=1024, le=65535,
                            description="Port to listen on")
            debug: bool = Field(default=False, description="Enable debug mode")
            trusted_hosts: List[str] = Field(
                default_factory=list,
                description="List of trusted host addresses")

            @validator("host", pre=True)
            @classmethod
            def validate_host(cls, v: Any) -> str:                """
                Validate and convert host to a valid IP address or hostname.

                Args:                v: Host value to validate

                Returns:                Validated host string
                """
                try:                return str(IPvAnyAddress.validate(v))
                except ValidationError:                return str(v)

                def is_debug_enabled(self) -> bool:                    """Check if debug mode is enabled."""
                return self.debug

                def is_host_trusted(self, host: str) -> bool:                    """
                    Check if a host is in the trusted hosts list.

                    Args:                    host: Host to check for trust

                    Returns:                    Whether the host is trusted
                    """
                return host in self.trusted_hosts or not self.trusted_hosts

                def load_config_from_env() -> Config:                    """
                    Load configuration from environment variables.

                    Returns:                    Config: Configuration object initialized from environment variables
                    """
                    env_vars: Dict[str, str] = {}

                    # Load from .env file if it exists
                    env_file_path = os.path.join(
                        os.path.dirname(__file__), "..", ".env")
                    if os.path.exists(env_file_path):                        with open(env_file_path, encoding="utf-8") as env_file:                            for line in env_file:                                line = line.strip()
                                if line and not line.startswith("#"):                                    key, value = line.split("=", 1)
                                    env_vars[key.strip()] = value.strip()

                                    # Override with actual environment
                                    # variables
                                    env_vars.update(os.environ)

                                    # Extract relevant config items
                                    config_data: Dict[str, Any] = {}

                                    # Host configuration
                                    config_data["host"] = env_vars.get(
                                        "HOST", "127.0.0.1")

                                    # Port configuration
                                    try:                                        config_data["port"] = int(
                                            env_vars.get("PORT", 8000))
                                        except ValueError:                                            config_data["port"] = 8000

                                            # Debug configuration
                                            config_data["debug"] = env_vars.get(
                                                "DEBUG", "false").lower() in (
                                                "true", "1", "yes")

                                            # Trusted hosts configuration
                                            trusted_hosts_str = env_vars.get(
                                                "TRUSTED_HOSTS", "")
                                            config_data["trusted_hosts"] = [
                                                h.strip() for h in trusted_hosts_str.split(",")
                                                if h.strip()
                                            ] if trusted_hosts_str else []

                                            return Config(**config_data)

                                            # Create a config instance
                                            config = load_config_from_env()

                                            if __name__ == "__main__":                                                print("Backend configuration:")
                                                print(f"  HOST: {config.host}")
                                                print(f"  PORT: {config.port}")
                                                print(
                                                    f"  DEBUG: {config.debug}")
                                                print(
                                                    f"  TRUSTED_HOSTS: {config.trusted_hosts}")
