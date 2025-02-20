#!/usr/bin/env python3
"""
Configuration module for the SutazAI backend.

This module defines a configuration class using Pydantic to ensure that
all required parameters are specified with appropriate defaults.
"""

from typing import List

from pydantic import BaseSettings


class Config(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8000
    debug: bool = False
    trusted_hosts: List[str] = []

    class Config:
        env_file = ".env"


# For convenience, create a config instance
config = Config()

if __name__ == "__main__":
    print("Backend configuration:")
    print(f"  HOST: {config.host}")
    print(f"  PORT: {config.port}")
    print(f"  DEBUG: {config.debug}")
    print(f"  TRUSTED_HOSTS: {config.trusted_hosts}")
