"""
SutazAI Exceptions Package

Provides a comprehensive set of custom exceptions
for the SutazAI system.
"""

from .system_exceptions import (
    ComponentInitializationError,
    ConfigurationError,
    ResourceAllocationError,
    SutazAIBaseException,
    global_exception_handler,
)

__all__ = [
    "SutazAIBaseException",
    "ConfigurationError",
    "ResourceAllocationError",
    "ComponentInitializationError",
    "global_exception_handler",
]
