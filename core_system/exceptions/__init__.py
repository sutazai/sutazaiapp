"""
SutazAI Exceptions Package

Provides a comprehensive set of custom exceptions
for the SutazAI system.
"""

from .system_exceptions import (
    ComponentInitializationError,
    ConfigurationError,
    ResourceAllocationError,
    SecurityViolationError,
    SutazAIBaseException,
    global_exception_handler,
)

__all__ = [
    "SutazAIBaseException",
    "ConfigurationError",
    "SecurityViolationError",
    "ResourceAllocationError",
    "ComponentInitializationError",
    "global_exception_handler",
]
