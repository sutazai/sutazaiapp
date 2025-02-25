"""
Routers Package Initialization

Provides centralized router management for the SutazAI backend.
"""

from .core import core_router
from .health import health_router

__all__ = ["core_router", "health_router"]
