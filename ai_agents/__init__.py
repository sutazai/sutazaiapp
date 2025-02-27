"""
Sutazaiapp AI Agents Module

This module provides a comprehensive framework for AI agent management,
including base classes, interfaces, and utility functions for agent orchestration.
"""

from .base_agent import BaseAgent, AgentError

__all__ = [
    'BaseAgent',
    'AgentError'
]

# Version of the AI Agents module
__version__ = '0.1.0'
