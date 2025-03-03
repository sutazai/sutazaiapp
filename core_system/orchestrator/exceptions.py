#!/usr/bin/env python3.11
"""
Exceptions for Supreme AI Orchestrator

This module defines custom exceptions used throughout the orchestrator system.
"""

class OrchestratorError(Exception):
    """Base exception for orchestrator errors"""
    pass


class AgentError(OrchestratorError):
    """Exception raised for errors in agent operations"""
    pass


class SyncError(OrchestratorError):
    """Exception raised for synchronization errors"""
    pass


class QueueError(OrchestratorError):
    """Exception raised for task queue errors"""
    pass


class TaskError(OrchestratorError):
    """Exception raised for task-related errors"""
    pass


class ConfigError(OrchestratorError):
    """Exception raised for configuration errors"""
    pass


class AuthenticationError(OrchestratorError):
    """Exception raised for authentication errors"""
    pass


class CommunicationError(OrchestratorError):
    """Exception raised for inter-server communication errors"""
    pass


class ResourceError(OrchestratorError):
    """Exception raised for resource allocation errors"""
    pass


class ValidationError(OrchestratorError):
    """Exception raised for data validation errors"""
    pass 