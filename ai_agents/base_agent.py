#!/usr/bin/env python3
"""
Base Agent Module

This module provides the base agent class that all specific agent types inherit from.
It includes core functionality for initialization, execution, and cleanup.
"""

import logging
from typing import Dict, Any, List, Optional
from abc import ABC, abstractmethod
from datetime import datetime
import threading
from enum import Enum

# Import AgentStatus from its new location
# from .agent_manager import AgentStatus
from .agent_status import AgentStatus

# Import AgentConfig
from ai_agents.agent_framework import AgentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("logs/base_agent.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class AgentError(Exception):
    """Base exception for agent-related errors."""

    pass


class BaseAgent(ABC):
    """Base class for all AI agents."""

    def __init__(self, config: AgentConfig):
        """
        Initialize the base agent.

        Args:
            config: Agent configuration object (AgentConfig dataclass)
        """
        self.config = config
        # Access attributes directly from the AgentConfig object
        self.model_id = config.model_id  # Store the model ID
        self.capabilities = [cap.value for cap in config.capabilities] # Store as strings
        self.metadata = {"name": config.name, "description": config.description}
        self.status = AgentStatus.INITIALIZING
        self._is_initialized = False
        self._is_running = False
        self._stop_event = threading.Event()
        self.execution_log: List[Dict[str, Any]] = []
        self.error_log: List[Dict[str, Any]] = []
        self.history: List[Dict[str, Any]] = []
        self._last_heartbeat: Optional[float] = None  # Initialize to None

    def initialize(self) -> None:
        """
        Initialize the agent.

        This method should be implemented by subclasses to perform any necessary
        setup, such as loading models, connecting to services, etc.

        Raises:
            AgentError: If initialization fails
        """
        if self._is_initialized:
            raise AgentError("Agent already initialized")

        try:
            self._initialize()
            self._is_initialized = True
            logger.info(f"Agent {self.metadata.get('agent_id', 'unknown')} initialized")
        except Exception as e:
            logger.error(f"Error initializing agent: {str(e)}")
            raise AgentError(f"Failed to initialize agent: {str(e)}")

    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute a task.

        Args:
            task: Task configuration dictionary

        Returns:
            Dict[str, Any]: Task results

        Raises:
            AgentError: If execution fails
        """
        if not self._is_initialized:
            raise AgentError("Agent not initialized")

        if self._is_running:
            raise AgentError("Agent already running")

        try:
            self._is_running = True
            start_time = datetime.utcnow()

            # Validate task
            self._validate_task(task)

            # Execute task
            result = self._execute(task)

            # Update metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(execution_time)

            return result

        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            raise AgentError(f"Failed to execute task: {str(e)}")
        finally:
            self._is_running = False

    def cleanup(self) -> None:
        """
        Clean up agent resources.

        This method should be implemented by subclasses to perform any necessary
        cleanup, such as closing connections, saving state, etc.

        Raises:
            AgentError: If cleanup fails
        """
        if not self._is_initialized:
            return

        try:
            self._cleanup()
            self._is_initialized = False
            logger.info(f"Agent {self.metadata.get('agent_id', 'unknown')} cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up agent: {str(e)}")
            raise AgentError(f"Failed to clean up agent: {str(e)}")

    def get_capabilities(self) -> List[str]:
        """
        Get list of agent capabilities.

        Returns:
            List[str]: List of capability identifiers
        """
        return self.capabilities.copy()

    def has_capability(self, capability: str) -> bool:
        """
        Check if agent has a specific capability.

        Args:
            capability: Capability identifier

        Returns:
            bool: True if agent has the capability
        """
        return capability in self.capabilities

    def get_metadata(self) -> Dict[str, Any]:
        """
        Get agent metadata.

        Returns:
            Dict[str, Any]: Agent metadata dictionary
        """
        return self.metadata.copy()

    def get_model_info(self) -> Dict[str, Any]:
        """
        Get model information.

        Returns:
            Dict[str, Any]: Model information dictionary
        """
        return {"model_id": self.model_id}

    def is_initialized(self) -> bool:
        """
        Check if agent is initialized.

        Returns:
            bool: True if agent is initialized
        """
        return self._is_initialized

    def is_running(self) -> bool:
        """
        Check if agent is running.

        Returns:
            bool: True if agent is running
        """
        return self._is_running

    def _validate_task(self, task: Dict[str, Any]) -> None:
        """
        Validate task configuration.

        Args:
            task: Task configuration dictionary

        Raises:
            AgentError: If task configuration is invalid
        """
        required_fields = ["type", "parameters"]

        for field in required_fields:
            if field not in task:
                raise AgentError(f"Missing required task field: {field}")

        if task["type"] not in self.capabilities:
            raise AgentError(f"Agent does not support task type: {task['type']}")

    def _update_metrics(self, execution_time: float) -> None:
        """
        Update agent metrics.

        Args:
            execution_time: Task execution time in seconds
        """
        # This method can be overridden by subclasses to track custom metrics
        pass

    @abstractmethod
    def _initialize(self) -> None:
        """Internal initialization method to be implemented by subclasses."""
        pass

    @abstractmethod
    def _execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Internal execution method to be implemented by subclasses."""
        pass

    @abstractmethod
    def _cleanup(self) -> None:
        """Internal cleanup method to be implemented by subclasses."""
        pass


class BaseAgentImplementation(BaseAgent):
    """Concrete implementation of BaseAgent for testing purposes."""

    def __init__(self, config: Dict[str, Any]):
        """Initialize the test agent implementation."""
        super().__init__(config)
        # Add 'test' capability to support test tasks
        if "test" not in self.capabilities:
            self.capabilities.append("test")
        self._validate_config(config)
        self.is_initialized = False
        self.history = []
        self._simulate_error = False
        self._last_heartbeat = datetime.utcnow().timestamp()

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate agent configuration."""
        required_fields = ["type", "model", "capabilities"]

        for field in required_fields:
            if field not in config:
                raise AgentError(f"Missing required configuration field: {field}")

        if "name" not in config["model"] or "version" not in config["model"]:
            raise AgentError("Model configuration must include name and version")

    def initialize(self) -> None:
        """Initialize the test agent."""
        if not self.is_initialized:
            self._initialize()
            self.is_initialized = True
            logger.info(f"Agent {self.metadata.get('agent_id', 'unknown')} initialized")

    def execute(self, task: Dict[str, Any] = None) -> Dict[str, Any]:
        """Execute a test task."""
        if not task:
            task = {"type": "test", "parameters": {}}

        if not self.is_initialized:
            raise AgentError("Agent not initialized")

        if self._is_running:
            raise AgentError("Agent already running")

        try:
            self._is_running = True
            start_time = datetime.utcnow()

            # When simulating an error, clear history first to match test expectations
            if self._simulate_error:
                self.clear_history()
                error = "Simulated execution error"
                self._log_execution(False, error=error)
                raise AgentError(error)

            # Skip validation for simulate_error case to test error handling
            self._validate_task(task)

            # Execute task
            result = self._execute(task)

            # Update metrics
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            self._update_metrics(execution_time)

            return result

        except Exception as e:
            logger.error(f"Error executing task: {str(e)}")
            if (
                not self._simulate_error
            ):  # Only log if not already logged for simulated error
                self._log_execution(False, error=str(e))
            raise AgentError(f"Failed to execute task: {str(e)}")
        finally:
            self._is_running = False

    def cleanup(self) -> None:
        """Clean up test agent resources."""
        if self.is_initialized:
            self._cleanup()
            self.is_initialized = False
            logger.info(f"Agent {self.metadata.get('agent_id', 'unknown')} cleaned up")

    def _initialize(self) -> None:
        """Internal initialization implementation."""
        # Simulation of initialization
        self.is_initialized = True
        logger.info("Test agent initialized")

    def _execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Internal execution implementation."""
        # Simulation of execution
        result = {"status": "success", "message": "Test execution completed"}
        self._log_execution(True, result)
        return result

    def _cleanup(self) -> None:
        """Internal cleanup implementation."""
        # Simulation of cleanup
        self.is_initialized = False
        logger.info("Test agent cleaned up")

    def _log_execution(
        self, success: bool, result: Dict[str, Any] = None, error: str = None
    ) -> None:
        """Log execution details."""
        execution_log = {
            "timestamp": datetime.utcnow().timestamp(),
            "success": success,
            "duration": 0.1,  # Simulated duration
            "result": result,
            "error": error,
        }
        self.history.append(execution_log)

    def get_history(self) -> List[Dict[str, Any]]:
        """Get execution history."""
        return self.history

    def clear_history(self) -> None:
        """Clear execution history."""
        self.history = []

    def update_heartbeat(self) -> None:
        """Update agent heartbeat."""
        self._last_heartbeat = datetime.utcnow().timestamp()

    def get_heartbeat(self) -> float:
        """Get last heartbeat timestamp."""
        return self._last_heartbeat
