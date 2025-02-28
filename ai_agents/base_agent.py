from typing import Dict, List, Optional

#!/usr/bin/env python3.11
"""Base Agent Module
This module defines the BaseAgent class used as the foundation for all AI agents in \
    the application.
"""

from abc import ABC, abstractmethod
from typing import Any, dict, list, Optional
from loguru import logger
import uuid
import json
from datetime import datetime
import os


class BaseAgent(ABC):
    """
    Abstract base class for AI agents in the Sutazaiapp system.
    Provides a standardized interface for agent initialization,
    configuration, and execution.
    """

    def __init__(
        self,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None):
        """
        Initialize a base agent with optional name and configuration.

        Args:
        name: Optional name for the agent
        config: Optional configuration dictionary
        """
        self.id = str(uuid.uuid4())
        self.name = name or f"Agent_{self.id[:8]}"
        self.config = config or {}

        # Configure agent-specific logger
        log_path = os.path.join("logs", f"{self.name}.log")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        logger.add(log_path, rotation="10 MB", level="INFO")
        self.logger = logger.bind(agent_id=self.id, agent_name=self.name)

        self.logger.info("Initializing agent: %s", self.name)
        self._validate_config()

        def _validate_config(self):
            """
            Validate the agent's configuration.
            Subclasses should override this method to provide specific validation.
            """
            # Basic configuration validation can be implemented here
        pass

        @abstractmethod
        def initialize(self) -> bool:
            """
            Initialize the agent's resources and prepare for execution.

            Returns:
            bool: Whether initialization was successful
            """
        pass

        @abstractmethod
        def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
            """
            Execute a given task using the agent's capabilities.

            Args:
            task: A dictionary describing the task to be executed

            Returns:
            Dict containing task execution results
            """
        pass

        def shutdown(self):
            """
            Gracefully shut down the agent and release resources.
            """
            self.logger.info("Shutting down agent: %s", self.name)

            def __repr__(self) -> str:
            return f"{self.__class__.__name__}(id={self.id}, name={self.name})"

            def __str__(self) -> str:
            return self.name


            class AgentError(Exception):
                """
                Custom exception for agent-related errors

                Attributes:
                message (str): Error message
                task (Dict): Task that caused the error
                details (Dict): Additional error details
                """

                def __init__(
                    self,
                    message: str,
                    task: Optional[Dict[str, Any]] = None,
                    details: Optional[Dict[str, Any]] = None,
                    ):
                    self.message = message
                    self.task = task
                    self.details = details or {}
                    super().__init__(self.message)


                    class BaseAgentImplementation:
                        """Base implementation class providing common agent functionality."""

                        def __init__(
                            self,
                            agent_name: str = "base_agent",
                            log_dir: str = "logs"):
                            """
                            Initialize the base agent with required attributes

                            Args:
                            agent_name (str): Name of the agent
                            log_dir (str): Directory for storing logs
                            """
                            self.agent_name = agent_name
                            self.log_dir = log_dir
                            os.makedirs(log_dir, exist_ok=True)
                            self.performance_history: List[Dict[str, Any]] = []

                            def _log_performance(
                                self,
                                task: Dict[str, Any],
                                result: Dict[str, Any]) -> None:
                                """
                                Log agent performance for each task

                                Args:
                                task (Dict): Executed task details
                                result (Dict): Task execution results
                                """
                                performance_record: Dict[str, Any] = {
                                "task": task,
                                "result": result,
                                "success": result.get(
                                    "status",
                                    "failed") == "success",
                                "timestamp": datetime.now().isoformat(),
                                }

                                self.performance_history.append(
                                    performance_record)

                                # Optional: Persist performance history
                                self._persist_performance_history()

                                def _persist_performance_history(
                                    self,
                                    max_records: int = 100) -> None:
                                    """
                                    Persist performance history to a file, maintaining a maximum number of records

                                    Args:
                                    max_records (
                                        int): Maximum number of performance records to keep
                                    """
                                    try:
                                        history_file = os.path.join(
                                            self.log_dir,
                                            f"{self.agent_name}_performance_history.json")

                                        # Trim history if exceeding max records
                                        trimmed_history = self.performance_history[-max_records:]

                                        with open(
                                            history_file,
                                            "w",
                                            encoding="utf-8") as f:
                                        json.dump(
                                            trimmed_history,
                                            f,
                                            indent=2,
                                            default=str)

                                        except (
                                            IOError,
                                            json.JSONDecodeError) as e:
                                            logger.error(
                                                "Failed to persist performance history: %s",
                                                e)

                                            def get_performance_summary(
                                                self) -> Dict[str, Any]:
                                                """
                                                Get a summary of agent performance

                                                Returns:
                                                Dict containing performance metrics
                                                """
                                                total_tasks = len(
                                                    self.performance_history)
                                                if total_tasks == 0:
                                                return {"total_tasks": 0, "success_rate": 0.0}

                                                successful_tasks = sum(
                                                    1 for record in self.performance_history if record.get("success", False))
                                                success_rate = (
                                                    successful_tasks / total_tasks) * 100

                                            return {
                                            "total_tasks": total_tasks,
                                            "successful_tasks": successful_tasks,
                                            "success_rate": success_rate,
                                            "last_execution": (
                                                self.performance_history[-1]["timestamp"] if self.performance_history else None),
                                            }
