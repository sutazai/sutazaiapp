from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging

# Configure logger
logger = logging.getLogger(__name__)

# Add module-level docstring
"""Base Agent Module
This module defines the BaseAgent class used as the foundation for all AI agents in the application.
"""


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


class BaseAgent:
    def __init__(self, agent_name: str = "base_agent", log_dir: str = "logs"):
        """
        Initialize the base agent with required attributes

        Args:
            agent_name (str): Name of the agent
            log_dir (str): Directory for storing logs
        """
        self.agent_name = agent_name
        self.log_dir = log_dir
        self.performance_history: List[Dict[str, Any]] = []

    def _log_performance(self, task: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Log agent performance for each task

        Args:
            task (Dict): Executed task details
            result (Dict): Task execution results
        """
        performance_record: Dict[str, Any] = {
            "task": task,
            "result": result,
            "success": result.get("status", "failed") == "success",
            "timestamp": datetime.now(),
        }

        self.performance_history.append(performance_record)

        # Optional: Persist performance history
        self._persist_performance_history()

    def _persist_performance_history(self, max_records: int = 100) -> None:
        """
        Persist performance history to a file, maintaining a maximum number of records

        Args:
            max_records (int): Maximum number of performance records to keep
        """
        try:
            history_file = f"{self.log_dir}/{self.agent_name}_performance_history.json"

            # Trim history if exceeding max records
            trimmed_history = self.performance_history[-max_records:]

            with open(history_file, "w", encoding="utf-8") as f:
                json.dump(trimmed_history, f, indent=2, default=str)

        except (IOError, json.JSONDecodeError) as e:
            logger.error(f"Failed to persist performance history: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of agent performance

        Returns:
            Dict containing performance metrics
        """
        total_tasks = len(self.performance_history)
        if total_tasks == 0:
            return {"total_tasks": 0, "success_rate": 0.0}

        successful_tasks = sum(
            1 for record in self.performance_history if record.get("success", False)
        )
        success_rate = (successful_tasks / total_tasks) * 100

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": success_rate,
            "last_execution": (
                self.performance_history[-1]["timestamp"]
                if self.performance_history
                else None
            ),
        }
