import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict

from loguru import logger


class BaseAgent(ABC):
    """
    Abstract Base Agent: Standardized Interface for AI Agents

    Provides a consistent framework for:
    - Agent initialization
    - Task execution
    - Performance tracking
    - Error handling
    """

    def __init__(
        self,
        agent_name: str = None,
        log_dir: str = "/opt/sutazai_project/SutazAI/logs",
    ):
        """
        Initialize base agent with unique identifier and logging

        Args:
            agent_name (str): Unique name for the agent
            log_dir (str): Directory for logging agent activities
        """
        self.agent_id = str(uuid.uuid4())
        self.agent_name = agent_name or self.__class__.__name__
        self.log_dir = log_dir

        # Comprehensive logging setup
        logger.add(
            f"{log_dir}/{self.agent_name}_agent.log",
            rotation="10 MB",
            level="INFO",
        )

        # Performance tracking
        self.performance_history = []

        logger.info(
            f"🤖 Agent {self.agent_name} (ID: {self.agent_id}) initialized"
        )

    @abstractmethod
    def execute(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """
        Abstract method for task execution

        Args:
            task (Dict): Task specification with required parameters

        Returns:
            Dict: Task execution results

        Raises:
            NotImplementedError: If not implemented by subclass
        """
        raise NotImplementedError("Subclasses must implement execute method")

    def _log_performance(self, task: Dict[str, Any], result: Dict[str, Any]):
        """
        Log agent performance for each task

        Args:
            task (Dict): Executed task details
            result (Dict): Task execution results
        """
        performance_record = {
            "task": task,
            "result": result,
            "success": result.get("status", "failed") == "success",
            "timestamp": datetime.now(),
        }

        self.performance_history.append(performance_record)

        # Optional: Persist performance history
        self._persist_performance_history()

    def _persist_performance_history(self, max_records: int = 100):
        """
        Persist performance history to a file, maintaining a maximum number of records

        Args:
            max_records (int): Maximum number of performance records to keep
        """
        try:
            history_file = (
                f"{self.log_dir}/{self.agent_name}_performance_history.json"
            )

            # Trim history if exceeding max records
            trimmed_history = self.performance_history[-max_records:]

            with open(history_file, "w") as f:
                json.dump(trimmed_history, f, indent=2, default=str)

        except Exception as e:
            logger.error(f"Failed to persist performance history: {e}")

    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Generate a summary of agent performance

        Returns:
            Dict: Performance metrics and summary
        """
        if not self.performance_history:
            return {"total_tasks": 0, "success_rate": 0}

        total_tasks = len(self.performance_history)
        successful_tasks = sum(
            1 for record in self.performance_history if record["success"]
        )

        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks,
        }


class AgentError(Exception):
    """
    Custom exception for agent-related errors

    Provides detailed error information for agent execution
    """

    def __init__(
        self, message: str, agent_id: str = None, task: Dict[str, Any] = None
    ):
        """
        Initialize agent-specific error

        Args:
            message (str): Error description
            agent_id (str, optional): ID of the agent that encountered the error
            task (Dict, optional): Task details when error occurred
        """
        self.agent_id = agent_id
        self.task = task
        super().__init__(message)

    def __str__(self):
        """
        Provide a comprehensive error representation

        Returns:
            str: Detailed error description
        """
        error_details = f"Agent Error: {super().__str__()}"
        if self.agent_id:
            error_details += f"\nAgent ID: {self.agent_id}"
        if self.task:
            error_details += f"\nTask: {self.task}"
        return error_details
