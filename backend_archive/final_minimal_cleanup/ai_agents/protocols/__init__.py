"""
Protocols Package

This package provides functionality for inter-agent communication and task processing.
"""

from .message_protocol import Message, MessageType, MessageProtocol
from .agent_communication import AgentCommunication
from .parallel_processing import (
    TaskDistributionStrategy,
    SubTask,
    ParallelTask,
    ParallelTaskProcessor,
)

__all__ = [
    "Message",
    "MessageType",
    "MessageProtocol",
    "AgentCommunication",
    "TaskDistributionStrategy",
    "SubTask",
    "ParallelTask",
    "ParallelTaskProcessor",
]
