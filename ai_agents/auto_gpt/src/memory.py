"""
Memory management module for AutoGPT agent.

This module provides classes and utilities for managing the agent's memory,
including conversation history and task context.
"""

from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
import json
import os


@dataclass
class Message:
    """Represents a message in the conversation history."""

    role: str
    content: str
    timestamp: datetime = None

    def __post_init__(self):
        """Initialize timestamp if not provided."""
        if self.timestamp is None:
            self.timestamp = datetime.now()

    def to_dict(self) -> Dict:
        """Convert message to dictionary format."""
        return {"role": self.role, "content": self.content, "timestamp": self.timestamp.isoformat()}

    @classmethod
    def from_dict(cls, data: Dict) -> "Message":
        """Create message from dictionary format."""
        return cls(role=data["role"], content=data["content"], timestamp=datetime.fromisoformat(data["timestamp"]))


class Memory:
    """Manages the agent's memory including conversation history and context."""

    def __init__(self, max_messages: int = 10, persist_path: Optional[str] = None):
        """
        Initialize memory manager.

        Args:
            max_messages: Maximum number of messages to keep in memory
            persist_path: Path to persist memory to disk (optional)
        """
        self.max_messages = max_messages
        self.persist_path = persist_path
        self.messages: List[Message] = []
        self.context: Dict = {}

        if persist_path and os.path.exists(persist_path):
            self.load()

    def add_message(self, role: str, content: str) -> None:
        """
        Add a new message to the conversation history.

        Args:
            role: Role of the message sender (e.g., "user", "assistant")
            content: Content of the message
        """
        message = Message(role=role, content=content)
        self.messages.append(message)

        # Maintain maximum message limit
        if len(self.messages) > self.max_messages:
            self.messages = self.messages[-self.max_messages :]

        if self.persist_path:
            self.save()

    def get_messages(self) -> List[Message]:
        """Get all messages in the conversation history."""
        return self.messages

    def clear_messages(self) -> None:
        """Clear all messages from the conversation history."""
        self.messages = []
        if self.persist_path:
            self.save()

    def update_context(self, key: str, value: any) -> None:
        """
        Update a value in the context dictionary.

        Args:
            key: Context key to update
            value: New value for the key
        """
        self.context[key] = value
        if self.persist_path:
            self.save()

    def get_context(self, key: str) -> Optional[any]:
        """
        Get a value from the context dictionary.

        Args:
            key: Context key to retrieve

        Returns:
            Value associated with the key, or None if not found
        """
        return self.context.get(key)

    def clear_context(self) -> None:
        """Clear all values from the context dictionary."""
        self.context = {}
        if self.persist_path:
            self.save()

    def save(self) -> None:
        """Save memory state to disk if persist_path is set."""
        if not self.persist_path:
            return

        data = {"messages": [msg.to_dict() for msg in self.messages], "context": self.context}

        os.makedirs(os.path.dirname(self.persist_path), exist_ok=True)
        with open(self.persist_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self) -> None:
        """Load memory state from disk if persist_path is set."""
        if not self.persist_path or not os.path.exists(self.persist_path):
            return

        with open(self.persist_path, "r") as f:
            data = json.load(f)

        self.messages = [Message.from_dict(msg) for msg in data["messages"]]
        self.context = data["context"]
