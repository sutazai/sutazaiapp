"""
Message Protocol for Agent Communication

This module defines the standardized message format and protocol for communication
between agents in the SutazaiApp system.
"""

import uuid
import json
import time
from typing import Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass, field, asdict


class MessageType(Enum):
    """Type of message for agent communication."""

    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    QUERY = "query"
    RESPONSE = "response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    NOTIFICATION = "notification"
    HUMAN_INPUT_REQUEST = "human_input_request"
    HUMAN_INPUT_RESPONSE = "human_input_response"
    BROADCAST = "broadcast"


@dataclass
class Message:
    """
    Standardized message for agent communication.
    """

    message_type: MessageType
    sender_id: str
    content: Dict[str, Any]
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    recipient_id: Optional[str] = None
    correlation_id: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    ttl: Optional[int] = None  # Time to live in seconds
    priority: int = 1  # 1 (highest) to 5 (lowest)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary."""
        result = asdict(self)
        result["message_type"] = self.message_type.value
        return result

    def to_json(self) -> str:
        """Convert message to JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Create message from dictionary."""
        # Convert message_type string to enum
        if isinstance(data["message_type"], str):
            data["message_type"] = MessageType(data["message_type"])
        return cls(**data)

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """Create message from JSON string."""
        data = json.loads(json_str)
        return cls.from_dict(data)

    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        if self.ttl is None:
            return False
        return (time.time() - self.timestamp) > self.ttl


class MessageProtocol:
    """
    Protocol for handling agent messages.

    Provides methods for creating, validating, and processing messages
    between agents.
    """

    @staticmethod
    def create_message(
        message_type: MessageType,
        sender_id: str,
        content: Dict[str, Any],
        recipient_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        ttl: Optional[int] = None,
        priority: int = 1,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Create a new message for agent communication.

        Args:
            message_type: Type of message
            sender_id: ID of the sending agent
            content: Message content
            recipient_id: ID of the receiving agent (optional)
            correlation_id: ID to correlate related messages (optional)
            ttl: Time to live in seconds (optional)
            priority: Message priority (1-5, 1 is highest)
            metadata: Additional metadata for the message

        Returns:
            Message: Newly created message
        """
        return Message(
            message_type=message_type,
            sender_id=sender_id,
            content=content,
            recipient_id=recipient_id,
            correlation_id=correlation_id,
            ttl=ttl,
            priority=priority,
            metadata=metadata or {},
        )

    @staticmethod
    def validate_message(message: Message) -> bool:
        """
        Validate a message.

        Args:
            message: Message to validate

        Returns:
            bool: True if message is valid, False otherwise
        """
        # Check required fields
        if not message.sender_id or not isinstance(message.content, dict):
            return False

        # Validate TTL
        if message.ttl is not None and message.ttl <= 0:
            return False

        # Validate priority
        if not 1 <= message.priority <= 5:
            return False

        # Check if message is expired
        if message.is_expired():
            return False

        return True

    @staticmethod
    def create_response_message(
        request_message: Message,
        content: Dict[str, Any],
        sender_id: str,
        message_type: Optional[MessageType] = None,
    ) -> Message:
        """
        Create a response message for a request message.

        Args:
            request_message: Original request message
            content: Response content
            sender_id: ID of the responding agent
            message_type: Type of response message (default: auto-determine)

        Returns:
            Message: Response message
        """
        # Auto-determine response type based on request type
        if message_type is None:
            if request_message.message_type == MessageType.TASK_REQUEST:
                message_type = MessageType.TASK_RESPONSE
            elif request_message.message_type == MessageType.QUERY:
                message_type = MessageType.RESPONSE
            elif request_message.message_type == MessageType.HUMAN_INPUT_REQUEST:
                message_type = MessageType.HUMAN_INPUT_RESPONSE
            else:
                message_type = MessageType.RESPONSE

        return Message(
            message_type=message_type,
            sender_id=sender_id,
            recipient_id=request_message.sender_id,
            content=content,
            correlation_id=request_message.message_id,
            priority=request_message.priority,
        )

    @staticmethod
    def create_error_message(
        request_message: Message,
        error: str,
        sender_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """
        Create an error message in response to a request message.

        Args:
            request_message: Original request message
            error: Error message
            sender_id: ID of the responding agent
            details: Additional error details

        Returns:
            Message: Error message
        """
        content = {"error": error, "details": details or {}}

        return Message(
            message_type=MessageType.ERROR,
            sender_id=sender_id,
            recipient_id=request_message.sender_id,
            content=content,
            correlation_id=request_message.message_id,
            priority=max(1, request_message.priority - 1),  # Higher priority for errors
        )
