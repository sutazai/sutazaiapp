#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MessageProtocol - Standardized message protocol for agent communication
"""

import uuid
import time
import json
from typing import Dict, Any, Optional


class MessageProtocol:
    """
    Standardized message protocol for agent communication.
    """

    @staticmethod
    def create_message(
        content: str,
        sender: str,
        message_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a standard message object.

        Args:
            content: The content of the message
            sender: The sender identifier
            message_type: The type of message (default: "text")
            metadata: Optional metadata for the message

        Returns:
            A standardized message dictionary
        """
        if metadata is None:
            metadata = {}

        return {
            "id": str(uuid.uuid4()),
            "content": content,
            "sender": sender,
            "type": message_type,
            "timestamp": time.time(),
            "metadata": metadata,
        }

    @staticmethod
    def create_response_message(
        content: str,
        sender: str,
        request_id: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a response message that references a request.

        Args:
            content: The content of the response
            sender: The sender identifier
            request_id: The ID of the original request message
            metadata: Optional metadata for the message

        Returns:
            A standardized response message dictionary
        """
        if metadata is None:
            metadata = {}

        metadata["request_id"] = request_id

        return MessageProtocol.create_message(
            content=content, sender=sender, message_type="response", metadata=metadata
        )

    @staticmethod
    def create_error_message(
        error: str,
        sender: str,
        request_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create an error message.

        Args:
            error: The error message
            sender: The sender identifier
            request_id: Optional ID of the original request message
            metadata: Optional metadata for the message

        Returns:
            A standardized error message dictionary
        """
        if metadata is None:
            metadata = {}

        if request_id:
            metadata["request_id"] = request_id

        return MessageProtocol.create_message(
            content=error, sender=sender, message_type="error", metadata=metadata
        )

    @staticmethod
    def parse_message(message_data: str) -> Dict[str, Any]:
        """
        Parse a message from its string representation.

        Args:
            message_data: String representation of the message

        Returns:
            The parsed message as a dictionary
        """
        try:
            return json.loads(message_data)
        except json.JSONDecodeError:
            return {
                "id": str(uuid.uuid4()),
                "content": message_data,
                "sender": "unknown",
                "type": "text",
                "timestamp": time.time(),
                "metadata": {},
            }
