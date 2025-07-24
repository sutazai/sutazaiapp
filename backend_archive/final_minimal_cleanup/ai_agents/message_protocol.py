#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
MessageProtocol - Standardized message protocol for agent communication
"""

import uuid
import time
import json
from typing import Dict, Any, Optional
import traceback
import logging
from enum import Enum

# Import core definitions needed by the handler/other parts of the module
# from .message_protocol import Message, MessageType, create_message, parse_message

logger = logging.getLogger(__name__)

# --- Core Message Definitions ---
class MessageType(Enum):
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    AGENT_CONTROL = "agent_control"
    CONTROL_RESPONSE = "control_response"
    HEARTBEAT = "heartbeat"
    ERROR = "error"

# Define the Message class structure
class Message:
    """Represents a message exchanged between agents or system components."""
    def __init__(self,
                 msg_type: MessageType,
                 payload: Dict[str, Any],
                 msg_id: Optional[str] = None,
                 timestamp: Optional[float] = None,
                 correlation_id: Optional[str] = None,
                 sender_agent: Optional[str] = None,
                 target_agent: Optional[str] = None):
        self.id = msg_id or str(uuid.uuid4())
        self.type = msg_type
        self.payload = payload
        self.timestamp = timestamp or time.time()
        self.correlation_id = correlation_id
        self.sender_agent = sender_agent
        self.target_agent = target_agent

    def to_json(self) -> str:
        """Serialize the message to a JSON string."""
        # Convert enum to string for serialization
        data = self.__dict__.copy()
        data['type'] = self.type.value
        return json.dumps(data)

    @staticmethod
    def from_dict(data: Dict[str, Any]) -> 'Message':
        """Deserialize a message from a dictionary."""
        msg_type_str = data.get('type')
        if not msg_type_str:
            raise ValueError("Message data missing 'type' field")
        try:
            msg_type = MessageType(msg_type_str)
        except ValueError:
            raise ValueError(f"Invalid message type: {msg_type_str}")

        return Message(
            msg_type=msg_type,
            payload=data.get('payload', {}),
            msg_id=data.get('id'),
            timestamp=data.get('timestamp'),
            correlation_id=data.get('correlation_id'),
            sender_agent=data.get('sender_agent'),
            target_agent=data.get('target_agent')
        )

def create_message(msg_type: MessageType, payload: Dict[str, Any], correlation_id: Optional[str] = None) -> Message:
    """Helper function to create a Message object."""
    return Message(msg_type=msg_type, payload=payload, correlation_id=correlation_id)

def parse_message(json_string: str) -> Message:
    """Helper function to parse a JSON string into a Message object."""
    try:
        data = json.loads(json_string)
        return Message.from_dict(data)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Error parsing message: {e}")

# --- Message Handler Class ---

class MessageHandler:
    """Handles encoding/decoding and processing of messages."""

    def __init__(self, agent_manager=None):
        self.agent_manager = agent_manager

    def encode_message(self, message: Message) -> bytes:
        """Encode a Message object into bytes."""
        return message.to_json().encode('utf-8')

    def decode_message(self, data: bytes) -> Message:
        """Decode bytes into a Message object."""
        try:
            json_data = data.decode('utf-8')
            return parse_message(json_data)
        except Exception as e:
            logger.error(f"Failed to decode message: {e}")
            raise ValueError("Invalid message format")

    async def process_message(self, message: Message) -> Message:
        """Process an incoming message and generate a response."""
        try:
            response_payload: Dict[str, Any] = {}
            if message.type == MessageType.TASK_REQUEST:
                # Get target agent
                agent_id = message.target_agent
                if not agent_id:
                     return create_message(MessageType.ERROR, {"error": "No target agent specified"})

                # Delegate task to AgentManager
                if not self.agent_manager:
                     logger.error("AgentManager not available in MessageHandler")
                     return create_message(MessageType.ERROR, {"error": "Agent manager not configured"})

                # Assume agent_manager.handle_task returns the result payload
                result = await self.agent_manager.handle_task(agent_id, message.payload)
                response_payload = result
                response_type = MessageType.TASK_RESPONSE

            elif message.type == MessageType.AGENT_CONTROL:
                # Handle control messages (e.g., start, stop, configure)
                control_action = message.payload.get("action")
                agent_id = message.payload.get("agent_id")
                config = message.payload.get("config")

                if not agent_id or not control_action:
                    return create_message(MessageType.ERROR, {"error": "Missing agent_id or action in control message"})

                if not self.agent_manager:
                     logger.error("AgentManager not available in MessageHandler")
                     return create_message(MessageType.ERROR, {"error": "Agent manager not configured"})

                # Placeholder for control logic
                if control_action == "start":
                    success = self.agent_manager.start_agent(agent_id)
                    response_payload = {"status": "started" if success else "failed_to_start"}
                elif control_action == "stop":
                    success = self.agent_manager.stop_agent(agent_id)
                    response_payload = {"status": "stopped" if success else "failed_to_stop"}
                elif control_action == "configure":
                    # Requires update_agent_config implementation
                    # success = self.agent_manager.update_agent_config(agent_id, config)
                    # response_payload = {"status": "configured" if success else "failed_to_configure"}
                    response_payload = {"status": "configure_not_implemented"}
                else:
                     response_payload = {"error": f"Unsupported control action: {control_action}"}

                response_type = MessageType.CONTROL_RESPONSE

            elif message.type == MessageType.HEARTBEAT:
                 # Respond to heartbeat
                 response_payload = {"status": "alive", "timestamp": time.time()}
                 response_type = MessageType.HEARTBEAT

            else:
                response_payload = {"error": f"Unsupported message type: {message.type}"}
                response_type = MessageType.ERROR

            # Create and return response message
            return create_message(response_type, response_payload, message.id)

        except Exception as e:
            logger.error(f"Error processing message: {e}\n{traceback.format_exc()}")
            return create_message(
                MessageType.ERROR,
                {"error": f"Internal server error: {str(e)}", "details": traceback.format_exc()},
                message.id
            )
