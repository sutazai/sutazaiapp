"""
Message Protocol Module
Defines message types and protocols for agent communication
"""

import uuid
from enum import Enum
from typing import Dict, Any, Optional
from datetime import datetime
from dataclasses import dataclass

class MessageType(Enum):
    """Types of messages that can be sent between agents"""
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    QUERY = "query"
    QUERY_RESPONSE = "query_response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    COORDINATION = "coordination"
    KNOWLEDGE_SHARE = "knowledge_share"

@dataclass
class Message:
    """Base message class for agent communication"""
    message_id: str
    message_type: MessageType
    sender_id: str
    recipient_id: Optional[str]
    content: Dict[str, Any]
    timestamp: datetime
    priority: int = 5  # 1-10, 10 being highest priority
    
    def __post_init__(self):
        if isinstance(self.message_type, str):
            self.message_type = MessageType(self.message_type)

class MessageProtocol:
    """Protocol utilities for creating and managing messages"""
    
    @staticmethod
    def create_task_request(sender_id: str, recipient_id: str, task: Dict[str, Any]) -> Message:
        """Create a task request message"""
        return Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.TASK_REQUEST,
            sender_id=sender_id,
            recipient_id=recipient_id,
            content={
                "task": task,
                "requested_at": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            priority=7
        )
        
    @staticmethod
    def create_response_message(request_message: Message, content: Dict[str, Any], sender_id: str) -> Message:
        """Create a response message to a request"""
        response_type = MessageType.TASK_RESPONSE if request_message.message_type == MessageType.TASK_REQUEST else MessageType.QUERY_RESPONSE
        
        return Message(
            message_id=str(uuid.uuid4()),
            message_type=response_type,
            sender_id=sender_id,
            recipient_id=request_message.sender_id,
            content={
                "response": content,
                "request_id": request_message.message_id,
                "responded_at": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            priority=request_message.priority
        )
        
    @staticmethod
    def create_error_message(request_message: Message, error: str, sender_id: str, details: Optional[Dict[str, Any]] = None) -> Message:
        """Create an error response message"""
        return Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.ERROR,
            sender_id=sender_id,
            recipient_id=request_message.sender_id,
            content={
                "error": error,
                "request_id": request_message.message_id,
                "details": details or {},
                "error_time": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            priority=9  # High priority for errors
        )
        
    @staticmethod
    def create_status_update(sender_id: str, status: str, details: Optional[Dict[str, Any]] = None) -> Message:
        """Create a status update message"""
        return Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.STATUS_UPDATE,
            sender_id=sender_id,
            recipient_id=None,  # Broadcast
            content={
                "status": status,
                "details": details or {},
                "status_time": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            priority=3
        )
        
    @staticmethod
    def create_heartbeat(sender_id: str, health_data: Optional[Dict[str, Any]] = None) -> Message:
        """Create a heartbeat message"""
        return Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.HEARTBEAT,
            sender_id=sender_id,
            recipient_id=None,  # Broadcast
            content={
                "health": health_data or {},
                "heartbeat_time": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            priority=1  # Low priority
        )
        
    @staticmethod
    def create_coordination_message(sender_id: str, coordination_type: str, data: Dict[str, Any], target_agents: Optional[list] = None) -> Message:
        """Create a coordination message"""
        return Message(
            message_id=str(uuid.uuid4()),
            message_type=MessageType.COORDINATION,
            sender_id=sender_id,
            recipient_id=None,  # Broadcast or to specific agents
            content={
                "coordination_type": coordination_type,
                "data": data,
                "target_agents": target_agents,
                "coordination_time": datetime.now().isoformat()
            },
            timestamp=datetime.now(),
            priority=6
        )