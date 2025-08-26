"""
Base message schemas and enums for all agent communications.
"""
from datetime import datetime
from typing import Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field
import uuid


class MessageType(str, Enum):
    """All possible message types in the system"""
    # Agent messages
    AGENT_REGISTRATION = "agent.registration"
    AGENT_HEARTBEAT = "agent.heartbeat"
    AGENT_STATUS = "agent.status"
    AGENT_CAPABILITY = "agent.capability"
    
    # Task messages
    TASK_REQUEST = "task.request"
    TASK_RESPONSE = "task.response"
    TASK_STATUS = "task.status"
    TASK_ASSIGNMENT = "task.assignment"
    TASK_COMPLETION = "task.completion"
    
    # Resource messages
    RESOURCE_REQUEST = "resource.request"
    RESOURCE_ALLOCATION = "resource.allocation"
    RESOURCE_RELEASE = "resource.release"
    RESOURCE_STATUS = "resource.status"
    
    # System messages
    SYSTEM_HEALTH = "system.health"
    SYSTEM_ALERT = "system.alert"
    ERROR = "system.error"
    ACKNOWLEDGEMENT = "system.ack"


class Priority(int, Enum):
    """Message priority levels"""
    LOW = 0
    NORMAL = 1
    HIGH = 2
    CRITICAL = 3


class TaskStatus(str, Enum):
    """Task lifecycle states"""
    PENDING = "pending"
    QUEUED = "queued"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class ResourceType(str, Enum):
    """Types of system resources"""
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    DISK = "disk"
    NETWORK = "network"
    CUSTOM = "custom"


class AgentStatus(str, Enum):
    """Agent operational states"""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    OVERLOADED = "overloaded"
    OFFLINE = "offline"
    ERROR = "error"
    SHUTDOWN = "shutdown"


class BaseMessage(BaseModel):
    """
    Base schema for all messages in the system.
    Every message must inherit from this class.
    """
    message_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message_type: MessageType
    source_agent: str = Field(..., description="ID of the sending agent")
    target_agent: Optional[str] = Field(None, description="ID of target agent, None for broadcast")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = Field(None, description="ID to correlate related messages")
    priority: Priority = Field(Priority.NORMAL, description="Message priority")
    ttl: Optional[int] = Field(None, description="Time to live in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")
    
    class Config:
        use_enum_values = True
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }