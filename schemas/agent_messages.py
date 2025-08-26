"""
Agent-related message schemas for registration, heartbeat, and status.
"""
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import Field
from .base import BaseMessage, MessageType, AgentStatus


class AgentRegistrationMessage(BaseMessage):
    """
    Message sent when an agent registers with the system.
    """
    message_type: MessageType = MessageType.AGENT_REGISTRATION
    agent_id: str = Field(..., description="Unique identifier for the agent")
    agent_type: str = Field(..., description="Type of agent (e.g., 'orchestrator', 'worker')")
    capabilities: List[str] = Field(default_factory=list, description="List of agent capabilities")
    version: str = Field(..., description="Agent version")
    host: str = Field(..., description="Host where agent is running")
    port: int = Field(..., description="Port where agent is listening")
    max_concurrent_tasks: int = Field(1, description="Maximum concurrent tasks")
    supported_message_types: List[str] = Field(default_factory=list)
    configuration: Dict[str, Any] = Field(default_factory=dict)


class AgentHeartbeatMessage(BaseMessage):
    """
    Periodic heartbeat message to indicate agent is alive.
    """
    message_type: MessageType = MessageType.AGENT_HEARTBEAT
    agent_id: str
    status: AgentStatus
    current_load: float = Field(0.0, ge=0.0, le=1.0, description="Current load (0-1)")
    active_tasks: int = Field(0, ge=0)
    available_capacity: int = Field(0, ge=0)
    cpu_usage: float = Field(0.0, ge=0.0, le=100.0)
    memory_usage: float = Field(0.0, ge=0.0, le=100.0)
    last_task_completed: Optional[datetime] = None
    uptime_seconds: float = Field(0.0, ge=0.0)
    error_count: int = Field(0, ge=0)


class AgentStatusMessage(BaseMessage):
    """
    Detailed status message with agent metrics and state.
    """
    message_type: MessageType = MessageType.AGENT_STATUS
    agent_id: str
    status: AgentStatus
    metrics: Dict[str, Any] = Field(default_factory=dict)
    active_task_ids: List[str] = Field(default_factory=list)
    queued_task_ids: List[str] = Field(default_factory=list)
    resource_usage: Dict[str, float] = Field(default_factory=dict)
    error_messages: List[str] = Field(default_factory=list)
    last_error_time: Optional[datetime] = None
    configuration_hash: Optional[str] = None


class AgentCapabilityMessage(BaseMessage):
    """
    Message describing agent capabilities and requirements.
    """
    message_type: MessageType = MessageType.AGENT_CAPABILITY
    agent_id: str
    capabilities: Dict[str, Any] = Field(
        default_factory=dict,
        description="Detailed capability descriptions"
    )
    requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource requirements for operations"
    )
    performance_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Historical performance metrics"
    )
    supported_task_types: List[str] = Field(default_factory=list)
    max_task_size: Optional[int] = Field(None, description="Maximum task size in bytes")
    timeout_seconds: int = Field(300, description="Default task timeout")