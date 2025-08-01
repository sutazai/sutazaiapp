"""
Base Models for Backend Components
Shared data models and schemas
"""

from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import BaseModel, Field
from enum import Enum

class MessageType(Enum):
    """Message types for inter-component communication"""
    INFO = "info"
    ERROR = "error"
    WARNING = "warning"
    SUCCESS = "success"
    TASK = "task"
    RESPONSE = "response"

class Message(BaseModel):
    """Base message model"""
    id: str = Field(..., description="Unique message identifier")
    type: MessageType = Field(..., description="Message type")
    content: Dict[str, Any] = Field(..., description="Message content")
    sender: Optional[str] = Field(None, description="Message sender")
    recipient: Optional[str] = Field(None, description="Message recipient")
    timestamp: datetime = Field(default_factory=datetime.now, description="Message timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class TaskStatus(Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class Task(BaseModel):
    """Base task model"""
    id: str = Field(..., description="Unique task identifier")
    type: str = Field(..., description="Task type")
    description: str = Field(..., description="Task description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    status: TaskStatus = Field(TaskStatus.PENDING, description="Task status")
    created_at: datetime = Field(default_factory=datetime.now, description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    result: Optional[Dict[str, Any]] = Field(None, description="Task result")
    error: Optional[str] = Field(None, description="Error message if failed")

class AgentInfo(BaseModel):
    """Agent information model"""
    id: str = Field(..., description="Agent identifier")
    name: str = Field(..., description="Agent name")
    type: str = Field(..., description="Agent type")
    status: str = Field(..., description="Agent status")
    capabilities: List[str] = Field(default_factory=list, description="Agent capabilities")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class SystemMetrics(BaseModel):
    """System metrics model"""
    timestamp: datetime = Field(default_factory=datetime.now, description="Metrics timestamp")
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_percent: float = Field(..., description="Memory usage percentage")
    disk_percent: float = Field(..., description="Disk usage percentage")
    active_agents: int = Field(..., description="Number of active agents")
    pending_tasks: int = Field(..., description="Number of pending tasks")
    completed_tasks: int = Field(..., description="Number of completed tasks")

class ServiceHealth(BaseModel):
    """Service health status model"""
    service_name: str = Field(..., description="Service name")
    status: str = Field(..., description="Health status")
    last_check: datetime = Field(default_factory=datetime.now, description="Last health check")
    details: Dict[str, Any] = Field(default_factory=dict, description="Health details")
    uptime: Optional[float] = Field(None, description="Service uptime in seconds")