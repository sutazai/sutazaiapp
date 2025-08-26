"""
Task-related message schemas for request, assignment, and completion.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import Field
from .base import BaseMessage, MessageType, Priority, TaskStatus


class TaskRequestMessage(BaseMessage):
    """
    Message requesting task execution.
    """
    message_type: MessageType = MessageType.TASK_REQUEST
    task_id: str = Field(..., description="Unique task identifier")
    task_type: str = Field(..., description="Type of task to execute")
    payload: Dict[str, Any] = Field(default_factory=dict, description="Task data")
    requirements: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resource and capability requirements"
    )
    priority: Priority = Priority.NORMAL
    timeout_seconds: int = Field(300, description="Task timeout in seconds")
    retry_count: int = Field(0, description="Number of retries attempted")
    max_retries: int = Field(3, description="Maximum retries allowed")
    dependencies: List[str] = Field(default_factory=list, description="Task dependencies")
    callback_url: Optional[str] = Field(None, description="Webhook for completion")


class TaskResponseMessage(BaseMessage):
    """
    Initial response to task request.
    """
    message_type: MessageType = MessageType.TASK_RESPONSE
    task_id: str
    accepted: bool = Field(..., description="Whether task was accepted")
    assigned_agent: Optional[str] = Field(None, description="Agent assigned to task")
    estimated_completion: Optional[datetime] = None
    rejection_reason: Optional[str] = None
    queue_position: Optional[int] = None


class TaskStatusUpdateMessage(BaseMessage):
    """
    Progress update for an in-progress task.
    """
    message_type: MessageType = MessageType.TASK_STATUS
    task_id: str
    status: TaskStatus
    progress: float = Field(0.0, ge=0.0, le=1.0, description="Progress (0-1)")
    message: Optional[str] = Field(None, description="Status message")
    current_step: Optional[str] = None
    total_steps: Optional[int] = None
    elapsed_seconds: float = Field(0.0, ge=0.0)
    estimated_remaining_seconds: Optional[float] = None
    intermediate_results: Optional[Dict[str, Any]] = None


class TaskAssignmentMessage(BaseMessage):
    """
    Message assigning a task to an agent.
    """
    message_type: MessageType = MessageType.TASK_ASSIGNMENT
    task_id: str
    assigned_agent: str = Field(..., description="Agent ID assigned to task")
    assignment_time: datetime = Field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    priority_boost: int = Field(0, description="Priority adjustment")
    resource_allocation: Dict[str, Any] = Field(
        default_factory=dict,
        description="Resources allocated for task"
    )
    execution_constraints: Dict[str, Any] = Field(default_factory=dict)


class TaskCompletionMessage(BaseMessage):
    """
    Message indicating task completion.
    """
    message_type: MessageType = MessageType.TASK_COMPLETION
    task_id: str
    status: TaskStatus = Field(..., description="Final task status")
    result: Optional[Dict[str, Any]] = Field(None, description="Task results")
    error: Optional[str] = Field(None, description="Error message if failed")
    error_details: Optional[Dict[str, Any]] = None
    execution_time_seconds: float = Field(..., ge=0.0)
    resource_usage: Dict[str, float] = Field(
        default_factory=dict,
        description="Resources consumed"
    )
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Performance metrics")
    completion_time: datetime = Field(default_factory=datetime.utcnow)