"""
Agent Models for SutazAI
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum
from app.schemas.message_types import TaskPriority


class TaskStatus(str, Enum):
    """Task execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentTask(BaseModel):
    """
    Agent task model for tracking agent execution
    """
    id: str = Field(..., description="Unique task identifier")
    agent_name: str = Field(..., description="Name of the agent to execute the task")
    task_description: str = Field(..., description="Description of the task to be performed")
    
    # Optional fields with defaults
    status: TaskStatus = Field(default=TaskStatus.PENDING, description="Current task status")
    priority: TaskPriority = Field(default=TaskPriority.NORMAL, description="Task priority")
    
    # Timestamps
    created_at: datetime = Field(default_factory=datetime.now, description="Task creation timestamp")
    started_at: Optional[datetime] = Field(default=None, description="Task start timestamp")
    completed_at: Optional[datetime] = Field(default=None, description="Task completion timestamp")
    
    # Task data
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Task parameters")
    context: Dict[str, Any] = Field(default_factory=dict, description="Execution context")
    
    # Results
    result: Optional[Dict[str, Any]] = Field(default=None, description="Task execution result")
    error_message: Optional[str] = Field(default=None, description="Error message if task failed")
    
    # Metadata
    max_retries: int = Field(default=3, description="Maximum number of retry attempts")
    retry_count: int = Field(default=0, description="Current retry count")
    timeout_seconds: Optional[int] = Field(default=None, description="Task timeout in seconds")
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }
        
    def is_completed(self) -> bool:
        """Check if task is in a completed state"""
        return self.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]
    
    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return (
            self.status == TaskStatus.FAILED and 
            self.retry_count < self.max_retries
        )
    
    def mark_started(self) -> None:
        """Mark task as started"""
        self.status = TaskStatus.RUNNING
        self.started_at = datetime.now()
    
    def mark_completed(self, result: Dict[str, Any]) -> None:
        """Mark task as completed with result"""
        self.status = TaskStatus.COMPLETED
        self.completed_at = datetime.now()
        self.result = result
    
    def mark_failed(self, error_message: str) -> None:
        """Mark task as failed with error message"""
        self.status = TaskStatus.FAILED
        self.completed_at = datetime.now()
        self.error_message = error_message
        self.retry_count += 1
    
    def mark_cancelled(self) -> None:
        """Mark task as cancelled"""
        self.status = TaskStatus.CANCELLED
        self.completed_at = datetime.now() 
