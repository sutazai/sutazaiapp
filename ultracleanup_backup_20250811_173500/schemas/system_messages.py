"""
System-level message schemas for health, alerts, and errors.
"""
from typing import Dict, Any, Optional, List
from datetime import datetime
from pydantic import Field
from .base import BaseMessage, MessageType, Priority


class SystemHealthMessage(BaseMessage):
    """
    System-wide health status message.
    """
    message_type: MessageType = MessageType.SYSTEM_HEALTH
    healthy: bool = Field(..., description="Overall system health")
    components: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict,
        description="Health status of each component"
    )
    metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="System-wide metrics"
    )
    active_agents: int = Field(0, ge=0)
    total_tasks_processed: int = Field(0, ge=0)
    error_rate: float = Field(0.0, ge=0.0, le=1.0)
    average_response_time_ms: float = Field(0.0, ge=0.0)
    warnings: List[str] = Field(default_factory=list)
    degraded_services: List[str] = Field(default_factory=list)


class SystemAlertMessage(BaseMessage):
    """
    System alert or notification message.
    """
    message_type: MessageType = MessageType.SYSTEM_ALERT
    alert_id: str = Field(..., description="Unique alert identifier")
    severity: str = Field(..., description="Alert severity (info/warning/error/critical)")
    category: str = Field(..., description="Alert category")
    title: str = Field(..., description="Alert title")
    description: str = Field(..., description="Detailed description")
    affected_components: List[str] = Field(default_factory=list)
    recommended_action: Optional[str] = None
    auto_resolved: bool = Field(False, description="Can be auto-resolved")
    expires_at: Optional[datetime] = None
    context: Dict[str, Any] = Field(default_factory=dict)


class ErrorMessage(BaseMessage):
    """
    Error message for failures and exceptions.
    """
    message_type: MessageType = MessageType.ERROR
    error_id: str = Field(..., description="Unique error identifier")
    error_code: str = Field(..., description="Error code for categorization")
    error_message: str = Field(..., description="Human-readable error message")
    error_type: str = Field(..., description="Type of error")
    severity: Priority = Priority.HIGH
    stack_trace: Optional[str] = None
    affected_task_id: Optional[str] = None
    affected_agent_id: Optional[str] = None
    recovery_suggestion: Optional[str] = None
    retry_possible: bool = Field(False, description="Whether retry might succeed")
    context: Dict[str, Any] = Field(default_factory=dict, description="Error context")


class AcknowledgementMessage(BaseMessage):
    """
    Simple acknowledgement message for confirming receipt.
    """
    message_type: MessageType = MessageType.ACKNOWLEDGEMENT
    ack_message_id: str = Field(..., description="ID of message being acknowledged")
    ack_status: str = Field(..., description="received/processed/rejected")
    processing_time_ms: Optional[float] = None
    notes: Optional[str] = None