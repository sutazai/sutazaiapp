"""
Resource-related message schemas for allocation and management.
"""
from typing import Dict, Any, Optional, List
from pydantic import Field
from .base import BaseMessage, MessageType, Priority, ResourceType


class ResourceRequestMessage(BaseMessage):
    """
    Request for resource allocation.
    """
    message_type: MessageType = MessageType.RESOURCE_REQUEST
    request_id: str = Field(..., description="Unique request identifier")
    requesting_agent: str = Field(..., description="Agent requesting resources")
    task_id: Optional[str] = Field(None, description="Associated task ID")
    resources: Dict[ResourceType, float] = Field(
        ...,
        description="Requested resources and amounts"
    )
    priority: Priority = Priority.NORMAL
    duration_seconds: int = Field(60, description="How long resources are needed")
    exclusive: bool = Field(False, description="Whether exclusive access is needed")
    preemptible: bool = Field(True, description="Can be preempted by higher priority")
    minimum_acceptable: Optional[Dict[ResourceType, float]] = Field(
        None,
        description="Minimum acceptable resource levels"
    )


class ResourceAllocationMessage(BaseMessage):
    """
    Response to resource request with allocation details.
    """
    message_type: MessageType = MessageType.RESOURCE_ALLOCATION
    request_id: str
    allocation_id: str = Field(..., description="Unique allocation identifier")
    allocated: bool = Field(..., description="Whether resources were allocated")
    allocated_resources: Dict[ResourceType, float] = Field(
        default_factory=dict,
        description="Actually allocated resources"
    )
    expires_at: datetime = Field(..., description="When allocation expires")
    partial_allocation: bool = Field(False, description="Whether partially fulfilled")
    rejection_reason: Optional[str] = None
    queue_position: Optional[int] = Field(None, description="Position if queued")
    alternative_resources: Optional[Dict[ResourceType, float]] = None


class ResourceReleaseMessage(BaseMessage):
    """
    Release allocated resources.
    """
    message_type: MessageType = MessageType.RESOURCE_RELEASE
    allocation_id: str = Field(..., description="Allocation to release")
    releasing_agent: str = Field(..., description="Agent releasing resources")
    task_id: Optional[str] = Field(None, description="Associated task that completed")
    actual_usage: Dict[ResourceType, float] = Field(
        default_factory=dict,
        description="Actual resource usage"
    )
    early_release: bool = Field(False, description="Released before expiration")
    release_reason: Optional[str] = None


class ResourceStatusMessage(BaseMessage):
    """
    Current resource availability and usage status.
    """
    message_type: MessageType = MessageType.RESOURCE_STATUS
    total_capacity: Dict[ResourceType, float] = Field(
        default_factory=dict,
        description="Total system capacity"
    )
    available_capacity: Dict[ResourceType, float] = Field(
        default_factory=dict,
        description="Currently available"
    )
    allocated_capacity: Dict[ResourceType, float] = Field(
        default_factory=dict,
        description="Currently allocated"
    )
    reserved_capacity: Dict[ResourceType, float] = Field(
        default_factory=dict,
        description="Reserved for future"
    )
    active_allocations: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="List of active allocations"
    )
    pending_requests: int = Field(0, description="Number of pending requests")
    allocation_metrics: Dict[str, float] = Field(
        default_factory=dict,
        description="Allocation performance metrics"
    )