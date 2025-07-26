"""
Agent Status Module

Defines agent status enumeration and related functionality.
"""

from enum import Enum, auto
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime


class AgentStatus(Enum):
    """Enumeration of possible agent statuses"""
    INACTIVE = "inactive"
    STARTING = "starting"
    ACTIVE = "active"
    BUSY = "busy"
    PAUSED = "paused"
    ERROR = "error"
    STOPPING = "stopping"
    MAINTENANCE = "maintenance"
    
    @classmethod
    def from_string(cls, status_str: str) -> 'AgentStatus':
        """Convert string to AgentStatus enum"""
        status_map = {status.value: status for status in cls}
        return status_map.get(status_str.lower(), cls.INACTIVE)
    
    def is_operational(self) -> bool:
        """Check if agent is in an operational state"""
        return self in [AgentStatus.ACTIVE, AgentStatus.BUSY]
    
    def can_accept_tasks(self) -> bool:
        """Check if agent can accept new tasks"""
        return self == AgentStatus.ACTIVE


@dataclass
class AgentStatusInfo:
    """Detailed agent status information"""
    status: AgentStatus
    last_updated: datetime
    current_task: Optional[str] = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = None
    uptime_seconds: float = 0.0
    
    def __post_init__(self):
        if self.performance_metrics is None:
            self.performance_metrics = {}
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "status": self.status.value,
            "last_updated": self.last_updated.isoformat(),
            "current_task": self.current_task,
            "error_message": self.error_message,
            "performance_metrics": self.performance_metrics,
            "uptime_seconds": self.uptime_seconds,
            "is_operational": self.status.is_operational(),
            "can_accept_tasks": self.status.can_accept_tasks()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentStatusInfo':
        """Create from dictionary representation"""
        return cls(
            status=AgentStatus.from_string(data.get("status", "inactive")),
            last_updated=datetime.fromisoformat(data["last_updated"]),
            current_task=data.get("current_task"),
            error_message=data.get("error_message"),
            performance_metrics=data.get("performance_metrics", {}),
            uptime_seconds=data.get("uptime_seconds", 0.0)
        )


class StatusTransitionValidator:
    """Validates agent status transitions"""
    
    VALID_TRANSITIONS = {
        AgentStatus.INACTIVE: [AgentStatus.STARTING, AgentStatus.MAINTENANCE],
        AgentStatus.STARTING: [AgentStatus.ACTIVE, AgentStatus.ERROR, AgentStatus.INACTIVE],
        AgentStatus.ACTIVE: [AgentStatus.BUSY, AgentStatus.PAUSED, AgentStatus.STOPPING, AgentStatus.ERROR, AgentStatus.MAINTENANCE],
        AgentStatus.BUSY: [AgentStatus.ACTIVE, AgentStatus.ERROR, AgentStatus.STOPPING],
        AgentStatus.PAUSED: [AgentStatus.ACTIVE, AgentStatus.STOPPING, AgentStatus.ERROR],
        AgentStatus.ERROR: [AgentStatus.INACTIVE, AgentStatus.STARTING, AgentStatus.MAINTENANCE],
        AgentStatus.STOPPING: [AgentStatus.INACTIVE],
        AgentStatus.MAINTENANCE: [AgentStatus.INACTIVE, AgentStatus.STARTING]
    }
    
    @classmethod
    def is_valid_transition(cls, from_status: AgentStatus, to_status: AgentStatus) -> bool:
        """Check if status transition is valid"""
        valid_next_states = cls.VALID_TRANSITIONS.get(from_status, [])
        return to_status in valid_next_states
    
    @classmethod
    def get_valid_transitions(cls, from_status: AgentStatus) -> list[AgentStatus]:
        """Get list of valid next statuses"""
        return cls.VALID_TRANSITIONS.get(from_status, [])