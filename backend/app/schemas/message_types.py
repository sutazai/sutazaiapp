"""
Canonical message type enumerations for inter-agent communication.

Aligned to IMPORTANT/COMPREHENSIVE_ENGINEERING_STANDARDS.md (single canonical implementation per concept).
Includes a superset of values used across modules to avoid breaking changes.
"""
from __future__ import annotations

from enum import Enum


class MessageType(Enum):
    # Basic protocol
    TASK_REQUEST = "task_request"
    TASK_RESPONSE = "task_response"
    QUERY = "query"
    QUERY_RESPONSE = "query_response"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    HEARTBEAT = "heartbeat"
    COORDINATION = "coordination"
    KNOWLEDGE_SHARE = "knowledge_share"

    # Orchestrator/bus specific
    TASK_ASSIGNMENT = "task_assignment"
    TASK_RESULT = "task_result"
    TASK_COMPLETION = "task_completion"
    TASK_STATUS_UPDATE = "task_status_update"
    COORDINATION_REQUEST = "coordination_request"
    COORDINATION_RESPONSE = "coordination_response"
    WORKFLOW_EVENT = "workflow_event"
    SYSTEM_NOTIFICATION = "system_notification"
    BROADCAST = "broadcast"
    DIRECT_MESSAGE = "direct_message"
    SYSTEM_UPDATE = "system_update"
    AGENT_ANNOUNCEMENT = "agent_announcement"
    CAPABILITY_QUERY = "capability_query"
    DATA_BROADCAST = "data_broadcast"
    CHAT_MESSAGE = "chat_message"
    NOTIFICATION = "notification"
    ALERT = "alert"
    HEALTH_CHECK = "health_check"
    PERFORMANCE_METRIC = "performance_metric"
    # Generic bus styles used in some modules
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    SYSTEM = "system"


class MessagePriority(Enum):
    URGENT = 5
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1
    BACKGROUND = 0

    @classmethod
    def from_value(cls, value):
        # Accept int, str name, or enum instance
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            # Accept both canonical scale (0..5) and agent-bus scale (CRITICAL=1 .. BACKGROUND=5)
            # First try direct match
            for m in cls:
                if m.value == value:
                    return m
            # Fallback: interpret 1..5 as CRITICAL..BACKGROUND
            mapping = {1: cls.CRITICAL, 2: cls.HIGH, 3: cls.NORMAL, 4: cls.LOW, 5: cls.BACKGROUND}
            return mapping.get(value, cls.NORMAL)
        if isinstance(value, str):
            try:
                return cls[value.upper()]
            except Exception:
                return cls.NORMAL
        return cls.NORMAL


class TaskPriority(str, Enum):
    """
    Canonical task priority across the platform.

    - String values preserve API compatibility in models and JSON
    - Provides rank() for numeric comparisons and scheduling algorithms
    - Accepts multiple legacy names via from_value()
    """
    BACKGROUND = "background"
    LOW = "low"
    MEDIUM = "medium"
    NORMAL = "normal"
    HIGH = "high"
    URGENT = "urgent"       # legacy alias often meaning CRITICAL
    CRITICAL = "critical"
    EMERGENCY = "emergency"

    _RANK = {
        BACKGROUND: 0,
        LOW: 1,
        MEDIUM: 2,
        NORMAL: 2,
        HIGH: 3,
        URGENT: 4,
        CRITICAL: 4,
        EMERGENCY: 5,
    }

    @property
    def rank(self) -> int:
        return self._RANK[self]

    @classmethod
    def from_value(cls, value):
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            v = value.strip().lower()
            # normalize common synonyms
            aliases = {
                "med": "medium",
                "norm": "normal",
                "crit": "critical",
                "urgent": "urgent",
                "emerg": "emergency",
            }
            v = aliases.get(v, v)
            for member in cls:
                if member.value == v or member.name.lower() == v:
                    return member
            # Numeric-as-string
            if v.isdigit():
                return cls.from_value(int(v))
            return cls.NORMAL
        if isinstance(value, int):
            mapping = {
                0: cls.BACKGROUND,
                1: cls.LOW,
                2: cls.NORMAL,
                3: cls.HIGH,
                4: cls.CRITICAL,
                5: cls.EMERGENCY,
            }
            return mapping.get(value, cls.NORMAL)
        return cls.NORMAL


class AlertSeverity(str, Enum):
    """Canonical alert severity for monitoring and oversight"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @property
    def level(self) -> int:
        return {self.INFO: 1, self.WARNING: 2, self.ERROR: 3, self.CRITICAL: 4}[self]
