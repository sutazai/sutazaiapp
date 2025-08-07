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
