"""
Canonical message model for inter-agent communication.

Goals:
- Provide a single Message dataclass used across modules
- Normalize field names (sender/recipient/payload)
- Be tolerant of legacy shapes (sender_id/content/etc.) via from_dict
- Avoid external dependencies beyond stdlib and our enums
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional

from .message_types import MessageType, MessagePriority


class DeliveryMode(Enum):
    FIRE_AND_FORGET = "fire_and_forget"
    AT_LEAST_ONCE = "at_least_once"
    EXACTLY_ONCE = "exactly_once"
    REQUEST_RESPONSE = "request_response"


@dataclass
class Message:
    """Canonical message structure.

    Field names are normalized to:
    - sender: str
    - recipient: Optional[str] (None for broadcast)
    - payload: Dict[str, Any]

    The model remains lenient when parsing from dicts that use sender_id/recipient_id/content
    and integer or named priorities. Extra optional fields support advanced flows but are
    not required by all modules.
    """

    id: str
    type: MessageType
    sender: str
    recipient: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    priority: MessagePriority = MessagePriority.NORMAL

    # Optional/advanced fields
    ttl: Optional[int] = None
    correlation_id: Optional[str] = None
    requires_response: bool = False
    delivery_mode: DeliveryMode = DeliveryMode.FIRE_AND_FORGET
    reply_to: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a portable dict representation."""
        return {
            "id": self.id,
            "type": self.type.value,
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "ttl": self.ttl,
            "correlation_id": self.correlation_id,
            "requires_response": self.requires_response,
            "delivery_mode": getattr(self.delivery_mode, "value", self.delivery_mode),
            "reply_to": self.reply_to,
            "retry_count": self.retry_count,
            "max_retries": self.max_retries,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """Parse from various legacy/canonical shapes.

        Accepts both sender/sender_id, recipient/recipient_id, payload/content.
        Accepts MessageType names or values. Priority accepts ints, names, or enum.
        """
        # Resolve type
        mtype_raw = data.get("type") or data.get("message_type") or data.get("messageType")
        if isinstance(mtype_raw, MessageType):
            mtype = mtype_raw
        else:
            try:
                # Try as value string
                mtype = MessageType(mtype_raw)
            except Exception:
                # Try as name
                try:
                    mtype = MessageType[str(mtype_raw).upper()]
                except Exception:
                    mtype = MessageType.EVENT

        # Normalize fields
        sender = data.get("sender") or data.get("sender_id") or data.get("from") or "system"
        recipient = data.get("recipient") or data.get("recipient_id") or data.get("to")
        payload = data.get("payload") or data.get("content") or {}

        # Timestamp
        ts = data.get("timestamp")
        if isinstance(ts, datetime):
            timestamp = ts
        else:
            try:
                timestamp = datetime.fromisoformat(ts) if ts else datetime.now()
            except Exception:
                timestamp = datetime.now()

        # Priority
        priority = MessagePriority.from_value(data.get("priority", MessagePriority.NORMAL.value))

        # Delivery mode
        dm_raw = data.get("delivery_mode")
        if isinstance(dm_raw, DeliveryMode):
            delivery_mode = dm_raw
        else:
            try:
                delivery_mode = DeliveryMode(dm_raw) if dm_raw else DeliveryMode.FIRE_AND_FORGET
            except Exception:
                delivery_mode = DeliveryMode.FIRE_AND_FORGET

        return cls(
            id=str(data.get("id")),
            type=mtype,
            sender=sender,
            recipient=recipient,
            payload=payload,
            timestamp=timestamp,
            priority=priority,
            ttl=data.get("ttl"),
            correlation_id=data.get("correlation_id"),
            requires_response=bool(data.get("requires_response", False)),
            delivery_mode=delivery_mode,
            reply_to=data.get("reply_to"),
            retry_count=int(data.get("retry_count", 0) or 0),
            max_retries=int(data.get("max_retries", 3) or 3),
            metadata=data.get("metadata", {}),
        )

    def create_response(self, payload: Dict[str, Any]) -> "Message":
        """Create a response message, correlating to this message."""
        import uuid

        return Message(
            id=f"resp_{uuid.uuid4().hex[:8]}",
            type=MessageType.RESPONSE,
            sender=self.recipient or "system",
            recipient=self.sender,
            payload=payload,
            priority=self.priority,
            correlation_id=self.id,
            delivery_mode=DeliveryMode.FIRE_AND_FORGET,
        )

    # Back-compat helpers for modules expecting different field names
    def is_expired(self) -> bool:
        """Check if message has expired based on TTL."""
        if not self.ttl:
            return False
        try:
            elapsed = (datetime.now() - self.timestamp).total_seconds()
            return elapsed > int(self.ttl)
        except Exception:
            return False

    @property
    def sender_id(self) -> str:
        return self.sender

    @property
    def recipient_id(self) -> Optional[str]:
        return self.recipient

    @property
    def content(self) -> Dict[str, Any]:
        return self.payload
