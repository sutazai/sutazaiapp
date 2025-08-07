"""
Message adapters to translate to canonical enums from heterogeneous message dicts.

Non-destructive: used at integration points; original modules can keep their enums.
"""
from __future__ import annotations

from typing import Any, Dict, Tuple

from .message_types import MessageType, MessagePriority


def to_canonical_enums(message_dict: Dict[str, Any]) -> Tuple[MessageType, MessagePriority]:
    """Return canonical (MessageType, MessagePriority) from a message dict.

    Accepts either name fields (message_type_name/priority_name) or value fields
    (message_type/priority) where type may be a string name or string value.
    Handles differing integer scales used by some modules for priority.
    """
    mt_raw = message_dict.get('message_type_name') or message_dict.get('message_type')
    pr_raw = message_dict.get('priority_name') if 'priority_name' in message_dict else message_dict.get('priority')

    # MessageType mapping
    mt: MessageType
    if isinstance(mt_raw, str):
        # Try direct name
        try:
            mt = MessageType[mt_raw.upper()]
        except Exception:
            # Try value string
            try:
                mt = MessageType(mt_raw)
            except Exception:
                mt = MessageType.EVENT if 'event' in (mt_raw or '') else MessageType.SYSTEM
    else:
        # Unknown type -> default
        mt = MessageType.EVENT

    # MessagePriority mapping (robust to int scale and names)
    pr = MessagePriority.NORMAL
    if pr_raw is not None:
        pr = MessagePriority.from_value(pr_raw)

    return mt, pr

