"""
Event utilities for consistent handler registration across components.

This avoids duplicate code in registry/services/workflow engines and ensures:
- Lazy initialization of event handler lists
- Optional de-duplication of the same callable per event_type
"""
from typing import Callable, Dict, List


def register_event_handler(event_handlers: Dict[str, List[Callable]], event_type: str, handler: Callable, *, allow_duplicates: bool = False) -> None:
    """Register a handler under event_type into the event_handlers mapping.

    - Initializes list if missing
    - Prevents duplicates by identity unless allow_duplicates=True
    """
    if event_handlers is None:
        raise ValueError("event_handlers mapping is required")
    if not isinstance(event_type, str) or not event_type:
        raise ValueError("event_type must be a non-empty string")
    if not callable(handler):
        raise ValueError("handler must be callable")

    lst = event_handlers.setdefault(event_type, [])
    if allow_duplicates or handler not in lst:
        lst.append(handler)

