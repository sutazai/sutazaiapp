"""
SutazAI Copy Internals Module
-----------------------------
A simplified version of copy utilities for the SutazAI system.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Set, TypeVar


# Simple utility functions for copying and model manipulation
# This is a simplified version that doesn't rely on Pydantic internals

def object_setattr(obj: Any, name: str, value: Any) -> None:
    """Set attribute on object, bypassing __setattr__."""
    object.__setattr__(obj, name, value)


def deep_copy(obj: Any) -> Any:
    """Perform a deep copy of an object."""
    return deepcopy(obj)


def is_sequence_like(obj: Any) -> bool:
    """Check if an object is sequence-like (list, tuple, etc.)."""
    return isinstance(obj, (list, tuple, set))


def is_dict_like(obj: Any) -> bool:
    """Check if an object is dict-like."""
    return isinstance(obj, dict)


def copy_dict_exclude(d: Dict[str, Any], exclude_keys: Set[str]) -> Dict[str, Any]:
    """Copy a dictionary excluding certain keys."""
    return {k: v for k, v in d.items() if k not in exclude_keys}


def copy_dict_include(d: Dict[str, Any], include_keys: Set[str]) -> Dict[str, Any]:
    """Copy a dictionary including only certain keys."""
    return {k: v for k, v in d.items() if k in include_keys}


T = TypeVar('T')


def copy_and_set_values(
    obj: T,
    values: Dict[str, Any],
    *,
    deep: bool = False,
) -> T:
    """
    Create a copy of an object with new values.
    
    Args:
        obj: The object to copy
        values: New values to set
        deep: Whether to perform a deep copy
        
    Returns:
        A new instance with the values set
    """
    if deep:
        values = deepcopy(values)
    
    # Create a new instance of the same class
    cls = obj.__class__
    new_obj = cls.__new__(cls)
    
    # Set the values
    for key, value in values.items():
        setattr(new_obj, key, value)
    
    return new_obj


# Add any other utility functions needed for the system here
