"""
Validation helpers for API schemas.
"""
import re
from typing import Optional


_MODEL_NAME_RE = re.compile(r'^[a-zA-Z0-9._:-]+$')


def validate_model_name(value: Optional[str]) -> Optional[str]:
    """
    Validate model identifier format used in API requests.
    - Returns stripped value when valid, raises ValueError when invalid.
    """
    if value is None:
        return value
    v = value.strip()
    if not _MODEL_NAME_RE.match(v):
        raise ValueError("Invalid model name format")
    return v

