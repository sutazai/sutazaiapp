"""
Backward-compatible config shim.
Centralizes settings to backend/app/core/config.py (single source of truth) while preserving imports.
"""
from functools import lru_cache
from app.core.config import Settings as _AppSettings, get_settings as _get_settings

# Preserve historical class name for external imports
AppSettings = _AppSettings

@lru_cache()
def get_settings() -> AppSettings:
    return _get_settings()

# Convenience instance (kept for compatibility)
settings = get_settings()
