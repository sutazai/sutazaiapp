# Proxy module for pydantic_settings
# This makes sure we have the right BaseSettings regardless of pydantic version
try:
    # Try to import from pydantic_settings (for pydantic v2)
    from pydantic_settings import BaseSettings as _BaseSettings

    BaseSettings = _BaseSettings
except ImportError:
    # Fall back to pydantic v1 import
    try:
        pass
    except ImportError:
        raise ImportError(
            "Neither pydantic_settings nor pydantic BaseSettings could be imported"
        )

__version__ = "proxy.1.0.1"
