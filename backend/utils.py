#!/usr/bin/env python3.11
"""Utility Module

Provides utility functions for the SutazAI backend, including caching mechanisms
and helper functions for common operations.
"""

import time
from functools import wraps
from typing import Any, Dict

# Simple in-memory cache
cache_data: Dict[str, Any] = {}
cache_expiry: Dict[str, float] = {}


def cache(expire: int = 60):
    """Simple in-memory cache decorator.

    Args:
        expire: Cache expiry time in seconds
    """

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Create a cache key from the function name and arguments
            key = f"{func.__name__}:{str(args)}:{str(kwargs)}"

            # Check if the result is in cache and not expired
            current_time = time.time()
            if key in cache_data and cache_expiry.get(key, 0) > current_time:
                return cache_data[key]

            # Call the function and cache the result
            result = await func(*args, **kwargs)
            cache_data[key] = result
            cache_expiry[key] = current_time + expire

            return result
        return wrapper
    return decorator


def clear_cache() -> None:
    """Clear all cached data."""
    cache_data.clear()
    cache_expiry.clear()
