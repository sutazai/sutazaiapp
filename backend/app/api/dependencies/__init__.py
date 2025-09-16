"""
API Dependencies for FastAPI
"""

from app.api.dependencies.auth import (
    get_current_user,
    get_current_user_optional,
    get_current_active_user,
    get_current_verified_user,
    get_current_superuser,
    rate_limiter,
    strict_rate_limiter,
    oauth2_scheme,
    oauth2_scheme_optional
)

__all__ = [
    "get_current_user",
    "get_current_user_optional",
    "get_current_active_user",
    "get_current_verified_user",
    "get_current_superuser",
    "rate_limiter",
    "strict_rate_limiter",
    "oauth2_scheme",
    "oauth2_scheme_optional"
]