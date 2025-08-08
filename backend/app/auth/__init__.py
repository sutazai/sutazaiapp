"""
Authentication module for SUTAZAI system
Provides JWT-based authentication with secure password hashing
"""

from .jwt_handler import JWTHandler, create_access_token, create_refresh_token, verify_token
from .password import hash_password, verify_password
from .models import User, UserCreate, UserLogin, Token, TokenData
from .dependencies import get_current_user, get_current_active_user, require_admin

__all__ = [
    "JWTHandler",
    "create_access_token",
    "create_refresh_token",
    "verify_token",
    "hash_password",
    "verify_password",
    "User",
    "UserCreate",
    "UserLogin",
    "Token",
    "TokenData",
    "get_current_user",
    "get_current_active_user",
    "require_admin",
]