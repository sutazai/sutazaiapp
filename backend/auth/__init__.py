"""
Authentication package for SutazAI application
Contains JWT handling, OTP validation, and security utilities
"""

from .jwt_handler import create_access_token, verify_token, get_current_user
from .otp_handler import generate_otp, validate_otp, send_otp_email
from .dependencies import get_current_active_user, require_admin
from .security import hash_password, verify_password

__all__ = [
    "create_access_token",
    "verify_token", 
    "get_current_user",
    "generate_otp",
    "validate_otp",
    "send_otp_email",
    "get_current_active_user",
    "require_admin",
    "hash_password",
    "verify_password"
]