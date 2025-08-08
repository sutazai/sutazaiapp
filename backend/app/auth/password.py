"""
Password hashing and verification using bcrypt
Provides secure password management for SUTAZAI authentication
"""

import bcrypt
from typing import Union


def hash_password(password: str) -> str:
    """
    Hash a password using bcrypt with a random salt
    
    Args:
        password: Plain text password to hash
        
    Returns:
        Hashed password string
    """
    # Convert password to bytes
    password_bytes = password.encode('utf-8')
    
    # Generate salt and hash password
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password_bytes, salt)
    
    # Return as string for storage
    return hashed.decode('utf-8')


def verify_password(plain_password: str, hashed_password: Union[str, bytes]) -> bool:
    """
    Verify a plain password against a hashed password
    
    Args:
        plain_password: Plain text password to verify
        hashed_password: Previously hashed password to compare against
        
    Returns:
        True if password matches, False otherwise
    """
    try:
        # Convert to bytes if needed
        password_bytes = plain_password.encode('utf-8')
        
        if isinstance(hashed_password, str):
            hashed_bytes = hashed_password.encode('utf-8')
        else:
            hashed_bytes = hashed_password
        
        # Check password
        return bcrypt.checkpw(password_bytes, hashed_bytes)
    except Exception:
        # Any error in verification means password doesn't match
        return False


def is_password_strong(password: str) -> tuple[bool, str]:
    """
    Check if password meets security requirements
    
    Args:
        password: Password to check
        
    Returns:
        Tuple of (is_valid, message)
    """
    if len(password) < 8:
        return False, "Password must be at least 8 characters long"
    
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in password)
    
    if not has_upper:
        return False, "Password must contain at least one uppercase letter"
    if not has_lower:
        return False, "Password must contain at least one lowercase letter"
    if not has_digit:
        return False, "Password must contain at least one digit"
    if not has_special:
        return False, "Password must contain at least one special character"
    
    return True, "Password is strong"