#!/usr/bin/env python3
"""
Authentication Dependencies for FastAPI
Provides dependency injection for user authentication and authorization
"""

from typing import Optional
from fastapi import HTTPException, status, Depends
from sqlalchemy.orm import Session
from backend.database import get_session
from backend.database.models import User, UserRole
from .jwt_handler import get_current_user

def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """Get current active user dependency"""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Inactive user"
        )
    return current_user


def require_admin(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require admin role dependency"""
    if current_user.role != UserRole.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def require_verified_user(
    current_user: User = Depends(get_current_active_user)
) -> User:
    """Require verified user dependency"""
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Email verification required"
        )
    return current_user


def optional_user(
    db: Session = Depends(get_session),
    current_user: Optional[User] = None
) -> Optional[User]:
    """Optional user dependency for public endpoints"""
    try:
        return get_current_user()
    except HTTPException:
        return None