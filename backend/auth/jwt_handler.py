#!/usr/bin/env python3
"""
JWT Token Handling for SutazAI Authentication
Manages creation, validation, and decoding of JWT tokens
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
import jwt
from jwt.exceptions import InvalidTokenError
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
from backend.config import Config
from backend.database import get_session
from backend.database.models import User, UserSession
from loguru import logger

# Security scheme
security = HTTPBearer()
config = Config()


def create_access_token(
    data: Dict[str, Any], 
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create JWT access token"""
    to_encode = data.copy()
    
    if expires_delta:
        expire = datetime.now(timezone.utc) + expires_delta
    else:
        expire = datetime.now(timezone.utc) + timedelta(
            minutes=config.security.access_token_expire_minutes
        )
    
    to_encode.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    
    try:
        encoded_jwt = jwt.encode(
            to_encode, 
            config.security.secret_key, 
            algorithm=config.security.algorithm
        )
        return encoded_jwt
    except Exception as e:
        logger.error(f"Failed to create access token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create access token"
        )


def create_refresh_token(user_id: int) -> str:
    """Create JWT refresh token"""
    data = {
        "sub": str(user_id),
        "type": "refresh"
    }
    
    expire = datetime.now(timezone.utc) + timedelta(
        days=config.security.refresh_token_expire_days
    )
    data.update({"exp": expire, "iat": datetime.now(timezone.utc)})
    
    try:
        encoded_jwt = jwt.encode(
            data,
            config.security.secret_key,
            algorithm=config.security.algorithm
        )
        return encoded_jwt
    except Exception as e:
        logger.error(f"Failed to create refresh token: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Could not create refresh token"
        )


def verify_token(token: str) -> Dict[str, Any]:
    """Verify and decode JWT token"""
    try:
        payload = jwt.decode(
            token,
            config.security.secret_key,
            algorithms=[config.security.algorithm]
        )
        
        # Check if token has expired
        exp = payload.get("exp")
        if exp is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token missing expiration"
            )
        
        if datetime.fromtimestamp(exp, tz=timezone.utc) < datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired"
            )
        
        return payload
    
    except InvalidTokenError as e:
        logger.warning(f"Invalid token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        logger.error(f"Token verification error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_session)
) -> User:
    """Get current user from JWT token"""
    token = credentials.credentials
    
    try:
        payload = verify_token(token)
        user_id: str = payload.get("sub")
        
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials"
            )
        
        # Get user from database
        user = db.query(User).filter(User.id == int(user_id)).first()
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found"
            )
        
        # Check if user is active
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Inactive user"
            )
        
        # Check if user account is locked
        if user.is_locked():
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Account is locked due to too many failed login attempts"
            )
        
        return user
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials"
        )


def create_user_session(
    db: Session,
    user: User,
    access_token: str,
    refresh_token: str,
    ip_address: str = None,
    user_agent: str = None
) -> UserSession:
    """Create a new user session"""
    
    # Calculate token expiration
    expires_at = datetime.now(timezone.utc) + timedelta(
        minutes=config.security.access_token_expire_minutes
    )
    
    session = UserSession(
        user_id=user.id,
        session_token=access_token,
        refresh_token=refresh_token,
        expires_at=expires_at,
        ip_address=ip_address,
        user_agent=user_agent
    )
    
    db.add(session)
    db.commit()
    db.refresh(session)
    
    logger.info(f"Created session for user {user.username}")
    return session


def invalidate_user_session(db: Session, session_token: str) -> bool:
    """Invalidate a user session"""
    session = db.query(UserSession).filter(
        UserSession.session_token == session_token
    ).first()
    
    if session:
        session.is_active = False
        db.commit()
        logger.info(f"Invalidated session for user ID {session.user_id}")
        return True
    
    return False


def invalidate_all_user_sessions(db: Session, user_id: int) -> int:
    """Invalidate all sessions for a user"""
    count = db.query(UserSession).filter(
        UserSession.user_id == user_id,
        UserSession.is_active == True
    ).update({"is_active": False})
    
    db.commit()
    logger.info(f"Invalidated {count} sessions for user ID {user_id}")
    return count


def refresh_access_token(
    refresh_token: str,
    db: Session
) -> Dict[str, str]:
    """Refresh access token using refresh token"""
    try:
        payload = verify_token(refresh_token)
        
        # Check if it's a refresh token
        if payload.get("type") != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type"
            )
        
        user_id = int(payload.get("sub"))
        
        # Get user
        user = db.query(User).filter(User.id == user_id).first()
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Check if refresh token exists in database
        session = db.query(UserSession).filter(
            UserSession.refresh_token == refresh_token,
            UserSession.is_active == True
        ).first()
        
        if not session:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Create new tokens
        new_access_token = create_access_token({"sub": str(user.id)})
        new_refresh_token = create_refresh_token(user.id)
        
        # Update session
        session.session_token = new_access_token
        session.refresh_token = new_refresh_token
        session.expires_at = datetime.now(timezone.utc) + timedelta(
            minutes=config.security.access_token_expire_minutes
        )
        db.commit()
        
        return {
            "access_token": new_access_token,
            "refresh_token": new_refresh_token,
            "token_type": "bearer"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error refreshing token: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not refresh token"
        )