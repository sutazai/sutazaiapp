"""
Authentication endpoints for user registration, login, and token management
Implements OAuth2 with Password Flow for authentication
"""

from typing import Annotated
from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_
from datetime import datetime, timezone, timedelta
import logging

from app.core.database import get_db
from app.core.security import security
from app.core.config import settings
from app.models.user import (
    User, UserCreate, UserResponse, UserLogin,
    Token, PasswordReset, PasswordResetConfirm
)
from app.api.dependencies.auth import (
    get_current_user, get_current_active_user,
    strict_rate_limiter
)
from app.services.email import email_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Register a new user
    
    Args:
        user_data: User registration data
        db: Database session
        
    Returns:
        Created user information
        
    Raises:
        HTTPException: If username or email already exists
    """
    # Check if user already exists
    result = await db.execute(
        select(User).where(
            or_(
                User.email == user_data.email,
                User.username == user_data.username
            )
        )
    )
    existing_user = result.scalar_one_or_none()
    
    if existing_user:
        if existing_user.email == user_data.email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email already registered"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already taken"
            )
    
    # Create new user
    db_user = User(
        email=user_data.email,
        username=user_data.username,
        full_name=user_data.full_name,
        hashed_password=security.get_password_hash(user_data.password),
        is_active=user_data.is_active,
        is_superuser=user_data.is_superuser,
        is_verified=False,  # Require email verification
        created_at=datetime.now(timezone.utc)
    )
    
    db.add(db_user)
    await db.commit()
    await db.refresh(db_user)
    
    logger.info(f"New user registered: {db_user.username} ({db_user.email})")
    
    # Generate and send verification email
    verification_token = security.generate_email_verification_token(db_user.email)
    email_sent = await email_service.send_verification_email(db_user.email, verification_token)
    
    if email_sent:
        logger.info(f"Verification email sent to {db_user.email}")
    else:
        logger.warning(f"Failed to send verification email to {db_user.email}, token: {verification_token}")
    
    return UserResponse.model_validate(db_user)


@router.post("/login", response_model=Token)
async def login(
    form_data: Annotated[OAuth2PasswordRequestForm, Depends()],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    OAuth2 compatible token login
    
    Args:
        form_data: OAuth2 password request form (username and password)
        db: Database session
        
    Returns:
        Access and refresh tokens
        
    Raises:
        HTTPException: If authentication fails
    """
    # Find user by username or email
    result = await db.execute(
        select(User).where(
            or_(
                User.username == form_data.username,
                User.email == form_data.username
            )
        )
    )
    user = result.scalar_one_or_none()
    
    # Check if user exists and password is correct
    if not user or not security.verify_password(form_data.password, user.hashed_password):
        # Increment failed login attempts if user exists
        if user:
            user.failed_login_attempts += 1
            
            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.account_locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
                await db.commit()
                
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Account locked due to multiple failed login attempts. Try again in 30 minutes."
                )
            
            await db.commit()
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if account is locked
    if user.account_locked_until and user.account_locked_until > datetime.now(timezone.utc):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Account locked until {user.account_locked_until}"
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account"
        )
    
    # Create access and refresh tokens
    access_token_data = {
        "sub": user.username,
        "user_id": user.id,
        "email": user.email
    }
    
    access_token = security.create_access_token(data=access_token_data)
    refresh_token = security.create_refresh_token(data=access_token_data)
    
    # Reset failed login attempts and update last login
    user.failed_login_attempts = 0
    user.account_locked_until = None
    user.last_login = datetime.now(timezone.utc)
    user.refresh_token = refresh_token  # Store refresh token (optional)
    
    await db.commit()
    
    logger.info(f"User logged in: {user.username}")
    
    return Token(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer",
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
    )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    db: Annotated[AsyncSession, Depends(get_db)],
    refresh_token: str = Body(..., embed=True)
):
    """
    Refresh access token using refresh token
    
    Args:
        refresh_token: Valid refresh token
        db: Database session
        
    Returns:
        New access and refresh tokens
        
    Raises:
        HTTPException: If refresh token is invalid
    """
    try:
        # Verify refresh token
        payload = security.verify_token(refresh_token, token_type="refresh")
        username = payload.get("sub")
        user_id = payload.get("user_id")
        
        if not username or not user_id:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid refresh token"
            )
        
        # Get user from database
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if not user or not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found or inactive"
            )
        
        # Create new tokens
        access_token_data = {
            "sub": user.username,
            "user_id": user.id,
            "email": user.email
        }
        
        new_access_token = security.create_access_token(data=access_token_data)
        new_refresh_token = security.create_refresh_token(data=access_token_data)
        
        # Update refresh token in database
        user.refresh_token = new_refresh_token
        await db.commit()
        
        return Token(
            access_token=new_access_token,
            refresh_token=new_refresh_token,
            token_type="bearer",
            expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Token refresh error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid refresh token"
        )


@router.post("/logout")
async def logout(
    current_user: Annotated[User, Depends(get_current_active_user)],
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Logout current user (invalidate refresh token)
    
    Args:
        current_user: Current authenticated user
        db: Database session
        
    Returns:
        Success message
    """
    # Clear refresh token
    current_user.refresh_token = None
    await db.commit()
    
    logger.info(f"User logged out: {current_user.username}")
    
    return {"message": "Successfully logged out"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[User, Depends(get_current_active_user)]
):
    """
    Get current user information
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current user information
    """
    return UserResponse.model_validate(current_user)


@router.post("/password-reset", dependencies=[Depends(strict_rate_limiter)])
async def request_password_reset(
    reset_data: PasswordReset,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Request password reset token
    Rate limited to prevent abuse
    
    Args:
        reset_data: Email address for reset
        db: Database session
        
    Returns:
        Success message (always returns success to prevent email enumeration)
    """
    # Find user by email
    result = await db.execute(
        select(User).where(User.email == reset_data.email)
    )
    user = result.scalar_one_or_none()
    
    if user:
        # Generate reset token
        reset_token = security.generate_password_reset_token(user.email)
        
        # Send password reset email
        email_sent = await email_service.send_password_reset_email(user.email, reset_token)
        
        if email_sent:
            logger.info(f"Password reset email sent to {user.email}")
        else:
            logger.warning(f"Failed to send password reset email to {user.email}, but token generated: {reset_token}")
    
    # Always return success to prevent email enumeration
    return {
        "message": "If the email exists, a password reset link has been sent"
    }


@router.post("/password-reset/confirm")
async def confirm_password_reset(
    reset_confirm: PasswordResetConfirm,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Confirm password reset with token
    
    Args:
        reset_confirm: Reset token and new password
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If token is invalid or expired
    """
    # Verify reset token
    email = security.verify_password_reset_token(reset_confirm.token)
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token"
        )
    
    # Find user and update password
    result = await db.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    # Update password
    user.hashed_password = security.get_password_hash(reset_confirm.new_password)
    user.password_changed_at = datetime.now(timezone.utc)
    user.failed_login_attempts = 0
    user.account_locked_until = None
    
    await db.commit()
    
    logger.info(f"Password reset for user: {user.username}")
    
    return {"message": "Password successfully reset"}


@router.post("/verify-email/{token}")
async def verify_email(
    token: str,
    db: Annotated[AsyncSession, Depends(get_db)]
):
    """
    Verify user email with token
    
    Args:
        token: Email verification token
        db: Database session
        
    Returns:
        Success message
        
    Raises:
        HTTPException: If token is invalid
    """
    # Verify the token and extract email
    email = security.verify_email_token(token)
    
    if not email:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token"
        )
    
    # Find user by email
    result = await db.execute(
        select(User).where(User.email == email)
    )
    user = result.scalar_one_or_none()
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )
    
    if user.is_verified:
        return {"message": "Email already verified"}
    
    # Mark user as verified
    user.is_verified = True
    user.verified_at = datetime.now(timezone.utc)
    
    await db.commit()
    await db.refresh(user)
    
    logger.info(f"Email verified for user: {user.username} ({user.email})")
    
    return {"message": "Email successfully verified"}