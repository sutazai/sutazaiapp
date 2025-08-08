"""
Authentication API Router for SUTAZAI
Provides endpoints for user registration, login, and token management
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status, Body
from fastapi.security import HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import get_db
from .models import UserCreate, UserLogin, Token, UserResponse, PasswordChange
from .service import AuthService
from .dependencies import get_current_active_user, require_admin


router = APIRouter(
    prefix="/api/v1/auth",
    tags=["Authentication"],
    responses={
        401: {"description": "Unauthorized"},
        403: {"description": "Forbidden"},
    }
)

security = HTTPBearer()


@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(
    user_data: UserCreate,
    db: AsyncSession = Depends(get_db)
):
    """
    Register a new user account
    
    - **username**: Unique username (3-50 characters, alphanumeric + underscore/hyphen)
    - **email**: Valid email address
    - **password**: Strong password (min 8 chars, must include upper, lower, digit, special)
    """
    try:
        user = await AuthService.register_user(db, user_data)
        return UserResponse.model_validate(user)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed"
        )


@router.post("/login", response_model=Token)
async def login(
    login_data: UserLogin,
    db: AsyncSession = Depends(get_db)
):
    """
    Login with username and password
    
    Returns JWT access token and refresh token
    """
    # Authenticate user
    user = await AuthService.authenticate_user(db, login_data)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    
    # Create and return tokens
    try:
        token_response = await AuthService.create_token_response(user)
        return token_response
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create authentication token"
        )


@router.post("/refresh", response_model=Token)
async def refresh_token(
    refresh_token: str = Body(..., embed=True),
    db: AsyncSession = Depends(get_db)
):
    """
    Refresh access token using refresh token
    
    - **refresh_token**: Valid refresh token obtained from login
    """
    try:
        token_response = await AuthService.refresh_user_token(db, refresh_token)
        return token_response
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Get current authenticated user information
    
    Requires valid JWT access token
    """
    return UserResponse.model_validate(current_user)


@router.post("/change-password", response_model=dict)
async def change_password(
    password_data: PasswordChange,
    current_user: Annotated[UserResponse, Depends(get_current_active_user)],
    db: AsyncSession = Depends(get_db)
):
    """
    Change current user's password
    
    - **current_password**: Current password for verification
    - **new_password**: New password (must meet security requirements)
    """
    try:
        success = await AuthService.change_password(
            db,
            current_user,
            password_data.current_password,
            password_data.new_password
        )
        
        if success:
            return {"message": "Password changed successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Failed to change password"
            )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )


@router.post("/logout", response_model=dict)
async def logout(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Logout current user
    
    Note: With JWT, logout is typically handled client-side by removing the token.
    This endpoint can be used for server-side logging or token blacklisting if implemented.
    """
    # In a production system, you might want to:
    # 1. Add the token to a blacklist
    # 2. Log the logout event
    # 3. Clear any server-side session data
    
    return {"message": "Successfully logged out"}


@router.get("/users", response_model=list[UserResponse])
async def list_users(
    current_user: Annotated[UserResponse, Depends(require_admin)],
    db: AsyncSession = Depends(get_db),
    skip: int = 0,
    limit: int = 100
):
    """
    List all users (Admin only)
    
    - **skip**: Number of users to skip (pagination)
    - **limit**: Maximum number of users to return
    """
    from sqlalchemy import select
    from .models import User
    
    result = await db.execute(
        select(User)
        .offset(skip)
        .limit(limit)
    )
    users = result.scalars().all()
    
    return [UserResponse.model_validate(user) for user in users]


@router.put("/users/{user_id}/admin", response_model=dict)
async def update_user_admin_status(
    user_id: int,
    current_user: Annotated[UserResponse, Depends(require_admin)],
    db: AsyncSession = Depends(get_db),
    is_admin: bool = Body(..., embed=True)
):
    """
    Update user admin status (Admin only)
    
    - **user_id**: ID of user to update
    - **is_admin**: New admin status
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot modify your own admin status"
        )
    
    success = await AuthService.update_user_admin_status(db, user_id, is_admin)
    
    if success:
        return {"message": f"User admin status updated to {is_admin}"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )


@router.delete("/users/{user_id}", response_model=dict)
async def deactivate_user(
    user_id: int,
    current_user: Annotated[UserResponse, Depends(require_admin)],
    db: AsyncSession = Depends(get_db)
):
    """
    Deactivate a user account (Admin only)
    
    - **user_id**: ID of user to deactivate
    """
    if user_id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Cannot deactivate your own account"
        )
    
    from sqlalchemy import update
    from .models import User
    
    result = await db.execute(
        update(User)
        .where(User.id == user_id)
        .values(is_active=False)
    )
    await db.commit()
    
    if result.rowcount > 0:
        return {"message": "User account deactivated"}
    else:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="User not found"
        )


@router.get("/verify", response_model=dict)
async def verify_token(
    current_user: Annotated[UserResponse, Depends(get_current_active_user)]
):
    """
    Verify if the current JWT token is valid
    
    Returns user info if token is valid
    """
    return {
        "valid": True,
        "user_id": current_user.id,
        "username": current_user.username,
        "is_admin": current_user.is_admin
    }