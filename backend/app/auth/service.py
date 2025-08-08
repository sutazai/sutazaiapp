"""
Authentication service for SUTAZAI
Handles user registration, login, and authentication operations
"""

from typing import Optional
from datetime import datetime, timedelta, timezone

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update, or_
from sqlalchemy.exc import IntegrityError

from .models import User, UserCreate, UserLogin, Token
from .password import hash_password, verify_password, is_password_strong
from .jwt_handler import create_access_token, create_refresh_token, verify_token


class AuthService:
    """Service class for authentication operations"""
    
    @staticmethod
    async def register_user(
        db: AsyncSession,
        user_data: UserCreate
    ) -> User:
        """
        Register a new user
        
        Args:
            db: Database session
            user_data: User registration data
            
        Returns:
            Created user object
            
        Raises:
            ValueError: If username/email already exists or password is weak
        """
        # Check password strength
        is_strong, message = is_password_strong(user_data.password)
        if not is_strong:
            raise ValueError(message)
        
        # Check if user already exists
        result = await db.execute(
            select(User).where(
                or_(
                    User.username == user_data.username.lower(),
                    User.email == user_data.email.lower()
                )
            )
        )
        existing_user = result.scalar_one_or_none()
        
        if existing_user:
            if existing_user.username == user_data.username.lower():
                raise ValueError("Username already registered")
            else:
                raise ValueError("Email already registered")
        
        # Hash password
        hashed_password = hash_password(user_data.password)
        
        # Create new user
        new_user = User(
            username=user_data.username.lower(),
            email=user_data.email.lower(),
            password_hash=hashed_password,
            is_active=True,
            is_admin=False,  # Default to non-admin
            created_at=datetime.now(timezone.utc)
        )
        
        try:
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)
            return new_user
        except IntegrityError:
            await db.rollback()
            raise ValueError("User registration failed")
    
    @staticmethod
    async def authenticate_user(
        db: AsyncSession,
        login_data: UserLogin
    ) -> Optional[User]:
        """
        Authenticate a user with username/password
        
        Args:
            db: Database session
            login_data: Login credentials
            
        Returns:
            User object if authentication successful, None otherwise
        """
        # Find user by username (case-insensitive)
        result = await db.execute(
            select(User).where(
                User.username == login_data.username.lower()
            )
        )
        user = result.scalar_one_or_none()
        
        if not user:
            return None
        
        # Check if account is locked
        if user.locked_until and user.locked_until > datetime.now(timezone.utc):
            return None
        
        # Verify password
        if not verify_password(login_data.password, user.password_hash):
            # Increment failed login attempts
            user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
            
            # Lock account after 5 failed attempts
            if user.failed_login_attempts >= 5:
                user.locked_until = datetime.now(timezone.utc) + timedelta(minutes=30)
            
            await db.commit()
            return None
        
        # Reset failed login attempts on successful login
        user.failed_login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now(timezone.utc)
        
        await db.commit()
        return user
    
    @staticmethod
    async def create_token_response(user: User) -> Token:
        """
        Create JWT token response for authenticated user
        
        Args:
            user: Authenticated user object
            
        Returns:
            Token response with access and refresh tokens
        """
        # Create access token
        access_token = create_access_token(
            user_id=user.id,
            username=user.username,
            email=user.email,
            is_admin=user.is_admin if hasattr(user, 'is_admin') else False,
            scopes=["read", "write", "admin"] if getattr(user, 'is_admin', False) else ["read", "write"]
        )
        
        # Create refresh token
        refresh_token = create_refresh_token(user_id=user.id)
        
        # Create user response (without sensitive data)
        from .models import UserResponse
        user_response = UserResponse(
            id=user.id,
            username=user.username,
            email=user.email,
            is_active=user.is_active,
            is_admin=getattr(user, 'is_admin', False),
            created_at=user.created_at
        )
        
        return Token(
            access_token=access_token,
            refresh_token=refresh_token,
            token_type="bearer",
            expires_in=1800,  # 30 minutes
            user=user_response
        )
    
    @staticmethod
    async def refresh_user_token(
        db: AsyncSession,
        refresh_token: str
    ) -> Token:
        """
        Refresh access token using refresh token
        
        Args:
            db: Database session
            refresh_token: Valid refresh token
            
        Returns:
            New token response
            
        Raises:
            ValueError: If refresh token is invalid
        """
        try:
            # Verify refresh token
            payload = verify_token(refresh_token, token_type="refresh")
            user_id = int(payload.get("sub"))
            
            # Get user from database
            result = await db.execute(
                select(User).where(User.id == user_id)
            )
            user = result.scalar_one_or_none()
            
            if not user or not user.is_active:
                raise ValueError("Invalid user or account inactive")
            
            # Create new tokens
            return await AuthService.create_token_response(user)
            
        except Exception as e:
            raise ValueError(f"Failed to refresh token: {str(e)}")
    
    @staticmethod
    async def change_password(
        db: AsyncSession,
        user: User,
        current_password: str,
        new_password: str
    ) -> bool:
        """
        Change user password
        
        Args:
            db: Database session
            user: User object
            current_password: Current password for verification
            new_password: New password to set
            
        Returns:
            True if password changed successfully
            
        Raises:
            ValueError: If current password is wrong or new password is weak
        """
        # Verify current password
        if not verify_password(current_password, user.password_hash):
            raise ValueError("Current password is incorrect")
        
        # Check new password strength
        is_strong, message = is_password_strong(new_password)
        if not is_strong:
            raise ValueError(message)
        
        # Hash and update password
        hashed_password = hash_password(new_password)
        user.password_hash = hashed_password
        user.updated_at = datetime.now(timezone.utc)
        
        await db.commit()
        return True
    
    @staticmethod
    async def get_user_by_id(
        db: AsyncSession,
        user_id: int
    ) -> Optional[User]:
        """
        Get user by ID
        
        Args:
            db: Database session
            user_id: User ID
            
        Returns:
            User object if found
        """
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_user_by_email(
        db: AsyncSession,
        email: str
    ) -> Optional[User]:
        """
        Get user by email
        
        Args:
            db: Database session
            email: User email
            
        Returns:
            User object if found
        """
        result = await db.execute(
            select(User).where(User.email == email.lower())
        )
        return result.scalar_one_or_none()
    
    @staticmethod
    async def update_user_admin_status(
        db: AsyncSession,
        user_id: int,
        is_admin: bool
    ) -> bool:
        """
        Update user admin status (admin only operation)
        
        Args:
            db: Database session
            user_id: User ID to update
            is_admin: New admin status
            
        Returns:
            True if updated successfully
        """
        result = await db.execute(
            update(User)
            .where(User.id == user_id)
            .values(is_admin=is_admin, updated_at=datetime.now(timezone.utc))
        )
        await db.commit()
        return result.rowcount > 0