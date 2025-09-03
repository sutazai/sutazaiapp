"""
Authentication dependencies for FastAPI
Provides dependency injection for authentication and authorization
"""

from typing import Optional, Annotated
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from jose import JWTError
import logging
from datetime import datetime, timezone
import time

from app.core.security import security
from app.core.database import get_db
from app.models.user import User, TokenData
from app.services.connections import service_connections

logger = logging.getLogger(__name__)

# OAuth2 scheme for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

# Optional OAuth2 scheme (for endpoints that work with or without auth)
oauth2_scheme_optional = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login", auto_error=False)


async def get_current_user(
    token: Annotated[str, Depends(oauth2_scheme)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> User:
    """
    Get current authenticated user from JWT token
    
    Args:
        token: JWT access token from Authorization header
        db: Database session
        
    Returns:
        Current authenticated user
        
    Raises:
        HTTPException: If authentication fails
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        # Verify and decode the token
        payload = security.verify_token(token, token_type="access")
        username: str = payload.get("sub")
        user_id: int = payload.get("user_id")
        
        if username is None or user_id is None:
            raise credentials_exception
            
        # Create token data
        token_data = TokenData(username=username, user_id=user_id)
        
    except HTTPException:
        raise credentials_exception
    except Exception as e:
        logger.error(f"Token validation error: {e}")
        raise credentials_exception
    
    # Get user from database
    result = await db.execute(
        select(User).where(User.id == token_data.user_id)
    )
    user = result.scalar_one_or_none()
    
    if user is None:
        raise credentials_exception
    
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
            detail="Inactive user"
        )
    
    # Update last login time
    user.last_login = datetime.now(timezone.utc)
    await db.commit()
    await db.refresh(user)
    
    return user


async def get_current_user_optional(
    token: Annotated[Optional[str], Depends(oauth2_scheme_optional)],
    db: Annotated[AsyncSession, Depends(get_db)]
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise
    Used for endpoints that work with or without authentication
    
    Args:
        token: Optional JWT access token
        db: Database session
        
    Returns:
        Current user or None
    """
    if not token:
        return None
        
    try:
        return await get_current_user(token, db)
    except HTTPException:
        return None


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Get current active user
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Current active user
        
    Raises:
        HTTPException: If user is not active
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user"
        )
    return current_user


async def get_current_verified_user(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    """
    Get current verified user
    
    Args:
        current_user: Current active user
        
    Returns:
        Current verified user
        
    Raises:
        HTTPException: If user is not verified
    """
    if not current_user.is_verified:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Please verify your email first"
        )
    return current_user


async def get_current_superuser(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    """
    Get current superuser
    
    Args:
        current_user: Current active user
        
    Returns:
        Current superuser
        
    Raises:
        HTTPException: If user is not a superuser
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )
    return current_user


class RateLimiter:
    """
    Rate limiter dependency for protecting endpoints using Redis sliding window
    """
    def __init__(self, calls: int = 10, period: int = 60):
        """
        Initialize rate limiter
        
        Args:
            calls: Number of allowed calls
            period: Time period in seconds
        """
        self.calls = calls
        self.period = period
    
    async def __call__(self, request: Request, user: User = Depends(get_current_user_optional)) -> bool:
        """
        Check if request should be rate limited using Redis sliding window algorithm
        
        Args:
            request: FastAPI request object for IP extraction
            user: Optional current user
            
        Returns:
            True if allowed, raises exception if rate limited
        """
        # Get Redis connection from ServiceConnections singleton
        redis_client = service_connections.redis_client
        if not redis_client:
            # If Redis is unavailable, allow request but log warning
            logger.warning("Redis unavailable for rate limiting - allowing request")
            return True
        
        # Get identifier (user ID or IP address)
        if user:
            identifier = f"rate_limit:user:{user.id}"
        else:
            # Extract IP from request
            client_ip = request.client.host if request.client else "unknown"
            identifier = f"rate_limit:ip:{client_ip}"
        
        try:
            current_time = time.time()
            window_start = current_time - self.period
            
            # Use Redis sorted set for sliding window
            # Remove expired entries
            await redis_client.zremrangebyscore(identifier, 0, window_start)
            
            # Count current entries in window
            current_count = await redis_client.zcard(identifier)
            
            if current_count >= self.calls:
                # Calculate time until oldest entry expires
                oldest_entry = await redis_client.zrange(identifier, 0, 0, withscores=True)
                if oldest_entry:
                    reset_time = oldest_entry[0][1] + self.period
                    retry_after = int(reset_time - current_time)
                else:
                    retry_after = self.period
                
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail=f"Rate limit exceeded. Try again in {retry_after} seconds",
                    headers={"Retry-After": str(retry_after)}
                )
            
            # Add current request to the window
            await redis_client.zadd(identifier, {str(current_time): current_time})
            
            # Set expiry on the key
            await redis_client.expire(identifier, self.period)
            
            return True
            
        except HTTPException:
            # Re-raise HTTP exceptions
            raise
        except Exception as e:
            # Log Redis errors but don't block requests
            logger.error(f"Rate limiter Redis error: {e}")
            return True


# Dependency instances
rate_limiter = RateLimiter(calls=100, period=60)  # 100 calls per minute
strict_rate_limiter = RateLimiter(calls=5, period=60)  # 5 calls per minute for sensitive endpoints