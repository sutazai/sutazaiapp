"""
Authentication dependencies for FastAPI
Provides dependency injection for authenticated routes
"""

from typing import Optional, Annotated, List, Callable
from datetime import datetime, timezone

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, update

from app.core.database import get_db
from .jwt_handler import verify_token
from .models import User, TokenData


# Security scheme
security = HTTPBearer(auto_error=False)


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get current authenticated user from JWT token
    
    Args:
        credentials: HTTP Bearer token from request
        db: Database session
        
    Returns:
        User object if authenticated, None otherwise
        
    Raises:
        HTTPException: If token is invalid or user not found
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Verify and decode token
        payload = verify_token(credentials.credentials, token_type="access")
        
        # Extract user ID
        user_id = int(payload.get("sub"))
        if user_id is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Get user from database
        result = await db.execute(
            select(User).where(User.id == user_id)
        )
        user = result.scalar_one_or_none()
        
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Update last login time
        await db.execute(
            update(User)
            .where(User.id == user_id)
            .values(last_login=datetime.now(timezone.utc))
        )
        await db.commit()
        
        return user
        
    except ValueError as e:
        # Token verification failed
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    except Exception as e:
        # Other errors
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )


async def get_current_active_user(
    current_user: Annotated[User, Depends(get_current_user)]
) -> User:
    """
    Get current active user
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        User object if active
        
    Raises:
        HTTPException: If user is inactive or locked
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Inactive user account"
        )
    
    # Check if account is locked
    if current_user.locked_until:
        if current_user.locked_until > datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Account locked until {current_user.locked_until}"
            )
    
    return current_user


async def require_admin(
    current_user: Annotated[User, Depends(get_current_active_user)]
) -> User:
    """
    Require admin privileges
    
    Args:
        current_user: Current active user
        
    Returns:
        User object if admin
        
    Raises:
        HTTPException: If user is not admin
    """
    if not current_user.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required"
        )
    
    return current_user


async def get_optional_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
    db: AsyncSession = Depends(get_db)
) -> Optional[User]:
    """
    Get current user if authenticated, None otherwise
    Used for endpoints that work both authenticated and anonymous
    
    Args:
        credentials: Optional HTTP Bearer token
        db: Database session
        
    Returns:
        User object if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    try:
        # Try to get user, but don't raise exception if it fails
        user = await get_current_user(credentials, db)
        return user
    except HTTPException:
        return None


def require_permissions(permissions: List[str]) -> Callable:
    """
    Create a dependency that requires specific permissions
    
    Args:
        permissions: List of required permissions (e.g., ["hardware:optimize", "system:monitor"])
        
    Returns:
        Dependency function that validates permissions
        
    Example:
        @router.post("/optimize")
        async def optimize(user = Depends(require_permissions(["hardware:optimize"]))):
            # Only users with hardware:optimize permission can access this
            pass
    """
    def permission_dependency(
        current_user: Annotated[User, Depends(get_current_active_user)]
    ) -> User:
        """
        Validate user has required permissions
        
        Args:
            current_user: Current authenticated active user
            
        Returns:
            User object if permissions are satisfied
            
        Raises:
            HTTPException: If user lacks required permissions
        """
        # Admin users have all permissions
        if current_user.is_admin:
            return current_user
        
        # Check user permissions
        user_permissions = getattr(current_user, 'permissions', [])
        
        # If user_permissions is a string, convert to list
        if isinstance(user_permissions, str):
            user_permissions = [user_permissions]
        elif user_permissions is None:
            user_permissions = []
        
        # Check if user has all required permissions
        missing_permissions = []
        for required_permission in permissions:
            # Check for exact match or wildcard match
            if not any(
                user_perm == required_permission or 
                user_perm.endswith('*') and required_permission.startswith(user_perm[:-1])
                for user_perm in user_permissions
            ):
                missing_permissions.append(required_permission)
        
        if missing_permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required permissions: {', '.join(missing_permissions)}"
            )
        
        return current_user
    
    return permission_dependency


def require_any_permission(permissions: List[str]) -> Callable:
    """
    Create a dependency that requires ANY of the specified permissions
    
    Args:
        permissions: List of permissions (user needs only one)
        
    Returns:
        Dependency function that validates permissions
    """
    def permission_dependency(
        current_user: Annotated[User, Depends(get_current_active_user)]
    ) -> User:
        """Validate user has at least one required permission"""
        # Admin users have all permissions
        if current_user.is_admin:
            return current_user
        
        # Check user permissions
        user_permissions = getattr(current_user, 'permissions', [])
        
        # If user_permissions is a string, convert to list
        if isinstance(user_permissions, str):
            user_permissions = [user_permissions]
        elif user_permissions is None:
            user_permissions = []
        
        # Check if user has any of the required permissions
        for required_permission in permissions:
            if any(
                user_perm == required_permission or 
                user_perm.endswith('*') and required_permission.startswith(user_perm[:-1])
                for user_perm in user_permissions
            ):
                return current_user
        
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Requires one of: {', '.join(permissions)}"
        )
    
    return permission_dependency