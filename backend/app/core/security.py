"""
Security utilities for JWT authentication and password hashing
Implements OAuth2 with Password and Bearer for secure authentication
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
import secrets
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# Password hashing context with bcrypt
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Generate a secure secret key if not set properly
if settings.SECRET_KEY == "sutazai-secret-key-2025-change-in-production":
    # Generate a cryptographically secure secret key
    REAL_SECRET_KEY = secrets.token_urlsafe(32)
    logger.warning(f"Generated new SECRET_KEY for JWT. Add this to .env: SECRET_KEY={REAL_SECRET_KEY}")
else:
    REAL_SECRET_KEY = settings.SECRET_KEY


class SecurityUtils:
    """Utility class for security operations"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a plain password against a hashed password
        
        Args:
            plain_password: The plain text password
            hashed_password: The bcrypt hashed password
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """
        Hash a password using bcrypt
        
        Args:
            password: Plain text password
            
        Returns:
            Bcrypt hashed password
        """
        return pwd_context.hash(password)
    
    @staticmethod
    def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT access token
        
        Args:
            data: The payload data to encode
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT token
        """
        to_encode = data.copy()
        
        # Set expiration time
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        
        # Add standard JWT claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        })
        
        # Create the JWT token
        encoded_jwt = jwt.encode(
            to_encode, 
            REAL_SECRET_KEY, 
            algorithm=settings.ALGORITHM
        )
        
        return encoded_jwt
    
    @staticmethod
    def create_refresh_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """
        Create a JWT refresh token with longer expiration
        
        Args:
            data: The payload data to encode
            expires_delta: Optional custom expiration time
            
        Returns:
            Encoded JWT refresh token
        """
        to_encode = data.copy()
        
        # Refresh tokens expire in 7 days by default
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(days=7)
        
        # Add standard JWT claims
        to_encode.update({
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh"
        })
        
        # Create the JWT token
        encoded_jwt = jwt.encode(
            to_encode, 
            REAL_SECRET_KEY, 
            algorithm=settings.ALGORITHM
        )
        
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
        """
        Verify and decode a JWT token
        
        Args:
            token: The JWT token to verify
            token_type: Expected token type ("access" or "refresh")
            
        Returns:
            Decoded token payload
            
        Raises:
            HTTPException: If token is invalid or expired
        """
        credentials_exception = HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
        
        try:
            # Decode the token
            payload = jwt.decode(
                token, 
                REAL_SECRET_KEY, 
                algorithms=[settings.ALGORITHM]
            )
            
            # Verify token type
            if payload.get("type") != token_type:
                raise credentials_exception
            
            # Check if token has required fields
            if "sub" not in payload:
                raise credentials_exception
                
            return payload
            
        except JWTError as e:
            logger.error(f"JWT verification error: {e}")
            raise credentials_exception
        except Exception as e:
            logger.error(f"Token verification error: {e}")
            raise credentials_exception
    
    @staticmethod
    def generate_password_reset_token(email: str) -> str:
        """
        Generate a password reset token
        
        Args:
            email: User's email address
            
        Returns:
            Password reset token
        """
        data = {"sub": email, "purpose": "password_reset"}
        return SecurityUtils.create_access_token(
            data=data,
            expires_delta=timedelta(hours=1)  # Reset tokens expire in 1 hour
        )
    
    @staticmethod
    def verify_password_reset_token(token: str) -> Optional[str]:
        """
        Verify a password reset token
        
        Args:
            token: The reset token to verify
            
        Returns:
            Email address if valid, None otherwise
        """
        try:
            payload = SecurityUtils.verify_token(token, token_type="access")
            
            if payload.get("purpose") != "password_reset":
                return None
                
            email: str = payload.get("sub")
            return email
            
        except HTTPException:
            return None


# Create a global security instance
security = SecurityUtils()