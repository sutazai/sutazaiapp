"""
Security utilities for JWT authentication and password hashing
Implements OAuth2 with Password and Bearer for secure authentication
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Tuple
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status
import logging
import re
from app.core.config import settings

logger = logging.getLogger(__name__)

# Password hashing context with bcrypt
# Set ident="2b" to avoid passlib wrap bug detection issues
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__ident="2b",  # Use 2b format to avoid wrap bug detection
    bcrypt__rounds=12     # Standard security rounds
)

# JWT secret key is now managed through the secrets manager
# The settings.SECRET_KEY property handles generation and warnings


class SecurityUtils:
    """Utility class for security operations"""
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """
        Verify a plain password against a hashed password
        Truncates password to 72 bytes to match hashing behavior
        
        Args:
            plain_password: The plain text password
            hashed_password: The bcrypt hashed password
            
        Returns:
            True if password matches, False otherwise
        """
        try:
            # Truncate to 72 bytes to match get_password_hash behavior
            password_bytes = plain_password.encode('utf-8')[:72]
            truncated_password = password_bytes.decode('utf-8', errors='ignore')
            return pwd_context.verify(truncated_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @staticmethod
    def get_password_hash(password: str) -> str:
        """
        Hash a password using bcrypt
        Truncates password to 72 bytes to comply with bcrypt limitations
        
        Args:
            password: Plain text password
            
        Returns:
            Bcrypt hashed password
        """
        # Bcrypt has a 72-byte limit, truncate password if needed
        # This is safe as 72 bytes provides sufficient entropy
        password_bytes = password.encode('utf-8')[:72]
        truncated_password = password_bytes.decode('utf-8')
        return pwd_context.hash(truncated_password)
    
    @staticmethod
    def validate_password_strength(password: str) -> Tuple[bool, Optional[str]]:
        """
        Validate password strength according to security requirements
        
        Requirements:
        - Minimum 8 characters
        - At least one uppercase letter
        - At least one lowercase letter
        - At least one digit
        - At least one special character
        
        Args:
            password: Plain text password to validate
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if not password:
            return False, "Password cannot be empty"
        
        if len(password) < 8:
            return False, "Password must be at least 8 characters long"
        
        if len(password) > 128:
            return False, "Password must be no more than 128 characters long"
        
        # Check for at least one uppercase letter
        if not re.search(r'[A-Z]', password):
            return False, "Password must contain at least one uppercase letter"
        
        # Check for at least one lowercase letter
        if not re.search(r'[a-z]', password):
            return False, "Password must contain at least one lowercase letter"
        
        # Check for at least one digit
        if not re.search(r'\d', password):
            return False, "Password must contain at least one digit"
        
        # Check for at least one special character
        if not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            return False, "Password must contain at least one special character"
        
        # Check for common weak passwords
        common_weak_passwords = [
            'password', 'password123', '12345678', 'qwerty', 'abc123',
            '123456789', 'letmein', 'welcome', 'monkey', '1q2w3e4r'
        ]
        if password.lower() in common_weak_passwords:
            return False, "Password is too common, please choose a stronger password"
        
        return True, None
    
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
            settings.SECRET_KEY, 
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
            settings.SECRET_KEY, 
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
                settings.SECRET_KEY, 
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

    
    @staticmethod
    def generate_email_verification_token(email: str) -> str:
        """
        Generate email verification token
        
        Args:
            email: User's email address
            
        Returns:
            JWT token for email verification
        """
        expire = datetime.now(timezone.utc) + timedelta(hours=24)  # 24 hour expiry
        data = {
            "sub": email,
            "type": "email_verification",
            "exp": expire
        }
        
        encoded_jwt = jwt.encode(data, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_email_token(token: str) -> Optional[str]:
        """
        Verify email verification token
        
        Args:
            token: JWT token to verify
            
        Returns:
            Email address if valid, None otherwise
        """
        try:
            payload = jwt.decode(
                token, 
                settings.SECRET_KEY, 
                algorithms=[settings.ALGORITHM]
            )
            
            # Check token type
            if payload.get("type") != "email_verification":
                return None
            
            email: str = payload.get("sub")
            return email
            
        except JWTError as e:
            logger.error(f"Email verification token error: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error verifying email token: {e}")
            return None


# Create a global security instance
security = SecurityUtils()