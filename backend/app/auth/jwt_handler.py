"""
JWT Token Handler for SUTAZAI Authentication System
Implements secure JWT token creation and validation
"""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any

import jwt
from jwt import PyJWTError

# JWT secret must be provided via environment variable
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY") or os.getenv("JWT_SECRET")
if not JWT_SECRET_KEY:
    raise ValueError(
        "JWT_SECRET_KEY or JWT_SECRET environment variable is required. "
        "Generate a secure secret with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
    )

JWT_ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7


class JWTHandler:
    """Handler for JWT token operations"""
    
    def __init__(self, secret_key: str = None, algorithm: str = None):
        if secret_key:
            self.secret_key = secret_key
        else:
            self.secret_key = JWT_SECRET_KEY
        self.algorithm = algorithm or JWT_ALGORITHM
        self.access_token_expire = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        self.refresh_token_expire = timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    
    def create_access_token(
        self,
        user_id: int,
        username: str,
        email: str,
        is_admin: bool = False,
        scopes: list = None,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new access token"""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + self.access_token_expire
        
        payload = {
            "sub": str(user_id),  # Subject (user ID)
            "username": username,
            "email": email,
            "is_admin": is_admin,
            "scopes": scopes or [],
            "type": "access",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "iss": "sutazai",  # Issuer
        }
        
        encoded_jwt = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def create_refresh_token(
        self,
        user_id: int,
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """Create a new refresh token"""
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + self.refresh_token_expire
        
        payload = {
            "sub": str(user_id),
            "type": "refresh",
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "iss": "sutazai",
        }
        
        encoded_jwt = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.secret_key,
                algorithms=[self.algorithm],
                issuer="sutazai"
            )
            
            # Verify token type
            if payload.get("type") != token_type:
                raise ValueError(f"Invalid token type. Expected {token_type}, got {payload.get('type')}")
            
            return payload
            
        except jwt.ExpiredSignatureError:
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            raise ValueError(f"Invalid token: {str(e)}")
    
    def refresh_access_token(self, refresh_token: str) -> tuple[str, str]:
        """Use refresh token to get new access token"""
        try:
            # Verify refresh token
            payload = self.verify_token(refresh_token, token_type="refresh")
            user_id = int(payload["sub"])
            
            # In production, you would fetch user details from database here
            # For now, we'll create a basic access token
            # This should be enhanced to fetch actual user data
            
            # Create new access token
            # Note: In production, fetch user details from database
            new_access_token = self.create_access_token(
                user_id=user_id,
                username="",  # Should be fetched from database
                email="",  # Should be fetched from database
                is_admin=False,  # Should be fetched from database
                scopes=[]  # Should be fetched from database
            )
            
            return new_access_token, refresh_token
            
        except Exception as e:
            raise ValueError(f"Failed to refresh token: {str(e)}")


# Global JWT handler instance
jwt_handler = JWTHandler()


# Convenience functions
def create_access_token(
    user_id: int,
    username: str,
    email: str,
    is_admin: bool = False,
    scopes: list = None,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create an access token"""
    return jwt_handler.create_access_token(
        user_id=user_id,
        username=username,
        email=email,
        is_admin=is_admin,
        scopes=scopes,
        expires_delta=expires_delta
    )


def create_refresh_token(
    user_id: int,
    expires_delta: Optional[timedelta] = None
) -> str:
    """Create a refresh token"""
    return jwt_handler.create_refresh_token(user_id=user_id, expires_delta=expires_delta)


def verify_token(token: str, token_type: str = "access") -> Dict[str, Any]:
    """Verify a JWT token"""
    return jwt_handler.verify_token(token=token, token_type=token_type)