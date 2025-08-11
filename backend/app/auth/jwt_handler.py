"""
ULTRA-SECURE JWT Token Handler for SUTAZAI Authentication System
Implements RS256 JWT with RSA keys for maximum security
"""

import os
import secrets
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path

import jwt
from jwt import PyJWTError

# RSA Key Configuration for RS256
JWT_PRIVATE_KEY_PATH = os.getenv("JWT_PRIVATE_KEY_PATH", "/opt/sutazaiapp/secrets/jwt/private_key.pem")
JWT_PUBLIC_KEY_PATH = os.getenv("JWT_PUBLIC_KEY_PATH", "/opt/sutazaiapp/secrets/jwt/public_key.pem")
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")  # Fallback to environment variable
JWT_ALGORITHM = "RS256"  # Upgraded to RS256 for ULTRA-SECURITY
ACCESS_TOKEN_EXPIRE_MINUTES = 30
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Load RSA Keys with fallback to HS256
def load_rsa_keys():
    """Load RSA private and public keys for JWT signing and verification"""
    try:
        # Check if cryptography is available
        try:
            from cryptography.hazmat.primitives import serialization
            from cryptography.hazmat.primitives.asymmetric import rsa
        except ImportError:
            import logging
            logger = logging.getLogger(__name__)
            logger.info("Cryptography library not available, using HS256 instead of RS256")
            return None, None
        
        # Load private key
        if os.path.exists(JWT_PRIVATE_KEY_PATH):
            with open(JWT_PRIVATE_KEY_PATH, 'rb') as f:
                private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None  # No password for now, can be enhanced
                )
        else:
            raise FileNotFoundError(f"Private key not found at {JWT_PRIVATE_KEY_PATH}")
        
        # Load public key
        if os.path.exists(JWT_PUBLIC_KEY_PATH):
            with open(JWT_PUBLIC_KEY_PATH, 'rb') as f:
                public_key = serialization.load_pem_public_key(f.read())
        else:
            raise FileNotFoundError(f"Public key not found at {JWT_PUBLIC_KEY_PATH}")
        
        return private_key, public_key
        
    except Exception as e:
        # Log warning and return None to fallback to HS256
        import logging
        logger = logging.getLogger(__name__)
        logger.warning(f"RSA keys not available, falling back to HS256: {e}")
        return None, None

# Load keys at module level with fallback
PRIVATE_KEY, PUBLIC_KEY = load_rsa_keys()

# Use HS256 if RSA keys are not available
if PRIVATE_KEY is None or PUBLIC_KEY is None:
    JWT_ALGORITHM = "HS256"
    if not JWT_SECRET_KEY:
        raise ValueError("JWT_SECRET_KEY must be set when RSA keys are not available")
else:
    JWT_ALGORITHM = "RS256"


class JWTHandler:
    """Handler for JWT token operations"""
    
    def __init__(self, secret_key: str = None, algorithm: str = None):
        self.algorithm = algorithm or JWT_ALGORITHM
        
        # Set signing key based on algorithm
        if self.algorithm == "RS256":
            self.signing_key = PRIVATE_KEY
            self.verification_key = PUBLIC_KEY
        else:  # HS256
            if secret_key:
                self.signing_key = secret_key
                self.verification_key = secret_key
            else:
                self.signing_key = JWT_SECRET_KEY
                self.verification_key = JWT_SECRET_KEY
        
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
        
        encoded_jwt = jwt.encode(payload, self.signing_key, algorithm=self.algorithm)
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
        
        encoded_jwt = jwt.encode(payload, self.signing_key, algorithm=self.algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str, token_type: str = "access") -> Dict[str, Any]:
        """Verify and decode a JWT token"""
        try:
            payload = jwt.decode(
                token,
                self.verification_key,
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
