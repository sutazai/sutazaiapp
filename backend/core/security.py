#!/usr/bin/env python3
"""
SutazAI Security Manager
Authentication, authorization, and security utilities
"""

import logging
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import jwt
import bcrypt
from passlib.context import CryptContext

from .config import settings

logger = logging.getLogger(__name__)


class SecurityManager:
    """Security and authentication manager"""
    
    def __init__(self):
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
        self.jwt_algorithm = settings.ALGORITHM
        self.jwt_secret = settings.JWT_SECRET
        self.jwt_expire_hours = settings.ACCESS_TOKEN_EXPIRE_MINUTES // 60
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt"""
        return self.pwd_context.hash(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash"""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def create_access_token(self, data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(hours=self.jwt_expire_hours)
        
        to_encode.update({"exp": expire})
        
        encoded_jwt = jwt.encode(to_encode, self.jwt_secret, algorithm=self.jwt_algorithm)
        return encoded_jwt
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify JWT token and return payload"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return payload
        except jwt.ExpiredSignatureError:
            logger.warning("Token has expired")
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid token")
            return None
    
    def generate_api_key(self, user_id: str, scopes: List[str] = None) -> str:
        """Generate API key for user"""
        data = {
            "user_id": user_id,
            "type": "api_key",
            "scopes": scopes or ["read", "write"],
            "created_at": datetime.utcnow().isoformat()
        }
        
        # API keys don't expire by default
        return jwt.encode(data, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify API key and return payload"""
        try:
            payload = jwt.decode(api_key, self.jwt_secret, algorithms=[self.jwt_algorithm])
            if payload.get("type") == "api_key":
                return payload
            return None
        except jwt.InvalidTokenError:
            logger.warning("Invalid API key")
            return None
    
    def sanitize_input(self, input_text: str) -> str:
        """Sanitize user input to prevent injection attacks"""
        if not input_text:
            return ""
        
        # Remove potential SQL injection patterns
        dangerous_patterns = [
            "SELECT", "INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
            "EXEC", "EXECUTE", "UNION", "SCRIPT", "JAVASCRIPT", "VBSCRIPT",
            "<script>", "</script>", "eval(", "javascript:", "onload=", "onerror="
        ]
        
        sanitized = input_text
        for pattern in dangerous_patterns:
            sanitized = sanitized.replace(pattern.lower(), "")
            sanitized = sanitized.replace(pattern.upper(), "")
        
        return sanitized.strip()
    
    def validate_file_upload(self, filename: str, file_size: int) -> Dict[str, Any]:
        """Validate file upload"""
        result = {"valid": False, "errors": []}
        
        # Check file extension
        allowed_extensions = settings.SUPPORTED_DOC_TYPES.split(",")
        file_extension = filename.split(".")[-1].lower() if "." in filename else ""
        
        if file_extension not in allowed_extensions:
            result["errors"].append(f"File type '{file_extension}' not allowed")
        
        # Check file size (limit to 100MB)
        max_size = 100 * 1024 * 1024  # 100MB in bytes
        if file_size > max_size:
            result["errors"].append(f"File size {file_size} exceeds limit of {max_size} bytes")
        
        # Check filename for suspicious content
        suspicious_chars = ["<", ">", ":", "\"", "|", "?", "*", "\\", "/"]
        if any(char in filename for char in suspicious_chars):
            result["errors"].append("Filename contains suspicious characters")
        
        result["valid"] = len(result["errors"]) == 0
        return result
    
    def check_rate_limit(self, user_id: str, endpoint: str, max_requests: int = 100, window_minutes: int = 60) -> Dict[str, Any]:
        """Check rate limit for user/endpoint combination"""
        # This is a simplified rate limiter
        # In production, you'd use Redis or similar for distributed rate limiting
        
        # For now, return success (implement proper rate limiting with Redis)
        return {
            "allowed": True,
            "remaining": max_requests - 1,
            "reset_time": datetime.utcnow() + timedelta(minutes=window_minutes)
        }
    
    def generate_csrf_token(self, user_id: str) -> str:
        """Generate CSRF token"""
        data = {
            "user_id": user_id,
            "type": "csrf",
            "created_at": datetime.utcnow().isoformat()
        }
        
        # CSRF tokens expire in 1 hour
        expire = datetime.utcnow() + timedelta(hours=1)
        data["exp"] = expire
        
        return jwt.encode(data, self.jwt_secret, algorithm=self.jwt_algorithm)
    
    def verify_csrf_token(self, token: str, user_id: str) -> bool:
        """Verify CSRF token"""
        try:
            payload = jwt.decode(token, self.jwt_secret, algorithms=[self.jwt_algorithm])
            return (payload.get("type") == "csrf" and 
                   payload.get("user_id") == user_id)
        except jwt.InvalidTokenError:
            return False
    
    def audit_log(self, action: str, user_id: str, details: Dict[str, Any] = None):
        """Log security-relevant actions"""
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "action": action,
            "user_id": user_id,
            "details": details or {}
        }
        
        # Log to security log
        logger.info(f"Security audit: {log_entry}")
        
        # TODO: Store in database for audit trail
        
    def encrypt_sensitive_data(self, data: str) -> str:
        """Encrypt sensitive data (placeholder)"""
        # In production, use proper encryption like Fernet
        return data  # Placeholder
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """Decrypt sensitive data (placeholder)"""
        # In production, use proper decryption like Fernet
        return encrypted_data  # Placeholder