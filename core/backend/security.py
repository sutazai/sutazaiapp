"""
Security Management for SutazAI
==============================

Basic security implementation for the SutazAI system.
"""

import hashlib
import secrets
from typing import Optional, Dict, Any
from datetime import datetime, timedelta

from .config import Settings
from .utils import setup_logging

logger = setup_logging(__name__)


class SecurityManager:
    """Basic security management"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.api_keys = set()
        self.rate_limits = {}
    
    def generate_api_key(self) -> str:
        """Generate a new API key"""
        key = secrets.token_urlsafe(32)
        self.api_keys.add(key)
        return key
    
    def validate_api_key(self, key: str) -> bool:
        """Validate API key"""
        return key in self.api_keys
    
    def hash_password(self, password: str) -> str:
        """Hash password using SHA-256"""
        salt = secrets.token_hex(16)
        return hashlib.sha256((password + salt).encode()).hexdigest() + ":" + salt
    
    def verify_password(self, password: str, hashed: str) -> bool:
        """Verify password against hash"""
        try:
            password_hash, salt = hashed.split(":")
            return hashlib.sha256((password + salt).encode()).hexdigest() == password_hash
        except:
            return False
    
    async def check_rate_limit(self, identifier: str, limit: int = 100, window: int = 3600) -> bool:
        """Simple rate limiting"""
        now = datetime.now()
        
        if identifier not in self.rate_limits:
            self.rate_limits[identifier] = []
        
        # Clean old entries
        self.rate_limits[identifier] = [
            timestamp for timestamp in self.rate_limits[identifier]
            if now - timestamp < timedelta(seconds=window)
        ]
        
        # Check limit
        if len(self.rate_limits[identifier]) >= limit:
            return False
        
        # Add current request
        self.rate_limits[identifier].append(now)
        return True