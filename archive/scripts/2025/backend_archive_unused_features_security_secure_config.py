#!/usr/bin/env python3
"""
Secure Configuration Management
Enterprise-grade security configuration for SutazAI V7
"""

import os
import secrets
import logging
from typing import Dict, Any, Optional
from cryptography.fernet import Fernet
from pathlib import Path
import json

logger = logging.getLogger("SecureConfig")

class SecureConfigManager:
    """Enterprise-grade configuration management with encryption and secret handling"""
    
    def __init__(self, config_dir: str = "/opt/sutazaiapp/config/secure"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.encryption_key = self._get_or_create_encryption_key()
        self.cipher = Fernet(self.encryption_key)
        
    def _get_or_create_encryption_key(self) -> bytes:
        """Get or create encryption key for secure storage"""
        key_file = self.config_dir / ".encryption_key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read()
        else:
            # Generate new encryption key
            key = Fernet.generate_key()
            # Store with restricted permissions
            with open(key_file, 'wb') as f:
                f.write(key)
            os.chmod(key_file, 0o600)  # Owner read/write only
            logger.info("Generated new encryption key for secure configuration")
            return key
    
    def get_secret(self, secret_name: str, default: Optional[str] = None) -> str:
        """Get secret from secure storage with environment variable fallback"""
        
        # Priority order:
        # 1. Environment variable
        # 2. Encrypted file storage
        # 3. Default value (only for development)
        
        # Check environment variable first
        env_value = os.environ.get(secret_name)
        if env_value:
            return env_value
            
        # Check encrypted storage
        secret_file = self.config_dir / f"{secret_name}.enc"
        if secret_file.exists():
            try:
                with open(secret_file, 'rb') as f:
                    encrypted_data = f.read()
                decrypted_value = self.cipher.decrypt(encrypted_data).decode()
                return decrypted_value
            except Exception as e:
                logger.error(f"Failed to decrypt secret {secret_name}: {e}")
        
        # Use default only in development
        if default and os.environ.get("SUTAZAI_ENV", "production") == "development":
            logger.warning(f"Using default value for secret {secret_name} in development mode")
            return default
            
        # Generate secure random secret if none exists
        if secret_name in ["AUTH_SECRET_KEY", "JWT_SECRET", "ENCRYPTION_KEY"]:
            logger.info(f"Generating secure random secret for {secret_name}")
            secure_secret = secrets.token_urlsafe(64)
            self.store_secret(secret_name, secure_secret)
            return secure_secret
            
        raise ValueError(f"Secret {secret_name} not found and no default provided")
    
    def store_secret(self, secret_name: str, secret_value: str) -> bool:
        """Store secret in encrypted file storage"""
        try:
            secret_file = self.config_dir / f"{secret_name}.enc"
            encrypted_data = self.cipher.encrypt(secret_value.encode())
            
            with open(secret_file, 'wb') as f:
                f.write(encrypted_data)
            os.chmod(secret_file, 0o600)  # Owner read/write only
            
            logger.info(f"Stored encrypted secret: {secret_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to store secret {secret_name}: {e}")
            return False
    
    def rotate_secret(self, secret_name: str) -> str:
        """Rotate a secret by generating a new value"""
        new_secret = secrets.token_urlsafe(64)
        if self.store_secret(secret_name, new_secret):
            logger.info(f"Rotated secret: {secret_name}")
            return new_secret
        else:
            raise RuntimeError(f"Failed to rotate secret: {secret_name}")

# Global secure config manager instance
secure_config = SecureConfigManager()

# Secure configuration functions
def get_auth_secret() -> str:
    """Get authentication secret key"""
    return secure_config.get_secret("AUTH_SECRET_KEY")

def get_grafana_password() -> str:
    """Get Grafana admin password"""
    return secure_config.get_secret("GRAFANA_ADMIN_PASSWORD")

def get_postgres_password() -> str:
    """Get PostgreSQL password"""
    return secure_config.get_secret("POSTGRES_PASSWORD")

def get_qdrant_api_key() -> str:
    """Get Qdrant API key"""
    return secure_config.get_secret("QDRANT_API_KEY")

def get_jwt_secret() -> str:
    """Get JWT signing secret"""
    return secure_config.get_secret("JWT_SECRET")

def get_encryption_key() -> str:
    """Get application encryption key"""
    return secure_config.get_secret("ENCRYPTION_KEY")

# CORS configuration
def get_allowed_origins() -> list:
    """Get allowed CORS origins based on environment"""
    env = os.environ.get("SUTAZAI_ENV", "production")
    
    if env == "development":
        return [
            "http://localhost:3000",
            "http://localhost:8501",
            "http://localhost:8000",
            "http://127.0.0.1:3000",
            "http://127.0.0.1:8501",
            "http://127.0.0.1:8000"
        ]
    elif env == "staging":
        return [
            "https://staging.sutazai.company.com",
            "https://staging-admin.sutazai.company.com"
        ]
    else:  # production
        return [
            "https://sutazai.company.com",
            "https://admin.sutazai.company.com",
            "https://api.sutazai.company.com"
        ]

def get_allowed_hosts() -> list:
    """Get allowed hosts for the application"""
    env = os.environ.get("SUTAZAI_ENV", "production")
    
    if env == "development":
        return ["localhost", "127.0.0.1", "0.0.0.0"]
    elif env == "staging":
        return ["staging.sutazai.company.com", "staging-admin.sutazai.company.com"]
    else:  # production
        return ["sutazai.company.com", "admin.sutazai.company.com", "api.sutazai.company.com"]

# Rate limiting configuration
def get_rate_limits() -> Dict[str, str]:
    """Get rate limiting configuration"""
    return {
        "default": "100/minute",
        "chat": "10/minute",
        "upload": "5/minute",
        "auth": "5/minute",
        "model_inference": "20/minute",
        "admin": "50/minute"
    }

# Security headers configuration
def get_security_headers() -> Dict[str, str]:
    """Get security headers configuration"""
    return {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
        "Content-Security-Policy": "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; media-src 'self'; connect-src 'self'",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), camera=()"
    }

if __name__ == "__main__":
    # Test the secure configuration
    print("Testing secure configuration...")
    print(f"Auth secret length: {len(get_auth_secret())}")
    print(f"Allowed origins: {get_allowed_origins()}")
    print(f"Rate limits: {get_rate_limits()}")
    print("Secure configuration test completed.")