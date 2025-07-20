#!/usr/bin/env python3
"""
Secure Configuration Management for SutazAI
Replaces hardcoded credentials with environment-based configuration
"""

import os
import secrets
import hashlib
from typing import Optional, Dict, Any
from pathlib import Path
import logging

# Handle pydantic version compatibility
try:
    from pydantic_settings import BaseSettings
    from pydantic import Field, field_validator
except ImportError:
    try:
        from pydantic import BaseSettings, Field, validator as field_validator
    except ImportError:
        # Fallback to basic implementation without pydantic
        BaseSettings = object
        def Field(*args, **kwargs):
            return kwargs.get('default')
        def field_validator(*args, **kwargs):
            def decorator(func):
                return func
            return decorator

logger = logging.getLogger(__name__)

class SecurityConfig(BaseSettings):
    """Secure configuration management with validation"""
    
    # Database Configuration
    POSTGRES_HOST: str = Field(default="localhost", env="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(default=5432, env="POSTGRES_PORT")
    POSTGRES_DB: str = Field(default="sutazai", env="POSTGRES_DB")
    POSTGRES_USER: str = Field(default="sutazai", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., env="POSTGRES_PASSWORD")
    
    # Redis Configuration
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Security Configuration
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    JWT_SECRET: str = Field(..., env="JWT_SECRET")
    API_KEY: str = Field(..., env="API_KEY")
    
    # Application Configuration
    DEBUG: bool = Field(default=False, env="DEBUG")
    ENVIRONMENT: str = Field(default="production", env="ENVIRONMENT")
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    
    # External Services
    OLLAMA_HOST: str = Field(default="localhost", env="OLLAMA_HOST")
    OLLAMA_PORT: int = Field(default=11434, env="OLLAMA_PORT")
    
    # Monitoring
    GRAFANA_ADMIN_PASSWORD: str = Field(..., env="GRAFANA_ADMIN_PASSWORD")
    PROMETHEUS_ENABLED: bool = Field(default=True, env="PROMETHEUS_ENABLED")
    
    # SSL/TLS Configuration
    SSL_ENABLED: bool = Field(default=True, env="SSL_ENABLED")
    SSL_CERT_PATH: str = Field(default="/opt/sutazaiapp/ssl/cert.pem", env="SSL_CERT_PATH")
    SSL_KEY_PATH: str = Field(default="/opt/sutazaiapp/ssl/key.pem", env="SSL_KEY_PATH")
    
    class Config:
        env_file = "/opt/sutazaiapp/.env"
        env_file_encoding = "utf-8"
        case_sensitive = True
    
    @field_validator("POSTGRES_PASSWORD", "SECRET_KEY", "JWT_SECRET", "API_KEY", "GRAFANA_ADMIN_PASSWORD")
    def validate_secure_passwords(cls, v, field=None):
        """Validate that passwords meet security requirements"""
        if not v or v in ["CHANGE_ME_SECURE_PASSWORD", "CHANGE_ME_SECRET_KEY", 
                         "CHANGE_ME_JWT_SECRET", "CHANGE_ME_API_KEY", 
                         "CHANGE_ME_GRAFANA_PASSWORD"]:
            field_name = getattr(field, 'name', 'password field') if field else 'password field'
            raise ValueError(f"{field_name} must be set to a secure value")
        
        if len(v) < 16:
            field_name = getattr(field, 'name', 'password field') if field else 'password field'
            raise ValueError(f"{field_name} must be at least 16 characters long")
        
        return v
    
    @field_validator("ENVIRONMENT")
    def validate_environment(cls, v):
        """Validate environment setting"""
        if v not in ["development", "staging", "production"]:
            raise ValueError("ENVIRONMENT must be one of: development, staging, production")
        return v
    
    @property
    def database_url(self) -> str:
        """Get database URL"""
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    @property
    def ollama_url(self) -> str:
        """Get Ollama URL"""
        return f"http://{self.OLLAMA_HOST}:{self.OLLAMA_PORT}"

class SecretManager:
    """Secure secret management utilities"""
    
    @staticmethod
    def generate_secure_password(length: int = 32) -> str:
        """Generate a cryptographically secure password"""
        return secrets.token_urlsafe(length)
    
    @staticmethod
    def generate_secret_key() -> str:
        """Generate a secure secret key for application"""
        return secrets.token_urlsafe(64)
    
    @staticmethod
    def hash_password(password: str, salt: Optional[str] = None) -> tuple[str, str]:
        """Hash a password with salt"""
        if salt is None:
            salt = secrets.token_hex(16)
        
        password_hash = hashlib.pbkdf2_hmac(
            'sha256',
            password.encode('utf-8'),
            salt.encode('utf-8'),
            100000  # iterations
        )
        
        return password_hash.hex(), salt
    
    @staticmethod
    def verify_password(password: str, password_hash: str, salt: str) -> bool:
        """Verify a password against its hash"""
        computed_hash, _ = SecretManager.hash_password(password, salt)
        return secrets.compare_digest(computed_hash, password_hash)

def initialize_secure_config() -> SecurityConfig:
    """Initialize secure configuration with validation"""
    env_file = Path("/opt/sutazaiapp/.env")
    
    if not env_file.exists():
        logger.warning(".env file not found. Creating template...")
        create_env_template()
        raise FileNotFoundError(
            "Configuration file not found. Please copy .env.template to .env and configure it."
        )
    
    try:
        config = SecurityConfig()
        logger.info("Secure configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load secure configuration: {e}")
        raise

def create_env_template():
    """Create .env template with secure defaults"""
    template_path = Path("/opt/sutazaiapp/.env.template")
    env_path = Path("/opt/sutazaiapp/.env")
    
    if template_path.exists() and not env_path.exists():
        # Generate secure values
        postgres_password = SecretManager.generate_secure_password()
        secret_key = SecretManager.generate_secret_key()
        jwt_secret = SecretManager.generate_secret_key()
        api_key = SecretManager.generate_secure_password()
        grafana_password = SecretManager.generate_secure_password()
        
        # Read template
        with open(template_path, 'r') as f:
            content = f.read()
        
        # Replace placeholders
        content = content.replace("CHANGE_ME_SECURE_PASSWORD", postgres_password)
        content = content.replace("CHANGE_ME_SECRET_KEY", secret_key)
        content = content.replace("CHANGE_ME_JWT_SECRET", jwt_secret)
        content = content.replace("CHANGE_ME_API_KEY", api_key)
        content = content.replace("CHANGE_ME_GRAFANA_PASSWORD", grafana_password)
        
        # Write .env file
        with open(env_path, 'w') as f:
            f.write(content)
        
        # Set secure permissions
        os.chmod(env_path, 0o600)
        
        logger.info("Generated secure .env file with random passwords")

def get_config() -> SecurityConfig:
    """Get singleton configuration instance"""
    if not hasattr(get_config, '_instance'):
        get_config._instance = initialize_secure_config()
    return get_config._instance

# Singleton instance
config = None

def load_config():
    """Load configuration singleton"""
    global config
    if config is None:
        config = get_config()
    return config