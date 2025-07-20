"""
Configuration management for SutazAI Backend
============================================

Centralized configuration system using Pydantic settings.
Supports environment variables and configuration files.
"""

import os
from functools import lru_cache
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application settings
    app_name: str = "SutazAI Unified Backend"
    app_version: str = "8.0.0"
    debug: bool = Field(default=False, env="DEBUG")
    log_level: str = Field(default="INFO", env="LOG_LEVEL")
    
    # Server settings
    host: str = Field(default="0.0.0.0", env="HOST")
    port: int = Field(default=8000, env="PORT")
    workers: int = Field(default=1, env="WORKERS")
    
    # Database settings
    database_url: str = Field(env="DATABASE_URL", default="postgresql://sutazai@postgres:5432/sutazai")
    redis_url: str = Field(env="REDIS_URL", default="redis://redis:6379")
    
    # Vector database settings
    chromadb_url: str = Field(env="CHROMADB_URL", default="http://chromadb:8000")
    qdrant_url: str = Field(env="QDRANT_URL", default="http://qdrant:6333")
    
    # AI model settings
    ollama_url: str = Field(env="OLLAMA_URL", default="http://ollama:11434")
    default_model: str = Field(env="DEFAULT_MODEL", default="llama3")
    max_context_length: int = Field(env="MAX_CONTEXT_LENGTH", default=4096)
    
    # Security settings
    secret_key: str = Field(env="SECRET_KEY", default="your-secret-key-change-in-production")
    jwt_algorithm: str = Field(env="JWT_ALGORITHM", default="HS256")
    access_token_expire_minutes: int = Field(env="ACCESS_TOKEN_EXPIRE_MINUTES", default=30)
    
    # Performance settings
    max_concurrent_tasks: int = Field(env="MAX_CONCURRENT_TASKS", default=10)
    task_timeout_seconds: int = Field(env="TASK_TIMEOUT_SECONDS", default=300)
    request_timeout_seconds: int = Field(env="REQUEST_TIMEOUT_SECONDS", default=30)
    
    # Monitoring settings
    enable_metrics: bool = Field(env="ENABLE_METRICS", default=True)
    metrics_port: int = Field(env="METRICS_PORT", default=9090)
    
    # CORS settings
    cors_origins: List[str] = Field(
        env="CORS_ORIGINS",
        default=["http://localhost:8501", "http://frontend:8501", "*"]
    )
    
    # File storage settings
    upload_directory: str = Field(env="UPLOAD_DIRECTORY", default="/tmp/uploads")
    max_upload_size: int = Field(env="MAX_UPLOAD_SIZE", default=100 * 1024 * 1024)  # 100MB
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


def get_database_config() -> dict:
    """Get database configuration dictionary"""
    settings = get_settings()
    return {
        "url": settings.database_url,
        "echo": settings.debug,
        "pool_size": 20,
        "max_overflow": 30,
        "pool_timeout": 30,
        "pool_recycle": 1800
    }


def get_redis_config() -> dict:
    """Get Redis configuration dictionary"""
    settings = get_settings()
    return {
        "url": settings.redis_url,
        "decode_responses": True,
        "socket_timeout": 5,
        "socket_connect_timeout": 5,
        "retry_on_timeout": True
    }


def get_model_config() -> dict:
    """Get AI model configuration dictionary"""
    settings = get_settings()
    return {
        "ollama_url": settings.ollama_url,
        "default_model": settings.default_model,
        "max_context_length": settings.max_context_length,
        "timeout": settings.request_timeout_seconds
    }