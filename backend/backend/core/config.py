"""
Backend Core Configuration
Configuration settings for backend components
"""

from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    """Application settings"""
    
    # Database settings
    database_url: str = "postgresql://sutazai:sutazai_password@postgres:5432/sutazai"
    redis_url: str = "redis://:redis_password@redis:6379/0"
    
    # AI Services
    ollama_url: str = "http://ollama:11434"
    chromadb_url: str = "http://chromadb:8000"
    qdrant_url: str = "http://qdrant:6333"
    
    # Enterprise features
    enterprise_mode: bool = True
    monitoring_enabled: bool = True
    
    # Agent settings
    max_agents: int = 100
    agent_timeout: int = 300
    
    # Memory settings
    max_short_term_memory: int = 100
    max_long_term_memory: int = 1000
    max_shared_memory: int = 500
    
    # Workflow settings
    max_workflow_tasks: int = 50
    workflow_timeout: int = 3600
    
    # Security
    jwt_secret: str = "your-secret-key"
    jwt_algorithm: str = "HS256"
    jwt_expiration: int = 86400
    
    class Config:
        env_file = ".env"
        case_sensitive = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

# Convenience function for backward compatibility
def get_config() -> Dict[str, Any]:
    """Get configuration as dictionary"""
    settings = get_settings()
    return settings.model_dump()