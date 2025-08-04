"""
Configuration management using Pydantic Settings
"""
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import field_validator, Field
from functools import lru_cache
import os
import secrets

class Settings(BaseSettings):
    """
    Application settings with validation
    """
    # Core Settings
    PROJECT_NAME: str = "SutazAI"
    ENVIRONMENT: str = Field("development", env="ENVIRONMENT")
    DEBUG: bool = Field(False, env="DEBUG")
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")
    
    # System Configuration (environment variables from .env)
    TZ: str = Field("UTC", env="TZ")
    SUTAZAI_ENV: str = Field("local", env="SUTAZAI_ENV")
    LOCAL_IP: str = Field("127.0.0.1", env="LOCAL_IP")
    DEPLOYMENT_ID: str = Field("", env="DEPLOYMENT_ID")
    
    # Security
    SECRET_KEY: str = Field("default-secret-key-change-in-production", env="SECRET_KEY")
    JWT_SECRET: str = Field("default-jwt-secret-change-in-production", env="JWT_SECRET")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Database
    POSTGRES_HOST: str = Field("postgres", env="POSTGRES_HOST")
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = Field("sutazai", env="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field("sutazai_password", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field("sutazai", env="POSTGRES_DB")
    DATABASE_URL: Optional[str] = Field(None, env="DATABASE_URL")
    
    @property
    def computed_database_url(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Redis
    REDIS_HOST: str = Field("redis", env="REDIS_HOST")
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    
    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"
    
    # Vector Databases
    CHROMADB_HOST: str = Field("chromadb", env="CHROMADB_HOST")
    CHROMADB_PORT: int = 8001  # Changed from 8000 to avoid conflict with backend
    CHROMADB_API_KEY: Optional[str] = None
    
    QDRANT_HOST: str = Field("qdrant", env="QDRANT_HOST")
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    
    # Neo4j Configuration
    NEO4J_PASSWORD: str = Field("neo4j_password", env="NEO4J_PASSWORD")
    
    # Monitoring Configuration
    GRAFANA_PASSWORD: str = Field("admin", env="GRAFANA_PASSWORD")
    
    # Model Configuration - Emergency small models to prevent freezing
    OLLAMA_HOST: str = Field("http://ollama:9005", env="OLLAMA_HOST")
    OLLAMA_ORIGINS: str = Field("*", env="OLLAMA_ORIGINS")
    OLLAMA_NUM_PARALLEL: str = Field("2", env="OLLAMA_NUM_PARALLEL")
    OLLAMA_MAX_LOADED_MODELS: str = Field("2", env="OLLAMA_MAX_LOADED_MODELS")
    
    DEFAULT_MODEL: str = "tinyllama:1.1b"  # Ultra-small model to prevent system overload
    FALLBACK_MODEL: str = "qwen2.5:3b"  # Larger model for complex tasks when resources allow
    EMBEDDING_MODEL: str = "nomic-embed-text"
    MODEL_TIMEOUT: int = 300  # seconds
    
    # Performance optimization settings
    MODEL_PRELOAD_ENABLED: bool = True
    MODEL_CACHE_SIZE: int = 2  # Number of models to keep in memory
    
    @field_validator("OLLAMA_HOST", mode="before")
    @classmethod
    def validate_ollama_host(cls, v: str) -> str:
        """Ensure OLLAMA_HOST has proper format"""
        if v == "0.0.0.0" or v == "ollama":
            return "http://ollama:9005"
        if not v.startswith("http"):
            return f"http://{v}:9005"
        return v
    
    # GPU Configuration
    ENABLE_GPU: bool = Field(False, env="ENABLE_GPU")
    GPU_MEMORY_FRACTION: float = 0.8
    
    # Feature Flags
    ENABLE_MONITORING: bool = Field(True, env="ENABLE_MONITORING")
    ENABLE_LOGGING: bool = Field(True, env="ENABLE_LOGGING")
    ENABLE_HEALTH_CHECKS: bool = Field(True, env="ENABLE_HEALTH_CHECKS")
    
    # Performance Tuning
    MAX_WORKERS: int = Field(4, env="MAX_WORKERS")
    CONNECTION_POOL_SIZE: int = Field(20, env="CONNECTION_POOL_SIZE")
    CACHE_TTL: int = Field(3600, env="CACHE_TTL")
    
    # Resource Limits
    MAX_CONCURRENT_AGENTS: int = 10
    MAX_MODEL_INSTANCES: int = 5
    CACHE_SIZE_GB: int = 10
    MAX_UPLOAD_SIZE_MB: int = 100
    
    # Monitoring
    METRICS_ENABLED: bool = True
    PROMETHEUS_PORT: int = 9090
    
    # Paths
    DATA_PATH: str = "/data"
    MODELS_PATH: str = "/data/models"
    LOGS_PATH: str = "/logs"
    
    @field_validator("SECRET_KEY", mode="before")
    @classmethod
    def validate_secret_key(cls, v: Optional[str]) -> str:
        if not v:
            return secrets.token_urlsafe(32)
        return v
    
    class Config:
        env_file = ".env"
        case_sensitive = True

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

settings = get_settings()