"""
Configuration management using Pydantic Settings
"""
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import validator, Field
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
    
    # Security
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    BACKEND_CORS_ORIGINS: List[str] = ["*"]
    
    # Database
    POSTGRES_HOST: str
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str
    POSTGRES_PASSWORD: str
    POSTGRES_DB: str
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Redis
    REDIS_HOST: str
    REDIS_PORT: int = 6379
    REDIS_PASSWORD: Optional[str] = None
    
    @property
    def REDIS_URL(self) -> str:
        if self.REDIS_PASSWORD:
            return f"redis://:{self.REDIS_PASSWORD}@{self.REDIS_HOST}:{self.REDIS_PORT}/0"
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"
    
    # Vector Databases
    CHROMADB_HOST: str
    CHROMADB_PORT: int = 8000
    CHROMADB_API_KEY: Optional[str] = None
    
    QDRANT_HOST: str
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    
    # Model Configuration
    OLLAMA_HOST: str = "http://ollama:11434"
    DEFAULT_MODEL: str = "qwen2.5:3b"  # Faster 3B model for better response times
    FALLBACK_MODEL: str = "deepseek-r1:8b"  # Larger model for complex tasks
    EMBEDDING_MODEL: str = "nomic-embed-text"
    MODEL_TIMEOUT: int = 300  # seconds
    
    # Performance optimization settings
    MODEL_PRELOAD_ENABLED: bool = True
    MODEL_CACHE_SIZE: int = 3  # Number of models to keep in memory
    
    # Advanced Model Manager settings
    ADVANCED_CACHING_ENABLED: bool = True
    BATCH_SIZE: int = 8  # Default batch size for request batching
    BATCH_TIMEOUT_MS: int = 100  # Batch timeout in milliseconds
    STREAMING_ENABLED: bool = True  # Enable streaming responses
    GPU_ACCELERATION: bool = True  # Enable GPU acceleration when available
    
    # Cache management
    CACHE_WARMUP_ON_STARTUP: bool = True
    CACHE_ARTIFACT_RETENTION_DAYS: int = 7
    PERFORMANCE_MONITORING: bool = True
    
    @validator("OLLAMA_HOST", pre=True)
    def validate_ollama_host(cls, v: str) -> str:
        """Ensure OLLAMA_HOST has proper format"""
        if v == "0.0.0.0" or v == "ollama":
            return "http://ollama:11434"
        if not v.startswith("http"):
            return f"http://{v}:11434"
        return v
    
    # GPU Configuration
    ENABLE_GPU: bool = True
    GPU_MEMORY_FRACTION: float = 0.8
    
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
    
    @validator("SECRET_KEY", pre=True)
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