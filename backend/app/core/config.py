"""
Configuration management using Pydantic Settings
"""
from typing import List, Optional, Dict, Any
from pydantic_settings import BaseSettings, SettingsConfigDict
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
    
    # Security - REQUIRED environment variables for production security
    SECRET_KEY: str = Field(..., env="SECRET_KEY", description="Required: Application secret key")
    JWT_SECRET: str = Field(..., env="JWT_SECRET", description="Required: JWT signing secret")
    JWT_ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # API Configuration
    API_V1_PREFIX: str = "/api/v1"
    # Security: Specify exact allowed origins instead of wildcard
    # These are the legitimate services that need CORS access
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:10011",  # Frontend Streamlit UI
        "http://localhost:10010",  # Backend itself (for Swagger UI)
        "http://localhost:3000",   # Development frontend (if needed)
        "http://127.0.0.1:10011",  # Alternative localhost
        "http://127.0.0.1:10010",  # Alternative localhost backend
        # Add production domains here when deployed
        # "https://sutazai.example.com",
    ]
    
    # Database
    POSTGRES_HOST: str = Field("postgres", env="POSTGRES_HOST")
    POSTGRES_PORT: int = 5432
    POSTGRES_USER: str = Field("sutazai", env="POSTGRES_USER")
    # Security: do not embed default secrets in code. Expect runtime env to provide.
    POSTGRES_PASSWORD: str = Field("", env="POSTGRES_PASSWORD")
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
    CHROMADB_PORT: int = 8000
    CHROMADB_API_KEY: Optional[str] = None
    
    QDRANT_HOST: str = Field("qdrant", env="QDRANT_HOST")
    QDRANT_PORT: int = 6333
    QDRANT_API_KEY: Optional[str] = None
    
    # Neo4j Configuration
    # Security: do not embed default secrets in code. Expect runtime env to provide.
    NEO4J_PASSWORD: str = Field("", env="NEO4J_PASSWORD")
    
    # Monitoring Configuration
    # Security: do not embed default secrets in code. Expect runtime env to provide.
    GRAFANA_PASSWORD: str = Field("", env="GRAFANA_PASSWORD")
    
    # Model Configuration - Fixed to use correct Ollama internal port
    OLLAMA_HOST: str = Field("http://ollama:11434", env="OLLAMA_HOST")
    OLLAMA_ORIGINS: str = Field("*", env="OLLAMA_ORIGINS")
    OLLAMA_NUM_PARALLEL: str = Field("2", env="OLLAMA_NUM_PARALLEL")
    OLLAMA_MAX_LOADED_MODELS: str = Field("2", env="OLLAMA_MAX_LOADED_MODELS")
    
    DEFAULT_MODEL: str = "tinyllama"  # Fixed to use available tinyllama model
    FALLBACK_MODEL: str = "tinyllama"  # Fixed to use available tinyllama model
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
            return "http://ollama:11434"
        if not v.startswith("http"):
            # Default to the correct Ollama internal port (11434)
            return f"http://{v}:11434"
        return v
    
    @field_validator("JWT_SECRET", "SECRET_KEY", mode="after")
    @classmethod
    def validate_secrets(cls, v: str, info) -> str:
        """Ensure secrets are not using default/insecure values"""
        insecure_values = [
            "default-secret-key-change-in-production",
            "default-jwt-secret-change-in-production",
            "changeme",
            "secret",
            "password",
            "123456",
            "admin"
        ]
        
        if not v or len(v) < 32:
            raise ValueError(f"{info.field_name} must be at least 32 characters long for security")
        
        if v.lower() in insecure_values or any(bad in v.lower() for bad in ["default", "change", "todo"]):
            raise ValueError(
                f"{info.field_name} is using an insecure default value. "
                f"Generate a secure secret with: python -c 'import secrets; print(secrets.token_urlsafe(32))'"
            )
        
        return v

    @field_validator("POSTGRES_PASSWORD", "NEO4J_PASSWORD", "GRAFANA_PASSWORD", mode="after")
    @classmethod
    def validate_service_passwords(cls, v: str, info) -> str:
        """Enforce non-empty, non-insecure passwords in non-local environments."""
        # Allow empty in explicit local/dev setups to avoid blocking local runs,
        # but require strong values in staging/production.
        env = os.getenv("SUTAZAI_ENV", os.getenv("ENVIRONMENT", "local")).lower()
        insecure_values = {"", "password", "admin", "changeme", "default"}
        if env in {"production", "prod", "staging"}:
            if v is None or v.strip().lower() in insecure_values or len(v) < 8:
                raise ValueError(
                    f"{info.field_name} must be provided via environment and be at least 8 characters in {env}"
                )
        return v
    
    # GPU Configuration
    ENABLE_GPU: bool = Field(False, env="ENABLE_GPU")
    GPU_MEMORY_FRACTION: float = 0.8
    
    # Feature Flags
    ENABLE_MONITORING: bool = Field(True, env="ENABLE_MONITORING")
    ENABLE_LOGGING: bool = Field(True, env="ENABLE_LOGGING")
    ENABLE_HEALTH_CHECKS: bool = Field(True, env="ENABLE_HEALTH_CHECKS")
    
    # Optional Features (disabled by default)
    ENABLE_FSDP: bool = Field(False, env="ENABLE_FSDP")
    ENABLE_TABBY: bool = Field(False, env="ENABLE_TABBY")
    TABBY_URL: str = Field("http://tabbyml:8080", env="TABBY_URL")
    TABBY_API_KEY: Optional[str] = Field(None, env="TABBY_API_KEY")
    
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
    
    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
        extra="ignore",  # Ignore unrelated env vars to prevent ValidationError
    )

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()

settings = get_settings()
