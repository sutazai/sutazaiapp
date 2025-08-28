"""
SutazAI Platform Configuration
Centralized configuration management for all services
"""

from pydantic_settings import BaseSettings
from typing import Optional
import os


class Settings(BaseSettings):
    """Application settings with environment variable support"""
    
    # Application
    APP_NAME: str = "SutazAI Platform API"
    APP_VERSION: str = "4.0.0"
    DEBUG: bool = False
    API_V1_STR: str = "/api/v1"
    
    # Server
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    
    # PostgreSQL
    POSTGRES_HOST: str = "localhost"
    POSTGRES_PORT: int = 10000
    POSTGRES_USER: str = "jarvis"
    POSTGRES_PASSWORD: str = "sutazai_secure_2024"
    POSTGRES_DB: str = "jarvis_ai"
    
    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Redis
    REDIS_HOST: str = "sutazai-redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
    
    # RabbitMQ
    RABBITMQ_HOST: str = "sutazai-rabbitmq"
    RABBITMQ_PORT: int = 5672
    RABBITMQ_USER: str = "sutazai"
    RABBITMQ_PASSWORD: str = "sutazai_secure_2024"
    
    @property
    def RABBITMQ_URL(self) -> str:
        return f"amqp://{self.RABBITMQ_USER}:{self.RABBITMQ_PASSWORD}@{self.RABBITMQ_HOST}:{self.RABBITMQ_PORT}/"
    
    # Neo4j
    NEO4J_HOST: str = "sutazai-neo4j"
    NEO4J_BOLT_PORT: int = 7687
    NEO4J_HTTP_PORT: int = 7474
    NEO4J_USER: str = "neo4j"
    NEO4J_PASSWORD: str = "sutazai_secure_2024"
    
    @property
    def NEO4J_URL(self) -> str:
        return f"bolt://{self.NEO4J_HOST}:{self.NEO4J_BOLT_PORT}"
    
    # Vector Databases
    CHROMADB_HOST: str = "sutazai-chromadb"
    CHROMADB_PORT: int = 8000
    CHROMADB_TOKEN: str = "sutazai-secure-token-2024"
    
    QDRANT_HOST: str = "sutazai-qdrant"
    QDRANT_HTTP_PORT: int = 6334
    QDRANT_GRPC_PORT: int = 6333
    
    FAISS_HOST: str = "sutazai-faiss"
    FAISS_PORT: int = 8000
    
    # Service Discovery
    CONSUL_HOST: str = "sutazai-consul"
    CONSUL_PORT: int = 8500
    
    # API Gateway
    KONG_HOST: str = "sutazai-kong"
    KONG_ADMIN_PORT: int = 8001
    KONG_PROXY_PORT: int = 8000
    
    # Ollama
    OLLAMA_HOST: str = "host.docker.internal"
    OLLAMA_PORT: int = 11434
    
    # Security
    SECRET_KEY: str = "sutazai-secret-key-2025-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # Connection Pool Settings
    DB_POOL_SIZE: int = 10
    DB_MAX_OVERFLOW: int = 20
    DB_POOL_TIMEOUT: int = 30
    DB_POOL_RECYCLE: int = 1800
    
    # CORS
    CORS_ORIGINS: list = ["*"]
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Create global settings instance
settings = Settings()