"""SutazAI Backend Configuration Management"""

import os
import sys
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache
from pathlib import Path
import logging

# Temporarily modify sys.path to handle potential import issues during setup
original_path = sys.path[:]
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, project_root)

# Remove our path to ensure we're using the installed package
if "/opt/sutazaiapp" in sys.path:
    sys.path.remove("/opt/sutazaiapp")

# Restore path
sys.path = original_path

logger = logging.getLogger(__name__)

# Get the project root directory
ROOT_DIR = Path(__file__).parent.parent.parent.absolute()


class Settings(BaseSettings):
    API_HOST: str = Field(default="0.0.0.0")
    API_PORT: int = Field(default=8000)
    DEBUG: bool = Field(default=False)

    # Database
    DATABASE_URL: str = Field(
        default="sqlite:///./sutazai.db", description="SQLite database URL"
    )

    # SQLAlchemy URL (used directly by the ORM)
    SQLALCHEMY_DATABASE_URL: str = Field(
        default="sqlite:///./sutazai.db",
        description="SQLAlchemy database URL - will use DATABASE_URL if not provided",
    )

    # Security
    SECRET_KEY: str = Field(
        default="changethisinproduction",
        description="Secret key for JWT token encryption",
    )
    ALGORITHM: str = Field(default="HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30)

    # Qdrant vector database
    QDRANT_HOST: str = Field(default="localhost")
    QDRANT_PORT: int = Field(default=6333)
    VECTOR_DB_URL: str = Field(default="http://localhost:6333")

    # Sentence Transformer model
    SENTENCE_TRANSFORMER_MODEL: str = Field(default="all-MiniLM-L6-v2")

    # Models
    DEFAULT_MODEL: str = Field(default="gpt-3.5-turbo")
    MODEL_DEPLOYMENT_PATH: str = Field(default=str(ROOT_DIR / "models"))

    # Storage paths
    UPLOADS_DIR: Path = Field(default=ROOT_DIR / "uploads")
    WORKSPACE_DIR: Path = Field(default=ROOT_DIR / "workspace")
    DIAGRAMS_DIR: Path = Field(default=ROOT_DIR / "diagrams")

    # Additional fields that might be in .env
    APP_ENV: str = Field(default="development")
    LOG_LEVEL: str = Field(default="INFO")
    WEB_PORT: int = Field(default=3000)
    JWT_SECRET: str = Field(
        default="yo52f8be20200280918846f666d1a68fc0c8e3c8157bf3f34830ba58bc1d488e4e"
    )
    JWT_ALGORITHM: str = Field(default="HS256")
    JWT_EXPIRATION: int = Field(default=86400)
    DB_TYPE: str = Field(default="sqlite")
    DB_PATH: str = Field(default="/opt/sutazaiapp/storage/sutazai.db")
    MODEL_CACHE_DIR: str = Field(default="./model_management")
    MAX_TOKENS: int = Field(default=2048)
    TEMPERATURE: float = Field(default=0.7)
    MAX_DOCUMENT_SIZE_MB: int = Field(default=10)
    SUPPORTED_DOC_TYPES: str = Field(default="pdf,docx,txt,md")
    ENABLE_PROMETHEUS: bool = Field(default=True)
    PROMETHEUS_PORT: int = Field(default=9090)
    GRAFANA_PORT: int = Field(default=3001)
    LOGS_DIR: str = Field(default="./logs")
    UPLOAD_DIR: str = Field(default="./uploads")
    TEMP_DIR: str = Field(default="./tmp")
    BEHIND_PROXY: bool = Field(default=False)
    ENFORCE_HTTPS: bool = Field(default=False)
    SQLITE_PATH: str = Field(default="/opt/sutazaiapp/storage/sutazai.db")
    VECTOR_STORE_URL: str = Field(default="http://localhost:6333")
    VECTOR_STORE_COLLECTION: str = Field(default="sutazai_vectors")
    API_WORKERS: int = Field(default=4)
    API_TIMEOUT: int = Field(default=300)
    NODE_EXPORTER_PORT: int = Field(default=9100)
    SSL_CERT_PATH: str = Field(default="/opt/sutazaiapp/ssl/cert.pem")
    SSL_KEY_PATH: str = Field(default="/opt/sutazaiapp/ssl/key.pem")
    LOG_DIR: str = Field(default="/opt/sutazaiapp/logs")
    GPT4ALL_MODEL_PATH: str = Field(
        default="/opt/sutazaiapp/model_management/GPT4All/gpt4all.bin"
    )
    DEEPSEEK_MODEL_PATH: str = Field(
        default="/opt/sutazaiapp/model_management/DeepSeek-Coder-33B"
    )

    # Using the pydantic-settings v2 format
    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": True,
        "extra": "ignore",
    }

    def __init__(self, **data):
        super().__init__(**data)
        # If SQLALCHEMY_DATABASE_URL is not set, use DATABASE_URL
        if not self.SQLALCHEMY_DATABASE_URL:
            self.SQLALCHEMY_DATABASE_URL = self.DATABASE_URL


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.

    Returns:
        Settings: Application settings
    """
    try:
        settings = Settings()
        logger.info("Loaded settings")
        return settings
    except Exception as e:
        logger.error(f"Error loading settings: {e}")
        # Return default settings as fallback
        return Settings()


# Initialize settings object for direct import
settings = get_settings()
