from typing import List, Optional, Union, Dict, Any
from pydantic import AnyHttpUrl, PostgresDsn, validator, Field
from pydantic_settings import BaseSettings
from functools import lru_cache
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Determine Project Root dynamically relative to this config file
# Assuming this file is at backend/core/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class AppSettings(BaseSettings):
    # --- Core Settings ---
    PROJECT_NAME: str = "SutazAI AGI/ASI System"
    PROJECT_VERSION: str = "1.0.0"
    DEBUG_MODE: bool = Field(False, env="DEBUG_MODE")
    SECRET_KEY: str = Field(default="sutazai_super_secret_key_change_in_production", env="SECRET_KEY")
    JWT_SECRET: str = Field(default="sutazai_jwt_secret_change_in_production", env="JWT_SECRET")

    # --- API Settings ---
    API_V1_STR: str = "/api/v1"
    SERVER_HOST: str = Field("0.0.0.0", env="SERVER_HOST")
    SERVER_PORT: int = Field(8000, env="SERVER_PORT")

    # --- CORS ---
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = Field(
        default=[], env="BACKEND_CORS_ORIGINS"
    )

    @validator("BACKEND_CORS_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v: Union[str, List[str]]) -> Union[List[str], str]:
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, (list, str)):
            return v
        raise ValueError(v)

    # --- Database ---
    DATABASE_URL: str = Field(default="postgresql://sutazai:sutazai_secure_password@postgres:5432/sutazai", env="DATABASE_URL")
    REDIS_URL: str = Field(default="redis://redis:6379", env="REDIS_URL")
    MONGODB_URL: str = Field(default="mongodb://sutazai:sutazai_mongo_password@mongodb:27017/sutazai", env="MONGODB_URL")

    # --- Vector Database URLs ---
    CHROMADB_URL: str = Field(default="http://chromadb:8000", env="CHROMADB_URL")
    QDRANT_URL: str = Field(default="http://qdrant:6333", env="QDRANT_URL")
    FAISS_URL: str = Field(default="http://faiss:8088", env="FAISS_URL")

    # --- AI Model Services ---
    OLLAMA_URL: str = Field(default="http://ollama:11434", env="OLLAMA_URL")

    # --- AI Agent Services ---
    AUTOGPT_URL: str = Field(default="http://autogpt:8000", env="AUTOGPT_URL")
    LOCALAGI_URL: str = Field(default="http://localagi:8080", env="LOCALAGI_URL")
    TABBYML_URL: str = Field(default="http://tabbyml:8080", env="TABBYML_URL")
    AGENTZERO_URL: str = Field(default="http://agentzero:8000", env="AGENTZERO_URL")
    BIGAGI_URL: str = Field(default="http://bigagi:3000", env="BIGAGI_URL")

    # --- Web Automation Services ---
    BROWSER_USE_URL: str = Field(default="http://browser-use:8080", env="BROWSER_USE_URL")
    SKYVERN_URL: str = Field(default="http://skyvern:8080", env="SKYVERN_URL")

    # --- Document Processing ---
    DOCUMIND_URL: str = Field(default="http://documind:8080", env="DOCUMIND_URL")

    # --- Financial Analysis ---
    FINROBOT_URL: str = Field(default="http://finrobot:8080", env="FINROBOT_URL")

    # --- Code Generation Services ---
    GPT_ENGINEER_URL: str = Field(default="http://gpt-engineer:8080", env="GPT_ENGINEER_URL")
    AIDER_URL: str = Field(default="http://aider:8080", env="AIDER_URL")

    # --- Framework Services ---
    LANGFLOW_URL: str = Field(default="http://langflow:7860", env="LANGFLOW_URL")
    DIFY_URL: str = Field(default="http://dify:5001", env="DIFY_URL")
    PYTORCH_URL: str = Field(default="http://pytorch-service:8080", env="PYTORCH_URL")
    TENSORFLOW_URL: str = Field(default="http://tensorflow-service:8080", env="TENSORFLOW_URL")
    JAX_URL: str = Field(default="http://jax-service:8080", env="JAX_URL")

    # --- Specialized Services ---
    AWESOME_CODE_AI_URL: str = Field(default="http://awesome-code-ai:8089", env="AWESOME_CODE_AI_URL")

    # --- Performance Settings ---
    MAX_CONCURRENT_REQUESTS: int = Field(default=100, env="MAX_CONCURRENT_REQUESTS")
    WORKER_COUNT: int = Field(default=4, env="WORKER_COUNT")

    # --- Cache Settings ---
    CACHE_TTL: int = Field(default=300, env="CACHE_TTL")  # 5 minutes
    CACHE_MAX_SIZE: int = Field(default=10000, env="CACHE_MAX_SIZE")

    # --- Model Settings ---
    DEFAULT_MODEL: str = Field(default="deepseek-coder:6.7b", env="DEFAULT_MODEL")
    MAX_LOADED_MODELS: int = Field(default=5, env="MAX_LOADED_MODELS")
    MODEL_TIMEOUT: int = Field(default=300, env="MODEL_TIMEOUT")  # 5 minutes

    # --- Agent Settings ---
    MAX_ACTIVE_AGENTS: int = Field(default=10, env="MAX_ACTIVE_AGENTS")
    AGENT_TIMEOUT: int = Field(default=600, env="AGENT_TIMEOUT")  # 10 minutes

    # --- Paths ---
    LOG_DIR: Path = Field(PROJECT_ROOT / "logs")
    DATA_DIR: Path = Field(PROJECT_ROOT / "data")
    UPLOAD_DIR: Path = Field(DATA_DIR / "uploads")
    DOCUMENT_DIR: Path = Field(DATA_DIR / "documents")
    CONFIG_DIR: Path = Field(PROJECT_ROOT / "backend/config")
    WORKSPACE_PATH: str = Field(default="/workspace", env="WORKSPACE_PATH")
    MODELS_PATH: str = Field(default="/models", env="MODELS_PATH")
    DOCUMENTS_PATH: str = Field(default="/documents", env="DOCUMENTS_PATH")
    LOGS_PATH: str = Field(default="/logs", env="LOGS_PATH")

    # --- File Uploads ---
    SUPPORTED_DOC_TYPES: str = Field("pdf,docx,txt,md,csv", env="SUPPORTED_DOC_TYPES")

    # --- Logging ---
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    # --- Other Settings ---
    SQLALCHEMY_ECHO: bool = Field(False, env="SQLALCHEMY_ECHO")
    SERVER_NAME: Optional[str] = Field(None, env="SERVER_NAME")
    SSL_REDIRECT: bool = Field(False, env="SSL_REDIRECT")

    # --- JWT Settings ---
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(60 * 24 * 8, env="ACCESS_TOKEN_EXPIRE_MINUTES")
    ALGORITHM: str = "HS256"

    # --- Monitoring ---
    METRICS_ENABLED: bool = Field(default=True, env="METRICS_ENABLED")
    HEALTH_CHECK_INTERVAL: int = Field(default=60, env="HEALTH_CHECK_INTERVAL")

    # --- Backup Settings ---
    BACKUP_ENABLED: bool = Field(default=True, env="BACKUP_ENABLED")
    BACKUP_INTERVAL: int = Field(default=3600, env="BACKUP_INTERVAL")
    BACKUP_RETENTION_DAYS: int = Field(default=30, env="BACKUP_RETENTION_DAYS")

    @property
    def all_service_urls(self) -> Dict[str, str]:
        """Get all service URLs"""
        return {
            "ollama": self.OLLAMA_URL,
            "autogpt": self.AUTOGPT_URL,
            "localagi": self.LOCALAGI_URL,
            "tabbyml": self.TABBYML_URL,
            "agentzero": self.AGENTZERO_URL,
            "bigagi": self.BIGAGI_URL,
            "browser_use": self.BROWSER_USE_URL,
            "skyvern": self.SKYVERN_URL,
            "documind": self.DOCUMIND_URL,
            "finrobot": self.FINROBOT_URL,
            "gpt_engineer": self.GPT_ENGINEER_URL,
            "aider": self.AIDER_URL,
            "langflow": self.LANGFLOW_URL,
            "dify": self.DIFY_URL,
            "pytorch": self.PYTORCH_URL,
            "tensorflow": self.TENSORFLOW_URL,
            "jax": self.JAX_URL,
            "awesome_code_ai": self.AWESOME_CODE_AI_URL,
            "chromadb": self.CHROMADB_URL,
            "qdrant": self.QDRANT_URL,
            "faiss": self.FAISS_URL,
        }

    class Config:
        case_sensitive = True
        env_file = PROJECT_ROOT / ".env"
        env_file_encoding = 'utf-8'

@lru_cache() # Cache the settings object for performance
def get_settings() -> AppSettings:
    logger.info("Loading application settings...")
    try:
        settings = AppSettings()
        # Create directories if they don't exist
        settings.LOG_DIR.mkdir(parents=True, exist_ok=True)
        settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
        settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        settings.DOCUMENT_DIR.mkdir(parents=True, exist_ok=True)
        settings.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        
        # Create additional directories for container paths
        from pathlib import Path
        Path(settings.WORKSPACE_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.MODELS_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.DOCUMENTS_PATH).mkdir(parents=True, exist_ok=True)
        Path(settings.LOGS_PATH).mkdir(parents=True, exist_ok=True)
        
        logger.info("Application settings loaded successfully.")
        return settings
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to load application settings: {e}", exc_info=True)
        # Create a minimal settings object for development
        try:
            settings = AppSettings(
                SECRET_KEY="sutazai_dev_secret_key_change_in_production",
                JWT_SECRET="sutazai_dev_jwt_secret_change_in_production"
            )
            logger.warning("Using development settings due to configuration error")
            return settings
        except:
            raise ValueError(f"Could not load settings. Check environment variables and .env file. Error: {e}")

# Global settings instance
settings = get_settings() 