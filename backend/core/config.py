from typing import List, Optional, Union, Dict, Any
from pydantic import AnyHttpUrl, PostgresDsn, validator, Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

# Determine Project Root dynamically relative to this config file
# Assuming this file is at backend/core/config.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

class AppSettings(BaseSettings):
    # --- Core Settings ---
    PROJECT_NAME: str = "SutazAI"
    PROJECT_VERSION: str = "1.0.0" # Default version, could be loaded from version.txt
    DEBUG_MODE: bool = Field(False, env="DEBUG_MODE")
    SECRET_KEY: str = Field("dev-secret-key-change-in-production", env="SECRET_KEY")

    # --- API Settings ---
    API_V1_STR: str = "/api/v1"
    SERVER_HOST: str = Field("0.0.0.0", env="SERVER_HOST")
    SERVER_PORT: int = Field(8000, env="SERVER_PORT")

    # --- CORS ---
    # BACKEND_CORS_ORIGINS is a JSON-formatted list of origins
    # e.g., '["http://localhost", "http://localhost:4200", "http://localhost:3000"]'
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
    POSTGRES_SERVER: str = Field("postgres", env="POSTGRES_HOST")
    POSTGRES_USER: str = Field("sutazai", env="POSTGRES_USER") 
    POSTGRES_PASSWORD: str = Field("sutazai_password", env="POSTGRES_PASSWORD")
    POSTGRES_DB: str = Field("sutazai", env="POSTGRES_DB")
    DATABASE_URL: Optional[str] = Field(None, env="DATABASE_URL")
    DATABASE_URI: Optional[PostgresDsn] = None # Assembled below

    @validator("DATABASE_URI", pre=True)
    def assemble_db_connection(cls, v: Optional[str], values: Dict[str, Any]) -> Any:
        # If DATABASE_URL is provided, use it directly
        if values.get("DATABASE_URL"):
            return values.get("DATABASE_URL")
        # Otherwise build from components
        if isinstance(v, str):
            return v
        return PostgresDsn.build(
            scheme="postgresql+asyncpg", # Use asyncpg driver
            username=values.get("POSTGRES_USER"),
            password=values.get("POSTGRES_PASSWORD"),
            host=values.get("POSTGRES_SERVER"),
            path=f"/{values.get('POSTGRES_DB') or ''}",
        )

    # --- Paths ---
    # Define paths relative to the project root
    LOG_DIR: Path = Field(default_factory=lambda: PROJECT_ROOT / "logs")
    DATA_DIR: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    UPLOAD_DIR: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "uploads")
    DOCUMENT_DIR: Path = Field(default_factory=lambda: PROJECT_ROOT / "data" / "documents")
    CONFIG_DIR: Path = Field(default_factory=lambda: PROJECT_ROOT / "backend" / "config")

    # --- File Uploads ---
    SUPPORTED_DOC_TYPES: str = Field("pdf,docx,txt,md,csv", env="SUPPORTED_DOC_TYPES") # Comma-separated

    # --- Logging ---
    LOG_LEVEL: str = Field("INFO", env="LOG_LEVEL")

    # --- Other Settings from old config (adapt as needed) ---
    SQLALCHEMY_ECHO: bool = Field(False, env="SQLALCHEMY_ECHO") # For dev debug
    SERVER_NAME: Optional[str] = Field(None, env="SERVER_NAME") # For specific deployments
    SSL_REDIRECT: bool = Field(False, env="SSL_REDIRECT")

    # --- JWT Settings (Example) ---
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(60 * 24 * 8, env="ACCESS_TOKEN_EXPIRE_MINUTES") # 8 days
    ALGORITHM: str = "HS256"

    # Pydantic v2 settings configuration
    model_config = SettingsConfigDict(
        case_sensitive=True,
        env_file=str(PROJECT_ROOT / ".env"),  # Load .env from project root
        env_file_encoding='utf-8',
        extra="ignore",  # Ignore unrelated env vars (prevents ValidationError)
    )

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
        # Note: Config dir likely exists, but doesn't hurt to ensure
        settings.CONFIG_DIR.mkdir(parents=True, exist_ok=True) 
        logger.info("Application settings loaded successfully.")
        return settings
    except Exception as e:
        logger.critical(f"CRITICAL: Failed to load application settings: {e}", exc_info=True)
        # Depending on policy, you might exit here or raise a specific configuration error
        raise ValueError(f"Could not load settings. Check environment variables and .env file. Error: {e}")

# Instantiate once at module level for convenience if needed elsewhere,
# but prefer using the get_settings() function via Depends for FastAPI routes.
settings = get_settings() 
