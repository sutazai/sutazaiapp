import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings:
    """
    Configuration settings for the SutazAI backend.
    Loads settings from environment variables with sensible defaults.
    """
    # Database settings
    database_url = os.getenv(
        "DATABASE_URL", 
        "postgresql://sutazai_user:sutazai_postgres_password@postgres:5432/sutazai"
    )
    
    # Redis settings
    redis_url = os.getenv(
        "REDIS_URL", 
        "redis://:sutazai_redis_password@redis:6379/0"
    )
    
    # Application settings
    debug = os.getenv("DEBUG", "false").lower() == "true"
    api_port = int(os.getenv("PORT", 8000))
    model_port = int(os.getenv("MODEL_PORT", 8001))
    
    # Security settings
    secret_key = os.getenv("SECRET_KEY", "default_secret_key")
    algorithm = os.getenv("ALGORITHM", "HS256")
    access_token_expire_minutes = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
    
    # GPU settings
    gpu_enabled = os.getenv("GPU_ENABLED", "false").lower() == "true"
    
    # Logging settings
    log_level = os.getenv("LOG_LEVEL", "INFO")
    
    # Model settings
    model_path = os.getenv(
        "MODEL_PATH", 
        "/models/DeepSeek-Coder-33B/ggml-model-q4_0.gguf"
    )

# Create a singleton settings instance
settings = Settings()