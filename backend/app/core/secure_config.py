"""
Secure Configuration Module for SutazAI Backend
Centralizes all environment variables and secrets management
Following enterprise security best practices
"""

import os
import logging
from typing import Optional, Any
from functools import lru_cache

logger = logging.getLogger(__name__)


class SecurityException(Exception):
    """Custom exception for security configuration issues"""
    pass


class SecureConfig:
    """
    Centralized secure configuration management
    All sensitive values MUST be loaded from environment variables
    NO hardcoded secrets allowed in production code
    """
    
    def __init__(self):
        """Initialize secure configuration with validation"""
        self._validate_required_vars()
        self._warn_insecure_defaults()
    
    def _validate_required_vars(self):
        """Validate that critical environment variables are set"""
        required_vars = [
            "POSTGRES_USER",
            "POSTGRES_DB",
        ]
        
        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)
        
        if missing_vars and os.getenv("SUTAZAI_ENV") == "production":
            raise SecurityException(
                f"Critical environment variables missing in production: {', '.join(missing_vars)}"
            )
    
    def _warn_insecure_defaults(self):
        """Warn about insecure default values being used"""
        if os.getenv("SUTAZAI_ENV") == "production":
            # Check for default passwords
            if self.postgres_password == "sutazai123":
                logger.critical("SECURITY WARNING: Default PostgreSQL password detected in production!")
            if self.jwt_secret == "dev-secret-key":
                logger.critical("SECURITY WARNING: Default JWT secret detected in production!")
            if self.secret_key == "dev-secret-key":
                logger.critical("SECURITY WARNING: Default secret key detected in production!")
    
    # Database Configuration
    @property
    def postgres_host(self) -> str:
        """PostgreSQL host"""
        return os.getenv("POSTGRES_HOST", "sutazai-postgres")
    
    @property
    def postgres_port(self) -> int:
        """PostgreSQL port"""
        return int(os.getenv("POSTGRES_PORT", "5432"))
    
    @property
    def postgres_user(self) -> str:
        """PostgreSQL user"""
        return os.getenv("POSTGRES_USER", "sutazai")
    
    @property
    def postgres_password(self) -> str:
        """PostgreSQL password - MUST be set via environment variable in production"""
        password = os.getenv("POSTGRES_PASSWORD")
        if not password:
            if os.getenv("SUTAZAI_ENV") == "production":
                raise SecurityException("POSTGRES_PASSWORD must be set in production")
            # Only use default in development
            password = "sutazai123"
            logger.warning("Using default PostgreSQL password - ONLY for development!")
        return password
    
    @property
    def postgres_db(self) -> str:
        """PostgreSQL database name"""
        return os.getenv("POSTGRES_DB", "sutazai")
    
    @property
    def database_url(self) -> str:
        """Constructed PostgreSQL connection URL"""
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    @property
    def sync_database_url(self) -> str:
        """Synchronous PostgreSQL connection URL"""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )
    
    # Redis Configuration
    @property
    def redis_host(self) -> str:
        """Redis host"""
        return os.getenv("REDIS_HOST", "sutazai-redis")
    
    @property
    def redis_port(self) -> int:
        """Redis port"""
        return int(os.getenv("REDIS_PORT", "6379"))
    
    @property
    def redis_password(self) -> Optional[str]:
        """Redis password (optional)"""
        return os.getenv("REDIS_PASSWORD")
    
    @property
    def redis_url(self) -> str:
        """Constructed Redis connection URL"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/0"
        return f"redis://{self.redis_host}:{self.redis_port}/0"
    
    # Neo4j Configuration
    @property
    def neo4j_uri(self) -> str:
        """Neo4j connection URI"""
        host = os.getenv("NEO4J_HOST", "sutazai-neo4j")
        port = os.getenv("NEO4J_PORT", "7687")
        return f"bolt://{host}:{port}"
    
    @property
    def neo4j_user(self) -> str:
        """Neo4j username"""
        return os.getenv("NEO4J_USER", "neo4j")
    
    @property
    def neo4j_password(self) -> str:
        """Neo4j password - MUST be set via environment variable in production"""
        password = os.getenv("NEO4J_PASSWORD")
        if not password:
            if os.getenv("SUTAZAI_ENV") == "production":
                raise SecurityException("NEO4J_PASSWORD must be set in production")
            # Only use default in development
            password = "sutazai_neo4j_password"
            logger.warning("Using default Neo4j password - ONLY for development!")
        return password
    
    # ChromaDB Configuration
    @property
    def chromadb_host(self) -> str:
        """ChromaDB host"""
        return os.getenv("CHROMADB_HOST", "sutazai-chromadb")
    
    @property
    def chromadb_port(self) -> int:
        """ChromaDB port"""
        return int(os.getenv("CHROMADB_PORT", "8000"))
    
    @property
    def chromadb_url(self) -> str:
        """ChromaDB API URL"""
        return f"http://{self.chromadb_host}:{self.chromadb_port}"
    
    @property
    def chromadb_api_key(self) -> Optional[str]:
        """ChromaDB API key - MUST be set via environment variable in production"""
        api_key = os.getenv("CHROMADB_API_KEY")
        if not api_key:
            if os.getenv("SUTAZAI_ENV") == "production":
                logger.warning("CHROMADB_API_KEY not set - ChromaDB may be unsecured")
            return None
        return api_key
    
    # Qdrant Configuration
    @property
    def qdrant_host(self) -> str:
        """Qdrant host"""
        return os.getenv("QDRANT_HOST", "sutazai-qdrant")
    
    @property
    def qdrant_port(self) -> int:
        """Qdrant port"""
        return int(os.getenv("QDRANT_PORT", "6333"))
    
    @property
    def qdrant_url(self) -> str:
        """Qdrant API URL"""
        return f"http://{self.qdrant_host}:{self.qdrant_port}"
    
    # Security Configuration
    @property
    def secret_key(self) -> str:
        """Application secret key for session management"""
        key = os.getenv("SECRET_KEY")
        if not key:
            if os.getenv("SUTAZAI_ENV") == "production":
                raise SecurityException("SECRET_KEY must be set in production")
            # Only use default in development
            key = "dev-secret-key"
            logger.warning("Using default secret key - ONLY for development!")
        return key
    
    @property
    def jwt_secret(self) -> str:
        """JWT secret key for token signing"""
        secret = os.getenv("JWT_SECRET", os.getenv("JWT_SECRET_KEY"))
        if not secret:
            if os.getenv("SUTAZAI_ENV") == "production":
                raise SecurityException("JWT_SECRET must be set in production")
            # Only use default in development
            secret = "dev-secret-key"
            logger.warning("Using default JWT secret - ONLY for development!")
        return secret
    
    @property
    def jwt_algorithm(self) -> str:
        """JWT signing algorithm"""
        return os.getenv("JWT_ALGORITHM", "HS256")
    
    @property
    def jwt_expiry_minutes(self) -> int:
        """JWT token expiry in minutes"""
        return int(os.getenv("JWT_EXPIRY_MINUTES", "30"))
    
    # Ollama Configuration
    @property
    def ollama_host(self) -> str:
        """Ollama AI model server host"""
        return os.getenv("OLLAMA_HOST", "sutazai-ollama")
    
    @property
    def ollama_port(self) -> int:
        """Ollama AI model server port"""
        return int(os.getenv("OLLAMA_PORT", "11434"))
    
    @property
    def ollama_url(self) -> str:
        """Ollama API URL"""
        return f"http://{self.ollama_host}:{self.ollama_port}"
    
    # AgentOps Configuration
    @property
    def agentops_api_key(self) -> Optional[str]:
        """AgentOps API key for agent debugging"""
        return os.getenv("AGENTOPS_API_KEY")
    
    @property
    def agentops_endpoint(self) -> str:
        """AgentOps endpoint URL"""
        return os.getenv("AGENTOPS_ENDPOINT", "http://localhost:8000")
    
    # Application Configuration
    @property
    def sutazai_env(self) -> str:
        """Application environment (development/staging/production)"""
        return os.getenv("SUTAZAI_ENV", "development")
    
    @property
    def debug(self) -> bool:
        """Debug mode flag"""
        return os.getenv("DEBUG", "false").lower() == "true"
    
    @property
    def log_level(self) -> str:
        """Application log level"""
        return os.getenv("LOG_LEVEL", "INFO")
    
    @property
    def cors_origins(self) -> list:
        """CORS allowed origins"""
        origins = os.getenv("CORS_ORIGINS", "http://localhost:10011,http://localhost:3000")
        return [origin.strip() for origin in origins.split(",")]
    
    def get_safe_config_dict(self) -> dict:
        """
        Get configuration dictionary with sensitive values masked
        Used for logging and debugging without exposing secrets
        """
        return {
            "sutazai_env": self.sutazai_env,
            "postgres_host": self.postgres_host,
            "postgres_port": self.postgres_port,
            "postgres_user": self.postgres_user,
            "postgres_password": "***MASKED***",
            "postgres_db": self.postgres_db,
            "redis_host": self.redis_host,
            "redis_port": self.redis_port,
            "redis_password": "***MASKED***" if self.redis_password else None,
            "neo4j_uri": self.neo4j_uri,
            "neo4j_user": self.neo4j_user,
            "neo4j_password": "***MASKED***",
            "chromadb_url": self.chromadb_url,
            "chromadb_api_key": "***MASKED***" if self.chromadb_api_key else None,
            "qdrant_url": self.qdrant_url,
            "ollama_url": self.ollama_url,
            "secret_key": "***MASKED***",
            "jwt_secret": "***MASKED***",
            "jwt_algorithm": self.jwt_algorithm,
            "jwt_expiry_minutes": self.jwt_expiry_minutes,
            "debug": self.debug,
            "log_level": self.log_level,
            "cors_origins": self.cors_origins,
        }


@lru_cache()
def get_secure_config() -> SecureConfig:
    """
    Get singleton instance of secure configuration
    Cached to avoid repeated environment variable lookups
    """
    return SecureConfig()


# Export convenience function
config = get_secure_config()