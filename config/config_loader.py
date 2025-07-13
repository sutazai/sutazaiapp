"""
Secure Configuration Loader for SutazAI
Handles environment variables and secure configuration loading
"""

import os
import sys
import toml
import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

class ConfigLoader:
    """Secure configuration loader with environment variable support"""

    def __init__(self, base_path: Optional[str] = None):
        self.base_path = Path(base_path) if base_path else Path(__file__).parent.parent
        self.config_path = self.base_path / "config"
        self._load_environment()

    def _load_environment(self):
        """Load environment variables from .env file"""
        env_file = self.base_path / ".env"

        if env_file.exists():
            load_dotenv(env_file)
            logger.info(f"Loaded environment variables from {env_file}")
        else:
            logger.warning(f".env file not found at {env_file}")
            logger.warning("Using environment variables from system only")

    def load_orchestrator_config(self) -> Dict[str, Any]:
        """Load orchestrator configuration with environment variable substitution"""
        config_file = self.config_path / "orchestrator.toml"

        if not config_file.exists():
            raise FileNotFoundError(f"Orchestrator config not found: {config_file}")

        # Load base configuration
        with open(config_file, 'r') as f:
            config = toml.load(f)

        # Apply environment variable substitutions
        self._substitute_env_vars(config)

        return config

    def _substitute_env_vars(self, config: Dict[str, Any]):
        """Recursively substitute environment variables in configuration"""

        # Primary server API key
        if 'primary_server' in config:
            api_key = os.getenv('PRIMARY_API_KEY')
            if api_key:
                config['primary_server']['api_key'] = api_key
            else:
                logger.error("PRIMARY_API_KEY environment variable not set")
                raise ValueError("PRIMARY_API_KEY is required")

        # Secondary server API key
        if 'secondary_server' in config:
            api_key = os.getenv('SECONDARY_API_KEY')
            if api_key:
                config['secondary_server']['api_key'] = api_key
            else:
                logger.error("SECONDARY_API_KEY environment variable not set")
                raise ValueError("SECONDARY_API_KEY is required")

        # Database password
        if 'database' in config:
            password = os.getenv('POSTGRES_PASSWORD')
            if password:
                config['database']['password'] = password
            else:
                logger.error("POSTGRES_PASSWORD environment variable not set")
                raise ValueError("POSTGRES_PASSWORD is required")

        # Redis password
        if 'redis' in config:
            password = os.getenv('REDIS_PASSWORD')
            if password:
                config['redis']['password'] = password
            # Redis password is optional, don't raise error if not set

        # Application secret key
        secret_key = os.getenv('SECRET_KEY')
        if secret_key:
            config['secret_key'] = secret_key
        else:
            logger.error("SECRET_KEY environment variable not set")
            raise ValueError("SECRET_KEY is required")

    def get_database_url(self) -> str:
        """Get database URL from environment or construct from config"""
        # Try to get full DATABASE_URL first
        db_url = os.getenv('DATABASE_URL')
        if db_url:
            return db_url

        # Construct from individual components
        db_host = os.getenv('DB_HOST', 'localhost')
        db_port = os.getenv('DB_PORT', '5432')
        db_name = os.getenv('DB_NAME', 'sutazai')
        db_user = os.getenv('DB_USER', 'sutazai')
        db_password = os.getenv('POSTGRES_PASSWORD')

        if not db_password:
            raise ValueError("Database password not configured (POSTGRES_PASSWORD or DATABASE_URL required)")

        return f"postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

    def get_redis_url(self) -> str:
        """Get Redis URL from environment or construct from config"""
        # Try to get full REDIS_URL first
        redis_url = os.getenv('REDIS_URL')
        if redis_url:
            return redis_url

        # Construct from individual components
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = os.getenv('REDIS_PORT', '6379')
        redis_password = os.getenv('REDIS_PASSWORD')
        redis_db = os.getenv('REDIS_DB', '0')

        if redis_password:
            return f"redis://:{redis_password}@{redis_host}:{redis_port}/{redis_db}"
        else:
            return f"redis://{redis_host}:{redis_port}/{redis_db}"

    def get_smtp_config(self) -> Dict[str, Any]:
        """Get SMTP configuration from environment"""
        return {
            'server': os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            'port': int(os.getenv('SMTP_PORT', '587')),
            'username': os.getenv('SMTP_USER'),
            'password': os.getenv('SMTP_PASSWORD'),
            'use_tls': os.getenv('SMTP_USE_TLS', 'true').lower() == 'true'
        }

    def get_ssh_config(self) -> Dict[str, Any]:
        """Get SSH configuration for deployment"""
        return {
            'user': os.getenv('SSH_USER', 'sutazaiapp_dev'),
            'host': os.getenv('DEPLOY_SERVER', '192.168.100.100'),
            'key_path': os.getenv('SSH_KEY_PATH', '/root/.ssh/sutazaiapp_sync_key')
        }

    def get_app_config(self) -> Dict[str, Any]:
        """Get general application configuration"""
        return {
            'debug': os.getenv('DEBUG', 'false').lower() == 'true',
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'environment': os.getenv('ENVIRONMENT', 'production'),
            'secret_key': os.getenv('SECRET_KEY'),
            'admin_email': os.getenv('ADMIN_EMAIL'),
            'admin_phone': os.getenv('ADMIN_PHONE')
        }

    def validate_required_vars(self) -> bool:
        """Validate that all required environment variables are set"""
        required_vars = [
            'PRIMARY_API_KEY',
            'SECONDARY_API_KEY',
            'SECRET_KEY',
            'POSTGRES_PASSWORD'
        ]

        missing_vars = []
        for var in required_vars:
            if not os.getenv(var):
                missing_vars.append(var)

        if missing_vars:
            logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
            return False

        return True

    def generate_secure_config(self) -> Dict[str, str]:
        """Generate secure random values for configuration"""
        import secrets
        import string

        def generate_password(length=32):
            alphabet = string.ascii_letters + string.digits + "!@#$%^&*"
            return ''.join(secrets.choice(alphabet) for _ in range(length))

        def generate_api_key(length=64):
            return secrets.token_urlsafe(length)

        return {
            'PRIMARY_API_KEY': generate_api_key(),
            'SECONDARY_API_KEY': generate_api_key(),
            'SECRET_KEY': generate_api_key(),
            'POSTGRES_PASSWORD': generate_password(),
            'REDIS_PASSWORD': generate_password()
        }

    def create_env_file(self, secure_values: Optional[Dict[str, str]] = None):
        """Create .env file with secure values"""
        env_file = self.base_path / ".env"

        if env_file.exists():
            logger.warning(f".env file already exists at {env_file}")
            return

        if not secure_values:
            secure_values = self.generate_secure_config()

        env_content = f"""# SutazAI Environment Configuration
# Generated on {os.path.now().isoformat()}
# WARNING: Keep this file secure and never commit to version control

# Database Configuration
DATABASE_URL=postgresql://sutazai:{secure_values['POSTGRES_PASSWORD']}@localhost:5432/sutazai
POSTGRES_PASSWORD={secure_values['POSTGRES_PASSWORD']}

# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD={secure_values['REDIS_PASSWORD']}

# API Keys and Secrets
PRIMARY_API_KEY={secure_values['PRIMARY_API_KEY']}
SECONDARY_API_KEY={secure_values['SECONDARY_API_KEY']}
SECRET_KEY={secure_values['SECRET_KEY']}

# SMTP Configuration (update with your values)
SMTP_USER=your_smtp_username
SMTP_PASSWORD=your_smtp_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587

# SSH and Deployment Configuration
SSH_USER=sutazaiapp_dev
DEPLOY_SERVER=192.168.100.100
SSH_KEY_PATH=/root/.ssh/sutazaiapp_sync_key

# Application Settings
DEBUG=false
LOG_LEVEL=INFO
ENVIRONMENT=production

# Administrative Contact (for alerts only)
ADMIN_EMAIL=admin@yourcompany.com
ADMIN_PHONE=+1234567890
"""

        with open(env_file, 'w') as f:
            f.write(env_content)

        # Set secure permissions
        os.chmod(env_file, 0o600)

        logger.info(f"Created .env file with secure values at {env_file}")
        logger.warning("Please update SMTP and administrative contact information in .env file")

# Global configuration loader instance
config_loader = ConfigLoader()

def get_config() -> ConfigLoader:
    """Get the global configuration loader instance"""
    return config_loader

def load_orchestrator_config() -> Dict[str, Any]:
    """Convenience function to load orchestrator configuration"""
    return config_loader.load_orchestrator_config()

def get_database_url() -> str:
    """Convenience function to get database URL"""
    return config_loader.get_database_url()

def get_redis_url() -> str:
    """Convenience function to get Redis URL"""
    return config_loader.get_redis_url()
