#!/usr/bin/env python3
"""
Multi-Environment Configuration Manager for SutazAI
Version: 1.0.0

DESCRIPTION:
    Sophisticated configuration management system that handles multiple
    environments (dev, staging, production) with secure secret management,
    environment-specific overrides, and configuration validation.

PURPOSE:
    - Manage configurations across multiple environments
    - Secure secret handling and rotation
    - Environment-specific configuration overrides
    - Configuration drift detection and remediation
    - Automated configuration deployment and validation
    - Configuration versioning and rollback

USAGE:
    python multi-environment-config-manager.py [command] [options]

REQUIREMENTS:
    - Python 3.8+
    - Cryptography for secret encryption
    - YAML and JSON support
    - File system access
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
import hashlib
import base64
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set, Union
import tempfile
import shutil
from dataclasses import dataclass, asdict, field
import yaml
import subprocess
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
CONFIG_DIR = PROJECT_ROOT / "config"
ENVIRONMENTS_DIR = CONFIG_DIR / "environments"
SECRETS_DIR = PROJECT_ROOT / "secrets"
TEMPLATES_DIR = CONFIG_DIR / "templates"
LOG_DIR = PROJECT_ROOT / "logs"
CONFIG_DB = LOG_DIR / "config_management.db"

# Ensure directories exist
for directory in [CONFIG_DIR, ENVIRONMENTS_DIR, SECRETS_DIR, TEMPLATES_DIR, LOG_DIR]:
    directory.mkdir(exist_ok=True)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / "config-manager.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TEST = "test"
    LOCAL = "local"

class ConfigType(Enum):
    """Configuration types"""
    APPLICATION = "application"
    DATABASE = "database"
    SECRETS = "secrets"
    NETWORKING = "networking"
    MONITORING = "monitoring"
    AI_MODELS = "ai_models"
    INFRASTRUCTURE = "infrastructure"

class SecretType(Enum):
    """Secret types"""
    PASSWORD = "password"
    API_KEY = "api_key"
    CERTIFICATE = "certificate"
    TOKEN = "token"
    CONNECTION_STRING = "connection_string"

@dataclass
class ConfigValue:
    """Individual configuration value"""
    key: str
    value: Any
    value_type: str
    is_secret: bool = False
    environment_specific: bool = True
    description: str = ""
    required: bool = True
    validation_rules: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)

@dataclass
class SecretValue:
    """Encrypted secret value"""
    key: str
    encrypted_value: str
    secret_type: SecretType
    rotation_interval: timedelta
    last_rotated: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EnvironmentConfig:
    """Complete environment configuration"""
    environment: Environment
    name: str
    config_values: Dict[str, ConfigValue]
    secrets: Dict[str, SecretValue]
    overrides: Dict[str, Any]
    version: str
    created_at: datetime
    updated_at: datetime
    checksum: str
    validation_errors: List[str] = field(default_factory=list)

@dataclass
class ConfigTemplate:
    """Configuration template"""
    name: str
    config_type: ConfigType
    template_content: str
    variables: List[str]
    required_secrets: List[str]
    validation_schema: Dict[str, Any]
    description: str

class SecretManager:
    """Secure secret management with encryption"""
    
    def __init__(self, master_key: Optional[str] = None):
        self.master_key = master_key or self._get_or_create_master_key()
        self.cipher = self._create_cipher()
    
    def _get_or_create_master_key(self) -> str:
        """Get or create master encryption key"""
        key_file = SECRETS_DIR / ".master.key"
        
        if key_file.exists():
            with open(key_file, 'rb') as f:
                return f.read().decode()
        else:
            # Generate new master key
            key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(key)
            
            # Secure the key file
            os.chmod(key_file, 0o600)
            logger.info("Generated new master encryption key")
            return key.decode()
    
    def _create_cipher(self) -> Fernet:
        """Create cipher from master key"""
        if isinstance(self.master_key, str):
            key = self.master_key.encode()
        else:
            key = self.master_key
        
        return Fernet(key)
    
    def encrypt_secret(self, value: str) -> str:
        """Encrypt a secret value"""
        encrypted = self.cipher.encrypt(value.encode())
        return base64.b64encode(encrypted).decode()
    
    def decrypt_secret(self, encrypted_value: str) -> str:
        """Decrypt a secret value"""
        try:
            encrypted_bytes = base64.b64decode(encrypted_value.encode())
            decrypted = self.cipher.decrypt(encrypted_bytes)
            return decrypted.decode()
        except Exception as e:
            logger.error(f"Failed to decrypt secret: {e}")
            raise
    
    def generate_password(self, length: int = 32, include_symbols: bool = True) -> str:
        """Generate secure random password"""
        import secrets
        import string
        
        alphabet = string.ascii_letters + string.digits
        if include_symbols:
            alphabet += "!@#$%^&*"
        
        return ''.join(secrets.choice(alphabet) for _ in range(length))
    
    def generate_api_key(self, prefix: str = "sutazai") -> str:
        """Generate API key"""
        import secrets
        random_part = secrets.token_urlsafe(32)
        return f"{prefix}_{random_part}"

class ConfigDatabase:
    """SQLite database for configuration management"""
    
    def __init__(self, db_file: Path):
        self.db_file = db_file
        self._init_database()
    
    def _init_database(self):
        """Initialize database schema"""
        with sqlite3.connect(self.db_file) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS environments (
                    environment TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    version TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    checksum TEXT NOT NULL,
                    config_data TEXT NOT NULL,
                    validation_errors TEXT
                );
                
                CREATE TABLE IF NOT EXISTS config_values (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    environment TEXT NOT NULL,
                    key TEXT NOT NULL,
                    value TEXT,
                    value_type TEXT NOT NULL,
                    is_secret INTEGER DEFAULT 0,
                    environment_specific INTEGER DEFAULT 1,
                    description TEXT,
                    required INTEGER DEFAULT 1,
                    validation_rules TEXT,
                    tags TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (environment) REFERENCES environments (environment)
                );
                
                CREATE TABLE IF NOT EXISTS secrets (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    environment TEXT NOT NULL,
                    key TEXT NOT NULL,
                    encrypted_value TEXT NOT NULL,
                    secret_type TEXT NOT NULL,
                    rotation_interval INTEGER,
                    last_rotated TEXT NOT NULL,
                    metadata TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (environment) REFERENCES environments (environment)
                );
                
                CREATE TABLE IF NOT EXISTS config_templates (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL UNIQUE,
                    config_type TEXT NOT NULL,
                    template_content TEXT NOT NULL,
                    variables TEXT,
                    required_secrets TEXT,
                    validation_schema TEXT,
                    description TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
                
                CREATE TABLE IF NOT EXISTS config_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    environment TEXT NOT NULL,
                    version TEXT NOT NULL,
                    change_type TEXT NOT NULL,
                    changes TEXT NOT NULL,
                    changed_by TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (environment) REFERENCES environments (environment)
                );
                
                CREATE INDEX IF NOT EXISTS idx_config_values_env_key ON config_values(environment, key);
                CREATE INDEX IF NOT EXISTS idx_secrets_env_key ON secrets(environment, key);
                CREATE INDEX IF NOT EXISTS idx_config_history_env ON config_history(environment);
            """)
    
    def store_environment_config(self, env_config: EnvironmentConfig):
        """Store environment configuration"""
        with sqlite3.connect(self.db_file) as conn:
            # Store main environment record
            conn.execute("""
                INSERT OR REPLACE INTO environments
                (environment, name, version, created_at, updated_at, checksum, 
                 config_data, validation_errors)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                env_config.environment.value,
                env_config.name,
                env_config.version,
                env_config.created_at.isoformat(),
                env_config.updated_at.isoformat(),
                env_config.checksum,
                json.dumps(asdict(env_config), default=str),
                json.dumps(env_config.validation_errors)
            ))
            
            # Store config values
            conn.execute("DELETE FROM config_values WHERE environment = ?", 
                        (env_config.environment.value,))
            
            for key, config_value in env_config.config_values.items():
                conn.execute("""
                    INSERT INTO config_values
                    (environment, key, value, value_type, is_secret, environment_specific,
                     description, required, validation_rules, tags)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    env_config.environment.value,
                    key,
                    json.dumps(config_value.value) if not isinstance(config_value.value, str) else config_value.value,
                    config_value.value_type,
                    1 if config_value.is_secret else 0,
                    1 if config_value.environment_specific else 0,
                    config_value.description,
                    1 if config_value.required else 0,
                    json.dumps(config_value.validation_rules),
                    json.dumps(config_value.tags)
                ))
            
            # Store secrets
            conn.execute("DELETE FROM secrets WHERE environment = ?", 
                        (env_config.environment.value,))
            
            for key, secret in env_config.secrets.items():
                conn.execute("""
                    INSERT INTO secrets
                    (environment, key, encrypted_value, secret_type, rotation_interval,
                     last_rotated, metadata)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (
                    env_config.environment.value,
                    key,
                    secret.encrypted_value,
                    secret.secret_type.value,
                    int(secret.rotation_interval.total_seconds()),
                    secret.last_rotated.isoformat(),
                    json.dumps(secret.metadata)
                ))
    
    def get_environment_config(self, environment: Environment) -> Optional[EnvironmentConfig]:
        """Get environment configuration"""
        with sqlite3.connect(self.db_file) as conn:
            # Get main environment record
            row = conn.execute("""
                SELECT environment, name, version, created_at, updated_at, 
                       checksum, config_data, validation_errors
                FROM environments WHERE environment = ?
            """, (environment.value,)).fetchone()
            
            if not row:
                return None
            
            # Parse stored config data
            config_data = json.loads(row[6])
            
            # Get config values
            config_rows = conn.execute("""
                SELECT key, value, value_type, is_secret, environment_specific,
                       description, required, validation_rules, tags
                FROM config_values WHERE environment = ?
            """, (environment.value,)).fetchall()
            
            config_values = {}
            for config_row in config_rows:
                value = config_row[1]
                if config_row[2] in ['dict', 'list'] and isinstance(value, str):
                    try:
                        value = json.loads(value)
                    except:
                        pass
                
                config_values[config_row[0]] = ConfigValue(
                    key=config_row[0],
                    value=value,
                    value_type=config_row[2],
                    is_secret=bool(config_row[3]),
                    environment_specific=bool(config_row[4]),
                    description=config_row[5] or "",
                    required=bool(config_row[6]),
                    validation_rules=json.loads(config_row[7]) if config_row[7] else [],
                    tags=json.loads(config_row[8]) if config_row[8] else []
                )
            
            # Get secrets
            secret_rows = conn.execute("""
                SELECT key, encrypted_value, secret_type, rotation_interval,
                       last_rotated, metadata
                FROM secrets WHERE environment = ?
            """, (environment.value,)).fetchall()
            
            secrets = {}
            for secret_row in secret_rows:
                secrets[secret_row[0]] = SecretValue(
                    key=secret_row[0],
                    encrypted_value=secret_row[1],
                    secret_type=SecretType(secret_row[2]),
                    rotation_interval=timedelta(seconds=secret_row[3]),
                    last_rotated=datetime.fromisoformat(secret_row[4]),
                    metadata=json.loads(secret_row[5]) if secret_row[5] else {}
                )
            
            return EnvironmentConfig(
                environment=Environment(row[0]),
                name=row[1],
                config_values=config_values,
                secrets=secrets,
                overrides=config_data.get('overrides', {}),
                version=row[2],
                created_at=datetime.fromisoformat(row[3]),
                updated_at=datetime.fromisoformat(row[4]),
                checksum=row[5],
                validation_errors=json.loads(row[7]) if row[7] else []
            )
    
    def list_environments(self) -> List[str]:
        """List all configured environments"""
        with sqlite3.connect(self.db_file) as conn:
            rows = conn.execute("SELECT environment FROM environments ORDER BY environment").fetchall()
            return [row[0] for row in rows]
    
    def store_template(self, template: ConfigTemplate):
        """Store configuration template"""
        with sqlite3.connect(self.db_file) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO config_templates
                (name, config_type, template_content, variables, required_secrets,
                 validation_schema, description)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                template.name,
                template.config_type.value,
                template.template_content,
                json.dumps(template.variables),
                json.dumps(template.required_secrets),
                json.dumps(template.validation_schema),
                template.description
            ))
    
    def get_template(self, name: str) -> Optional[ConfigTemplate]:
        """Get configuration template"""
        with sqlite3.connect(self.db_file) as conn:
            row = conn.execute("""
                SELECT name, config_type, template_content, variables, required_secrets,
                       validation_schema, description
                FROM config_templates WHERE name = ?
            """, (name,)).fetchone()
            
            if not row:
                return None
            
            return ConfigTemplate(
                name=row[0],
                config_type=ConfigType(row[1]),
                template_content=row[2],
                variables=json.loads(row[3]) if row[3] else [],
                required_secrets=json.loads(row[4]) if row[4] else [],
                validation_schema=json.loads(row[5]) if row[5] else {},
                description=row[6] or ""
            )

class ConfigValidator:
    """Configuration validation and compliance checking"""
    
    def __init__(self):
        self.validation_rules = {
            'required': self._validate_required,
            'type': self._validate_type,
            'range': self._validate_range,
            'pattern': self._validate_pattern,
            'url': self._validate_url,
            'port': self._validate_port,
            'email': self._validate_email,
            'secure_password': self._validate_secure_password
        }
    
    def validate_environment_config(self, env_config: EnvironmentConfig) -> List[str]:
        """Validate complete environment configuration"""
        errors = []
        
        # Validate each config value
        for key, config_value in env_config.config_values.items():
            value_errors = self.validate_config_value(config_value)
            errors.extend([f"{key}: {error}" for error in value_errors])
        
        # Environment-specific validations
        env_errors = self._validate_environment_specific(env_config)
        errors.extend(env_errors)
        
        return errors
    
    def validate_config_value(self, config_value: ConfigValue) -> List[str]:
        """Validate individual configuration value"""
        errors = []
        
        # Check required values
        if config_value.required and (config_value.value is None or config_value.value == ""):
            errors.append("Value is required but not provided")
            return errors
        
        # Apply validation rules
        for rule in config_value.validation_rules:
            if isinstance(rule, str):
                rule_name = rule
                rule_params = {}
            elif isinstance(rule, dict):
                rule_name = list(rule.keys())[0]
                rule_params = rule[rule_name]
            else:
                continue
            
            if rule_name in self.validation_rules:
                try:
                    self.validation_rules[rule_name](config_value.value, rule_params)
                except ValueError as e:
                    errors.append(str(e))
        
        return errors
    
    def _validate_required(self, value: Any, params: Dict) -> None:
        """Validate required field"""
        if value is None or value == "":
            raise ValueError("Required value is missing")
    
    def _validate_type(self, value: Any, params: Dict) -> None:
        """Validate type"""
        expected_type = params.get('type', 'string')
        
        type_mapping = {
            'string': str,
            'integer': int,
            'float': float,
            'boolean': bool,
            'list': list,
            'dict': dict
        }
        
        if expected_type in type_mapping:
            if not isinstance(value, type_mapping[expected_type]):
                raise ValueError(f"Expected {expected_type}, got {type(value).__name__}")
    
    def _validate_range(self, value: Any, params: Dict) -> None:
        """Validate numeric range"""
        if isinstance(value, (int, float)):
            min_val = params.get('min')
            max_val = params.get('max')
            
            if min_val is not None and value < min_val:
                raise ValueError(f"Value {value} is below minimum {min_val}")
            
            if max_val is not None and value > max_val:
                raise ValueError(f"Value {value} is above maximum {max_val}")
    
    def _validate_pattern(self, value: Any, params: Dict) -> None:
        """Validate regex pattern"""
        import re
        
        if isinstance(value, str):
            pattern = params.get('pattern', '')
            if pattern and not re.match(pattern, value):
                raise ValueError(f"Value does not match pattern: {pattern}")
    
    def _validate_url(self, value: Any, params: Dict) -> None:
        """Validate URL format"""
        import re
        
        if isinstance(value, str):
            url_pattern = r'^https?://(?:[-\w.])+(?:[:\d]+)?(?:/(?:[\w/_.])*(?:\?(?:[\w&=%.])*)?(?:#(?:\w*))?)?$'
            if not re.match(url_pattern, value):
                raise ValueError("Invalid URL format")
    
    def _validate_port(self, value: Any, params: Dict) -> None:
        """Validate port number"""
        if isinstance(value, int):
            if not (1 <= value <= 65535):
                raise ValueError("Port must be between 1 and 65535")
    
    def _validate_email(self, value: Any, params: Dict) -> None:
        """Validate email format"""
        import re
        
        if isinstance(value, str):
            email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
            if not re.match(email_pattern, value):
                raise ValueError("Invalid email format")
    
    def _validate_secure_password(self, value: Any, params: Dict) -> None:
        """Validate password security"""
        if isinstance(value, str):
            min_length = params.get('min_length', 12)
            require_uppercase = params.get('require_uppercase', True)
            require_lowercase = params.get('require_lowercase', True)
            require_digits = params.get('require_digits', True)
            require_symbols = params.get('require_symbols', True)
            
            if len(value) < min_length:
                raise ValueError(f"Password must be at least {min_length} characters")
            
            if require_uppercase and not any(c.isupper() for c in value):
                raise ValueError("Password must contain uppercase letters")
            
            if require_lowercase and not any(c.islower() for c in value):
                raise ValueError("Password must contain lowercase letters")
            
            if require_digits and not any(c.isdigit() for c in value):
                raise ValueError("Password must contain digits")
            
            if require_symbols and not any(c in "!@#$%^&*()_+-=[]{}|;:,.<>?" for c in value):
                raise ValueError("Password must contain special characters")
    
    def _validate_environment_specific(self, env_config: EnvironmentConfig) -> List[str]:
        """Environment-specific validation rules"""
        errors = []
        
        # Production-specific validations
        if env_config.environment == Environment.PRODUCTION:
            # Check for debug settings
            debug_settings = ['DEBUG', 'ENABLE_DEBUG', 'LOG_LEVEL']
            for setting in debug_settings:
                if setting in env_config.config_values:
                    value = env_config.config_values[setting].value
                    if setting == 'DEBUG' and value not in [False, 'false', '0']:
                        errors.append("DEBUG mode should be disabled in production")
                    elif setting == 'LOG_LEVEL' and value in ['DEBUG', 'TRACE']:
                        errors.append("Log level should not be DEBUG or TRACE in production")
            
            # Check for secure connections
            ssl_settings = ['ENABLE_SSL', 'USE_TLS', 'SECURE_CONNECTIONS']
            for setting in ssl_settings:
                if setting in env_config.config_values:
                    value = env_config.config_values[setting].value
                    if value not in [True, 'true', '1']:
                        errors.append(f"Secure connections should be enabled in production: {setting}")
        
        # Check for required secrets
        required_secrets = ['DATABASE_PASSWORD', 'JWT_SECRET', 'ENCRYPTION_KEY']
        for secret_key in required_secrets:
            if secret_key not in env_config.secrets and secret_key not in env_config.config_values:
                errors.append(f"Required secret missing: {secret_key}")
        
        return errors

class TemplateEngine:
    """Configuration template processing engine"""
    
    def __init__(self, secret_manager: SecretManager):
        self.secret_manager = secret_manager
    
    def render_template(self, template: ConfigTemplate, variables: Dict[str, Any], 
                       environment: Environment) -> str:
        """Render configuration template with variables"""
        import jinja2
        
        # Create Jinja2 environment
        jinja_env = jinja2.Environment(
            loader=jinja2.BaseLoader(),
            undefined=jinja2.StrictUndefined
        )
        
        # Add custom filters
        jinja_env.filters['generate_password'] = self._generate_password_filter
        jinja_env.filters['generate_secret'] = self._generate_secret_filter
        jinja_env.filters['encrypt'] = self._encrypt_filter
        
        # Add environment-specific functions
        template_vars = {
            **variables,
            'environment': environment.value,
            'is_production': environment == Environment.PRODUCTION,
            'is_development': environment == Environment.DEVELOPMENT,
            'generate_password': self.secret_manager.generate_password,
            'generate_api_key': self.secret_manager.generate_api_key
        }
        
        try:
            template_obj = jinja_env.from_string(template.template_content)
            rendered = template_obj.render(**template_vars)
            return rendered
        
        except jinja2.exceptions.TemplateError as e:
            raise ValueError(f"Template rendering error: {e}")
    
    def _generate_password_filter(self, length: int = 32) -> str:
        """Jinja2 filter to generate passwords"""
        return self.secret_manager.generate_password(length)
    
    def _generate_secret_filter(self, secret_type: str = "password") -> str:
        """Jinja2 filter to generate secrets"""
        if secret_type == "password":
            return self.secret_manager.generate_password()
        elif secret_type == "api_key":
            return self.secret_manager.generate_api_key()
        else:
            return self.secret_manager.generate_password()
    
    def _encrypt_filter(self, value: str) -> str:
        """Jinja2 filter to encrypt values"""
        return self.secret_manager.encrypt_secret(value)

class ConfigDeployer:
    """Deploy configurations to target environments"""
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
    
    async def deploy_environment_config(self, env_config: EnvironmentConfig) -> bool:
        """Deploy configuration to environment"""
        logger.info(f"Deploying configuration for environment: {env_config.environment.value}")
        
        try:
            # Generate environment files
            await self._generate_env_file(env_config)
            await self._generate_compose_override(env_config)
            await self._generate_config_files(env_config)
            
            # Update secrets
            await self._deploy_secrets(env_config)
            
            # Validate deployment
            await self._validate_deployment(env_config)
            
            logger.info(f"Configuration deployed successfully: {env_config.environment.value}")
            return True
            
        except Exception as e:
            logger.error(f"Configuration deployment failed: {e}")
            return False
    
    async def _generate_env_file(self, env_config: EnvironmentConfig):
        """Generate .env file for environment"""
        env_file = self.project_root / f".env.{env_config.environment.value}"
        
        with open(env_file, 'w') as f:
            f.write(f"# Generated configuration for {env_config.environment.value}\n")
            f.write(f"# Generated at: {datetime.now().isoformat()}\n")
            f.write(f"# Version: {env_config.version}\n\n")
            
            # Write config values
            for key, config_value in env_config.config_values.items():
                if not config_value.is_secret:
                    value = config_value.value
                    if isinstance(value, (dict, list)):
                        value = json.dumps(value)
                    f.write(f"{key}={value}\n")
            
            # Add environment indicator
            f.write(f"\nSUTAZAI_ENV={env_config.environment.value}\n")
            f.write(f"DEPLOYMENT_ENV={env_config.environment.value}\n")
    
    async def _generate_compose_override(self, env_config: EnvironmentConfig):
        """Generate docker-compose override for environment"""
        override_file = self.project_root / f"docker-compose.{env_config.environment.value}.yml"
        
        # Environment-specific overrides
        override_config = {
            'version': '3.8',
            'services': {}
        }
        
        # Add environment-specific service configurations
        if env_config.environment == Environment.PRODUCTION:
            override_config['services'].update({
                'backend': {
                    'restart': 'always',
                    'deploy': {
                        'replicas': 3,
                        'resources': {
                            'limits': {'memory': '2g', 'cpus': '1.0'}
                        }
                    }
                },
                'frontend': {
                    'restart': 'always',
                    'deploy': {
                        'replicas': 2,
                        'resources': {
                            'limits': {'memory': '1g', 'cpus': '0.5'}
                        }
                    }
                }
            })
        elif env_config.environment == Environment.DEVELOPMENT:
            override_config['services'].update({
                'backend': {
                    'volumes': ['./backend:/app'],
                    'environment': ['DEBUG=true']
                }
            })
        
        # Apply custom overrides
        if env_config.overrides:
            for service, overrides in env_config.overrides.items():
                if service not in override_config['services']:
                    override_config['services'][service] = {}
                override_config['services'][service].update(overrides)
        
        with open(override_file, 'w') as f:
            yaml.dump(override_config, f, default_flow_style=False)
    
    async def _generate_config_files(self, env_config: EnvironmentConfig):
        """Generate environment-specific configuration files"""
        config_dir = self.project_root / "config" / env_config.environment.value
        config_dir.mkdir(exist_ok=True)
        
        # Generate configuration files based on templates
        # This would use the template engine to generate specific config files
        pass
    
    async def _deploy_secrets(self, env_config: EnvironmentConfig):
        """Deploy secrets to secure storage"""
        secrets_dir = self.project_root / "secrets" / env_config.environment.value
        secrets_dir.mkdir(exist_ok=True, mode=0o700)
        
        for key, secret in env_config.secrets.items():
            secret_file = secrets_dir / f"{key.lower()}.txt"
            
            # Decrypt and write secret
            decrypted_value = self.secret_manager.decrypt_secret(secret.encrypted_value)
            with open(secret_file, 'w') as f:
                f.write(decrypted_value)
            
            # Secure the file
            os.chmod(secret_file, 0o600)
    
    async def _validate_deployment(self, env_config: EnvironmentConfig):
        """Validate deployed configuration"""
        # Check that files were created
        env_file = self.project_root / f".env.{env_config.environment.value}"
        if not env_file.exists():
            raise Exception("Environment file was not created")
        
        # Validate file contents
        with open(env_file, 'r') as f:
            content = f.read()
            if f"SUTAZAI_ENV={env_config.environment.value}" not in content:
                raise Exception("Environment file does not contain correct environment setting")

class MultiEnvironmentConfigManager:
    """Main configuration management system"""
    
    def __init__(self):
        self.database = ConfigDatabase(CONFIG_DB)
        self.secret_manager = SecretManager()
        self.validator = ConfigValidator()
        self.template_engine = TemplateEngine(self.secret_manager)
        self.deployer = ConfigDeployer(PROJECT_ROOT)
        
        # Load default templates
        self._load_default_templates()
    
    def _load_default_templates(self):
        """Load default configuration templates"""
        templates = [
            ConfigTemplate(
                name="docker_compose_base",
                config_type=ConfigType.INFRASTRUCTURE,
                template_content=self._get_docker_compose_template(),
                variables=["environment", "replicas", "resources"],
                required_secrets=["database_password", "redis_password"],
                validation_schema={},
                description="Base Docker Compose configuration"
            ),
            ConfigTemplate(
                name="app_config",
                config_type=ConfigType.APPLICATION,
                template_content=self._get_app_config_template(),
                variables=["environment", "debug_enabled", "log_level"],
                required_secrets=["jwt_secret", "encryption_key"],
                validation_schema={},
                description="Application configuration"
            )
        ]
        
        for template in templates:
            self.database.store_template(template)
    
    def create_environment(self, environment: Environment, name: str) -> EnvironmentConfig:
        """Create new environment configuration"""
        logger.info(f"Creating environment configuration: {environment.value}")
        
        # Generate default configuration values
        config_values = self._generate_default_config_values(environment)
        
        # Generate required secrets
        secrets = self._generate_default_secrets(environment)
        
        # Create environment config
        env_config = EnvironmentConfig(
            environment=environment,
            name=name,
            config_values=config_values,
            secrets=secrets,
            overrides={},
            version="1.0.0",
            created_at=datetime.now(),
            updated_at=datetime.now(),
            checksum="",
            validation_errors=[]
        )
        
        # Calculate checksum
        env_config.checksum = self._calculate_checksum(env_config)
        
        # Validate configuration
        validation_errors = self.validator.validate_environment_config(env_config)
        env_config.validation_errors = validation_errors
        
        # Store configuration
        self.database.store_environment_config(env_config)
        
        logger.info(f"Environment created: {environment.value}")
        return env_config
    
    def update_environment_config(self, environment: Environment, updates: Dict[str, Any]) -> EnvironmentConfig:
        """Update environment configuration"""
        env_config = self.database.get_environment_config(environment)
        if not env_config:
            raise ValueError(f"Environment not found: {environment.value}")
        
        # Apply updates
        for key, value in updates.items():
            if key in env_config.config_values:
                env_config.config_values[key].value = value
            else:
                # Create new config value
                env_config.config_values[key] = ConfigValue(
                    key=key,
                    value=value,
                    value_type=type(value).__name__,
                    is_secret=False,
                    environment_specific=True
                )
        
        # Update metadata
        env_config.updated_at = datetime.now()
        env_config.version = self._increment_version(env_config.version)
        env_config.checksum = self._calculate_checksum(env_config)
        
        # Validate
        env_config.validation_errors = self.validator.validate_environment_config(env_config)
        
        # Store
        self.database.store_environment_config(env_config)
        
        return env_config
    
    async def deploy_environment(self, environment: Environment) -> bool:
        """Deploy environment configuration"""
        env_config = self.database.get_environment_config(environment)
        if not env_config:
            raise ValueError(f"Environment not found: {environment.value}")
        
        # Check for validation errors
        if env_config.validation_errors:
            logger.error(f"Cannot deploy environment with validation errors: {env_config.validation_errors}")
            return False
        
        # Deploy configuration
        return await self.deployer.deploy_environment_config(env_config)
    
    def rotate_secrets(self, environment: Environment, secret_keys: List[str] = None) -> bool:
        """Rotate secrets for environment"""
        env_config = self.database.get_environment_config(environment)
        if not env_config:
            raise ValueError(f"Environment not found: {environment.value}")
        
        secrets_to_rotate = secret_keys or list(env_config.secrets.keys())
        
        for secret_key in secrets_to_rotate:
            if secret_key in env_config.secrets:
                secret = env_config.secrets[secret_key]
                
                # Check if rotation is needed
                if datetime.now() - secret.last_rotated > secret.rotation_interval:
                    # Generate new secret
                    if secret.secret_type == SecretType.PASSWORD:
                        new_value = self.secret_manager.generate_password()
                    elif secret.secret_type == SecretType.API_KEY:
                        new_value = self.secret_manager.generate_api_key()
                    else:
                        new_value = self.secret_manager.generate_password()
                    
                    # Encrypt and store
                    secret.encrypted_value = self.secret_manager.encrypt_secret(new_value)
                    secret.last_rotated = datetime.now()
                    
                    logger.info(f"Rotated secret: {secret_key}")
        
        # Update configuration
        env_config.updated_at = datetime.now()
        env_config.checksum = self._calculate_checksum(env_config)
        self.database.store_environment_config(env_config)
        
        return True
    
    def compare_environments(self, env1: Environment, env2: Environment) -> Dict[str, Any]:
        """Compare configurations between environments"""
        config1 = self.database.get_environment_config(env1)
        config2 = self.database.get_environment_config(env2)
        
        if not config1 or not config2:
            raise ValueError("One or both environments not found")
        
        differences = {
            "config_differences": {},
            "secret_differences": {},
            "only_in_env1": [],
            "only_in_env2": []
        }
        
        # Compare config values
        all_keys = set(config1.config_values.keys()) | set(config2.config_values.keys())
        
        for key in all_keys:
            if key in config1.config_values and key in config2.config_values:
                value1 = config1.config_values[key].value
                value2 = config2.config_values[key].value
                if value1 != value2:
                    differences["config_differences"][key] = {
                        env1.value: value1,
                        env2.value: value2
                    }
            elif key in config1.config_values:
                differences["only_in_env1"].append(key)
            else:
                differences["only_in_env2"].append(key)
        
        # Compare secrets (keys only, not values)
        secret_keys1 = set(config1.secrets.keys())
        secret_keys2 = set(config2.secrets.keys())
        
        differences["secret_differences"] = {
            "only_in_env1": list(secret_keys1 - secret_keys2),
            "only_in_env2": list(secret_keys2 - secret_keys1),
            "common": list(secret_keys1 & secret_keys2)
        }
        
        return differences
    
    def _generate_default_config_values(self, environment: Environment) -> Dict[str, ConfigValue]:
        """Generate default configuration values for environment"""
        config_values = {}
        
        # Common configuration
        common_configs = {
            "SUTAZAI_ENV": environment.value,
            "LOG_LEVEL": "INFO" if environment == Environment.PRODUCTION else "DEBUG",
            "DEBUG": environment != Environment.PRODUCTION,
            "ENABLE_MONITORING": True,
            "MAX_WORKERS": 4 if environment == Environment.PRODUCTION else 2,
            "TIMEOUT": 30,
            "ENABLE_SSL": environment == Environment.PRODUCTION
        }
        
        for key, value in common_configs.items():
            config_values[key] = ConfigValue(
                key=key,
                value=value,
                value_type=type(value).__name__,
                is_secret=False,
                environment_specific=True,
                description=f"Default {key} setting",
                required=True
            )
        
        # Environment-specific configurations
        if environment == Environment.PRODUCTION:
            prod_configs = {
                "REPLICAS": 3,
                "MEMORY_LIMIT": "2g",
                "CPU_LIMIT": "1.0",
                "ENABLE_CACHING": True,
                "CACHE_TTL": 3600
            }
            
            for key, value in prod_configs.items():
                config_values[key] = ConfigValue(
                    key=key,
                    value=value,
                    value_type=type(value).__name__,
                    is_secret=False,
                    environment_specific=True,
                    description=f"Production {key} setting"
                )
        
        return config_values
    
    def _generate_default_secrets(self, environment: Environment) -> Dict[str, SecretValue]:
        """Generate default secrets for environment"""
        secrets = {}
        
        # Required secrets
        secret_configs = {
            "DATABASE_PASSWORD": (SecretType.PASSWORD, timedelta(days=90)),
            "REDIS_PASSWORD": (SecretType.PASSWORD, timedelta(days=90)),
            "JWT_SECRET": (SecretType.TOKEN, timedelta(days=365)),
            "ENCRYPTION_KEY": (SecretType.TOKEN, timedelta(days=365)),
            "API_KEY": (SecretType.API_KEY, timedelta(days=180))
        }
        
        for key, (secret_type, rotation_interval) in secret_configs.items():
            if secret_type == SecretType.PASSWORD:
                secret_value = self.secret_manager.generate_password()
            elif secret_type == SecretType.API_KEY:
                secret_value = self.secret_manager.generate_api_key()
            else:
                secret_value = self.secret_manager.generate_password(64)
            
            encrypted_value = self.secret_manager.encrypt_secret(secret_value)
            
            secrets[key] = SecretValue(
                key=key,
                encrypted_value=encrypted_value,
                secret_type=secret_type,
                rotation_interval=rotation_interval,
                last_rotated=datetime.now(),
                metadata={"environment": environment.value}
            )
        
        return secrets
    
    def _calculate_checksum(self, env_config: EnvironmentConfig) -> str:
        """Calculate configuration checksum"""
        # Create a deterministic representation
        config_data = {
            "environment": env_config.environment.value,
            "config_values": {k: v.value for k, v in env_config.config_values.items()},
            "secret_keys": list(env_config.secrets.keys()),
            "overrides": env_config.overrides
        }
        
        config_str = json.dumps(config_data, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _increment_version(self, current_version: str) -> str:
        """Increment version number"""
        try:
            parts = current_version.split('.')
            patch = int(parts[2]) + 1
            return f"{parts[0]}.{parts[1]}.{patch}"
        except:
            return "1.0.1"
    
    def _get_docker_compose_template(self) -> str:
        """Get Docker Compose template"""
        return """
version: '3.8'

services:
  backend:
    {% if environment == 'production' %}
    deploy:
      replicas: {{ replicas | default(3) }}
      resources:
        limits:
          memory: {{ resources.memory | default('2g') }}
          cpus: {{ resources.cpus | default('1.0') }}
    restart: always
    {% else %}
    volumes:
      - ./backend:/app
    {% endif %}
    environment:
      - SUTAZAI_ENV={{ environment }}
      - DEBUG={{ 'false' if environment == 'production' else 'true' }}
"""
    
    def _get_app_config_template(self) -> str:
        """Get application configuration template"""
        return """
# Application Configuration for {{ environment }}
# Generated at: {{ ansible_date_time.iso8601 }}

[app]
environment = {{ environment }}
debug = {{ debug_enabled | default(false) }}
log_level = {{ log_level | default('INFO') }}

[security]
jwt_secret = {{ jwt_secret }}
encryption_key = {{ encryption_key }}

[database]
host = postgres
port = 5432
name = sutazai
user = sutazai
password = {{ database_password }}

[redis]
host = redis
port = 6379
password = {{ redis_password }}
"""

async def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Multi-Environment Configuration Manager for SutazAI"
    )
    parser.add_argument(
        "command",
        choices=["create", "update", "deploy", "compare", "rotate", "list", "info"],
        help="Command to execute"
    )
    parser.add_argument(
        "--environment", "-e",
        choices=["development", "staging", "production", "test", "local"],
        help="Target environment"
    )
    parser.add_argument(
        "--name", "-n",
        help="Environment name"
    )
    parser.add_argument(
        "--config",
        help="Configuration updates (JSON format)"
    )
    parser.add_argument(
        "--compare-with",
        help="Environment to compare with"
    )
    parser.add_argument(
        "--secrets",
        nargs="*",
        help="Specific secrets to rotate"
    )
    
    args = parser.parse_args()
    
    config_manager = MultiEnvironmentConfigManager()
    
    try:
        if args.command == "create":
            if not args.environment or not args.name:
                logger.error("Environment and name are required for create")
                sys.exit(1)
            
            environment = Environment(args.environment)
            env_config = config_manager.create_environment(environment, args.name)
            
            print(f"Created environment: {env_config.environment.value}")
            print(f"Version: {env_config.version}")
            print(f"Config values: {len(env_config.config_values)}")
            print(f"Secrets: {len(env_config.secrets)}")
            
            if env_config.validation_errors:
                print("Validation errors:")
                for error in env_config.validation_errors:
                    print(f"  - {error}")
        
        elif args.command == "update":
            if not args.environment or not args.config:
                logger.error("Environment and config are required for update")
                sys.exit(1)
            
            environment = Environment(args.environment)
            updates = json.loads(args.config)
            
            env_config = config_manager.update_environment_config(environment, updates)
            
            print(f"Updated environment: {env_config.environment.value}")
            print(f"New version: {env_config.version}")
        
        elif args.command == "deploy":
            if not args.environment:
                logger.error("Environment is required for deploy")
                sys.exit(1)
            
            environment = Environment(args.environment)
            success = config_manager.deploy_environment(environment)
            
            if success:
                print(f"Successfully deployed environment: {environment.value}")
            else:
                print(f"Failed to deploy environment: {environment.value}")
                sys.exit(1)
        
        elif args.command == "compare":
            if not args.environment or not args.compare_with:
                logger.error("Two environments are required for compare")
                sys.exit(1)
            
            env1 = Environment(args.environment)
            env2 = Environment(args.compare_with)
            
            differences = config_manager.compare_environments(env1, env2)
            
            print(f"Comparing {env1.value} vs {env2.value}:")
            print(json.dumps(differences, indent=2))
        
        elif args.command == "rotate":
            if not args.environment:
                logger.error("Environment is required for rotate")
                sys.exit(1)
            
            environment = Environment(args.environment)
            success = config_manager.rotate_secrets(environment, args.secrets)
            
            if success:
                print(f"Successfully rotated secrets for: {environment.value}")
            else:
                print(f"Failed to rotate secrets for: {environment.value}")
        
        elif args.command == "list":
            environments = config_manager.database.list_environments()
            print("Configured environments:")
            for env in environments:
                print(f"  - {env}")
        
        elif args.command == "info":
            if not args.environment:
                logger.error("Environment is required for info")
                sys.exit(1)
            
            environment = Environment(args.environment)
            env_config = config_manager.database.get_environment_config(environment)
            
            if not env_config:
                print(f"Environment not found: {environment.value}")
                sys.exit(1)
            
            print(f"Environment: {env_config.environment.value}")
            print(f"Name: {env_config.name}")
            print(f"Version: {env_config.version}")
            print(f"Created: {env_config.created_at}")
            print(f"Updated: {env_config.updated_at}")
            print(f"Config values: {len(env_config.config_values)}")
            print(f"Secrets: {len(env_config.secrets)}")
            print(f"Validation status: {'✓ Valid' if not env_config.validation_errors else '✗ Has errors'}")
            
            if env_config.validation_errors:
                print("Validation errors:")
                for error in env_config.validation_errors:
                    print(f"  - {error}")
    
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())