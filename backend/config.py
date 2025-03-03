#!/usr/bin/env python3.11
"""Configuration Module

This module provides configuration management for the SutazAI backend.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from loguru import logger


@dataclass
class ServerConfig:
    """Server configuration settings."""
    workers: int = 4
    keepalive: int = 60
    timeout: int = 300
    limit_max_requests: int = 0
    max_requests_jitter: int = 0
    limit_concurrency: int = 1000
    backlog: int = 2048


@dataclass
class Config:
    """Main configuration class for the SutazAI backend."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize configuration.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.config_path = config_path or os.getenv(
            "SUTAZAI_CONFIG",
            "/opt/sutazaiapp/config/config.yml",
        )
        
        # Load configuration
        self._config = self._load_config()
        
        # Server settings
        self.host = self._config.get("host", "0.0.0.0")
        self.port = int(self._config.get("port", 8000))
        self.debug = bool(self._config.get("debug", False))
        
        # Initialize server config
        self.server = ServerConfig(
            workers=int(self._config.get("server", {}).get("workers", 4)),
            keepalive=int(self._config.get("server", {}).get("keepalive", 60)),
            timeout=int(self._config.get("server", {}).get("timeout", 300)),
            limit_max_requests=int(
                self._config.get("server", {}).get("limit_max_requests", 0)
            ),
            max_requests_jitter=int(
                self._config.get("server", {}).get("max_requests_jitter", 0)
            ),
            limit_concurrency=int(
                self._config.get("server", {}).get("limit_concurrency", 1000)
            ),
            backlog=int(self._config.get("server", {}).get("backlog", 2048)),
        )
        
        # Database settings
        self.db_url = self._config.get(
            "database_url",
            "postgresql://postgres:postgres@localhost:5432/sutazai",
        )
        
        # Redis settings
        self.redis_url = self._config.get(
            "redis_url",
            "redis://localhost:6379/0",
        )
        
        # Security settings
        self.secret_key = self._config.get(
            "secret_key",
            os.getenv("SUTAZAI_SECRET_KEY", "your-secret-key"),
        )
        self.token_expire_minutes = int(
            self._config.get("token_expire_minutes", 60 * 24)  # 24 hours
        )
        
        # Logging settings
        self.log_level = self._config.get("log_level", "INFO")
        self.log_file = self._config.get(
            "log_file",
            "/opt/sutazaiapp/logs/backend.log",
        )
        
        # Configure logging
        self._setup_logging()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file.
        
        Returns:
            Dict containing configuration values
        """
        try:
            if self.config_path is None:
                logger.warning("Config path is None, using default empty config")
                return {}
                
            # Ensure config_path is str before creating Path
            config_path_str = str(self.config_path) if self.config_path is not None else ""
            config_path = Path(config_path_str)
            if not config_path.exists():
                logger.warning(f"Config file not found: {config_path}")
                return {}
                
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
                
            return config or {}
            
        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return {}
            
    def _setup_logging(self) -> None:
        """Configure logging settings."""
        # Ensure log_file is not None before creating Path
        log_file = self.log_file
        if log_file is None:
            log_file = "/opt/sutazaiapp/logs/backend.log"
            
        log_dir = Path(log_file).parent
        log_dir.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=self.log_level,
            rotation="10 MB",
            compression="zip",
            retention="1 week",
        )
        
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        return self._config.get(key, default)
        
    def __str__(self) -> str:
        """String representation of configuration."""
        return f"Config(debug={self.debug}, host={self.host}, port={self.port})"


def load_config(config_path: Optional[str] = None) -> Config:
    """Load configuration from file.
    
    Args:
        config_path: Optional path to configuration file
        
    Returns:
        Config instance
    """
    return Config(config_path)
