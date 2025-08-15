#!/usr/bin/env python3
"""
MCP Update Manager Configuration

Centralized configuration management for the MCP automation system.
Provides environment-specific settings, security configurations, and
operational parameters following Enforcement Rules compliance.

Author: Claude AI Assistant (python-architect.md)
Created: 2025-08-15 11:20:00 UTC
Version: 1.0.0
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from enum import Enum


class UpdateMode(Enum):
    """Update modes for MCP server management."""
    STAGING_ONLY = "staging_only"
    PRODUCTION = "production"
    ROLLBACK = "rollback"
    HEALTH_CHECK = "health_check"


class LogLevel(Enum):
    """Logging levels for the automation system."""
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


@dataclass
class SecurityConfig:
    """Security configuration for MCP operations."""
    verify_checksums: bool = True
    require_signature_validation: bool = False  # Future enhancement
    staging_isolation: bool = True
    max_download_size_mb: int = 500
    allowed_registries: List[str] = None
    quarantine_duration_minutes: int = 30
    
    def __post_init__(self):
        if self.allowed_registries is None:
            self.allowed_registries = ["https://registry.npmjs.org"]


@dataclass
class PerformanceConfig:
    """Performance and resource configuration."""
    max_concurrent_downloads: int = 3
    download_timeout_seconds: int = 300
    health_check_timeout_seconds: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: int = 5
    staging_timeout_minutes: int = 10
    cleanup_retention_days: int = 7


@dataclass
class PathConfig:
    """File system path configuration."""
    mcp_root: Path
    automation_root: Path
    staging_root: Path
    backup_root: Path
    logs_root: Path
    wrappers_root: Path
    
    def __post_init__(self):
        # Ensure all paths are Path objects
        for field_name, field_value in self.__dict__.items():
            if isinstance(field_value, str):
                setattr(self, field_name, Path(field_value))
        
        # Create directories if they don't exist
        for path in [self.staging_root, self.backup_root, self.logs_root]:
            path.mkdir(parents=True, exist_ok=True)


@dataclass
class MonitoringConfig:
    """Monitoring and alerting configuration."""
    enable_metrics: bool = True
    metrics_port: int = 9090
    enable_health_endpoints: bool = True
    health_check_interval_minutes: int = 5
    alert_on_failures: bool = True
    prometheus_namespace: str = "mcp_automation"


class MCPAutomationConfig:
    """
    Main configuration class for MCP automation system.
    
    Provides centralized configuration management with environment-specific
    overrides, validation, and secure defaults following organizational standards.
    """
    
    def __init__(self, config_file: Optional[Path] = None):
        """
        Initialize configuration with optional config file override.
        
        Args:
            config_file: Optional path to configuration file
        """
        self.logger = self._setup_logging()
        
        # Load base configuration
        self._load_base_config()
        
        # Load environment overrides
        self._load_environment_config()
        
        # Load file overrides if provided
        if config_file and config_file.exists():
            self._load_file_config(config_file)
        
        # Validate configuration
        self._validate_config()
        
        self.logger.info("MCP automation configuration initialized successfully")
    
    def _setup_logging(self) -> logging.Logger:
        """Setup basic logging for configuration initialization."""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def _load_base_config(self):
        """Load base configuration with secure defaults."""
        # Base paths
        sutazai_root = Path(os.getenv("SUTAZAI_ROOT", "/opt/sutazaiapp"))
        mcp_root = sutazai_root / "scripts" / "mcp"
        
        self.paths = PathConfig(
            mcp_root=mcp_root,
            automation_root=mcp_root / "automation",
            staging_root=mcp_root / "automation" / "staging",
            backup_root=mcp_root / "automation" / "backups",
            logs_root=sutazai_root / "logs" / "mcp_automation",
            wrappers_root=mcp_root / "wrappers"
        )
        
        self.security = SecurityConfig()
        self.performance = PerformanceConfig()
        self.monitoring = MonitoringConfig()
        
        # Operational settings
        self.update_mode = UpdateMode.STAGING_ONLY
        self.log_level = LogLevel.INFO
        self.dry_run = False
        self.enable_auto_rollback = True
        self.notification_webhooks: List[str] = []
        
        # MCP Server Registry
        self.mcp_servers = {
            "files": {"package": "@modelcontextprotocol/server-filesystem", "wrapper": "files.sh"},
            "context7": {"package": "@modelcontextprotocol/server-context", "wrapper": "context7.sh"},
            "http_fetch": {"package": "@modelcontextprotocol/server-fetch", "wrapper": "http_fetch.sh"},
            "ddg": {"package": "@modelcontextprotocol/server-search", "wrapper": "ddg.sh"},
            "sequentialthinking": {"package": "@modelcontextprotocol/server-sequential", "wrapper": "sequentialthinking.sh"},
            "nx-mcp": {"package": "@modelcontextprotocol/server-nx", "wrapper": "nx-mcp.sh"},
            "extended-memory": {"package": "@modelcontextprotocol/server-memory", "wrapper": "extended-memory.sh"},
            "mcp_ssh": {"package": "@modelcontextprotocol/server-ssh", "wrapper": "mcp_ssh.sh"},
            "ultimatecoder": {"package": "@modelcontextprotocol/server-coder", "wrapper": "ultimatecoder.sh"},
            "postgres": {"package": "@modelcontextprotocol/server-postgres", "wrapper": "postgres.sh"},
            "playwright-mcp": {"package": "@modelcontextprotocol/server-playwright", "wrapper": "playwright-mcp.sh"},
            "memory-bank-mcp": {"package": "@modelcontextprotocol/server-memorybank", "wrapper": "memory-bank-mcp.sh"},
            "puppeteer-mcp": {"package": "@modelcontextprotocol/server-puppeteer", "wrapper": "puppeteer-mcp.sh"},
            "knowledge-graph-mcp": {"package": "@modelcontextprotocol/server-knowledge", "wrapper": "knowledge-graph-mcp.sh"},
            "compass-mcp": {"package": "@modelcontextprotocol/server-compass", "wrapper": "compass-mcp.sh"},
        }
    
    def _load_environment_config(self):
        """Load configuration from environment variables."""
        # Security overrides
        if os.getenv("MCP_DISABLE_CHECKSUM_VERIFICATION") == "true":
            self.security.verify_checksums = False
            self.logger.warning("Checksum verification disabled via environment variable")
        
        # Performance overrides
        if max_downloads := os.getenv("MCP_MAX_CONCURRENT_DOWNLOADS"):
            self.performance.max_concurrent_downloads = int(max_downloads)
        
        if download_timeout := os.getenv("MCP_DOWNLOAD_TIMEOUT"):
            self.performance.download_timeout_seconds = int(download_timeout)
        
        # Operational overrides
        if update_mode := os.getenv("MCP_UPDATE_MODE"):
            try:
                self.update_mode = UpdateMode(update_mode)
            except ValueError:
                self.logger.warning(f"Invalid update mode: {update_mode}, using default")
        
        if log_level := os.getenv("MCP_LOG_LEVEL"):
            try:
                self.log_level = LogLevel(log_level.upper())
            except ValueError:
                self.logger.warning(f"Invalid log level: {log_level}, using default")
        
        if os.getenv("MCP_DRY_RUN") == "true":
            self.dry_run = True
            self.logger.info("Dry run mode enabled via environment variable")
        
        # Webhook notifications
        if webhook_urls := os.getenv("MCP_NOTIFICATION_WEBHOOKS"):
            self.notification_webhooks = webhook_urls.split(",")
    
    def _load_file_config(self, config_file: Path):
        """Load configuration from JSON file."""
        try:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
            
            # Merge file configuration
            for section, values in file_config.items():
                if hasattr(self, section) and isinstance(getattr(self, section), dict):
                    getattr(self, section).update(values)
                else:
                    setattr(self, section, values)
            
            self.logger.info(f"Loaded configuration from {config_file}")
        
        except Exception as e:
            self.logger.error(f"Failed to load config file {config_file}: {e}")
            raise
    
    def _validate_config(self):
        """Validate configuration for consistency and security."""
        # Validate paths exist and are accessible
        if not self.paths.mcp_root.exists():
            raise ValueError(f"MCP root directory does not exist: {self.paths.mcp_root}")
        
        if not self.paths.wrappers_root.exists():
            raise ValueError(f"MCP wrappers directory does not exist: {self.paths.wrappers_root}")
        
        # Validate performance settings
        if self.performance.max_concurrent_downloads < 1:
            raise ValueError("max_concurrent_downloads must be at least 1")
        
        if self.performance.download_timeout_seconds < 30:
            raise ValueError("download_timeout_seconds must be at least 30")
        
        # Validate security settings
        if self.security.max_download_size_mb < 1:
            raise ValueError("max_download_size_mb must be at least 1")
        
        if not self.security.allowed_registries:
            raise ValueError("allowed_registries cannot be empty")
        
        # Validate MCP server configurations
        for server_name, config in self.mcp_servers.items():
            wrapper_path = self.paths.wrappers_root / config["wrapper"]
            if not wrapper_path.exists():
                self.logger.warning(f"Wrapper script not found for {server_name}: {wrapper_path}")
    
    def get_server_config(self, server_name: str) -> Optional[Dict[str, Any]]:
        """Get configuration for a specific MCP server."""
        return self.mcp_servers.get(server_name)
    
    def get_all_servers(self) -> List[str]:
        """Get list of all configured MCP servers."""
        return list(self.mcp_servers.keys())
    
    def get_staging_path(self, server_name: str) -> Path:
        """Get staging path for a specific server."""
        return self.paths.staging_root / server_name
    
    def get_backup_path(self, server_name: str) -> Path:
        """Get backup path for a specific server."""
        return self.paths.backup_root / server_name
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for serialization."""
        config_dict = {}
        
        # Convert dataclass objects
        for attr_name in ['security', 'performance', 'monitoring']:
            if hasattr(self, attr_name):
                config_dict[attr_name] = asdict(getattr(self, attr_name))
        
        # Convert paths
        if hasattr(self, 'paths'):
            config_dict['paths'] = {
                name: str(path) for name, path in asdict(self.paths).items()
            }
        
        # Add other attributes
        for attr_name in ['update_mode', 'log_level', 'dry_run', 'enable_auto_rollback']:
            if hasattr(self, attr_name):
                attr_value = getattr(self, attr_name)
                if isinstance(attr_value, Enum):
                    config_dict[attr_name] = attr_value.value
                else:
                    config_dict[attr_name] = attr_value
        
        config_dict['mcp_servers'] = self.mcp_servers
        config_dict['notification_webhooks'] = self.notification_webhooks
        
        return config_dict
    
    def save_config(self, output_file: Path):
        """Save current configuration to file."""
        try:
            config_dict = self.to_dict()
            with open(output_file, 'w') as f:
                json.dump(config_dict, f, indent=2)
            
            self.logger.info(f"Configuration saved to {output_file}")
        
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise


# Global configuration instance
_config_instance: Optional[MCPAutomationConfig] = None


def get_config(config_file: Optional[Path] = None) -> MCPAutomationConfig:
    """
    Get global configuration instance (singleton pattern).
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        MCPAutomationConfig instance
    """
    global _config_instance
    
    if _config_instance is None:
        _config_instance = MCPAutomationConfig(config_file)
    
    return _config_instance


def reload_config(config_file: Optional[Path] = None) -> MCPAutomationConfig:
    """
    Reload global configuration instance.
    
    Args:
        config_file: Optional configuration file path
        
    Returns:
        New MCPAutomationConfig instance
    """
    global _config_instance
    _config_instance = MCPAutomationConfig(config_file)
    return _config_instance


if __name__ == "__main__":
    # Configuration validation and testing
    config = get_config()
    print("MCP Automation Configuration:")
    print(f"  MCP Root: {config.paths.mcp_root}")
    print(f"  Update Mode: {config.update_mode.value}")
    print(f"  Dry Run: {config.dry_run}")
    print(f"  Servers: {len(config.mcp_servers)}")
    
    # Test configuration serialization
    config_dict = config.to_dict()
    print(f"  Configuration sections: {list(config_dict.keys())}")