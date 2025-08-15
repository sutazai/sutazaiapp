#!/usr/bin/env python3
"""
MCP Automation System

Automated MCP server download, update, and version management system with
comprehensive safety mechanisms, monitoring, and zero-disruption deployment.

This package provides:
- Configuration management with environment-specific overrides
- Version tracking and rollback capabilities  
- Safe download handling with integrity validation
- Main orchestration service with async coordination
- Comprehensive error handling and audit trails

Author: Claude AI Assistant (python-architect.md)
Created: 2025-08-15 11:42:00 UTC
Version: 1.0.0
"""

import logging
import sys
from pathlib import Path

# Package metadata
__version__ = "1.0.0"
__author__ = "Claude AI Assistant (python-architect.md)"
__description__ = "Automated MCP server management with safety and monitoring"
__license__ = "MIT"

# Configure package-level logging
logger = logging.getLogger(__name__)

# Ensure Python version compatibility
MIN_PYTHON = (3, 8)
if sys.version_info < MIN_PYTHON:
    sys.exit(f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required")

# Package imports with error handling
try:
    from .config import (
        MCPAutomationConfig,
        get_config,
        reload_config,
        UpdateMode,
        LogLevel,
        SecurityConfig,
        PerformanceConfig,
        PathConfig,
        MonitoringConfig
    )
    
    from .version_manager import (
        VersionManager,
        VersionInfo,
        OperationRecord,
        VersionState,
        OperationType
    )
    
    from .download_manager import (
        DownloadManager,
        DownloadProgress,
        DownloadResult,
        DownloadState,
        ValidationResult,
        SecurityError,
        ValidationError
    )
    
    from .mcp_update_manager import (
        MCPUpdateManager,
        UpdateJob,
        UpdateSummary,
        UpdateStatus,
        UpdatePriority
    )
    
    logger.debug("MCP Automation System imported successfully")

except ImportError as e:
    logger.error(f"Failed to import MCP automation components: {e}")
    # Don't fail completely, but log the error
    pass

# Public API
__all__ = [
    # Configuration
    'MCPAutomationConfig',
    'get_config',
    'reload_config',
    'UpdateMode',
    'LogLevel',
    'SecurityConfig',
    'PerformanceConfig',
    'PathConfig',
    'MonitoringConfig',
    
    # Version Management
    'VersionManager',
    'VersionInfo',
    'OperationRecord',
    'VersionState',
    'OperationType',
    
    # Download Management
    'DownloadManager',
    'DownloadProgress',
    'DownloadResult',
    'DownloadState',
    'ValidationResult',
    'SecurityError',
    'ValidationError',
    
    # Update Management
    'MCPUpdateManager',
    'UpdateJob',
    'UpdateSummary',
    'UpdateStatus',
    'UpdatePriority',
    
    # Metadata
    '__version__',
    '__author__',
    '__description__',
    '__license__'
]

def get_package_info():
    """Get package information for debugging and monitoring."""
    return {
        'name': __name__,
        'version': __version__,
        'author': __author__,
        'description': __description__,
        'license': __license__,
        'python_version': f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        'path': str(Path(__file__).parent),
        'components': {
            'config': 'MCPAutomationConfig' in globals(),
            'version_manager': 'VersionManager' in globals(),
            'download_manager': 'DownloadManager' in globals(),
            'update_manager': 'MCPUpdateManager' in globals()
        }
    }

def setup_logging(level=logging.INFO, log_file=None):
    """
    Setup package-wide logging configuration.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional log file path
    """
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Setup console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Setup file handler if specified
    handlers = [console_handler]
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Configure package logger
    package_logger = logging.getLogger(__name__)
    package_logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in package_logger.handlers[:]:
        package_logger.removeHandler(handler)
    
    # Add new handlers
    for handler in handlers:
        package_logger.addHandler(handler)
    
    package_logger.info(f"MCP Automation System logging configured (level: {logging.getLevelName(level)})")

# Auto-configure basic logging if not already configured
if not logger.handlers:
    setup_logging()

logger.info(f"MCP Automation System v{__version__} initialized")