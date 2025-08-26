"""
Production-Grade Logging Configuration for SutazAI System
Centralized logging setup following enterprise standards and Rule 8 compliance
Replaces all print() statements with structured logging
"""

import logging
import logging.config
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import json


class ProductionLoggingFormatter(logging.Formatter):
    """
    Custom formatter for production logging with structured output
    Includes timestamp, level, module, function, line number, and message
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record with production-grade structure"""
        
        # Add extra context information
        record.timestamp = datetime.utcnow().isoformat() + 'Z'
        record.module_name = record.module if hasattr(record, 'module') else record.name
        record.function_name = record.funcName
        record.line_number = record.lineno
        
        # Create structured log entry
        log_entry = {
            'timestamp': record.timestamp,
            'level': record.levelname,
            'logger': record.name,
            'module': record.module_name,
            'function': record.function_name,
            'line': record.line_number,
            'message': record.getMessage(),
            'process_id': os.getpid(),
            'thread_id': record.thread
        }
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # For console output, use readable format
        if hasattr(record, 'console_output') and record.console_output:
            return f"[{record.timestamp}] {record.levelname:8} | {record.name:20} | {record.getMessage()}"
        
        # For file output, use JSON format
        return json.dumps(log_entry, separators=(',', ':'))


class SutazAILogger:
    """
    Centralized logger configuration for SutazAI system
    Provides production-grade logging with multiple handlers and formatters
    """
    
    _instance: Optional['SutazAILogger'] = None
    _configured: bool = False
    
    def __new__(cls) -> 'SutazAILogger':
        """Singleton pattern for logging configuration"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize logging configuration if not already done"""
        if not self._configured:
            self.setup_logging()
            self._configured = True
    
    def setup_logging(self) -> None:
        """
        Configure production-grade logging system
        Sets up multiple handlers: console, file, error file
        """
        
        # Determine environment and log level
        environment = os.getenv('SUTAZAI_ENV', 'development')
        log_level = os.getenv('SUTAZAI_LOG_LEVEL', 'INFO' if environment == 'production' else 'DEBUG')
        
        # Create logs directory if it doesn't exist
        log_dir = Path('/opt/sutazaiapp/logs')
        log_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(getattr(logging, log_level.upper()))
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler with readable format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ProductionLoggingFormatter()
        
        # Create simple console formatter for readability
        console_format = logging.Formatter(
            '[%(asctime)s] %(levelname)-8s | %(name)-20s | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_format)
        root_logger.addHandler(console_handler)
        
        # File handler for all logs
        file_handler = logging.FileHandler(
            log_dir / f'sutazai-{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8'
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(ProductionLoggingFormatter())
        root_logger.addHandler(file_handler)
        
        # Error file handler for warnings and errors only
        error_handler = logging.FileHandler(
            log_dir / f'sutazai-errors-{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8'
        )
        error_handler.setLevel(logging.WARNING)
        error_handler.setFormatter(ProductionLoggingFormatter())
        root_logger.addHandler(error_handler)
        
        # Performance handler for performance-related logs
        perf_handler = logging.FileHandler(
            log_dir / f'sutazai-performance-{datetime.now().strftime("%Y%m%d")}.log',
            encoding='utf-8'
        )
        perf_handler.setLevel(logging.INFO)
        perf_handler.addFilter(lambda record: 'performance' in record.name.lower() or 'perf' in record.getMessage().lower())
        perf_handler.setFormatter(ProductionLoggingFormatter())
        root_logger.addHandler(perf_handler)
        
        # Configure specific loggers
        self._configure_module_loggers()
        
        # Log successful configuration
        logger = logging.getLogger(__name__)
        logger.info(f"Production logging configured successfully for environment: {environment}")
        logger.info(f"Log level set to: {log_level}")
        logger.info(f"Log directory: {log_dir}")
    
    def _configure_module_loggers(self) -> None:
        """Configure specific loggers for different modules"""
        
        # MCP-specific logger
        mcp_logger = logging.getLogger('mcp')
        mcp_logger.setLevel(logging.INFO)
        
        # Agent-specific logger
        agent_logger = logging.getLogger('agent')
        agent_logger.setLevel(logging.INFO)
        
        # Database-specific logger
        db_logger = logging.getLogger('database')
        db_logger.setLevel(logging.INFO)
        
        # Performance-specific logger
        perf_logger = logging.getLogger('performance')
        perf_logger.setLevel(logging.INFO)
        
        # Test-specific logger
        test_logger = logging.getLogger('test')
        test_logger.setLevel(logging.DEBUG)
    
    @staticmethod
    def get_logger(name: str) -> logging.Logger:
        """
        Get a configured logger for the specified module
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Configured logger instance
        """
        return logging.getLogger(name)
    
    @staticmethod
    def log_performance_metric(
        operation: str,
        duration_ms: float,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Log performance metrics in structured format
        
        Args:
            operation: Name of the operation being measured
            duration_ms: Duration in milliseconds
            details: Additional details about the operation
        """
        logger = logging.getLogger('performance')
        
        metric_data = {
            'operation': operation,
            'duration_ms': duration_ms,
            'timestamp': datetime.utcnow().isoformat() + 'Z'
        }
        
        if details:
            metric_data.update(details)
        
        logger.info(f"PERFORMANCE_METRIC: {json.dumps(metric_data)}")
    
    @staticmethod
    def log_security_event(
        event_type: str,
        details: Dict[str, Any],
        severity: str = 'INFO'
    ) -> None:
        """
        Log security-related events
        
        Args:
            event_type: Type of security event
            details: Event details
            severity: Log severity level
        """
        logger = logging.getLogger('security')
        
        security_data = {
            'event_type': event_type,
            'timestamp': datetime.utcnow().isoformat() + 'Z',
            'details': details
        }
        
        level = getattr(logging, severity.upper(), logging.INFO)
        logger.log(level, f"SECURITY_EVENT: {json.dumps(security_data)}")


def initialize_logging() -> None:
    """Initialize the SutazAI logging system"""
    SutazAILogger()


def get_logger(name: str) -> logging.Logger:
    """
    Convenience function to get a properly configured logger
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    # Ensure logging is initialized
    if not SutazAILogger._configured:
        initialize_logging()
    
    return SutazAILogger.get_logger(name)


# Convenience functions to replace common print patterns
def log_info(message: str, logger_name: Optional[str] = None) -> None:
    """Replace print() with info-level logging"""
    logger = get_logger(logger_name or 'sutazai')
    logger.info(message)


def log_debug(message: str, logger_name: Optional[str] = None) -> None:
    """Replace print() with debug-level logging"""
    logger = get_logger(logger_name or 'sutazai')
    logger.debug(message)


def log_warning(message: str, logger_name: Optional[str] = None) -> None:
    """Replace print() with warning-level logging"""
    logger = get_logger(logger_name or 'sutazai')
    logger.warning(message)


def log_error(message: str, exception: Optional[Exception] = None, logger_name: Optional[str] = None) -> None:
    """Replace print() with error-level logging"""
    logger = get_logger(logger_name or 'sutazai')
    if exception:
        logger.error(message, exc_info=exception)
    else:
        logger.error(message)


# Initialize logging when module is imported
initialize_logging()