"""
Logger utility for SutazAI AGI/ASI System
Provides structured logging with file and console output
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_object = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
            'message': record.getMessage(),
        }
        
        # Add exception info if present
        if record.exc_info:
            log_object['exception'] = self.formatException(record.exc_info)
            
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'created', 'filename', 
                          'funcName', 'levelname', 'levelno', 'lineno', 
                          'module', 'msecs', 'pathname', 'process', 
                          'processName', 'relativeCreated', 'thread', 
                          'threadName', 'exc_info', 'exc_text', 'stack_info']:
                log_object[key] = value
                
        return json.dumps(log_object)

def setup_logger(
    name: str = "sutazai",
    log_file: Optional[str] = None,
    level: str = "INFO",
    max_bytes: int = 10 * 1024 * 1024,  # 10MB
    backup_count: int = 5,
    json_format: bool = False
) -> logging.Logger:
    """
    Set up a logger with file and console handlers
    
    Args:
        name: Logger name
        log_file: Path to log file (creates logs/ directory if needed)
        level: Logging level
        max_bytes: Maximum size of log file before rotation
        backup_count: Number of backup files to keep
        json_format: Use JSON formatting for logs
        
    Returns:
        Configured logger instance
    """
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Create formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG if level == "DEBUG" else logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        # Create logs directory if it doesn't exist
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create rotating file handler
        file_handler = RotatingFileHandler(
            log_file,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

# Create module-specific loggers
def get_logger(module_name: str) -> logging.Logger:
    """Get a logger for a specific module"""
    return logging.getLogger(f"sutazai.{module_name}")

# Utility functions
def log_function_call(logger: logging.Logger):
    """Decorator to log function calls"""
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

def log_async_function_call(logger: logging.Logger):
    """Decorator to log async function calls"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = await func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {str(e)}")
                raise
        return wrapper
    return decorator

# Performance logging
class PerformanceLogger:
    """Context manager for logging performance metrics"""
    
    def __init__(self, logger: logging.Logger, operation: str):
        self.logger = logger
        self.operation = operation
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        self.logger.debug(f"Starting {self.operation}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        duration = (datetime.now() - self.start_time).total_seconds()
        if exc_type:
            self.logger.error(
                f"{self.operation} failed after {duration:.3f}s: {exc_val}",
                extra={'duration': duration, 'error': str(exc_val)}
            )
        else:
            self.logger.info(
                f"{self.operation} completed in {duration:.3f}s",
                extra={'duration': duration}
            )

# Default logger instance
default_logger = setup_logger()

__all__ = [
    'setup_logger',
    'get_logger',
    'log_function_call',
    'log_async_function_call',
    'PerformanceLogger',
    'JSONFormatter',
    'default_logger'
] 