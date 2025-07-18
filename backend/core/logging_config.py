#!/usr/bin/env python3
"""
SutazAI Logging Configuration
Centralized logging setup
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime
import json

from .config import settings


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging"""
    
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname", 
                          "filename", "module", "lineno", "funcName", "created", "msecs",
                          "relativeCreated", "thread", "threadName", "processName", 
                          "process", "exc_info", "exc_text", "stack_info", "getMessage"]:
                log_entry[key] = value
        
        return json.dumps(log_entry)


def setup_logging():
    """Setup logging configuration"""
    
    # Create logs directory
    log_dir = Path(settings.LOGS_PATH)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # File handler
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir / "sutazai.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    # Error file handler
    error_handler = logging.handlers.RotatingFileHandler(
        log_dir / "sutazai_errors.log",
        maxBytes=10*1024*1024,  # 10MB
        backupCount=10
    )
    error_handler.setLevel(logging.ERROR)
    
    # Set formatters
    if settings.LOG_FORMAT == "json":
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    error_handler.setFormatter(formatter)
    
    # Add handlers to root logger
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(error_handler)
    
    # Configure specific loggers
    
    # Database logger
    db_logger = logging.getLogger("sqlalchemy")
    db_logger.setLevel(logging.WARNING)
    
    # HTTP client logger
    http_logger = logging.getLogger("httpx")
    http_logger.setLevel(logging.WARNING)
    
    # Redis logger
    redis_logger = logging.getLogger("aioredis")
    redis_logger.setLevel(logging.WARNING)
    
    # MongoDB logger
    mongo_logger = logging.getLogger("motor")
    mongo_logger.setLevel(logging.WARNING)
    
    # Uvicorn logger
    uvicorn_logger = logging.getLogger("uvicorn")
    uvicorn_logger.setLevel(logging.INFO)
    
    # FastAPI logger
    fastapi_logger = logging.getLogger("fastapi")
    fastapi_logger.setLevel(logging.INFO)
    
    # Create application logger
    app_logger = logging.getLogger("sutazai")
    app_logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    
    app_logger.info("Logging configuration initialized")
    
    return app_logger


# Global logger instance
logger = setup_logging()