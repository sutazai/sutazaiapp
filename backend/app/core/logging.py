"""
Logging configuration for SutazAI
"""
import logging
import sys
from typing import Optional
from pathlib import Path
import json
from datetime import datetime

from pythonjsonlogger import jsonlogger

# Custom log formatter
class CustomJsonFormatter(jsonlogger.JsonFormatter):
    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)
        log_record['timestamp'] = datetime.utcnow().isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    json_logs: bool = True
) -> logging.Logger:
    """
    Setup logging configuration
    
    Args:
        log_level: Logging level
        log_file: Optional log file path
        json_logs: Whether to use JSON formatting
        
    Returns:
        Logger instance
    """
    # Create logger
    logger = logging.getLogger("sutazai")
    logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if json_logs:
        formatter = CustomJsonFormatter()
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=104857600,  # 100MB
            backupCount=10
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance"""
    return logging.getLogger(f"sutazai.{name}")