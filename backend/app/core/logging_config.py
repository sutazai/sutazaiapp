"""
Enterprise-grade logging configuration
Compliant with Professional Project Standards Rule 5
Implements structured logging, audit trails, and security monitoring
"""

import logging
import logging.handlers
import json
import sys
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import traceback
from pythonjsonlogger import jsonlogger
import uuid

# Log directory with proper permissions
LOG_DIR = Path("/opt/sutazaiapp/logs")
LOG_DIR.mkdir(exist_ok=True, mode=0o755)

class SecurityAuditLogger:
    """Specialized logger for security-related events"""
    
    def __init__(self):
        self.logger = logging.getLogger("security.audit")
        self.logger.setLevel(logging.INFO)
        
        # Security audit file handler with rotation
        audit_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "security_audit.log",
            maxBytes=10_000_000,  # 10MB
            backupCount=100  # Keep 100 backup files
        )
        audit_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(audit_handler)
    
    def log_authentication(self, event_type: str, user_id: Optional[str] = None, 
                          success: bool = True, ip_address: str = None, **kwargs):
        """Log authentication events"""
        self.logger.info(
            "Authentication event",
            extra={
                "event_type": event_type,
                "user_id": user_id,
                "success": success,
                "ip_address": ip_address,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "correlation_id": str(uuid.uuid4()),
                **kwargs
            }
        )
    
    def log_authorization(self, user_id: str, resource: str, action: str, 
                         granted: bool, **kwargs):
        """Log authorization decisions"""
        self.logger.info(
            "Authorization check",
            extra={
                "user_id": user_id,
                "resource": resource,
                "action": action,
                "granted": granted,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **kwargs
            }
        )
    
    def log_data_access(self, user_id: str, table: str, operation: str, 
                       record_count: int = 1, **kwargs):
        """Log data access for compliance"""
        self.logger.info(
            "Data access",
            extra={
                "user_id": user_id,
                "table": table,
                "operation": operation,
                "record_count": record_count,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **kwargs
            }
        )
    
    def log_security_violation(self, violation_type: str, details: Dict[str, Any], 
                              severity: str = "HIGH"):
        """Log security violations and potential attacks"""
        self.logger.critical(
            f"SECURITY VIOLATION: {violation_type}",
            extra={
                "violation_type": violation_type,
                "severity": severity,
                "details": details,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "alert": True
            }
        )


class PerformanceLogger:
    """Logger for performance metrics and monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger("performance.metrics")
        self.logger.setLevel(logging.INFO)
        
        # Performance metrics file handler
        perf_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "performance_metrics.log",
            maxBytes=10_000_000,
            backupCount=50
        )
        perf_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(perf_handler)
    
    def log_request(self, endpoint: str, method: str, duration_ms: float, 
                   status_code: int, **kwargs):
        """Log API request performance"""
        self.logger.info(
            "API request",
            extra={
                "endpoint": endpoint,
                "method": method,
                "duration_ms": duration_ms,
                "status_code": status_code,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **kwargs
            }
        )
    
    def log_database_query(self, query_type: str, table: str, duration_ms: float, 
                          rows_affected: int = 0, **kwargs):
        """Log database query performance"""
        self.logger.info(
            "Database query",
            extra={
                "query_type": query_type,
                "table": table,
                "duration_ms": duration_ms,
                "rows_affected": rows_affected,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **kwargs
            }
        )
    
    def log_cache_operation(self, operation: str, key: str, hit: bool, 
                           duration_ms: float, **kwargs):
        """Log cache operations"""
        self.logger.info(
            "Cache operation",
            extra={
                "operation": operation,
                "key": key,
                "cache_hit": hit,
                "duration_ms": duration_ms,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **kwargs
            }
        )


class JsonFormatter(jsonlogger.JsonFormatter):
    """Custom JSON formatter with additional fields"""
    
    def add_fields(self, log_record, record, message_dict):
        super(JsonFormatter, self).add_fields(log_record, record, message_dict)
        
        # Add custom fields
        log_record['timestamp'] = datetime.now(timezone.utc).isoformat()
        log_record['level'] = record.levelname
        log_record['logger'] = record.name
        log_record['module'] = record.module
        log_record['function'] = record.funcName
        log_record['line'] = record.lineno
        
        # Add process and thread info for debugging
        log_record['process_id'] = os.getpid()
        log_record['thread_name'] = record.threadName
        
        # Add environment info
        log_record['environment'] = os.getenv('APP_ENV', 'development')
        log_record['app_name'] = os.getenv('APP_NAME', 'sutazai-backend')
        
        # Add exception info if present
        if record.exc_info:
            log_record['exception'] = traceback.format_exception(*record.exc_info)


class ErrorTracker:
    """Track and analyze errors for monitoring"""
    
    def __init__(self):
        self.logger = logging.getLogger("error.tracker")
        self.logger.setLevel(logging.ERROR)
        
        # Error tracking file handler
        error_handler = logging.handlers.RotatingFileHandler(
            LOG_DIR / "errors.log",
            maxBytes=10_000_000,
            backupCount=100
        )
        error_handler.setFormatter(JsonFormatter())
        self.logger.addHandler(error_handler)
    
    def track_error(self, error: Exception, context: Dict[str, Any] = None):
        """Track application errors with context"""
        self.logger.error(
            f"Application error: {type(error).__name__}",
            extra={
                "error_type": type(error).__name__,
                "error_message": str(error),
                "context": context or {},
                "stack_trace": traceback.format_exc(),
                "timestamp": datetime.now(timezone.utc).isoformat()
            },
            exc_info=True
        )
    
    def track_validation_error(self, field: str, value: Any, error: str, **kwargs):
        """Track validation errors"""
        self.logger.warning(
            "Validation error",
            extra={
                "field": field,
                "value": str(value)[:100],  # Truncate for security
                "error": error,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                **kwargs
            }
        )


def configure_logging():
    """Configure comprehensive logging for the application"""
    
    # Determine log level from environment
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    
    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level))
    
    # Remove default handlers
    root_logger.handlers = []
    
    # Console handler with JSON formatting for structured logs
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(console_handler)
    
    # File handler for all application logs
    app_handler = logging.handlers.RotatingFileHandler(
        LOG_DIR / "application.log",
        maxBytes=50_000_000,  # 50MB
        backupCount=10
    )
    app_handler.setFormatter(JsonFormatter())
    root_logger.addHandler(app_handler)
    
    # Configure specific loggers
    configure_module_loggers()
    
    # Initialize specialized loggers
    security_logger = SecurityAuditLogger()
    performance_logger = PerformanceLogger()
    error_tracker = ErrorTracker()
    
    return security_logger, performance_logger, error_tracker


def configure_module_loggers():
    """Configure logging for specific modules"""
    
    # Reduce noise from third-party libraries
    logging.getLogger('sqlalchemy.engine').setLevel(logging.WARNING)
    logging.getLogger('aioredis').setLevel(logging.WARNING)
    logging.getLogger('neo4j').setLevel(logging.WARNING)
    logging.getLogger('uvicorn.access').setLevel(logging.WARNING)
    
    # Critical modules with DEBUG level in development
    if os.getenv('DEBUG', 'false').lower() == 'true':
        logging.getLogger('app.core.security').setLevel(logging.DEBUG)
        logging.getLogger('app.api.dependencies.auth').setLevel(logging.DEBUG)
        logging.getLogger('app.services.connections').setLevel(logging.DEBUG)


# Initialize loggers on import
security_audit_logger, performance_logger, error_tracker = configure_logging()

# Export for use in other modules
__all__ = [
    'security_audit_logger',
    'performance_logger', 
    'error_tracker',
    'configure_logging',
    'JsonFormatter'
]