"""
Request/Response Logging Middleware
Comprehensive logging of all HTTP interactions with correlation IDs
"""

import time
import logging
import uuid
import json
from typing import Callable
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to log all HTTP requests and responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request/response with correlation ID
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Generate correlation ID
        correlation_id = str(uuid.uuid4())
        request.state.correlation_id = correlation_id
        
        # Extract request details
        method = request.method
        url = str(request.url)
        path = request.url.path
        client_host = request.client.host if request.client else "unknown"
        user_agent = request.headers.get("user-agent", "unknown")
        
        # Log request
        request_time = datetime.now(timezone.utc)
        logger.info(
            f"[{correlation_id}] Incoming request: {method} {path}",
            extra={
                "correlation_id": correlation_id,
                "method": method,
                "path": path,
                "url": url,
                "client_host": client_host,
                "user_agent": user_agent,
                "timestamp": request_time.isoformat()
            }
        )
        
        # Start timer
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response
            logger.info(
                f"[{correlation_id}] Response: {response.status_code} ({process_time:.3f}s)",
                extra={
                    "correlation_id": correlation_id,
                    "method": method,
                    "path": path,
                    "status_code": response.status_code,
                    "process_time": process_time,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                }
            )
            
            # Add correlation ID to response headers
            response.headers["X-Correlation-ID"] = correlation_id
            response.headers["X-Process-Time"] = f"{process_time:.3f}"
            
            return response
            
        except Exception as e:
            # Calculate processing time even on error
            process_time = time.time() - start_time
            
            # Log error
            logger.error(
                f"[{correlation_id}] Request failed: {type(e).__name__}: {str(e)} ({process_time:.3f}s)",
                extra={
                    "correlation_id": correlation_id,
                    "method": method,
                    "path": path,
                    "exception_type": type(e).__name__,
                    "exception_message": str(e),
                    "process_time": process_time,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                },
                exc_info=True
            )
            
            # Re-raise exception
            raise


class StructuredLoggingHandler(logging.Handler):
    """Custom logging handler that outputs structured JSON logs"""
    
    def emit(self, record: logging.LogRecord):
        """
        Emit a log record as structured JSON
        
        Args:
            record: Log record to emit
        """
        try:
            # Build structured log entry
            log_entry = {
                "timestamp": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "function": record.funcName,
                "line": record.lineno
            }
            
            # Add correlation ID if available
            if hasattr(record, 'correlation_id'):
                log_entry["correlation_id"] = record.correlation_id
            
            # Add extra fields
            for key, value in record.__dict__.items():
                if key not in ['name', 'msg', 'args', 'created', 'filename', 'funcName', 
                              'levelname', 'levelno', 'lineno', 'module', 'msecs', 
                              'message', 'pathname', 'process', 'processName', 'relativeCreated', 
                              'stack_info', 'thread', 'threadName', 'exc_info', 'exc_text']:
                    log_entry[key] = value
            
            # Add exception info if present
            if record.exc_info:
                log_entry["exception"] = self.format_exception(record.exc_info)
            
            # Output as JSON
            print(json.dumps(log_entry))
            
        except Exception:
            self.handleError(record)
    
    def format_exception(self, exc_info):
        """Format exception information"""
        import traceback
        return {
            "type": exc_info[0].__name__ if exc_info[0] else "Unknown",
            "message": str(exc_info[1]) if exc_info[1] else "",
            "traceback": traceback.format_exception(*exc_info)
        }


def configure_structured_logging(log_level: str = "INFO"):
    """
    Configure structured logging for the application
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Get root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    
    # Remove existing handlers
    root_logger.handlers.clear()
    
    # Add structured logging handler
    structured_handler = StructuredLoggingHandler()
    root_logger.addHandler(structured_handler)
    
    logger.info("Structured logging configured", extra={"log_level": log_level})


# Export for easy import
__all__ = [
    'RequestLoggingMiddleware',
    'StructuredLoggingHandler',
    'configure_structured_logging'
]
