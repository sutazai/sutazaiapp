"""
System Exceptions module.

This module defines custom exceptions as well as helper functions to log
critical errors in a type‑safe manner.
"""

import logging
import traceback
import json
from datetime import datetime
import os
from typing import Dict, Any, Optional, Tuple, Type, TypeVar, Union, List
from types import TracebackType

# Define generic exception type
ExceptionType = TypeVar('ExceptionType', bound=BaseException)

logger = logging.getLogger(__name__)

class ExceptionTracker:
    """
    Advanced exception tracking and reporting mechanism.
    """
    _instance = None
    _exceptions_log: List[Dict[str, Any]] = []

    def __new__(cls):
        if not cls._instance:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def log_exception(
        cls, 
        exception: Exception, 
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log and track exceptions with comprehensive details.
        """
        exception_details = {
            "timestamp": datetime.now().isoformat(),
            "type": type(exception).__name__,
            "message": str(exception),
            "traceback": traceback.format_exc(),
            "context": context or {},
            "unique_id": f"EXC-{hash(exception)}"
        }
        
        cls._exceptions_log.append(exception_details)
        
        # Log to file
        log_dir = "/opt/sutazai_project/SutazAI/logs/exceptions"
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"{exception_details['unique_id']}.json")
        
        with open(log_file, 'w') as f:
            json.dump(exception_details, f, indent=4)
        
        # Log to system logger
        logger.error(
            f"Exception Logged: {exception_details['unique_id']}\n"
            f"Type: {exception_details['type']}\n"
            f"Message: {exception_details['message']}"
        )
        
        return exception_details['unique_id']

def log_critical_exception(
    exc: Exception, 
    context: Optional[Dict[str, Any]] = None
) -> str:
    """
    Logs a critical exception with comprehensive tracking.
    """
    tracker = ExceptionTracker()
    return tracker.log_exception(exc, context)

class SutazAIBaseException(Exception):
    """
    Enhanced base exception for SutazAI system with comprehensive tracking.
    """
    
    def __init__(
        self, 
        message: str, 
        error_code: Optional[str] = None, 
        context: Optional[dict] = None
    ):
        """
        Initialize a base exception with detailed information.
        
        Args:
            message (str): Descriptive error message
            error_code (Optional[str]): Unique error identifier
            context (Optional[dict]): Additional context about the error
        """
        super().__init__(message)
        self.message = message
        self.error_code = error_code or "SUTAZAI_UNKNOWN_ERROR"
        self.context = context or {}
        self.trace = traceback.format_exc()
        
        # Use ExceptionTracker for comprehensive logging
        ExceptionTracker().log_exception(
            self, 
            context={
                "error_code": self.error_code,
                **self.context
            }
        )
    
    def to_dict(self) -> dict:
        """
        Convert exception to a comprehensive dictionary for serialization.
        
        Returns:
            dict: Structured exception information
        """
        return {
            "error_code": self.error_code,
            "message": self.message,
            "context": self.context,
            "trace": self.trace,
            "timestamp": datetime.now().isoformat()
        }

# Custom exception classes
class ConfigurationError(SutazAIBaseException):
    """Raised when there's an issue with system configuration."""
    def __init__(self, message: str, config_key: Optional[str] = None):
        super().__init__(
            message, 
            error_code="SUTAZAI_CONFIG_ERROR",
            context={"config_key": config_key}
        )

class SecurityViolationError(SutazAIBaseException):
    """Raised when a security policy is violated."""
    def __init__(self, message: str, security_context: Optional[dict] = None):
        super().__init__(
            message,
            error_code="SUTAZAI_SECURITY_VIOLATION",
            context=security_context
        )

class ResourceAllocationError(SutazAIBaseException):
    """Raised when resource allocation fails."""
    def __init__(self, message: str, resource_details: Optional[dict] = None):
        super().__init__(
            message,
            error_code="SUTAZAI_RESOURCE_ALLOCATION_ERROR",
            context=resource_details
        )

class ComponentInitializationError(SutazAIBaseException):
    """Raised when a system component fails to initialize."""
    def __init__(self, component_name: str, reason: Optional[str] = None):
        super().__init__(
            f"Failed to initialize component: {component_name}",
            error_code="SUTAZAI_COMPONENT_INIT_ERROR",
            context={
                "component": component_name,
                "reason": reason
            }
        )

def global_exception_handler(
    exc_type: Type[BaseException], 
    exc_value: BaseException, 
    exc_traceback: Optional[TracebackType]
) -> None:
    """
    Global exception handler for unhandled exceptions.
    
    Args:
        exc_type (Type[BaseException]): Exception type
        exc_value (BaseException): Exception instance
        exc_traceback (Optional[TracebackType]): Traceback object
    """
    logger = logging.getLogger('SutazAI.GlobalExceptionHandler')
    logger.critical(
        "Unhandled Exception",
        exc_info=(exc_type, exc_value, exc_traceback)
    )
    # Optionally trigger emergency protocols or notifications 

class OptExcInfo:
    def __init__(
        self, 
        exc_info: Optional[Tuple[Type[BaseException], BaseException, Optional[TracebackType]]] = None
    ):
        self.exc_info = exc_info

    def get_traceback(self) -> Optional[str]:
        if self.exc_info:
            return ''.join(traceback.format_exception(*self.exc_info))
        return None 