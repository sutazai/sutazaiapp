"""
SutazAI Exception Classes
Comprehensive exception hierarchy for error handling and system reliability
"""

import logging
from typing import Optional, Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class SutazaiException(Exception):
    """Base exception class for SutazAI system"""
    
    def __init__(self, message: str, error_code: Optional[str] = None, context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code
        self.context = context or {}
        self.timestamp = datetime.now()
        
        # Log the exception
        logger.error(f"SutazaiException: {message} (code: {error_code})")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary"""
        return {
            "exception_type": self.__class__.__name__,
            "message": self.message,
            "error_code": self.error_code,
            "context": self.context,
            "timestamp": self.timestamp.isoformat()
        }

class SecurityException(SutazaiException):
    """Security-related exceptions"""
    
    def __init__(self, message: str, threat_level: str = "medium", **kwargs):
        super().__init__(message, error_code="SECURITY_ERROR", **kwargs)
        self.threat_level = threat_level

class AuthenticationException(SecurityException):
    """Authentication failures"""
    
    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, error_code="AUTH_ERROR", **kwargs)

class AuthorizationException(SecurityException):
    """Authorization failures"""
    
    def __init__(self, message: str = "Authorization failed", **kwargs):
        super().__init__(message, error_code="AUTHZ_ERROR", **kwargs)

class ValidationException(SutazaiException):
    """Input validation errors"""
    
    def __init__(self, message: str, field: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="VALIDATION_ERROR", **kwargs)
        self.field = field

class ConfigurationException(SutazaiException):
    """Configuration errors"""
    
    def __init__(self, message: str, config_key: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="CONFIG_ERROR", **kwargs)
        self.config_key = config_key

class DatabaseException(SutazaiException):
    """Database operation errors"""
    
    def __init__(self, message: str, operation: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DB_ERROR", **kwargs)
        self.operation = operation

class NeuralNetworkException(SutazaiException):
    """Neural network operation errors"""
    
    def __init__(self, message: str, network_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="NEURAL_ERROR", **kwargs)
        self.network_id = network_id

class ModelException(SutazaiException):
    """Model management errors"""
    
    def __init__(self, message: str, model_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="MODEL_ERROR", **kwargs)
        self.model_name = model_name

class TaskException(SutazaiException):
    """Task processing errors"""
    
    def __init__(self, message: str, task_id: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="TASK_ERROR", **kwargs)
        self.task_id = task_id

class APIException(SutazaiException):
    """API operation errors"""
    
    def __init__(self, message: str, status_code: int = 500, **kwargs):
        super().__init__(message, error_code="API_ERROR", **kwargs)
        self.status_code = status_code

class DeploymentException(SutazaiException):
    """Deployment operation errors"""
    
    def __init__(self, message: str, deployment_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="DEPLOYMENT_ERROR", **kwargs)
        self.deployment_type = deployment_type

class MonitoringException(SutazaiException):
    """Monitoring system errors"""
    
    def __init__(self, message: str, metric_name: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="MONITORING_ERROR", **kwargs)
        self.metric_name = metric_name

class PerformanceException(SutazaiException):
    """Performance-related errors"""
    
    def __init__(self, message: str, component: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="PERFORMANCE_ERROR", **kwargs)
        self.component = component

class SystemException(SutazaiException):
    """System-level errors"""
    
    def __init__(self, message: str, system_component: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="SYSTEM_ERROR", **kwargs)
        self.system_component = system_component

class ResourceException(SutazaiException):
    """Resource management errors"""
    
    def __init__(self, message: str, resource_type: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="RESOURCE_ERROR", **kwargs)
        self.resource_type = resource_type

class NetworkException(SutazaiException):
    """Network operation errors"""
    
    def __init__(self, message: str, endpoint: Optional[str] = None, **kwargs):
        super().__init__(message, error_code="NETWORK_ERROR", **kwargs)
        self.endpoint = endpoint

class TimeoutException(SutazaiException):
    """Timeout errors"""
    
    def __init__(self, message: str, timeout_duration: Optional[float] = None, **kwargs):
        super().__init__(message, error_code="TIMEOUT_ERROR", **kwargs)
        self.timeout_duration = timeout_duration

def handle_exception(exc: Exception, context: Optional[Dict[str, Any]] = None) -> SutazaiException:
    """Convert generic exceptions to SutazaiException"""
    if isinstance(exc, SutazaiException):
        return exc
    
    return SutazaiException(
        message=str(exc),
        error_code="GENERIC_ERROR",
        context=context or {"original_exception": exc.__class__.__name__}
    )