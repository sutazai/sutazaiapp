"""
SutazaiApp Exception Module

Defines custom exceptions and exception handlers for consistent error handling
across the application.
"""

import logging
import uuid
from typing import Any, Dict, Optional, List, Awaitable

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse, Response
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException as StarletteHTTPException

from backend.core.config import settings

# Configure logger
logger = logging.getLogger("sutazaiapp.exceptions")


# Base exception class
class SutazaiException(Exception):
    """Base exception class for SutazaiApp"""

    def __init__(
        self,
        message: str,
        status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
        error_code: str = "INTERNAL_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.details = details or {}
        super().__init__(self.message)


# Specific exception types
class NotFoundException(SutazaiException):
    """Exception raised when a resource is not found"""

    def __init__(
        self,
        message: str = "Resource not found",
        resource_type: Optional[str] = None,
        resource_id: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        resource_details = {}
        if resource_type:
            resource_details["resource_type"] = resource_type
        if resource_id:
            resource_details["resource_id"] = resource_id

        error_code = "NOT_FOUND"
        if resource_type:
            error_code = f"{resource_type.upper()}_NOT_FOUND"

        super().__init__(
            message=message,
            status_code=status.HTTP_404_NOT_FOUND,
            error_code=error_code,
            details=details or resource_details,
        )


class ValidationException(SutazaiException):
    """Exception raised for validation errors"""

    def __init__(
        self,
        message: str = "Validation error",
        field_errors: Optional[Dict[str, str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        validation_details = {}
        if field_errors:
            validation_details["field_errors"] = field_errors

        super().__init__(
            message=message,
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            error_code="VALIDATION_ERROR",
            details=details or validation_details,
        )


class AuthenticationException(SutazaiException):
    """Exception raised for authentication errors"""

    def __init__(
        self,
        message: str = "Authentication error",
        error_code: str = "AUTHENTICATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_401_UNAUTHORIZED,
            error_code=error_code,
            details=details,
        )


class AuthorizationException(SutazaiException):
    """Exception raised for authorization errors"""

    def __init__(
        self,
        message: str = "Authorization error",
        required_permissions: Optional[List[str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        authorization_details = {}
        if required_permissions:
            authorization_details["required_permissions"] = required_permissions

        super().__init__(
            message=message,
            status_code=status.HTTP_403_FORBIDDEN,
            error_code="AUTHORIZATION_ERROR",
            details=details or authorization_details,
        )


class ServiceUnavailableException(SutazaiException):
    """Exception raised when a service is unavailable"""

    def __init__(
        self,
        message: str = "Service unavailable",
        service_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        service_details = {}
        if service_name:
            service_details["service_name"] = service_name

        error_code = "SERVICE_UNAVAILABLE"
        if service_name:
            error_code = f"{service_name.upper()}_UNAVAILABLE"

        super().__init__(
            message=message,
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            error_code=error_code,
            details=details or service_details,
        )


class RateLimitException(SutazaiException):
    """Exception raised when rate limit is exceeded"""

    def __init__(
        self,
        message: str = "Rate limit exceeded",
        limit: Optional[int] = None,
        reset_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        rate_limit_details = {}
        if limit:
            rate_limit_details["limit"] = limit
        if reset_after:
            rate_limit_details["reset_after_seconds"] = reset_after

        super().__init__(
            message=message,
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            error_code="RATE_LIMIT_EXCEEDED",
            details=details or rate_limit_details,
        )


class FileProcessingException(SutazaiException):
    """Exception raised during file processing"""

    def __init__(
        self,
        message: str = "File processing error",
        filename: Optional[str] = None,
        file_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ):
        file_details = {}
        if filename:
            file_details["filename"] = filename
        if file_type:
            file_details["file_type"] = file_type

        super().__init__(
            message=message,
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            error_code="FILE_PROCESSING_ERROR",
            details=details or file_details,
        )


class InvalidRequestException(SutazaiException):
    """Exception raised for invalid requests"""

    def __init__(
        self, message: str = "Invalid request", details: Optional[Dict[str, Any]] = None
    ):
        super().__init__(
            message=message,
            status_code=status.HTTP_400_BAD_REQUEST,
            error_code="INVALID_REQUEST",
            details=details,
        )


# Exception handlers
def create_error_response(
    status_code: int,
    message: str,
    error_code: str = "INTERNAL_ERROR",
    details: Optional[Dict[str, Any]] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a standardized error response"""
    response = {
        "success": False,
        "message": message,
        "error_code": error_code,
        "request_id": request_id,
    }

    if details and (settings.DEBUG or error_code != "INTERNAL_ERROR"):
        response["details"] = details

    return response


async def sutazai_exception_handler(
    request: Request, exc: SutazaiException
) -> Response:
    """Handler for SutazaiException and its subclasses"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Log the exception
    logger.error(
        f"SutazaiException handled: {exc.error_code} - {exc.message} - "
        f"Request ID: {request_id}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            status_code=exc.status_code,
            message=exc.message,
            error_code=exc.error_code,
            details=exc.details,
            request_id=request_id,
        ),
    )


async def http_exception_handler(
    request: Request, exc: StarletteHTTPException
) -> Response:
    """Handler for FastAPI and Starlette HTTP exceptions"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Map HTTP status code to error code
    error_code = "HTTP_ERROR"
    if exc.status_code == 404:
        error_code = "NOT_FOUND"
    elif exc.status_code == 401:
        error_code = "UNAUTHORIZED"
    elif exc.status_code == 403:
        error_code = "FORBIDDEN"
    elif exc.status_code == 429:
        error_code = "RATE_LIMIT_EXCEEDED"

    # Log the exception
    logger.error(
        f"HTTP exception handled: {error_code} ({exc.status_code}) - "
        f"{exc.detail} - Request ID: {request_id}"
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=create_error_response(
            status_code=exc.status_code,
            message=str(exc.detail),
            error_code=error_code,
            request_id=request_id,
        ),
    )


async def validation_exception_handler(
    request: Request, exc: RequestValidationError
) -> Response:
    """Handler for request validation errors"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Extract field errors from the exception
    field_errors = {}
    for error in exc.errors():
        # Get the field path
        loc = error.get("loc", [])
        if len(loc) > 1:  # First element is usually 'body', 'query', etc.
            field = ".".join(str(item) for item in loc[1:])
            field_errors[field] = error.get("msg", "Invalid value")

    # Create a user-friendly message
    message = "Validation error"
    if field_errors:
        fields_list = ", ".join(field_errors.keys())
        message = f"Validation error in fields: {fields_list}"

    # Log the exception
    logger.error(f"Validation exception handled: {message} - Request ID: {request_id}")

    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=create_error_response(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            message=message,
            error_code="VALIDATION_ERROR",
            details={"field_errors": field_errors},
            request_id=request_id,
        ),
    )


async def general_exception_handler(request: Request, exc: Exception) -> Response:
    """Handler for all unhandled exceptions"""
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))

    # Log the exception
    logger.exception(f"Unhandled exception: {str(exc)} - Request ID: {request_id}")

    # Only include detailed error message in debug mode
    error_message = str(exc) if settings.DEBUG else "An unexpected error occurred"

    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=create_error_response(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            message=error_message,
            request_id=request_id,
        ),
    )


def register_exception_handlers(app: FastAPI) -> None:
    """Register all exception handlers with the FastAPI app"""
    # Custom exception handlers
    app.add_exception_handler(SutazaiException, sutazai_exception_handler) # type: ignore [arg-type]

    # Standard FastAPI exception handlers
    app.add_exception_handler(StarletteHTTPException, http_exception_handler) # type: ignore [arg-type]
    app.add_exception_handler(RequestValidationError, validation_exception_handler) # type: ignore [arg-type]

    # Catch-all handler for unhandled exceptions
    app.add_exception_handler(Exception, general_exception_handler) # type: ignore [arg-type]


class ServiceError(Exception):
    """Base exception for service-related errors."""

    pass


class ValidationError(ServiceError):
    """Exception raised for validation errors."""

    pass


class ConfigurationError(ServiceError):
    """Exception raised for configuration errors."""

    pass


class ResourceError(ServiceError):
    """Exception raised for resource-related errors."""

    pass


class ProcessingError(ServiceError):
    """Exception raised for processing errors."""

    pass


class AuthenticationError(ServiceError):
    """Exception raised for authentication errors."""

    pass


class AuthorizationError(ServiceError):
    """Exception raised for authorization errors."""

    pass


class DependencyError(ServiceError):
    """Exception raised for dependency-related errors."""

    pass


class FileError(ServiceError):
    """Exception raised for file-related errors."""

    pass


class NetworkError(ServiceError):
    """Exception raised for network-related errors."""

    pass


class DatabaseError(ServiceError):
    """Exception raised for database-related errors."""

    pass


class CacheError(ServiceError):
    """Exception raised for cache-related errors."""

    pass


class RateLimitError(ServiceError):
    """Exception raised for rate limit violations."""

    pass


class TimeoutError(ServiceError):
    """Exception raised for timeout errors."""

    pass


class ResourceNotFoundError(ResourceError):
    """Exception raised when a requested resource is not found."""

    pass


class ResourceConflictError(ResourceError):
    """Exception raised when there is a conflict with an existing resource."""

    pass


class ResourceLimitError(ResourceError):
    """Exception raised when a resource limit is exceeded."""

    pass


class InvalidInputError(ValidationError):
    """Exception raised for invalid input data."""

    pass


class MissingRequiredError(ValidationError):
    """Exception raised when required data is missing."""

    pass


class FormatError(ValidationError):
    """Exception raised for format-related errors."""

    pass


class ConfigurationMissingError(ConfigurationError):
    """Exception raised when required configuration is missing."""

    pass


class ConfigurationInvalidError(ConfigurationError):
    """Exception raised when configuration is invalid."""

    pass


class ProcessingTimeoutError(ProcessingError):
    """Exception raised when processing times out."""

    pass


class ProcessingFailedError(ProcessingError):
    """Exception raised when processing fails."""

    pass


class FileNotFoundError(FileError):
    """Exception raised when a file is not found."""

    pass


class FilePermissionError(FileError):
    """Exception raised when there are file permission issues."""

    pass


class FileFormatError(FileError):
    """Exception raised when a file format is invalid."""

    pass


class FileSizeError(FileError):
    """Exception raised when a file size limit is exceeded."""

    pass


class NetworkConnectionError(NetworkError):
    """Exception raised for network connection errors."""

    pass


class NetworkTimeoutError(NetworkError):
    """Exception raised for network timeout errors."""

    pass


class DatabaseConnectionError(DatabaseError):
    """Exception raised for database connection errors."""

    pass


class DatabaseQueryError(DatabaseError):
    """Exception raised for database query errors."""

    pass


class CacheConnectionError(CacheError):
    """Exception raised for cache connection errors."""

    pass


class CacheKeyError(CacheError):
    """Exception raised for cache key errors."""

    pass


class RateLimitExceededError(RateLimitError):
    """Exception raised when rate limit is exceeded."""

    pass


class RateLimitResetError(RateLimitError):
    """Exception raised when rate limit reset fails."""

    pass


class OperationTimeoutError(TimeoutError):
    """Exception raised when an operation times out."""

    pass


class RequestTimeoutError(TimeoutError):
    """Exception raised when a request times out."""

    pass
