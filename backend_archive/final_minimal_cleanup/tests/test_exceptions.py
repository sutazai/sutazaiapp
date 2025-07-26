"""
Test Core Exceptions

This module contains tests for the core exceptions.
"""

import unittest

import sys

sys.path.append("/opt/sutazaiapp")

from backend.core.exceptions import (
    ServiceError,
    ValidationError,
    ConfigurationError,
    ResourceError,
    ProcessingError,
    AuthenticationError,
    AuthorizationError,
    DependencyError,
    FileError,
    NetworkError,
    DatabaseError,
    CacheError,
    RateLimitError,
    TimeoutError,
    ResourceNotFoundError,
    ResourceConflictError,
    ResourceLimitError,
    InvalidInputError,
    MissingRequiredError,
    FormatError,
    ConfigurationMissingError,
    ConfigurationInvalidError,
    ProcessingTimeoutError,
    ProcessingFailedError,
    FileNotFoundError,
    FilePermissionError,
    FileFormatError,
    FileSizeError,
    NetworkConnectionError,
    NetworkTimeoutError,
    DatabaseConnectionError,
    DatabaseQueryError,
    CacheConnectionError,
    CacheKeyError,
    RateLimitExceededError,
    RateLimitResetError,
    OperationTimeoutError,
    RequestTimeoutError,
)


class TestCoreExceptions(unittest.TestCase):
    """Tests for core exceptions."""

    def test_service_error(self):
        """Test ServiceError base class."""
        error = ServiceError("Test error")
        self.assertEqual(str(error), "Test error")
        self.assertIsInstance(error, Exception)

    def test_validation_error(self):
        """Test ValidationError."""
        error = ValidationError("Invalid input")
        self.assertEqual(str(error), "Invalid input")
        self.assertIsInstance(error, ServiceError)

    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid configuration")
        self.assertEqual(str(error), "Invalid configuration")
        self.assertIsInstance(error, ServiceError)

    def test_resource_error(self):
        """Test ResourceError."""
        error = ResourceError("Resource error")
        self.assertEqual(str(error), "Resource error")
        self.assertIsInstance(error, ServiceError)

    def test_processing_error(self):
        """Test ProcessingError."""
        error = ProcessingError("Processing failed")
        self.assertEqual(str(error), "Processing failed")
        self.assertIsInstance(error, ServiceError)

    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Authentication failed")
        self.assertEqual(str(error), "Authentication failed")
        self.assertIsInstance(error, ServiceError)

    def test_authorization_error(self):
        """Test AuthorizationError."""
        error = AuthorizationError("Not authorized")
        self.assertEqual(str(error), "Not authorized")
        self.assertIsInstance(error, ServiceError)

    def test_dependency_error(self):
        """Test DependencyError."""
        error = DependencyError("Missing dependency")
        self.assertEqual(str(error), "Missing dependency")
        self.assertIsInstance(error, ServiceError)

    def test_file_error(self):
        """Test FileError."""
        error = FileError("File error")
        self.assertEqual(str(error), "File error")
        self.assertIsInstance(error, ServiceError)

    def test_network_error(self):
        """Test NetworkError."""
        error = NetworkError("Network error")
        self.assertEqual(str(error), "Network error")
        self.assertIsInstance(error, ServiceError)

    def test_database_error(self):
        """Test DatabaseError."""
        error = DatabaseError("Database error")
        self.assertEqual(str(error), "Database error")
        self.assertIsInstance(error, ServiceError)

    def test_cache_error(self):
        """Test CacheError."""
        error = CacheError("Cache error")
        self.assertEqual(str(error), "Cache error")
        self.assertIsInstance(error, ServiceError)

    def test_rate_limit_error(self):
        """Test RateLimitError."""
        error = RateLimitError("Rate limit exceeded")
        self.assertEqual(str(error), "Rate limit exceeded")
        self.assertIsInstance(error, ServiceError)

    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError("Operation timed out")
        self.assertEqual(str(error), "Operation timed out")
        self.assertIsInstance(error, ServiceError)

    def test_resource_not_found_error(self):
        """Test ResourceNotFoundError."""
        error = ResourceNotFoundError("Resource not found")
        self.assertEqual(str(error), "Resource not found")
        self.assertIsInstance(error, ResourceError)

    def test_resource_conflict_error(self):
        """Test ResourceConflictError."""
        error = ResourceConflictError("Resource conflict")
        self.assertEqual(str(error), "Resource conflict")
        self.assertIsInstance(error, ResourceError)

    def test_resource_limit_error(self):
        """Test ResourceLimitError."""
        error = ResourceLimitError("Resource limit reached")
        self.assertEqual(str(error), "Resource limit reached")
        self.assertIsInstance(error, ResourceError)

    def test_invalid_input_error(self):
        """Test InvalidInputError."""
        error = InvalidInputError("Invalid input")
        self.assertEqual(str(error), "Invalid input")
        self.assertIsInstance(error, ValidationError)

    def test_missing_required_error(self):
        """Test MissingRequiredError."""
        error = MissingRequiredError("Missing required field")
        self.assertEqual(str(error), "Missing required field")
        self.assertIsInstance(error, ValidationError)

    def test_format_error(self):
        """Test FormatError."""
        error = FormatError("Invalid format")
        self.assertEqual(str(error), "Invalid format")
        self.assertIsInstance(error, ValidationError)

    def test_configuration_missing_error(self):
        """Test ConfigurationMissingError."""
        error = ConfigurationMissingError("Missing configuration")
        self.assertEqual(str(error), "Missing configuration")
        self.assertIsInstance(error, ConfigurationError)

    def test_configuration_invalid_error(self):
        """Test ConfigurationInvalidError."""
        error = ConfigurationInvalidError("Invalid configuration")
        self.assertEqual(str(error), "Invalid configuration")
        self.assertIsInstance(error, ConfigurationError)

    def test_processing_timeout_error(self):
        """Test ProcessingTimeoutError."""
        error = ProcessingTimeoutError("Processing timed out")
        self.assertEqual(str(error), "Processing timed out")
        self.assertIsInstance(error, ProcessingError)

    def test_processing_failed_error(self):
        """Test ProcessingFailedError."""
        error = ProcessingFailedError("Processing failed")
        self.assertEqual(str(error), "Processing failed")
        self.assertIsInstance(error, ProcessingError)

    def test_file_not_found_error(self):
        """Test FileNotFoundError."""
        error = FileNotFoundError("File not found")
        self.assertEqual(str(error), "File not found")
        self.assertIsInstance(error, FileError)

    def test_file_permission_error(self):
        """Test FilePermissionError."""
        error = FilePermissionError("Permission denied")
        self.assertEqual(str(error), "Permission denied")
        self.assertIsInstance(error, FileError)

    def test_file_format_error(self):
        """Test FileFormatError."""
        error = FileFormatError("Invalid file format")
        self.assertEqual(str(error), "Invalid file format")
        self.assertIsInstance(error, FileError)

    def test_file_size_error(self):
        """Test FileSizeError."""
        error = FileSizeError("File too large")
        self.assertEqual(str(error), "File too large")
        self.assertIsInstance(error, FileError)

    def test_network_connection_error(self):
        """Test NetworkConnectionError."""
        error = NetworkConnectionError("Connection failed")
        self.assertEqual(str(error), "Connection failed")
        self.assertIsInstance(error, NetworkError)

    def test_network_timeout_error(self):
        """Test NetworkTimeoutError."""
        error = NetworkTimeoutError("Network timeout")
        self.assertEqual(str(error), "Network timeout")
        self.assertIsInstance(error, NetworkError)

    def test_database_connection_error(self):
        """Test DatabaseConnectionError."""
        error = DatabaseConnectionError("Database connection failed")
        self.assertEqual(str(error), "Database connection failed")
        self.assertIsInstance(error, DatabaseError)

    def test_database_query_error(self):
        """Test DatabaseQueryError."""
        error = DatabaseQueryError("Query failed")
        self.assertEqual(str(error), "Query failed")
        self.assertIsInstance(error, DatabaseError)

    def test_cache_connection_error(self):
        """Test CacheConnectionError."""
        error = CacheConnectionError("Cache connection failed")
        self.assertEqual(str(error), "Cache connection failed")
        self.assertIsInstance(error, CacheError)

    def test_cache_key_error(self):
        """Test CacheKeyError."""
        error = CacheKeyError("Cache key not found")
        self.assertEqual(str(error), "Cache key not found")
        self.assertIsInstance(error, CacheError)

    def test_rate_limit_exceeded_error(self):
        """Test RateLimitExceededError."""
        error = RateLimitExceededError("Rate limit exceeded")
        self.assertEqual(str(error), "Rate limit exceeded")
        self.assertIsInstance(error, RateLimitError)

    def test_rate_limit_reset_error(self):
        """Test RateLimitResetError."""
        error = RateLimitResetError("Rate limit reset")
        self.assertEqual(str(error), "Rate limit reset")
        self.assertIsInstance(error, RateLimitError)

    def test_operation_timeout_error(self):
        """Test OperationTimeoutError."""
        error = OperationTimeoutError("Operation timed out")
        self.assertEqual(str(error), "Operation timed out")
        self.assertIsInstance(error, TimeoutError)

    def test_request_timeout_error(self):
        """Test RequestTimeoutError."""
        error = RequestTimeoutError("Request timed out")
        self.assertEqual(str(error), "Request timed out")
        self.assertIsInstance(error, TimeoutError)


if __name__ == "__main__":
    unittest.main()
