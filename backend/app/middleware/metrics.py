"""
Prometheus Metrics Middleware
Comprehensive instrumentation for all API endpoints
"""

import time
import logging
from typing import Callable, Optional
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from app.core.config import settings

# Import prometheus_client - required for production metrics
from prometheus_client import Counter, Histogram, Gauge, Info

logger = logging.getLogger(__name__)

# HTTP Metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status_code']
)

http_request_duration_seconds = Histogram(
    'http_request_duration_seconds',
    'HTTP request latency',
    ['method', 'endpoint'],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0)
)

http_request_size_bytes = Histogram(
    'http_request_size_bytes',
    'HTTP request size in bytes',
    ['method', 'endpoint']
)

http_response_size_bytes = Histogram(
    'http_response_size_bytes',
    'HTTP response size in bytes',
    ['method', 'endpoint']
)

http_requests_in_progress = Gauge(
    'http_requests_in_progress',
    'Number of HTTP requests in progress',
    ['method', 'endpoint']
)

http_exceptions_total = Counter(
    'http_exceptions_total',
    'Total HTTP exceptions',
    ['method', 'endpoint', 'exception_type']
)

# Authentication Metrics
auth_login_total = Counter(
    'auth_login_total',
    'Total login attempts',
    ['status']
)

auth_login_failures_total = Counter(
    'auth_login_failures_total',
    'Total failed login attempts',
    ['reason']
)

auth_token_generation_total = Counter(
    'auth_token_generation_total',
    'Total tokens generated',
    ['token_type']
)

auth_account_lockouts_total = Counter(
    'auth_account_lockouts_total',
    'Total account lockouts'
)

auth_password_resets_total = Counter(
    'auth_password_resets_total',
    'Total password reset requests'
)

auth_email_verifications_total = Counter(
    'auth_email_verifications_total',
    'Total email verifications'
)

# Database Metrics
db_queries_total = Counter(
    'db_queries_total',
    'Total database queries',
    ['operation', 'table']
)

db_query_duration_seconds = Histogram(
    'db_query_duration_seconds',
    'Database query duration',
    ['operation', 'table'],
    buckets=(0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)
)

db_connection_pool_size = Gauge(
    'db_connection_pool_size',
    'Database connection pool size'
)

db_connection_pool_in_use = Gauge(
    'db_connection_pool_in_use',
    'Database connections currently in use'
)

db_connection_errors_total = Counter(
    'db_connection_errors_total',
    'Total database connection errors',
    ['error_type']
)

# Redis Cache Metrics
cache_hits_total = Counter(
    'cache_hits_total',
    'Total cache hits',
    ['cache_key_prefix']
)

cache_misses_total = Counter(
    'cache_misses_total',
    'Total cache misses',
    ['cache_key_prefix']
)

cache_set_operations_total = Counter(
    'cache_set_operations_total',
    'Total cache set operations'
)

cache_delete_operations_total = Counter(
    'cache_delete_operations_total',
    'Total cache delete operations'
)

cache_operation_duration_seconds = Histogram(
    'cache_operation_duration_seconds',
    'Cache operation duration',
    ['operation']
)

# RabbitMQ Metrics
rabbitmq_messages_published_total = Counter(
    'rabbitmq_messages_published_total',
    'Total messages published to RabbitMQ',
    ['exchange', 'routing_key']
)

rabbitmq_messages_consumed_total = Counter(
    'rabbitmq_messages_consumed_total',
    'Total messages consumed from RabbitMQ',
    ['queue']
)

rabbitmq_message_processing_duration_seconds = Histogram(
    'rabbitmq_message_processing_duration_seconds',
    'Message processing duration',
    ['queue']
)

rabbitmq_connection_errors_total = Counter(
    'rabbitmq_connection_errors_total',
    'Total RabbitMQ connection errors'
)

# Vector Database Metrics
vector_db_operations_total = Counter(
    'vector_db_operations_total',
    'Total vector database operations',
    ['db_type', 'operation']
)

vector_db_operation_duration_seconds = Histogram(
    'vector_db_operation_duration_seconds',
    'Vector database operation duration',
    ['db_type', 'operation']
)

vector_db_collection_size = Gauge(
    'vector_db_collection_size',
    'Number of vectors in collection',
    ['db_type', 'collection']
)

# External API Metrics
external_api_calls_total = Counter(
    'external_api_calls_total',
    'Total external API calls',
    ['service', 'endpoint']
)

external_api_call_duration_seconds = Histogram(
    'external_api_call_duration_seconds',
    'External API call duration',
    ['service', 'endpoint']
)

external_api_errors_total = Counter(
    'external_api_errors_total',
    'Total external API errors',
    ['service', 'error_type']
)

# System Metrics
app_info = Info('app', 'Application information')
app_info.info({
    'name': settings.APP_NAME,
    'version': settings.APP_VERSION,
    'environment': settings.ENVIRONMENT
})


class PrometheusMetricsMiddleware(BaseHTTPMiddleware):
    """Middleware to collect Prometheus metrics for all requests"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and collect metrics
        
        Args:
            request: Incoming HTTP request
            call_next: Next middleware/handler
            
        Returns:
            HTTP response
        """
        # Extract endpoint from path
        endpoint = request.url.path
        method = request.method
        
        # Increment in-progress counter
        http_requests_in_progress.labels(method=method, endpoint=endpoint).inc()
        
        # Measure request size
        request_size = int(request.headers.get('content-length', 0))
        if request_size > 0:
            http_request_size_bytes.labels(method=method, endpoint=endpoint).observe(request_size)
        
        # Start timer
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            
            # Record metrics
            status_code = response.status_code
            http_requests_total.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code
            ).inc()
            
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            # Measure response size
            response_size = int(response.headers.get('content-length', 0))
            if response_size > 0:
                http_response_size_bytes.labels(
                    method=method,
                    endpoint=endpoint
                ).observe(response_size)
            
            return response
            
        except Exception as e:
            # Record exception
            http_exceptions_total.labels(
                method=method,
                endpoint=endpoint,
                exception_type=type(e).__name__
            ).inc()
            
            # Calculate duration even on error
            duration = time.time() - start_time
            http_request_duration_seconds.labels(
                method=method,
                endpoint=endpoint
            ).observe(duration)
            
            # Re-raise exception
            raise
            
        finally:
            # Decrement in-progress counter
            http_requests_in_progress.labels(method=method, endpoint=endpoint).dec()


# Helper functions for metric recording
def record_auth_login(success: bool, failure_reason: str = None):
    """Record login attempt metrics"""
    status = 'success' if success else 'failure'
    auth_login_total.labels(status=status).inc()
    
    if not success and failure_reason:
        auth_login_failures_total.labels(reason=failure_reason).inc()


def record_token_generation(token_type: str):
    """Record token generation"""
    auth_token_generation_total.labels(token_type=token_type).inc()


def record_account_lockout():
    """Record account lockout"""
    auth_account_lockouts_total.inc()


def record_password_reset():
    """Record password reset request"""
    auth_password_resets_total.inc()


def record_email_verification():
    """Record email verification"""
    auth_email_verifications_total.inc()


def record_db_query(operation: str, table: str, duration: float):
    """Record database query metrics"""
    db_queries_total.labels(operation=operation, table=table).inc()
    db_query_duration_seconds.labels(operation=operation, table=table).observe(duration)


def record_db_error(error_type: str):
    """Record database error"""
    db_connection_errors_total.labels(error_type=error_type).inc()


def record_cache_operation(operation: str, hit: bool = None, key_prefix: str = "default"):
    """Record cache operation metrics"""
    if operation == "get":
        if hit:
            cache_hits_total.labels(cache_key_prefix=key_prefix).inc()
        else:
            cache_misses_total.labels(cache_key_prefix=key_prefix).inc()
    elif operation == "set":
        cache_set_operations_total.inc()
    elif operation == "delete":
        cache_delete_operations_total.inc()


def record_rabbitmq_publish(exchange: str, routing_key: str):
    """Record RabbitMQ message publication"""
    rabbitmq_messages_published_total.labels(
        exchange=exchange,
        routing_key=routing_key
    ).inc()


def record_rabbitmq_consume(queue: str, duration: float):
    """Record RabbitMQ message consumption"""
    rabbitmq_messages_consumed_total.labels(queue=queue).inc()
    rabbitmq_message_processing_duration_seconds.labels(queue=queue).observe(duration)


def record_vector_db_operation(db_type: str, operation: str, duration: float):
    """Record vector database operation"""
    vector_db_operations_total.labels(db_type=db_type, operation=operation).inc()
    vector_db_operation_duration_seconds.labels(db_type=db_type, operation=operation).observe(duration)


def record_external_api_call(service: str, endpoint: str, duration: float, success: bool = True, error_type: str = None):
    """Record external API call metrics"""
    external_api_calls_total.labels(service=service, endpoint=endpoint).inc()
    external_api_call_duration_seconds.labels(service=service, endpoint=endpoint).observe(duration)
    
    if not success and error_type:
        external_api_errors_total.labels(service=service, error_type=error_type).inc()


# Export all metrics for easy import
__all__ = [
    'PrometheusMetricsMiddleware',
    'record_auth_login',
    'record_token_generation',
    'record_account_lockout',
    'record_password_reset',
    'record_email_verification',
    'record_db_query',
    'record_db_error',
    'record_cache_operation',
    'record_rabbitmq_publish',
    'record_rabbitmq_consume',
    'record_vector_db_operation',
    'record_external_api_call',
    'http_requests_total',
    'http_request_duration_seconds',
    'db_queries_total',
    'cache_hits_total',
    'cache_misses_total'
]
