"""
Enterprise-level middleware for FastAPI
Includes rate limiting, circuit breaker, caching, and request tracking
"""

from fastapi import Request, Response, HTTPException
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
import time
import uuid
from typing import Callable
import logging
from pybreaker import CircuitBreaker, CircuitBreakerError
from aiocache import Cache
from aiocache.serializers import JsonSerializer
from prometheus_client import Counter, Histogram, Gauge
import asyncio

logger = logging.getLogger(__name__)

# Prometheus metrics
request_count = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
request_duration = Histogram('http_request_duration_seconds', 'HTTP request duration', ['method', 'endpoint'])
active_requests = Gauge('http_requests_active', 'Active HTTP requests')
circuit_breaker_failures = Counter('circuit_breaker_failures_total', 'Circuit breaker failures', ['service'])

# Rate limiter instance
limiter = Limiter(key_func=get_remote_address, default_limits=["200/minute", "2000/hour"])

# Cache instance
cache = Cache(Cache.MEMORY, serializer=JsonSerializer())

# Circuit breakers for external services
service_breakers = {
    "database": CircuitBreaker(
        fail_max=5,
        timeout_duration=60,
        expected_exception=Exception
    ),
    "redis": CircuitBreaker(
        fail_max=3,
        timeout_duration=30,
        expected_exception=Exception
    ),
    "ollama": CircuitBreaker(
        fail_max=10,
        timeout_duration=120,
        expected_exception=Exception
    ),
}


class RequestIDMiddleware(BaseHTTPMiddleware):
    """Add unique request ID to all requests for tracing"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        
        return response


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Track request performance and add timing headers"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()
        active_requests.inc()
        
        try:
            response = await call_next(request)
            
            # Add timing headers
            process_time = time.time() - start_time
            response.headers["X-Process-Time"] = f"{process_time:.4f}"
            
            # Record metrics
            request_count.labels(
                method=request.method,
                endpoint=request.url.path,
                status=response.status_code
            ).inc()
            
            request_duration.labels(
                method=request.method,
                endpoint=request.url.path
            ).observe(process_time)
            
            return response
        
        finally:
            active_requests.dec()


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add security headers to all responses"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=()"
        
        # Remove server header for security
        if "server" in response.headers:
            del response.headers["server"]
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Centralized error handling with proper logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        
        except CircuitBreakerError as e:
            logger.error(f"Circuit breaker open: {str(e)}")
            circuit_breaker_failures.labels(service="unknown").inc()
            
            return JSONResponse(
                status_code=503,
                content={
                    "detail": "Service temporarily unavailable. Please try again later.",
                    "error_type": "circuit_breaker_open",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        
        except RateLimitExceeded:
            logger.warning(f"Rate limit exceeded for {get_remote_address(request)}")
            
            return JSONResponse(
                status_code=429,
                content={
                    "detail": "Rate limit exceeded. Please slow down your requests.",
                    "error_type": "rate_limit_exceeded",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )
        
        except HTTPException as e:
            # Let FastAPI handle HTTP exceptions normally
            raise
        
        except Exception as e:
            logger.exception(f"Unhandled exception: {str(e)}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "detail": "Internal server error. Please contact support.",
                    "error_type": "internal_error",
                    "request_id": getattr(request.state, "request_id", None)
                }
            )


def get_circuit_breaker(service_name: str) -> CircuitBreaker:
    """Get circuit breaker for a specific service"""
    return service_breakers.get(service_name, service_breakers["database"])


async def call_with_circuit_breaker(service_name: str, func: Callable, *args, **kwargs):
    """Execute function with circuit breaker protection"""
    breaker = get_circuit_breaker(service_name)
    
    try:
        return await breaker.call_async(func, *args, **kwargs)
    except CircuitBreakerError as e:
        circuit_breaker_failures.labels(service=service_name).inc()
        logger.error(f"Circuit breaker open for {service_name}: {str(e)}")
        raise HTTPException(
            status_code=503,
            detail=f"Service {service_name} is temporarily unavailable"
        )


def setup_enterprise_middleware(app):
    """Setup all enterprise middleware"""
    
    # Add middleware in order (last added = first executed)
    app.add_middleware(ErrorHandlingMiddleware)
    app.add_middleware(SecurityHeadersMiddleware)
    app.add_middleware(PerformanceMiddleware)
    app.add_middleware(RequestIDMiddleware)
    
    # Add rate limiter
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
    
    logger.info("Enterprise middleware configured successfully")
    
    return limiter
