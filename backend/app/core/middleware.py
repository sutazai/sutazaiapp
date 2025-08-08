"""
Performance and Security Middleware
"""

import time
import uuid
from typing import Callable
from fastapi import Request, Response, status, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.gzip import GZipMiddleware
import logging
from datetime import datetime
import json

from app.core.performance import performance_optimizer, rate_limiter
from app.core.security import xss_protection, decode_jwt, AuthError

logger = logging.getLogger(__name__)

security = HTTPBearer(auto_error=False)

def jwt_required(scopes=None):
    scopes = scopes or []
    def _dep(credentials: HTTPAuthorizationCredentials = Depends(security)):
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
        try:
            return decode_jwt(credentials.credentials, required_scopes=scopes)
        except AuthError as e:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=str(e))
    return _dep


class PerformanceMiddleware(BaseHTTPMiddleware):
    """Track request performance and add optimizations"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Start timing
        start_time = time.time()
        
        # Add request ID to headers
        response = await call_next(request)
        
        # Calculate duration
        duration = time.time() - start_time
        
        # Add performance headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Response-Time"] = f"{duration:.3f}"
        response.headers["X-Server-Time"] = datetime.utcnow().isoformat()
        
        # Log slow requests
        if duration > 1.0:
            logger.warning(
                f"Slow request: {request.method} {request.url.path} "
                f"took {duration:.2f}s (ID: {request_id})"
            )
        
        # Track timing
        endpoint = f"{request.method}:{request.url.path}"
        performance_optimizer._request_timings[endpoint].append(duration)
        
        return response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Rate limiting middleware"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier (IP or API key)
        client_id = request.client.host if request.client else "unknown"
        
        # Check rate limit
        if not await rate_limiter.acquire():
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={"error": "Rate limit exceeded"},
                headers={"Retry-After": str(int(rate_limiter.get_wait_time()))}
            )
        
        response = await call_next(request)
        return response


class CacheMiddleware(BaseHTTPMiddleware):
    """Cache middleware for GET requests"""
    
    CACHEABLE_PATHS = [
        "/api/v1/models/list",
        "/api/v1/agents/list",
        "/api/v1/system/status"
    ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Only cache GET requests on specific paths
        if request.method != "GET" or request.url.path not in self.CACHEABLE_PATHS:
            return await call_next(request)
        
        # Generate cache key
        cache_key = f"response:{request.url.path}:{request.url.query}"
        
        # Check cache
        cached = await performance_optimizer.cache_manager.get(cache_key)
        if cached:
            return Response(
                content=cached["content"],
                status_code=cached["status_code"],
                headers=cached["headers"],
                media_type="application/json"
            )
        
        # Get response
        response = await call_next(request)
        
        # Cache successful responses
        if response.status_code == 200:
            # Read response body
            body = b""
            async for chunk in response.body_iterator:
                body += chunk
            
            # Cache response
            await performance_optimizer.cache_manager.set(
                cache_key,
                {
                    "content": body,
                    "status_code": response.status_code,
                    "headers": dict(response.headers)
                },
                ttl=300  # 5 minutes
            )
            
            # Return new response with body
            return Response(
                content=body,
                status_code=response.status_code,
                headers=dict(response.headers),
                media_type="application/json"
            )
        
        return response


class CompressionMiddleware(GZipMiddleware):
    """Enhanced compression middleware"""
    
    def __init__(self, app, minimum_size: int = 500, compresslevel: int = 6):
        super().__init__(app, minimum_size=minimum_size, compresslevel=compresslevel)


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add comprehensive security headers to prevent XSS and other attacks"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)
        
        # Add comprehensive security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        response.headers["Permissions-Policy"] = "geolocation=(), microphone=(), camera=(), fullscreen=(), payment=()"
        
        # Enhanced CSP for comprehensive XSS protection
        csp_directives = [
            "default-src 'self'",
            "script-src 'self' 'strict-dynamic'",
            "style-src 'self' 'unsafe-inline'",
            "img-src 'self' data: https:",
            "font-src 'self'",
            "connect-src 'self'",
            "media-src 'self'",
            "object-src 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "frame-ancestors 'none'",
            "upgrade-insecure-requests",
            "block-all-mixed-content"
        ]
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # Additional security headers
        response.headers["Cross-Origin-Embedder-Policy"] = "require-corp"
        response.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        response.headers["Cross-Origin-Resource-Policy"] = "same-site"
        
        return response


class XSSProtectionMiddleware(BaseHTTPMiddleware):
    """Middleware for comprehensive XSS protection"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Sanitize request data
        if request.method in ["POST", "PUT", "PATCH"]:
            try:
                # Get request body if it exists
                body = await request.body()
                if body:
                    # Parse JSON body
                    try:
                        request_data = json.loads(body.decode())
                        
                        # Sanitize the request data
                        sanitized_data = await xss_protection.process_request({
                            "body": request_data,
                            "path": str(request.url.path),
                            "method": request.method
                        })
                        
                        # Create new request with sanitized data
                        sanitized_body = json.dumps(sanitized_data.get("body", {})).encode()
                        
                        # Replace request body
                        request._body = sanitized_body
                        
                    except json.JSONDecodeError:
                        # If not JSON, treat as text and sanitize
                        text_content = body.decode('utf-8', errors='ignore')
                        try:
                            sanitized_text = xss_protection.validator.validate_input(text_content, "text")
                            request._body = sanitized_text.encode()
                        except ValueError as e:
                            logger.warning(f"XSS protection blocked request: {e}")
                            return JSONResponse(
                                status_code=400,
                                content={"error": "Request blocked for security reasons", "details": str(e)}
                            )
                            
            except Exception as e:
                logger.error(f"XSS protection error: {e}")
                # Continue with original request if sanitization fails
                pass
        
        # Process the request
        response = await call_next(request)
        
        # Sanitize response if it's JSON
        if response.headers.get("content-type", "").startswith("application/json"):
            try:
                # Read response body
                body = b""
                async for chunk in response.body_iterator:
                    body += chunk
                
                if body:
                    # Parse and sanitize response data
                    response_data = json.loads(body.decode())
                    sanitized_response = await xss_protection.sanitize_response(response_data)
                    
                    # Create new response with sanitized data
                    sanitized_body = json.dumps(sanitized_response).encode()
                    
                    return Response(
                        content=sanitized_body,
                        status_code=response.status_code,
                        headers=dict(response.headers),
                        media_type="application/json"
                    )
            except Exception as e:
                logger.error(f"Response sanitization error: {e}")
                # Return original response if sanitization fails
                pass
        
        return response


class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Global error handling with logging"""
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            # Log error with request details
            logger.error(
                f"Unhandled error: {str(e)}\n"
                f"Request: {request.method} {request.url}\n"
                f"Headers: {dict(request.headers)}\n",
                exc_info=True
            )
            
            # Return error response
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={
                    "error": "Internal server error",
                    "message": str(e) if logger.level <= logging.DEBUG else "An error occurred",
                    "request_id": getattr(request.state, "request_id", "unknown")
                }
            )


class MetricsMiddleware(BaseHTTPMiddleware):
    """Collect metrics for monitoring"""
    
    def __init__(self, app):
        super().__init__(app)
        self.request_count = 0
        self.error_count = 0
        self.response_times = []
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        self.request_count += 1
        start_time = time.time()
        
        response = await call_next(request)
        
        duration = time.time() - start_time
        self.response_times.append(duration)
        
        # Keep only last 1000 response times
        if len(self.response_times) > 1000:
            self.response_times = self.response_times[-1000:]
        
        # Count errors
        if response.status_code >= 400:
            self.error_count += 1
        
        # Add metrics to response headers (for debugging)
        if request.url.path == "/metrics":
            metrics = {
                "request_count": self.request_count,
                "error_count": self.error_count,
                "error_rate": self.error_count / self.request_count if self.request_count > 0 else 0,
                "avg_response_time": sum(self.response_times) / len(self.response_times) if self.response_times else 0,
                "p95_response_time": sorted(self.response_times)[int(len(self.response_times) * 0.95)] if self.response_times else 0
            }
            return JSONResponse(content=metrics)
        
        return response