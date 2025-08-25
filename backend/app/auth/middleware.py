"""
Authentication Middleware for FastAPI
Provides security layer for API endpoints with rate limiting and audit logging
"""

import os
import time
import logging
from typing import Optional, Dict, Set, List
from datetime import datetime, timezone
from collections import defaultdict

from fastapi import Request, Response, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware

from app.auth.jwt_handler import verify_token
from app.auth.models import User
from app.core.database import get_db

logger = logging.getLogger(__name__)

# Rate limiting storage (in production, use Redis)
rate_limit_storage = defaultdict(list)

# Public endpoints that don't require authentication
PUBLIC_ENDPOINTS = {
    "/docs",
    "/redoc", 
    "/openapi.json",
    "/health",
    "/health-emergency",
    "/favicon.ico",
    "/api/v1/auth/login",
    "/api/v1/auth/register",
    "/api/v1/auth/refresh",
    "/api/v1/auth/forgot-password",
    "/api/v1/system/status",
}

# Admin-only endpoints
ADMIN_ENDPOINTS = {
    "/api/v1/system/restart",
    "/api/v1/system/shutdown", 
    "/api/v1/admin/",
    "/api/v1/users/admin/",
}

# Rate limiting configuration
RATE_LIMITS = {
    "default": {"requests": 100, "window": 60},  # 100 req/min default
    "auth": {"requests": 10, "window": 60},      # 10 req/min for auth endpoints
    "admin": {"requests": 50, "window": 60},     # 50 req/min for admin endpoints
}


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Security middleware that handles:
    1. Authentication verification
    2. Rate limiting
    3. Audit logging
    4. Security headers
    """
    
    def __init__(self, app, enable_rate_limiting: bool = True):
        super().__init__(app)
        self.enable_rate_limiting = enable_rate_limiting
        self.security_scheme = HTTPBearer(auto_error=False)
    
    async def dispatch(self, request: Request, call_next):
        """Process request through security layers"""
        start_time = time.time()
        
        try:
            # Add security headers to all responses
            response = await self._add_security_headers(request, call_next)
            
            # Log request for audit trail
            await self._audit_log(request, response, start_time)
            
            return response
            
        except HTTPException as e:
            # Log security violations
            logger.warning(
                f"Security violation: {e.detail} - "
                f"IP: {self._get_client_ip(request)} - "
                f"Path: {request.url.path}"
            )
            raise
        except Exception as e:
            logger.error(f"Security middleware error: {e}")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal security error"
            )
    
    async def _add_security_headers(self, request: Request, call_next):
        """Add security headers and process authentication"""
        
        # Check if endpoint requires authentication
        if not self._is_public_endpoint(request.url.path):
            
            # Rate limiting check
            if self.enable_rate_limiting:
                await self._check_rate_limit(request)
            
            # Authentication check
            user = await self._authenticate_request(request)
            
            # Authorization check for admin endpoints
            if self._is_admin_endpoint(request.url.path):
                if not user or not user.is_admin:
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail="Admin access required"
                    )
            
            # Add user to request state for use in endpoints
            request.state.user = user
        
        # Process the request
        response = await call_next(request)
        
        # Add security headers to response
        security_headers = {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY", 
            "X-XSS-Protection": "1; mode=block",
            "Strict-Transport-Security": "max-age=31536000; includeSubDomains",
            "Content-Security-Policy": "default-src 'self'",
            "Referrer-Policy": "strict-origin-when-cross-origin",
            "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        }
        
        for header, value in security_headers.items():
            response.headers[header] = value
            
        return response
    
    def _is_public_endpoint(self, path: str) -> bool:
        """Check if endpoint is public (doesn't require auth)"""
        # Exact match check
        if path in PUBLIC_ENDPOINTS:
            return True
        
        # Pattern matching for dynamic routes
        public_patterns = ["/docs", "/redoc", "/openapi", "/health"]
        return any(path.startswith(pattern) for pattern in public_patterns)
    
    def _is_admin_endpoint(self, path: str) -> bool:
        """Check if endpoint requires admin privileges"""
        return any(path.startswith(pattern) for pattern in ADMIN_ENDPOINTS)
    
    async def _authenticate_request(self, request: Request) -> Optional[User]:
        """Authenticate the request and return user"""
        
        # Extract Authorization header
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Missing or invalid authorization header",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Extract token
        token = auth_header.split(" ")[1] if " " in auth_header else None
        if not token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token format",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        try:
            # Verify JWT token
            payload = verify_token(token, token_type="access")
            user_id = int(payload.get("sub"))
            
            if user_id is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid token payload",
                    headers={"WWW-Authenticate": "Bearer"},
                )
            
            # Get user from database (simplified for middleware)
            # In production, this should use a cached user lookup
            # For now, create a basic user object from token payload
            user = User(
                id=user_id,
                username=payload.get("username", "unknown"),
                email=payload.get("email", ""),
                is_active=payload.get("is_active", True),
                is_admin=payload.get("is_admin", False),
            )
            
            return user
            
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail=f"Token verification failed: {str(e)}",
                headers={"WWW-Authenticate": "Bearer"},
            )
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed",
                headers={"WWW-Authenticate": "Bearer"},
            )
    
    async def _check_rate_limit(self, request: Request):
        """Check rate limiting for the request"""
        client_ip = self._get_client_ip(request)
        path = request.url.path
        
        # Determine rate limit category
        if path.startswith("/api/v1/auth/"):
            limit_config = RATE_LIMITS["auth"]
        elif self._is_admin_endpoint(path):
            limit_config = RATE_LIMITS["admin"]
        else:
            limit_config = RATE_LIMITS["default"]
        
        # Check rate limit
        current_time = time.time()
        window_start = current_time - limit_config["window"]
        
        # Clean old requests
        rate_limit_storage[client_ip] = [
            req_time for req_time in rate_limit_storage[client_ip]
            if req_time > window_start
        ]
        
        # Check if limit exceeded
        if len(rate_limit_storage[client_ip]) >= limit_config["requests"]:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail=f"Rate limit exceeded: {limit_config['requests']} requests per {limit_config['window']} seconds",
                headers={
                    "X-RateLimit-Limit": str(limit_config["requests"]),
                    "X-RateLimit-Window": str(limit_config["window"]),
                    "X-RateLimit-Remaining": "0",
                    "Retry-After": str(int(limit_config["window"]))
                }
            )
        
        # Add current request
        rate_limit_storage[client_ip].append(current_time)
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP address handling proxies"""
        # Check for forwarded headers (nginx, cloudflare, etc.)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip
        
        # Fallback to direct connection
        return request.client.host if request.client else "unknown"
    
    async def _audit_log(self, request: Request, response: Response, start_time: float):
        """Log request for security audit trail"""
        
        # Calculate response time
        response_time = (time.time() - start_time) * 1000  # ms
        
        # Extract user info if available
        user_info = "anonymous"
        if hasattr(request.state, "user") and request.state.user:
            user_info = f"user:{request.state.user.id}:{request.state.user.username}"
        
        # Log security-relevant requests
        if (response.status_code >= 400 or 
            request.url.path.startswith("/api/v1/admin/") or
            request.method in ["POST", "PUT", "DELETE", "PATCH"]):
            
            logger.info(
                f"AUDIT: {request.method} {request.url.path} | "
                f"Status: {response.status_code} | "
                f"User: {user_info} | "
                f"IP: {self._get_client_ip(request)} | "
                f"Time: {response_time:.1f}ms | "
                f"UA: {request.headers.get('User-Agent', 'unknown')[:100]}"
            )


class APIKeyMiddleware(BaseHTTPMiddleware):
    """
    Alternative middleware for API key authentication
    Useful for service-to-service communication
    """
    
    def __init__(self, app, api_keys: Set[str] = None):
        super().__init__(app)
        self.valid_api_keys = api_keys or set()
        # Load API keys from environment
        env_keys = os.getenv("VALID_API_KEYS", "").split(",")
        self.valid_api_keys.update(key.strip() for key in env_keys if key.strip())
    
    async def dispatch(self, request: Request, call_next):
        """Check API key for service endpoints"""
        
        # Only apply to service endpoints
        if request.url.path.startswith("/api/v1/service/"):
            api_key = request.headers.get("X-API-Key")
            
            if not api_key or api_key not in self.valid_api_keys:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid or missing API key",
                    headers={"WWW-Authenticate": "ApiKey"},
                )
        
        response = await call_next(request)
        return response