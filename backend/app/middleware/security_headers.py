"""
Security Headers Middleware for SutazAI
Implements comprehensive security headers following OWASP best practices
"""

from fastapi import Request
from fastapi.responses import Response
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Callable
import hashlib
import secrets
from datetime import datetime

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add comprehensive security headers to all responses
    Following OWASP Security Headers Project recommendations
    """
    
    def __init__(self, app, environment: str = "production"):
        super().__init__(app)
        self.environment = environment
        self.nonce_length = 32
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Generate CSP nonce for this request
        nonce = secrets.token_urlsafe(self.nonce_length)
        request.state.csp_nonce = nonce
        
        # Process the request
        response = await call_next(request)
        
        # Add security headers to response
        self._add_security_headers(response, nonce, request)
        
        return response
    
    def _add_security_headers(self, response: Response, nonce: str, request: Request):
        """Add comprehensive security headers to response"""
        
        # 1. X-Frame-Options - Prevent clickjacking attacks
        response.headers["X-Frame-Options"] = "DENY"
        
        # 2. X-Content-Type-Options - Prevent MIME sniffing
        response.headers["X-Content-Type-Options"] = "nosniff"
        
        # 3. X-XSS-Protection - Enable XSS filter (legacy browsers)
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # 4. Referrer-Policy - Control referrer information
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # 5. Permissions-Policy (formerly Feature-Policy)
        permissions = [
            "accelerometer=()",
            "camera=()",
            "geolocation=()",
            "gyroscope=()",
            "magnetometer=()",
            "microphone=()",
            "payment=()",
            "usb=()"
        ]
        response.headers["Permissions-Policy"] = ", ".join(permissions)
        
        # 6. Content-Security-Policy - Comprehensive CSP
        csp_directives = self._get_csp_directives(nonce)
        response.headers["Content-Security-Policy"] = "; ".join(csp_directives)
        
        # 7. Strict-Transport-Security - Force HTTPS (only in production)
        if self.environment == "production":
            # max-age=31536000 (1 year), includeSubDomains, preload
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # 8. X-Permitted-Cross-Domain-Policies - Control Adobe products
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        
        # 9. Cache-Control for sensitive data
        request_path = str(request.url.path)
        if "/api/" in request_path or "/auth/" in request_path:
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
            response.headers["Pragma"] = "no-cache"
            response.headers["Expires"] = "0"
        
        # 10. X-Download-Options - Prevent IE from executing downloads
        response.headers["X-Download-Options"] = "noopen"
        
    def _get_csp_directives(self, nonce: str) -> list:
        """Build Content Security Policy directives"""
        
        # Define allowed sources
        self_sources = ["'self'"]
        
        # In development, be slightly more permissive
        if self.environment == "development":
            script_sources = self_sources + [f"'nonce-{nonce}'", "'unsafe-inline'", "http://localhost:*"]
            style_sources = self_sources + ["'unsafe-inline'", "http://localhost:*"]
            img_sources = self_sources + ["data:", "http://localhost:*", "https:"]
            connect_sources = self_sources + ["http://localhost:*", "ws://localhost:*"]
        else:
            # Production: Strict CSP
            script_sources = self_sources + [f"'nonce-{nonce}'"]
            style_sources = self_sources + [f"'nonce-{nonce}'"]
            img_sources = self_sources + ["data:", "https:"]
            connect_sources = self_sources + ["https:"]
        
        directives = [
            f"default-src {' '.join(self_sources)}",
            f"script-src {' '.join(script_sources)}",
            f"style-src {' '.join(style_sources)}",
            f"img-src {' '.join(img_sources)}",
            f"font-src {' '.join(self_sources)}",
            f"connect-src {' '.join(connect_sources)}",
            "object-src 'none'",
            "frame-ancestors 'none'",
            "base-uri 'self'",
            "form-action 'self'",
            "upgrade-insecure-requests"
        ]
        
        # Add report-uri if configured
        if self.environment == "production":
            directives.append("report-uri /api/csp-report")
        
        return directives


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Rate limiting middleware to prevent abuse
    """
    
    def __init__(self, app, requests_per_minute: int = 60):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.request_counts = {}
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier (IP address)
        client_ip = request.client.host
        
        # Check rate limit
        current_minute = datetime.now().strftime("%Y%m%d%H%M")
        key = f"{client_ip}:{current_minute}"
        
        if key not in self.request_counts:
            self.request_counts[key] = 0
            
        self.request_counts[key] += 1
        
        if self.request_counts[key] > self.requests_per_minute:
            # Rate limit exceeded
            response = Response(
                content="Rate limit exceeded",
                status_code=429,
                headers={
                    "Retry-After": "60",
                    "X-RateLimit-Limit": str(self.requests_per_minute),
                    "X-RateLimit-Remaining": "0",
                    "X-RateLimit-Reset": str(int(datetime.now().timestamp()) + 60)
                }
            )
            return response
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        response.headers["X-RateLimit-Limit"] = str(self.requests_per_minute)
        response.headers["X-RateLimit-Remaining"] = str(
            self.requests_per_minute - self.request_counts[key]
        )
        response.headers["X-RateLimit-Reset"] = str(int(datetime.now().timestamp()) + 60)
        
        # Clean up old entries (simple cleanup, could be improved)
        if len(self.request_counts) > 10000:
            # Keep only recent entries
            cutoff = datetime.now().strftime("%Y%m%d%H%M")
            self.request_counts = {
                k: v for k, v in self.request_counts.items() 
                if k.split(":")[1] >= cutoff
            }
        
        return response


def setup_security_middleware(app, environment: str = "production"):
    """
    Setup all security middleware for the application
    
    Args:
        app: FastAPI application instance
        environment: Current environment (development/production)
    """
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
    
    # Add security headers middleware
    app.add_middleware(SecurityHeadersMiddleware, environment=environment)
    
    # Add rate limiting
    app.add_middleware(RateLimitMiddleware, requests_per_minute=60)
    
    # Add trusted host validation (prevent host header injection)
    if environment == "production":
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=["sutazai.com", "*.sutazai.com", "localhost"]
        )
        
        # Force HTTPS in production
        app.add_middleware(HTTPSRedirectMiddleware)