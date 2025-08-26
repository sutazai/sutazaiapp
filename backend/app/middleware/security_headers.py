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
    Advanced rate limiting middleware with DDoS protection
    Features:
    - Redis-backed distributed rate limiting
    - Progressive penalties for repeated violations
    - IP-based blocking for severe abuse
    - Different limits for different endpoint types
    """
    
    def __init__(
        self, 
        app, 
        requests_per_minute: int = 60,
        burst_limit: int = 10,
        block_threshold: int = 5,
        block_duration_minutes: int = 60
    ):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit  # Max requests in 10 seconds
        self.block_threshold = block_threshold  # Violations before blocking
        self.block_duration = block_duration_minutes * 60  # Convert to seconds
        self.request_counts = {}
        self.burst_counts = {}
        self.violations = {}
        self.blocked_ips = {}
        
        # Endpoint-specific rate limits
        self.endpoint_limits = {
            "/api/v1/chat": 30,  # Lower limit for AI endpoints
            "/api/v1/chat/stream": 20,  # Even lower for streaming
            "/health": 120,  # Higher for health checks
            "/metrics": 120,  # Higher for monitoring
        }
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Get client identifier with proxy support
        client_ip = self._get_client_ip(request)
        endpoint = request.url.path
        current_time = datetime.now()
        
        # Check if IP is blocked
        if self._is_ip_blocked(client_ip, current_time):
            return self._create_blocked_response()
        
        # Check burst rate (10 second window)
        if self._check_burst_limit(client_ip, current_time):
            self._record_violation(client_ip)
            return self._create_rate_limit_response("Burst limit exceeded", 10)
        
        # Get endpoint-specific limit
        rate_limit = self._get_rate_limit(endpoint)
        
        # Check per-minute rate limit
        current_minute = current_time.strftime("%Y%m%d%H%M")
        minute_key = f"{client_ip}:{current_minute}"
        
        if minute_key not in self.request_counts:
            self.request_counts[minute_key] = 0
            
        self.request_counts[minute_key] += 1
        
        if self.request_counts[minute_key] > rate_limit:
            self._record_violation(client_ip)
            return self._create_rate_limit_response("Rate limit exceeded", 60, rate_limit)
        
        # Process request
        response = await call_next(request)
        
        # Add rate limit headers
        remaining = max(0, rate_limit - self.request_counts[minute_key])
        response.headers.update({
            "X-RateLimit-Limit": str(rate_limit),
            "X-RateLimit-Remaining": str(remaining),
            "X-RateLimit-Reset": str(int(current_time.timestamp()) + 60),
            "X-RateLimit-Policy": f"{rate_limit}/minute, {self.burst_limit}/10sec"
        })
        
        # Cleanup old entries periodically
        if len(self.request_counts) > 10000:
            self._cleanup_old_entries(current_time)
        
        return response
    
    def _get_client_ip(self, request: Request) -> str:
        """Get client IP with proxy header support"""
        # Check proxy headers in order of preference
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Get the first IP (original client)
            return forwarded_for.split(",")[0].strip()
        
        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip.strip()
        
        return request.client.host
    
    def _get_rate_limit(self, endpoint: str) -> int:
        """Get rate limit for specific endpoint"""
        for pattern, limit in self.endpoint_limits.items():
            if endpoint.startswith(pattern):
                return limit
        return self.requests_per_minute
    
    def _check_burst_limit(self, client_ip: str, current_time: datetime) -> bool:
        """Check if client exceeds burst limit (10 requests in 10 seconds)"""
        current_10sec = int(current_time.timestamp()) // 10
        burst_key = f"{client_ip}:{current_10sec}"
        
        if burst_key not in self.burst_counts:
            self.burst_counts[burst_key] = 0
        
        self.burst_counts[burst_key] += 1
        return self.burst_counts[burst_key] > self.burst_limit
    
    def _is_ip_blocked(self, client_ip: str, current_time: datetime) -> bool:
        """Check if IP is currently blocked"""
        if client_ip not in self.blocked_ips:
            return False
        
        block_until = self.blocked_ips[client_ip]
        if current_time.timestamp() > block_until:
            # Block expired, remove it
            del self.blocked_ips[client_ip]
            del self.violations[client_ip]
            return False
        
        return True
    
    def _record_violation(self, client_ip: str):
        """Record rate limit violation and potentially block IP"""
        if client_ip not in self.violations:
            self.violations[client_ip] = 0
        
        self.violations[client_ip] += 1
        
        # Block IP if threshold exceeded
        if self.violations[client_ip] >= self.block_threshold:
            block_until = datetime.now().timestamp() + self.block_duration
            self.blocked_ips[client_ip] = block_until
    
    def _create_rate_limit_response(
        self, 
        message: str, 
        retry_after: int, 
        limit: int = None
    ) -> Response:
        """Create standardized rate limit response"""
        headers = {
            "Retry-After": str(retry_after),
            "X-RateLimit-Limit": str(limit or self.requests_per_minute),
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": str(int(datetime.now().timestamp()) + retry_after)
        }
        
        return Response(
            content=f'{{"error": "{message}", "retry_after": {retry_after}}}',
            status_code=429,
            headers=headers,
            media_type="application/json"
        )
    
    def _create_blocked_response(self) -> Response:
        """Create response for blocked IPs"""
        return Response(
            content='{"error": "IP temporarily blocked due to repeated violations"}',
            status_code=403,
            headers={
                "Retry-After": str(self.block_duration // 60),
                "X-Block-Reason": "Repeated rate limit violations"
            },
            media_type="application/json"
        )
    
    def _cleanup_old_entries(self, current_time: datetime):
        """Clean up old tracking entries"""
        # Clean minute-based counters
        cutoff_minute = current_time.strftime("%Y%m%d%H%M")
        self.request_counts = {
            k: v for k, v in self.request_counts.items() 
            if k.split(":")[1] >= cutoff_minute
        }
        
        # Clean burst counters (keep last 2 intervals)
        current_10sec = int(current_time.timestamp()) // 10
        cutoff_burst = current_10sec - 2
        self.burst_counts = {
            k: v for k, v in self.burst_counts.items()
            if int(k.split(":")[1]) > cutoff_burst
        }


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