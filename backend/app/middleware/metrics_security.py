"""
Metrics Security Middleware
Protects metrics endpoints with JWT authentication and role-based access control
"""

import logging
from typing import Optional, Callable
from fastapi import Request, Response, HTTPException, status
from fastapi.responses import PlainTextResponse, JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger(__name__)


class MetricsAuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to protect metrics endpoints with JWT authentication
    Requires admin or monitoring role to access metrics
    """
    
    def __init__(self, app, protected_paths: list = None):
        super().__init__(app)
        self.protected_paths = protected_paths or [
            "/metrics",
            "/api/v1/metrics",
            "/api/v1/health/detailed",
            "/api/v1/health/circuit-breakers",
            "/api/v1/cache/stats"
        ]
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check authentication for protected metrics endpoints"""
        
        # Check if path needs protection
        path = request.url.path
        if not self._needs_protection(path):
            # Not a protected endpoint, continue normally
            return await call_next(request)
        
        # Extract and verify authentication
        try:
            # Get authorization header
            auth_header = request.headers.get("Authorization")
            if not auth_header or not auth_header.startswith("Bearer "):
                logger.warning(f"Unauthorized metrics access attempt to {path} - missing auth header")
                return self._create_unauthorized_response(path)
            
            # Extract token
            token = auth_header.replace("Bearer ", "")
            
            # Verify token using existing JWT handler
            from app.auth.jwt_handler import verify_token
            
            try:
                # Verify token and get payload
                payload = verify_token(token, token_type="access")
                
                # Check for required permissions
                user_id = payload.get("sub")
                scopes = payload.get("scopes", [])
                
                # Require admin or monitoring scope for metrics
                if not self._has_metrics_permission(scopes):
                    logger.warning(f"Forbidden metrics access attempt to {path} by user {user_id} - insufficient permissions")
                    return self._create_forbidden_response(path)
                
                # Log successful access
                logger.info(f"Authorized metrics access to {path} by user {user_id}")
                
                # Add user info to request state for downstream use
                request.state.user_id = user_id
                request.state.user_scopes = scopes
                
            except ValueError as e:
                # Token verification failed
                logger.warning(f"Invalid token for metrics access to {path}: {e}")
                return self._create_unauthorized_response(path, str(e))
            
        except Exception as e:
            logger.error(f"Error in metrics authentication for {path}: {e}")
            return self._create_error_response()
        
        # Authentication successful, proceed with request
        response = await call_next(request)
        
        # Add security headers to metrics responses
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        
        return response
    
    def _needs_protection(self, path: str) -> bool:
        """Check if path requires authentication"""
        # Exact match or prefix match
        for protected_path in self.protected_paths:
            if path == protected_path or path.startswith(protected_path + "/"):
                return True
        return False
    
    def _has_metrics_permission(self, scopes: list) -> bool:
        """Check if user has permission to access metrics"""
        # Require admin, monitoring, or metrics scope
        required_scopes = {"admin", "monitoring", "metrics", "system:read"}
        return bool(required_scopes.intersection(scopes))
    
    def _create_unauthorized_response(self, path: str, detail: str = "Authentication required") -> Response:
        """Create 401 Unauthorized response"""
        
        # Different response format for Prometheus endpoint
        if path == "/metrics":
            return PlainTextResponse(
                content=f"# ERROR: {detail}\n# Authentication required for metrics access",
                status_code=status.HTTP_401_UNAUTHORIZED,
                headers={
                    "WWW-Authenticate": "Bearer",
                    "X-Error": detail
                }
            )
        
        # JSON response for API endpoints
        return JSONResponse(
            content={
                "error": "Unauthorized",
                "detail": detail,
                "path": path
            },
            status_code=status.HTTP_401_UNAUTHORIZED,
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    def _create_forbidden_response(self, path: str) -> Response:
        """Create 403 Forbidden response"""
        
        # Different response format for Prometheus endpoint
        if path == "/metrics":
            return PlainTextResponse(
                content="# ERROR: Forbidden\n# Insufficient permissions for metrics access",
                status_code=status.HTTP_403_FORBIDDEN
            )
        
        # JSON response for API endpoints
        return JSONResponse(
            content={
                "error": "Forbidden",
                "detail": "Insufficient permissions to access metrics",
                "required_scopes": ["admin", "monitoring", "metrics"],
                "path": path
            },
            status_code=status.HTTP_403_FORBIDDEN
        )
    
    def _create_error_response(self) -> Response:
        """Create 500 Internal Server Error response"""
        return JSONResponse(
            content={
                "error": "Internal Server Error",
                "detail": "Error processing authentication"
            },
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR
        )


class APIKeyAuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Alternative authentication using API keys for monitoring systems
    Useful for Prometheus scrapers and monitoring tools
    """
    
    def __init__(self, app, api_keys: dict = None):
        super().__init__(app)
        self.api_keys = api_keys or {}
        self.protected_paths = ["/metrics", "/api/v1/metrics"]
        
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """Check API key authentication for metrics endpoints"""
        
        path = request.url.path
        if path not in self.protected_paths:
            return await call_next(request)
        
        # Check for API key in header or query parameter
        api_key = request.headers.get("X-API-Key") or request.query_params.get("api_key")
        
        if not api_key:
            # Fall back to JWT authentication
            return await call_next(request)
        
        # Validate API key
        if api_key not in self.api_keys:
            logger.warning(f"Invalid API key attempt for {path}")
            return JSONResponse(
                content={"error": "Invalid API key"},
                status_code=status.HTTP_401_UNAUTHORIZED
            )
        
        # Add key info to request
        key_info = self.api_keys[api_key]
        request.state.api_key_name = key_info.get("name", "unknown")
        request.state.api_key_scopes = key_info.get("scopes", [])
        
        logger.info(f"API key authentication successful for {path} using key: {request.state.api_key_name}")
        
        return await call_next(request)


def setup_metrics_security(app, environment: str = "production"):
    """
    Setup metrics security middleware
    
    Args:
        app: FastAPI application instance
        environment: Current environment
    """
    import os
    import json
    
    # Add metrics authentication middleware
    app.add_middleware(MetricsAuthenticationMiddleware)
    
    # Optionally add API key authentication for monitoring tools
    api_keys_config = os.getenv("METRICS_API_KEYS")
    if api_keys_config:
        try:
            api_keys = json.loads(api_keys_config)
            app.add_middleware(APIKeyAuthenticationMiddleware, api_keys=api_keys)
            logger.info(f"API key authentication enabled for metrics with {len(api_keys)} keys")
        except json.JSONDecodeError:
            logger.error("Failed to parse METRICS_API_KEYS configuration")
    
    logger.info("Metrics security middleware configured successfully")