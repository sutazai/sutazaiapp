"""
CORS Security Configuration for SutazAI Backend
Provides secure CORS configuration with explicit origin whitelisting
"""

import os
import logging
from typing import Dict, List, Any

logger = logging.getLogger(__name__)


class CORSSecurityManager:
    """CORS Security configuration manager"""
    
    def __init__(self):
        self.allowed_origins = self._get_allowed_origins()
        
    def _get_allowed_origins(self) -> List[str]:
        """Get allowed origins from environment or use secure defaults"""
        # Get from environment variable
        origins_env = os.getenv("SUTAZAI_ALLOWED_ORIGINS", "")
        
        if origins_env:
            # Parse comma-separated origins
            origins = [origin.strip() for origin in origins_env.split(",")]
            # Filter out wildcards for security
            origins = [origin for origin in origins if "*" not in origin]
        else:
            # Secure defaults for development/production
            origins = [
                "http://localhost:10011",  # Frontend Streamlit
                "http://127.0.0.1:10011",
                "http://localhost:3000",   # Development frontend
                "http://127.0.0.1:3000",
                "http://localhost:8080",   # Alternative dev port
                "http://127.0.0.1:8080"
            ]
        
        logger.info(f"CORS allowed origins: {origins}")
        return origins
    
    def get_allowed_origins(self) -> List[str]:
        """Get list of allowed origins"""
        return self.allowed_origins
    
    def get_cors_middleware_config(self) -> Dict[str, Any]:
        """Get CORS middleware configuration"""
        return {
            "allow_origins": self.allowed_origins,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": [
                "Accept",
                "Accept-Language", 
                "Content-Language",
                "Content-Type",
                "Authorization",
                "X-Requested-With",
                "X-CSRFToken"
            ],
            "expose_headers": ["X-Total-Count", "X-Page-Count"]
        }


# Global CORS security manager instance
_cors_manager = None


def get_cors_manager() -> CORSSecurityManager:
    """Get or create CORS security manager singleton"""
    global _cors_manager
    
    if _cors_manager is None:
        _cors_manager = CORSSecurityManager()
    
    return _cors_manager


def get_secure_cors_config(context: str = "api") -> Dict[str, Any]:
    """Get secure CORS configuration for specific context"""
    manager = get_cors_manager()
    config = manager.get_cors_middleware_config()
    
    # Add context-specific configuration
    if context == "api":
        # API endpoints may need additional headers
        config["allow_headers"].extend([
            "X-API-Key",
            "X-Request-ID",
            "X-Client-Version"
        ])
    elif context == "websocket":
        # WebSocket specific configuration
        config["allow_credentials"] = True
    
    return config


def validate_cors_security() -> bool:
    """Validate CORS security configuration"""
    manager = get_cors_manager()
    allowed_origins = manager.get_allowed_origins()
    
    # Check for wildcard origins (security risk)
    for origin in allowed_origins:
        if "*" in origin:
            logger.critical(f"SECURITY RISK: Wildcard origin detected: {origin}")
            return False
    
    # Ensure we have at least one allowed origin
    if not allowed_origins:
        logger.critical("SECURITY RISK: No allowed origins configured")
        return False
    
    # Check for overly permissive origins in production
    env = os.getenv("SUTAZAI_ENV", "production")
    if env == "production":
        risky_origins = [
            "http://localhost",
            "http://127.0.0.1",
            "http://0.0.0.0"
        ]
        
        for origin in allowed_origins:
            # Check for localhost without specific port in production
            if any(origin.startswith(risky) and origin.count(":") == 1 for risky in risky_origins):
                logger.warning(f"PRODUCTION WARNING: Localhost origin without port: {origin}")
    
    logger.info("CORS security validation passed")
    return True


# Legacy compatibility - create a CORS security object
class _CORSSecurityCompat:
    """Compatibility wrapper for legacy CORS usage"""
    
    def get_cors_middleware_config(self) -> Dict[str, Any]:
        return get_secure_cors_config()
    
    def get_allowed_origins(self) -> List[str]:
        return get_cors_manager().get_allowed_origins()


# Create legacy compatibility instance
cors_security = _CORSSecurityCompat()