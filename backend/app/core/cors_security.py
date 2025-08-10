"""
CORS Security Configuration
Centralized CORS configuration with explicit origin whitelisting for security
"""
import os
from typing import List
from pydantic import BaseModel


class CORSSecurityConfig(BaseModel):
    """Secure CORS configuration with explicit origin whitelisting"""
    
    # Core application origins
    CORE_ORIGINS: List[str] = [
        "http://localhost:10011",   # Frontend Streamlit UI
        "http://localhost:10010",   # Backend (for Swagger UI)
        "http://127.0.0.1:10011",   # Alternative localhost frontend
        "http://127.0.0.1:10010",   # Alternative localhost backend
    ]
    
    # Development origins (only in development)
    DEV_ORIGINS: List[str] = [
        "http://localhost:3000",    # React dev server
        "http://localhost:8501",    # Alternative Streamlit port
        "http://127.0.0.1:3000",    # Alternative localhost
        "http://127.0.0.1:8501",    # Alternative localhost
    ]
    
    # Monitoring origins (for metrics and dashboards)
    MONITORING_ORIGINS: List[str] = [
        "http://localhost:10200",   # Prometheus
        "http://localhost:10201",   # Grafana
        "http://localhost:10202",   # Loki
        "http://localhost:10203",   # AlertManager
        "http://127.0.0.1:10200",   # Alternative localhost
        "http://127.0.0.1:10201",   # Alternative localhost
        "http://127.0.0.1:10202",   # Alternative localhost
        "http://127.0.0.1:10203",   # Alternative localhost
    ]
    
    # Service mesh origins (for inter-service communication)
    SERVICE_ORIGINS: List[str] = [
        "http://localhost:8090",    # Ollama Integration
        "http://localhost:8589",    # AI Agent Orchestrator
        "http://localhost:11110",   # Hardware Resource Optimizer
        "http://127.0.0.1:8090",    # Alternative localhost
        "http://127.0.0.1:8589",    # Alternative localhost
        "http://127.0.0.1:11110",   # Alternative localhost
    ]

    def get_allowed_origins(self, include_monitoring: bool = True, include_services: bool = True) -> List[str]:
        """Get secure list of allowed origins based on environment"""
        allowed_origins = self.CORE_ORIGINS.copy()
        
        # Add development origins only in development environment
        env = os.getenv("SUTAZAI_ENV", "development").lower()
        if env in ["development", "dev", "local"]:
            allowed_origins.extend(self.DEV_ORIGINS)
        
        # Add monitoring origins for services that need to access metrics
        if include_monitoring:
            allowed_origins.extend(self.MONITORING_ORIGINS)
            
        # Add service mesh origins for inter-service communication
        if include_services:
            allowed_origins.extend(self.SERVICE_ORIGINS)
        
        # Add production origins from environment variables
        prod_origins = os.getenv("CORS_ALLOWED_ORIGINS", "").split(",")
        prod_origins = [origin.strip() for origin in prod_origins if origin.strip()]
        allowed_origins.extend(prod_origins)
        
        # Remove duplicates and return
        return list(set(allowed_origins))

    def get_cors_middleware_config(self, include_monitoring: bool = True, include_services: bool = True) -> dict:
        """Get complete CORS middleware configuration"""
        return {
            "allow_origins": self.get_allowed_origins(include_monitoring, include_services),
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
            "allow_headers": ["*"],  # Can be more restrictive if needed
            "expose_headers": ["*"],  # Can be more restrictive if needed
        }


# Global instance
cors_security = CORSSecurityConfig()


def get_secure_cors_config(service_type: str = "api") -> dict:
    """
    Get secure CORS configuration for different service types
    
    Args:
        service_type: Type of service ("api", "monitoring", "agent", "minimal")
    """
    if service_type == "monitoring":
        # Monitoring services need limited access
        return cors_security.get_cors_middleware_config(include_monitoring=False, include_services=False)
    elif service_type == "agent":
        # Agent services need service mesh access
        return cors_security.get_cors_middleware_config(include_monitoring=True, include_services=True)
    elif service_type == "minimal":
        # Minimal services only need core access
        return {
            "allow_origins": cors_security.CORE_ORIGINS,
            "allow_credentials": True,
            "allow_methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"],
            "expose_headers": ["Content-Type"],
        }
    else:
        # Full API access
        return cors_security.get_cors_middleware_config()


def validate_cors_security() -> bool:
    """Validate that no wildcard origins are configured"""
    all_origins = cors_security.get_allowed_origins()
    
    # Check for wildcard patterns
    wildcards = ["*", "*.localhost", "*.local", "*://localhost:*"]
    
    for origin in all_origins:
        if any(wildcard in origin for wildcard in wildcards):
            return False
            
    return True