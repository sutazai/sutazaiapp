"""
SutazAI Web Framework Configuration Guide

Provides a comprehensive overview and best practices for 
configuring FastAPI and Uvicorn in the SutazAI backend.

Key Considerations:
- Performance optimization
- Security hardening
- Scalability strategies
"""

from typing import Any, Dict, Optional

from pydantic import BaseModel, Field


class WebServerConfig(BaseModel):
    """
    Comprehensive web server configuration model.
    
    Defines best practices and recommended settings for 
    FastAPI and Uvicorn deployment.
    """
    
    # Server Basics
    host: str = Field(default="0.0.0.0", description="Binding host address")
    port: int = Field(default=8000, ge=1024, le=65535, description="Port number")
    
    # Performance Tuning
    workers: Optional[int] = Field(
        default=None, 
        description="Number of worker processes"
    )
    
    # Security Settings
    cors_origins: Optional[list] = Field(
        default_factory=list, 
        description="Allowed CORS origins"
    )
    
    # Logging and Monitoring
    log_level: str = Field(
        default="info", 
        pattern="^(debug|info|warning|error|critical)$",
        description="Logging verbosity level"
    )
    
    # Advanced Configuration
    reload: bool = Field(
        default=False, 
        description="Enable auto-reload for development"
    )
    
    def get_uvicorn_config(self) -> Dict[str, Any]:
        """
        Generate Uvicorn configuration dictionary.
        
        Returns:
            Dict with Uvicorn server configuration
        """
        return {
            "host": self.host,
            "port": self.port,
            "workers": self.workers,
            "log_level": self.log_level,
            "reload": self.reload
        }
    
    def validate_config(self) -> bool:
        """
        Perform comprehensive configuration validation.
        
        Returns:
            Boolean indicating configuration validity
        """
        # Add advanced validation logic
        if self.port < 1024 and not self.host.startswith("127."):
            return False
        
        return True

def create_recommended_config() -> WebServerConfig:
    """
    Generate a recommended web server configuration.
    
    Returns:
        WebServerConfig with optimized default settings
    """
    return WebServerConfig(
        host="0.0.0.0",
        port=8000,
        workers=4,  # Adjust based on CPU cores
        cors_origins=["https://sutazai.local", "http://localhost:3000"],
        log_level="info",
        reload=False
    )

def main():
    """
    Demonstrate web framework configuration usage.
    """
    # Create recommended configuration
    config = create_recommended_config()
    
    # Validate configuration
    if config.validate_config():
        print("✅ Web Server Configuration Validated")
        
        # Display Uvicorn configuration
        uvicorn_config = config.get_uvicorn_config()
        print("Uvicorn Configuration:")
        for key, value in uvicorn_config.items():
            print(f"{key}: {value}")
    else:
        print("❌ Configuration Validation Failed")

if __name__ == "__main__":
    main() 