"""
Base Service Class for SutazAI
Provides common functionality for all services
"""

import logging
from abc import ABC
from typing import Dict, Any, Optional
from datetime import datetime


class BaseService(ABC):
    """
    Abstract base class for all SutazAI services
    Provides common functionality like logging, health checks, and lifecycle management
    """
    
    def __init__(self, service_name: Optional[str] = None):
        """
        Initialize the base service
        
        Args:
            service_name: Name of the service (defaults to class name)
        """
        self.service_name = service_name or self.__class__.__name__
        self.logger = logging.getLogger(f"sutazai.{self.service_name}")
        self.started_at = datetime.now()
        self.is_healthy = True
        self._metrics: Dict[str, Any] = {}
        
        self.logger.info(f"{self.service_name} service initialized")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get health status of the service
        
        Returns:
            Dictionary containing health information
        """
        return {
            "service_name": self.service_name,
            "healthy": self.is_healthy,
            "started_at": self.started_at.isoformat(),
            "uptime_seconds": (datetime.now() - self.started_at).total_seconds(),
            "metrics": self._metrics.copy()
        }
    
    def mark_unhealthy(self, reason: str):
        """
        Mark the service as unhealthy
        
        Args:
            reason: Reason for unhealthy status
        """
        self.is_healthy = False
        self.logger.warning(f"{self.service_name} marked as unhealthy: {reason}")
    
    def mark_healthy(self):
        """Mark the service as healthy"""
        self.is_healthy = True
        self.logger.info(f"{self.service_name} marked as healthy")
    
    def update_metric(self, key: str, value: Any):
        """
        Update a service metric
        
        Args:
            key: Metric name
            value: Metric value
        """
        self._metrics[key] = value
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all service metrics"""
        return self._metrics.copy()
    
    async def start(self):
        """Start the service (override in subclasses)"""
        self.logger.info(f"Starting {self.service_name} service")
    
    async def stop(self):
        """Stop the service (override in subclasses)"""
        self.logger.info(f"Stopping {self.service_name} service")
    
    async def restart(self):
        """Restart the service"""
        self.logger.info(f"Restarting {self.service_name} service")
        await self.stop()
        await self.start()
    
    def __str__(self) -> str:
        """String representation of the service"""
        return f"{self.service_name}(healthy={self.is_healthy})"
    
    def __repr__(self) -> str:
        """Detailed representation of the service"""
        return (f"{self.__class__.__name__}("
                f"service_name='{self.service_name}', "
                f"healthy={self.is_healthy}, "
                f"started_at='{self.started_at.isoformat()}')") 