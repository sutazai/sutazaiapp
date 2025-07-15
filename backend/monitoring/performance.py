"""Performance Monitor for SutazAI"""
import time
import psutil
import logging
from typing import Dict, Any
from collections import deque

logger = logging.getLogger(__name__)

class PerformanceMonitor:
    def __init__(self):
        self.metrics = deque(maxlen=1000)
        self.monitoring_active = False
    
    def record_metric(self, name: str, value: float, unit: str):
        """Record a performance metric"""
        metric = {
            "timestamp": time.time(),
            "name": name,
            "value": value,
            "unit": unit
        }
        self.metrics.append(metric)
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        return {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "disk_percent": psutil.disk_usage('/').percent
        }
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary"""
        return {
            "total_metrics": len(self.metrics),
            "system_metrics": self.get_system_metrics()
        }

# Global instance
performance_monitor = PerformanceMonitor()
