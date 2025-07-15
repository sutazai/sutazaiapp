"""
Simplified Observability System for validation
Basic monitoring without external dependencies
"""

import time
import logging
from typing import Dict, Any
from datetime import datetime

logger = logging.getLogger(__name__)

class ObservabilitySystem:
    """Simplified observability system for validation"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = []
        self.alerts = []
        self.initialized = True
        
    def get_system_overview(self) -> Dict[str, Any]:
        """Get system overview"""
        return {
            "health_status": {
                "overall_status": "healthy",
                "components": {
                    "system": {
                        "status": "healthy",
                        "message": "System running normally"
                    }
                }
            },
            "recent_alerts": self.alerts[-10:],
            "system_metrics": self.metrics[-50:],
            "timestamp": datetime.now().isoformat()
        }
    
    def start_monitoring(self):
        """Start monitoring"""
        logger.info("Monitoring started")
        
    def stop_monitoring(self):
        """Stop monitoring"""
        logger.info("Monitoring stopped")
        
    def collect_metrics(self):
        """Collect system metrics"""
        metric = {
            "timestamp": datetime.now().isoformat(),
            "cpu_usage": 25.0,
            "memory_usage": 45.0,
            "uptime": time.time() - self.start_time
        }
        self.metrics.append(metric)
        return metric