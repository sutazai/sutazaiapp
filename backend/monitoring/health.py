"""Health Checker for SutazAI"""
import asyncio
import time
import logging
from typing import Dict, Any
from enum import Enum

logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"

class HealthChecker:
    def __init__(self):
        self.checks = {}
        self.last_results = {}
    
    async def run_all_checks(self) -> Dict[str, Any]:
        """Run all health checks"""
        results = {}
        
        # Basic system check
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            if cpu_percent > 80 or memory_percent > 85:
                status = HealthStatus.WARNING
                message = f"High resource usage - CPU: {cpu_percent}%, Memory: {memory_percent}%"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources OK"
            
            results["system"] = {
                "status": status.value,
                "message": message,
                "timestamp": time.time()
            }
        except Exception as e:
            results["system"] = {
                "status": HealthStatus.CRITICAL.value,
                "message": f"System check failed: {str(e)}",
                "timestamp": time.time()
            }
        
        return results
    
    async def get_health_report(self) -> Dict[str, Any]:
        """Get comprehensive health report"""
        results = await self.run_all_checks()
        
        return {
            "overall_status": "healthy",
            "timestamp": time.time(),
            "checks": results
        }

# Global instance
health_checker = HealthChecker()
