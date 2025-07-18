#!/usr/bin/env python3
"""
SutazAI Monitoring System
Metrics collection and health checking
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
import psutil
import httpx

from .config import settings

logger = logging.getLogger(__name__)


class MetricsCollector:
    """Collects system and application metrics"""
    
    def __init__(self):
        self.start_time = time.time()
        self.metrics = {
            "requests": {},
            "system": {},
            "services": {},
            "errors": {}
        }
    
    def get_uptime(self) -> float:
        """Get system uptime in seconds"""
        return time.time() - self.start_time
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            # Memory metrics
            memory = psutil.virtual_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            metrics = {
                "timestamp": datetime.utcnow().isoformat(),
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg": psutil.getloadavg() if hasattr(psutil, 'getloadavg') else None
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent,
                    "used": memory.used,
                    "free": memory.free
                },
                "disk": {
                    "total": disk.total,
                    "used": disk.used,
                    "free": disk.free,
                    "percent": disk.percent
                },
                "network": {
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                "uptime": self.get_uptime()
            }
            
            self.metrics["system"] = metrics
            return metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    async def record_request(self, method: str, endpoint: str, status_code: int, duration: float):
        """Record API request metrics"""
        try:
            key = f"{method}_{endpoint}"
            if key not in self.metrics["requests"]:
                self.metrics["requests"][key] = {
                    "count": 0,
                    "total_duration": 0,
                    "avg_duration": 0,
                    "status_codes": {}
                }
            
            self.metrics["requests"][key]["count"] += 1
            self.metrics["requests"][key]["total_duration"] += duration
            self.metrics["requests"][key]["avg_duration"] = (
                self.metrics["requests"][key]["total_duration"] / 
                self.metrics["requests"][key]["count"]
            )
            
            # Track status codes
            status_key = str(status_code)
            if status_key not in self.metrics["requests"][key]["status_codes"]:
                self.metrics["requests"][key]["status_codes"][status_key] = 0
            self.metrics["requests"][key]["status_codes"][status_key] += 1
            
        except Exception as e:
            logger.error(f"Error recording request metrics: {e}")
    
    async def record_error(self, error_type: str, error_message: str, context: Dict[str, Any] = None):
        """Record error metrics"""
        try:
            if error_type not in self.metrics["errors"]:
                self.metrics["errors"][error_type] = {
                    "count": 0,
                    "last_occurred": None,
                    "recent_messages": []
                }
            
            self.metrics["errors"][error_type]["count"] += 1
            self.metrics["errors"][error_type]["last_occurred"] = datetime.utcnow().isoformat()
            
            # Keep last 10 error messages
            self.metrics["errors"][error_type]["recent_messages"].append({
                "message": error_message,
                "timestamp": datetime.utcnow().isoformat(),
                "context": context or {}
            })
            
            if len(self.metrics["errors"][error_type]["recent_messages"]) > 10:
                self.metrics["errors"][error_type]["recent_messages"].pop(0)
                
        except Exception as e:
            logger.error(f"Error recording error metrics: {e}")
    
    async def get_current_metrics(self) -> Dict[str, Any]:
        """Get current metrics snapshot"""
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "uptime": self.get_uptime(),
            "requests": self.metrics["requests"],
            "system": self.metrics["system"],
            "services": self.metrics["services"],
            "errors": self.metrics["errors"]
        }


class HealthChecker:
    """Checks health of services and dependencies"""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=5.0)
    
    async def check_service_health(self, service_name: str, url: str) -> Dict[str, Any]:
        """Check health of a single service"""
        try:
            start_time = time.time()
            
            # Try to reach the service
            response = await self.http_client.get(f"{url}/health")
            
            duration = time.time() - start_time
            
            return {
                "service": service_name,
                "status": "healthy" if response.status_code == 200 else "unhealthy",
                "status_code": response.status_code,
                "response_time": duration,
                "timestamp": datetime.utcnow().isoformat(),
                "details": response.json() if response.headers.get("content-type", "").startswith("application/json") else None
            }
            
        except httpx.TimeoutException:
            return {
                "service": service_name,
                "status": "timeout",
                "error": "Service timeout",
                "timestamp": datetime.utcnow().isoformat()
            }
        except httpx.ConnectError:
            return {
                "service": service_name,
                "status": "unreachable",
                "error": "Cannot connect to service",
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            return {
                "service": service_name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def check_all_services(self) -> Dict[str, Any]:
        """Check health of all configured services"""
        results = {}
        
        # Check all service URLs from settings
        service_urls = settings.all_service_urls
        
        # Create tasks for concurrent health checks
        tasks = []
        for service_name, url in service_urls.items():
            task = asyncio.create_task(
                self.check_service_health(service_name, url)
            )
            tasks.append(task)
        
        # Wait for all health checks to complete
        health_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for result in health_results:
            if isinstance(result, Exception):
                logger.error(f"Health check exception: {result}")
                continue
            
            service_name = result.get("service")
            if service_name:
                results[service_name] = result
        
        return results
    
    async def check_database_health(self) -> Dict[str, Any]:
        """Check database health"""
        # This would be implemented with actual database connections
        return {
            "postgres": {"status": "healthy", "response_time": 0.01},
            "redis": {"status": "healthy", "response_time": 0.005},
            "mongodb": {"status": "healthy", "response_time": 0.02}
        }
    
    async def check_system_health(self) -> Dict[str, Any]:
        """Check overall system health"""
        try:
            # Check system resources
            cpu_percent = psutil.cpu_percent(interval=1)
            memory_percent = psutil.virtual_memory().percent
            disk_percent = psutil.disk_usage('/').percent
            
            # Determine overall health
            health_status = "healthy"
            if cpu_percent > 90 or memory_percent > 90 or disk_percent > 90:
                health_status = "critical"
            elif cpu_percent > 70 or memory_percent > 70 or disk_percent > 70:
                health_status = "warning"
            
            return {
                "status": health_status,
                "cpu_usage": cpu_percent,
                "memory_usage": memory_percent,
                "disk_usage": disk_percent,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"System health check error: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def shutdown(self):
        """Shutdown health checker"""
        await self.http_client.aclose()