"""
Monitoring and Metrics Collection for SutazAI
=============================================

Comprehensive monitoring, metrics collection, and health checking.
"""

import asyncio
import time
import psutil
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict, deque

from .config import Settings
from .utils import setup_logging

logger = setup_logging(__name__)


class MetricsCollector:
    """Collects and manages system and application metrics"""
    
    def __init__(self):
        self.request_metrics = defaultdict(lambda: {
            "count": 0,
            "total_time": 0.0,
            "error_count": 0,
            "response_times": deque(maxlen=1000)
        })
        self.system_metrics = {}
        self.custom_metrics = defaultdict(float)
        self.start_time = time.time()
    
    async def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        processing_time: float
    ):
        """Record request metrics"""
        key = f"{method}:{endpoint}"
        metrics = self.request_metrics[key]
        
        metrics["count"] += 1
        metrics["total_time"] += processing_time
        metrics["response_times"].append(processing_time)
        
        if status_code >= 400:
            metrics["error_count"] += 1
    
    def record_custom_metric(self, name: str, value: float):
        """Record custom metric"""
        self.custom_metrics[name] = value
    
    def increment_counter(self, name: str, value: float = 1.0):
        """Increment counter metric"""
        self.custom_metrics[name] += value
    
    async def collect_system_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            load_avg = psutil.getloadavg() if hasattr(psutil, 'getloadavg') else [0, 0, 0]
            
            # Memory metrics
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            
            # Network metrics
            network = psutil.net_io_counters()
            
            self.system_metrics = {
                "cpu": {
                    "percent": cpu_percent,
                    "count": cpu_count,
                    "load_avg_1m": load_avg[0],
                    "load_avg_5m": load_avg[1],
                    "load_avg_15m": load_avg[2]
                },
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "used": memory.used,
                    "percent": memory.percent,
                    "swap_total": swap.total,
                    "swap_used": swap.used,
                    "swap_percent": swap.percent
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
                "uptime": time.time() - self.start_time
            }
            
            return self.system_metrics
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
            return {}
    
    def get_request_metrics(self) -> Dict[str, Any]:
        """Get request metrics summary"""
        summary = {}
        
        for endpoint, metrics in self.request_metrics.items():
            if metrics["count"] > 0:
                avg_time = metrics["total_time"] / metrics["count"]
                error_rate = metrics["error_count"] / metrics["count"] * 100
                
                # Calculate percentiles
                response_times = sorted(metrics["response_times"])
                if response_times:
                    p50 = response_times[len(response_times) // 2]
                    p95 = response_times[int(len(response_times) * 0.95)]
                    p99 = response_times[int(len(response_times) * 0.99)]
                else:
                    p50 = p95 = p99 = 0
                
                summary[endpoint] = {
                    "request_count": metrics["count"],
                    "error_count": metrics["error_count"],
                    "error_rate": error_rate,
                    "avg_response_time": avg_time,
                    "p50_response_time": p50,
                    "p95_response_time": p95,
                    "p99_response_time": p99
                }
        
        return summary
    
    def get_prometheus_metrics(self) -> str:
        """Generate Prometheus-format metrics"""
        lines = []
        
        # System metrics
        if self.system_metrics:
            lines.extend([
                f"# HELP sutazai_cpu_percent CPU usage percentage",
                f"# TYPE sutazai_cpu_percent gauge",
                f"sutazai_cpu_percent {self.system_metrics['cpu']['percent']}",
                "",
                f"# HELP sutazai_memory_used_bytes Memory used in bytes",
                f"# TYPE sutazai_memory_used_bytes gauge",
                f"sutazai_memory_used_bytes {self.system_metrics['memory']['used']}",
                "",
                f"# HELP sutazai_memory_percent Memory usage percentage",
                f"# TYPE sutazai_memory_percent gauge",
                f"sutazai_memory_percent {self.system_metrics['memory']['percent']}",
                "",
                f"# HELP sutazai_disk_used_bytes Disk used in bytes",
                f"# TYPE sutazai_disk_used_bytes gauge",
                f"sutazai_disk_used_bytes {self.system_metrics['disk']['used']}",
                "",
                f"# HELP sutazai_uptime_seconds System uptime in seconds",
                f"# TYPE sutazai_uptime_seconds counter",
                f"sutazai_uptime_seconds {self.system_metrics['uptime']}",
                ""
            ])
        
        # Request metrics
        for endpoint, metrics in self.request_metrics.items():
            if metrics["count"] > 0:
                method, path = endpoint.split(":", 1)
                avg_time = metrics["total_time"] / metrics["count"]
                
                lines.extend([
                    f"# HELP sutazai_requests_total Total number of requests",
                    f"# TYPE sutazai_requests_total counter",
                    f'sutazai_requests_total{{method="{method}",path="{path}"}} {metrics["count"]}',
                    "",
                    f"# HELP sutazai_request_duration_seconds Request duration",
                    f"# TYPE sutazai_request_duration_seconds histogram",
                    f'sutazai_request_duration_seconds_sum{{method="{method}",path="{path}"}} {metrics["total_time"]}',
                    f'sutazai_request_duration_seconds_count{{method="{method}",path="{path}"}} {metrics["count"]}',
                    "",
                    f"# HELP sutazai_request_errors_total Request errors",
                    f"# TYPE sutazai_request_errors_total counter",
                    f'sutazai_request_errors_total{{method="{method}",path="{path}"}} {metrics["error_count"]}',
                    ""
                ])
        
        # Custom metrics
        for name, value in self.custom_metrics.items():
            lines.extend([
                f"# HELP sutazai_{name} Custom metric {name}",
                f"# TYPE sutazai_{name} gauge",
                f"sutazai_{name} {value}",
                ""
            ])
        
        return "\n".join(lines)


class HealthChecker:
    """Comprehensive health checking system"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.health_checks = {}
        self.last_check_time = {}
        self.check_interval = 30  # seconds
    
    def register_health_check(self, name: str, check_func: callable):
        """Register a health check function"""
        self.health_checks[name] = check_func
        self.last_check_time[name] = 0
    
    async def run_health_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check"""
        if name not in self.health_checks:
            return {"status": "unknown", "error": "Health check not found"}
        
        try:
            start_time = time.time()
            result = await self.health_checks[name]()
            execution_time = time.time() - start_time
            
            if isinstance(result, dict):
                result["execution_time"] = execution_time
                result["checked_at"] = datetime.now().isoformat()
            else:
                result = {
                    "status": "healthy" if result else "unhealthy",
                    "execution_time": execution_time,
                    "checked_at": datetime.now().isoformat()
                }
            
            self.last_check_time[name] = time.time()
            return result
            
        except Exception as e:
            logger.error(f"Health check {name} failed: {e}")
            return {
                "status": "unhealthy",
                "error": str(e),
                "execution_time": 0,
                "checked_at": datetime.now().isoformat()
            }
    
    async def run_all_health_checks(self) -> Dict[str, Any]:
        """Run all registered health checks"""
        results = {}
        
        for name in self.health_checks:
            results[name] = await self.run_health_check(name)
        
        # Determine overall health
        overall_status = "healthy"
        unhealthy_services = []
        
        for name, result in results.items():
            if result.get("status") == "unhealthy":
                overall_status = "unhealthy"
                unhealthy_services.append(name)
            elif result.get("status") == "degraded":
                if overall_status == "healthy":
                    overall_status = "degraded"
        
        return {
            "overall_status": overall_status,
            "unhealthy_services": unhealthy_services,
            "checks": results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def check_database_health(self, db_manager) -> Dict[str, Any]:
        """Check database health"""
        try:
            health = await db_manager.health_check()
            
            # Determine overall database health
            all_healthy = all(
                check.get("status") == "healthy"
                for check in health.values()
            )
            
            return {
                "status": "healthy" if all_healthy else "unhealthy",
                "details": health
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_model_health(self) -> Dict[str, Any]:
        """Check AI model service health"""
        try:
            import aiohttp
            
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.settings.ollama_url}/api/tags",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        models = await response.json()
                        return {
                            "status": "healthy",
                            "model_count": len(models.get("models", [])),
                            "available_models": [m["name"] for m in models.get("models", [])]
                        }
                    else:
                        return {
                            "status": "unhealthy",
                            "error": f"HTTP {response.status}"
                        }
                        
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
    
    async def check_vector_store_health(self) -> Dict[str, Any]:
        """Check vector store health"""
        try:
            import aiohttp
            
            health_results = {}
            
            # Check ChromaDB
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.settings.chromadb_url}/api/v1/heartbeat",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        health_results["chromadb"] = {
                            "status": "healthy" if response.status == 200 else "unhealthy",
                            "response_code": response.status
                        }
            except Exception as e:
                health_results["chromadb"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Check Qdrant
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"{self.settings.qdrant_url}/health",
                        timeout=aiohttp.ClientTimeout(total=5)
                    ) as response:
                        health_results["qdrant"] = {
                            "status": "healthy" if response.status == 200 else "unhealthy",
                            "response_code": response.status
                        }
            except Exception as e:
                health_results["qdrant"] = {
                    "status": "unhealthy",
                    "error": str(e)
                }
            
            # Determine overall status
            all_healthy = all(
                result.get("status") == "healthy"
                for result in health_results.values()
            )
            
            return {
                "status": "healthy" if all_healthy else "unhealthy",
                "details": health_results
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }


class AlertManager:
    """Manages system alerts and notifications"""
    
    def __init__(self):
        self.alerts = []
        self.alert_thresholds = {
            "cpu_usage": 80.0,
            "memory_usage": 85.0,
            "disk_usage": 90.0,
            "error_rate": 5.0,
            "response_time": 2.0
        }
        self.alert_history = deque(maxlen=1000)
    
    def check_thresholds(self, metrics: Dict[str, Any]):
        """Check metrics against thresholds and generate alerts"""
        current_alerts = []
        
        # Check system metrics
        if "cpu" in metrics:
            cpu_usage = metrics["cpu"].get("percent", 0)
            if cpu_usage > self.alert_thresholds["cpu_usage"]:
                current_alerts.append({
                    "type": "cpu_high",
                    "severity": "warning",
                    "message": f"High CPU usage: {cpu_usage:.1f}%",
                    "value": cpu_usage,
                    "threshold": self.alert_thresholds["cpu_usage"]
                })
        
        if "memory" in metrics:
            memory_usage = metrics["memory"].get("percent", 0)
            if memory_usage > self.alert_thresholds["memory_usage"]:
                current_alerts.append({
                    "type": "memory_high",
                    "severity": "warning",
                    "message": f"High memory usage: {memory_usage:.1f}%",
                    "value": memory_usage,
                    "threshold": self.alert_thresholds["memory_usage"]
                })
        
        if "disk" in metrics:
            disk_usage = metrics["disk"].get("percent", 0)
            if disk_usage > self.alert_thresholds["disk_usage"]:
                current_alerts.append({
                    "type": "disk_high",
                    "severity": "critical",
                    "message": f"High disk usage: {disk_usage:.1f}%",
                    "value": disk_usage,
                    "threshold": self.alert_thresholds["disk_usage"]
                })
        
        # Update alerts
        timestamp = datetime.now()
        for alert in current_alerts:
            alert["timestamp"] = timestamp.isoformat()
            
            # Add to history
            self.alert_history.append(alert)
            
            # Log alert
            level = "critical" if alert["severity"] == "critical" else "warning"
            getattr(logger, level)(f"ALERT: {alert['message']}")
        
        self.alerts = current_alerts
        return current_alerts
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get currently active alerts"""
        return self.alerts
    
    def get_alert_history(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get alert history for specified hours"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            alert for alert in self.alert_history
            if datetime.fromisoformat(alert["timestamp"]) > cutoff_time
        ]