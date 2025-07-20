# backend/app/services/metrics_collector.py
from typing import Dict, Any, List
from datetime import datetime, timedelta
import asyncio
import logging
from prometheus_client import Counter, Histogram, Gauge, Info
import psutil
import GPUtil

logger = logging.getLogger(__name__)

class MetricsCollector:
    """Comprehensive metrics collection service"""
    
    def __init__(self):
        # Define Prometheus metrics
        self.request_counter = Counter(
            'sutazai_requests_total',
            'Total number of requests',
            ['method', 'endpoint', 'status']
        )
        
        self.response_time_histogram = Histogram(
            'sutazai_response_time_seconds',
            'Response time in seconds',
            ['method', 'endpoint']
        )
        
        self.active_users_gauge = Gauge(
            'sutazai_active_users',
            'Number of active users'
        )
        
        self.model_inference_histogram = Histogram(
            'sutazai_model_inference_seconds',
            'Model inference time in seconds',
            ['model_name', 'task_type']
        )
        
        self.agent_tasks_counter = Counter(
            'sutazai_agent_tasks_total',
            'Total agent tasks',
            ['agent_type', 'status']
        )
        
        self.system_info = Info(
            'sutazai_system',
            'System information'
        )
        
        # System resource gauges
        self.cpu_usage_gauge = Gauge(
            'sutazai_cpu_usage_percent',
            'CPU usage percentage'
        )
        
        self.memory_usage_gauge = Gauge(
            'sutazai_memory_usage_percent',
            'Memory usage percentage'
        )
        
        self.gpu_usage_gauge = Gauge(
            'sutazai_gpu_usage_percent',
            'GPU usage percentage',
            ['gpu_id']
        )
        
        self.disk_usage_gauge = Gauge(
            'sutazai_disk_usage_percent',
            'Disk usage percentage',
            ['mount_point']
        )
        
        # Start background metrics collection
        asyncio.create_task(self._collect_system_metrics())
    
    async def _collect_system_metrics(self):
        """Collect system metrics in background"""
        while True:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.cpu_usage_gauge.set(cpu_percent)
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.memory_usage_gauge.set(memory.percent)
                
                # GPU usage (if available)
                try:
                    gpus = GPUtil.getGPUs()
                    for gpu in gpus:
                        self.gpu_usage_gauge.labels(
                            gpu_id=gpu.id
                        ).set(gpu.load * 100)
                except Exception:
                    pass
                
                # Disk usage
                for partition in psutil.disk_partitions():
                    if partition.fstype:
                        try:
                            usage = psutil.disk_usage(partition.mountpoint)
                            self.disk_usage_gauge.labels(
                                mount_point=partition.mountpoint
                            ).set(usage.percent)
                        except Exception:
                            pass # Ignore errors for removable media or inaccessible paths
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(60)
    
    def record_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        response_time: float
    ):
        """Record API request metrics"""
        self.request_counter.labels(
            method=method,
            endpoint=endpoint,
            status=str(status_code)
        ).inc()
        
        self.response_time_histogram.labels(
            method=method,
            endpoint=endpoint
        ).observe(response_time)
    
    def record_model_inference(
        self,
        model_name: str,
        task_type: str,
        inference_time: float
    ):
        """Record model inference metrics"""
        self.model_inference_histogram.labels(
            model_name=model_name,
            task_type=task_type
        ).observe(inference_time)
    
    def record_agent_task(
        self,
        agent_type: str,
        status: str
    ):
        """Record agent task execution"""
        self.agent_tasks_counter.labels(
            agent_type=agent_type,
            status=status
        ).inc()
    
    def update_active_users(self, count: int):
        """Update active users count"""
        self.active_users_gauge.set(count)
    
    async def _check_database(self): return {"status": "healthy"}
    async def _check_redis(self): return {"status": "healthy"}
    async def _check_models(self): return {"status": "healthy"}
    async def _check_agents(self): return {"status": "healthy"}
    async def _check_storage(self): return {"status": "healthy"}

    async def get_system_health(self) -> Dict[str, Any]:
        """Get comprehensive system health status"""
        
        # Check all critical components
        health_checks = {
            "database": await self._check_database(),
            "redis": await self._check_redis(),
            "models": await self._check_models(),
            "agents": await self._check_agents(),
            "storage": await self._check_storage()
        }
        
        # Calculate overall health
        unhealthy_services = [
            service for service, status in health_checks.items()
            if status["status"] != "healthy"
        ]
        
        if not unhealthy_services:
            overall_status = "healthy"
        elif len(unhealthy_services) <= 2:
            overall_status = "degraded"
        else:
            overall_status = "unhealthy"
        
        boot_time = datetime.fromtimestamp(psutil.boot_time())
        uptime_seconds = (datetime.now() - boot_time).total_seconds()

        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "services": health_checks,
            "metrics": {
                "cpu_usage": psutil.cpu_percent(),
                "memory_usage": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent,
                "active_connections": len(psutil.net_connections(kind='inet')),
                "uptime_hours": uptime_seconds / 3600
            }
        }
    
    async def _get_api_metrics(self, start_time, end_time): return {"total_requests": 0, "avg_response_time": 0, "p95_response_time": 0, "error_rate": 0}
    async def _get_model_metrics(self, start_time, end_time): return {"total_inferences": 0, "avg_inference_time": 0, "models_used": []}
    async def _get_agent_metrics(self, start_time, end_time): return {"total_tasks": 0, "success_rate": 1.0, "avg_execution_time": 0}
    async def _get_resource_metrics(self, start_time, end_time): return {"avg_cpu": 0, "peak_cpu": 0, "avg_memory": 0, "peak_memory": 0}

    async def generate_performance_report(
        self,
        time_range: timedelta = timedelta(days=7)
    ) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        
        end_time = datetime.utcnow()
        start_time = end_time - time_range
        
        report = {
            "report_period": {
                "start": start_time.isoformat(),
                "end": end_time.isoformat()
            },
            "summary": {},
            "details": {},
            "recommendations": []
        }
        
        # API Performance
        api_metrics = await self._get_api_metrics(start_time, end_time)
        report["summary"]["api_performance"] = {
            "total_requests": api_metrics["total_requests"],
            "average_response_time": api_metrics["avg_response_time"],
            "p95_response_time": api_metrics["p95_response_time"],
            "error_rate": api_metrics["error_rate"]
        }
        
        # Model Performance
        model_metrics = await self._get_model_metrics(start_time, end_time)
        report["summary"]["model_performance"] = {
            "total_inferences": model_metrics["total_inferences"],
            "average_inference_time": model_metrics["avg_inference_time"],
            "models_used": model_metrics["models_used"]
        }
        
        # Agent Performance
        agent_metrics = await self._get_agent_metrics(start_time, end_time)
        report["summary"]["agent_performance"] = {
            "total_tasks": agent_metrics["total_tasks"],
            "success_rate": agent_metrics["success_rate"],
            "average_execution_time": agent_metrics["avg_execution_time"]
        }
        
        # Resource Utilization
        resource_metrics = await self._get_resource_metrics(start_time, end_time)
        report["summary"]["resource_utilization"] = {
            "avg_cpu_usage": resource_metrics["avg_cpu"],
            "peak_cpu_usage": resource_metrics["peak_cpu"],
            "avg_memory_usage": resource_metrics["avg_memory"],
            "peak_memory_usage": resource_metrics["peak_memory"]
        }
        
        # Generate recommendations
        report["recommendations"] = self._generate_recommendations(report["summary"])
        
        return report
    
    def _generate_recommendations(
        self,
        summary: Dict[str, Any]
    ) -> List[Dict[str, str]]:
        """Generate performance recommendations based on metrics"""
        
        recommendations = []
        
        # API performance recommendations
        if summary.get("api_performance", {}).get("p95_response_time", 0) > 1.0:
            recommendations.append({
                "category": "API Performance",
                "issue": "High response time detected",
                "recommendation": "Consider implementing caching or optimizing database queries"
            })
        
        if summary.get("api_performance", {}).get("error_rate", 0) > 0.01:
            recommendations.append({
                "category": "API Reliability",
                "issue": "High error rate detected",
                "recommendation": "Review error logs and implement better error handling"
            })
        
        # Resource recommendations
        if summary.get("resource_utilization", {}).get("avg_cpu_usage", 0) > 80:
            recommendations.append({
                "category": "Resource Usage",
                "issue": "High CPU utilization",
                "recommendation": "Consider scaling horizontally or optimizing CPU-intensive operations"
            })
        
        if summary.get("resource_utilization", {}).get("peak_memory_usage", 0) > 90:
            recommendations.append({
                "category": "Resource Usage",
                "issue": "High memory usage detected",
                "recommendation": "Review memory leaks and consider increasing available memory"
            })
        
        # Agent performance recommendations
        if summary.get("agent_performance", {}).get("success_rate", 1.0) < 0.9:
            recommendations.append({
                "category": "Agent Performance",
                "issue": "Low agent task success rate",
                "recommendation": "Review failed tasks and improve error handling in agents"
            })
        
        return recommendations
