#!/usr/bin/env python3
"""
SutazAI v9 Health Check Endpoints
Provides comprehensive health monitoring for all services
"""

import asyncio
import aiohttp
import psutil
import time
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import subprocess
import aioredis
import consul.aio
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    status: HealthStatus
    message: str
    duration_ms: float
    timestamp: float
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}

@dataclass
class SystemHealthReport:
    """Complete system health report"""
    overall_status: HealthStatus
    timestamp: float
    uptime_seconds: float
    version: str
    checks: List[HealthCheck]
    metrics: Dict[str, Any]
    dependencies: Dict[str, HealthStatus]

class HealthCheckRegistry:
    """Registry for health check functions"""
    
    def __init__(self):
        self.checks = {}
        self.dependencies = {}
    
    def register_check(self, name: str, check_function, timeout: int = 30):
        """Register a health check function"""
        self.checks[name] = {
            'function': check_function,
            'timeout': timeout
        }
    
    def register_dependency(self, name: str, endpoint: str, timeout: int = 10):
        """Register an external dependency"""
        self.dependencies[name] = {
            'endpoint': endpoint,
            'timeout': timeout
        }

class HealthMonitor:
    """Main health monitoring system"""
    
    def __init__(self, service_name: str = "sutazai-service"):
        self.service_name = service_name
        self.registry = HealthCheckRegistry()
        self.start_time = time.time()
        self.version = "9.0.0"
        
        # Register built-in health checks
        self._register_builtin_checks()
    
    def _register_builtin_checks(self):
        """Register built-in health checks"""
        self.registry.register_check("system_resources", self._check_system_resources)
        self.registry.register_check("disk_space", self._check_disk_space)
        self.registry.register_check("memory_usage", self._check_memory_usage)
        self.registry.register_check("cpu_usage", self._check_cpu_usage)
        self.registry.register_check("process_status", self._check_process_status)
        
        # Register dependencies
        self.registry.register_dependency("redis", "redis://redis-primary:6379")
        self.registry.register_dependency("consul", "http://consul:8500/v1/status/leader")
        self.registry.register_dependency("prometheus", "http://prometheus:9090/-/ready")
        self.registry.register_dependency("grafana", "http://grafana:3000/api/health")
    
    async def get_health_report(self) -> SystemHealthReport:
        """Generate comprehensive health report"""
        start_time = time.time()
        checks = []
        
        # Run all registered health checks
        for name, check_config in self.registry.checks.items():
            try:
                check_start = time.time()
                
                # Run check with timeout
                result = await asyncio.wait_for(
                    check_config['function'](),
                    timeout=check_config['timeout']
                )
                
                duration_ms = (time.time() - check_start) * 1000
                
                check = HealthCheck(
                    name=name,
                    status=result.get('status', HealthStatus.UNKNOWN),
                    message=result.get('message', ''),
                    duration_ms=duration_ms,
                    timestamp=time.time(),
                    details=result.get('details', {})
                )
                checks.append(check)
                
            except asyncio.TimeoutError:
                checks.append(HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check timed out after {check_config['timeout']}s",
                    duration_ms=check_config['timeout'] * 1000,
                    timestamp=time.time()
                ))
            except Exception as e:
                checks.append(HealthCheck(
                    name=name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {str(e)}",
                    duration_ms=0,
                    timestamp=time.time()
                ))
        
        # Check dependencies
        dependencies = await self._check_dependencies()
        
        # Calculate overall status
        overall_status = self._calculate_overall_status(checks, dependencies)
        
        # Get system metrics
        metrics = await self._get_system_metrics()
        
        report = SystemHealthReport(
            overall_status=overall_status,
            timestamp=time.time(),
            uptime_seconds=time.time() - self.start_time,
            version=self.version,
            checks=checks,
            metrics=metrics,
            dependencies=dependencies
        )
        
        return report
    
    async def _check_dependencies(self) -> Dict[str, HealthStatus]:
        """Check external dependencies"""
        dependency_status = {}
        
        async with aiohttp.ClientSession() as session:
            for name, config in self.registry.dependencies.items():
                try:
                    if name == "redis":
                        status = await self._check_redis_dependency(config['endpoint'])
                    else:
                        status = await self._check_http_dependency(session, config['endpoint'], config['timeout'])
                    
                    dependency_status[name] = status
                    
                except Exception as e:
                    logger.warning(f"Failed to check dependency {name}: {e}")
                    dependency_status[name] = HealthStatus.UNHEALTHY
        
        return dependency_status
    
    async def _check_redis_dependency(self, redis_url: str) -> HealthStatus:
        """Check Redis connectivity"""
        try:
            redis = await aioredis.from_url(redis_url)
            await redis.ping()
            await redis.close()
            return HealthStatus.HEALTHY
        except Exception:
            return HealthStatus.UNHEALTHY
    
    async def _check_http_dependency(self, session: aiohttp.ClientSession, url: str, timeout: int) -> HealthStatus:
        """Check HTTP dependency"""
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as response:
                if response.status == 200:
                    return HealthStatus.HEALTHY
                else:
                    return HealthStatus.DEGRADED
        except Exception:
            return HealthStatus.UNHEALTHY
    
    def _calculate_overall_status(self, checks: List[HealthCheck], dependencies: Dict[str, HealthStatus]) -> HealthStatus:
        """Calculate overall system health status"""
        # Count status types
        check_statuses = [check.status for check in checks]
        all_statuses = check_statuses + list(dependencies.values())
        
        unhealthy_count = all_statuses.count(HealthStatus.UNHEALTHY)
        degraded_count = all_statuses.count(HealthStatus.DEGRADED)
        
        # Determine overall status
        if unhealthy_count > 0:
            return HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            return HealthStatus.DEGRADED
        else:
            return HealthStatus.HEALTHY
    
    async def _get_system_metrics(self) -> Dict[str, Any]:
        """Get system performance metrics"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            return {
                'cpu_usage_percent': cpu_percent,
                'memory_usage_percent': memory.percent,
                'memory_available_gb': round(memory.available / (1024**3), 2),
                'disk_usage_percent': round((disk.used / disk.total) * 100, 2),
                'disk_free_gb': round(disk.free / (1024**3), 2),
                'load_average': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None,
                'process_count': len(psutil.pids())
            }
        except Exception as e:
            logger.error(f"Failed to get system metrics: {e}")
            return {}
    
    # Built-in health check functions
    
    async def _check_system_resources(self) -> Dict[str, Any]:
        """Check system resource availability"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # Determine status based on resource usage
            if cpu_percent > 90 or memory.percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "System resources critically low"
            elif cpu_percent > 80 or memory.percent > 85:
                status = HealthStatus.DEGRADED
                message = "System resources under pressure"
            else:
                status = HealthStatus.HEALTHY
                message = "System resources normal"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_available_gb': round(memory.available / (1024**3), 2)
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Failed to check system resources: {str(e)}"
            }
    
    async def _check_disk_space(self) -> Dict[str, Any]:
        """Check disk space availability"""
        try:
            disk = psutil.disk_usage('/')
            usage_percent = (disk.used / disk.total) * 100
            
            if usage_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Disk space critically low"
            elif usage_percent > 85:
                status = HealthStatus.DEGRADED
                message = "Disk space low"
            else:
                status = HealthStatus.HEALTHY
                message = "Disk space normal"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'usage_percent': round(usage_percent, 2),
                    'free_gb': round(disk.free / (1024**3), 2),
                    'total_gb': round(disk.total / (1024**3), 2)
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Failed to check disk space: {str(e)}"
            }
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage"""
        try:
            memory = psutil.virtual_memory()
            
            if memory.percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "Memory usage critically high"
            elif memory.percent > 85:
                status = HealthStatus.DEGRADED
                message = "Memory usage high"
            else:
                status = HealthStatus.HEALTHY
                message = "Memory usage normal"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'percent': memory.percent,
                    'available_gb': round(memory.available / (1024**3), 2),
                    'total_gb': round(memory.total / (1024**3), 2)
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Failed to check memory usage: {str(e)}"
            }
    
    async def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage"""
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent > 90:
                status = HealthStatus.UNHEALTHY
                message = "CPU usage critically high"
            elif cpu_percent > 80:
                status = HealthStatus.DEGRADED
                message = "CPU usage high"
            else:
                status = HealthStatus.HEALTHY
                message = "CPU usage normal"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'percent': cpu_percent,
                    'count': psutil.cpu_count(),
                    'load_avg': list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Failed to check CPU usage: {str(e)}"
            }
    
    async def _check_process_status(self) -> Dict[str, Any]:
        """Check process status"""
        try:
            process_count = len(psutil.pids())
            
            # Check for zombie processes
            zombie_count = 0
            for proc in psutil.process_iter(['status']):
                try:
                    if proc.info['status'] == psutil.STATUS_ZOMBIE:
                        zombie_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            
            if zombie_count > 10:
                status = HealthStatus.DEGRADED
                message = f"High number of zombie processes: {zombie_count}"
            else:
                status = HealthStatus.HEALTHY
                message = "Process status normal"
            
            return {
                'status': status,
                'message': message,
                'details': {
                    'total_processes': process_count,
                    'zombie_processes': zombie_count
                }
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Failed to check process status: {str(e)}"
            }

# FastAPI/Starlette health endpoints
class HealthEndpoints:
    """HTTP endpoints for health monitoring"""
    
    def __init__(self, health_monitor: HealthMonitor):
        self.health_monitor = health_monitor
    
    async def health_check(self, request) -> Dict[str, Any]:
        """Basic health check endpoint"""
        try:
            report = await self.health_monitor.get_health_report()
            
            # Return simplified response for basic health check
            return {
                'status': report.overall_status.value,
                'timestamp': report.timestamp,
                'uptime': report.uptime_seconds,
                'version': report.version
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'timestamp': time.time(),
                'error': str(e)
            }
    
    async def health_detailed(self, request) -> Dict[str, Any]:
        """Detailed health check endpoint"""
        try:
            report = await self.health_monitor.get_health_report()
            
            # Convert dataclasses to dictionaries
            checks_dict = [asdict(check) for check in report.checks]
            
            # Convert enum values to strings
            for check in checks_dict:
                check['status'] = check['status'].value
            
            dependencies_dict = {k: v.value for k, v in report.dependencies.items()}
            
            return {
                'status': report.overall_status.value,
                'timestamp': report.timestamp,
                'uptime': report.uptime_seconds,
                'version': report.version,
                'checks': checks_dict,
                'dependencies': dependencies_dict,
                'metrics': report.metrics
            }
        except Exception as e:
            logger.error(f"Detailed health check failed: {e}")
            return {
                'status': HealthStatus.UNHEALTHY.value,
                'timestamp': time.time(),
                'error': str(e)
            }
    
    async def liveness_probe(self, request) -> Dict[str, Any]:
        """Kubernetes liveness probe endpoint"""
        # Simple check - if we can respond, we're alive
        return {
            'status': 'alive',
            'timestamp': time.time()
        }
    
    async def readiness_probe(self, request) -> Dict[str, Any]:
        """Kubernetes readiness probe endpoint"""
        try:
            # Check critical dependencies only
            critical_checks = ['system_resources', 'disk_space']
            
            for check_name in critical_checks:
                if check_name in self.health_monitor.registry.checks:
                    result = await self.health_monitor.registry.checks[check_name]['function']()
                    if result.get('status') == HealthStatus.UNHEALTHY:
                        return {
                            'status': 'not_ready',
                            'reason': f"Critical check failed: {check_name}",
                            'timestamp': time.time()
                        }
            
            return {
                'status': 'ready',
                'timestamp': time.time()
            }
        except Exception as e:
            return {
                'status': 'not_ready',
                'reason': str(e),
                'timestamp': time.time()
            }
    
    async def metrics_endpoint(self, request) -> str:
        """Prometheus metrics endpoint"""
        try:
            report = await self.health_monitor.get_health_report()
            
            # Generate Prometheus metrics format
            metrics = []
            
            # Overall health status (1 = healthy, 0.5 = degraded, 0 = unhealthy)
            status_value = {
                HealthStatus.HEALTHY: 1,
                HealthStatus.DEGRADED: 0.5,
                HealthStatus.UNHEALTHY: 0
            }.get(report.overall_status, 0)
            
            metrics.append(f'sutazai_health_status {status_value}')
            metrics.append(f'sutazai_uptime_seconds {report.uptime_seconds}')
            
            # Individual check statuses
            for check in report.checks:
                check_value = {
                    HealthStatus.HEALTHY: 1,
                    HealthStatus.DEGRADED: 0.5,
                    HealthStatus.UNHEALTHY: 0
                }.get(check.status, 0)
                
                metrics.append(f'sutazai_health_check{{name="{check.name}"}} {check_value}')
                metrics.append(f'sutazai_health_check_duration_seconds{{name="{check.name}"}} {check.duration_ms/1000}')
            
            # Dependency statuses
            for dep_name, dep_status in report.dependencies.items():
                dep_value = {
                    HealthStatus.HEALTHY: 1,
                    HealthStatus.DEGRADED: 0.5,
                    HealthStatus.UNHEALTHY: 0
                }.get(dep_status, 0)
                
                metrics.append(f'sutazai_dependency_status{{name="{dep_name}"}} {dep_value}')
            
            # System metrics
            if report.metrics:
                for metric_name, metric_value in report.metrics.items():
                    if isinstance(metric_value, (int, float)):
                        metrics.append(f'sutazai_system_{metric_name} {metric_value}')
            
            return '\n'.join(metrics) + '\n'
            
        except Exception as e:
            logger.error(f"Metrics endpoint failed: {e}")
            return f'sutazai_health_status 0\n# Error: {str(e)}\n'

# Example usage and service-specific health monitors

class AIModelHealthMonitor(HealthMonitor):
    """Health monitor specifically for AI model services"""
    
    def __init__(self, model_name: str):
        super().__init__(f"ai-model-{model_name}")
        self.model_name = model_name
        self._register_ai_specific_checks()
    
    def _register_ai_specific_checks(self):
        """Register AI model specific health checks"""
        self.registry.register_check("model_loaded", self._check_model_loaded)
        self.registry.register_check("gpu_memory", self._check_gpu_memory)
        self.registry.register_check("inference_queue", self._check_inference_queue)
        self.registry.register_check("model_performance", self._check_model_performance)
    
    async def _check_model_loaded(self) -> Dict[str, Any]:
        """Check if AI model is properly loaded"""
        # This would integrate with your specific AI model loading system
        try:
            # Placeholder - implement actual model status check
            model_loaded = True  # Replace with actual check
            
            if model_loaded:
                return {
                    'status': HealthStatus.HEALTHY,
                    'message': f"Model {self.model_name} is loaded and ready",
                    'details': {'model_name': self.model_name}
                }
            else:
                return {
                    'status': HealthStatus.UNHEALTHY,
                    'message': f"Model {self.model_name} is not loaded"
                }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Failed to check model status: {str(e)}"
            }
    
    async def _check_gpu_memory(self) -> Dict[str, Any]:
        """Check GPU memory usage"""
        try:
            # This would use nvidia-ml-py or similar to check GPU memory
            # Placeholder implementation
            gpu_memory_percent = 75  # Replace with actual GPU memory check
            
            if gpu_memory_percent > 95:
                status = HealthStatus.UNHEALTHY
                message = "GPU memory critically high"
            elif gpu_memory_percent > 85:
                status = HealthStatus.DEGRADED
                message = "GPU memory high"
            else:
                status = HealthStatus.HEALTHY
                message = "GPU memory normal"
            
            return {
                'status': status,
                'message': message,
                'details': {'gpu_memory_percent': gpu_memory_percent}
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNKNOWN,
                'message': f"Could not check GPU memory: {str(e)}"
            }
    
    async def _check_inference_queue(self) -> Dict[str, Any]:
        """Check inference request queue length"""
        try:
            # This would check your actual inference queue
            queue_length = 5  # Replace with actual queue length check
            
            if queue_length > 100:
                status = HealthStatus.DEGRADED
                message = "Inference queue backed up"
            else:
                status = HealthStatus.HEALTHY
                message = "Inference queue normal"
            
            return {
                'status': status,
                'message': message,
                'details': {'queue_length': queue_length}
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Failed to check inference queue: {str(e)}"
            }
    
    async def _check_model_performance(self) -> Dict[str, Any]:
        """Check model inference performance"""
        try:
            # This would check actual model performance metrics
            avg_inference_time = 2.5  # Replace with actual metric
            
            if avg_inference_time > 30:
                status = HealthStatus.DEGRADED
                message = "Model inference slow"
            else:
                status = HealthStatus.HEALTHY
                message = "Model performance normal"
            
            return {
                'status': status,
                'message': message,
                'details': {'avg_inference_time_seconds': avg_inference_time}
            }
        except Exception as e:
            return {
                'status': HealthStatus.UNHEALTHY,
                'message': f"Failed to check model performance: {str(e)}"
            }

# Integration with FastAPI/Starlette
def setup_health_endpoints(app, health_monitor: HealthMonitor):
    """Setup health endpoints for FastAPI/Starlette app"""
    health_endpoints = HealthEndpoints(health_monitor)
    
    # Add routes
    app.add_route('/health', health_endpoints.health_check, methods=['GET'])
    app.add_route('/health/detailed', health_endpoints.health_detailed, methods=['GET'])
    app.add_route('/health/live', health_endpoints.liveness_probe, methods=['GET'])
    app.add_route('/health/ready', health_endpoints.readiness_probe, methods=['GET'])
    app.add_route('/metrics', health_endpoints.metrics_endpoint, methods=['GET'])

# Example usage
async def main():
    """Example usage of health monitoring system"""
    
    # Create health monitor
    health_monitor = HealthMonitor("sutazai-main-service")
    
    # Add custom health check
    async def check_database():
        # Custom database health check
        return {
            'status': HealthStatus.HEALTHY,
            'message': 'Database connection OK'
        }
    
    health_monitor.registry.register_check("database", check_database)
    
    # Get health report
    report = await health_monitor.get_health_report()
    
    print(f"Overall Status: {report.overall_status.value}")
    print(f"Uptime: {report.uptime_seconds:.2f} seconds")
    print(f"Version: {report.version}")
    
    for check in report.checks:
        print(f"  {check.name}: {check.status.value} - {check.message}")
    
    for dep_name, dep_status in report.dependencies.items():
        print(f"  {dep_name} (dep): {dep_status.value}")

if __name__ == "__main__":
    asyncio.run(main())