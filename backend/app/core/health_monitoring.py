#!/usr/bin/env python3
"""
Ultra-Enhanced Health Monitoring System for SutazAI
Provides comprehensive observability with zero performance impact

Features:
- Individual service health checks with detailed metrics
- Circuit breaker status integration
- Response time tracking per service
- Intelligent caching with cache warming
- Prometheus metrics integration
- Graceful degradation under load
- Zero-performance-impact design
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Callable, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import psutil
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded" 
    UNHEALTHY = "unhealthy"
    TIMEOUT = "timeout"
    CIRCUIT_OPEN = "circuit_open"
    UNKNOWN = "unknown"


class SystemStatus(Enum):
    """Overall system health status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class ServiceMetrics:
    """Detailed metrics for a single service"""
    name: str
    status: ServiceStatus
    response_time_ms: float = 0.0
    last_check: Optional[datetime] = None
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    error_message: Optional[str] = None
    consecutive_failures: int = 0
    total_checks: int = 0
    success_rate: float = 0.0
    circuit_breaker_state: Optional[str] = None
    circuit_breaker_failures: int = 0
    uptime_percentage: float = 0.0
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealthReport:
    """Complete system health report"""
    overall_status: SystemStatus
    timestamp: datetime
    services: Dict[str, ServiceMetrics]
    performance_metrics: Dict[str, Any]
    system_resources: Dict[str, Any]
    alerts: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class HealthMonitoringService:
    """
    Ultra-high-performance health monitoring with zero impact design
    
    Features:
    - Async-first architecture with connection pooling
    - Intelligent caching to minimize overhead  
    - Circuit breaker integration for resilience
    - Per-service response time tracking
    - Prometheus metrics export
    - Smart cache warming and invalidation
    """
    
    def __init__(self, cache_service=None, pool_manager=None):
        self._cache_service = cache_service
        self._pool_manager = pool_manager
        self._service_checkers: Dict[str, Callable] = {}
        self._service_history: Dict[str, List[ServiceMetrics]] = {}
        self._circuit_breakers: Dict[str, Any] = {}
        
        # Performance optimization settings
        self._cache_ttl = {
            'basic_health': 15,  # Basic health cached for 15s
            'detailed_health': 30,  # Detailed health cached for 30s
            'service_metrics': 60,  # Service metrics cached for 60s
            'system_resources': 10   # System resources cached for 10s
        }
        
        # Health check timeouts (aggressive for zero-impact)
        self._service_timeouts = {
            'redis': 0.5,        # 500ms for Redis
            'database': 1.0,     # 1s for Database  
            'ollama': 2.0,       # 2s for Ollama
            'task_queue': 0.5,   # 500ms for Task Queue
            'agents': 1.0,       # 1s for Agents
            'vector_db': 1.0     # 1s for Vector DBs
        }
        
        # Register default service checkers
        self._register_default_checkers()
        
        logger.info("Health monitoring service initialized with zero-impact design")
    
    def _register_default_checkers(self):
        """Register default health check functions"""
        self._service_checkers.update({
            'redis': self._check_redis_health,
            'database': self._check_database_health,
            'ollama': self._check_ollama_health,
            'task_queue': self._check_task_queue_health,
            'vector_db_qdrant': self._check_qdrant_health,
            'vector_db_chromadb': self._check_chromadb_health,
            'hardware_optimizer': self._check_hardware_optimizer_health,
            'ai_orchestrator': self._check_ai_orchestrator_health,
        })
    
    def register_service_checker(self, service_name: str, checker_func: Callable):
        """Register a custom service health checker"""
        self._service_checkers[service_name] = checker_func
        logger.info(f"Registered custom health checker for service: {service_name}")
    
    def register_circuit_breaker(self, service_name: str, circuit_breaker):
        """Register circuit breaker for a service"""
        self._circuit_breakers[service_name] = circuit_breaker
        logger.info(f"Registered circuit breaker for service: {service_name}")
    
    async def get_basic_health(self) -> Dict[str, Any]:
        """
        Get basic health status (ultra-fast, <50ms target)
        Optimized for high-frequency health checks
        """
        cache_key = "health:basic:status"
        
        # Try cache first for instant response
        if self._cache_service:
            cached = await self._cache_service.get(cache_key)
            if cached:
                return cached
        
        # Perform   critical checks only
        critical_services = ['redis', 'database']
        start_time = time.time()
        
        service_statuses = {}
        overall_status = SystemStatus.HEALTHY
        
        # Check critical services with aggressive timeouts
        critical_tasks = []
        for service_name in critical_services:
            if service_name in self._service_checkers:
                task = asyncio.create_task(
                    self._check_service_with_timeout(
                        service_name, 
                        self._service_checkers[service_name],
                        timeout=self._service_timeouts.get(service_name, 1.0)
                    )
                )
                critical_tasks.append((service_name, task))
        
        # Wait for critical services (max 2s total)
        try:
            done, pending = await asyncio.wait(
                [task for _, task in critical_tasks],
                timeout=2.0,
                return_when=asyncio.ALL_COMPLETED
            )
            
            for service_name, task in critical_tasks:
                if task in done:
                    try:
                        metrics = await task
                        service_statuses[service_name] = metrics.status.value
                        if metrics.status not in [ServiceStatus.HEALTHY]:
                            overall_status = SystemStatus.DEGRADED
                    except Exception as e:
                        service_statuses[service_name] = ServiceStatus.TIMEOUT.value
                        overall_status = SystemStatus.DEGRADED
                        logger.warning(f"Critical service {service_name} check failed: {e}")
                else:
                    service_statuses[service_name] = ServiceStatus.TIMEOUT.value
                    overall_status = SystemStatus.DEGRADED
                    task.cancel()
        
        except Exception as e:
            logger.error(f"Error in basic health check: {e}")
            overall_status = SystemStatus.CRITICAL
        
        response_time = (time.time() - start_time) * 1000
        
        result = {
            "status": overall_status.value,
            "timestamp": datetime.now().isoformat(),
            "services": service_statuses,
            "response_time_ms": round(response_time, 2),
            "check_type": "basic"
        }
        
        # Cache the result
        if self._cache_service:
            await self._cache_service.set(
                cache_key, 
                result, 
                ttl=self._cache_ttl['basic_health']
            )
        
        return result
    
    async def get_detailed_health(self) -> SystemHealthReport:
        """
        Get comprehensive health report with all services
        Includes circuit breaker status, metrics, and recommendations
        """
        cache_key = "health:detailed:report" 
        
        # Try cache first
        if self._cache_service:
            cached = await self._cache_service.get(cache_key)
            if cached:
                # Reconstruct dataclass from cached dict
                return self._reconstruct_health_report(cached)
        
        start_time = time.time()
        
        # Check all registered services in parallel
        service_tasks = []
        for service_name, checker_func in self._service_checkers.items():
            task = asyncio.create_task(
                self._check_service_with_timeout(
                    service_name, 
                    checker_func,
                    timeout=self._service_timeouts.get(service_name, 2.0)
                )
            )
            service_tasks.append((service_name, task))
        
        # Gather results with timeout
        service_metrics = {}
        try:
            done, pending = await asyncio.wait(
                [task for _, task in service_tasks],
                timeout=10.0,  # Max 10s for detailed check
                return_when=asyncio.ALL_COMPLETED
            )
            
            for service_name, task in service_tasks:
                if task in done:
                    try:
                        metrics = await task
                        service_metrics[service_name] = metrics
                    except Exception as e:
                        service_metrics[service_name] = self._create_error_metrics(
                            service_name, str(e)
                        )
                        logger.error(f"Service {service_name} check failed: {e}")
                else:
                    service_metrics[service_name] = self._create_timeout_metrics(service_name)
                    task.cancel()
        
        except Exception as e:
            logger.error(f"Error in detailed health check: {e}")
        
        # Determine overall system status
        overall_status = self._calculate_overall_status(service_metrics)
        
        # Get system resource metrics
        system_resources = await self._get_system_resources()
        
        # Get performance metrics
        performance_metrics = await self._get_performance_metrics()
        performance_metrics["health_check_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        # Generate alerts and recommendations
        alerts = self._generate_alerts(service_metrics, system_resources)
        recommendations = self._generate_recommendations(service_metrics, system_resources)
        
        # Create comprehensive report
        report = SystemHealthReport(
            overall_status=overall_status,
            timestamp=datetime.now(),
            services=service_metrics,
            performance_metrics=performance_metrics,
            system_resources=system_resources,
            alerts=alerts,
            recommendations=recommendations
        )
        
        # Cache the result (serialize for caching)
        if self._cache_service:
            cached_data = self._serialize_health_report(report)
            await self._cache_service.set(
                cache_key,
                cached_data,
                ttl=self._cache_ttl['detailed_health']
            )
        
        # Store in history for trending
        await self._update_service_history(service_metrics)
        
        return report
    
    async def _check_service_with_timeout(self, service_name: str, checker_func: Callable, timeout: float) -> ServiceMetrics:
        """Execute service health check with timeout and circuit breaker integration"""
        start_time = time.time()
        
        try:
            # Check if circuit breaker exists for this service
            circuit_breaker = self._circuit_breakers.get(service_name)
            
            if circuit_breaker:
                # Use circuit breaker
                try:
                    if hasattr(circuit_breaker, 'is_healthy') and not circuit_breaker.is_healthy:
                        # Circuit is open
                        return ServiceMetrics(
                            name=service_name,
                            status=ServiceStatus.CIRCUIT_OPEN,
                            response_time_ms=0.0,
                            last_check=datetime.now(),
                            error_message=f"Circuit breaker is {circuit_breaker.state.value}",
                            circuit_breaker_state=circuit_breaker.state.value,
                            circuit_breaker_failures=getattr(circuit_breaker, 'consecutive_failures', 0)
                        )
                    
                    # Execute through circuit breaker
                    result = await circuit_breaker.call(checker_func)
                    response_time = (time.time() - start_time) * 1000
                    
                    metrics = ServiceMetrics(
                        name=service_name,
                        status=ServiceStatus.HEALTHY,
                        response_time_ms=round(response_time, 2),
                        last_check=datetime.now(),
                        last_success=datetime.now(),
                        circuit_breaker_state=circuit_breaker.state.value if hasattr(circuit_breaker, 'state') else 'closed'
                    )
                    
                    # Add circuit breaker stats if available
                    if hasattr(circuit_breaker, 'get_stats'):
                        cb_stats = circuit_breaker.get_stats()
                        metrics.custom_metrics['circuit_breaker'] = cb_stats
                    
                    return metrics
                    
                except Exception as cb_error:
                    response_time = (time.time() - start_time) * 1000
                    return ServiceMetrics(
                        name=service_name,
                        status=ServiceStatus.UNHEALTHY,
                        response_time_ms=round(response_time, 2),
                        last_check=datetime.now(),
                        last_failure=datetime.now(),
                        error_message=str(cb_error),
                        circuit_breaker_state=circuit_breaker.state.value if hasattr(circuit_breaker, 'state') else 'unknown'
                    )
            
            else:
                # No circuit breaker - direct execution with timeout
                result = await asyncio.wait_for(checker_func(), timeout=timeout)
                response_time = (time.time() - start_time) * 1000
                
                return ServiceMetrics(
                    name=service_name,
                    status=ServiceStatus.HEALTHY,
                    response_time_ms=round(response_time, 2),
                    last_check=datetime.now(),
                    last_success=datetime.now()
                )
        
        except asyncio.TimeoutError:
            response_time = (time.time() - start_time) * 1000
            return ServiceMetrics(
                name=service_name,
                status=ServiceStatus.TIMEOUT,
                response_time_ms=round(response_time, 2),
                last_check=datetime.now(),
                last_failure=datetime.now(),
                error_message=f"Health check timed out after {timeout}s"
            )
        
        except Exception as e:
            response_time = (time.time() - start_time) * 1000
            return ServiceMetrics(
                name=service_name,
                status=ServiceStatus.UNHEALTHY,
                response_time_ms=round(response_time, 2),
                last_check=datetime.now(),
                last_failure=datetime.now(),
                error_message=str(e)
            )
    
    # Service-specific health check methods
    async def _check_redis_health(self) -> bool:
        """Check Redis health"""
        if self._pool_manager:
            from app.core.connection_pool import get_redis
            redis_client = await get_redis()
            await redis_client.ping()
            return True
        return False
    
    async def _check_database_health(self) -> bool:
        """Check PostgreSQL health"""
        if self._pool_manager:
            await self._pool_manager.execute_db_query("SELECT 1", fetch_one=True)
            return True
        return False
    
    async def _check_ollama_health(self) -> bool:
        """Check Ollama health"""
        try:
            # Use HTTP client to check Ollama API
            if self._pool_manager:
                async with await self._pool_manager.get_http_client('ollama') as client:
                    response = await client.get("http://sutazai-ollama:11434/api/tags")
                    return response.status_code == 200
        except Exception:
            return False
        return False
    
    async def _check_task_queue_health(self) -> bool:
        """Check task queue health"""
        try:
            from app.core.task_queue import get_task_queue
            task_queue = await get_task_queue()
            return task_queue is not None
        except Exception:
            return False
    
    async def _check_qdrant_health(self) -> bool:
        """Check Qdrant vector database health"""
        try:
            if self._pool_manager:
                async with await self._pool_manager.get_http_client('vector_db') as client:
                    response = await client.get("http://sutazai-qdrant:6333/health")
                    return response.status_code == 200
        except Exception:
            return False
    
    async def _check_chromadb_health(self) -> bool:
        """Check ChromaDB health using v2 API"""
        try:
            if self._pool_manager:
                headers = {"X-Chroma-Token": "sk-dcebf71d6136dafc1405f3d3b6f7a9ce43723e36f93542fb"}
                async with await self._pool_manager.get_http_client('vector_db') as client:
                    response = await client.get("http://sutazai-chromadb:8000/api/v2/heartbeat", headers=headers)
                    return response.status_code == 200
        except Exception:
            return False
    
    async def _check_hardware_optimizer_health(self) -> bool:
        """Check Hardware Resource Optimizer health"""
        try:
            if self._pool_manager:
                async with await self._pool_manager.get_http_client('agents') as client:
                    response = await client.get("http://sutazai-hardware-resource-optimizer:8080/health")
                    return response.status_code == 200
        except Exception:
            return False
    
    async def _check_ai_orchestrator_health(self) -> bool:
        """Check AI Agent Orchestrator health"""
        try:
            if self._pool_manager:
                async with await self._pool_manager.get_http_client('agents') as client:
                    response = await client.get("http://sutazai-ai-agent-orchestrator:8589/health")
                    return response.status_code == 200
        except Exception:
            return False
    
    def _create_error_metrics(self, service_name: str, error_message: str) -> ServiceMetrics:
        """Create metrics for failed service check"""
        return ServiceMetrics(
            name=service_name,
            status=ServiceStatus.UNHEALTHY,
            response_time_ms=0.0,
            last_check=datetime.now(),
            last_failure=datetime.now(),
            error_message=error_message
        )
    
    def _create_timeout_metrics(self, service_name: str) -> ServiceMetrics:
        """Create metrics for timed out service check"""
        return ServiceMetrics(
            name=service_name,
            status=ServiceStatus.TIMEOUT,
            response_time_ms=self._service_timeouts.get(service_name, 2.0) * 1000,
            last_check=datetime.now(),
            last_failure=datetime.now(),
            error_message=f"Health check timed out after {self._service_timeouts.get(service_name, 2.0)}s"
        )
    
    def _calculate_overall_status(self, service_metrics: Dict[str, ServiceMetrics]) -> SystemStatus:
        """Calculate overall system status from service metrics"""
        if not service_metrics:
            return SystemStatus.UNKNOWN
        
        critical_services = ['redis', 'database']
        critical_unhealthy = 0
        total_unhealthy = 0
        
        for service_name, metrics in service_metrics.items():
            if metrics.status in [ServiceStatus.UNHEALTHY, ServiceStatus.TIMEOUT, ServiceStatus.CIRCUIT_OPEN]:
                total_unhealthy += 1
                if service_name in critical_services:
                    critical_unhealthy += 1
        
        # If any critical service is down, system is critical
        if critical_unhealthy > 0:
            return SystemStatus.CRITICAL
        
        # If more than 50% of services are unhealthy, system is degraded
        if total_unhealthy > len(service_metrics) * 0.5:
            return SystemStatus.DEGRADED
        
        # If any service is unhealthy but not critical, system is degraded
        if total_unhealthy > 0:
            return SystemStatus.DEGRADED
        
        return SystemStatus.HEALTHY
    
    async def _get_system_resources(self) -> Dict[str, Any]:
        """Get system resource metrics with caching"""
        cache_key = "health:system_resources"
        
        if self._cache_service:
            cached = await self._cache_service.get(cache_key)
            if cached:
                return cached
        
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            resources = {
                "cpu": {
                    "usage_percent": round(cpu_percent, 1),
                    "load_average": list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else None
                },
                "memory": {
                    "usage_percent": round(memory.percent, 1),
                    "available_mb": round(memory.available / 1024 / 1024, 1),
                    "used_mb": round(memory.used / 1024 / 1024, 1),
                    "total_mb": round(memory.total / 1024 / 1024, 1)
                },
                "disk": {
                    "usage_percent": round(disk.percent, 1),
                    "free_gb": round(disk.free / 1024 / 1024 / 1024, 1),
                    "used_gb": round(disk.used / 1024 / 1024 / 1024, 1),
                    "total_gb": round(disk.total / 1024 / 1024 / 1024, 1)
                }
            }
            
            # Cache the result
            if self._cache_service:
                await self._cache_service.set(
                    cache_key,
                    resources,
                    ttl=self._cache_ttl['system_resources']
                )
            
            return resources
            
        except Exception as e:
            logger.error(f"Error getting system resources: {e}")
            return {"error": str(e)}
    
    async def _get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics"""
        metrics = {}
        
        try:
            # Cache performance metrics
            if self._cache_service:
                cache_stats = self._cache_service.get_stats()
                metrics["cache"] = cache_stats
            
            # Connection pool metrics
            if self._pool_manager:
                pool_stats = self._pool_manager.get_stats()
                metrics["connection_pools"] = pool_stats
            
            # Circuit breaker metrics
            if self._circuit_breakers:
                cb_metrics = {}
                for name, breaker in self._circuit_breakers.items():
                    if hasattr(breaker, 'get_stats'):
                        cb_metrics[name] = breaker.get_stats()
                if cb_metrics:
                    metrics["circuit_breakers"] = cb_metrics
        
        except Exception as e:
            logger.error(f"Error getting performance metrics: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    def _generate_alerts(self, service_metrics: Dict[str, ServiceMetrics], system_resources: Dict[str, Any]) -> List[str]:
        """Generate alerts based on current status"""
        alerts = []
        
        # Service alerts
        for service_name, metrics in service_metrics.items():
            if metrics.status == ServiceStatus.UNHEALTHY:
                alerts.append(f"CRITICAL: Service '{service_name}' is unhealthy - {metrics.error_message}")
            elif metrics.status == ServiceStatus.TIMEOUT:
                alerts.append(f"WARNING: Service '{service_name}' is not responding within timeout")
            elif metrics.status == ServiceStatus.CIRCUIT_OPEN:
                alerts.append(f"WARNING: Circuit breaker for '{service_name}' is open")
            elif metrics.response_time_ms > 5000:  # 5 second threshold
                alerts.append(f"WARNING: Service '{service_name}' has high response time: {metrics.response_time_ms}ms")
        
        # System resource alerts
        if system_resources and "error" not in system_resources:
            cpu_usage = system_resources.get("cpu", {}).get("usage_percent", 0)
            memory_usage = system_resources.get("memory", {}).get("usage_percent", 0)
            disk_usage = system_resources.get("disk", {}).get("usage_percent", 0)
            
            if cpu_usage > 90:
                alerts.append(f"CRITICAL: High CPU usage: {cpu_usage}%")
            elif cpu_usage > 75:
                alerts.append(f"WARNING: Elevated CPU usage: {cpu_usage}%")
            
            if memory_usage > 95:
                alerts.append(f"CRITICAL: High memory usage: {memory_usage}%")
            elif memory_usage > 85:
                alerts.append(f"WARNING: Elevated memory usage: {memory_usage}%")
            
            if disk_usage > 95:
                alerts.append(f"CRITICAL: Disk space critically low: {disk_usage}%")
            elif disk_usage > 85:
                alerts.append(f"WARNING: Disk space running low: {disk_usage}%")
        
        return alerts
    
    def _generate_recommendations(self, service_metrics: Dict[str, ServiceMetrics], system_resources: Dict[str, Any]) -> List[str]:
        """Generate recommendations for system optimization"""
        recommendations = []
        
        # Service recommendations
        failing_services = [name for name, metrics in service_metrics.items() 
                          if metrics.status in [ServiceStatus.UNHEALTHY, ServiceStatus.TIMEOUT]]
        
        if failing_services:
            recommendations.append(f"Consider restarting or investigating these services: {', '.join(failing_services)}")
        
        slow_services = [name for name, metrics in service_metrics.items() 
                        if metrics.response_time_ms > 2000]
        
        if slow_services:
            recommendations.append(f"Optimize performance for slow services: {', '.join(slow_services)}")
        
        # Resource recommendations
        if system_resources and "error" not in system_resources:
            memory_usage = system_resources.get("memory", {}).get("usage_percent", 0)
            cpu_usage = system_resources.get("cpu", {}).get("usage_percent", 0)
            
            if memory_usage > 80:
                recommendations.append("Consider increasing available memory or optimizing memory usage")
            
            if cpu_usage > 80:
                recommendations.append("Consider scaling horizontally or optimizing CPU-intensive operations")
        
        return recommendations
    
    def _serialize_health_report(self, report: SystemHealthReport) -> Dict[str, Any]:
        """Serialize health report for caching"""
        return {
            "overall_status": report.overall_status.value,
            "timestamp": report.timestamp.isoformat(),
            "services": {
                name: {
                    "name": metrics.name,
                    "status": metrics.status.value,
                    "response_time_ms": metrics.response_time_ms,
                    "last_check": metrics.last_check.isoformat() if metrics.last_check else None,
                    "last_success": metrics.last_success.isoformat() if metrics.last_success else None,
                    "last_failure": metrics.last_failure.isoformat() if metrics.last_failure else None,
                    "error_message": metrics.error_message,
                    "consecutive_failures": metrics.consecutive_failures,
                    "total_checks": metrics.total_checks,
                    "success_rate": metrics.success_rate,
                    "circuit_breaker_state": metrics.circuit_breaker_state,
                    "circuit_breaker_failures": metrics.circuit_breaker_failures,
                    "uptime_percentage": metrics.uptime_percentage,
                    "custom_metrics": metrics.custom_metrics
                }
                for name, metrics in report.services.items()
            },
            "performance_metrics": report.performance_metrics,
            "system_resources": report.system_resources,
            "alerts": report.alerts,
            "recommendations": report.recommendations
        }
    
    def _reconstruct_health_report(self, cached_data: Dict[str, Any]) -> SystemHealthReport:
        """Reconstruct health report from cached data"""
        services = {}
        for name, data in cached_data.get("services", {}).items():
            services[name] = ServiceMetrics(
                name=data["name"],
                status=ServiceStatus(data["status"]),
                response_time_ms=data["response_time_ms"],
                last_check=datetime.fromisoformat(data["last_check"]) if data["last_check"] else None,
                last_success=datetime.fromisoformat(data["last_success"]) if data["last_success"] else None,
                last_failure=datetime.fromisoformat(data["last_failure"]) if data["last_failure"] else None,
                error_message=data["error_message"],
                consecutive_failures=data["consecutive_failures"],
                total_checks=data["total_checks"],
                success_rate=data["success_rate"],
                circuit_breaker_state=data["circuit_breaker_state"],
                circuit_breaker_failures=data["circuit_breaker_failures"],
                uptime_percentage=data["uptime_percentage"],
                custom_metrics=data["custom_metrics"]
            )
        
        return SystemHealthReport(
            overall_status=SystemStatus(cached_data["overall_status"]),
            timestamp=datetime.fromisoformat(cached_data["timestamp"]),
            services=services,
            performance_metrics=cached_data["performance_metrics"],
            system_resources=cached_data["system_resources"],
            alerts=cached_data["alerts"],
            recommendations=cached_data["recommendations"]
        )
    
    async def _update_service_history(self, service_metrics: Dict[str, ServiceMetrics]):
        """Update service history for trending analysis"""
        try:
            for service_name, metrics in service_metrics.items():
                if service_name not in self._service_history:
                    self._service_history[service_name] = []
                
                # Keep last 100 entries per service
                history = self._service_history[service_name]
                history.append(metrics)
                
                if len(history) > 100:
                    history.pop(0)  # Remove oldest entry
                    
        except Exception as e:
            logger.error(f"Error updating service history: {e}")
    
    async def get_prometheus_metrics(self) -> str:
        """Generate Prometheus metrics format"""
        try:
            report = await self.get_detailed_health()
            
            metrics_lines = [
                "# HELP sutazai_service_health Service health status (1=healthy, 0=unhealthy)",
                "# TYPE sutazai_service_health gauge"
            ]
            
            for service_name, metrics in report.services.items():
                health_value = 1 if metrics.status == ServiceStatus.HEALTHY else 0
                metrics_lines.append(
                    f'sutazai_service_health{{service="{service_name}",status="{metrics.status.value}"}} {health_value}'
                )
                
                # Add response time metrics
                metrics_lines.extend([
                    f'sutazai_service_response_time_ms{{service="{service_name}"}} {metrics.response_time_ms}',
                ])
                
                # Add circuit breaker metrics if available
                if metrics.circuit_breaker_state:
                    cb_health = 1 if metrics.circuit_breaker_state == "closed" else 0
                    metrics_lines.append(
                        f'sutazai_circuit_breaker_health{{service="{service_name}",state="{metrics.circuit_breaker_state}"}} {cb_health}'
                    )
            
            # Add system status
            system_health = 1 if report.overall_status == SystemStatus.HEALTHY else 0
            metrics_lines.extend([
                "",
                "# HELP sutazai_system_health Overall system health status (1=healthy, 0=unhealthy)",
                "# TYPE sutazai_system_health gauge",
                f'sutazai_system_health{{status="{report.overall_status.value}"}} {system_health}'
            ])
            
            # Add system resource metrics
            if report.system_resources and "error" not in report.system_resources:
                metrics_lines.extend([
                    "",
                    "# HELP sutazai_cpu_usage_percent CPU usage percentage",
                    "# TYPE sutazai_cpu_usage_percent gauge",
                    f"sutazai_cpu_usage_percent {report.system_resources['cpu']['usage_percent']}",
                    "",
                    "# HELP sutazai_memory_usage_percent Memory usage percentage", 
                    "# TYPE sutazai_memory_usage_percent gauge",
                    f"sutazai_memory_usage_percent {report.system_resources['memory']['usage_percent']}",
                    "",
                    "# HELP sutazai_disk_usage_percent Disk usage percentage",
                    "# TYPE sutazai_disk_usage_percent gauge",
                    f"sutazai_disk_usage_percent {report.system_resources['disk']['usage_percent']}"
                ])
            
            return "\n".join(metrics_lines)
            
        except Exception as e:
            logger.error(f"Error generating Prometheus metrics: {e}")
            return f"# Error generating metrics: {str(e)}"


# Global health monitoring service instance
_health_monitor = None
_health_monitor_lock = asyncio.Lock()


async def get_health_monitoring_service(cache_service=None, pool_manager=None) -> HealthMonitoringService:
    """Get global health monitoring service instance"""
    global _health_monitor
    
    async with _health_monitor_lock:
        if _health_monitor is None:
            _health_monitor = HealthMonitoringService(cache_service, pool_manager)
        
        # Update services if provided
        if cache_service:
            _health_monitor._cache_service = cache_service
        if pool_manager:
            _health_monitor._pool_manager = pool_manager
        
        return _health_monitor