#!/usr/bin/env python3
"""
Service Monitor with Circuit Breaker Integration for SutazAI

This module monitors all SutazAI services and applies circuit breaker patterns
to prevent cascading failures across the AI agent ecosystem.

Author: SutazAI Infrastructure Team
Version: 1.0.0
"""

import asyncio
import logging
import json
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import aiohttp
import docker
import redis
from circuit_breaker import (
    CircuitBreaker, HttpCircuitBreaker, CircuitConfig, 
    CircuitBreakerRegistry, CircuitBreakerOpenError, CircuitState
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ServiceEndpoint:
    """Service endpoint configuration"""
    name: str
    url: str
    health_path: str = "/health"
    timeout: float = 30.0
    critical: bool = True
    dependencies: List[str] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

class SutazAIServiceMonitor:
    """
    Comprehensive service monitor for SutazAI with circuit breaker protection
    """
    
    def __init__(self, redis_client: Optional[redis.Redis] = None):
        self.redis_client = redis_client or self._create_redis_client()
        self.circuit_registry = CircuitBreakerRegistry(self.redis_client)
        self.docker_client = docker.from_env()
        
        # Service configurations
        self.services = self._initialize_service_configs()
        
        # Monitoring state
        self.monitoring_active = False
        self.last_health_check = {}
        
    def _create_redis_client(self) -> redis.Redis:
        """Create Redis client with fallback"""
        try:
            client = redis.Redis(
                host='localhost',
                port=6379,
                decode_responses=True,
                socket_connect_timeout=5
            )
            client.ping()
            return client
        except Exception as e:
            logger.warning(f"Redis not available: {e}")
            return None
    
    def _initialize_service_configs(self) -> Dict[str, ServiceEndpoint]:
        """Initialize service endpoint configurations"""
        return {
            # Core Infrastructure
            "redis": ServiceEndpoint(
                name="redis",
                url="redis://localhost:6379",
                health_path="/ping",
                critical=True
            ),
            "postgres": ServiceEndpoint(
                name="postgres", 
                url="postgresql://localhost:5432",
                health_path="/",
                critical=True
            ),
            
            # AI Infrastructure
            "ollama": ServiceEndpoint(
                name="ollama",
                url="http://localhost:11434",
                health_path="/api/tags",
                critical=True,
                timeout=60.0
            ),
            
            # Vector Stores
            "chromadb": ServiceEndpoint(
                name="chromadb",
                url="http://localhost:8000",
                health_path="/api/v1/heartbeat",
                critical=True
            ),
            "qdrant": ServiceEndpoint(
                name="qdrant",
                url="http://localhost:6333",
                health_path="/health",
                critical=True
            ),
            
            # SutazAI Core
            "backend": ServiceEndpoint(
                name="backend",
                url="http://localhost:8000",
                health_path="/health",
                critical=True,
                dependencies=["redis", "postgres", "ollama"]
            ),
            "frontend": ServiceEndpoint(
                name="frontend",
                url="http://localhost:8501",
                health_path="/_stcore/health",
                critical=True,
                dependencies=["backend"]
            ),
            
            # AI Agents
            "letta": ServiceEndpoint(
                name="letta",
                url="http://localhost:8010",
                health_path="/health",
                critical=False,
                dependencies=["ollama", "postgres"]
            ),
            "autogpt": ServiceEndpoint(
                name="autogpt",
                url="http://localhost:8020",
                health_path="/health",
                critical=False,
                dependencies=["ollama", "redis"]
            ),
            "localagi": ServiceEndpoint(
                name="localagi",
                url="http://localhost:8030",
                health_path="/health",
                critical=False,
                dependencies=["ollama"]
            ),
            "langchain": ServiceEndpoint(
                name="langchain",
                url="http://localhost:8040",
                health_path="/health",
                critical=False,
                dependencies=["ollama", "chromadb"]
            ),
            "crewai": ServiceEndpoint(
                name="crewai",
                url="http://localhost:8050",
                health_path="/health",
                critical=False,
                dependencies=["ollama", "redis"]
            ),
            
            # Workflow Engines
            "n8n": ServiceEndpoint(
                name="n8n",
                url="http://localhost:5678",
                health_path="/healthz",
                critical=False,
                dependencies=["postgres"]
            ),
            "dify": ServiceEndpoint(
                name="dify",
                url="http://localhost:3000",
                health_path="/health",
                critical=False,
                dependencies=["postgres", "redis"]
            ),
            
            # Monitoring
            "prometheus": ServiceEndpoint(
                name="prometheus",
                url="http://localhost:9090",
                health_path="/-/healthy",
                critical=False
            ),
            "grafana": ServiceEndpoint(
                name="grafana",
                url="http://localhost:3001",
                health_path="/api/health",
                critical=False,
                dependencies=["prometheus"]
            )
        }
    
    async def check_service_health(self, service: ServiceEndpoint) -> Dict[str, Any]:
        """
        Check health of a single service with circuit breaker protection
        """
        circuit_config = CircuitConfig(
            failure_threshold=3,
            recovery_timeout=60,
            success_threshold=2,
            timeout=service.timeout
        )
        
        health_status = {
            'service': service.name,
            'status': 'unknown',
            'response_time': None,
            'error': None,
            'circuit_breaker_state': 'unknown',
            'timestamp': time.time()
        }
        
        try:
            # Get circuit breaker for this service
            if service.url.startswith('http'):
                circuit_breaker = self.circuit_registry.get_http_circuit_breaker(
                    service.name,
                    service.url,
                    circuit_config
                )
                
                async with circuit_breaker:
                    start_time = time.time()
                    response = await circuit_breaker.get(service.health_path)
                    response_time = time.time() - start_time
                    
                    if response.status == 200:
                        health_status.update({
                            'status': 'healthy',
                            'response_time': response_time,
                            'circuit_breaker_state': circuit_breaker.state.value
                        })
                    else:
                        health_status.update({
                            'status': 'unhealthy',
                            'response_time': response_time,
                            'error': f"HTTP {response.status}",
                            'circuit_breaker_state': circuit_breaker.state.value
                        })
            else:
                # Handle non-HTTP services (Redis, Postgres)
                circuit_breaker = self.circuit_registry.get_circuit_breaker(
                    service.name,
                    circuit_config
                )
                
                async def health_check():
                    if service.name == "redis":
                        return await self._check_redis_health()
                    elif service.name == "postgres":
                        return await self._check_postgres_health()
                    else:
                        raise Exception(f"Unknown service type: {service.name}")
                
                start_time = time.time()
                result = await circuit_breaker.call(health_check)
                response_time = time.time() - start_time
                
                health_status.update({
                    'status': 'healthy' if result else 'unhealthy',
                    'response_time': response_time,
                    'circuit_breaker_state': circuit_breaker.state.value
                })
                
        except CircuitBreakerOpenError:
            health_status.update({
                'status': 'circuit_open',
                'error': 'Circuit breaker is open',
                'circuit_breaker_state': 'open'
            })
            
        except Exception as e:
            health_status.update({
                'status': 'error',
                'error': str(e),
                'circuit_breaker_state': 'unknown'
            })
        
        # Store health status
        self.last_health_check[service.name] = health_status
        
        # Store in Redis if available
        if self.redis_client:
            try:
                key = f"health:{service.name}"
                self.redis_client.setex(key, 300, json.dumps(health_status))
            except Exception as e:
                logger.warning(f"Failed to store health status in Redis: {e}")
        
        return health_status
    
    async def _check_redis_health(self) -> bool:
        """Check Redis health"""
        try:
            if self.redis_client:
                self.redis_client.ping()
                return True
            return False
        except Exception:
            return False
    
    async def _check_postgres_health(self) -> bool:
        """Check PostgreSQL health"""
        try:
            # This would require psycopg2 or asyncpg
            # For now, check if container is running
            containers = self.docker_client.containers.list()
            for container in containers:
                if 'postgres' in container.name and container.status == 'running':
                    return True
            return False
        except Exception:
            return False
    
    async def check_all_services(self) -> Dict[str, Dict[str, Any]]:
        """Check health of all services concurrently"""
        tasks = []
        
        for service in self.services.values():
            task = asyncio.create_task(
                self.check_service_health(service),
                name=f"health_check_{service.name}"
            )
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        health_report = {}
        for i, result in enumerate(results):
            service_name = list(self.services.keys())[i]
            
            if isinstance(result, Exception):
                health_report[service_name] = {
                    'service': service_name,
                    'status': 'error',
                    'error': str(result),
                    'timestamp': time.time()
                }
            else:
                health_report[service_name] = result
        
        return health_report
    
    async def get_service_dependencies(self, service_name: str) -> List[str]:
        """Get dependencies for a service"""
        if service_name in self.services:
            return self.services[service_name].dependencies
        return []
    
    async def check_dependency_chain(self, service_name: str) -> Dict[str, Any]:
        """Check entire dependency chain for a service"""
        dependencies = await self.get_service_dependencies(service_name)
        
        dependency_status = {}
        for dep in dependencies:
            if dep in self.services:
                status = await self.check_service_health(self.services[dep])
                dependency_status[dep] = status
                
                # Recursively check dependencies
                sub_deps = await self.check_dependency_chain(dep)
                if sub_deps:
                    dependency_status[dep]['dependencies'] = sub_deps
        
        return dependency_status
    
    async def get_critical_services_status(self) -> Dict[str, str]:
        """Get status of critical services only"""
        critical_services = {
            name: service for name, service in self.services.items()
            if service.critical
        }
        
        status_summary = {}
        for name, service in critical_services.items():
            if name in self.last_health_check:
                status_summary[name] = self.last_health_check[name]['status']
            else:
                health = await self.check_service_health(service)
                status_summary[name] = health['status']
        
        return status_summary
    
    async def start_continuous_monitoring(self, interval: int = 30):
        """Start continuous health monitoring"""
        self.monitoring_active = True
        logger.info(f"Starting continuous monitoring with {interval}s interval")
        
        while self.monitoring_active:
            try:
                health_report = await self.check_all_services()
                
                # Log critical service issues
                for service_name, health in health_report.items():
                    service = self.services.get(service_name)
                    if service and service.critical and health['status'] != 'healthy':
                        logger.warning(
                            f"Critical service {service_name} is {health['status']}: "
                            f"{health.get('error', 'Unknown error')}"
                        )
                
                # Store aggregated health report
                if self.redis_client:
                    try:
                        self.redis_client.setex(
                            "health:all_services",
                            300,
                            json.dumps(health_report)
                        )
                    except Exception as e:
                        logger.warning(f"Failed to store health report: {e}")
                
            except Exception as e:
                logger.error(f"Error in continuous monitoring: {e}")
            
            await asyncio.sleep(interval)
    
    def stop_monitoring(self):
        """Stop continuous monitoring"""
        self.monitoring_active = False
        logger.info("Stopped continuous monitoring")
    
    def get_circuit_breaker_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get circuit breaker statistics for all services"""
        return self.circuit_registry.get_all_stats()
    
    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers (emergency recovery)"""
        self.circuit_registry.reset_all()
        logger.info("All circuit breakers have been reset")

class ServiceHealthAPI:
    """REST API for service health and circuit breaker management"""
    
    def __init__(self, monitor: SutazAIServiceMonitor):
        self.monitor = monitor
    
    async def health_summary(self) -> Dict[str, Any]:
        """Get health summary"""
        critical_status = await self.monitor.get_critical_services_status()
        
        overall_health = "healthy"
        if any(status != "healthy" for status in critical_status.values()):
            overall_health = "degraded"
        if all(status in ["error", "circuit_open"] for status in critical_status.values()):
            overall_health = "critical"
        
        return {
            "overall_health": overall_health,
            "critical_services": critical_status,
            "circuit_breaker_stats": self.monitor.get_circuit_breaker_stats(),
            "timestamp": time.time()
        }
    
    async def detailed_health(self) -> Dict[str, Any]:
        """Get detailed health report"""
        return await self.monitor.check_all_services()
    
    async def service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health for specific service"""
        if service_name not in self.monitor.services:
            return {"error": f"Service {service_name} not found"}
        
        return await self.monitor.check_service_health(
            self.monitor.services[service_name]
        )
    
    async def reset_circuit_breaker(self, service_name: str) -> Dict[str, str]:
        """Reset circuit breaker for specific service"""
        try:
            breaker = self.monitor.circuit_registry.get_circuit_breaker(service_name)
            with breaker._lock:
                breaker._transition_to(CircuitState.CLOSED)
                breaker._failure_count = 0
                breaker._success_count = 0
            
            return {"status": "success", "message": f"Circuit breaker reset for {service_name}"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

# Example usage
async def main():
    """Example usage of the service monitor"""
    monitor = SutazAIServiceMonitor()
    api = ServiceHealthAPI(monitor)
    
    # Check all services once
    health_report = await monitor.check_all_services()
    print("Health Report:")
    print(json.dumps(health_report, indent=2))
    
    # Get health summary
    summary = await api.health_summary()
    print("\nHealth Summary:")
    print(json.dumps(summary, indent=2))
    
    # Start continuous monitoring (in background)
    # monitor_task = asyncio.create_task(monitor.start_continuous_monitoring(30))
    
    # Stop monitoring
    # monitor.stop_monitoring()

if __name__ == "__main__":
    asyncio.run(main())