#!/usr/bin/env python3
"""
MCP Automation Health Monitor
Comprehensive health checking and monitoring service for MCP infrastructure
"""

import asyncio
import json
import logging
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import httpx
import psutil
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


class ComponentType(Enum):
    """Types of components to monitor"""
    MCP_SERVER = "mcp_server"
    AUTOMATION_SERVICE = "automation_service"
    DATABASE = "database"
    CACHE = "cache"
    MESSAGE_QUEUE = "message_queue"
    API_ENDPOINT = "api_endpoint"
    CONTAINER = "container"
    SYSTEM_RESOURCE = "system_resource"


@dataclass
class HealthCheck:
    """Individual health check result"""
    name: str
    component_type: ComponentType
    status: HealthStatus
    message: str
    latency_ms: float
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class ComponentHealth:
    """Health status of a system component"""
    name: str
    type: ComponentType
    overall_status: HealthStatus
    checks: List[HealthCheck] = field(default_factory=list)
    uptime_seconds: float = 0
    last_healthy: Optional[datetime] = None
    consecutive_failures: int = 0
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SystemHealth:
    """Overall system health status"""
    status: HealthStatus
    healthy_components: int
    degraded_components: int
    unhealthy_components: int
    total_components: int
    components: Dict[str, ComponentHealth] = field(default_factory=dict)
    alerts: List[Dict[str, Any]] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)


class MCPHealthMonitor:
    """Comprehensive health monitoring for MCP automation infrastructure"""
    
    def __init__(self,
                 config_path: Optional[str] = None,
                 check_interval: int = 30,
                 alert_threshold: int = 3):
        """
        Initialize health monitor
        
        Args:
            config_path: Path to health check configuration
            check_interval: Interval between health checks in seconds
            alert_threshold: Number of failures before alerting
        """
        self.config_path = Path(config_path) if config_path else None
        self.check_interval = check_interval
        self.alert_threshold = alert_threshold
        
        # Component health tracking
        self.components: Dict[str, ComponentHealth] = {}
        self.system_health = SystemHealth(
            status=HealthStatus.UNKNOWN,
            healthy_components=0,
            degraded_components=0,
            unhealthy_components=0,
            total_components=0
        )
        
        # Load configuration
        self.config = self._load_config()
        
        # HTTP client for API checks
        self.http_client = httpx.AsyncClient(timeout=10.0)
        
        # Health check history
        self.check_history: List[SystemHealth] = []
        self.max_history_size = 100
        
    def _load_config(self) -> Dict[str, Any]:
        """Load health check configuration"""
        default_config = {
            'mcp_servers': [
                'filesystem', 'github', 'postgres', 'browser', 'docker',
                'kubernetes', 'gitlab', 'slack', 'google-drive', 'linear',
                'sentry', 'aws', 'google-cloud', 'azure', 'terraform',
                'extended-memory', 'browser-use'
            ],
            'api_endpoints': [
                {'name': 'backend', 'url': 'http://localhost:10010/health'},
                {'name': 'frontend', 'url': 'http://localhost:10011/'},
                {'name': 'prometheus', 'url': 'http://localhost:10200/-/healthy'},
                {'name': 'grafana', 'url': 'http://localhost:10201/api/health'},
                {'name': 'ollama', 'url': 'http://localhost:10104/api/version'}
            ],
            'databases': [
                {'name': 'postgres', 'type': 'postgresql', 'host': 'localhost', 'port': 10000},
                {'name': 'redis', 'type': 'redis', 'host': 'localhost', 'port': 10001},
                {'name': 'neo4j', 'type': 'neo4j', 'host': 'localhost', 'port': 10002}
            ],
            'containers': [
                'sutazai-backend', 'sutazai-frontend', 'sutazai-postgres',
                'sutazai-redis', 'sutazai-neo4j', 'sutazai-ollama'
            ],
            'system_thresholds': {
                'cpu_percent': 80,
                'memory_percent': 85,
                'disk_percent': 90,
                'network_errors': 100
            }
        }
        
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    default_config.update(loaded_config)
            except Exception as e:
                logger.error(f"Failed to load config from {self.config_path}: {e}")
                
        return default_config
        
    async def check_mcp_server(self, server_name: str) -> HealthCheck:
        """
        Check health of an MCP server
        
        Args:
            server_name: Name of the MCP server
            
        Returns:
            HealthCheck result
        """
        start_time = time.time()
        
        try:
            wrapper_path = Path(f"/opt/sutazaiapp/scripts/mcp/wrappers/{server_name}.sh")
            
            if not wrapper_path.exists():
                return HealthCheck(
                    name=server_name,
                    component_type=ComponentType.MCP_SERVER,
                    status=HealthStatus.UNKNOWN,
                    message=f"Wrapper script not found: {wrapper_path}",
                    latency_ms=0,
                    timestamp=datetime.now(),
                    error="FileNotFoundError"
                )
                
            # Run selfcheck
            result = subprocess.run(
                [str(wrapper_path), "--selfcheck"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            latency_ms = (time.time() - start_time) * 1000
            
            if result.returncode == 0:
                status = HealthStatus.HEALTHY
                message = "MCP server is responding normally"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"MCP server check failed: {result.stderr}"
                
            return HealthCheck(
                name=server_name,
                component_type=ComponentType.MCP_SERVER,
                status=status,
                message=message,
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                metadata={'stdout': result.stdout, 'stderr': result.stderr}
            )
            
        except subprocess.TimeoutExpired:
            return HealthCheck(
                name=server_name,
                component_type=ComponentType.MCP_SERVER,
                status=HealthStatus.CRITICAL,
                message="Health check timed out",
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error="TimeoutError"
            )
        except Exception as e:
            return HealthCheck(
                name=server_name,
                component_type=ComponentType.MCP_SERVER,
                status=HealthStatus.CRITICAL,
                message=f"Health check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )
            
    async def check_api_endpoint(self, endpoint: Dict[str, str]) -> HealthCheck:
        """
        Check health of an API endpoint
        
        Args:
            endpoint: Dictionary with 'name' and 'url' keys
            
        Returns:
            HealthCheck result
        """
        start_time = time.time()
        name = endpoint['name']
        url = endpoint['url']
        
        try:
            response = await self.http_client.get(url)
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                status = HealthStatus.HEALTHY
                message = f"API endpoint responding normally (status: {response.status_code})"
            elif 200 < response.status_code < 400:
                status = HealthStatus.DEGRADED
                message = f"API endpoint redirecting (status: {response.status_code})"
            else:
                status = HealthStatus.UNHEALTHY
                message = f"API endpoint error (status: {response.status_code})"
                
            return HealthCheck(
                name=name,
                component_type=ComponentType.API_ENDPOINT,
                status=status,
                message=message,
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                metadata={'status_code': response.status_code, 'url': url}
            )
            
        except httpx.TimeoutException:
            return HealthCheck(
                name=name,
                component_type=ComponentType.API_ENDPOINT,
                status=HealthStatus.CRITICAL,
                message=f"API endpoint timeout",
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error="TimeoutError",
                metadata={'url': url}
            )
        except Exception as e:
            return HealthCheck(
                name=name,
                component_type=ComponentType.API_ENDPOINT,
                status=HealthStatus.CRITICAL,
                message=f"API endpoint check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e),
                metadata={'url': url}
            )
            
    async def check_database(self, db_config: Dict[str, Any]) -> HealthCheck:
        """
        Check health of a database
        
        Args:
            db_config: Database configuration
            
        Returns:
            HealthCheck result
        """
        start_time = time.time()
        name = db_config['name']
        db_type = db_config['type']
        
        try:
            if db_type == 'postgresql':
                import psycopg2
                conn = psycopg2.connect(
                    host=db_config['host'],
                    port=db_config['port'],
                    database=db_config.get('database', 'sutazai'),
                    user=db_config.get('user', 'sutazai'),
                    password=db_config.get('password', os.getenv('POSTGRES_PASSWORD')),
                    connect_timeout=5
                )
                cursor = conn.cursor()
                cursor.execute("SELECT 1")
                cursor.close()
                conn.close()
                
            elif db_type == 'redis':
                import redis
                r = redis.Redis(
                    host=db_config['host'],
                    port=db_config['port'],
                    socket_connect_timeout=5
                )
                r.ping()
                
            elif db_type == 'neo4j':
                # Neo4j health check via HTTP API
                response = await self.http_client.get(
                    f"http://{db_config['host']}:{db_config.get('http_port', 10003)}/db/data/"
                )
                if response.status_code != 200:
                    raise Exception(f"Neo4j health check failed: {response.status_code}")
                    
            latency_ms = (time.time() - start_time) * 1000
            
            return HealthCheck(
                name=name,
                component_type=ComponentType.DATABASE,
                status=HealthStatus.HEALTHY,
                message=f"{db_type} database is healthy",
                latency_ms=latency_ms,
                timestamp=datetime.now(),
                metadata={'type': db_type}
            )
            
        except Exception as e:
            return HealthCheck(
                name=name,
                component_type=ComponentType.DATABASE,
                status=HealthStatus.CRITICAL,
                message=f"{db_type} database check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e),
                metadata={'type': db_type}
            )
            
    async def check_container(self, container_name: str) -> HealthCheck:
        """
        Check health of a Docker container
        
        Args:
            container_name: Name of the container
            
        Returns:
            HealthCheck result
        """
        start_time = time.time()
        
        try:
            import docker
            client = docker.from_env()
            
            try:
                container = client.containers.get(container_name)
                latency_ms = (time.time() - start_time) * 1000
                
                if container.status == 'running':
                    # Check if container has health check
                    if container.attrs.get('State', {}).get('Health'):
                        health_status = container.attrs['State']['Health']['Status']
                        if health_status == 'healthy':
                            status = HealthStatus.HEALTHY
                            message = "Container is running and healthy"
                        elif health_status == 'unhealthy':
                            status = HealthStatus.UNHEALTHY
                            message = "Container is running but unhealthy"
                        else:
                            status = HealthStatus.DEGRADED
                            message = f"Container health: {health_status}"
                    else:
                        status = HealthStatus.HEALTHY
                        message = "Container is running (no health check configured)"
                else:
                    status = HealthStatus.UNHEALTHY
                    message = f"Container status: {container.status}"
                    
                return HealthCheck(
                    name=container_name,
                    component_type=ComponentType.CONTAINER,
                    status=status,
                    message=message,
                    latency_ms=latency_ms,
                    timestamp=datetime.now(),
                    metadata={'container_status': container.status}
                )
                
            except docker.errors.NotFound:
                return HealthCheck(
                    name=container_name,
                    component_type=ComponentType.CONTAINER,
                    status=HealthStatus.CRITICAL,
                    message="Container not found",
                    latency_ms=(time.time() - start_time) * 1000,
                    timestamp=datetime.now(),
                    error="ContainerNotFound"
                )
                
        except Exception as e:
            return HealthCheck(
                name=container_name,
                component_type=ComponentType.CONTAINER,
                status=HealthStatus.CRITICAL,
                message=f"Container check failed: {str(e)}",
                latency_ms=(time.time() - start_time) * 1000,
                timestamp=datetime.now(),
                error=str(e)
            )
            
    async def check_system_resources(self) -> List[HealthCheck]:
        """
        Check system resource utilization
        
        Returns:
            List of HealthCheck results for system resources
        """
        checks = []
        thresholds = self.config['system_thresholds']
        
        # CPU check
        cpu_percent = psutil.cpu_percent(interval=1)
        cpu_status = HealthStatus.HEALTHY
        if cpu_percent > thresholds['cpu_percent']:
            cpu_status = HealthStatus.DEGRADED if cpu_percent < 90 else HealthStatus.CRITICAL
            
        checks.append(HealthCheck(
            name='cpu',
            component_type=ComponentType.SYSTEM_RESOURCE,
            status=cpu_status,
            message=f"CPU usage: {cpu_percent:.1f}%",
            latency_ms=0,
            timestamp=datetime.now(),
            metadata={'usage_percent': cpu_percent, 'threshold': thresholds['cpu_percent']}
        ))
        
        # Memory check
        memory = psutil.virtual_memory()
        memory_status = HealthStatus.HEALTHY
        if memory.percent > thresholds['memory_percent']:
            memory_status = HealthStatus.DEGRADED if memory.percent < 95 else HealthStatus.CRITICAL
            
        checks.append(HealthCheck(
            name='memory',
            component_type=ComponentType.SYSTEM_RESOURCE,
            status=memory_status,
            message=f"Memory usage: {memory.percent:.1f}%",
            latency_ms=0,
            timestamp=datetime.now(),
            metadata={
                'usage_percent': memory.percent,
                'available_gb': memory.available / (1024**3),
                'threshold': thresholds['memory_percent']
            }
        ))
        
        # Disk check
        for partition in psutil.disk_partitions():
            if partition.mountpoint in ['/', '/opt', '/var']:
                usage = psutil.disk_usage(partition.mountpoint)
                disk_status = HealthStatus.HEALTHY
                if usage.percent > thresholds['disk_percent']:
                    disk_status = HealthStatus.DEGRADED if usage.percent < 95 else HealthStatus.CRITICAL
                    
                checks.append(HealthCheck(
                    name=f'disk_{partition.mountpoint.replace("/", "_")}',
                    component_type=ComponentType.SYSTEM_RESOURCE,
                    status=disk_status,
                    message=f"Disk usage ({partition.mountpoint}): {usage.percent:.1f}%",
                    latency_ms=0,
                    timestamp=datetime.now(),
                    metadata={
                        'mountpoint': partition.mountpoint,
                        'usage_percent': usage.percent,
                        'free_gb': usage.free / (1024**3),
                        'threshold': thresholds['disk_percent']
                    }
                ))
                
        return checks
        
    async def perform_health_checks(self) -> SystemHealth:
        """
        Perform all configured health checks
        
        Returns:
            SystemHealth with all check results
        """
        all_checks = []
        components = {}
        
        # Check MCP servers
        mcp_tasks = []
        for server in self.config['mcp_servers']:
            mcp_tasks.append(self.check_mcp_server(server))
            
        mcp_results = await asyncio.gather(*mcp_tasks, return_exceptions=True)
        
        for result in mcp_results:
            if isinstance(result, HealthCheck):
                all_checks.append(result)
                if result.name not in components:
                    components[result.name] = ComponentHealth(
                        name=result.name,
                        type=result.component_type,
                        overall_status=result.status
                    )
                components[result.name].checks.append(result)
                
        # Check API endpoints
        api_tasks = []
        for endpoint in self.config['api_endpoints']:
            api_tasks.append(self.check_api_endpoint(endpoint))
            
        api_results = await asyncio.gather(*api_tasks, return_exceptions=True)
        
        for result in api_results:
            if isinstance(result, HealthCheck):
                all_checks.append(result)
                if result.name not in components:
                    components[result.name] = ComponentHealth(
                        name=result.name,
                        type=result.component_type,
                        overall_status=result.status
                    )
                components[result.name].checks.append(result)
                
        # Check databases
        db_tasks = []
        for db in self.config['databases']:
            db_tasks.append(self.check_database(db))
            
        db_results = await asyncio.gather(*db_tasks, return_exceptions=True)
        
        for result in db_results:
            if isinstance(result, HealthCheck):
                all_checks.append(result)
                if result.name not in components:
                    components[result.name] = ComponentHealth(
                        name=result.name,
                        type=result.component_type,
                        overall_status=result.status
                    )
                components[result.name].checks.append(result)
                
        # Check containers
        container_tasks = []
        for container in self.config['containers']:
            container_tasks.append(self.check_container(container))
            
        container_results = await asyncio.gather(*container_tasks, return_exceptions=True)
        
        for result in container_results:
            if isinstance(result, HealthCheck):
                all_checks.append(result)
                if result.name not in components:
                    components[result.name] = ComponentHealth(
                        name=result.name,
                        type=result.component_type,
                        overall_status=result.status
                    )
                components[result.name].checks.append(result)
                
        # Check system resources
        system_checks = await self.check_system_resources()
        for check in system_checks:
            all_checks.append(check)
            if check.name not in components:
                components[check.name] = ComponentHealth(
                    name=check.name,
                    type=check.component_type,
                    overall_status=check.status
                )
            components[check.name].checks.append(check)
            
        # Calculate overall system health
        healthy_count = sum(1 for c in components.values() if c.overall_status == HealthStatus.HEALTHY)
        degraded_count = sum(1 for c in components.values() if c.overall_status == HealthStatus.DEGRADED)
        unhealthy_count = sum(1 for c in components.values() if c.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL])
        
        if unhealthy_count > 0:
            overall_status = HealthStatus.CRITICAL if unhealthy_count > len(components) // 3 else HealthStatus.UNHEALTHY
        elif degraded_count > 0:
            overall_status = HealthStatus.DEGRADED
        else:
            overall_status = HealthStatus.HEALTHY
            
        system_health = SystemHealth(
            status=overall_status,
            healthy_components=healthy_count,
            degraded_components=degraded_count,
            unhealthy_components=unhealthy_count,
            total_components=len(components),
            components=components,
            timestamp=datetime.now()
        )
        
        # Update component tracking
        self.components = components
        self.system_health = system_health
        
        # Add to history
        self.check_history.append(system_health)
        if len(self.check_history) > self.max_history_size:
            self.check_history.pop(0)
            
        # Generate alerts if needed
        self._generate_alerts(system_health)
        
        return system_health
        
    def _generate_alerts(self, system_health: SystemHealth):
        """Generate alerts based on health status"""
        system_health.alerts.clear()
        
        for component in system_health.components.values():
            if component.overall_status in [HealthStatus.UNHEALTHY, HealthStatus.CRITICAL]:
                # Check if this is a persistent issue
                if component.consecutive_failures >= self.alert_threshold:
                    alert = {
                        'severity': 'critical' if component.overall_status == HealthStatus.CRITICAL else 'warning',
                        'component': component.name,
                        'type': component.type.value,
                        'message': f"{component.name} has been unhealthy for {component.consecutive_failures} consecutive checks",
                        'timestamp': datetime.now().isoformat(),
                        'checks': [asdict(check) for check in component.checks[-3:]]
                    }
                    system_health.alerts.append(alert)
                    logger.warning(f"Alert generated: {alert['message']}")
                    
    async def start_monitoring_loop(self):
        """Start continuous health monitoring"""
        logger.info(f"Starting health monitoring loop with {self.check_interval}s interval")
        
        while True:
            try:
                health = await self.perform_health_checks()
                logger.info(f"Health check completed: {health.status.value} "
                          f"({health.healthy_components}/{health.total_components} healthy)")
                
                # Log any critical issues
                for alert in health.alerts:
                    logger.error(f"ALERT: {alert['message']}")
                    
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
                
            await asyncio.sleep(self.check_interval)
            
    def get_health_summary(self) -> Dict[str, Any]:
        """Get current health summary"""
        return {
            'status': self.system_health.status.value,
            'healthy': self.system_health.healthy_components,
            'degraded': self.system_health.degraded_components,
            'unhealthy': self.system_health.unhealthy_components,
            'total': self.system_health.total_components,
            'timestamp': self.system_health.timestamp.isoformat(),
            'alerts': self.system_health.alerts,
            'components': {
                name: {
                    'status': comp.overall_status.value,
                    'type': comp.type.value,
                    'consecutive_failures': comp.consecutive_failures,
                    'last_check': comp.checks[-1].timestamp.isoformat() if comp.checks else None
                }
                for name, comp in self.system_health.components.items()
            }
        }
        
    async def cleanup(self):
        """Cleanup resources"""
        await self.http_client.aclose()


async def main():
    """Main function for testing"""
    monitor = MCPHealthMonitor()
    
    # Perform one health check
    health = await monitor.perform_health_checks()
    
    # Print summary
    summary = monitor.get_health_summary()
    logger.info(json.dumps(summary, indent=2))
    
    # Cleanup
    await monitor.cleanup()


if __name__ == "__main__":
    asyncio.run(main())