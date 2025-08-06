#!/usr/bin/env python3
"""
Distributed Health Monitor for 131 AI Agents with Ollama Integration
Monitors system health, detects bottlenecks, and manages degradation levels
"""

import asyncio
import json
import logging
import os
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

import aiohttp
import aioredis
import consul.aio
import psutil
from prometheus_client import Counter, Gauge, Histogram, generate_latest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Prometheus metrics
health_check_duration = Histogram(
    'health_check_duration_seconds',
    'Time spent performing health checks',
    ['component', 'check_type']
)

agent_health_status = Gauge(
    'agent_health_status',
    'Health status of agents (0=unhealthy, 1=healthy)',
    ['agent_name', 'agent_type']
)

ollama_queue_depth = Gauge(
    'ollama_queue_depth',
    'Current depth of Ollama request queue',
    ['priority']
)

system_degradation_level = Gauge(
    'system_degradation_level',
    'Current system degradation level (0-3)'
)

circuit_breaker_status = Gauge(
    'circuit_breaker_status',
    'Circuit breaker status (0=closed, 1=open, 2=half-open)',
    ['agent_name']
)

resource_utilization = Gauge(
    'resource_utilization_percent',
    'Resource utilization percentage',
    ['resource_type']
)

# Enums
class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"

class DegradationLevel(Enum):
    NORMAL = 0      # <60% load
    MINOR = 1       # 60-80% load
    MAJOR = 2       # 80-95% load
    CRITICAL = 3    # >95% load

class CircuitState(Enum):
    CLOSED = 0
    OPEN = 1
    HALF_OPEN = 2

# Data classes
@dataclass
class AgentHealth:
    name: str
    agent_type: str
    status: HealthStatus
    last_check: datetime
    response_time_ms: float
    error_count: int = 0
    circuit_state: CircuitState = CircuitState.CLOSED
    metadata: Dict = field(default_factory=dict)

@dataclass
class SystemMetrics:
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    gpu_percent: float
    gpu_memory_percent: float
    network_connections: int
    queue_depths: Dict[str, int]
    active_agents: int
    degradation_level: DegradationLevel

@dataclass
class BottleneckInfo:
    component: str
    severity: str
    description: str
    metrics: Dict
    suggested_actions: List[str]

class DistributedHealthMonitor:
    """
    Monitors health of distributed Ollama system with 131 agents
    """
    
    def __init__(self):
        self.redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379')
        self.consul_url = os.getenv('CONSUL_URL', 'http://localhost:8500')
        self.prometheus_url = os.getenv('PROMETHEUS_URL', 'http://localhost:9090')
        self.ollama_gateway_url = os.getenv('OLLAMA_GATEWAY_URL', 'http://localhost:11435')
        
        # Configuration
        self.check_interval = int(os.getenv('CHECK_INTERVAL', '30'))
        self.unhealthy_threshold = int(os.getenv('UNHEALTHY_THRESHOLD', '3'))
        self.memory_threshold = float(os.getenv('MEMORY_THRESHOLD_PERCENT', '85'))
        self.gpu_threshold = float(os.getenv('GPU_THRESHOLD_PERCENT', '90'))
        
        # State tracking
        self.agent_health: Dict[str, AgentHealth] = {}
        self.metrics_history: deque = deque(maxlen=100)
        self.bottlenecks: List[BottleneckInfo] = []
        self.current_degradation = DegradationLevel.NORMAL
        
        # Expected agents
        self.expected_opus_agents = 36
        self.expected_sonnet_agents = 95
        self.total_expected_agents = 131
        
        # Async resources
        self.redis_client: Optional[aioredis.Redis] = None
        self.consul_client: Optional[consul.aio.Consul] = None
        self.http_session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize async resources"""
        self.redis_client = await aioredis.create_redis_pool(self.redis_url)
        self.consul_client = consul.aio.Consul(
            host=self.consul_url.split('://')[1].split(':')[0],
            port=int(self.consul_url.split(':')[-1])
        )
        self.http_session = aiohttp.ClientSession()
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.redis_client:
            self.redis_client.close()
            await self.redis_client.wait_closed()
        if self.http_session:
            await self.http_session.close()
            
    async def run(self):
        """Main monitoring loop"""
        await self.initialize()
        
        try:
            while True:
                start_time = time.time()
                
                # Perform all health checks
                await asyncio.gather(
                    self.check_agent_health(),
                    self.check_ollama_health(),
                    self.check_system_resources(),
                    self.check_queue_health(),
                    self.check_network_health(),
                    return_exceptions=True
                )
                
                # Analyze for bottlenecks
                self.analyze_bottlenecks()
                
                # Update degradation level
                await self.update_degradation_level()
                
                # Export metrics
                await self.export_metrics()
                
                # Log summary
                self.log_health_summary()
                
                # Sleep until next check
                elapsed = time.time() - start_time
                sleep_time = max(0, self.check_interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except Exception as e:
            logger.error(f"Monitor error: {e}")
        finally:
            await self.cleanup()
            
    @health_check_duration.labels(component='agents', check_type='service_discovery').time()
    async def check_agent_health(self):
        """Check health of all registered agents"""
        try:
            # Get all registered agents from Consul
            _, services = await self.consul_client.catalog.services()
            
            agent_checks = []
            for service_name in services:
                if service_name.startswith('agent-'):
                    agent_checks.append(self._check_single_agent(service_name))
                    
            results = await asyncio.gather(*agent_checks, return_exceptions=True)
            
            # Update metrics
            healthy_count = sum(1 for r in results if isinstance(r, AgentHealth) and r.status == HealthStatus.HEALTHY)
            degraded_count = sum(1 for r in results if isinstance(r, AgentHealth) and r.status == HealthStatus.DEGRADED)
            
            logger.info(f"Agent health: {healthy_count} healthy, {degraded_count} degraded, "
                       f"{len(results) - healthy_count - degraded_count} unhealthy")
                       
        except Exception as e:
            logger.error(f"Error checking agent health: {e}")
            
    async def _check_single_agent(self, agent_name: str) -> AgentHealth:
        """Check health of a single agent"""
        try:
            # Get agent details from Consul
            _, services = await self.consul_client.health.service(agent_name)
            if not services:
                return AgentHealth(
                    name=agent_name,
                    agent_type="unknown",
                    status=HealthStatus.UNKNOWN,
                    last_check=datetime.now(),
                    response_time_ms=0
                )
                
            service = services[0]
            agent_address = service['Service']['Address']
            agent_port = service['Service']['Port']
            agent_type = service['Service'].get('Tags', ['unknown'])[0]
            
            # Check agent health endpoint
            start = time.time()
            async with self.http_session.get(
                f"http://{agent_address}:{agent_port}/health",
                timeout=aiohttp.ClientTimeout(total=5)
            ) as response:
                response_time = (time.time() - start) * 1000
                
                if response.status == 200:
                    health_data = await response.json()
                    status = HealthStatus.HEALTHY
                else:
                    status = HealthStatus.UNHEALTHY
                    
            health = AgentHealth(
                name=agent_name,
                agent_type=agent_type,
                status=status,
                last_check=datetime.now(),
                response_time_ms=response_time
            )
            
            # Update tracking
            self.agent_health[agent_name] = health
            
            # Update Prometheus metric
            agent_health_status.labels(
                agent_name=agent_name,
                agent_type=agent_type
            ).set(1 if status == HealthStatus.HEALTHY else 0)
            
            return health
            
        except asyncio.TimeoutError:
            logger.warning(f"Timeout checking agent {agent_name}")
            return AgentHealth(
                name=agent_name,
                agent_type="unknown",
                status=HealthStatus.UNHEALTHY,
                last_check=datetime.now(),
                response_time_ms=5000
            )
        except Exception as e:
            logger.error(f"Error checking agent {agent_name}: {e}")
            return AgentHealth(
                name=agent_name,
                agent_type="unknown",
                status=HealthStatus.UNKNOWN,
                last_check=datetime.now(),
                response_time_ms=0
            )
            
    @health_check_duration.labels(component='ollama', check_type='service').time()
    async def check_ollama_health(self):
        """Check Ollama service health"""
        try:
            async with self.http_session.get(
                f"{self.ollama_gateway_url}/api/tags",
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    logger.info(f"Ollama healthy with {len(data.get('models', []))} models loaded")
                else:
                    logger.warning(f"Ollama unhealthy: status {response.status}")
                    
        except Exception as e:
            logger.error(f"Error checking Ollama health: {e}")
            
    @health_check_duration.labels(component='system', check_type='resources').time()
    async def check_system_resources(self):
        """Monitor system resource usage"""
        try:
            # CPU and Memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            # GPU (if available)
            gpu_percent = 0
            gpu_memory_percent = 0
            
            try:
                import pynvml
                pynvml.nvmlInit()
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_percent = util.gpu
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    gpu_memory_percent = (mem_info.used / mem_info.total) * 100
            except:
                pass
                
            # Network connections
            connections = len(psutil.net_connections())
            
            # Create metrics object
            metrics = SystemMetrics(
                timestamp=datetime.now(),
                cpu_percent=cpu_percent,
                memory_percent=memory.percent,
                gpu_percent=gpu_percent,
                gpu_memory_percent=gpu_memory_percent,
                network_connections=connections,
                queue_depths={},
                active_agents=len([a for a in self.agent_health.values() if a.status == HealthStatus.HEALTHY]),
                degradation_level=self.current_degradation
            )
            
            # Store in history
            self.metrics_history.append(metrics)
            
            # Update Prometheus metrics
            resource_utilization.labels(resource_type='cpu').set(cpu_percent)
            resource_utilization.labels(resource_type='memory').set(memory.percent)
            resource_utilization.labels(resource_type='gpu').set(gpu_percent)
            resource_utilization.labels(resource_type='gpu_memory').set(gpu_memory_percent)
            
            # Check for resource warnings
            if memory.percent > self.memory_threshold:
                logger.warning(f"High memory usage: {memory.percent:.1f}%")
            if gpu_memory_percent > self.gpu_threshold:
                logger.warning(f"High GPU memory usage: {gpu_memory_percent:.1f}%")
                
        except Exception as e:
            logger.error(f"Error checking system resources: {e}")
            
    @health_check_duration.labels(component='queues', check_type='redis').time()
    async def check_queue_health(self):
        """Monitor request queue depths"""
        try:
            queue_depths = {}
            
            # Check priority queues
            for priority in range(1, 11):
                queue_key = f"ollama:queue:p{priority}"
                depth = await self.redis_client.llen(queue_key)
                queue_depths[f"priority_{priority}"] = depth
                
                # Update Prometheus metric
                ollama_queue_depth.labels(priority=str(priority)).set(depth)
                
            # Check total queue depth
            total_depth = sum(queue_depths.values())
            if total_depth > 1000:
                logger.warning(f"High queue depth: {total_depth} total requests")
                
            # Store in latest metrics
            if self.metrics_history:
                self.metrics_history[-1].queue_depths = queue_depths
                
        except Exception as e:
            logger.error(f"Error checking queue health: {e}")
            
    async def check_network_health(self):
        """Monitor network connectivity between components"""
        try:
            # Check key service connectivity
            services = [
                ('Redis', self.redis_url),
                ('Consul', self.consul_url),
                ('Ollama Gateway', self.ollama_gateway_url),
                ('Prometheus', self.prometheus_url)
            ]
            
            for service_name, url in services:
                try:
                    if service_name == 'Redis':
                        # Redis ping
                        await self.redis_client.ping()
                    else:
                        # HTTP health check
                        async with self.http_session.get(
                            url,
                            timeout=aiohttp.ClientTimeout(total=5)
                        ) as response:
                            if response.status >= 500:
                                logger.warning(f"{service_name} returned status {response.status}")
                except Exception as e:
                    logger.error(f"Cannot reach {service_name}: {e}")
                    
        except Exception as e:
            logger.error(f"Error checking network health: {e}")
            
    def analyze_bottlenecks(self):
        """Analyze system for bottlenecks"""
        self.bottlenecks.clear()
        
        if not self.metrics_history:
            return
            
        latest = self.metrics_history[-1]
        
        # Check Ollama capacity bottleneck
        total_queue_depth = sum(latest.queue_depths.values())
        if total_queue_depth > 100:
            self.bottlenecks.append(BottleneckInfo(
                component="Ollama Service",
                severity="HIGH",
                description=f"Request queue depth ({total_queue_depth}) exceeds threshold",
                metrics={
                    "queue_depth": total_queue_depth,
                    "ollama_parallel_limit": 2
                },
                suggested_actions=[
                    "Enable request batching",
                    "Increase cache TTL",
                    "Switch to lighter models (gpt-oss)",
                    "Implement more aggressive request filtering"
                ]
            ))
            
        # Check memory bottleneck
        if latest.memory_percent > self.memory_threshold:
            self.bottlenecks.append(BottleneckInfo(
                component="System Memory",
                severity="HIGH",
                description=f"Memory usage ({latest.memory_percent:.1f}%) exceeds threshold",
                metrics={
                    "memory_percent": latest.memory_percent,
                    "active_agents": latest.active_agents
                },
                suggested_actions=[
                    "Reduce agent memory limits",
                    "Enable memory-based autoscaling",
                    "Increase swap space",
                    "Disable non-critical agents"
                ]
            ))
            
        # Check GPU bottleneck
        if latest.gpu_memory_percent > self.gpu_threshold:
            self.bottlenecks.append(BottleneckInfo(
                component="GPU Memory",
                severity="CRITICAL",
                description=f"GPU memory usage ({latest.gpu_memory_percent:.1f}%) exceeds threshold",
                metrics={
                    "gpu_memory_percent": latest.gpu_memory_percent,
                    "gpu_percent": latest.gpu_percent
                },
                suggested_actions=[
                    "Unload unused models",
                    "Reduce OLLAMA_MAX_LOADED_MODELS",
                    "Switch to CPU-only mode for some requests",
                    "Implement model swapping strategy"
                ]
            ))
            
        # Check agent failures
        unhealthy_agents = [a for a in self.agent_health.values() if a.status == HealthStatus.UNHEALTHY]
        if len(unhealthy_agents) > self.total_expected_agents * 0.1:  # >10% unhealthy
            self.bottlenecks.append(BottleneckInfo(
                component="Agent Health",
                severity="HIGH",
                description=f"{len(unhealthy_agents)} agents are unhealthy",
                metrics={
                    "unhealthy_count": len(unhealthy_agents),
                    "total_agents": len(self.agent_health)
                },
                suggested_actions=[
                    "Check agent container logs",
                    "Verify network connectivity",
                    "Restart failed agents",
                    "Check for OOM kills"
                ]
            ))
            
    async def update_degradation_level(self):
        """Update system degradation level based on metrics"""
        if not self.metrics_history:
            return
            
        latest = self.metrics_history[-1]
        
        # Calculate overall system load
        system_load = max(
            latest.cpu_percent,
            latest.memory_percent,
            latest.gpu_memory_percent
        )
        
        # Determine degradation level
        if system_load >= 95:
            new_level = DegradationLevel.CRITICAL
        elif system_load >= 80:
            new_level = DegradationLevel.MAJOR
        elif system_load >= 60:
            new_level = DegradationLevel.MINOR
        else:
            new_level = DegradationLevel.NORMAL
            
        # Update if changed
        if new_level != self.current_degradation:
            logger.info(f"Degradation level changed: {self.current_degradation.name} -> {new_level.name}")
            self.current_degradation = new_level
            
            # Notify via Redis pub/sub
            await self.redis_client.publish(
                'system:degradation:level',
                json.dumps({
                    'level': new_level.value,
                    'level_name': new_level.name,
                    'timestamp': datetime.now().isoformat(),
                    'metrics': {
                        'cpu_percent': latest.cpu_percent,
                        'memory_percent': latest.memory_percent,
                        'gpu_memory_percent': latest.gpu_memory_percent
                    }
                })
            )
            
        # Update Prometheus metric
        system_degradation_level.set(new_level.value)
        
    async def export_metrics(self):
        """Export health status to Redis for other services"""
        try:
            health_summary = {
                'timestamp': datetime.now().isoformat(),
                'degradation_level': self.current_degradation.value,
                'agent_health': {
                    'total': len(self.agent_health),
                    'healthy': len([a for a in self.agent_health.values() if a.status == HealthStatus.HEALTHY]),
                    'degraded': len([a for a in self.agent_health.values() if a.status == HealthStatus.DEGRADED]),
                    'unhealthy': len([a for a in self.agent_health.values() if a.status == HealthStatus.UNHEALTHY])
                },
                'bottlenecks': [
                    {
                        'component': b.component,
                        'severity': b.severity,
                        'description': b.description
                    }
                    for b in self.bottlenecks
                ]
            }
            
            # Store in Redis with TTL
            await self.redis_client.setex(
                'health:summary:latest',
                300,  # 5 minute TTL
                json.dumps(health_summary)
            )
            
            # Also publish for real-time subscribers
            await self.redis_client.publish(
                'health:updates',
                json.dumps(health_summary)
            )
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
            
    def log_health_summary(self):
        """Log a summary of system health"""
        if not self.metrics_history:
            return
            
        latest = self.metrics_history[-1]
        
        logger.info(
            f"System Health Summary - "
            f"Degradation: {self.current_degradation.name}, "
            f"CPU: {latest.cpu_percent:.1f}%, "
            f"Memory: {latest.memory_percent:.1f}%, "
            f"GPU: {latest.gpu_memory_percent:.1f}%, "
            f"Active Agents: {latest.active_agents}/{self.total_expected_agents}, "
            f"Queue Depth: {sum(latest.queue_depths.values())}, "
            f"Bottlenecks: {len(self.bottlenecks)}"
        )
        
        # Log bottlenecks if any
        for bottleneck in self.bottlenecks:
            logger.warning(
                f"Bottleneck detected - {bottleneck.component}: {bottleneck.description}"
            )

async def main():
    """Main entry point"""
    monitor = DistributedHealthMonitor()
    
    try:
        await monitor.run()
    except KeyboardInterrupt:
        logger.info("Shutting down health monitor")
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        
if __name__ == "__main__":
    asyncio.run(main())