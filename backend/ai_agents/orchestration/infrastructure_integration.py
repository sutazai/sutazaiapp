"""
Infrastructure Integration for SutazAI Agent Orchestration
=========================================================

Comprehensive integration system that connects the orchestration framework
with Docker containers, Redis, Ollama, monitoring systems, and all infrastructure
components. Provides automated deployment, scaling, health management, and
resource optimization for the entire 38-agent ecosystem.

Key Features:
- Docker container lifecycle management
- Redis cluster integration and management
- Ollama model deployment and optimization
- Monitoring system integration (Prometheus, Grafana, Loki)
- Auto-scaling and resource optimization
- Health monitoring and self-healing
- Configuration management
- Network and service discovery
"""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
import docker
import redis.asyncio as redis
import httpx
import yaml
import psutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
import aiofiles

logger = logging.getLogger(__name__)


@dataclass
class ContainerSpec:
    """Container specification for agent deployment"""
    name: str
    image: str
    ports: Dict[str, int]
    environment: Dict[str, str]
    volumes: Dict[str, str]
    memory_limit: str
    cpu_limit: str
    health_check: Dict[str, Any]
    restart_policy: str = "unless-stopped"
    network: str = "sutazai-network"
    labels: Dict[str, str] = field(default_factory=dict)
    dependencies: List[str] = field(default_factory=list)


@dataclass
class ServiceSpec:
    """Service specification for infrastructure components"""
    name: str
    type: str  # redis, postgres, ollama, etc.
    container_spec: ContainerSpec
    scaling_config: Dict[str, Any]
    monitoring_config: Dict[str, Any]
    backup_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ResourceMetrics:
    """Resource utilization metrics"""
    cpu_percent: float
    memory_percent: float
    memory_usage_mb: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class HealthStatus:
    """Health status information"""
    status: str  # healthy, unhealthy, unknown
    last_check: datetime
    response_time_ms: float
    error_message: Optional[str] = None
    consecutive_failures: int = 0


class InfrastructureIntegration:
    """
    Comprehensive infrastructure integration and management system
    """
    
    def __init__(self, redis_url: str = "redis://redis:6379"):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.docker_client = docker.from_env()
        
        # Service Management
        self.services: Dict[str, ServiceSpec] = {}
        self.containers: Dict[str, docker.models.containers.Container] = {}
        self.health_status: Dict[str, HealthStatus] = {}
        
        # Resource Monitoring
        self.resource_metrics: Dict[str, ResourceMetrics] = {}
        self.performance_history: Dict[str, List[ResourceMetrics]] = {}
        
        # Configuration
        self.config = {
            "health_check_interval": 30,
            "resource_check_interval": 60,
            "auto_scaling_enabled": True,
            "auto_healing_enabled": True,
            "max_restart_attempts": 3,
            "scale_up_threshold": 0.8,
            "scale_down_threshold": 0.3,
            "monitoring_retention_hours": 24
        }
        
        # Ollama Integration
        self.ollama_config = {
            "base_url": "http://ollama:11434",
            "models": [],
            "model_optimization": True,
            "auto_model_management": True
        }
        
        # Background Tasks
        self.background_tasks: List[asyncio.Task] = []
        self.running = False
        
        logger.info("Infrastructure Integration initialized")
    
    async def initialize(self):
        """Initialize the infrastructure integration system"""
        logger.info("🚀 Initializing Infrastructure Integration...")
        
        # Connect to Redis
        self.redis_client = await redis.from_url(self.redis_url)
        
        # Load service specifications
        await self._load_service_specifications()
        
        # Initialize monitoring
        await self._initialize_monitoring()
        
        # Start background services
        self.background_tasks = [
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._resource_monitor()),
            asyncio.create_task(self._auto_scaler()),
            asyncio.create_task(self._auto_healer()),
            asyncio.create_task(self._ollama_manager())
        ]
        
        self.running = True
        logger.info("✅ Infrastructure Integration ready")
    
    async def shutdown(self):
        """Shutdown the infrastructure integration"""
        logger.info("🛑 Shutting down Infrastructure Integration...")
        
        self.running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("✅ Infrastructure Integration shutdown complete")
    
    # ==================== Service Management ====================
    
    async def _load_service_specifications(self):
        """Load service specifications for all infrastructure components"""
        
        # Core Infrastructure Services
        self.services.update({
            "redis": ServiceSpec(
                name="redis",
                type="database",
                container_spec=ContainerSpec(
                    name="sutazai-redis",
                    image="redis:7.2-alpine",
                    ports={"6379": 6379},
                    environment={
                        "REDIS_PASSWORD": "redis_password"
                    },
                    volumes={
                        "redis_data": "/data"
                    },
                    memory_limit="1G",
                    cpu_limit="1",
                    health_check={
                        "test": ["CMD-SHELL", "redis-cli -a redis_password ping"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5
                    }
                ),
                scaling_config={
                    "min_instances": 1,
                    "max_instances": 3,
                    "target_cpu": 70,
                    "target_memory": 80
                },
                monitoring_config={
                    "metrics_endpoint": "/metrics",
                    "log_level": "INFO"
                }
            ),
            
            "postgres": ServiceSpec(
                name="postgres",
                type="database",
                container_spec=ContainerSpec(
                    name="sutazai-postgres",
                    image="postgres:16.3-alpine",
                    ports={"5432": 5432},
                    environment={
                        "POSTGRES_USER": "sutazai",
                        "POSTGRES_PASSWORD": "sutazai_password",
                        "POSTGRES_DB": "sutazai"
                    },
                    volumes={
                        "postgres_data": "/var/lib/postgresql/data"
                    },
                    memory_limit="2G",
                    cpu_limit="2",
                    health_check={
                        "test": ["CMD-SHELL", "pg_isready -U sutazai"],
                        "interval": "10s",
                        "timeout": "5s",
                        "retries": 5
                    }
                ),
                scaling_config={
                    "min_instances": 1,
                    "max_instances": 1,  # Postgres doesn't scale horizontally easily
                    "backup_enabled": True
                },
                monitoring_config={
                    "metrics_endpoint": "/metrics",
                    "log_level": "INFO",
                    "slow_query_log": True
                }
            ),
            
            "ollama": ServiceSpec(
                name="ollama",
                type="ai_inference",
                container_spec=ContainerSpec(
                    name="sutazai-ollama",
                    image="ollama/ollama:latest",
                    ports={"11434": 11434},
                    environment={
                        "OLLAMA_HOST": "0.0.0.0",
                        "OLLAMA_ORIGINS": "*",
                        "OLLAMA_NUM_PARALLEL": "2",
                        "OLLAMA_MAX_LOADED_MODELS": "1"
                    },
                    volumes={
                        "ollama_data": "/root/.ollama",
                        "models_data": "/models"
                    },
                    memory_limit="8G",
                    cpu_limit="6",
                    health_check={
                        "test": ["CMD-SHELL", "ollama list > /dev/null || exit 1"],
                        "interval": "30s",
                        "timeout": "10s",
                        "retries": 5
                    },
                    labels={
                        "ai.model.provider": "ollama",
                        "ai.resource.intensive": "true"
                    }
                ),
                scaling_config={
                    "min_instances": 1,
                    "max_instances": 2,
                    "target_cpu": 75,
                    "target_memory": 85,
                    "model_based_scaling": True
                },
                monitoring_config={
                    "metrics_endpoint": "/metrics",
                    "model_performance_tracking": True,
                    "inference_monitoring": True
                }
            )
        })
        
        # Agent Container Specifications
        await self._load_agent_specifications()
        
        logger.info(f"Loaded {len(self.services)} service specifications")
    
    async def _load_agent_specifications(self):
        """Load specifications for all 38 AI agent containers"""
        
        # Base agent configuration template
        base_agent_config = {
            "memory_limit": "1G",
            "cpu_limit": "1",
            "network": "sutazai-network",
            "restart_policy": "unless-stopped",
            "environment": {
                "REDIS_URL": "redis://redis:6379",
                "BACKEND_URL": "http://backend-agi:8000",
                "OLLAMA_BASE_URL": "http://ollama:11434"
            },
            "volumes": {
                "agent_workspaces": "/app/workspace",
                "agent_outputs": "/app/outputs"
            },
            "health_check": {
                "test": ["CMD", "curl", "-f", "http://localhost:8080/health"],
                "interval": "30s",
                "timeout": "10s",
                "retries": 3
            }
        }
        
        # Define all 38 agents with their specific configurations
        agent_specs = {
            "agi-system-architect": {
                "image": "sutazai/agi-architect:latest",
                "ports": {"8201": 8080},
                "memory_limit": "2G",
                "cpu_limit": "2"
            },
            "autonomous-system-controller": {
                "image": "sutazai/autonomous-controller:latest", 
                "ports": {"8202": 8080},
                "memory_limit": "1.5G",
                "cpu_limit": "2"
            },
            "ai-agent-orchestrator": {
                "image": "sutazai/agent-orchestrator:latest",
                "ports": {"8203": 8080}
            },
            "complex-problem-solver": {
                "image": "sutazai/problem-solver:latest",
                "ports": {"8204": 8080},
                "memory_limit": "2G"
            },
            "deep-learning-brain-manager": {
                "image": "sutazai/brain-manager:latest",
                "ports": {"8205": 8080},
                "memory_limit": "3G",
                "cpu_limit": "2"
            },
            "senior-ai-engineer": {
                "image": "sutazai/ai-engineer:latest",
                "ports": {"8206": 8080},
                "memory_limit": "2G"
            },
            "senior-backend-developer": {
                "image": "sutazai/backend-dev:latest",
                "ports": {"8207": 8080}
            },
            "senior-frontend-developer": {
                "image": "sutazai/frontend-dev:latest",
                "ports": {"8208": 8080}
            },
            "code-generation-improver": {
                "image": "sutazai/code-improver:latest",
                "ports": {"8209": 8080}
            },
            "testing-qa-validator": {
                "image": "sutazai/qa-validator:latest",
                "ports": {"8210": 8080}
            },
            "infrastructure-devops-manager": {
                "image": "sutazai/devops-manager:latest",
                "ports": {"8211": 8080},
                "volumes": {
                    **base_agent_config["volumes"],
                    "/var/run/docker.sock": "/var/run/docker.sock:ro"
                }
            },
            "deployment-automation-master": {
                "image": "sutazai/deployment-master:latest",
                "ports": {"8212": 8080}
            },
            "hardware-resource-optimizer": {
                "image": "sutazai/hardware-optimizer:latest",
                "ports": {"8213": 8080}
            },
            "system-optimizer-reorganizer": {
                "image": "sutazai/system-optimizer:latest",
                "ports": {"8214": 8080}
            },
            "ollama-integration-specialist": {
                "image": "sutazai/ollama-specialist:latest",
                "ports": {"8215": 8080}
            },
                "ports": {"8216": 8080}
            },
            "context-optimization-engineer": {
                "image": "sutazai/context-optimizer:latest",
                "ports": {"8217": 8080}
            },
            "localagi-orchestration-manager": {
                "image": "sutazai/localagi-manager:latest",
                "ports": {"8218": 8080}
            },
            "agentzero-coordinator": {
                "image": "sutazai/agentzero:latest",
                "ports": {"8219": 8080}
            },
            "bigagi-system-manager": {
                "image": "sutazai/bigagi-manager:latest",
                "ports": {"8220": 8080}
            },
            "agentgpt-autonomous-executor": {
                "image": "sutazai/agentgpt:latest",
                "ports": {"8221": 8080}
            },
            "opendevin-code-generator": {
                "image": "sutazai/opendevin:latest",
                "ports": {"8222": 8080}
            },
            "langflow-workflow-designer": {
                "image": "sutazai/langflow:latest",
                "ports": {"8223": 8080}
            },
            "flowiseai-flow-manager": {
                "image": "sutazai/flowise:latest",
                "ports": {"8224": 8080}
            },
            "dify-automation-specialist": {
                "image": "sutazai/dify:latest",
                "ports": {"8225": 8080}
            },
            "task-assignment-coordinator": {
                "image": "sutazai/task-coordinator:latest",
                "ports": {"8226": 8080}
            },
            "semgrep-security-analyzer": {
                "image": "sutazai/semgrep:latest",
                "ports": {"8227": 8080}
            },
            "security-pentesting-specialist": {
                "image": "sutazai/pentester:latest",
                "ports": {"8228": 8080}
            },
            "kali-security-specialist": {
                "image": "sutazai/kali-specialist:latest",
                "ports": {"8229": 8080},
                "memory_limit": "2G"
            },
            "private-data-analyst": {
                "image": "sutazai/private-analyst:latest",
                "ports": {"8230": 8080}
            },
            "jarvis-voice-interface": {
                "image": "sutazai/jarvis:latest",
                "ports": {"8231": 8080}
            },
            "browser-automation-orchestrator": {
                "image": "sutazai/browser-automation:latest",
                "ports": {"8232": 8080}
            },
            "shell-automation-specialist": {
                "image": "sutazai/shell-specialist:latest",
                "ports": {"8233": 8080}
            },
            "ai-product-manager": {
                "image": "sutazai/product-manager:latest",
                "ports": {"8234": 8080}
            },
            "ai-scrum-master": {
                "image": "sutazai/scrum-master:latest",
                "ports": {"8235": 8080}
            },
            "ai-agent-creator": {
                "image": "sutazai/agent-creator:latest",
                "ports": {"8236": 8080}
            },
            "document-knowledge-manager": {
                "image": "sutazai/doc-manager:latest",
                "ports": {"8237": 8080}
            },
            "financial-analysis-specialist": {
                "image": "sutazai/financial-analyst:latest",
                "ports": {"8238": 8080}
            }
        }
        
        # Create service specs for all agents
        for agent_id, agent_config in agent_specs.items():
            # Merge with base config
            config = {**base_agent_config, **agent_config}
            
            container_spec = ContainerSpec(
                name=f"sutazai-{agent_id}",
                image=config["image"],
                ports=config["ports"],
                environment=config["environment"],
                volumes=config["volumes"],
                memory_limit=config["memory_limit"],
                cpu_limit=config["cpu_limit"],
                health_check=config["health_check"],
                restart_policy=config["restart_policy"],
                network=config["network"],
                labels={
                    "ai.agent.id": agent_id,
                    "ai.agent.type": "specialized",
                    "sutazai.component": "agent"
                }
            )
            
            self.services[agent_id] = ServiceSpec(
                name=agent_id,
                type="ai_agent",
                container_spec=container_spec,
                scaling_config={
                    "min_instances": 1,
                    "max_instances": 2,
                    "target_cpu": 80,
                    "target_memory": 85
                },
                monitoring_config={
                    "health_endpoint": "/health",
                    "metrics_endpoint": "/metrics",
                    "log_level": "INFO"
                }
            )
    
    # ==================== Container Lifecycle Management ====================
    
    async def deploy_service(self, service_name: str) -> bool:
        """Deploy a service container"""
        
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False
        
        service_spec = self.services[service_name]
        container_spec = service_spec.container_spec
        
        try:
            # Check if container already exists
            existing_container = self._get_container(container_spec.name)
            if existing_container:
                logger.info(f"Container {container_spec.name} already exists")
                return True
            
            # Prepare container configuration
            container_config = {
                "image": container_spec.image,
                "name": container_spec.name,
                "ports": container_spec.ports,
                "environment": container_spec.environment,
                "volumes": container_spec.volumes,
                "restart_policy": {"Name": container_spec.restart_policy},
                "network": container_spec.network,
                "labels": container_spec.labels,
                "mem_limit": container_spec.memory_limit,
                "cpu_period": 100000,
                "cpu_quota": int(float(container_spec.cpu_limit) * 100000),
                "healthcheck": {
                    "test": container_spec.health_check["test"],
                    "interval": self._parse_duration(container_spec.health_check["interval"]),
                    "timeout": self._parse_duration(container_spec.health_check["timeout"]),
                    "retries": container_spec.health_check["retries"]
                }
            }
            
            # Deploy dependencies first
            for dependency in container_spec.dependencies:
                if not await self.deploy_service(dependency):
                    logger.error(f"Failed to deploy dependency {dependency}")
                    return False
            
            # Create and start container
            container = self.docker_client.containers.run(
                detach=True,
                **container_config
            )
            
            self.containers[service_name] = container
            
            # Wait for container to be healthy
            await self._wait_for_health(container_spec.name, timeout=60)
            
            # Update health status
            self.health_status[service_name] = HealthStatus(
                status="healthy",
                last_check=datetime.now(),
                response_time_ms=0.0
            )
            
            logger.info(f"✅ Successfully deployed service: {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy service {service_name}: {e}")
            return False
    
    async def stop_service(self, service_name: str) -> bool:
        """Stop a service container"""
        
        try:
            container = self._get_container_by_service(service_name)
            if container:
                container.stop(timeout=30)
                logger.info(f"Stopped service: {service_name}")
                return True
            else:
                logger.warning(f"Service {service_name} not found or not running")
                return False
                
        except Exception as e:
            logger.error(f"Failed to stop service {service_name}: {e}")
            return False
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart a service container"""
        
        try:
            container = self._get_container_by_service(service_name)
            if container:
                container.restart(timeout=30)
                
                # Wait for health check
                container_spec = self.services[service_name].container_spec
                await self._wait_for_health(container_spec.name, timeout=60)
                
                logger.info(f"Restarted service: {service_name}")
                return True
            else:
                # If container doesn't exist, deploy it
                return await self.deploy_service(service_name)
                
        except Exception as e:
            logger.error(f"Failed to restart service {service_name}: {e}")
            return False
    
    async def scale_service(self, service_name: str, instances: int) -> bool:
        """Scale a service to the specified number of instances"""
        
        if service_name not in self.services:
            logger.error(f"Unknown service: {service_name}")
            return False
        
        service_spec = self.services[service_name]
        scaling_config = service_spec.scaling_config
        
        # Check scaling limits
        if instances < scaling_config["min_instances"]:
            instances = scaling_config["min_instances"]
        elif instances > scaling_config["max_instances"]:
            instances = scaling_config["max_instances"]
        
        try:
            # Get current instances
            current_containers = self._get_service_containers(service_name)
            current_count = len(current_containers)
            
            if instances > current_count:
                # Scale up
                for i in range(instances - current_count):
                    instance_name = f"{service_name}-{current_count + i + 1}"
                    if await self._deploy_service_instance(service_name, instance_name):
                        logger.info(f"Scaled up {service_name}: created instance {instance_name}")
            
            elif instances < current_count:
                # Scale down
                containers_to_remove = current_containers[instances:]
                for container in containers_to_remove:
                    container.stop(timeout=30)
                    container.remove()
                    logger.info(f"Scaled down {service_name}: removed instance {container.name}")
            
            logger.info(f"Successfully scaled {service_name} to {instances} instances")
            return True
            
        except Exception as e:
            logger.error(f"Failed to scale service {service_name}: {e}")
            return False
    
    # ==================== Health Monitoring ====================
    
    async def _health_monitor(self):
        """Monitor health of all services"""
        
        while self.running:
            try:
                health_checks = []
                
                for service_name in self.services:
                    health_checks.append(
                        asyncio.create_task(self._check_service_health(service_name))
                    )
                
                # Wait for all health checks to complete
                await asyncio.gather(*health_checks, return_exceptions=True)
                
                await asyncio.sleep(self.config["health_check_interval"])
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _check_service_health(self, service_name: str):
        """Check health of a specific service"""
        
        try:
            container = self._get_container_by_service(service_name)
            
            if not container:
                self.health_status[service_name] = HealthStatus(
                    status="unknown",
                    last_check=datetime.now(),
                    response_time_ms=0.0,
                    error_message="Container not found"
                )
                return
            
            # Check container status
            container.reload()
            if container.status != "running":
                self.health_status[service_name] = HealthStatus(
                    status="unhealthy",
                    last_check=datetime.now(),
                    response_time_ms=0.0,
                    error_message=f"Container status: {container.status}"
                )
                return
            
            # Perform HTTP health check if service has health endpoint
            service_spec = self.services[service_name]
            monitoring_config = service_spec.monitoring_config
            
            if "health_endpoint" in monitoring_config:
                start_time = time.time()
                
                # Get container IP and port
                container_ports = container.ports
                health_endpoint = monitoring_config["health_endpoint"]
                
                # Try to find the health check port
                health_port = None
                for port_spec, host_ports in container_ports.items():
                    if host_ports:
                        health_port = host_ports[0]["HostPort"]
                        break
                
                if health_port:
                    health_url = f"http://localhost:{health_port}{health_endpoint}"
                    
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(health_url)
                        
                        response_time = (time.time() - start_time) * 1000
                        
                        if response.status_code == 200:
                            self.health_status[service_name] = HealthStatus(
                                status="healthy",
                                last_check=datetime.now(),
                                response_time_ms=response_time,
                                consecutive_failures=0
                            )
                        else:
                            self._record_health_failure(service_name, f"HTTP {response.status_code}")
                else:
                    # No port mapping found, assume healthy if container is running
                    self.health_status[service_name] = HealthStatus(
                        status="healthy",
                        last_check=datetime.now(),
                        response_time_ms=0.0
                    )
            else:
                # No health endpoint, check container health
                health_info = container.attrs.get("State", {}).get("Health", {})
                if health_info:
                    docker_status = health_info.get("Status", "none")
                    if docker_status == "healthy":
                        status = "healthy"
                    elif docker_status == "unhealthy":
                        status = "unhealthy"
                    else:
                        status = "unknown"
                    
                    self.health_status[service_name] = HealthStatus(
                        status=status,
                        last_check=datetime.now(),
                        response_time_ms=0.0
                    )
                else:
                    # Assume healthy if container is running and no health check defined
                    self.health_status[service_name] = HealthStatus(
                        status="healthy",
                        last_check=datetime.now(),
                        response_time_ms=0.0
                    )
                    
        except Exception as e:
            self._record_health_failure(service_name, str(e))
    
    def _record_health_failure(self, service_name: str, error_message: str):
        """Record a health check failure"""
        
        current_status = self.health_status.get(service_name)
        consecutive_failures = (current_status.consecutive_failures + 1) if current_status else 1
        
        self.health_status[service_name] = HealthStatus(
            status="unhealthy",
            last_check=datetime.now(),
            response_time_ms=0.0,
            error_message=error_message,
            consecutive_failures=consecutive_failures
        )
        
        logger.warning(f"Health check failed for {service_name}: {error_message}")
    
    # ==================== Resource Monitoring ====================
    
    async def _resource_monitor(self):
        """Monitor resource utilization of all services"""
        
        while self.running:
            try:
                for service_name in self.services:
                    await self._collect_service_metrics(service_name)
                
                # Collect system-wide metrics
                await self._collect_system_metrics()
                
                await asyncio.sleep(self.config["resource_check_interval"])
                
            except Exception as e:
                logger.error(f"Resource monitor error: {e}")
                await asyncio.sleep(60)
    
    async def _collect_service_metrics(self, service_name: str):
        """Collect resource metrics for a specific service"""
        
        try:
            container = self._get_container_by_service(service_name)
            if not container:
                return
            
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate CPU percentage
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                       stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_cpu_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                              stats["precpu_stats"]["system_cpu_usage"]
            
            if system_cpu_delta > 0:
                cpu_percent = (cpu_delta / system_cpu_delta) * len(stats["cpu_stats"]["cpu_usage"]["percpu_usage"]) * 100.0
            else:
                cpu_percent = 0.0
            
            # Calculate memory usage
            memory_usage = stats["memory_stats"]["usage"]
            memory_limit = stats["memory_stats"]["limit"]
            memory_percent = (memory_usage / memory_limit) * 100.0 if memory_limit > 0 else 0.0
            
            # Network I/O
            networks = stats.get("networks", {})
            network_io = {}
            for interface, net_stats in networks.items():
                network_io[f"{interface}_rx_bytes"] = net_stats["rx_bytes"] 
                network_io[f"{interface}_tx_bytes"] = net_stats["tx_bytes"]
            
            # Create metrics object
            metrics = ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                memory_usage_mb=memory_usage / (1024 * 1024),
                disk_usage_percent=0.0,  # Would need additional logic for disk stats
                network_io=network_io
            )
            
            # Store current metrics
            self.resource_metrics[service_name] = metrics
            
            # Store in history
            if service_name not in self.performance_history:
                self.performance_history[service_name] = []
            
            history = self.performance_history[service_name]
            history.append(metrics)
            
            # Limit history size
            max_history = self.config["monitoring_retention_hours"] * 60  # per minute
            if len(history) > max_history:
                history.pop(0)
            
            # Store in Redis for persistence
            await self.redis_client.hset(
                f"metrics:{service_name}",
                "latest",
                json.dumps(asdict(metrics), default=str)
            )
            
        except Exception as e:
            logger.error(f"Failed to collect metrics for {service_name}: {e}")
    
    async def _collect_system_metrics(self):
        """Collect system-wide resource metrics"""
        
        try:
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            
            system_metrics = {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_usage_gb": memory.used / (1024**3),
                "disk_percent": disk.percent,
                "disk_usage_gb": disk.used / (1024**3),
                "timestamp": datetime.now().isoformat()
            }
            
            # Store in Redis
            await self.redis_client.hset(
                "metrics:system",
                "latest",
                json.dumps(system_metrics)
            )
            
        except Exception as e:
            logger.error(f"Failed to collect system metrics: {e}")
    
    # ==================== Auto-scaling ====================
    
    async def _auto_scaler(self):
        """Automatic service scaling based on resource utilization"""
        
        while self.running:
            try:
                if not self.config["auto_scaling_enabled"]:
                    await asyncio.sleep(300)  # Check every 5 minutes
                    continue
                
                for service_name, service_spec in self.services.items():
                    if service_spec.type == "ai_agent":  # Only scale AI agents
                        await self._evaluate_scaling(service_name)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Auto-scaler error: {e}")
                await asyncio.sleep(300)
    
    async def _evaluate_scaling(self, service_name: str):
        """Evaluate if a service needs scaling"""
        
        try:
            metrics = self.resource_metrics.get(service_name)
            if not metrics:
                return
            
            service_spec = self.services[service_name]
            scaling_config = service_spec.scaling_config
            
            current_instances = len(self._get_service_containers(service_name))
            scale_up_threshold = self.config["scale_up_threshold"] * 100
            scale_down_threshold = self.config["scale_down_threshold"] * 100
            
            # Check if scaling up is needed
            if (metrics.cpu_percent > scale_up_threshold or 
                metrics.memory_percent > scale_up_threshold):
                
                if current_instances < scaling_config["max_instances"]:
                    logger.info(f"Scaling up {service_name}: CPU={metrics.cpu_percent:.1f}%, MEM={metrics.memory_percent:.1f}%")
                    await self.scale_service(service_name, current_instances + 1)
            
            # Check if scaling down is possible
            elif (metrics.cpu_percent < scale_down_threshold and 
                  metrics.memory_percent < scale_down_threshold):
                
                if current_instances > scaling_config["min_instances"]:
                    # Check history to ensure consistent low usage
                    history = self.performance_history.get(service_name, [])
                    if len(history) >= 5:  # At least 5 data points
                        recent_avg_cpu = sum(m.cpu_percent for m in history[-5:]) / 5
                        recent_avg_mem = sum(m.memory_percent for m in history[-5:]) / 5
                        
                        if (recent_avg_cpu < scale_down_threshold and 
                            recent_avg_mem < scale_down_threshold):
                            logger.info(f"Scaling down {service_name}: sustained low usage")
                            await self.scale_service(service_name, current_instances - 1)
                            
        except Exception as e:
            logger.error(f"Failed to evaluate scaling for {service_name}: {e}")
    
    # ==================== Auto-healing ====================
    
    async def _auto_healer(self):
        """Automatic service healing for failed containers"""
        
        while self.running:
            try:
                if not self.config["auto_healing_enabled"]:
                    await asyncio.sleep(60)
                    continue
                
                for service_name, health_status in self.health_status.items():
                    if health_status.status == "unhealthy":
                        await self._heal_service(service_name, health_status)
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Auto-healer error: {e}")
                await asyncio.sleep(60)
    
    async def _heal_service(self, service_name: str, health_status: HealthStatus):
        """Attempt to heal an unhealthy service"""
        
        try:
            max_failures = self.config["max_restart_attempts"]
            
            if health_status.consecutive_failures >= max_failures:
                logger.error(f"Service {service_name} has exceeded max failure attempts ({max_failures})")
                return
            
            logger.warning(f"Attempting to heal unhealthy service: {service_name}")
            
            # Try restarting the service
            if await self.restart_service(service_name):
                logger.info(f"Successfully healed service: {service_name}")
                
                # Reset failure count
                self.health_status[service_name].consecutive_failures = 0
            else:
                logger.error(f"Failed to heal service: {service_name}")
                
        except Exception as e:
            logger.error(f"Auto-healing failed for {service_name}: {e}")
    
    # ==================== Ollama Integration ====================
    
    async def _ollama_manager(self):
        """Manage Ollama models and optimization"""
        
        while self.running:
            try:
                if not self.ollama_config["auto_model_management"]:
                    await asyncio.sleep(300)
                    continue
                
                # Check Ollama health
                if await self._check_ollama_health():
                    await self._manage_ollama_models()
                    await self._optimize_ollama_performance()
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Ollama manager error: {e}")
                await asyncio.sleep(300)
    
    async def _check_ollama_health(self) -> bool:
        """Check if Ollama is healthy and responsive"""
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{self.ollama_config['base_url']}/api/tags")
                return response.status_code == 200
                
        except Exception as e:
            logger.warning(f"Ollama health check failed: {e}")
            return False
    
    async def _manage_ollama_models(self):
        """Manage Ollama model lifecycle"""
        
        try:
            # Get list of available models
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.get(f"{self.ollama_config['base_url']}/api/tags")
                
                if response.status_code == 200:
                    data = response.json()
                    models = data.get("models", [])
                    
                    self.ollama_config["models"] = [
                        {
                            "name": model["name"],
                            "size": model["size"],
                            "digest": model["digest"],
                            "modified_at": model["modified_at"]
                        }
                        for model in models
                    ]
                    
                    logger.info(f"Ollama has {len(models)} models loaded")
                    
                    # Store model info in Redis
                    await self.redis_client.hset(
                        "ollama:models",
                        "list",
                        json.dumps(self.ollama_config["models"], default=str)
                    )
                    
        except Exception as e:
            logger.error(f"Failed to manage Ollama models: {e}")
    
    async def _optimize_ollama_performance(self):
        """Optimize Ollama performance based on usage patterns"""
        
        try:
            if not self.ollama_config["model_optimization"]:
                return
            
            # Get current resource usage
            ollama_metrics = self.resource_metrics.get("ollama")
            if not ollama_metrics:
                return
            
            # Optimize based on resource usage
            if ollama_metrics.memory_percent > 90:
                logger.warning("Ollama memory usage is high, consider model optimization")
                # Could implement model unloading/swapping here
            
            if ollama_metrics.cpu_percent > 95:
                logger.warning("Ollama CPU usage is high, consider load balancing")
                # Could implement request throttling here
                
        except Exception as e:
            logger.error(f"Failed to optimize Ollama performance: {e}")
    
    # ==================== Utility Methods ====================
    
    def _get_container(self, container_name: str) -> Optional[docker.models.containers.Container]:
        """Get container by name"""
        
        try:
            return self.docker_client.containers.get(container_name)
        except docker.errors.NotFound:
            return None
    
    def _get_container_by_service(self, service_name: str) -> Optional[docker.models.containers.Container]:
        """Get container by service name"""
        
        if service_name in self.services:
            container_name = self.services[service_name].container_spec.name
            return self._get_container(container_name)
        return None
    
    def _get_service_containers(self, service_name: str) -> List[docker.models.containers.Container]:
        """Get all containers for a service (for scaled services)"""
        
        containers = []
        try:
            all_containers = self.docker_client.containers.list(all=True)
            for container in all_containers:
                if service_name in container.name:
                    containers.append(container)
        except Exception as e:
            logger.error(f"Failed to get service containers for {service_name}: {e}")
        
        return containers
    
    async def _wait_for_health(self, container_name: str, timeout: int = 60) -> bool:
        """Wait for container to become healthy"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            container = self._get_container(container_name)
            if not container:
                return False
            
            container.reload()
            
            # Check Docker health status
            health_info = container.attrs.get("State", {}).get("Health", {})
            if health_info:
                status = health_info.get("Status", "none")
                if status == "healthy":
                    return True
                elif status == "unhealthy":
                    return False
            else:
                # If no health check defined, check if container is running
                if container.status == "running":
                    return True
            
            await asyncio.sleep(2)
        
        return False
    
    def _parse_duration(self, duration_str: str) -> int:
        """Parse duration string to nanoseconds"""
        
        if duration_str.endswith('s'):
            return int(float(duration_str[:-1]) * 1_000_000_000)
        elif duration_str.endswith('m'):
            return int(float(duration_str[:-1]) * 60 * 1_000_000_000)
        else:
            return int(duration_str) * 1_000_000_000
    
    async def _deploy_service_instance(self, service_name: str, instance_name: str) -> bool:
        """Deploy a specific instance of a service"""
        
        # Create a copy of the service spec with modified name
        service_spec = self.services[service_name]
        instance_spec = ContainerSpec(
            name=instance_name,
            image=service_spec.container_spec.image,
            ports={},  # Let Docker assign random ports for additional instances
            environment=service_spec.container_spec.environment,
            volumes=service_spec.container_spec.volumes,
            memory_limit=service_spec.container_spec.memory_limit,
            cpu_limit=service_spec.container_spec.cpu_limit,
            health_check=service_spec.container_spec.health_check,
            restart_policy=service_spec.container_spec.restart_policy,
            network=service_spec.container_spec.network,
            labels={
                **service_spec.container_spec.labels,
                "sutazai.instance": instance_name
            }
        )
        
        # Deploy the instance
        try:
            container_config = {
                "image": instance_spec.image,
                "name": instance_spec.name,
                "environment": instance_spec.environment,
                "volumes": instance_spec.volumes,
                "restart_policy": {"Name": instance_spec.restart_policy},
                "network": instance_spec.network,
                "labels": instance_spec.labels,
                "mem_limit": instance_spec.memory_limit,
                "cpu_period": 100000,
                "cpu_quota": int(float(instance_spec.cpu_limit) * 100000)
            }
            
            container = self.docker_client.containers.run(
                detach=True,
                **container_config
            )
            
            logger.info(f"Successfully deployed instance: {instance_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy instance {instance_name}: {e}")
            return False
    
    async def _initialize_monitoring(self):
        """Initialize monitoring integrations"""
        
        try:
            # Initialize Prometheus metrics if available
            # Initialize Grafana dashboards if available
            # Initialize Loki logging if available
            
            logger.info("Monitoring integrations initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize monitoring: {e}")
    
    # ==================== Public API Methods ====================
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        total_services = len(self.services)
        healthy_services = len([s for s in self.health_status.values() if s.status == "healthy"])
        running_containers = len([c for c in self.containers.values() if c.status == "running"])
        
        return {
            "total_services": total_services,
            "healthy_services": healthy_services,
            "running_containers": running_containers,
            "health_ratio": healthy_services / max(total_services, 1),
            "ollama_status": await self._check_ollama_health(),
            "ollama_models": len(self.ollama_config.get("models", [])),
            "auto_scaling_enabled": self.config["auto_scaling_enabled"],
            "auto_healing_enabled": self.config["auto_healing_enabled"],
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_service_status(self, service_name: Optional[str] = None) -> Dict[str, Any]:
        """Get service status information"""
        
        if service_name:
            if service_name not in self.services:
                return {"error": f"Unknown service: {service_name}"}
            
            health = self.health_status.get(service_name, HealthStatus("unknown", datetime.now(), 0.0))
            metrics = self.resource_metrics.get(service_name)
            
            return {
                "name": service_name,
                "health_status": health.status,
                "last_check": health.last_check.isoformat(),
                "response_time_ms": health.response_time_ms,
                "consecutive_failures": health.consecutive_failures,
                "cpu_percent": metrics.cpu_percent if metrics else 0.0,
                "memory_percent": metrics.memory_percent if metrics else 0.0,
                "memory_usage_mb": metrics.memory_usage_mb if metrics else 0.0
            }
        else:
            return {
                service_name: {
                    "health_status": health.status,
                    "last_check": health.last_check.isoformat(),
                    "cpu_percent": self.resource_metrics.get(service_name, ResourceMetrics(0, 0, 0, 0, {})).cpu_percent,
                    "memory_percent": self.resource_metrics.get(service_name, ResourceMetrics(0, 0, 0, 0, {})).memory_percent
                }
                for service_name, health in self.health_status.items()
            }
    
    async def deploy_all_agents(self) -> Dict[str, bool]:
        """Deploy all AI agent services"""
        
        results = {}
        
        # Deploy core infrastructure first
        core_services = ["redis", "postgres", "ollama"]
        for service_name in core_services:
            if service_name in self.services:
                results[service_name] = await self.deploy_service(service_name)
        
        # Deploy all AI agents
        agent_services = [name for name, spec in self.services.items() if spec.type == "ai_agent"]
        
        # Deploy agents in parallel
        deployment_tasks = []
        for service_name in agent_services:
            deployment_tasks.append(
                asyncio.create_task(self.deploy_service(service_name))
            )
        
        # Wait for all deployments to complete
        deployment_results = await asyncio.gather(*deployment_tasks, return_exceptions=True)
        
        for service_name, result in zip(agent_services, deployment_results):
            results[service_name] = isinstance(result, bool) and result
        
        return results


# ==================== Factory Function ====================

def create_infrastructure_integration(redis_url: str = "redis://redis:6379") -> InfrastructureIntegration:
    """Factory function to create infrastructure integration"""
    return InfrastructureIntegration(redis_url)


# ==================== Example Usage ====================

async def example_infrastructure_management():
    """Example of using the infrastructure integration"""
    
    # Initialize infrastructure integration
    integration = create_infrastructure_integration("redis://redis:6379")
    await integration.initialize()
    
    # Deploy core services
    await integration.deploy_service("redis")
    await integration.deploy_service("postgres")
    await integration.deploy_service("ollama")
    
    # Deploy all AI agents
    results = await integration.deploy_all_agents()
    print(f"Agent deployment results: {results}")
    
    # Get system status
    status = await integration.get_system_status()
    print(f"System status: {status}")
    
    # Scale a service
    await integration.scale_service("senior-ai-engineer", 2)
    
    await integration.shutdown()


if __name__ == "__main__":
    asyncio.run(example_infrastructure_management())