#!/usr/bin/env python3
"""
MCP Service Registry

Dynamic service discovery and management system for MCP servers and automation
components. Provides health monitoring, dependency tracking, and service coordination
with automatic failure detection and recovery mechanisms.

Author: Claude AI Assistant (ai-agent-orchestrator)
Created: 2025-08-15 11:54:00 UTC
Version: 1.0.0
"""

import asyncio
import json
import subprocess
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
import logging
import aiohttp

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import MCPAutomationConfig


class ServiceType(Enum):
    """Service types in the MCP ecosystem."""
    MCP_SERVER = "mcp-server"
    AUTOMATION = "automation"
    MONITORING = "monitoring"
    DATABASE = "database"
    API = "api"
    AGENT = "agent"
    UTILITY = "utility"


class ServiceStatus(Enum):
    """Service health status."""
    UNKNOWN = "unknown"
    INITIALIZING = "initializing"
    READY = "ready"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    ERROR = "error"
    STOPPED = "stopped"
    MAINTENANCE = "maintenance"


@dataclass
class ServiceInfo:
    """Service registration information."""
    name: str
    type: str
    version: str
    endpoint: str
    capabilities: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    status: ServiceStatus = ServiceStatus.UNKNOWN
    last_heartbeat: Optional[datetime] = None
    health_check_url: Optional[str] = None
    health_check_interval: int = 30
    startup_timeout: int = 60
    shutdown_timeout: int = 30
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return self.status in [ServiceStatus.READY, ServiceStatus.HEALTHY]
        
    def is_critical(self) -> bool:
        """Check if service is in critical state."""
        return self.status in [ServiceStatus.ERROR, ServiceStatus.STOPPED]


@dataclass
class ServiceHealth:
    """Service health check result."""
    service_name: str
    status: ServiceStatus
    response_time: float
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ServiceDependency:
    """Service dependency information."""
    service: str
    depends_on: str
    required: bool = True
    startup_order: int = 0
    health_check_before_start: bool = True


class ServiceRegistry:
    """
    Central service registry for MCP ecosystem.
    
    Manages service discovery, health monitoring, dependency resolution,
    and lifecycle coordination for all MCP components.
    """
    
    def __init__(self, config: Optional[MCPAutomationConfig] = None):
        """Initialize service registry."""
        self.config = config or MCPAutomationConfig()
        self.logger = self._setup_logging()
        
        # Service storage
        self.services: Dict[str, ServiceInfo] = {}
        self.health_history: Dict[str, List[ServiceHealth]] = {}
        self.dependencies: List[ServiceDependency] = []
        
        # Health monitoring
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        self._health_check_intervals: Dict[str, int] = {}
        
        # Service discovery
        self._discovery_enabled = True
        self._discovery_task: Optional[asyncio.Task] = None
        
        # HTTP session for health checks
        self._session: Optional[aiohttp.ClientSession] = None
        
    def _setup_logging(self) -> logging.Logger:
        """Configure logging."""
        logger = logging.getLogger("mcp.service_registry")
        logger.setLevel(self.config.log_level.value.upper())
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
        
    async def initialize(self) -> None:
        """Initialize service registry."""
        self.logger.info("Initializing service registry...")
        
        # Create HTTP session
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=10)
        )
        
        # Load service definitions
        await self._load_service_definitions()
        
        # Discover MCP servers
        await self._discover_mcp_servers()
        
        # Start discovery task
        if self._discovery_enabled:
            self._discovery_task = asyncio.create_task(self._continuous_discovery())
            
        self.logger.info(f"Service registry initialized with {len(self.services)} services")
        
    async def _load_service_definitions(self) -> None:
        """Load predefined service definitions."""
        # Core services
        core_services = [
            ServiceInfo(
                name="postgresql",
                type=ServiceType.DATABASE.value,
                version="15.0",
                endpoint="postgresql://localhost:10000",
                capabilities=["sql", "transactions", "replication"],
                health_check_url="postgresql://localhost:10000",
                status=ServiceStatus.UNKNOWN
            ),
            ServiceInfo(
                name="redis",
                type=ServiceType.DATABASE.value,
                version="7.0",
                endpoint="redis://localhost:10001",
                capabilities=["cache", "pubsub", "streams"],
                health_check_url="redis://localhost:10001",
                status=ServiceStatus.UNKNOWN
            ),
            ServiceInfo(
                name="backend-api",
                type=ServiceType.API.value,
                version="1.0.0",
                endpoint="http://localhost:10010",
                capabilities=["rest", "websocket", "graphql"],
                health_check_url="http://localhost:10010/health",
                dependencies=["postgresql", "redis"],
                status=ServiceStatus.UNKNOWN
            ),
            ServiceInfo(
                name="ollama",
                type=ServiceType.MCP_SERVER.value,
                version="latest",
                endpoint="http://localhost:10104",
                capabilities=["llm", "embeddings", "chat"],
                health_check_url="http://localhost:10104/api/version",
                status=ServiceStatus.UNKNOWN
            )
        ]
        
        for service in core_services:
            await self.register_service(service)
            
    async def _discover_mcp_servers(self) -> None:
        """Discover MCP servers from configuration."""
        try:
            mcp_config_path = Path("/opt/sutazaiapp/.mcp.json")
            if mcp_config_path.exists():
                with open(mcp_config_path) as f:
                    mcp_config = json.load(f)
                    
                for name, server_config in mcp_config.get("mcpServers", {}).items():
                    # Check if wrapper script exists
                    wrapper_path = Path(f"/opt/sutazaiapp/scripts/mcp/wrappers/{name}.sh")
                    if not wrapper_path.exists():
                        # Try alternate naming
                        wrapper_path = Path(server_config.get("command", ""))
                        
                    service_info = ServiceInfo(
                        name=f"mcp-{name}",
                        type=ServiceType.MCP_SERVER.value,
                        version="unknown",
                        endpoint=str(wrapper_path) if wrapper_path.exists() else server_config.get("command", ""),
                        capabilities=self._infer_capabilities(name),
                        metadata={
                            "config": server_config,
                            "wrapper": str(wrapper_path) if wrapper_path.exists() else None
                        },
                        status=ServiceStatus.UNKNOWN
                    )
                    
                    await self.register_service(service_info)
                    
        except Exception as e:
            self.logger.error(f"MCP server discovery failed: {e}")
            
    def _infer_capabilities(self, server_name: str) -> List[str]:
        """Infer capabilities from server name."""
        capability_map = {
            "github": ["git", "repository", "pr", "issue"],
            "files": ["filesystem", "read", "write"],
            "postgres": ["sql", "database", "query"],
            "http": ["web", "api", "fetch"],
            "browser": ["automation", "scraping", "testing"],
            "memory": ["storage", "persistence", "cache"],
            "language-server": ["lsp", "completion", "diagnostics"],
            "context": ["memory", "state", "history"],
            "sequential": ["planning", "reasoning", "analysis"],
            "ultimate": ["coding", "generation", "refactoring"]
        }
        
        for key, capabilities in capability_map.items():
            if key in server_name.lower():
                return capabilities
                
        return ["general"]
        
    async def register_service(self, service: ServiceInfo) -> None:
        """Register a new service."""
        self.services[service.name] = service
        self.health_history[service.name] = []
        
        self.logger.info(f"Registered service: {service.name} ({service.type})")
        
        # Start health monitoring if URL provided
        if service.health_check_url:
            await self._start_health_monitoring(service)
            
    async def unregister_service(self, service_name: str) -> bool:
        """Unregister a service."""
        if service_name in self.services:
            # Stop health monitoring
            if service_name in self._health_check_tasks:
                self._health_check_tasks[service_name].cancel()
                del self._health_check_tasks[service_name]
                
            del self.services[service_name]
            self.logger.info(f"Unregistered service: {service_name}")
            return True
            
        return False
        
    async def get_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Get service information."""
        return self.services.get(service_name)
        
    async def get_all_services(self) -> List[ServiceInfo]:
        """Get all registered services."""
        return list(self.services.values())
        
    async def get_services_by_type(self, service_type: str) -> List[ServiceInfo]:
        """Get services by type."""
        return [
            s for s in self.services.values()
            if s.type == service_type
        ]
        
    async def get_services_by_capability(self, capability: str) -> List[ServiceInfo]:
        """Get services by capability."""
        return [
            s for s in self.services.values()
            if capability in s.capabilities
        ]
        
    async def update_service_status(
        self,
        service_name: str,
        status: ServiceStatus,
        details: Optional[Dict[str, Any]] = None
    ) -> None:
        """Update service status."""
        if service_name in self.services:
            service = self.services[service_name]
            old_status = service.status
            service.status = status
            service.last_heartbeat = datetime.now(timezone.utc)
            
            if details:
                service.metadata.update(details)
                
            # Log status change
            if old_status != status:
                self.logger.info(
                    f"Service {service_name} status changed: {old_status.value} -> {status.value}"
                )
                
    async def check_health(self) -> List[str]:
        """Check health of all services and return unhealthy ones."""
        unhealthy = []
        
        for service_name, service in self.services.items():
            try:
                health = await self._check_service_health(service)
                
                # Update status
                await self.update_service_status(
                    service_name,
                    health.status,
                    health.details
                )
                
                # Track unhealthy services
                if not service.is_healthy():
                    unhealthy.append(service_name)
                    
                # Store health history
                if service_name in self.health_history:
                    self.health_history[service_name].append(health)
                    # Keep only last 100 entries
                    if len(self.health_history[service_name]) > 100:
                        self.health_history[service_name] = self.health_history[service_name][-100:]
                        
            except Exception as e:
                self.logger.error(f"Health check failed for {service_name}: {e}")
                unhealthy.append(service_name)
                
        return unhealthy
        
    async def _check_service_health(self, service: ServiceInfo) -> ServiceHealth:
        """Check health of a single service."""
        start_time = asyncio.get_event_loop().time()
        
        try:
            if service.health_check_url:
                # HTTP health check
                if service.health_check_url.startswith("http"):
                    async with self._session.get(service.health_check_url) as response:
                        response_time = asyncio.get_event_loop().time() - start_time
                        
                        if response.status == 200:
                            data = await response.json() if response.content_type == "application/json" else {}
                            return ServiceHealth(
                                service_name=service.name,
                                status=ServiceStatus.HEALTHY,
                                response_time=response_time,
                                details=data
                            )
                        else:
                            return ServiceHealth(
                                service_name=service.name,
                                status=ServiceStatus.UNHEALTHY,
                                response_time=response_time,
                                error=f"HTTP {response.status}"
                            )
                            
                # PostgreSQL health check
                elif service.health_check_url.startswith("postgresql"):
                    # Simple connection test
                    result = subprocess.run(
                        ["pg_isready", "-h", "localhost", "-p", "10000"],
                        capture_output=True,
                        timeout=5
                    )
                    response_time = asyncio.get_event_loop().time() - start_time
                    
                    if result.returncode == 0:
                        return ServiceHealth(
                            service_name=service.name,
                            status=ServiceStatus.HEALTHY,
                            response_time=response_time
                        )
                    else:
                        return ServiceHealth(
                            service_name=service.name,
                            status=ServiceStatus.UNHEALTHY,
                            response_time=response_time,
                            error="Connection failed"
                        )
                        
                # Redis health check
                elif service.health_check_url.startswith("redis"):
                    # Simple ping test
                    result = subprocess.run(
                        ["redis-cli", "-h", "localhost", "-p", "10001", "ping"],
                        capture_output=True,
                        timeout=5
                    )
                    response_time = asyncio.get_event_loop().time() - start_time
                    
                    if result.returncode == 0 and b"PONG" in result.stdout:
                        return ServiceHealth(
                            service_name=service.name,
                            status=ServiceStatus.HEALTHY,
                            response_time=response_time
                        )
                    else:
                        return ServiceHealth(
                            service_name=service.name,
                            status=ServiceStatus.UNHEALTHY,
                            response_time=response_time,
                            error="Ping failed"
                        )
                        
            # MCP server health check
            elif service.type == ServiceType.MCP_SERVER.value:
                # Check if wrapper script exists
                wrapper = service.metadata.get("wrapper")
                if wrapper and Path(wrapper).exists():
                    # Try selfcheck
                    result = subprocess.run(
                        [wrapper, "--selfcheck"],
                        capture_output=True,
                        timeout=10
                    )
                    response_time = asyncio.get_event_loop().time() - start_time
                    
                    if result.returncode == 0:
                        return ServiceHealth(
                            service_name=service.name,
                            status=ServiceStatus.HEALTHY,
                            response_time=response_time
                        )
                    else:
                        return ServiceHealth(
                            service_name=service.name,
                            status=ServiceStatus.DEGRADED,
                            response_time=response_time,
                            error="Selfcheck failed"
                        )
                        
            # Default: assume healthy if no check available
            return ServiceHealth(
                service_name=service.name,
                status=ServiceStatus.UNKNOWN,
                response_time=0
            )
            
        except asyncio.TimeoutError:
            return ServiceHealth(
                service_name=service.name,
                status=ServiceStatus.UNHEALTHY,
                response_time=asyncio.get_event_loop().time() - start_time,
                error="Health check timeout"
            )
        except Exception as e:
            return ServiceHealth(
                service_name=service.name,
                status=ServiceStatus.ERROR,
                response_time=asyncio.get_event_loop().time() - start_time,
                error=str(e)
            )
            
    async def _start_health_monitoring(self, service: ServiceInfo) -> None:
        """Start health monitoring for a service."""
        async def monitor():
            while True:
                try:
                    health = await self._check_service_health(service)
                    await self.update_service_status(
                        service.name,
                        health.status,
                        health.details
                    )
                    
                    # Store health history
                    if service.name in self.health_history:
                        self.health_history[service.name].append(health)
                        if len(self.health_history[service.name]) > 100:
                            self.health_history[service.name] = self.health_history[service.name][-100:]
                            
                    await asyncio.sleep(service.health_check_interval)
                    
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    self.logger.error(f"Health monitoring error for {service.name}: {e}")
                    await asyncio.sleep(60)
                    
        task = asyncio.create_task(monitor())
        self._health_check_tasks[service.name] = task
        
    async def _continuous_discovery(self) -> None:
        """Continuously discover new services."""
        while self._discovery_enabled:
            try:
                # Re-discover MCP servers
                await self._discover_mcp_servers()
                
                # Discover running containers
                await self._discover_docker_services()
                
                await asyncio.sleep(300)  # Every 5 minutes
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Service discovery error: {e}")
                await asyncio.sleep(600)
                
    async def _discover_docker_services(self) -> None:
        """Discover services from Docker containers."""
        try:
            result = subprocess.run(
                ["docker", "ps", "--format", "json"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                for line in result.stdout.strip().split('\n'):
                    if line:
                        container = json.loads(line)
                        service_name = container.get("Names", "").replace("sutazai-", "")
                        
                        if service_name and service_name not in self.services:
                            # Auto-register discovered container
                            service_info = ServiceInfo(
                                name=service_name,
                                type=ServiceType.UTILITY.value,
                                version="unknown",
                                endpoint=f"docker://{container.get('ID', '')}",
                                metadata={"container": container},
                                status=ServiceStatus.UNKNOWN
                            )
                            await self.register_service(service_info)
                            
        except Exception as e:
            self.logger.debug(f"Docker discovery skipped: {e}")
            
    async def get_dependency_graph(self) -> Dict[str, List[str]]:
        """Get service dependency graph."""
        graph = {}
        for service in self.services.values():
            graph[service.name] = service.dependencies
        return graph
        
    async def get_startup_order(self) -> List[str]:
        """Get services in startup order based on dependencies."""
        # Topological sort
        visited = set()
        stack = []
        
        def visit(service_name: str):
            if service_name in visited:
                return
            visited.add(service_name)
            
            service = self.services.get(service_name)
            if service:
                for dep in service.dependencies:
                    visit(dep)
            stack.append(service_name)
            
        for service_name in self.services:
            visit(service_name)
            
        return stack
        
    async def wait_for_service(
        self,
        service_name: str,
        timeout: int = 60
    ) -> bool:
        """Wait for service to become healthy."""
        start_time = asyncio.get_event_loop().time()
        
        while (asyncio.get_event_loop().time() - start_time) < timeout:
            service = self.services.get(service_name)
            if service and service.is_healthy():
                return True
            await asyncio.sleep(1)
            
        return False
        
    async def restart_service(self, service_name: str) -> bool:
        """Attempt to restart a service."""
        service = self.services.get(service_name)
        if not service:
            return False
            
        try:
            # Docker container restart
            if "docker://" in service.endpoint:
                container_id = service.endpoint.replace("docker://", "")
                result = subprocess.run(
                    ["docker", "restart", container_id],
                    capture_output=True
                )
                return result.returncode == 0
                
            # MCP server restart
            elif service.type == ServiceType.MCP_SERVER.value:
                wrapper = service.metadata.get("wrapper")
                if wrapper:
                    # Kill existing process
                    subprocess.run(["pkill", "-f", wrapper], capture_output=True)
                    await asyncio.sleep(2)
                    
                    # Start new process
                    subprocess.Popen([wrapper], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    return True
                    
            return False
            
        except Exception as e:
            self.logger.error(f"Failed to restart {service_name}: {e}")
            return False
            
    async def get_service_metrics(self, service_name: str) -> Dict[str, Any]:
        """Get service metrics and statistics."""
        service = self.services.get(service_name)
        if not service:
            return {}
            
        health_history = self.health_history.get(service_name, [])
        
        # Calculate metrics
        total_checks = len(health_history)
        if total_checks == 0:
            return {
                "service": service_name,
                "status": service.status.value,
                "uptime_percentage": 0,
                "average_response_time": 0,
                "total_checks": 0
            }
            
        healthy_checks = sum(1 for h in health_history if h.status == ServiceStatus.HEALTHY)
        response_times = [h.response_time for h in health_history if h.response_time > 0]
        
        return {
            "service": service_name,
            "status": service.status.value,
            "uptime_percentage": (healthy_checks / total_checks) * 100,
            "average_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "total_checks": total_checks,
            "last_check": health_history[-1].timestamp.isoformat() if health_history else None,
            "capabilities": service.capabilities,
            "dependencies": service.dependencies
        }
        
    async def shutdown(self) -> None:
        """Shutdown service registry."""
        self.logger.info("Shutting down service registry...")
        
        # Stop discovery
        self._discovery_enabled = False
        if self._discovery_task:
            self._discovery_task.cancel()
            
        # Stop health monitoring
        for task in self._health_check_tasks.values():
            task.cancel()
            
        # Wait for tasks
        all_tasks = list(self._health_check_tasks.values())
        if self._discovery_task:
            all_tasks.append(self._discovery_task)
        await asyncio.gather(*all_tasks, return_exceptions=True)
        
        # Close HTTP session
        if self._session:
            await self._session.close()
            
        self.logger.info("Service registry shutdown complete")