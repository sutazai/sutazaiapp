"""
MCP Process Orchestrator
Manages sequential MCP startup with proper dependency management and health checks
Implements production-ready orchestration patterns
"""
import asyncio
import logging
import os
import json
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import networkx as nx

from .mcp_protocol_translator import get_protocol_translator, MCPProtocolTranslator
from .mcp_resource_isolation import get_resource_manager, MCPResourceIsolationManager
from .service_mesh import ServiceMesh, ServiceInstance

logger = logging.getLogger(__name__)

class MCPServiceState(Enum):
    """MCP service lifecycle states"""
    PENDING = "pending"
    STARTING = "starting"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"
    STOPPING = "stopping"
    STOPPED = "stopped"

@dataclass
class MCPServiceDefinition:
    """Definition of an MCP service with dependencies"""
    name: str
    wrapper_script: str
    port: int
    dependencies: List[str] = field(default_factory=list)
    health_check_path: str = "/health"
    startup_timeout: float = 30.0
    startup_delay: float = 0.0
    required: bool = False
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    memory_limit_mb: int = 512
    cpu_limit_percent: float = 25.0
    max_retries: int = 3
    retry_delay: float = 5.0

@dataclass
class MCPServiceStatus:
    """Current status of an MCP service"""
    name: str
    state: MCPServiceState
    port: int
    started_at: Optional[datetime] = None
    last_health_check: Optional[datetime] = None
    health_check_failures: int = 0
    retry_count: int = 0
    error_message: Optional[str] = None
    dependencies_met: bool = False
    resource_allocated: bool = False

class MCPDependencyResolver:
    """Resolves and manages MCP service dependencies"""
    
    def __init__(self):
        self.dependency_graph = nx.DiGraph()
        
    def add_service(self, service: MCPServiceDefinition):
        """Add service to dependency graph"""
        self.dependency_graph.add_node(service.name, definition=service)
        
        for dep in service.dependencies:
            self.dependency_graph.add_edge(dep, service.name)
    
    def get_startup_order(self) -> List[List[str]]:
        """
        Get startup order respecting dependencies
        Returns list of service groups that can start in parallel
        """
        if not self.dependency_graph:
            return []
        
        # Check for cycles
        if not nx.is_directed_acyclic_graph(self.dependency_graph):
            cycles = list(nx.simple_cycles(self.dependency_graph))
            raise ValueError(f"Dependency cycles detected: {cycles}")
        
        # Get topological generations (levels that can run in parallel)
        try:
            generations = list(nx.topological_generations(self.dependency_graph))
            return generations
        except nx.NetworkXError as e:
            logger.error(f"Error resolving dependencies: {e}")
            # Fallback to individual services
            return [[node] for node in self.dependency_graph.nodes()]
    
    def check_dependencies(self, service_name: str, running_services: Set[str]) -> bool:
        """Check if service dependencies are met"""
        if service_name not in self.dependency_graph:
            return True
        
        dependencies = list(self.dependency_graph.predecessors(service_name))
        return all(dep in running_services for dep in dependencies)
    
    def get_dependents(self, service_name: str) -> List[str]:
        """Get services that depend on this service"""
        if service_name not in self.dependency_graph:
            return []
        
        return list(self.dependency_graph.successors(service_name))

class MCPProcessOrchestrator:
    """
    Orchestrates MCP process lifecycle with proper dependency management
    Implements sequential startup, health monitoring, and graceful shutdown
    """
    
    def __init__(self, mesh: Optional[ServiceMesh] = None):
        self.mesh = mesh
        self.services: Dict[str, MCPServiceDefinition] = {}
        self.service_status: Dict[str, MCPServiceStatus] = {}
        self.dependency_resolver = MCPDependencyResolver()
        self.protocol_translator: Optional[MCPProtocolTranslator] = None
        self.resource_manager: Optional[MCPResourceIsolationManager] = None
        self.running_services: Set[str] = set()
        self.health_check_tasks: Dict[str, asyncio.Task] = {}
        self._initialized = False
        
    async def initialize(self):
        """Initialize orchestrator components"""
        if self._initialized:
            return
        
        # Initialize components
        self.protocol_translator = await get_protocol_translator()
        self.resource_manager = await get_resource_manager()
        
        # Load service definitions
        await self._load_service_definitions()
        
        self._initialized = True
        logger.info("✅ MCP Process Orchestrator initialized")
    
    async def _load_service_definitions(self):
        """Load MCP service definitions from configuration"""
        # Define service startup order and dependencies
        service_configs = [
            # Core services (no dependencies)
            {
                "name": "files",
                "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/files.sh",
                "port": 11100,
                "dependencies": [],
                "capabilities": ["filesystem", "read", "write"],
                "required": True
            },
            {
                "name": "extended-memory",
                "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh",
                "port": 11101,
                "dependencies": [],
                "capabilities": ["memory", "persistence"],
                "required": True
            },
            
            # Database services (depend on files)
            {
                "name": "postgres",
                "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/postgres.sh",
                "port": 11102,
                "dependencies": ["files"],
                "capabilities": ["database", "sql"],
                "memory_limit_mb": 1024
            },
            
            # Network services (depend on files)
            {
                "name": "http",
                "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh",
                "port": 11103,
                "dependencies": ["files"],
                "capabilities": ["http", "fetch"]
            },
            {
                "name": "ddg",
                "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh",
                "port": 11104,
                "dependencies": ["http"],
                "capabilities": ["search", "web"]
            },
            
            # Development services (depend on core)
            {
                "name": "github",
                "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/github.sh",
                "port": 11105,
                "dependencies": ["files", "http"],
                "capabilities": ["vcs", "github"]
            },
            {
                "name": "language-server",
                "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/language-server.sh",
                "port": 11106,
                "dependencies": ["files"],
                "capabilities": ["lsp", "completion"],
                "memory_limit_mb": 768
            },
            
            # Browser automation (heavy resources)
            {
                "name": "puppeteer-mcp",
                "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/puppeteer-mcp.sh",
                "port": 11107,
                "dependencies": ["files", "http"],
                "capabilities": ["browser", "automation"],
                "memory_limit_mb": 1024,
                "cpu_limit_percent": 50.0,
                "startup_delay": 2.0
            },
            {
                "name": "playwright-mcp",
                "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp.sh",
                "port": 11108,
                "dependencies": ["files", "http"],
                "capabilities": ["browser", "testing"],
                "memory_limit_mb": 1024,
                "cpu_limit_percent": 50.0,
                "startup_delay": 2.0
            },
            
            # Advanced services
            {
                "name": "knowledge-graph-mcp",
                "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/knowledge-graph-mcp.sh",
                "port": 11109,
                "dependencies": ["extended-memory", "postgres"],
                "capabilities": ["graph", "knowledge"],
                "memory_limit_mb": 768
            },
            {
                "name": "sequentialthinking",
                "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/sequentialthinking.sh",
                "port": 11110,
                "dependencies": ["extended-memory"],
                "capabilities": ["reasoning", "analysis"]
            }
        ]
        
        # Create service definitions
        for config in service_configs:
            # Check if wrapper exists
            if not os.path.exists(config["wrapper"]):
                logger.warning(f"Wrapper not found for {config['name']}: {config['wrapper']}")
                continue
            
            service_def = MCPServiceDefinition(
                name=config["name"],
                wrapper_script=config["wrapper"],
                port=config.get("port", 11200),
                dependencies=config.get("dependencies", []),
                required=config.get("required", False),
                capabilities=config.get("capabilities", []),
                metadata=config.get("metadata", {}),
                memory_limit_mb=config.get("memory_limit_mb", 512),
                cpu_limit_percent=config.get("cpu_limit_percent", 25.0),
                startup_delay=config.get("startup_delay", 0.0)
            )
            
            self.services[config["name"]] = service_def
            self.dependency_resolver.add_service(service_def)
            
            # Initialize status
            self.service_status[config["name"]] = MCPServiceStatus(
                name=config["name"],
                state=MCPServiceState.PENDING,
                port=config["port"]
            )
        
        logger.info(f"Loaded {len(self.services)} MCP service definitions")
    
    async def start_all_services(self) -> Dict[str, Any]:
        """
        Start all MCP services in dependency order
        
        Returns:
            Status report with started, failed, and skipped services
        """
        if not self._initialized:
            await self.initialize()
        
        results = {
            "started": [],
            "failed": [],
            "skipped": [],
            "total": len(self.services)
        }
        
        try:
            # Get startup order
            startup_groups = self.dependency_resolver.get_startup_order()
            
            logger.info(f"Starting MCP services in {len(startup_groups)} groups")
            
            # Start services group by group
            for group_idx, service_group in enumerate(startup_groups):
                logger.info(f"Starting group {group_idx + 1}: {service_group}")
                
                # Start services in parallel within group
                tasks = []
                for service_name in service_group:
                    if service_name in self.services:
                        tasks.append(self._start_service(service_name))
                
                if tasks:
                    group_results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # Process results
                    for service_name, result in zip(service_group, group_results):
                        if isinstance(result, Exception):
                            logger.error(f"Failed to start {service_name}: {result}")
                            results["failed"].append(service_name)
                        elif result:
                            results["started"].append(service_name)
                        else:
                            results["failed"].append(service_name)
                
                # Brief delay between groups
                if group_idx < len(startup_groups) - 1:
                    await asyncio.sleep(1.0)
            
            # Handle services not in dependency graph
            for service_name in self.services:
                if service_name not in results["started"] and service_name not in results["failed"]:
                    results["skipped"].append(service_name)
            
            logger.info(f"MCP Service Startup Complete:")
            logger.info(f"  Started: {len(results['started'])}/{results['total']}")
            logger.info(f"  Failed: {len(results['failed'])}/{results['total']}")
            logger.info(f"  Skipped: {len(results['skipped'])}/{results['total']}")
            
        except Exception as e:
            logger.error(f"Error during service startup: {e}")
        
        return results
    
    async def _start_service(self, service_name: str) -> bool:
        """Start a single MCP service"""
        if service_name not in self.services:
            logger.error(f"Service {service_name} not found")
            return False
        
        service_def = self.services[service_name]
        status = self.service_status[service_name]
        
        try:
            # Update state
            status.state = MCPServiceState.STARTING
            
            # Check dependencies
            if not self.dependency_resolver.check_dependencies(service_name, self.running_services):
                logger.warning(f"Dependencies not met for {service_name}")
                status.state = MCPServiceState.FAILED
                status.error_message = "Dependencies not met"
                return False
            
            # Apply startup delay if specified
            if service_def.startup_delay > 0:
                await asyncio.sleep(service_def.startup_delay)
            
            # Allocate resources
            allocation = await self.resource_manager.allocate_resources(
                service_name=service_name,
                preferred_port=service_def.port,
                memory_limit_mb=service_def.memory_limit_mb,
                cpu_limit_percent=service_def.cpu_limit_percent
            )
            
            if not allocation:
                logger.error(f"Failed to allocate resources for {service_name}")
                status.state = MCPServiceState.FAILED
                status.error_message = "Resource allocation failed"
                return False
            
            status.resource_allocated = True
            status.port = allocation.port
            
            # Register with protocol translator
            success = await self.protocol_translator.register_mcp_service(
                service_name=service_name,
                wrapper_script=service_def.wrapper_script,
                port=allocation.port
            )
            
            if not success:
                logger.error(f"Failed to start {service_name}")
                await self.resource_manager.release_resources(service_name)
                status.state = MCPServiceState.FAILED
                status.error_message = "Process start failed"
                return False
            
            # Register with mesh if available
            if self.mesh:
                try:
                    await self.mesh.register_service(
                        service_name=f"mcp-{service_name}",
                        address="localhost",
                        port=allocation.port,
                        tags=["mcp", service_name] + service_def.capabilities,
                        metadata={
                            "wrapper": service_def.wrapper_script,
                            "capabilities": service_def.capabilities,
                            **service_def.metadata
                        }
                    )
                    logger.info(f"Registered {service_name} with service mesh")
                except Exception as e:
                    logger.warning(f"Failed to register {service_name} with mesh: {e}")
            
            # Update status
            status.state = MCPServiceState.HEALTHY
            status.started_at = datetime.now()
            status.dependencies_met = True
            self.running_services.add(service_name)
            
            # Start health monitoring
            self.health_check_tasks[service_name] = asyncio.create_task(
                self._monitor_service_health(service_name)
            )
            
            logger.info(f"✅ Started MCP service {service_name} on port {allocation.port}")
            return True
            
        except Exception as e:
            logger.error(f"Error starting {service_name}: {e}")
            status.state = MCPServiceState.FAILED
            status.error_message = str(e)
            
            # Cleanup on failure
            if status.resource_allocated:
                await self.resource_manager.release_resources(service_name)
            
            return False
    
    async def _monitor_service_health(self, service_name: str):
        """Monitor health of a running service"""
        if service_name not in self.services:
            return
        
        service_def = self.services[service_name]
        status = self.service_status[service_name]
        
        while service_name in self.running_services:
            try:
                await asyncio.sleep(10)  # Check every 10 seconds
                
                # Get health status
                health = await self.protocol_translator.get_service_health(service_name)
                
                if health["healthy"]:
                    status.last_health_check = datetime.now()
                    status.health_check_failures = 0
                    
                    if status.state == MCPServiceState.DEGRADED:
                        status.state = MCPServiceState.HEALTHY
                        logger.info(f"Service {service_name} recovered to healthy state")
                else:
                    status.health_check_failures += 1
                    
                    if status.health_check_failures >= 3:
                        status.state = MCPServiceState.DEGRADED
                        logger.warning(f"Service {service_name} degraded after {status.health_check_failures} failures")
                    
                    if status.health_check_failures >= 5:
                        # Attempt restart
                        if status.retry_count < service_def.max_retries:
                            logger.info(f"Attempting to restart {service_name}")
                            await self.restart_service(service_name)
                        else:
                            status.state = MCPServiceState.FAILED
                            logger.error(f"Service {service_name} failed after {status.retry_count} retries")
                            self.running_services.discard(service_name)
                            break
                
            except Exception as e:
                logger.error(f"Error monitoring {service_name}: {e}")
    
    async def stop_service(self, service_name: str) -> bool:
        """Stop a single MCP service"""
        if service_name not in self.services:
            return False
        
        status = self.service_status[service_name]
        
        try:
            # Update state
            status.state = MCPServiceState.STOPPING
            
            # Check dependents
            dependents = self.dependency_resolver.get_dependents(service_name)
            running_dependents = [d for d in dependents if d in self.running_services]
            
            if running_dependents:
                logger.warning(f"Service {service_name} has running dependents: {running_dependents}")
                # Could implement cascading shutdown here
            
            # Cancel health monitoring
            if service_name in self.health_check_tasks:
                self.health_check_tasks[service_name].cancel()
                del self.health_check_tasks[service_name]
            
            # Unregister from protocol translator
            await self.protocol_translator.unregister_mcp_service(service_name)
            
            # Release resources
            await self.resource_manager.release_resources(service_name)
            
            # Update status
            status.state = MCPServiceState.STOPPED
            self.running_services.discard(service_name)
            
            logger.info(f"✅ Stopped MCP service {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping {service_name}: {e}")
            return False
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart an MCP service"""
        status = self.service_status[service_name]
        status.retry_count += 1
        
        # Stop the service
        await self.stop_service(service_name)
        
        # Wait before restart
        await asyncio.sleep(self.services[service_name].retry_delay)
        
        # Start the service
        return await self._start_service(service_name)
    
    async def stop_all_services(self):
        """Stop all MCP services in reverse dependency order"""
        # Get reverse startup order
        startup_groups = self.dependency_resolver.get_startup_order()
        
        # Stop in reverse order
        for service_group in reversed(startup_groups):
            tasks = []
            for service_name in service_group:
                if service_name in self.running_services:
                    tasks.append(self.stop_service(service_name))
            
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
        
        logger.info("✅ Stopped all MCP services")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        return {
            "services": {
                name: {
                    "state": status.state.value,
                    "port": status.port,
                    "started_at": status.started_at.isoformat() if status.started_at else None,
                    "last_health_check": status.last_health_check.isoformat() if status.last_health_check else None,
                    "health_check_failures": status.health_check_failures,
                    "retry_count": status.retry_count,
                    "dependencies_met": status.dependencies_met,
                    "error": status.error_message
                }
                for name, status in self.service_status.items()
            },
            "summary": {
                "total": len(self.services),
                "running": len(self.running_services),
                "healthy": sum(1 for s in self.service_status.values() if s.state == MCPServiceState.HEALTHY),
                "degraded": sum(1 for s in self.service_status.values() if s.state == MCPServiceState.DEGRADED),
                "failed": sum(1 for s in self.service_status.values() if s.state == MCPServiceState.FAILED)
            }
        }

# Global orchestrator instance
_orchestrator: Optional[MCPProcessOrchestrator] = None

async def get_orchestrator(mesh: Optional[ServiceMesh] = None) -> MCPProcessOrchestrator:
    """Get or create process orchestrator"""
    global _orchestrator
    
    if _orchestrator is None:
        _orchestrator = MCPProcessOrchestrator(mesh)
        await _orchestrator.initialize()
    
    return _orchestrator