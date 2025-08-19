"""
DinD-to-Mesh Bridge
Connects Docker-in-Docker MCP orchestrator with the service mesh
Enables multi-client access through proper isolation and port mapping
"""
import asyncio
import logging
import httpx
import json
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import docker
from docker.errors import DockerException, NotFound

from .service_mesh import ServiceMesh, ServiceInstance, ServiceState
from .mcp_protocol_translator import get_protocol_translator

logger = logging.getLogger(__name__)

# Port mapping configuration
DIND_BASE_PORT = 11100  # Base port for mesh-accessible MCP services
DIND_PORT_RANGE = 100    # Support up to 100 MCP containers
# Try both possible hostnames for DinD
DIND_HOST = os.getenv("DIND_HOST", "sutazai-mcp-orchestrator-notls")  
DIND_API_PORT = 2375   # DinD Docker API port (internal port, no TLS)
DIND_MANAGER_PORT = 18081  # MCP Manager API port

@dataclass
class DinDMCPService:
    """Represents an MCP service running in DinD"""
    name: str
    container_id: str
    internal_port: int  # Port inside DinD
    mesh_port: int      # Port exposed to mesh (11100-11199)
    protocol: str = "stdio"  # stdio, http, grpc
    state: ServiceState = ServiceState.UNKNOWN
    labels: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_health_check: Optional[datetime] = None
    clients: List[str] = field(default_factory=list)  # Track connected clients

class DinDMeshBridge:
    """
    Bridge between Docker-in-Docker MCP orchestrator and service mesh
    Handles port mapping, service discovery, and multi-client access
    """
    
    def __init__(self, mesh: ServiceMesh, dind_host: str = DIND_HOST):
        self.mesh = mesh
        self.dind_host = dind_host
        self.dind_client: Optional[docker.DockerClient] = None
        self.mcp_services: Dict[str, DinDMCPService] = {}
        self.port_allocations: Dict[int, str] = {}  # mesh_port -> service_name
        self.next_port = DIND_BASE_PORT
        self.protocol_translator = None
        self.initialized = False
        self.monitor_task: Optional[asyncio.Task] = None
        # Add registry attribute for compatibility
        self.registry = {"mcp_services": []}
        
    async def initialize(self) -> bool:
        """
        Initialize connection to DinD orchestrator
        """
        try:
            logger.info(f"Initializing DinD-Mesh Bridge to {self.dind_host}...")
            
            # CRITICAL FIX: Try multiple connection methods for DinD
            connection_attempts = [
                # Method 1: Host-mapped port (most reliable from backend container on Linux)
                "tcp://172.17.0.1:12375",
                # Method 2: Direct container network connection (if dockerd listens on bridge)
                f"tcp://{self.dind_host}:2375",
                # Method 3: Known orchestrator container name on same network
                "tcp://sutazai-mcp-orchestrator:2375",
                # Method 4: Alternative hostname
                "tcp://sutazai-mcp-orchestrator-notls:2375",
                # Method 5: Host-mapped port via host.docker.internal (Docker Desktop)
                "tcp://host.docker.internal:12375"
            ]

            last_error = None
            for attempt, base_url in enumerate(connection_attempts, 1):
                try:
                    logger.info(f"DinD connection attempt {attempt}/{len(connection_attempts)}: {base_url}")
                    self.dind_client = docker.DockerClient(
                        base_url=base_url,
                        timeout=10  # Shorter timeout for faster fallback
                    )

                    # Test connection
                    version = self.dind_client.version()
                    logger.info(f"✅ Connected to DinD Docker v{version.get('Version','?')} via {base_url}")
                    break  # Success! Exit connection loop

                except Exception as e:
                    last_error = e
                    # Detailed diagnostics for failure
                    explanation = getattr(e, 'explanation', None)
                    err_cls = e.__class__.__name__
                    logger.warning(
                        f"DinD connection attempt {attempt} failed: class={err_cls} repr={e!r} "
                        f"explanation={explanation} base_url={base_url}"
                    )
                    if self.dind_client:
                        try:
                            self.dind_client.close()
                        except Exception as ce:
                            logger.debug(f"Error closing failed DockerClient: {ce!r}")
                        self.dind_client = None
                    continue

            # Check if any connection succeeded
            if not self.dind_client:
                logger.error(
                    f"All DinD connection attempts failed. last_error_class={getattr(last_error,'__class__',type(last_error)).__name__} "
                    f"last_error_repr={last_error!r}"
                )
                raise Exception(f"All DinD connection attempts failed. Last error: {last_error}")
            
            # Initialize protocol translator for STDIO MCPs
            self.protocol_translator = await get_protocol_translator()
            
            # Discover existing MCP containers
            await self.discover_mcp_containers()
            
            # Start monitoring task
            self.monitor_task = asyncio.create_task(self._monitor_containers())
            
            self.initialized = True
            logger.info(f"✅ DinD-Mesh Bridge initialized with {len(self.mcp_services)} MCP services")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize DinD-Mesh Bridge: {e}")
            return False
    
    async def discover_mcp_containers(self) -> List[DinDMCPService]:
        """
        Discover MCP containers running in DinD
        """
        if not self.dind_client:
            logger.error("DinD client not initialized")
            return []
        
        try:
            # CRITICAL FIX: Use broader filter to discover MCP containers
            # Try multiple discovery methods since container labels may vary
            containers = []
            
            # Method 1: Try original filter first
            try:
                containers = self.dind_client.containers.list(
                    filters={"label": "mcp.service=true"}
                )
                if containers:
                    logger.info(f"Found {len(containers)} containers with mcp.service=true label")
            except Exception as e:
                logger.warning(f"Method 1 (mcp.service=true) failed: {e}")
            
            # Method 2: Try broader name filter if no labeled containers found
            if not containers:
                try:
                    all_containers = self.dind_client.containers.list(all=True)
                    containers = [c for c in all_containers if c.name.startswith('mcp-')]
                    logger.info(f"Found {len(containers)} containers with mcp- name prefix")
                except Exception as e:
                    logger.warning(f"Method 2 (name prefix) failed: {e}")
            
            # Method 3: If still no containers, try all running containers
            if not containers:
                try:
                    all_containers = self.dind_client.containers.list()
                    logger.info(f"Total running containers in DinD: {len(all_containers)}")
                    # Log container names for debugging
                    for container in all_containers[:5]:  # Show first 5
                        logger.info(f"  Container: {container.name} (labels: {container.labels})")
                    # Use all running containers as MCP containers for now
                    containers = all_containers
                except Exception as e:
                    logger.warning(f"Method 3 (all containers) failed: {e}")
            
            discovered = []
            for container in containers:
                service = await self._register_container(container)
                if service:
                    discovered.append(service)
            
            logger.info(f"Discovered {len(discovered)} MCP containers in DinD")
            return discovered
            
        except Exception as e:
            logger.error(f"Failed to discover MCP containers: {e}")
            return []
    
    async def _register_container(self, container) -> Optional[DinDMCPService]:
        """
        Register a DinD container with the mesh
        """
        try:
            # Extract MCP metadata from container labels or infer from name
            labels = container.labels
            
            # CRITICAL FIX: Handle containers without explicit MCP labels
            if container.name.startswith('mcp-'):
                # Strip 'mcp-' prefix to get service name
                mcp_name = container.name[4:]  # Remove 'mcp-' prefix
            else:
                mcp_name = labels.get("mcp.name", container.name)
            
            mcp_protocol = labels.get("mcp.protocol", "stdio")  # Default to stdio
            mcp_port = int(labels.get("mcp.port", "8080"))  # Default port for HTTP MCPs
            
            # Allocate mesh port
            mesh_port = self._allocate_port(mcp_name)
            
            # Create service record
            service = DinDMCPService(
                name=mcp_name,
                container_id=container.id,
                internal_port=mcp_port,
                mesh_port=mesh_port,
                protocol=mcp_protocol,
                state=ServiceState.HEALTHY if container.status == "running" else ServiceState.UNHEALTHY,
                labels=labels,
                metadata={
                    "container_name": container.name,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                    "created": container.attrs['Created'],
                    "dind_host": self.dind_host
                }
            )
            
            # Register with mesh
            await self._register_with_mesh(service)
            
            # Store service
            self.mcp_services[mcp_name] = service
            
            # Update registry for compatibility (avoid duplicates)
            if not any(s.get("name") == mcp_name for s in self.registry["mcp_services"]):
                self.registry["mcp_services"].append({
                    "name": mcp_name,
                    "tags": ["mcp", mcp_protocol],
                    "metadata": {"capabilities": [], "mesh_port": mesh_port}
                })
            
            logger.info(f"Registered MCP service {mcp_name} on mesh port {mesh_port}")
            return service
            
        except Exception as e:
            logger.error(f"Failed to register container {container.name}: {e}")
            return None
    
    def _allocate_port(self, service_name: str) -> int:
        """
        Allocate a mesh-accessible port for MCP service
        """
        # Check if service already has a port
        if service_name in self.mcp_services:
            return self.mcp_services[service_name].mesh_port
        
        # Find next available port
        while self.next_port < DIND_BASE_PORT + DIND_PORT_RANGE:
            if self.next_port not in self.port_allocations:
                self.port_allocations[self.next_port] = service_name
                allocated = self.next_port
                self.next_port += 1
                return allocated
            self.next_port += 1
        
        raise ValueError(f"No available ports for MCP service {service_name}")
    
    async def _register_with_mesh(self, service: DinDMCPService) -> bool:
        """
        Register MCP service with the service mesh
        """
        try:
            instance = await self.mesh.register_service(
                service_name=f"mcp-{service.name}",
                address=self.dind_host,  # DinD host
                port=service.mesh_port,
                tags=[
                    "mcp",
                    service.name,
                    f"protocol-{service.protocol}",
                    "dind-isolated",
                    "multi-client"
                ],
                metadata={
                    **service.metadata,
                    "internal_port": service.internal_port,
                    "container_id": service.container_id,
                    "mesh_port": service.mesh_port
                }
            )
            
            logger.info(f"✅ Registered {service.name} with mesh on port {service.mesh_port}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register {service.name} with mesh: {e}")
            return False
    
    async def deploy_mcp(self, mcp_config: Dict[str, Any]) -> Optional[DinDMCPService]:
        """
        Deploy a new MCP container in DinD
        """
        if not self.dind_client:
            logger.error("DinD client not initialized")
            return None
        
        try:
            # Prepare container configuration
            mcp_name = mcp_config.get("name")
            image = mcp_config.get("image", "mcp-base:latest")
            environment = mcp_config.get("environment", {})
            volumes = mcp_config.get("volumes", {})
            
            # Add MCP labels
            labels = {
                "mcp.service": "true",
                "mcp.name": mcp_name,
                "mcp.protocol": mcp_config.get("protocol", "stdio"),
                "mcp.port": str(mcp_config.get("port", 0)),
                "mcp.deployed_by": "dind-mesh-bridge"
            }
            
            # Create container in DinD
            container = self.dind_client.containers.run(
                image=image,
                name=f"mcp-{mcp_name}",
                environment=environment,
                volumes=volumes,
                labels=labels,
                network="sutazai-dind-internal",
                detach=True,
                auto_remove=False
            )
            
            logger.info(f"Deployed MCP container {mcp_name} in DinD")
            
            # Register with mesh
            service = await self._register_container(container)
            
            return service
            
        except Exception as e:
            logger.error(f"Failed to deploy MCP {mcp_config.get('name')}: {e}")
            return None
    
    async def handle_client_request(
        self,
        service_name: str,
        client_id: str,
        request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Handle request from a client (Claude Code or Codex) to an MCP service
        Ensures proper isolation and multi-client access
        """
        if service_name not in self.mcp_services:
            return {"error": f"MCP service {service_name} not found"}
        
        service = self.mcp_services[service_name]
        
        # Track client connection
        if client_id not in service.clients:
            service.clients.append(client_id)
            logger.info(f"Client {client_id} connected to {service_name}")
        
        try:
            # Route based on protocol
            if service.protocol == "stdio":
                # Use protocol translator for STDIO MCPs
                response = await self.protocol_translator.translate_to_stdio(
                    service_name=service_name,
                    container_id=service.container_id,
                    request=request
                )
            elif service.protocol == "http":
                # Direct HTTP call to container
                async with httpx.AsyncClient() as client:
                    url = f"http://{self.dind_host}:{service.internal_port}"
                    response = await client.post(url, json=request, timeout=30.0)
                    response = response.json()
            else:
                response = {"error": f"Unsupported protocol: {service.protocol}"}
            
            return {
                "service": service_name,
                "client": client_id,
                "response": response,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to handle request from {client_id} to {service_name}: {e}")
            return {"error": str(e)}
    
    async def _monitor_containers(self):
        """
        Monitor DinD containers and update mesh registration
        """
        while self.initialized:
            try:
                await asyncio.sleep(30)  # Check every 30 seconds
                
                if not self.dind_client:
                    continue
                
                # Get current containers
                containers = self.dind_client.containers.list(
                    filters={"label": "mcp.service=true"}
                )
                
                current_names = {c.labels.get("mcp.name", c.name) for c in containers}
                registered_names = set(self.mcp_services.keys())
                
                # Find new containers
                new_containers = current_names - registered_names
                for name in new_containers:
                    container = next(c for c in containers if c.labels.get("mcp.name", c.name) == name)
                    await self._register_container(container)
                
                # Find removed containers
                removed_containers = registered_names - current_names
                for name in removed_containers:
                    await self._deregister_service(name)
                
                # Update health status
                for container in containers:
                    name = container.labels.get("mcp.name", container.name)
                    if name in self.mcp_services:
                        service = self.mcp_services[name]
                        service.state = ServiceState.HEALTHY if container.status == "running" else ServiceState.UNHEALTHY
                        service.last_health_check = datetime.now()
                
            except Exception as e:
                logger.error(f"Error in container monitor: {e}")
    
    async def _deregister_service(self, service_name: str):
        """
        Deregister MCP service from mesh
        """
        if service_name not in self.mcp_services:
            return
        
        service = self.mcp_services[service_name]
        
        try:
            # Deregister from mesh
            await self.mesh.discovery.deregister_service(f"mcp-{service_name}")
            
            # Free port allocation
            if service.mesh_port in self.port_allocations:
                del self.port_allocations[service.mesh_port]
            
            # Remove service record
            del self.mcp_services[service_name]
            
            logger.info(f"Deregistered MCP service {service_name}")
            
        except Exception as e:
            logger.error(f"Failed to deregister {service_name}: {e}")
    
    def get_service_status(self) -> Dict[str, Any]:
        """
        Get status of all MCP services in DinD
        """
        return {
            "total": len(self.mcp_services),
            "healthy": sum(1 for s in self.mcp_services.values() if s.state == ServiceState.HEALTHY),
            "services": {
                name: {
                    "state": service.state.value,
                    "mesh_port": service.mesh_port,
                    "internal_port": service.internal_port,
                    "protocol": service.protocol,
                    "clients": service.clients,
                    "container_id": service.container_id[:12]
                }
                for name, service in self.mcp_services.items()
            },
            "port_allocations": self.port_allocations,
            "dind_host": self.dind_host,
            "initialized": self.initialized
        }
    
    async def health_check_all(self) -> Dict[str, Any]:
        """
        Check health of all MCP services
        """
        health_report = {
            "services": {},
            "summary": {
                "total": len(self.mcp_services),
                "healthy": 0,
                "unhealthy": 0,
                "percentage_healthy": 0.0
            }
        }
        
        for name, service in self.mcp_services.items():
            is_healthy = service.state == ServiceState.HEALTHY
            health_report["services"][name] = {
                "healthy": is_healthy,
                "available": is_healthy,
                "process_running": is_healthy,
                "retry_count": 0,
                "last_check": service.last_health_check.isoformat() if service.last_health_check else None
            }
            if is_healthy:
                health_report["summary"]["healthy"] += 1
            else:
                health_report["summary"]["unhealthy"] += 1
        
        if health_report["summary"]["total"] > 0:
            health_report["summary"]["percentage_healthy"] = (
                health_report["summary"]["healthy"] / health_report["summary"]["total"] * 100
            )
        
        return health_report
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """
        Get status of a specific MCP service
        """
        if service_name not in self.mcp_services:
            return {"status": "not_found", "error": f"Service {service_name} not found"}
        
        service = self.mcp_services[service_name]
        return {
            "service": service_name,
            "status": service.state.value,
            "mesh_instances": 1,
            "adapter_instances": 0,
            "instances": [{
                "id": service.container_id[:12],
                "mesh_port": service.mesh_port,
                "internal_port": service.internal_port,
                "state": service.state.value
            }],
            "mesh_registration": True
        }
    
    async def call_mcp_service(self, service_name: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a method on an MCP service
        """
        if service_name not in self.mcp_services:
            raise Exception(f"Service {service_name} not found")
        
        # Route through handle_client_request for consistency
        return await self.handle_client_request(
            service_name=service_name,
            client_id="api-client",
            request={"method": method, "params": params}
        )
    
    async def restart_service(self, service_name: str) -> bool:
        """
        Restart an MCP service
        """
        if service_name not in self.mcp_services or not self.dind_client:
            return False
        
        try:
            service = self.mcp_services[service_name]
            container = self.dind_client.containers.get(service.container_id)
            container.restart()
            return True
        except Exception as e:
            logger.error(f"Failed to restart {service_name}: {e}")
            return False
    
    async def stop_service(self, service_name: str) -> bool:
        """
        Stop an MCP service
        """
        if service_name not in self.mcp_services or not self.dind_client:
            return False
        
        try:
            service = self.mcp_services[service_name]
            container = self.dind_client.containers.get(service.container_id)
            container.stop()
            await self._deregister_service(service_name)
            return True
        except Exception as e:
            logger.error(f"Failed to stop {service_name}: {e}")
            return False
    
    
    async def shutdown(self):
        """
        Shutdown the bridge and cleanup
        """
        logger.info("Shutting down DinD-Mesh Bridge...")
        
        # Stop monitor task
        if self.monitor_task:
            self.monitor_task.cancel()
        
        # Deregister all services
        for service_name in list(self.mcp_services.keys()):
            await self._deregister_service(service_name)
        
        # Close DinD client
        if self.dind_client:
            self.dind_client.close()
        
        self.initialized = False
        logger.info("DinD-Mesh Bridge shutdown complete")

# Global instance
_dind_bridge: Optional[DinDMeshBridge] = None

async def get_dind_bridge(mesh: Optional[ServiceMesh] = None) -> DinDMeshBridge:
    """Get or create DinD-Mesh bridge instance"""
    global _dind_bridge
    
    if _dind_bridge is None:
        if mesh is None:
            from .service_mesh import get_mesh
            mesh = await get_mesh()
        
        _dind_bridge = DinDMeshBridge(mesh)
        await _dind_bridge.initialize()
    
    return _dind_bridge