"""
MCP-Mesh Integration Bridge
Provides seamless integration between Model Context Protocol servers and the service mesh
"""
import asyncio
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib

from app.mesh.service_mesh import ServiceMesh, ServiceRequest, ServiceInstance, ServiceState

logger = logging.getLogger(__name__)

@dataclass
class MCPServiceConfig:
    """Configuration for an MCP service"""
    name: str
    wrapper_script: str
    port: int
    health_endpoint: str = "/health"
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    auto_restart: bool = True
    max_retries: int = 3

class MCPServiceAdapter:
    """Adapter for MCP services to work with the service mesh"""
    
    def __init__(self, config: MCPServiceConfig, mesh: ServiceMesh):
        self.config = config
        self.mesh = mesh
        self.process: Optional[subprocess.Popen] = None
        self.service_instance: Optional[ServiceInstance] = None
        self.retry_count = 0
        self.last_health_check = datetime.now()
        
    async def start(self) -> bool:
        """Start the MCP service and register with mesh"""
        try:
            # Start MCP service process
            self.process = subprocess.Popen(
                [self.config.wrapper_script, "--port", str(self.config.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env={
                    **os.environ,
                    "MCP_SERVICE_NAME": self.config.name,
                    "MCP_SERVICE_PORT": str(self.config.port)
                }
            )
            
            # Wait for service to be ready
            await self._wait_for_ready()
            
            # Register with service mesh
            self.service_instance = await self.mesh.register_service(
                service_name=f"mcp-{self.config.name}",
                address="localhost",
                port=self.config.port,
                tags=["mcp", self.config.name] + self.config.capabilities,
                metadata={
                    "mcp_service": True,
                    "wrapper_script": self.config.wrapper_script,
                    "capabilities": self.config.capabilities,
                    **self.config.metadata
                }
            )
            
            logger.info(f"MCP service {self.config.name} started and registered with mesh")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP service {self.config.name}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the MCP service and deregister from mesh"""
        try:
            # Deregister from mesh
            if self.service_instance:
                await self.mesh.discovery.deregister_service(self.service_instance.service_id)
            
            # Stop process
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            
            logger.info(f"MCP service {self.config.name} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop MCP service {self.config.name}: {e}")
            return False
    
    async def restart(self) -> bool:
        """Restart the MCP service"""
        await self.stop()
        await asyncio.sleep(2)  # Brief pause before restart
        return await self.start()
    
    async def health_check(self) -> bool:
        """Check health of the MCP service"""
        try:
            # Check process status
            if self.process and self.process.poll() is not None:
                logger.warning(f"MCP service {self.config.name} process died")
                if self.config.auto_restart and self.retry_count < self.config.max_retries:
                    self.retry_count += 1
                    return await self.restart()
                return False
            
            # Call health endpoint through mesh
            request = ServiceRequest(
                service_name=f"mcp-{self.config.name}",
                method="GET",
                path=self.config.health_endpoint,
                timeout=5.0,
                retry_count=1
            )
            
            result = await self.mesh.call_service(request)
            self.last_health_check = datetime.now()
            return result["status_code"] == 200
            
        except Exception as e:
            logger.error(f"Health check failed for MCP service {self.config.name}: {e}")
            return False
    
    async def _wait_for_ready(self, timeout: int = 30) -> bool:
        """Wait for service to be ready"""
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < timeout:
            try:
                # Simple TCP connection check
                reader, writer = await asyncio.open_connection('localhost', self.config.port)
                writer.close()
                await writer.wait_closed()
                return True
            except:
                await asyncio.sleep(1)
        
        raise TimeoutError(f"MCP service {self.config.name} failed to start within {timeout} seconds")

class MCPMeshBridge:
    """Bridge between MCP servers and the service mesh"""
    
    def __init__(self, mesh: ServiceMesh):
        self.mesh = mesh
        self.adapters: Dict[str, MCPServiceAdapter] = {}
        self.registry: Dict[str, Any] = self._load_mcp_registry()
        self._initialized = False
        
    def _load_mcp_registry(self) -> Dict[str, Any]:
        """Load MCP service registry from configuration"""
        return {
            "mcp_services": [
                {
                    "name": "postgres",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/postgres.sh",
                    "port": 11100,
                    "capabilities": ["sql", "database", "query"],
                    "tags": ["database", "sql"],
                    "metadata": {"database": "sutazai", "schema": "public"}
                },
                {
                    "name": "files",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/files.sh",
                    "port": 11101,
                    "capabilities": ["read", "write", "list", "delete"],
                    "tags": ["filesystem", "storage"],
                    "metadata": {"root": "/opt/sutazaiapp"}
                },
                {
                    "name": "http",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/http.sh",
                    "port": 11102,
                    "capabilities": ["fetch", "post", "put", "delete"],
                    "tags": ["network", "http"],
                    "metadata": {"timeout": 30}
                },
                {
                    "name": "ddg",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh",
                    "port": 11103,
                    "capabilities": ["search", "news", "images"],
                    "tags": ["search", "web"],
                    "metadata": {"engine": "duckduckgo"}
                },
                {
                    "name": "github",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/github.sh",
                    "port": 11104,
                    "capabilities": ["repos", "issues", "pulls", "actions"],
                    "tags": ["vcs", "github"],
                    "metadata": {"api_version": "v3"}
                },
                {
                    "name": "extended-memory",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh",
                    "port": 11105,
                    "capabilities": ["store", "retrieve", "search", "delete"],
                    "tags": ["memory", "persistence"],
                    "metadata": {"backend": "chromadb"}
                },
                {
                    "name": "puppeteer-mcp",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/puppeteer-mcp.sh",
                    "port": 11106,
                    "capabilities": ["screenshot", "pdf", "scrape", "navigate"],
                    "tags": ["browser", "automation"],
                    "metadata": {"headless": True}
                },
                {
                    "name": "playwright-mcp",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp.sh",
                    "port": 11107,
                    "capabilities": ["screenshot", "pdf", "scrape", "navigate", "multi-browser"],
                    "tags": ["browser", "automation"],
                    "metadata": {"browsers": ["chromium", "firefox", "webkit"]}
                }
            ]
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all MCP services and register with mesh"""
        if self._initialized:
            return {"status": "already_initialized", "services": list(self.adapters.keys())}
        
        started = []
        failed = []
        
        for service_config in self.registry.get("mcp_services", []):
            config = MCPServiceConfig(
                name=service_config["name"],
                wrapper_script=service_config["wrapper"],
                port=service_config["port"],
                capabilities=service_config.get("capabilities", []),
                metadata=service_config.get("metadata", {})
            )
            
            adapter = MCPServiceAdapter(config, self.mesh)
            
            if await adapter.start():
                self.adapters[service_config["name"]] = adapter
                started.append(service_config["name"])
            else:
                failed.append(service_config["name"])
        
        self._initialized = True
        
        return {
            "status": "initialized",
            "started": started,
            "failed": failed,
            "total": len(started) + len(failed)
        }
    
    async def shutdown(self) -> bool:
        """Shutdown all MCP services"""
        success = True
        for adapter in self.adapters.values():
            if not await adapter.stop():
                success = False
        
        self.adapters.clear()
        self._initialized = False
        return success
    
    async def call_mcp_service(
        self, 
        service_name: str, 
        method: str, 
        params: Dict[str, Any]
    ) -> Any:
        """Call an MCP service through the mesh"""
        
        # Ensure service is registered
        if service_name not in self.adapters:
            raise ValueError(f"MCP service {service_name} not found")
        
        # Create mesh request
        request = ServiceRequest(
            service_name=f"mcp-{service_name}",
            method="POST",
            path=f"/execute",
            body={
                "method": method,
                "params": params
            },
            timeout=30.0,
            retry_count=3
        )
        
        # Call through mesh with load balancing and circuit breaking
        result = await self.mesh.call_service(request)
        
        # Extract response body
        if result["status_code"] == 200:
            return result["body"]
        else:
            raise Exception(f"MCP call failed: {result}")
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status of an MCP service"""
        if service_name not in self.adapters:
            return {
                "service": service_name,
                "status": "not_found",
                "error": f"Service {service_name} not registered"
            }
        
        adapter = self.adapters[service_name]
        
        # Get mesh instances
        mesh_instances = await self.mesh.discovery.discover_services(f"mcp-{service_name}")
        
        # Check adapter instances
        adapter_healthy = await adapter.health_check()
        
        return {
            "service": service_name,
            "status": "healthy" if adapter_healthy else "unhealthy",
            "mesh_instances": len(mesh_instances),
            "adapter_instances": 1 if adapter.process and adapter.process.poll() is None else 0,
            "instances": [
                {
                    "id": inst.service_id,
                    "address": f"{inst.address}:{inst.port}",
                    "state": inst.state.value,
                    "metadata": inst.metadata
                }
                for inst in mesh_instances
            ],
            "mesh_registration": adapter.service_instance is not None
        }
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart an MCP service"""
        if service_name not in self.adapters:
            return False
        
        return await self.adapters[service_name].restart()
    
    async def stop_service(self, service_name: str) -> bool:
        """Stop an MCP service"""
        if service_name not in self.adapters:
            return False
        
        adapter = self.adapters[service_name]
        success = await adapter.stop()
        
        if success:
            del self.adapters[service_name]
        
        return success
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Health check all MCP services"""
        results = {}
        
        for name, adapter in self.adapters.items():
            healthy = await adapter.health_check()
            mesh_instances = await self.mesh.discovery.discover_services(f"mcp-{name}")
            
            results[name] = {
                "total_instances": len(mesh_instances),
                "healthy": len([i for i in mesh_instances if i.state == ServiceState.HEALTHY]),
                "unhealthy": len([i for i in mesh_instances if i.state == ServiceState.UNHEALTHY]),
                "unknown": len([i for i in mesh_instances if i.state == ServiceState.UNKNOWN]),
                "overall_health": "healthy" if healthy else "unhealthy"
            }
        
        return results

# Global bridge instance
_mcp_bridge: Optional[MCPMeshBridge] = None

async def get_mcp_bridge(mesh: ServiceMesh) -> MCPMeshBridge:
    """Get or create MCP bridge instance"""
    global _mcp_bridge
    
    if _mcp_bridge is None:
        _mcp_bridge = MCPMeshBridge(mesh)
    
    return _mcp_bridge

async def get_service_mesh() -> ServiceMesh:
    """Get service mesh instance"""
    from app.mesh.service_mesh import get_mesh
    return await get_mesh()

