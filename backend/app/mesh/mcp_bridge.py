"""
MCP-Mesh Integration Bridge (Fixed)
Provides seamless integration between Model Context Protocol servers and the service mesh
Handles missing services gracefully and supports dynamic registration
"""
import asyncio
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib

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
    required: bool = False  # Whether this service is required for system operation

class MCPServiceAdapter:
    """Adapter for MCP services to work with the service mesh"""
    
    def __init__(self, config: MCPServiceConfig, mesh=None):
        """Initialize adapter with optional mesh - can work without mesh"""
        self.config = config
        self.mesh = mesh
        self.process: Optional[subprocess.Popen] = None
        self.service_instance: Optional[Any] = None
        self.retry_count = 0
        self.last_health_check = datetime.now()
        self.available = False
        
    async def start(self) -> bool:
        """Start the MCP service and optionally register with mesh"""
        try:
            # Check if wrapper script exists
            if not os.path.exists(self.config.wrapper_script):
                if self.config.required:
                    logger.error(f"Required MCP service {self.config.name} wrapper not found: {self.config.wrapper_script}")
                    return False
                else:
                    logger.warning(f"Optional MCP service {self.config.name} wrapper not found, skipping")
                    return False
            
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
            
            # Wait for service to be ready (with shorter timeout)
            ready = await self._wait_for_ready(timeout=10)
            if not ready:
                logger.warning(f"MCP service {self.config.name} did not become ready in time")
                if self.process:
                    self.process.terminate()
                return False
            
            # Register with service mesh if available
            if self.mesh:
                try:
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
                    logger.info(f"✅ MCP service {self.config.name} started and registered with mesh")
                except Exception as mesh_error:
                    logger.warning(f"⚠️ MCP service {self.config.name} started but mesh registration failed: {mesh_error}")
                    # Service is still running, just not in mesh
            else:
                logger.info(f"✅ MCP service {self.config.name} started (standalone mode)")
            
            self.available = True
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP service {self.config.name}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the MCP service and deregister from mesh"""
        try:
            # Deregister from mesh if registered
            if self.mesh and self.service_instance:
                try:
                    await self.mesh.discovery.deregister_service(self.service_instance.service_id)
                except Exception as mesh_error:
                    logger.debug(f"Could not deregister from mesh: {mesh_error}")
            
            # Stop process
            if self.process:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            
            self.available = False
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
                self.available = False
                if self.config.auto_restart and self.retry_count < self.config.max_retries:
                    self.retry_count += 1
                    return await self.restart()
                return False
            
            # If we have mesh, try health check through it
            if self.mesh and self.service_instance:
                try:
                    from app.mesh.service_mesh import ServiceRequest
                    request = ServiceRequest(
                        service_name=f"mcp-{self.config.name}",
                        method="GET",
                        path=self.config.health_endpoint,
                        timeout=5.0,
                        retry_count=1
                    )
                    
                    result = await self.mesh.call_service(request)
                    self.last_health_check = datetime.now()
                    self.available = result.get("status_code") == 200
                    return self.available
                except Exception as mesh_error:
                    logger.debug(f"Mesh health check failed: {mesh_error}")
                    # Fall back to process check
            
            # Simple process check if no mesh or mesh check failed
            if self.process and self.process.poll() is None:
                self.last_health_check = datetime.now()
                self.available = True
                return True
            
            self.available = False
            return False
            
        except Exception as e:
            logger.error(f"Health check failed for MCP service {self.config.name}: {e}")
            self.available = False
            return False
    
    async def _wait_for_ready(self, timeout: int = 30) -> bool:
        """Wait for service to be ready"""
        start_time = datetime.now()
        while (datetime.now() - start_time).seconds < timeout:
            try:
                # Check if process is still running
                if self.process and self.process.poll() is not None:
                    return False
                
                # Simple TCP connection check
                reader, writer = await asyncio.open_connection('localhost', self.config.port)
                writer.close()
                await writer.wait_closed()
                return True
            except:
                await asyncio.sleep(1)
        
        return False  # Timeout without error

class MCPMeshBridge:
    """Bridge between MCP servers and the service mesh - works with or without mesh"""
    
    def __init__(self, mesh=None):
        """Initialize bridge with optional mesh"""
        self.mesh = mesh
        self.adapters: Dict[str, MCPServiceAdapter] = {}
        self.registry: Dict[str, Any] = self._load_mcp_registry()
        self._initialized = False
        
    def _load_mcp_registry(self) -> Dict[str, Any]:
        """Load MCP service registry from configuration"""
        # Check which wrappers actually exist
        wrapper_base = "/opt/sutazaiapp/scripts/mcp/wrappers"
        available_wrappers = []
        
        if os.path.exists(wrapper_base):
            for file in os.listdir(wrapper_base):
                if file.endswith('.sh'):
                    available_wrappers.append(file)
        
        logger.info(f"Found {len(available_wrappers)} available MCP wrappers")
        
        # Define all known services but mark availability
        all_services = [
            {
                "name": "postgres",
                "wrapper": f"{wrapper_base}/postgres.sh",
                "port": 11100,
                "capabilities": ["sql", "database", "query"],
                "tags": ["database", "sql"],
                "metadata": {"database": "sutazai", "schema": "public"},
                "required": False
            },
            {
                "name": "files",
                "wrapper": f"{wrapper_base}/files.sh",
                "port": 11101,
                "capabilities": ["read", "write", "list", "delete"],
                "tags": ["filesystem", "storage"],
                "metadata": {"root": "/opt/sutazaiapp"},
                "required": False
            },
            {
                "name": "http",
                "wrapper": f"{wrapper_base}/http_fetch.sh",
                "port": 11102,
                "capabilities": ["fetch", "post", "put", "delete"],
                "tags": ["network", "http"],
                "metadata": {"timeout": 30},
                "required": False
            },
            {
                "name": "ddg",
                "wrapper": f"{wrapper_base}/ddg.sh",
                "port": 11103,
                "capabilities": ["search", "news", "images"],
                "tags": ["search", "web"],
                "metadata": {"engine": "duckduckgo"},
                "required": False
            },
            {
                "name": "github",
                "wrapper": f"{wrapper_base}/github.sh",
                "port": 11104,
                "capabilities": ["repos", "issues", "pulls", "actions"],
                "tags": ["vcs", "github"],
                "metadata": {"api_version": "v3"},
                "required": False
            },
            {
                "name": "extended-memory",
                "wrapper": f"{wrapper_base}/extended-memory.sh",
                "port": 11105,
                "capabilities": ["store", "retrieve", "search", "delete"],
                "tags": ["memory", "persistence"],
                "metadata": {"backend": "chromadb"},
                "required": False
            },
            {
                "name": "language-server",
                "wrapper": f"{wrapper_base}/language-server.sh",
                "port": 11106,
                "capabilities": ["completion", "diagnostics", "formatting"],
                "tags": ["language", "lsp"],
                "metadata": {"version": "1.0.0"},
                "required": False
            }
        ]
        
        # Filter to only available services
        available_services = []
        for service in all_services:
            wrapper_name = os.path.basename(service["wrapper"])
            if wrapper_name in available_wrappers or os.path.exists(service["wrapper"]):
                available_services.append(service)
                logger.info(f"✓ MCP service {service['name']} is available")
            else:
                logger.warning(f"✗ MCP service {service['name']} wrapper not found, skipping")
        
        return {"mcp_services": available_services}
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all available MCP services and optionally register with mesh"""
        if self._initialized:
            return {"status": "already_initialized", "services": list(self.adapters.keys())}
        
        started = []
        failed = []
        skipped = []
        
        for service_config in self.registry.get("mcp_services", []):
            config = MCPServiceConfig(
                name=service_config["name"],
                wrapper_script=service_config["wrapper"],
                port=service_config["port"],
                capabilities=service_config.get("capabilities", []),
                metadata=service_config.get("metadata", {}),
                required=service_config.get("required", False)
            )
            
            # Check if wrapper exists before trying to start
            if not os.path.exists(config.wrapper_script):
                skipped.append(service_config["name"])
                logger.warning(f"Skipping MCP {service_config['name']}: wrapper not found")
                continue
            
            adapter = MCPServiceAdapter(config, self.mesh)
            
            if await adapter.start():
                self.adapters[service_config["name"]] = adapter
                started.append(service_config["name"])
            else:
                if config.required:
                    failed.append(service_config["name"])
                else:
                    skipped.append(service_config["name"])
        
        self._initialized = True
        
        # Log summary
        logger.info(f"MCP Bridge Initialization Complete:")
        logger.info(f"  Started: {len(started)} services")
        logger.info(f"  Failed: {len(failed)} services")
        logger.info(f"  Skipped: {len(skipped)} services")
        
        return {
            "status": "initialized",
            "started": started,
            "failed": failed,
            "skipped": skipped,
            "total": len(started) + len(failed) + len(skipped)
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
    
    async def initialize_services(self) -> bool:
        """Initialize MCP services from configuration if not already initialized"""
        if self._initialized and self.adapters:
            logger.info("MCP services already initialized")
            return True
        
        try:
            # Load MCP configuration
            config_file = "/opt/sutazaiapp/.mcp.json"
            if not os.path.exists(config_file):
                logger.warning(f"MCP configuration file not found: {config_file}")
                return False
            
            with open(config_file, 'r') as f:
                mcp_config = json.load(f)
            
            mcp_servers = mcp_config.get("mcpServers", {})
            if not mcp_servers:
                logger.warning("No MCP servers configured in .mcp.json")
                return False
            
            logger.info(f"Loading {len(mcp_servers)} MCP servers from configuration...")
            
            # Create service configs from .mcp.json
            service_configs = []
            port_counter = 11100  # Start port range for MCP services
            
            for name, server_config in mcp_servers.items():
                # Map .mcp.json format to our internal format
                wrapper_script = None
                
                # Check if it's using a wrapper script or direct command
                if server_config.get("command", "").startswith("/opt/sutazaiapp/scripts/mcp/wrappers/"):
                    wrapper_script = server_config["command"]
                else:
                    # For npm-based services, check if wrapper exists
                    wrapper_path = f"/opt/sutazaiapp/scripts/mcp/wrappers/{name}.sh"
                    if os.path.exists(wrapper_path):
                        wrapper_script = wrapper_path
                    else:
                        logger.debug(f"No wrapper found for {name}, skipping")
                        continue
                
                service_configs.append({
                    "name": name,
                    "wrapper": wrapper_script,
                    "port": port_counter,
                    "capabilities": [],  # Could be enhanced from metadata
                    "metadata": {"source": "mcp.json"},
                    "required": False  # None are required for basic operation
                })
                port_counter += 1
            
            # Update the registry with our service configs
            if service_configs:
                self.registry["mcp_services"] = service_configs
                # Now call the original initialize method
                result = await self.initialize()
                return len(result.get("started", [])) > 0 or len(result.get("services", [])) > 0
            else:
                logger.warning("No valid MCP service configurations found")
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize MCP services: {e}")
            return False
    
    async def call_mcp_service(
        self, 
        service_name: str, 
        method: str, 
        params: Dict[str, Any]
    ) -> Any:
        """Call an MCP service through the mesh or directly"""
        
        # Check if service is available
        if service_name not in self.adapters:
            available = list(self.adapters.keys())
            raise ValueError(f"MCP service {service_name} not available. Available services: {available}")
        
        adapter = self.adapters[service_name]
        if not adapter.available:
            raise RuntimeError(f"MCP service {service_name} is not healthy")
        
        # If we have mesh, use it
        if self.mesh:
            try:
                from app.mesh.service_mesh import ServiceRequest
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
                if result.get("status_code") == 200:
                    return result.get("body")
                else:
                    raise Exception(f"MCP call failed: {result}")
            except Exception as mesh_error:
                logger.warning(f"Mesh call failed, falling back to direct: {mesh_error}")
                # Fall back to direct call
        
        # Direct call without mesh (not implemented in this version)
        raise NotImplementedError("Direct MCP calls without mesh not yet implemented")
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status of an MCP service"""
        if service_name not in self.adapters:
            return {
                "service": service_name,
                "status": "not_found",
                "error": f"Service {service_name} not registered"
            }
        
        adapter = self.adapters[service_name]
        
        # Check adapter health
        adapter_healthy = await adapter.health_check()
        
        status = {
            "service": service_name,
            "status": "healthy" if adapter_healthy else "unhealthy",
            "available": adapter.available,
            "process_running": adapter.process is not None and adapter.process.poll() is None,
            "last_health_check": adapter.last_health_check.isoformat() if adapter.last_health_check else None,
            "retry_count": adapter.retry_count
        }
        
        # Add mesh info if available
        if self.mesh and adapter.service_instance:
            try:
                mesh_instances = await self.mesh.discovery.discover_services(f"mcp-{service_name}")
                status["mesh_instances"] = len(mesh_instances)
                status["mesh_registration"] = True
            except:
                status["mesh_instances"] = 0
                status["mesh_registration"] = False
        else:
            status["mesh_registration"] = False
        
        return status
    
    async def list_services(self) -> List[str]:
        """List all registered MCP services"""
        return list(self.adapters.keys())
    
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
            
            results[name] = {
                "healthy": healthy,
                "available": adapter.available,
                "process_running": adapter.process is not None and adapter.process.poll() is None,
                "retry_count": adapter.retry_count,
                "last_check": adapter.last_health_check.isoformat() if adapter.last_health_check else None
            }
        
        # Summary
        total = len(results)
        healthy_count = sum(1 for r in results.values() if r["healthy"])
        
        return {
            "services": results,
            "summary": {
                "total": total,
                "healthy": healthy_count,
                "unhealthy": total - healthy_count,
                "percentage_healthy": (healthy_count / total * 100) if total > 0 else 0
            }
        }
    
    async def register_dynamic_service(self, service_config: Dict[str, Any]) -> bool:
        """Dynamically register a new MCP service"""
        try:
            config = MCPServiceConfig(
                name=service_config["name"],
                wrapper_script=service_config["wrapper"],
                port=service_config.get("port", 11200),  # Default port range for dynamic
                capabilities=service_config.get("capabilities", []),
                metadata=service_config.get("metadata", {}),
                required=service_config.get("required", False)
            )
            
            # Check if already registered
            if config.name in self.adapters:
                logger.warning(f"Service {config.name} already registered")
                return False
            
            # Create and start adapter
            adapter = MCPServiceAdapter(config, self.mesh)
            if await adapter.start():
                self.adapters[config.name] = adapter
                logger.info(f"✅ Dynamically registered MCP service {config.name}")
                return True
            else:
                logger.error(f"Failed to start dynamically registered service {config.name}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering dynamic service: {e}")
            return False

# Global bridge instance
_mcp_bridge: Optional[MCPMeshBridge] = None

async def get_mcp_bridge(mesh=None) -> MCPMeshBridge:
    """Get or create MCP bridge instance - works with or without mesh"""
    global _mcp_bridge
    
    if _mcp_bridge is None:
        _mcp_bridge = MCPMeshBridge(mesh)
    
    return _mcp_bridge

async def get_service_mesh():
    """Get service mesh instance if available"""
    try:
        from app.mesh.service_mesh import get_mesh
        return await get_mesh()
    except Exception as e:
        logger.warning(f"Service mesh not available: {e}")
        return None