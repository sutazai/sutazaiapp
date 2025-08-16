"""
MCP-Mesh Bridge Service - Orchestrates MCP server lifecycle and mesh integration
Production-ready implementation for MCP-mesh coordination
"""
from __future__ import annotations

import asyncio
import logging
import yaml
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from datetime import datetime, timezone

from prometheus_client import Counter, Gauge, Histogram
import uvicorn

from .service_mesh import ServiceMesh, ServiceInstance, ServiceState, LoadBalancerStrategy
from .mcp_adapter import MCPServiceAdapter, create_mcp_adapter

logger = logging.getLogger(__name__)

# Metrics
mcp_bridge_operations = Counter('mcp_bridge_operations_total', 'MCP bridge operations', ['operation', 'status'])
mcp_registered_services = Gauge('mcp_bridge_registered_services', 'Number of registered MCP services')
mcp_active_instances = Gauge('mcp_bridge_active_instances', 'Number of active MCP instances', ['service'])
mcp_registration_duration = Histogram('mcp_bridge_registration_duration_seconds', 'Time to register MCP service')

class MCPMeshBridge:
    """
    Bridge between MCP servers and the service mesh
    Manages lifecycle of MCP adapters and their mesh registration
    """
    
    def __init__(self, 
                 mesh: ServiceMesh,
                 registry_path: str = "/opt/sutazaiapp/backend/config/mcp_mesh_registry.yaml"):
        self.mesh = mesh
        self.registry_path = Path(registry_path)
        self.registry: Dict[str, Any] = {}
        self.adapters: Dict[str, MCPServiceAdapter] = {}
        self.adapter_servers: Dict[str, Any] = {}  # Running uvicorn servers
        self.registered_services: Set[str] = set()
        self.running = False
        self._load_registry()
    
    def _load_registry(self):
        """Load MCP registry configuration"""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    self.registry = yaml.safe_load(f)
                logger.info(f"Loaded MCP registry with {len(self.registry.get('mcp_services', []))} services")
            else:
                logger.warning(f"Registry file not found: {self.registry_path}")
                self.registry = {"mcp_services": [], "global_config": {}}
        except Exception as e:
            logger.error(f"Failed to load MCP registry: {e}")
            self.registry = {"mcp_services": [], "global_config": {}}
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all MCP services in the mesh"""
        logger.info("Initializing MCP-Mesh Bridge")
        results = {
            "started": [],
            "failed": [],
            "registered": [],
            "errors": []
        }
        
        self.running = True
        mcp_bridge_operations.labels(operation="initialize", status="started").inc()
        
        # Process each MCP service configuration
        for service_config in self.registry.get('mcp_services', []):
            service_name = service_config.get('name')
            if not service_name:
                results["errors"].append("Service config missing name")
                continue
            
            try:
                # Deploy the MCP service
                success = await self._deploy_mcp_service(service_config)
                if success:
                    results["started"].append(service_name)
                    results["registered"].append(f"mcp-{service_name}")
                else:
                    results["failed"].append(service_name)
                    
            except Exception as e:
                logger.error(f"Failed to deploy MCP service {service_name}: {e}")
                results["failed"].append(service_name)
                results["errors"].append(f"{service_name}: {str(e)}")
        
        # Update metrics
        mcp_registered_services.set(len(results["registered"]))
        mcp_bridge_operations.labels(
            operation="initialize", 
            status="completed" if not results["failed"] else "partial"
        ).inc()
        
        logger.info(f"MCP-Mesh Bridge initialization complete: {len(results['started'])} started, {len(results['failed'])} failed")
        return results
    
    async def _deploy_mcp_service(self, config: Dict[str, Any]) -> bool:
        """Deploy a single MCP service with its instances"""
        service_name = config['name']
        mesh_service_name = config.get('service_name', f"mcp-{service_name}")
        instances = config.get('instances', 1)
        port_range = config.get('port_range', [11100, 11199])
        
        logger.info(f"Deploying MCP service {service_name} with {instances} instances")
        
        try:
            # Create MCP adapter
            adapter = create_mcp_adapter(service_name)
            if not adapter:
                logger.error(f"Failed to create adapter for {service_name}")
                return False
            
            self.adapters[service_name] = adapter
            
            # Start adapter instances (MCP processes)
            ports = await adapter.start(instances)
            if not ports:
                logger.error(f"No instances started for {service_name}")
                return False
            
            logger.info(f"Started {len(ports)} instances of {service_name} on ports {ports}")
            
            # Start HTTP server for the adapter
            # Each adapter gets one HTTP server that proxies to multiple MCP instances
            http_port = port_range[0] - 1000  # HTTP port offset
            
            server_config = uvicorn.Config(
                app=adapter.get_app(),
                host="0.0.0.0",
                port=http_port,
                log_level="warning",
                access_log=False
            )
            server = uvicorn.Server(server_config)
            
            # Run server in background
            asyncio.create_task(server.serve())
            self.adapter_servers[service_name] = server
            
            # Register with service mesh
            for i, port in enumerate(ports):
                instance_id = f"{mesh_service_name}-{i}"
                
                # Create service instance for mesh registration
                instance = ServiceInstance(
                    service_id=instance_id,
                    service_name=mesh_service_name,
                    address="localhost",
                    port=http_port,  # All instances use same HTTP adapter port
                    tags=config.get('tags', []),
                    metadata={
                        **config.get('metadata', {}),
                        'mcp_server': service_name,
                        'mcp_instance': i,
                        'mcp_process_port': port,  # Actual MCP process port
                        'adapter_port': http_port
                    },
                    state=ServiceState.HEALTHY,
                    weight=100
                )
                
                # Register with mesh
                registered = await self.mesh.register_service(
                    service_name=mesh_service_name,
                    address=instance.address,
                    port=instance.port,
                    tags=instance.tags,
                    metadata=instance.metadata
                )
                
                if registered:
                    self.registered_services.add(instance_id)
                    logger.info(f"Registered {instance_id} with service mesh")
                    
                    # Configure load balancer strategy
                    lb_strategy = config.get('load_balancer', 'ROUND_ROBIN')
                    if hasattr(LoadBalancerStrategy, lb_strategy):
                        self.mesh.load_balancer.set_strategy(
                            mesh_service_name,
                            LoadBalancerStrategy[lb_strategy]
                        )
            
            # Update metrics
            mcp_active_instances.labels(service=service_name).set(len(ports))
            
            return True
            
        except Exception as e:
            logger.error(f"Error deploying MCP service {service_name}: {e}")
            mcp_bridge_operations.labels(operation="deploy", status="failed").inc()
            return False
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status of an MCP service"""
        if service_name not in self.adapters:
            return {
                "service": service_name,
                "status": "not_found",
                "error": f"Service {service_name} not registered"
            }
        
        adapter = self.adapters[service_name]
        mesh_service_name = f"mcp-{service_name}"
        
        # Get instances from mesh
        mesh_instances = await self.mesh.discover_services(mesh_service_name)
        
        # Get adapter status
        adapter_instances = []
        for instance in adapter.instances.values():
            adapter_instances.append({
                "id": instance.service_id,
                "port": instance.port,
                "status": instance.health_status,
                "requests": instance.request_count,
                "errors": instance.error_count
            })
        
        return {
            "service": service_name,
            "status": "active" if adapter.running else "stopped",
            "mesh_instances": len(mesh_instances),
            "adapter_instances": len(adapter_instances),
            "instances": adapter_instances,
            "mesh_registration": mesh_service_name in self.registered_services
        }
    
    async def call_mcp_service(self, 
                               service_name: str,
                               method: str,
                               params: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP service through the mesh"""
        mesh_service_name = f"mcp-{service_name}"
        
        # Use mesh to discover and call service
        from .service_mesh import ServiceRequest
        
        request = ServiceRequest(
            service_name=mesh_service_name,
            method="POST",
            path="/execute",
            body={
                "method": method,
                "params": params
            }
        )
        
        try:
            result = await self.mesh.call_service(request)
            return result
        except Exception as e:
            logger.error(f"Failed to call MCP service {service_name}: {e}")
            raise
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Health check all MCP services"""
        results = {}
        
        for service_name, adapter in self.adapters.items():
            health_status = {
                "healthy": 0,
                "unhealthy": 0,
                "unknown": 0
            }
            
            for instance in adapter.instances.values():
                status = instance.health_status
                if status == "healthy":
                    health_status["healthy"] += 1
                elif status in ["unhealthy", "exited"]:
                    health_status["unhealthy"] += 1
                else:
                    health_status["unknown"] += 1
            
            results[service_name] = {
                "total_instances": len(adapter.instances),
                **health_status,
                "overall_health": "healthy" if health_status["healthy"] > 0 else "unhealthy"
            }
        
        return results
    
    async def stop_service(self, service_name: str) -> bool:
        """Stop a specific MCP service"""
        if service_name not in self.adapters:
            logger.warning(f"Service {service_name} not found")
            return False
        
        try:
            # Stop adapter instances
            adapter = self.adapters[service_name]
            await adapter.stop()
            
            # Stop HTTP server
            if service_name in self.adapter_servers:
                server = self.adapter_servers[service_name]
                server.should_exit = True
                del self.adapter_servers[service_name]
            
            # Deregister from mesh
            mesh_service_name = f"mcp-{service_name}"
            instances = await self.mesh.discover_services(mesh_service_name)
            for instance in instances:
                await self.mesh.deregister_service(
                    instance['service_id'],
                    mesh_service_name
                )
            
            # Clean up
            del self.adapters[service_name]
            mcp_active_instances.labels(service=service_name).set(0)
            
            logger.info(f"Stopped MCP service {service_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping MCP service {service_name}: {e}")
            return False
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart a specific MCP service"""
        # Find config
        service_config = None
        for config in self.registry.get('mcp_services', []):
            if config.get('name') == service_name:
                service_config = config
                break
        
        if not service_config:
            logger.error(f"No configuration found for service {service_name}")
            return False
        
        # Stop if running
        if service_name in self.adapters:
            await self.stop_service(service_name)
        
        # Start again
        return await self._deploy_mcp_service(service_config)
    
    async def shutdown(self):
        """Shutdown all MCP services and clean up"""
        logger.info("Shutting down MCP-Mesh Bridge")
        self.running = False
        
        # Stop all services
        for service_name in list(self.adapters.keys()):
            await self.stop_service(service_name)
        
        # Clear registrations
        self.registered_services.clear()
        mcp_registered_services.set(0)
        
        logger.info("MCP-Mesh Bridge shutdown complete")

# Singleton instance management
_bridge_instance: Optional[MCPMeshBridge] = None

async def get_mcp_bridge(mesh: ServiceMesh) -> MCPMeshBridge:
    """Get or create the MCP bridge singleton"""
    global _bridge_instance
    if _bridge_instance is None:
        _bridge_instance = MCPMeshBridge(mesh)
    return _bridge_instance

async def initialize_mcp_mesh_integration(mesh: ServiceMesh) -> Dict[str, Any]:
    """Initialize MCP-mesh integration (convenience function)"""
    bridge = await get_mcp_bridge(mesh)
    return await bridge.initialize()