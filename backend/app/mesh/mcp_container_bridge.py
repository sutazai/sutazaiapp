"""
MCP Container Bridge - Integrates containerized MCP servers with the service mesh
"""

import asyncio
import aiohttp
import json
import logging
from typing import Dict, List, Optional, Any
from pathlib import Path
import docker
from datetime import datetime

logger = logging.getLogger(__name__)


class MCPContainerBridge:
    """Bridge between containerized MCP servers and the service mesh"""
    
    def __init__(self, mesh_client=None):
        self.mesh_client = mesh_client
        self.docker_client = docker.from_env()
        self.port_registry_path = Path('/opt/sutazaiapp/config/ports/mcp_ports.json')
        self.mcp_services: Dict[str, Dict] = {}
        self.session = None
        self._running = False
        self._health_check_task = None
        
    async def initialize(self):
        """Initialize the MCP container bridge"""
        logger.info("Initializing MCP Container Bridge...")
        
        # Create aiohttp session
        self.session = aiohttp.ClientSession()
        
        # Discover MCP containers
        await self.discover_mcp_containers()
        
        # Register services with mesh
        await self.register_with_mesh()
        
        # Start health monitoring
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"MCP Container Bridge initialized with {len(self.mcp_services)} services")
    
    async def discover_mcp_containers(self):
        """Discover running MCP containers"""
        try:
            # Load port registry
            port_registry = {}
            if self.port_registry_path.exists():
                with open(self.port_registry_path, 'r') as f:
                    port_registry = json.load(f)
            
            # Find MCP containers
            containers = self.docker_client.containers.list(
                filters={'label': 'managed_by=mcp-orchestrator'}
            )
            
            for container in containers:
                labels = container.labels
                name = labels.get('mcp_name')
                port = labels.get('mcp_port')
                
                if name and port:
                    self.mcp_services[name] = {
                        'container_id': container.id,
                        'container_name': container.name,
                        'port': int(port),
                        'url': f'http://localhost:{port}',
                        'status': 'healthy' if container.status == 'running' else 'unhealthy',
                        'last_check': datetime.utcnow().isoformat()
                    }
                    
            logger.info(f"Discovered {len(self.mcp_services)} MCP containers")
            
        except Exception as e:
            logger.error(f"Failed to discover MCP containers: {e}")
    
    async def register_with_mesh(self):
        """Register MCP services with the service mesh"""
        if not self.mesh_client:
            logger.warning("No mesh client available, skipping registration")
            return
        
        try:
            for name, service in self.mcp_services.items():
                # Register as HTTP service in mesh
                await self.mesh_client.register_service(
                    name=f"mcp-{name}",
                    service_type='http',
                    config={
                        'url': service['url'],
                        'port': service['port'],
                        'protocol': 'http',
                        'health_check': f"{service['url']}/health"
                    }
                )
                logger.info(f"Registered MCP service '{name}' with mesh")
                
        except Exception as e:
            logger.error(f"Failed to register with mesh: {e}")
    
    async def call_mcp_service(self, service_name: str, method: str, params: Dict = None) -> Optional[Dict]:
        """Call an MCP service through its container"""
        if service_name not in self.mcp_services:
            logger.error(f"MCP service '{service_name}' not found")
            return None
        
        service = self.mcp_services[service_name]
        url = f"{service['url']}/api/mcp"
        
        try:
            payload = {
                'jsonrpc': '2.0',
                'method': method,
                'params': params or {},
                'id': 1
            }
            
            async with self.session.post(
                url,
                json=payload,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                if response.status == 200:
                    result = await response.json()
                    return result.get('result')
                else:
                    logger.error(f"MCP call failed with status {response.status}")
                    return None
                    
        except asyncio.TimeoutError:
            logger.error(f"Timeout calling MCP service '{service_name}'")
            return None
        except Exception as e:
            logger.error(f"Error calling MCP service '{service_name}': {e}")
            return None
    
    async def broadcast_to_mcps(self, method: str, params: Dict = None) -> Dict[str, Any]:
        """Broadcast a call to all MCP services"""
        results = {}
        
        tasks = []
        for name in self.mcp_services.keys():
            task = asyncio.create_task(self.call_mcp_service(name, method, params))
            tasks.append((name, task))
        
        for name, task in tasks:
            try:
                result = await task
                results[name] = result
            except Exception as e:
                logger.error(f"Failed to call {name}: {e}")
                results[name] = None
        
        return results
    
    async def _health_check_loop(self):
        """Continuously monitor MCP container health"""
        while self._running:
            try:
                await self._check_all_health()
                await asyncio.sleep(30)  # Check every 30 seconds
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def _check_all_health(self):
        """Check health of all MCP services"""
        for name, service in self.mcp_services.items():
            try:
                # Check container status
                container = self.docker_client.containers.get(service['container_id'])
                container_healthy = container.status == 'running'
                
                # Check HTTP health endpoint
                health_url = f"{service['url']}/health"
                async with self.session.get(
                    health_url,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    http_healthy = response.status == 200
                
                # Update status
                service['status'] = 'healthy' if (container_healthy and http_healthy) else 'unhealthy'
                service['last_check'] = datetime.utcnow().isoformat()
                
                if service['status'] == 'unhealthy':
                    logger.warning(f"MCP service '{name}' is unhealthy")
                    await self._attempt_recovery(name, service)
                    
            except Exception as e:
                logger.error(f"Health check failed for '{name}': {e}")
                service['status'] = 'unknown'
    
    async def _attempt_recovery(self, name: str, service: Dict):
        """Attempt to recover an unhealthy MCP service"""
        logger.info(f"Attempting recovery for MCP service '{name}'")
        
        try:
            container = self.docker_client.containers.get(service['container_id'])
            
            # Restart container
            container.restart(timeout=10)
            await asyncio.sleep(5)  # Wait for startup
            
            # Re-check health
            await self._check_all_health()
            
            if self.mcp_services[name]['status'] == 'healthy':
                logger.info(f"Successfully recovered MCP service '{name}'")
            else:
                logger.error(f"Failed to recover MCP service '{name}'")
                
        except Exception as e:
            logger.error(f"Recovery failed for '{name}': {e}")
    
    def get_service_status(self) -> Dict[str, Dict]:
        """Get status of all MCP services"""
        return {
            'services': self.mcp_services,
            'total': len(self.mcp_services),
            'healthy': sum(1 for s in self.mcp_services.values() if s['status'] == 'healthy'),
            'unhealthy': sum(1 for s in self.mcp_services.values() if s['status'] == 'unhealthy')
        }
    
    async def shutdown(self):
        """Shutdown the bridge"""
        logger.info("Shutting down MCP Container Bridge...")
        
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        if self.session:
            await self.session.close()
        
        logger.info("MCP Container Bridge shutdown complete")


# Integration with existing MCP bridge
class EnhancedMCPBridge:
    """Enhanced MCP Bridge with container support"""
    
    def __init__(self, mesh_client=None):
        self.mesh_client = mesh_client
        self.container_bridge = MCPContainerBridge(mesh_client)
        self.stdio_servers = {}  # Existing STDIO servers
        
    async def initialize(self):
        """Initialize the enhanced bridge"""
        # Initialize container bridge
        await self.container_bridge.initialize()
        
        # Load STDIO server configurations
        await self._load_stdio_servers()
        
        logger.info("Enhanced MCP Bridge initialized")
    
    async def _load_stdio_servers(self):
        """Load STDIO MCP server configurations"""
        # This would load from .mcp.json for non-containerized servers
        pass
    
    async def call_server(self, server_name: str, method: str, params: Dict = None) -> Optional[Dict]:
        """Call an MCP server (containerized or STDIO)"""
        
        # Check if it's a containerized service
        if server_name in self.container_bridge.mcp_services:
            return await self.container_bridge.call_mcp_service(server_name, method, params)
        
        # Check if it's a STDIO service
        if server_name in self.stdio_servers:
            # Handle STDIO protocol
            return await self._call_stdio_server(server_name, method, params)
        
        logger.error(f"MCP server '{server_name}' not found")
        return None
    
    async def _call_stdio_server(self, server_name: str, method: str, params: Dict = None) -> Optional[Dict]:
        """Call a STDIO MCP server"""
        # Implementation for STDIO protocol
        # This would use subprocess to communicate with STDIO servers
        pass
    
    def get_all_servers(self) -> Dict[str, str]:
        """Get all available MCP servers and their types"""
        servers = {}
        
        # Add containerized servers
        for name in self.container_bridge.mcp_services.keys():
            servers[name] = 'container'
        
        # Add STDIO servers
        for name in self.stdio_servers.keys():
            servers[name] = 'stdio'
        
        return servers
    
    async def shutdown(self):
        """Shutdown the bridge"""
        await self.container_bridge.shutdown()