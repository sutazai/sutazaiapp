"""
MCP-Service Mesh Integration Layer
Bridges STDIO MCP servers with HTTP service mesh
"""
import asyncio
import json
import logging
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
import httpx

from .service_mesh import ServiceMesh, ServiceInstance, ServiceState

logger = logging.getLogger(__name__)

class MCPMeshIntegration:
    """Integrates MCP servers into the service mesh"""
    
    def __init__(self, mesh: ServiceMesh):
        self.mesh = mesh
        self.mcp_processes: Dict[str, subprocess.Popen] = {}
        self.mcp_adapters: Dict[str, "MCPAdapter"] = {}
        
    async def register_mcp_server(
        self, 
        name: str, 
        wrapper_path: str,
        port: int
    ) -> ServiceInstance:
        """Register an MCP server with the mesh"""
        
        # Create HTTP adapter for STDIO MCP server
        adapter = MCPAdapter(name, wrapper_path, port)
        await adapter.start()
        
        # Register with service mesh
        instance = ServiceInstance(
            service_id=f"mcp-{name}-{port}",
            service_name=f"mcp-{name}",
            address="localhost",
            port=port,
            tags=["mcp", "adapter"],
            metadata={
                "wrapper": wrapper_path,
                "protocol": "http-stdio-bridge"
            },
            state=ServiceState.HEALTHY
        )
        
        # Register with mesh
        await self.mesh.register_service(instance)
        
        # Store adapter
        self.mcp_adapters[name] = adapter
        
        logger.info(f"Registered MCP server {name} on port {port}")
        return instance
    
    async def call_mcp_service(
        self,
        service_name: str,
        method: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call an MCP service through the mesh"""
        
        # Route through service mesh for load balancing
        instance = await self.mesh.discover_service(f"mcp-{service_name}")
        
        if not instance:
            raise ValueError(f"MCP service {service_name} not found")
        
        # Call through adapter
        adapter = self.mcp_adapters.get(service_name)
        if adapter:
            return await adapter.call(method, params)
        
        # Fallback to HTTP call
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"http://{instance.address}:{instance.port}/call",
                json={"method": method, "params": params},
                timeout=30.0
            )
            response.raise_for_status()
            return response.json()


class MCPAdapter:
    """HTTP-to-STDIO adapter for MCP servers"""
    
    def __init__(self, name: str, wrapper_path: str, port: int):
        self.name = name
        self.wrapper_path = wrapper_path
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.server: Optional[asyncio.AbstractServer] = None
        
    async def start(self):
        """Start the MCP process and HTTP adapter"""
        
        # Start MCP process
        self.process = subprocess.Popen(
            [self.wrapper_path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Start HTTP server
        from aiohttp import web
        
        app = web.Application()
        app.router.add_post('/call', self.handle_call)
        app.router.add_get('/health', self.handle_health)
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.port)
        await site.start()
        
        logger.info(f"MCP adapter for {self.name} started on port {self.port}")
    
    async def handle_call(self, request):
        """Handle HTTP call and forward to STDIO MCP"""
        from aiohttp import web
        
        try:
            data = await request.json()
            method = data.get("method")
            params = data.get("params", {})
            
            # Send to MCP process
            request_json = json.dumps({
                "jsonrpc": "2.0",
                "method": method,
                "params": params,
                "id": 1
            })
            
            self.process.stdin.write(request_json + "\n")
            self.process.stdin.flush()
            
            # Read response
            response_line = self.process.stdout.readline()
            response_data = json.loads(response_line)
            
            return web.json_response(response_data.get("result", {}))
            
        except Exception as e:
            logger.error(f"MCP adapter error: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
    
    async def handle_health(self, request):
        """Health check endpoint"""
        from aiohttp import web
        
        healthy = self.process and self.process.poll() is None
        return web.json_response({
            "status": "healthy" if healthy else "unhealthy",
            "service": self.name,
            "port": self.port
        })
    
    async def stop(self):
        """Stop the adapter and MCP process"""
        if self.process:
            self.process.terminate()
            self.process.wait(timeout=5)
        if self.server:
            self.server.close()
            await self.server.wait_closed()
