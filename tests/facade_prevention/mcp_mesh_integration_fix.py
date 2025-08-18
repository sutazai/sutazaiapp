#!/usr/bin/env python3
"""
MCP-Mesh Integration Fix Implementation
Properly bridges MCP stdio servers with HTTP service mesh
"""
import asyncio
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MCPServiceDefinition:
    """Definition of an MCP service for mesh integration"""
    name: str
    wrapper_path: str
    port: int
    capabilities: List[str] = field(default_factory=list)
    health_endpoint: str = "/health"
    metadata: Dict[str, Any] = field(default_factory=dict)

class MCPMeshBridge:
    """
    Proper bridge implementation that connects stdio MCP servers to HTTP mesh
    This replaces the broken/disabled integration
    """
    
    # Port allocation for MCP HTTP adapters (11100-11199 range)
    MCP_PORT_BASE = 11100
    
    # MCP service definitions based on .mcp.json
    MCP_SERVICES = [
        MCPServiceDefinition("language-server", "/opt/sutazaiapp/scripts/mcp/wrappers/language-server.sh", 11100, 
                           ["code", "definitions"]),
        MCPServiceDefinition("github", "npx -y @modelcontextprotocol/server-github", 11101, 
                           ["repository", "issues", "pulls"]),
        MCPServiceDefinition("ultimatecoder", "/opt/sutazaiapp/scripts/mcp/wrappers/ultimatecoder.sh", 11102, 
                           ["code-generation", "analysis"]),
        MCPServiceDefinition("sequentialthinking", "/opt/sutazaiapp/scripts/mcp/wrappers/sequentialthinking.sh", 11103, 
                           ["reasoning", "problem-solving"]),
        MCPServiceDefinition("context7", "/opt/sutazaiapp/scripts/mcp/wrappers/context7.sh", 11104, 
                           ["documentation", "examples"]),
        MCPServiceDefinition("files", "/opt/sutazaiapp/scripts/mcp/wrappers/files.sh", 11105, 
                           ["file-operations"]),
        MCPServiceDefinition("http", "/opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh", 11106, 
                           ["web-requests"]),
        MCPServiceDefinition("ddg", "/opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh", 11107, 
                           ["search"]),
        MCPServiceDefinition("postgres", "/opt/sutazaiapp/scripts/mcp/wrappers/postgres.sh", 11108, 
                           ["database", "sql"]),
        MCPServiceDefinition("extended-memory", "/opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh", 11109, 
                           ["persistence", "memory"]),
        MCPServiceDefinition("mcp_ssh", "/opt/sutazaiapp/scripts/mcp/wrappers/mcp_ssh.sh", 11110, 
                           ["remote-execution"]),
        MCPServiceDefinition("nx-mcp", "/opt/sutazaiapp/scripts/mcp/wrappers/nx-mcp.sh", 11111, 
                           ["monorepo", "workspace"]),
        MCPServiceDefinition("puppeteer-mcp (no longer in use)", "/opt/sutazaiapp/scripts/mcp/wrappers/puppeteer-mcp (no longer in use).sh", 11112, 
                           ["browser-automation"]),
        MCPServiceDefinition("memory-bank-mcp", "/opt/sutazaiapp/scripts/mcp/wrappers/memory-bank-mcp.sh", 11113, 
                           ["memory-management"]),
        MCPServiceDefinition("playwright-mcp", "/opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp.sh", 11114, 
                           ["browser-testing"]),
        MCPServiceDefinition("knowledge-graph-mcp", "/opt/sutazaiapp/scripts/mcp/wrappers/knowledge-graph-mcp.sh", 11115, 
                           ["knowledge-management"]),
        MCPServiceDefinition("compass-mcp", "/opt/sutazaiapp/scripts/mcp/wrappers/compass-mcp.sh", 11116, 
                           ["mcp-discovery"])
    ]
    
    def __init__(self, mesh_client=None):
        self.mesh_client = mesh_client
        self.adapters: Dict[str, 'MCPHTTPAdapter'] = {}
        self.registered_services: Dict[str, Dict] = {}
        
    async def initialize_all(self) -> Dict[str, Any]:
        """Initialize all MCP services with mesh integration"""
        results = {
            "started": [],
            "failed": [],
            "registered": [],
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        logger.info("Starting MCP-Mesh integration...")
        
        for mcp_def in self.MCP_SERVICES:
            try:
                # Create HTTP adapter for MCP
                adapter = MCPHTTPAdapter(mcp_def)
                
                # Start the adapter (launches MCP process and HTTP server)
                if await adapter.start():
                    self.adapters[mcp_def.name] = adapter
                    results["started"].append(mcp_def.name)
                    
                    # Register with mesh if available
                    if self.mesh_client:
                        registration = await self.register_with_mesh(mcp_def, adapter)
                        if registration:
                            results["registered"].append(mcp_def.name)
                            self.registered_services[mcp_def.name] = registration
                    
                    logger.info(f"✅ MCP {mcp_def.name} started on port {mcp_def.port}")
                else:
                    results["failed"].append(mcp_def.name)
                    logger.error(f"❌ Failed to start MCP {mcp_def.name}")
                    
            except Exception as e:
                results["failed"].append(mcp_def.name)
                logger.error(f"❌ Error initializing MCP {mcp_def.name}: {e}")
        
        # Summary
        logger.info(f"MCP-Mesh Integration Results:")
        logger.info(f"  Started: {len(results['started'])} services")
        logger.info(f"  Registered: {len(results['registered'])} with mesh")
        logger.info(f"  Failed: {len(results['failed'])} services")
        
        return results
    
    async def register_with_mesh(self, mcp_def: MCPServiceDefinition, adapter: 'MCPHTTPAdapter') -> Optional[Dict]:
        """Register MCP service with the service mesh"""
        try:
            # Create service instance for mesh
            service_data = {
                "service_id": f"mcp-{mcp_def.name}-{mcp_def.port}",
                "service_name": f"mcp-{mcp_def.name}",
                "address": "localhost",
                "port": mcp_def.port,
                "tags": ["mcp", "bridge"] + mcp_def.capabilities,
                "metadata": {
                    "wrapper": mcp_def.wrapper_path,
                    "protocol": "http-stdio-bridge",
                    "capabilities": mcp_def.capabilities,
                    **mcp_def.metadata
                },
                "health_check": {
                    "http": f"http://localhost:{mcp_def.port}{mcp_def.health_endpoint}",
                    "interval": "10s",
                    "timeout": "5s"
                }
            }
            
            # Register with mesh (would call actual mesh API)
            if self.mesh_client:
                await self.mesh_client.register_service(service_data)
            
            logger.info(f"Registered MCP {mcp_def.name} with service mesh")
            return service_data
            
        except Exception as e:
            logger.error(f"Failed to register MCP {mcp_def.name} with mesh: {e}")
            return None
    
    async def shutdown_all(self):
        """Shutdown all MCP adapters"""
        logger.info("Shutting down MCP-Mesh bridge...")
        
        for name, adapter in self.adapters.items():
            try:
                await adapter.stop()
                logger.info(f"Stopped MCP adapter: {name}")
            except Exception as e:
                logger.error(f"Error stopping MCP {name}: {e}")
        
        self.adapters.clear()
        self.registered_services.clear()


class MCPHTTPAdapter:
    """
    HTTP adapter that wraps stdio MCP servers
    Provides HTTP endpoints that translate to stdio communication
    """
    
    def __init__(self, definition: MCPServiceDefinition):
        self.definition = definition
        self.process: Optional[subprocess.Popen] = None
        self.server_task: Optional[asyncio.Task] = None
        self.running = False
        
    async def start(self) -> bool:
        """Start MCP process and HTTP server"""
        try:
            # Start MCP process
            self.process = subprocess.Popen(
                [self.definition.wrapper_path] if self.definition.wrapper_path.endswith('.sh') 
                else ['sh', '-c', self.definition.wrapper_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            
            # Start HTTP server
            self.server_task = asyncio.create_task(self._run_http_server())
            self.running = True
            
            # Wait briefly to ensure startup
            await asyncio.sleep(1)
            
            # Check if process is still running
            if self.process.poll() is not None:
                logger.error(f"MCP process {self.definition.name} died immediately")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP adapter for {self.definition.name}: {e}")
            return False
    
    async def stop(self):
        """Stop the adapter"""
        self.running = False
        
        if self.server_task:
            self.server_task.cancel()
            try:
                await self.server_task
            except asyncio.CancelledError:
                pass
        
        if self.process:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
    
    async def _run_http_server(self):
        """Run HTTP server that bridges to stdio MCP"""
        from aiohttp import web
        
        async def handle_health(request):
            """Health check endpoint"""
            healthy = self.process and self.process.poll() is None
            return web.json_response({
                "status": "healthy" if healthy else "unhealthy",
                "service": self.definition.name,
                "port": self.definition.port,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        
        async def handle_call(request):
            """Handle MCP call via HTTP"""
            try:
                data = await request.json()
                
                # Send to MCP process via stdio
                request_json = json.dumps({
                    "jsonrpc": "2.0",
                    "method": data.get("method", ""),
                    "params": data.get("params", {}),
                    "id": data.get("id", 1)
                })
                
                if self.process and self.process.stdin:
                    self.process.stdin.write(request_json + "\n")
                    self.process.stdin.flush()
                    
                    # Read response (simplified - real implementation needs better parsing)
                    if self.process.stdout:
                        response_line = self.process.stdout.readline()
                        if response_line:
                            response_data = json.loads(response_line)
                            return web.json_response(response_data)
                
                return web.json_response({"error": "MCP process not available"}, status=503)
                
            except Exception as e:
                logger.error(f"Error handling MCP call for {self.definition.name}: {e}")
                return web.json_response({"error": str(e)}, status=500)
        
        async def handle_info(request):
            """Service information endpoint"""
            return web.json_response({
                "name": self.definition.name,
                "port": self.definition.port,
                "capabilities": self.definition.capabilities,
                "metadata": self.definition.metadata,
                "status": "running" if self.running else "stopped"
            })
        
        # Create web application
        app = web.Application()
        app.router.add_get('/health', handle_health)
        app.router.add_get('/info', handle_info)
        app.router.add_post('/call', handle_call)
        
        # Run server
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', self.definition.port)
        await site.start()
        
        logger.info(f"HTTP adapter for MCP {self.definition.name} listening on port {self.definition.port}")
        
        # Keep running until stopped
        while self.running:
            await asyncio.sleep(1)
        
        await runner.cleanup()


async def test_integration():
    """Test the MCP-Mesh integration"""
    logger.info("Testing MCP-Mesh Integration...")
    
    # Create bridge
    bridge = MCPMeshBridge()
    
    # Initialize all MCPs
    results = await bridge.initialize_all()
    
    # Test health checks
    import httpx
    async with httpx.AsyncClient() as client:
        for service in results["started"]:
            port = 11100 + list(name for name, _, _ in [(s.name, s.wrapper_path, s.port) for s in bridge.MCP_SERVICES]).index(service)
            try:
                response = await client.get(f"http://localhost:{port}/health")
                if response.status_code == 200:
                    logger.info(f"✅ Health check passed for {service}")
                else:
                    logger.error(f"❌ Health check failed for {service}")
            except Exception as e:
                logger.error(f"❌ Could not reach {service}: {e}")
    
    # Keep running for testing
    await asyncio.sleep(60)
    
    # Shutdown
    await bridge.shutdown_all()


if __name__ == "__main__":
    asyncio.run(test_integration())