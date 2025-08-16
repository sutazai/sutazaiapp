"""
MCP Service Adapter - Bridge between STDIO-based MCP servers and HTTP service mesh
Implements real, working adapter for MCP-mesh integration
"""
from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from enum import Enum

import httpx
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import JSONResponse
from prometheus_client import Counter, Histogram, Gauge
import uvicorn

logger = logging.getLogger(__name__)

# Metrics for monitoring
mcp_requests_total = Counter('mcp_adapter_requests_total', 'Total MCP requests', ['server', 'method'])
mcp_request_duration = Histogram('mcp_adapter_request_duration_seconds', 'MCP request duration', ['server'])
mcp_errors_total = Counter('mcp_adapter_errors_total', 'Total MCP errors', ['server', 'error_type'])
mcp_active_processes = Gauge('mcp_adapter_active_processes', 'Active MCP processes', ['server'])

class MCPServerType(Enum):
    """Types of MCP servers based on their wrapper implementation"""
    DOCKER = "docker"  # Runs in Docker container (postgres, etc.)
    NPX = "npx"        # Runs via npx (language-server, etc.)
    NODE = "node"      # Direct node execution
    PYTHON = "python"  # Python-based MCP servers
    BINARY = "binary"  # Standalone binaries

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server"""
    name: str
    server_type: MCPServerType
    wrapper_path: str
    instances: int = 1
    port_range: List[int] = field(default_factory=lambda: [11100, 11199])
    health_check_path: str = "/health"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    environment: Dict[str, str] = field(default_factory=dict)
    timeout: float = 30.0
    retry_count: int = 3

class MCPProcess:
    """Manages a single MCP server process"""
    
    def __init__(self, config: MCPServerConfig, instance_id: int):
        self.config = config
        self.instance_id = instance_id
        self.process: Optional[subprocess.Popen] = None
        self.port: Optional[int] = None
        self.service_id = f"{config.name}-{instance_id}-{uuid.uuid4().hex[:8]}"
        self.start_time: Optional[float] = None
        self.request_count = 0
        self.error_count = 0
        self.last_health_check: Optional[float] = None
        self.health_status = "unknown"
        
    async def start(self, port: int) -> bool:
        """Start the MCP server process"""
        self.port = port
        wrapper_path = Path(self.config.wrapper_path)
        
        if not wrapper_path.exists():
            logger.error(f"MCP wrapper not found: {wrapper_path}")
            return False
            
        try:
            # Prepare environment with port information
            env = {
                **self.config.environment,
                "MCP_ADAPTER_PORT": str(port),
                "MCP_INSTANCE_ID": str(self.instance_id),
                "MCP_SERVICE_ID": self.service_id
            }
            
            # Start the MCP process
            logger.info(f"Starting MCP server {self.config.name} instance {self.instance_id} on port {port}")
            
            # For Docker-based MCP servers, we'll need special handling
            if self.config.server_type == MCPServerType.DOCKER:
                # Docker containers handle their own networking
                # We'll create a proxy instead of modifying the wrapper
                self.process = subprocess.Popen(
                    [str(wrapper_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=False  # Use binary mode for better STDIO handling
                )
            else:
                # For NPX/Node/Python servers, direct STDIO communication
                self.process = subprocess.Popen(
                    [str(wrapper_path)],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    text=False
                )
            
            self.start_time = time.time()
            mcp_active_processes.labels(server=self.config.name).inc()
            
            # Give process time to start
            await asyncio.sleep(2)
            
            # Check if process is still running
            if self.process.poll() is not None:
                logger.error(f"MCP process {self.config.name} exited immediately")
                return False
                
            logger.info(f"MCP server {self.config.name} instance {self.instance_id} started successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCP server {self.config.name}: {e}")
            return False
    
    async def send_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Send request to MCP server via STDIO"""
        if not self.process or self.process.poll() is not None:
            raise RuntimeError(f"MCP process {self.config.name} not running")
        
        try:
            # MCP protocol uses JSON-RPC over STDIO
            json_rpc_request = {
                "jsonrpc": "2.0",
                "id": request.get("id", str(uuid.uuid4())),
                "method": request.get("method", "execute"),
                "params": request.get("params", {})
            }
            
            # Send request
            request_bytes = (json.dumps(json_rpc_request) + "\n").encode()
            self.process.stdin.write(request_bytes)
            self.process.stdin.flush()
            
            # Read response (with timeout)
            response_line = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    None, self.process.stdout.readline
                ),
                timeout=self.config.timeout
            )
            
            if not response_line:
                raise RuntimeError("Empty response from MCP server")
            
            # Parse response
            response = json.loads(response_line.decode())
            self.request_count += 1
            
            return response
            
        except asyncio.TimeoutError:
            self.error_count += 1
            mcp_errors_total.labels(server=self.config.name, error_type="timeout").inc()
            raise RuntimeError(f"MCP request timeout after {self.config.timeout}s")
        except Exception as e:
            self.error_count += 1
            mcp_errors_total.labels(server=self.config.name, error_type="request_error").inc()
            raise RuntimeError(f"MCP request failed: {e}")
    
    async def health_check(self) -> bool:
        """Check if MCP process is healthy"""
        if not self.process:
            self.health_status = "not_started"
            return False
            
        if self.process.poll() is not None:
            self.health_status = "exited"
            return False
        
        try:
            # Send health check request
            health_request = {
                "method": "health",
                "params": {}
            }
            response = await self.send_request(health_request)
            
            self.last_health_check = time.time()
            self.health_status = "healthy"
            return True
            
        except Exception as e:
            logger.warning(f"Health check failed for {self.config.name}: {e}")
            self.health_status = "unhealthy"
            # Process might still be running but not responding properly
            # For now, we'll consider it unhealthy but keep it running
            return False
    
    async def stop(self):
        """Stop the MCP process"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.sleep(2)
                if self.process.poll() is None:
                    self.process.kill()
                mcp_active_processes.labels(server=self.config.name).dec()
            except Exception as e:
                logger.error(f"Error stopping MCP process: {e}")
            finally:
                self.process = None

class MCPServiceAdapter:
    """
    Adapter to expose MCP servers as HTTP services for mesh integration
    Each adapter manages multiple instances of an MCP server
    """
    
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.instances: Dict[int, MCPProcess] = {}
        self.app = FastAPI(title=f"MCP-{config.name}")
        self.next_port = config.port_range[0]
        self.running = False
        self._setup_routes()
    
    def _setup_routes(self):
        """Setup HTTP routes for MCP access"""
        
        @self.app.get("/health")
        async def health_check():
            """Health check endpoint for service mesh"""
            healthy_instances = sum(
                1 for inst in self.instances.values() 
                if inst.health_status == "healthy"
            )
            total_instances = len(self.instances)
            
            status = "healthy" if healthy_instances > 0 else "unhealthy"
            
            return {
                "service": self.config.name,
                "status": status,
                "instances": {
                    "healthy": healthy_instances,
                    "total": total_instances
                },
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        @self.app.post("/execute")
        async def execute_command(request: Request):
            """Execute MCP command"""
            try:
                body = await request.json()
                
                # Select a healthy instance (simple round-robin for now)
                healthy_instances = [
                    inst for inst in self.instances.values()
                    if inst.health_status == "healthy"
                ]
                
                if not healthy_instances:
                    raise HTTPException(status_code=503, detail="No healthy MCP instances available")
                
                # Round-robin selection
                instance = healthy_instances[0]  # TODO: Implement proper load balancing
                
                # Track metrics
                start_time = time.time()
                mcp_requests_total.labels(server=self.config.name, method=body.get("method", "unknown")).inc()
                
                # Execute request
                result = await instance.send_request(body)
                
                # Track duration
                duration = time.time() - start_time
                mcp_request_duration.labels(server=self.config.name).observe(duration)
                
                return result
                
            except Exception as e:
                mcp_errors_total.labels(server=self.config.name, error_type="execution_error").inc()
                logger.error(f"MCP execution error: {e}")
                raise HTTPException(status_code=500, detail=str(e))
        
        @self.app.get("/instances")
        async def list_instances():
            """List all MCP instances"""
            return {
                "service": self.config.name,
                "instances": [
                    {
                        "id": inst.service_id,
                        "instance_id": inst.instance_id,
                        "port": inst.port,
                        "status": inst.health_status,
                        "uptime": time.time() - inst.start_time if inst.start_time else 0,
                        "request_count": inst.request_count,
                        "error_count": inst.error_count
                    }
                    for inst in self.instances.values()
                ]
            }
    
    async def start(self, instances: int = None) -> List[int]:
        """Start MCP service instances"""
        if instances is None:
            instances = self.config.instances
        
        ports = []
        for i in range(instances):
            port = self.next_port + i
            if port > self.config.port_range[1]:
                logger.warning(f"Port range exhausted for {self.config.name}")
                break
            
            process = MCPProcess(self.config, i)
            if await process.start(port):
                self.instances[i] = process
                ports.append(port)
                logger.info(f"Started MCP instance {self.config.name}:{i} on port {port}")
            else:
                logger.error(f"Failed to start MCP instance {self.config.name}:{i}")
        
        self.running = True
        
        # Start health check loop
        asyncio.create_task(self._health_check_loop())
        
        return ports
    
    async def _health_check_loop(self):
        """Periodic health check for all instances"""
        while self.running:
            for instance in self.instances.values():
                try:
                    await instance.health_check()
                except Exception as e:
                    logger.error(f"Health check error for {instance.service_id}: {e}")
            
            await asyncio.sleep(10)  # Check every 10 seconds
    
    async def stop(self):
        """Stop all MCP instances"""
        self.running = False
        for instance in self.instances.values():
            await instance.stop()
        self.instances.clear()
    
    def get_app(self) -> FastAPI:
        """Get FastAPI app for this adapter"""
        return self.app

# Factory function to create adapters for known MCP servers
def create_mcp_adapter(server_name: str) -> Optional[MCPServiceAdapter]:
    """Create an MCP adapter for a known server"""
    
    # MCP server configurations based on existing wrappers
    configs = {
        "postgres": MCPServerConfig(
            name="postgres",
            server_type=MCPServerType.DOCKER,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/postgres.sh",
            instances=3,
            tags=["mcp", "database", "postgres"],
            metadata={"version": "1.0.0", "capabilities": ["sql", "transactions"]}
        ),
        "files": MCPServerConfig(
            name="files",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/files.sh",
            instances=2,
            tags=["mcp", "filesystem", "io"],
            metadata={"version": "1.0.0", "capabilities": ["read", "write", "watch"]}
        ),
        "language-server": MCPServerConfig(
            name="language-server",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/language-server.sh",
            instances=3,
            tags=["mcp", "language", "code-analysis"],
            metadata={"version": "1.0.0", "capabilities": ["completion", "diagnostics"]}
        ),
        "http": MCPServerConfig(
            name="http",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh",
            instances=2,
            tags=["mcp", "http", "web"],
            metadata={"version": "1.0.0", "capabilities": ["fetch", "scrape"]}
        ),
        "ddg": MCPServerConfig(
            name="ddg",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh",
            instances=2,
            tags=["mcp", "search", "web"],
            metadata={"version": "1.0.0", "capabilities": ["search", "suggestions"]}
        ),
        "extended-memory": MCPServerConfig(
            name="extended-memory",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh",
            instances=1,
            tags=["mcp", "memory", "persistence"],
            metadata={"version": "1.0.0", "capabilities": ["store", "retrieve", "search"]}
        ),
        "mcp_ssh": MCPServerConfig(
            name="mcp_ssh",
            server_type=MCPServerType.NODE,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/mcp_ssh.sh",
            instances=2,
            tags=["mcp", "ssh", "remote"],
            metadata={"version": "1.0.0", "capabilities": ["exec", "transfer"]}
        ),
        "ultimatecoder": MCPServerConfig(
            name="ultimatecoder",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/ultimatecoder.sh",
            instances=2,
            tags=["mcp", "codegen", "ai"],
            metadata={"version": "1.0.0", "capabilities": ["generate", "refactor"]}
        ),
        "sequentialthinking": MCPServerConfig(
            name="sequentialthinking",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/sequentialthinking.sh",
            instances=2,
            tags=["mcp", "reasoning", "ai"],
            metadata={"version": "1.0.0", "capabilities": ["analyze", "plan"]}
        ),
        "context7": MCPServerConfig(
            name="context7",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/context7.sh",
            instances=2,
            tags=["mcp", "context", "library"],
            metadata={"version": "1.0.0", "capabilities": ["docs", "examples"]}
        ),
        "nx-mcp": MCPServerConfig(
            name="nx-mcp",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/nx-mcp.sh",
            instances=1,
            tags=["mcp", "nx", "monorepo"],
            metadata={"version": "1.0.0", "capabilities": ["workspace", "generators"]}
        ),
        "puppeteer-mcp": MCPServerConfig(
            name="puppeteer-mcp",
            server_type=MCPServerType.NODE,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/puppeteer-mcp.sh",
            instances=2,
            tags=["mcp", "browser", "automation"],
            metadata={"version": "1.0.0", "capabilities": ["scrape", "interact"]}
        ),
        "memory-bank-mcp": MCPServerConfig(
            name="memory-bank-mcp",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/memory-bank-mcp.sh",
            instances=1,
            tags=["mcp", "memory", "bank"],
            metadata={"version": "1.0.0", "capabilities": ["store", "query"]}
        ),
        "playwright-mcp": MCPServerConfig(
            name="playwright-mcp",
            server_type=MCPServerType.NODE,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp.sh",
            instances=2,
            tags=["mcp", "browser", "testing"],
            metadata={"version": "1.0.0", "capabilities": ["test", "automate"]}
        ),
        "knowledge-graph-mcp": MCPServerConfig(
            name="knowledge-graph-mcp",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/knowledge-graph-mcp.sh",
            instances=1,
            tags=["mcp", "graph", "knowledge"],
            metadata={"version": "1.0.0", "capabilities": ["graph", "query", "reason"]}
        ),
        "compass-mcp": MCPServerConfig(
            name="compass-mcp",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/compass-mcp.sh",
            instances=1,
            tags=["mcp", "discovery", "recommendation"],
            metadata={"version": "1.0.0", "capabilities": ["discover", "recommend"]}
        )
    }
    
    if server_name not in configs:
        logger.warning(f"Unknown MCP server: {server_name}")
        return None
    
    return MCPServiceAdapter(configs[server_name])