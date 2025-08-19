"""
Consolidated MCP (Model Context Protocol) API Endpoints
Single, authoritative implementation following Rule 1: Real Implementation Only
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Body, BackgroundTasks
from pydantic import BaseModel, Field
import subprocess
import json
import asyncio
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

router = APIRouter()

# MCP server configuration (only working servers based on testing)
WORKING_MCP_SERVERS = {
    "files": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/files.sh",
        "type": "npx",
        "description": "File system operations"
    },
    "github": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/github.sh",
        "type": "npx",
        "description": "GitHub repository operations"
    },
    "http": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh",
        "type": "npx",
        "description": "HTTP/HTTPS fetch operations"
    },
    "ddg": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh",
        "type": "npx",
        "description": "DuckDuckGo search"
    },
    "language-server": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/language-server.sh",
        "type": "npx",
        "description": "Language server protocol"
    },
    "mcp_ssh": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/mcp_ssh.sh",
        "type": "npx",
        "description": "SSH operations"
    },
    "ultimatecoder": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/ultimatecoder.sh",
        "type": "python",
        "description": "Advanced coding assistance"
    },
    "context7": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/context7.sh",
        "type": "npx",
        "description": "Context management"
    },
    "compass-mcp": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/compass-mcp.sh",
        "type": "npx",
        "description": "Navigation and guidance"
    },
    "knowledge-graph-mcp": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/knowledge-graph-mcp.sh",
        "type": "npx",
        "description": "Knowledge graph operations"
    },
    "memory-bank-mcp": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/memory-bank-mcp.sh",
        "type": "npx",
        "description": "Memory storage and retrieval"
    },
    "nx-mcp": {
        "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/nx-mcp.sh",
        "type": "npx",
        "description": "NX monorepo operations"
    }
}

class MCPServerInfo(BaseModel):
    """Information about an MCP server"""
    name: str
    type: str
    description: str
    status: str
    wrapper: str
    last_tested: Optional[str] = None

class MCPHealthCheck(BaseModel):
    """Health check result for an MCP server"""
    server: str
    healthy: bool
    message: str
    timestamp: str

class MCPExecuteRequest(BaseModel):
    """Request to execute MCP server command"""
    server: str = Field(..., description="MCP server name")
    method: str = Field(..., description="Method to execute")
    params: Dict[str, Any] = Field(default_factory=dict, description="Method parameters")
    timeout: float = Field(30.0, description="Execution timeout in seconds")

class MCPExecuteResponse(BaseModel):
    """Response from MCP server execution"""
    server: str
    method: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str

async def check_mcp_server_health(server_name: str) -> bool:
    """Check if an MCP server is healthy"""
    if server_name not in WORKING_MCP_SERVERS:
        return False
    
    wrapper = WORKING_MCP_SERVERS[server_name]["wrapper"]
    try:
        # Run selfcheck
        result = await asyncio.create_subprocess_exec(
            "bash", wrapper, "--selfcheck",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        stdout, stderr = await result.communicate()
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Health check failed for {server_name}: {e}")
        return False

@router.get("/servers", response_model=List[MCPServerInfo])
async def list_mcp_servers():
    """List all available MCP servers with their status"""
    servers = []
    for name, config in WORKING_MCP_SERVERS.items():
        health = await check_mcp_server_health(name)
        servers.append(MCPServerInfo(
            name=name,
            type=config["type"],
            description=config["description"],
            status="healthy" if health else "unhealthy",
            wrapper=config["wrapper"],
            last_tested=datetime.utcnow().isoformat()
        ))
    return servers

@router.get("/servers/{server_name}", response_model=MCPServerInfo)
async def get_mcp_server_info(server_name: str):
    """Get detailed information about a specific MCP server"""
    if server_name not in WORKING_MCP_SERVERS:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server '{server_name}' not found"
        )
    
    config = WORKING_MCP_SERVERS[server_name]
    health = await check_mcp_server_health(server_name)
    
    return MCPServerInfo(
        name=server_name,
        type=config["type"],
        description=config["description"],
        status="healthy" if health else "unhealthy",
        wrapper=config["wrapper"],
        last_tested=datetime.utcnow().isoformat()
    )

@router.post("/servers/{server_name}/health", response_model=MCPHealthCheck)
async def check_mcp_server(server_name: str):
    """Perform health check on an MCP server"""
    if server_name not in WORKING_MCP_SERVERS:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server '{server_name}' not found"
        )
    
    healthy = await check_mcp_server_health(server_name)
    
    return MCPHealthCheck(
        server=server_name,
        healthy=healthy,
        message="Server is operational" if healthy else "Server health check failed",
        timestamp=datetime.utcnow().isoformat()
    )

@router.post("/execute", response_model=MCPExecuteResponse)
async def execute_mcp_command(request: MCPExecuteRequest):
    """Execute a command on an MCP server"""
    if request.server not in WORKING_MCP_SERVERS:
        raise HTTPException(
            status_code=404,
            detail=f"MCP server '{request.server}' not found"
        )
    
    # Check server health first
    if not await check_mcp_server_health(request.server):
        return MCPExecuteResponse(
            server=request.server,
            method=request.method,
            success=False,
            error="Server is not healthy",
            timestamp=datetime.utcnow().isoformat()
        )
    
    try:
        # Build command based on server type
        wrapper = WORKING_MCP_SERVERS[request.server]["wrapper"]
        
        # Create JSON-RPC request
        json_rpc = {
            "jsonrpc": "2.0",
            "method": request.method,
            "params": request.params,
            "id": 1
        }
        
        # Execute command with timeout
        process = await asyncio.create_subprocess_exec(
            "bash", wrapper,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Send JSON-RPC request
        stdout, stderr = await asyncio.wait_for(
            process.communicate(input=json.dumps(json_rpc).encode()),
            timeout=request.timeout
        )
        
        if process.returncode != 0:
            error_msg = stderr.decode() if stderr else "Unknown error"
            return MCPExecuteResponse(
                server=request.server,
                method=request.method,
                success=False,
                error=error_msg,
                timestamp=datetime.utcnow().isoformat()
            )
        
        # Parse response
        try:
            response = json.loads(stdout.decode())
            return MCPExecuteResponse(
                server=request.server,
                method=request.method,
                success=True,
                result=response.get("result"),
                error=response.get("error"),
                timestamp=datetime.utcnow().isoformat()
            )
        except json.JSONDecodeError:
            # Return raw output if not JSON
            return MCPExecuteResponse(
                server=request.server,
                method=request.method,
                success=True,
                result=stdout.decode(),
                timestamp=datetime.utcnow().isoformat()
            )
            
    except asyncio.TimeoutError:
        return MCPExecuteResponse(
            server=request.server,
            method=request.method,
            success=False,
            error=f"Command timed out after {request.timeout} seconds",
            timestamp=datetime.utcnow().isoformat()
        )
    except Exception as e:
        logger.error(f"Failed to execute MCP command: {e}")
        return MCPExecuteResponse(
            server=request.server,
            method=request.method,
            success=False,
            error=str(e),
            timestamp=datetime.utcnow().isoformat()
        )

@router.post("/init")
async def initialize_mcp_servers(background_tasks: BackgroundTasks):
    """Initialize all MCP servers"""
    
    async def run_init():
        """Run initialization in background"""
        try:
            process = await asyncio.create_subprocess_exec(
                "/opt/sutazaiapp/scripts/mcp/init_mcp_servers.sh",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
        except Exception as e:
            logger.error(f"MCP initialization failed: {e}")
    
    background_tasks.add_task(run_init)
    
    return {
        "status": "initializing",
        "message": "MCP server initialization started in background",
        "timestamp": datetime.utcnow().isoformat()
    }

@router.get("/health")
async def mcp_health_summary():
    """Get overall MCP system health"""
    total = len(WORKING_MCP_SERVERS)
    healthy = 0
    unhealthy = []
    
    for server_name in WORKING_MCP_SERVERS:
        if await check_mcp_server_health(server_name):
            healthy += 1
        else:
            unhealthy.append(server_name)
    
    return {
        "total_servers": total,
        "healthy": healthy,
        "unhealthy": len(unhealthy),
        "unhealthy_servers": unhealthy,
        "health_percentage": (healthy / total * 100) if total > 0 else 0,
        "status": "healthy" if healthy == total else "degraded" if healthy > 0 else "critical",
        "timestamp": datetime.utcnow().isoformat()
    }