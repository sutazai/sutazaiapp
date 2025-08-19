"""
MCP API Endpoints with Real Integration
Uses stdio-based MCP client to connect to running MCP servers
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Body, BackgroundTasks
from pydantic import BaseModel, Field
import asyncio
import logging
from datetime import datetime

# Import the real MCP client
from app.services.mcp_client import mcp_registry

logger = logging.getLogger(__name__)

router = APIRouter()

class MCPServerInfo(BaseModel):
    """Information about an MCP server"""
    name: str
    description: str
    connected: bool
    command: str
    tools_count: Optional[int] = None
    resources_count: Optional[int] = None

class MCPToolInfo(BaseModel):
    """Information about an MCP tool"""
    name: str
    description: Optional[str] = None
    inputSchema: Optional[Dict[str, Any]] = None

class MCPResourceInfo(BaseModel):
    """Information about an MCP resource"""
    uri: str
    name: Optional[str] = None
    description: Optional[str] = None
    mimeType: Optional[str] = None

class MCPToolCallRequest(BaseModel):
    """Request to call an MCP tool"""
    server: str = Field(..., description="MCP server name")
    tool: str = Field(..., description="Tool name")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

class MCPToolCallResponse(BaseModel):
    """Response from MCP tool call"""
    server: str
    tool: str
    success: bool
    result: Optional[Any] = None
    error: Optional[str] = None
    timestamp: str

class MCPResourceReadRequest(BaseModel):
    """Request to read an MCP resource"""
    server: str = Field(..., description="MCP server name")
    uri: str = Field(..., description="Resource URI")

class MCPResourceReadResponse(BaseModel):
    """Response from MCP resource read"""
    server: str
    uri: str
    success: bool
    contents: Optional[Any] = None
    mimeType: Optional[str] = None
    error: Optional[str] = None
    timestamp: str

@router.get("/servers", response_model=List[MCPServerInfo])
async def list_mcp_servers():
    """List all available MCP servers with their real connection status"""
    try:
        servers = await mcp_registry.list_servers()
        server_infos = []
        
        for server in servers:
            info = MCPServerInfo(
                name=server["name"],
                description=server["description"],
                connected=server["connected"],
                command=server["command"]
            )
            
            # If connected, get tool and resource counts
            if server["connected"]:
                try:
                    tools = await mcp_registry.get_server_tools(server["name"])
                    resources = await mcp_registry.get_server_resources(server["name"])
                    info.tools_count = len(tools)
                    info.resources_count = len(resources)
                except Exception as e:
                    logger.warning(f"Failed to get details for {server['name']}: {e}")
            
            server_infos.append(info)
        
        return server_infos
        
    except Exception as e:
        logger.error(f"Failed to list MCP servers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/servers/{server_name}", response_model=MCPServerInfo)
async def get_mcp_server_info(server_name: str):
    """Get detailed information about a specific MCP server"""
    try:
        servers = await mcp_registry.list_servers()
        server_info = None
        
        for server in servers:
            if server["name"] == server_name:
                server_info = server
                break
        
        if not server_info:
            raise HTTPException(
                status_code=404,
                detail=f"MCP server '{server_name}' not found"
            )
        
        info = MCPServerInfo(
            name=server_info["name"],
            description=server_info["description"],
            connected=server_info["connected"],
            command=server_info["command"]
        )
        
        # Try to connect and get details
        client = await mcp_registry.get_client(server_name)
        if client:
            try:
                tools = await client.list_tools()
                resources = await client.list_resources()
                info.tools_count = len(tools)
                info.resources_count = len(resources)
                info.connected = True
            except Exception as e:
                logger.warning(f"Failed to get details for {server_name}: {e}")
        
        return info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP server info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/servers/{server_name}/tools", response_model=List[MCPToolInfo])
async def list_server_tools(server_name: str):
    """List all tools available from an MCP server"""
    try:
        tools = await mcp_registry.get_server_tools(server_name)
        
        if tools is None:
            raise HTTPException(
                status_code=404,
                detail=f"MCP server '{server_name}' not found or not available"
            )
        
        return [
            MCPToolInfo(
                name=tool.get("name", ""),
                description=tool.get("description"),
                inputSchema=tool.get("inputSchema")
            )
            for tool in tools
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list tools for {server_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/servers/{server_name}/resources", response_model=List[MCPResourceInfo])
async def list_server_resources(server_name: str):
    """List all resources available from an MCP server"""
    try:
        resources = await mcp_registry.get_server_resources(server_name)
        
        if resources is None:
            raise HTTPException(
                status_code=404,
                detail=f"MCP server '{server_name}' not found or not available"
            )
        
        return [
            MCPResourceInfo(
                uri=resource.get("uri", ""),
                name=resource.get("name"),
                description=resource.get("description"),
                mimeType=resource.get("mimeType")
            )
            for resource in resources
        ]
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list resources for {server_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/tools/call", response_model=MCPToolCallResponse)
async def call_mcp_tool(request: MCPToolCallRequest):
    """Call a tool on an MCP server"""
    try:
        result = await mcp_registry.call_server_tool(
            request.server,
            request.tool,
            request.arguments
        )
        
        return MCPToolCallResponse(
            server=request.server,
            tool=request.tool,
            success=True,
            result=result,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to call tool {request.tool} on {request.server}: {e}")
        return MCPToolCallResponse(
            server=request.server,
            tool=request.tool,
            success=False,
            error=str(e),
            timestamp=datetime.utcnow().isoformat()
        )

@router.post("/resources/read", response_model=MCPResourceReadResponse)
async def read_mcp_resource(request: MCPResourceReadRequest):
    """Read a resource from an MCP server"""
    try:
        result = await mcp_registry.read_server_resource(
            request.server,
            request.uri
        )
        
        return MCPResourceReadResponse(
            server=request.server,
            uri=request.uri,
            success=True,
            contents=result.get("contents"),
            mimeType=result.get("mimeType"),
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Failed to read resource {request.uri} from {request.server}: {e}")
        return MCPResourceReadResponse(
            server=request.server,
            uri=request.uri,
            success=False,
            error=str(e),
            timestamp=datetime.utcnow().isoformat()
        )

@router.post("/connect/{server_name}")
async def connect_to_server(server_name: str):
    """Connect to a specific MCP server"""
    try:
        client = await mcp_registry.get_client(server_name)
        if client:
            return {
                "server": server_name,
                "connected": True,
                "message": f"Successfully connected to {server_name}",
                "timestamp": datetime.utcnow().isoformat()
            }
        else:
            raise HTTPException(
                status_code=503,
                detail=f"Failed to connect to MCP server '{server_name}'"
            )
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to connect to {server_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/disconnect/{server_name}")
async def disconnect_from_server(server_name: str):
    """Disconnect from a specific MCP server"""
    try:
        async with mcp_registry._lock:
            if server_name in mcp_registry.clients:
                await mcp_registry.clients[server_name].disconnect()
                del mcp_registry.clients[server_name]
                
                return {
                    "server": server_name,
                    "disconnected": True,
                    "message": f"Successfully disconnected from {server_name}",
                    "timestamp": datetime.utcnow().isoformat()
                }
            else:
                return {
                    "server": server_name,
                    "disconnected": False,
                    "message": f"Server {server_name} was not connected",
                    "timestamp": datetime.utcnow().isoformat()
                }
                
    except Exception as e:
        logger.error(f"Failed to disconnect from {server_name}: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def mcp_health_check():
    """Check overall MCP integration health"""
    try:
        servers = await mcp_registry.list_servers()
        total = len(servers)
        connected = sum(1 for s in servers if s["connected"])
        
        # Try to connect to a few servers to test
        test_servers = ["files", "github", "ddg"]
        test_results = {}
        
        for server_name in test_servers:
            if server_name in [s["name"] for s in servers]:
                try:
                    client = await mcp_registry.get_client(server_name)
                    if client:
                        tools = await client.list_tools()
                        test_results[server_name] = {
                            "connected": True,
                            "tools_count": len(tools)
                        }
                    else:
                        test_results[server_name] = {
                            "connected": False,
                            "error": "Failed to connect"
                        }
                except Exception as e:
                    test_results[server_name] = {
                        "connected": False,
                        "error": str(e)
                    }
        
        return {
            "total_servers": total,
            "connected_servers": connected,
            "health_percentage": (connected / total * 100) if total > 0 else 0,
            "status": "healthy" if connected > 0 else "critical",
            "test_results": test_results,
            "timestamp": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

@router.on_event("shutdown")
async def shutdown_event():
    """Cleanup MCP connections on shutdown"""
    await mcp_registry.disconnect_all()