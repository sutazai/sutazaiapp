"""
MCP STDIO API Endpoints - Real MCP server communication
Uses actual STDIO protocol to communicate with MCP servers
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Body
from pydantic import BaseModel, Field
import logging

from ....mesh.mcp_stdio_bridge import get_mcp_stdio_bridge, MCPSTDIOBridge

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp-stdio", tags=["mcp-stdio"])

# Request/Response Models
class MCPToolCallRequest(BaseModel):
    """Request to call an MCP tool"""
    server: str = Field(..., description="MCP server name (e.g., 'files', 'github')")
    tool: str = Field(..., description="Tool name to execute")
    arguments: Dict[str, Any] = Field(default_factory=dict, description="Tool arguments")

class MCPMethodCallRequest(BaseModel):
    """Request to call an MCP method directly"""
    server: str = Field(..., description="MCP server name")
    method: str = Field(..., description="Method to call (e.g., 'tools/list')")
    params: Dict[str, Any] = Field(default_factory=dict, description="Method parameters")

class MCPServerInfo(BaseModel):
    """MCP server information"""
    name: str
    configured: bool
    running: bool
    tools: List[str]

# API Endpoints
@router.get("/servers", response_model=List[str])
async def list_configured_servers(bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)):
    """
    List all configured MCP servers from .mcp.json
    """
    try:
        servers = list(bridge.config.get("mcpServers", {}).keys())
        return servers
    except Exception as e:
        logger.error(f"Failed to list MCP servers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/servers/{server_name}", response_model=MCPServerInfo)
async def get_server_info(
    server_name: str,
    bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)
):
    """
    Get detailed information about an MCP server
    """
    try:
        info = await bridge.get_server_info(server_name)
        return info
    except Exception as e:
        logger.error(f"Failed to get server info: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/servers/{server_name}/start")
async def start_server(
    server_name: str,
    bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)
):
    """
    Start an MCP server
    """
    try:
        success = await bridge.start_server(server_name)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to start server {server_name}")
        
        # Get server info after starting
        info = await bridge.get_server_info(server_name)
        return {
            "status": "started",
            "server": server_name,
            "info": info
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/servers/{server_name}/stop")
async def stop_server(
    server_name: str,
    bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)
):
    """
    Stop an MCP server
    """
    try:
        success = await bridge.stop_server(server_name)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to stop server {server_name}")
        
        return {
            "status": "stopped",
            "server": server_name
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop server: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/servers/{server_name}/tools")
async def list_server_tools(
    server_name: str,
    bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)
):
    """
    List all tools available on an MCP server
    """
    try:
        tools = await bridge.list_tools(server_name)
        return {
            "server": server_name,
            "tools": tools
        }
    except Exception as e:
        logger.error(f"Failed to list tools: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/call-tool")
async def call_tool(
    request: MCPToolCallRequest,
    bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)
):
    """
    Call a tool on an MCP server
    """
    try:
        result = await bridge.execute_tool(
            server_name=request.server,
            tool_name=request.tool,
            arguments=request.arguments
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Tool execution failed")
        
        return {
            "server": request.server,
            "tool": request.tool,
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to call tool: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/call-method")
async def call_method(
    request: MCPMethodCallRequest,
    bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)
):
    """
    Call a method directly on an MCP server (advanced)
    """
    try:
        result = await bridge.call_method(
            server_name=request.server,
            method=request.method,
            params=request.params
        )
        
        if result is None:
            raise HTTPException(status_code=500, detail="Method call failed")
        
        return {
            "server": request.server,
            "method": request.method,
            "result": result
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to call method: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize-all")
async def initialize_all_servers(bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)):
    """
    Initialize all configured MCP servers
    """
    try:
        results = await bridge.initialize_all()
        
        # Get status of all servers
        server_status = {}
        for server_name in results["started"] + results["failed"]:
            info = await bridge.get_server_info(server_name)
            server_status[server_name] = info
        
        return {
            "status": "initialized",
            "results": results,
            "servers": server_status
        }
    except Exception as e:
        logger.error(f"Failed to initialize servers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/shutdown-all")
async def shutdown_all_servers(bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)):
    """
    Shutdown all MCP servers
    """
    try:
        await bridge.shutdown_all()
        return {
            "status": "shutdown",
            "message": "All MCP servers have been stopped"
        }
    except Exception as e:
        logger.error(f"Failed to shutdown servers: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Convenience endpoints for common operations
@router.post("/files/read")
async def read_file(
    path: str = Body(..., description="File path to read"),
    bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)
):
    """Read a file using the files MCP server"""
    try:
        result = await bridge.execute_tool(
            server_name="files",
            tool_name="read_text_file",
            arguments={"path": path}
        )
        return {"path": path, "content": result}
    except Exception as e:
        logger.error(f"Failed to read file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/files/write")
async def write_file(
    path: str = Body(..., description="File path to write"),
    content: str = Body(..., description="File content"),
    bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)
):
    """Write a file using the files MCP server"""
    try:
        result = await bridge.execute_tool(
            server_name="files",
            tool_name="write_file",
            arguments={"path": path, "content": content}
        )
        return {"path": path, "result": result}
    except Exception as e:
        logger.error(f"Failed to write file: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/files/list")
async def list_directory(
    path: str = Body(..., description="Directory path to list"),
    bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)
):
    """List directory contents using the files MCP server"""
    try:
        result = await bridge.execute_tool(
            server_name="files",
            tool_name="list_directory",
            arguments={"path": path}
        )
        return {"path": path, "contents": result}
    except Exception as e:
        logger.error(f"Failed to list directory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/http/fetch")
async def fetch_url(
    url: str = Body(..., description="URL to fetch"),
    bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)
):
    """Fetch a URL using the http MCP server"""
    try:
        result = await bridge.execute_tool(
            server_name="http",
            tool_name="fetch",
            arguments={"url": url}
        )
        return {"url": url, "response": result}
    except Exception as e:
        logger.error(f"Failed to fetch URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_web(
    query: str = Body(..., description="Search query"),
    bridge: MCPSTDIOBridge = Depends(get_mcp_stdio_bridge)
):
    """Search the web using the ddg MCP server"""
    try:
        result = await bridge.execute_tool(
            server_name="ddg",
            tool_name="search",
            arguments={"query": query}
        )
        return {"query": query, "results": result}
    except Exception as e:
        logger.error(f"Failed to search: {e}")
        raise HTTPException(status_code=500, detail=str(e))