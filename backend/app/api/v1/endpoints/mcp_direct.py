"""
Direct MCP Bridge API - Works without wrapper scripts
"""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
import logging

from app.mcp_bridge import direct_bridge

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp-direct", tags=["mcp-direct"])

class MCPRequest(BaseModel):
    method: str
    params: Dict[str, Any] = {}

@router.get("/servers")
async def list_servers():
    """List available MCP servers"""
    return {"servers": direct_bridge.list_servers()}

@router.get("/servers/{server_name}/status")
async def get_server_status(server_name: str):
    """Get MCP server status"""
    return direct_bridge.get_server_status(server_name)

@router.post("/servers/{server_name}/start")
async def start_server(server_name: str):
    """Start an MCP server"""
    proc = await direct_bridge.start_server(server_name)
    if proc:
        return {"status": "started", "pid": proc.pid, "server": server_name}
    else:
        raise HTTPException(status_code=500, detail=f"Failed to start server {server_name}")

@router.post("/servers/{server_name}/stop")
async def stop_server(server_name: str):
    """Stop an MCP server"""
    await direct_bridge.stop_server(server_name)
    return {"status": "stopped", "server": server_name}

@router.post("/servers/{server_name}/call")
async def call_server(server_name: str, request: MCPRequest):
    """Call an MCP server method"""
    try:
        result = await direct_bridge.call_server(server_name, request.method, request.params)
        return result
    except Exception as e:
        logger.error(f"MCP call failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/test/{server_name}")
async def test_server(server_name: str):
    """Test basic connectivity to an MCP server"""
    try:
        # Try to start the server
        proc = await direct_bridge.start_server(server_name)
        if not proc:
            return {
                "server": server_name,
                "status": "failed",
                "error": "Could not start server"
            }
        
        # Try a simple call (this will vary by server)
        test_methods = {
            "files": ("listFiles", {"path": "/app"}),
            "http": ("fetch", {"url": "https://api.github.com"}),
            "ddg": ("search", {"query": "test"}),
            "github": ("listRepositories", {})
        }
        
        if server_name in test_methods:
            method, params = test_methods[server_name]
            try:
                result = await direct_bridge.call_server(server_name, method, params)
                return {
                    "server": server_name,
                    "status": "success",
                    "test_method": method,
                    "response": result
                }
            except Exception as e:
                return {
                    "server": server_name,
                    "status": "partial",
                    "message": "Server started but test call failed",
                    "error": str(e)
                }
        else:
            return {
                "server": server_name,
                "status": "started",
                "pid": proc.pid,
                "message": "Server started, no test method available"
            }
            
    except Exception as e:
        return {
            "server": server_name,
            "status": "error",
            "error": str(e)
        }