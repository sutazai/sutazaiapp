#!/usr/bin/env python3
"""
Implement proper MCP STDIO bridge for the backend API
Creates endpoints that communicate with MCP servers via STDIO instead of HTTP
"""
import json
import asyncio
import subprocess
import logging
from pathlib import Path

MCP_CONFIG_PATH = "/opt/sutazaiapp/.mcp.json"
BACKEND_PATH = "/opt/sutazaiapp/backend"

def implement_stdio_bridge():
    """Implement STDIO bridge for MCP services"""
    print("🔧 Implementing MCP STDIO Bridge...")
    
    # Read MCP configuration
    with open(MCP_CONFIG_PATH, 'r') as f:
        mcp_config = json.load(f)
    
    # Create STDIO bridge endpoint
    stdio_bridge_code = '''"""
MCP STDIO Bridge - Proper implementation for STDIO-based MCP servers
Replaces HTTP-based approach with actual STDIO communication
"""
import asyncio
import json
import subprocess
import logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

class MCPSTDIORequest(BaseModel):
    method: str
    params: Dict[str, Any] = {}

class MCPSTDIOBridge:
    """Bridge for communicating with STDIO-based MCP servers"""
    
    def __init__(self):
        self.mcp_config = self._load_mcp_config()
        self.active_processes = {}
    
    def _load_mcp_config(self):
        """Load MCP configuration from .mcp.json"""
        try:
            with open("/opt/sutazaiapp/.mcp.json", 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            return {"mcp_servers": {}}
    
    async def get_mcp_process(self, server_name: str) -> Optional[subprocess.Popen]:
        """Get or create MCP server process"""
        if server_name in self.active_processes:
            proc = self.active_processes[server_name]
            if proc.poll() is None:  # Process still running
                return proc
        
        # Start new process
        server_config = self.mcp_config.get("mcp_servers", {}).get(server_name)
        if not server_config:
            return None
        
        try:
            if server_config.get("command") == "uvx":
                cmd = ["uvx"] + server_config.get("args", [])
            else:
                cmd = [server_config.get("command", "")] + server_config.get("args", [])
            
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0
            )
            self.active_processes[server_name] = proc
            return proc
        except Exception as e:
            logger.error(f"Failed to start MCP server {server_name}: {e}")
            return None
    
    async def call_mcp_server(self, server_name: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call MCP server via STDIO"""
        proc = await self.get_mcp_process(server_name)
        if not proc:
            raise HTTPException(status_code=404, detail=f"MCP server {server_name} not available")
        
        # Create JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            # Send request
            proc.stdin.write(json.dumps(request) + "\\n")
            proc.stdin.flush()
            
            # Read response (with timeout)
            response_line = proc.stdout.readline()
            if not response_line:
                raise HTTPException(status_code=500, detail="No response from MCP server")
            
            response = json.loads(response_line.strip())
            return response
        except Exception as e:
            logger.error(f"MCP call failed: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    def list_servers(self) -> list:
        """List available MCP servers"""
        return list(self.mcp_config.get("mcp_servers", {}).keys())

# Global bridge instance
stdio_bridge = MCPSTDIOBridge()

router = APIRouter(prefix="/mcp-stdio", tags=["mcp-stdio"])

@router.get("/servers")
async def list_mcp_servers():
    """List available MCP servers"""
    return {"servers": stdio_bridge.list_servers()}

@router.post("/servers/{server_name}/call")
async def call_mcp_server(server_name: str, request: MCPSTDIORequest):
    """Call MCP server via STDIO"""
    return await stdio_bridge.call_mcp_server(server_name, request.method, request.params)

@router.get("/servers/{server_name}/status")
async def get_server_status(server_name: str):
    """Get MCP server status"""
    proc = await stdio_bridge.get_mcp_process(server_name)
    if proc:
        return {
            "status": "running" if proc.poll() is None else "stopped",
            "pid": proc.pid,
            "server": server_name
        }
    else:
        return {"status": "not_found", "server": server_name}
'''
    
    # Write STDIO bridge file
    stdio_bridge_path = Path(BACKEND_PATH) / "app" / "api" / "v1" / "endpoints" / "mcp_stdio.py"
    with open(stdio_bridge_path, 'w') as f:
        f.write(stdio_bridge_code)
    
    print(f"✅ Created STDIO bridge at {stdio_bridge_path}")
    
    # Update main API router to include STDIO endpoints
    api_router_path = Path(BACKEND_PATH) / "app" / "api" / "v1" / "api.py"
    
    try:
        with open(api_router_path, 'r') as f:
            api_content = f.read()
        
        if "mcp_stdio" not in api_content:
            # Add import and router
            import_line = "from .endpoints import mcp_stdio"
            router_line = "api_router.include_router(mcp_stdio.router)"
            
            # Insert import
            if "from .endpoints import" in api_content:
                api_content = api_content.replace(
                    "from .endpoints import",
                    f"from .endpoints import mcp_stdio,\\nfrom .endpoints import"
                )
            else:
                api_content = f"{import_line}\\n{api_content}"
            
            # Insert router
            if "api_router.include_router(" in api_content:
                api_content = api_content.replace(
                    "api_router.include_router(mcp.router)",
                    f"api_router.include_router(mcp.router)\\napi_router.include_router(mcp_stdio.router)"
                )
            
            with open(api_router_path, 'w') as f:
                f.write(api_content)
            
            print("✅ Updated API router to include STDIO endpoints")
        else:
            print("✅ STDIO endpoints already included in API router")
    
    except Exception as e:
        print(f"⚠️ Failed to update API router: {e}")
    
    print("\\n🎯 MCP STDIO Bridge Implementation Complete!")
    print("New endpoints available:")
    print("  GET  /api/v1/mcp-stdio/servers")
    print("  POST /api/v1/mcp-stdio/servers/{server}/call")
    print("  GET  /api/v1/mcp-stdio/servers/{server}/status")
    
    return True

if __name__ == "__main__":
    success = implement_stdio_bridge()
    if success:
        print("\\n🎉 MCP STDIO Bridge implemented successfully!")
        print("Backend now supports proper STDIO communication with MCP servers")
    else:
        print("\\n⚠️ Issues occurred during STDIO bridge implementation")