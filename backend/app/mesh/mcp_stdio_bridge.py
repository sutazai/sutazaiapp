"""
MCP STDIO Bridge - Real MCP server communication via stdin/stdout
This replaces the fake HTTP endpoints with actual MCP protocol communication
"""
import asyncio
import json
import logging
import subprocess
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import os

logger = logging.getLogger(__name__)

@dataclass
class MCPServer:
    """Represents an MCP server instance"""
    name: str
    command: str
    args: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    process: Optional[subprocess.Popen] = None
    reader: Optional[asyncio.StreamReader] = None
    writer: Optional[asyncio.StreamWriter] = None

class MCPSTDIOBridge:
    """Manages STDIO communication with MCP servers"""
    
    def __init__(self, config_path: str = "/app/.mcp.json"):
        self.config_path = config_path
        self.servers: Dict[str, MCPServer] = {}
        self.config: Dict[str, Any] = {}
        self._load_config()
        
    def _load_config(self):
        """Load MCP configuration from .mcp.json file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    self.config = json.load(f)
                    logger.info(f"Loaded MCP config from {self.config_path}")
            else:
                logger.warning(f"MCP config not found at {self.config_path}")
                # Fallback to hardcoded config for testing
                self.config = self._get_fallback_config()
        except Exception as e:
            logger.error(f"Failed to load MCP config: {e}")
            self.config = self._get_fallback_config()
    
    def _get_fallback_config(self) -> Dict[str, Any]:
        """Fallback configuration when .mcp.json is not available"""
        return {
            "mcpServers": {
                "files": {
                    "type": "stdio",
                    "command": "/opt/sutazaiapp/scripts/mcp/wrappers/files.sh",
                    "args": [],
                    "env": {}
                },
                "github": {
                    "type": "stdio", 
                    "command": "sh",
                    "args": ["-lc", "npx -y @modelcontextprotocol/server-github --repositories 'sutazai/sutazaiapp'"],
                    "env": {}
                },
                "http": {
                    "type": "stdio",
                    "command": "/opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh",
                    "args": [],
                    "env": {}
                },
                "ddg": {
                    "type": "stdio",
                    "command": "/opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh",
                    "args": [],
                    "env": {}
                }
            }
        }
    
    async def start_server(self, server_name: str) -> bool:
        """Start an MCP server process"""
        try:
            if server_name in self.servers and self.servers[server_name].process:
                logger.info(f"Server {server_name} already running")
                return True
            
            server_config = self.config.get("mcpServers", {}).get(server_name)
            if not server_config:
                logger.error(f"No configuration found for server {server_name}")
                return False
            
            # Prepare environment
            env = os.environ.copy()
            env.update(server_config.get("env", {}))
            
            # Start process with STDIO pipes
            command = server_config["command"]
            args = server_config.get("args", [])
            
            # Handle different command formats
            if isinstance(command, str) and command == "sh" and args:
                # Special case for shell commands
                full_command = ["sh"] + args
            else:
                full_command = [command] + args
            
            logger.info(f"Starting MCP server {server_name}: {full_command}")
            
            process = await asyncio.create_subprocess_exec(
                *full_command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            # Store server info
            self.servers[server_name] = MCPServer(
                name=server_name,
                command=command,
                args=args,
                env=server_config.get("env", {}),
                process=process,
                reader=process.stdout,
                writer=process.stdin
            )
            
            # Test server by listing tools
            test_result = await self.call_method(server_name, "tools/list", {})
            if test_result:
                logger.info(f"MCP server {server_name} started successfully")
                return True
            else:
                logger.error(f"MCP server {server_name} started but not responding")
                await self.stop_server(server_name)
                return False
                
        except Exception as e:
            logger.error(f"Failed to start MCP server {server_name}: {e}")
            return False
    
    async def stop_server(self, server_name: str) -> bool:
        """Stop an MCP server process"""
        try:
            if server_name not in self.servers:
                return True
            
            server = self.servers[server_name]
            if server.process:
                server.process.terminate()
                try:
                    await asyncio.wait_for(server.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    server.process.kill()
                    await server.process.wait()
            
            del self.servers[server_name]
            logger.info(f"MCP server {server_name} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop MCP server {server_name}: {e}")
            return False
    
    async def call_method(self, server_name: str, method: str, params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Call an MCP method via STDIO"""
        try:
            # Ensure server is running
            if server_name not in self.servers:
                if not await self.start_server(server_name):
                    return None
            
            server = self.servers[server_name]
            if not server.process or not server.writer or not server.reader:
                logger.error(f"Server {server_name} not properly initialized")
                return None
            
            # Create JSON-RPC request
            request = {
                "jsonrpc": "2.0",
                "id": str(uuid.uuid4()),
                "method": method,
                "params": params
            }
            
            # Send request
            request_json = json.dumps(request) + "\n"
            server.writer.write(request_json.encode())
            await server.writer.drain()
            
            # Read response with timeout
            try:
                response_line = await asyncio.wait_for(
                    server.reader.readline(),
                    timeout=30.0
                )
                
                if not response_line:
                    logger.error(f"Empty response from {server_name}")
                    return None
                
                response = json.loads(response_line.decode())
                
                # Check for errors
                if "error" in response:
                    logger.error(f"MCP error from {server_name}: {response['error']}")
                    return None
                
                return response.get("result")
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for response from {server_name}")
                # Restart server on timeout
                await self.stop_server(server_name)
                return None
                
        except Exception as e:
            logger.error(f"Failed to call method {method} on {server_name}: {e}")
            return None
    
    async def list_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """List available tools for a server"""
        result = await self.call_method(server_name, "tools/list", {})
        if result and "tools" in result:
            return result["tools"]
        return []
    
    async def execute_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> Optional[Any]:
        """Execute a tool on an MCP server"""
        result = await self.call_method(
            server_name,
            "tools/call",
            {
                "name": tool_name,
                "arguments": arguments
            }
        )
        return result
    
    async def get_server_info(self, server_name: str) -> Dict[str, Any]:
        """Get information about a server"""
        info = {
            "name": server_name,
            "configured": server_name in self.config.get("mcpServers", {}),
            "running": False,
            "tools": []
        }
        
        if server_name in self.servers:
            server = self.servers[server_name]
            info["running"] = server.process is not None and server.process.returncode is None
            
            if info["running"]:
                tools = await self.list_tools(server_name)
                info["tools"] = [tool.get("name") for tool in tools]
        
        return info
    
    async def initialize_all(self) -> Dict[str, Any]:
        """Initialize all configured MCP servers"""
        results = {
            "started": [],
            "failed": [],
            "total": 0
        }
        
        for server_name in self.config.get("mcpServers", {}).keys():
            results["total"] += 1
            if await self.start_server(server_name):
                results["started"].append(server_name)
            else:
                results["failed"].append(server_name)
        
        return results
    
    async def shutdown_all(self):
        """Shutdown all MCP servers"""
        servers = list(self.servers.keys())
        for server_name in servers:
            await self.stop_server(server_name)
        
        logger.info("All MCP servers shut down")

# Singleton instance
_mcp_bridge = None

async def get_mcp_stdio_bridge() -> MCPSTDIOBridge:
    """Get the singleton MCP STDIO bridge instance"""
    global _mcp_bridge
    if _mcp_bridge is None:
        _mcp_bridge = MCPSTDIOBridge()
    return _mcp_bridge