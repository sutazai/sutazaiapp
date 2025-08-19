"""
MCP Client Service - Real stdio-based implementation
Connects to running MCP servers via their stdio interfaces
"""
import asyncio
import json
import logging
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
import os

logger = logging.getLogger(__name__)

class MCPClient:
    """Client for communicating with MCP servers via stdio"""
    
    def __init__(self, server_name: str, command: List[str]):
        """
        Initialize MCP client
        
        Args:
            server_name: Name of the MCP server
            command: Command to execute the MCP server
        """
        self.server_name = server_name
        self.command = command
        self.process: Optional[asyncio.subprocess.Process] = None
        self.request_id = 0
        self._lock = asyncio.Lock()
        
    async def connect(self) -> bool:
        """Connect to the MCP server"""
        try:
            # Start the MCP server process
            self.process = await asyncio.create_subprocess_exec(
                *self.command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "NODE_ENV": "production"}
            )
            
            # Send initialization request
            init_request = {
                "jsonrpc": "2.0",
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {
                        "name": "sutazai-backend",
                        "version": "1.0.0"
                    }
                },
                "id": self._next_id()
            }
            
            response = await self._send_request(init_request)
            if response and "result" in response:
                logger.info(f"Connected to MCP server: {self.server_name}")
                return True
            else:
                logger.error(f"Failed to initialize MCP server {self.server_name}: {response}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to connect to MCP server {self.server_name}: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from the MCP server"""
        if self.process:
            try:
                self.process.terminate()
                await asyncio.wait_for(self.process.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                self.process.kill()
                await self.process.wait()
            finally:
                self.process = None
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """
        Call a tool on the MCP server
        
        Args:
            tool_name: Name of the tool to call
            arguments: Arguments to pass to the tool
            
        Returns:
            Tool execution result
        """
        request = {
            "jsonrpc": "2.0",
            "method": "tools/call",
            "params": {
                "name": tool_name,
                "arguments": arguments
            },
            "id": self._next_id()
        }
        
        response = await self._send_request(request)
        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            raise Exception(f"Tool call error: {response['error']}")
        else:
            raise Exception(f"Invalid response from MCP server: {response}")
    
    async def list_tools(self) -> List[Dict[str, Any]]:
        """List available tools from the MCP server"""
        request = {
            "jsonrpc": "2.0",
            "method": "tools/list",
            "params": {},
            "id": self._next_id()
        }
        
        response = await self._send_request(request)
        if response and "result" in response:
            return response["result"].get("tools", [])
        return []  # No MCP servers available
    async def list_resources(self) -> List[Dict[str, Any]]:
        """List available resources from the MCP server"""
        request = {
            "jsonrpc": "2.0",
            "method": "resources/list",
            "params": {},
            "id": self._next_id()
        }
        
        response = await self._send_request(request)
        if response and "result" in response:
            return response["result"].get("resources", [])
        return []  # No MCP servers available
    async def read_resource(self, uri: str) -> Dict[str, Any]:
        """Read a resource from the MCP server"""
        request = {
            "jsonrpc": "2.0",
            "method": "resources/read",
            "params": {"uri": uri},
            "id": self._next_id()
        }
        
        response = await self._send_request(request)
        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            raise Exception(f"Resource read error: {response['error']}")
        else:
            raise Exception(f"Invalid response from MCP server: {response}")
    
    async def _send_request(self, request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Send a JSON-RPC request to the MCP server"""
        if not self.process or not self.process.stdin or not self.process.stdout:
            raise Exception(f"MCP server {self.server_name} is not connected")
        
        async with self._lock:
            try:
                # Send request
                request_str = json.dumps(request) + "\n"
                self.process.stdin.write(request_str.encode())
                await self.process.stdin.drain()
                
                # Read response
                response_line = await asyncio.wait_for(
                    self.process.stdout.readline(),
                    timeout=30.0
                )
                
                if not response_line:
                    return None
                
                # Parse JSON-RPC response
                response = json.loads(response_line.decode())
                return response
                
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for response from {self.server_name}")
                return None
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse response from {self.server_name}: {e}")
                return None
            except Exception as e:
                logger.error(f"Error communicating with {self.server_name}: {e}")
                return None
    
    def _next_id(self) -> int:
        """Generate next request ID"""
        self.request_id += 1
        return self.request_id


class MCPRegistry:
    """Registry for managing multiple MCP clients"""
    
    # MCP server configurations
    SERVERS = {
        "files": {
            "command": ["npx", "-y", "@modelcontextprotocol/server-filesystem", "/opt/sutazaiapp"],
            "description": "File system operations"
        },
        "github": {
            "command": ["npx", "-y", "@modelcontextprotocol/server-github"],
            "description": "GitHub repository operations",
            "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": os.getenv("GITHUB_TOKEN", "")}
        },
        "ddg": {
            "command": ["npx", "-y", "@modelcontextprotocol/server-duckduckgo"],
            "description": "DuckDuckGo search"
        },
        "http": {
            "command": ["npx", "-y", "@modelcontextprotocol/server-fetch"],
            "description": "HTTP/HTTPS fetch operations"
        },
        "context7": {
            "command": ["npx", "-y", "@upstash/context7-mcp@latest"],
            "description": "Context management",
            "env": {
                "UPSTASH_VECTOR_REST_URL": os.getenv("UPSTASH_VECTOR_REST_URL", ""),
                "UPSTASH_VECTOR_REST_TOKEN": os.getenv("UPSTASH_VECTOR_REST_TOKEN", "")
            }
        },
        "ultimatecoder": {
            "command": ["python", "-m", "ultimatecoder_mcp.server"],
            "description": "Advanced coding assistance"
        },
        "nx-mcp": {
            "command": ["npx", "-y", "nx-mcp"],
            "description": "NX monorepo operations"
        },
        "memory-bank": {
            "command": ["npx", "-y", "memory-bank-mcp"],
            "description": "Memory storage and retrieval"
        },
        "knowledge-graph": {
            "command": ["npx", "-y", "knowledge-graph-mcp"],
            "description": "Knowledge graph operations"
        },
        "compass": {
            "command": ["npx", "-y", "compass-mcp"],
            "description": "Navigation and guidance"
        }
    }
    
    def __init__(self):
        self.clients: Dict[str, MCPClient] = {}
        self._lock = asyncio.Lock()
    
    async def get_client(self, server_name: str) -> Optional[MCPClient]:
        """Get or create an MCP client for the specified server"""
        async with self._lock:
            # Return existing client if connected
            if server_name in self.clients:
                return self.clients[server_name]
            
            # Create new client
            if server_name not in self.SERVERS:
                logger.error(f"Unknown MCP server: {server_name}")
                return None
            
            config = self.SERVERS[server_name]
            command = config["command"]
            
            # Set environment variables if specified
            if "env" in config:
                for key, value in config["env"].items():
                    if value:
                        os.environ[key] = value
            
            client = MCPClient(server_name, command)
            
            # Try to connect
            if await client.connect():
                self.clients[server_name] = client
                return client
            else:
                logger.error(f"Failed to connect to MCP server: {server_name}")
                return None
    
    async def disconnect_all(self):
        """Disconnect all MCP clients"""
        async with self._lock:
            for client in self.clients.values():
                await client.disconnect()
            self.clients.clear()
    
    async def list_servers(self) -> List[Dict[str, Any]]:
        """List all available MCP servers with their status"""
        servers = []
        for name, config in self.SERVERS.items():
            connected = name in self.clients
            servers.append({
                "name": name,
                "description": config["description"],
                "connected": connected,
                "command": " ".join(config["command"])
            })
        return servers
    
    async def get_server_tools(self, server_name: str) -> List[Dict[str, Any]]:
        """Get tools available from a specific MCP server"""
        client = await self.get_client(server_name)
        if client:
            return await client.list_tools()
        return []  # No MCP servers available
    async def get_server_resources(self, server_name: str) -> List[Dict[str, Any]]:
        """Get resources available from a specific MCP server"""
        client = await self.get_client(server_name)
        if client:
            return await client.list_resources()
        return []  # No MCP servers available
    async def call_server_tool(
        self, 
        server_name: str, 
        tool_name: str, 
        arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a tool on a specific MCP server"""
        client = await self.get_client(server_name)
        if not client:
            raise Exception(f"Failed to connect to MCP server: {server_name}")
        
        return await client.call_tool(tool_name, arguments)
    
    async def read_server_resource(self, server_name: str, uri: str) -> Dict[str, Any]:
        """Read a resource from a specific MCP server"""
        client = await self.get_client(server_name)
        if not client:
            raise Exception(f"Failed to connect to MCP server: {server_name}")
        
        return await client.read_resource(uri)


# Global registry instance
mcp_registry = MCPRegistry()