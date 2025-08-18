"""
Direct MCP Bridge - Bypasses wrapper scripts and communicates directly
"""
import asyncio
import json
import logging
import subprocess
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)

class DirectMCPBridge:
    """Direct bridge to MCP servers without wrapper scripts"""
    
    # Direct command mappings for MCP servers
    MCP_COMMANDS = {
        "files": ["npx", "@modelcontextprotocol/server-filesystem", "/opt/sutazaiapp"],
        "github": ["npx", "@modelcontextprotocol/server-github", "--repositories", "sutazai/sutazaiapp"],
        "http": ["npx", "@modelcontextprotocol/server-http"],
        "ddg": ["npx", "@modelcontextprotocol/server-duckduckgo"],
        "postgres": ["npx", "@modelcontextprotocol/server-postgres", "postgresql://sutazai:change_me_secure@sutazai-postgres:5432/sutazai"],
        "context7": ["npx", "context7-mcp"],
        "ultimatecoder": ["npx", "ultimatecoder-mcp"],
        "sequentialthinking": ["npx", "@modelcontextprotocol/server-sequential-thinking"],
        "extended-memory": ["npx", "@modelcontextprotocol/server-memory"],
        "nx-mcp": ["npx", "nx-mcp-server"],
        "puppeteer-mcp": ["npx", "@modelcontextprotocol/server-puppeteer"],
        "playwright-mcp": ["npx", "playwright-mcp-server"],
        "memory-bank-mcp": ["npx", "memory-bank-server"],
        "knowledge-graph-mcp": ["npx", "@modelcontextprotocol/server-knowledge-graph"],
        "compass-mcp": ["npx", "compass-mcp-server"],
        "language-server": ["npx", "@modelcontextprotocol/server-language"]
    }
    
    def __init__(self):
        self.active_processes = {}
        self.check_node_availability()
    
    def check_node_availability(self):
        """Check if Node.js and npx are available"""
        try:
            result = subprocess.run(["npx", "--version"], capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                logger.info(f"npx available: {result.stdout.strip()}")
            else:
                logger.warning("npx not available - installing Node.js may be required")
        except Exception as e:
            logger.error(f"npx check failed: {e}")
    
    async def start_server(self, server_name: str) -> Optional[subprocess.Popen]:
        """Start an MCP server process"""
        if server_name in self.active_processes:
            proc = self.active_processes[server_name]
            if proc.poll() is None:
                return proc
            # Process died, remove it
            del self.active_processes[server_name]
        
        command = self.MCP_COMMANDS.get(server_name)
        if not command:
            logger.error(f"Unknown MCP server: {server_name}")
            return None
        
        try:
            logger.info(f"Starting MCP server {server_name} with command: {' '.join(command)}")
            proc = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=0,
                env={
                    **subprocess.os.environ,
                    "NODE_ENV": "production",
                    "MCP_MODE": "stdio"
                }
            )
            self.active_processes[server_name] = proc
            
            # Give it a moment to start
            await asyncio.sleep(0.5)
            
            # Check if it's still running
            if proc.poll() is not None:
                stderr = proc.stderr.read()
                logger.error(f"MCP server {server_name} failed to start: {stderr}")
                del self.active_processes[server_name]
                return None
            
            logger.info(f"MCP server {server_name} started with PID {proc.pid}")
            return proc
            
        except Exception as e:
            logger.error(f"Failed to start MCP server {server_name}: {e}")
            return None
    
    async def call_server(self, server_name: str, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Call an MCP server method"""
        proc = await self.start_server(server_name)
        if not proc:
            raise Exception(f"Failed to start MCP server {server_name}")
        
        # Create JSON-RPC request
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        try:
            # Send request
            request_str = json.dumps(request) + "\n"
            proc.stdin.write(request_str)
            proc.stdin.flush()
            
            # Read response with timeout
            try:
                response_line = await asyncio.wait_for(
                    asyncio.get_event_loop().run_in_executor(None, proc.stdout.readline),
                    timeout=10.0
                )
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for response from {server_name}")
                raise Exception("MCP server response timeout")
            
            if not response_line:
                raise Exception("No response from MCP server")
            
            response = json.loads(response_line.strip())
            return response
            
        except Exception as e:
            logger.error(f"MCP call to {server_name} failed: {e}")
            # Try to get error output
            if proc.stderr:
                stderr = proc.stderr.read()
                if stderr:
                    logger.error(f"MCP server stderr: {stderr}")
            raise
    
    def list_servers(self) -> list:
        """List available MCP servers"""
        return list(self.MCP_COMMANDS.keys())
    
    def get_server_status(self, server_name: str) -> Dict[str, Any]:
        """Get status of an MCP server"""
        if server_name not in self.MCP_COMMANDS:
            return {"status": "unknown", "server": server_name}
        
        if server_name in self.active_processes:
            proc = self.active_processes[server_name]
            if proc.poll() is None:
                return {
                    "status": "running",
                    "pid": proc.pid,
                    "server": server_name
                }
            else:
                return {
                    "status": "stopped",
                    "exit_code": proc.poll(),
                    "server": server_name
                }
        
        return {"status": "not_started", "server": server_name}
    
    async def stop_server(self, server_name: str):
        """Stop an MCP server"""
        if server_name in self.active_processes:
            proc = self.active_processes[server_name]
            if proc.poll() is None:
                proc.terminate()
                try:
                    await asyncio.wait_for(
                        asyncio.get_event_loop().run_in_executor(None, proc.wait),
                        timeout=5.0
                    )
                except asyncio.TimeoutError:
                    proc.kill()
                    proc.wait()
            del self.active_processes[server_name]
            logger.info(f"Stopped MCP server {server_name}")
    
    async def cleanup(self):
        """Clean up all active processes"""
        for server_name in list(self.active_processes.keys()):
            await self.stop_server(server_name)

# Global instance
direct_bridge = DirectMCPBridge()