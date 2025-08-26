"""
Connection Manager for MCP Servers

Handles connections to MCP servers using the official Python MCP SDK
with robust retry logic, connection pooling, and health monitoring.
"""

import asyncio
import json
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from mcp import ClientSession, StdioServerParameters
from mcp.client import stdio
from mcp.types import CallToolResult, GetPromptResult, ListPromptsResult, ListToolsResult

from .models import (
    ConnectionState,
    ConnectionType,
    HealthCheckResult,
    HealthStatus,
    MCPCapability,
    ServerConfig,
    ServerStatus,
)


class ConnectionManager:
    """
    Manages connections to MCP servers with retry logic and health monitoring.
    
    Supports multiple connection types:
    - STDIO: Standard input/output communication
    - HTTP: RESTful API communication
    - WebSocket: Real-time bidirectional communication
    - TCP: Direct TCP socket communication
    """
    
    def __init__(self) -> None:
        self._connections: Dict[str, ClientSession] = {}
        self._connection_states: Dict[str, ConnectionState] = {}
        self._processes: Dict[str, subprocess.Popen] = {}
        self._retry_counts: Dict[str, int] = {}
        self._last_activity: Dict[str, datetime] = {}
        
        # Connection pooling and limits
        self._active_connections = 0
        self._max_connections = 50
        
    async def connect_server(
        self, 
        config: ServerConfig, 
        force_reconnect: bool = False
    ) -> Tuple[bool, Optional[str]]:
        """
        Establish connection to an MCP server.
        
        Args:
            config: Server configuration
            force_reconnect: Force reconnection even if already connected
            
        Returns:
            Tuple of (success, error_message)
        """
        server_name = config.name
        
        try:
            # Check if already connected
            if not force_reconnect and self._is_connected(server_name):
                logger.debug(f"Server {server_name} already connected")
                return True, None
            
            # Disconnect existing connection if forcing reconnect
            if force_reconnect and server_name in self._connections:
                await self.disconnect_server(server_name)
            
            # Check connection limits
            if self._active_connections >= self._max_connections:
                return False, "Maximum connections reached"
            
            logger.info(f"Connecting to MCP server: {server_name}")
            
            # Initialize connection state
            self._connection_states[server_name] = ConnectionState(
                server_name=server_name,
                connection_type=config.connection_type,
            )
            
            # Connect based on connection type
            if config.connection_type == ConnectionType.STDIO:
                success, error = await self._connect_stdio(config)
            elif config.connection_type == ConnectionType.HTTP:
                success, error = await self._connect_http(config)
            elif config.connection_type == ConnectionType.WEBSOCKET:
                success, error = await self._connect_websocket(config)
            else:
                return False, f"Unsupported connection type: {config.connection_type}"
            
            if success:
                self._active_connections += 1
                self._connection_states[server_name].is_connected = True
                self._connection_states[server_name].connection_time = datetime.utcnow()
                self._last_activity[server_name] = datetime.utcnow()
                self._retry_counts[server_name] = 0
                
                logger.success(f"Successfully connected to {server_name}")
                
                # Test connection with a capabilities request
                await self._test_connection(server_name, config)
                
            else:
                self._retry_counts[server_name] = self._retry_counts.get(server_name, 0) + 1
                self._connection_states[server_name].connection_errors += 1
                self._connection_states[server_name].last_error = error
                self._connection_states[server_name].last_error_time = datetime.utcnow()
                
                logger.error(f"Failed to connect to {server_name}: {error}")
            
            return success, error
            
        except Exception as e:
            error_msg = f"Connection error: {str(e)}"
            logger.exception(f"Error connecting to {server_name}: {error_msg}")
            
            if server_name in self._connection_states:
                self._connection_states[server_name].connection_errors += 1
                self._connection_states[server_name].last_error = error_msg
                self._connection_states[server_name].last_error_time = datetime.utcnow()
            
            return False, error_msg
    
    async def _connect_stdio(self, config: ServerConfig) -> Tuple[bool, Optional[str]]:
        """Connect to STDIO MCP server"""
        try:
            # Start the server process
            process_args = [config.command] + config.args
            
            # Set working directory if specified
            cwd = str(config.working_directory) if config.working_directory else None
            
            # Prepare environment
            env = dict(config.environment) if config.environment else None
            
            logger.debug(f"Starting process: {' '.join(process_args)}")
            
            # Create server parameters
            server_params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=env,
            )
            
            # Create stdio transport and session
            stdio_transport = stdio.get_stdio_transport(server_params)
            session = ClientSession(stdio_transport)
            
            # Initialize the session
            await session.initialize()
            
            # Store connection
            self._connections[config.name] = session
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    async def _connect_http(self, config: ServerConfig) -> Tuple[bool, Optional[str]]:
        """Connect to HTTP MCP server"""
        # HTTP connections are stateless, so we just validate the endpoint
        try:
            import httpx
            
            base_url = config.base_url or f"http://{config.host}:{config.port}"
            
            async with httpx.AsyncClient(timeout=config.request_timeout) as client:
                # Try to get server capabilities
                response = await client.get(f"{base_url}/mcp/capabilities")
                
                if response.status_code == 200:
                    # Store connection info (HTTP is stateless)
                    self._connection_states[config.name].transport_info = {
                        "base_url": base_url,
                        "client_type": "httpx"
                    }
                    return True, None
                else:
                    return False, f"HTTP {response.status_code}: {response.text}"
                    
        except Exception as e:
            return False, str(e)
    
    async def _connect_websocket(self, config: ServerConfig) -> Tuple[bool, Optional[str]]:
        """Connect to WebSocket MCP server"""
        try:
            import websockets
            
            url = config.base_url or f"ws://{config.host}:{config.port}"
            
            # Connect to WebSocket
            websocket = await websockets.connect(url)
            
            # Store connection (simplified for now)
            self._connection_states[config.name].transport_info = {
                "websocket_url": url,
                "websocket": websocket
            }
            
            return True, None
            
        except Exception as e:
            return False, str(e)
    
    async def _test_connection(self, server_name: str, config: ServerConfig) -> None:
        """Test connection by requesting capabilities"""
        try:
            capabilities = await self.list_tools(server_name, timeout=config.health_check_timeout)
            
            if capabilities:
                logger.debug(f"Server {server_name} has {len(capabilities)} capabilities")
            else:
                logger.warning(f"Server {server_name} reported no capabilities")
                
        except Exception as e:
            logger.warning(f"Failed to test connection to {server_name}: {e}")
    
    async def disconnect_server(self, server_name: str) -> bool:
        """Disconnect from an MCP server"""
        try:
            logger.info(f"Disconnecting from MCP server: {server_name}")
            
            # Close session
            if server_name in self._connections:
                session = self._connections[server_name]
                await session.close()
                del self._connections[server_name]
                self._active_connections -= 1
            
            # Terminate process if exists
            if server_name in self._processes:
                process = self._processes[server_name]
                if process.poll() is None:  # Process is still running
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                del self._processes[server_name]
            
            # Update connection state
            if server_name in self._connection_states:
                self._connection_states[server_name].is_connected = False
                self._connection_states[server_name].connection_time = None
            
            # Clean up tracking
            if server_name in self._last_activity:
                del self._last_activity[server_name]
            
            logger.success(f"Disconnected from {server_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error disconnecting from {server_name}: {e}")
            return False
    
    def _is_connected(self, server_name: str) -> bool:
        """Check if server is connected"""
        return (
            server_name in self._connections and
            server_name in self._connection_states and
            self._connection_states[server_name].is_connected
        )
    
    async def list_tools(
        self, 
        server_name: str, 
        timeout: Optional[float] = None
    ) -> List[MCPCapability]:
        """List available tools from a server"""
        if not self._is_connected(server_name):
            raise ConnectionError(f"Not connected to {server_name}")
        
        try:
            session = self._connections[server_name]
            
            # Update activity timestamp
            self._last_activity[server_name] = datetime.utcnow()
            
            # Get tools from server
            result = await asyncio.wait_for(
                session.list_tools(),
                timeout=timeout
            )
            
            # Convert to our capability format
            capabilities = []
            if hasattr(result, 'tools'):
                for tool in result.tools:
                    capability = MCPCapability(
                        name=tool.name,
                        description=getattr(tool, 'description', ''),
                        parameters=getattr(tool, 'inputSchema', {}).get('properties', {}),
                        required=getattr(tool, 'inputSchema', {}).get('required', [])
                    )
                    capabilities.append(capability)
            
            # Update connection metrics
            state = self._connection_states[server_name]
            state.requests_sent += 1
            state.responses_received += 1
            
            return capabilities
            
        except Exception as e:
            logger.error(f"Failed to list tools from {server_name}: {e}")
            
            # Update error metrics
            if server_name in self._connection_states:
                state = self._connection_states[server_name]
                state.requests_sent += 1
                state.connection_errors += 1
                state.last_error = str(e)
                state.last_error_time = datetime.utcnow()
            
            raise
    
    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None
    ) -> Any:
        """Call a tool on an MCP server"""
        if not self._is_connected(server_name):
            raise ConnectionError(f"Not connected to {server_name}")
        
        try:
            session = self._connections[server_name]
            
            # Update activity timestamp
            self._last_activity[server_name] = datetime.utcnow()
            
            start_time = time.time()
            
            # Call the tool
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments),
                timeout=timeout
            )
            
            # Calculate latency
            latency = time.time() - start_time
            
            # Update connection metrics
            state = self._connection_states[server_name]
            state.requests_sent += 1
            state.responses_received += 1
            
            # Update average latency
            if state.avg_latency == 0:
                state.avg_latency = latency
            else:
                state.avg_latency = (state.avg_latency + latency) / 2
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to call tool {tool_name} on {server_name}: {e}")
            
            # Update error metrics
            if server_name in self._connection_states:
                state = self._connection_states[server_name]
                state.requests_sent += 1
                state.connection_errors += 1
                state.last_error = str(e)
                state.last_error_time = datetime.utcnow()
            
            raise
    
    async def health_check_server(
        self, 
        server_name: str, 
        timeout: float = 10.0
    ) -> HealthCheckResult:
        """Perform health check on an MCP server"""
        start_time = time.time()
        
        try:
            # Check if connected
            if not self._is_connected(server_name):
                return HealthCheckResult(
                    status=HealthStatus.UNHEALTHY,
                    is_running=False,
                    is_responsive=False,
                    error_message="Not connected",
                    error_code="NOT_CONNECTED"
                )
            
            # Try to list capabilities
            capabilities = await self.list_tools(server_name, timeout=timeout)
            response_time = time.time() - start_time
            
            # Determine health status based on response time and capabilities
            if response_time > timeout * 0.8:
                status = HealthStatus.DEGRADED
            else:
                status = HealthStatus.HEALTHY
            
            return HealthCheckResult(
                status=status,
                response_time=response_time,
                is_running=True,
                is_responsive=True,
                capabilities_count=len(capabilities)
            )
            
        except asyncio.TimeoutError:
            return HealthCheckResult(
                status=HealthStatus.UNHEALTHY,
                response_time=timeout,
                is_running=True,
                is_responsive=False,
                error_message="Health check timeout",
                error_code="TIMEOUT"
            )
        except Exception as e:
            return HealthCheckResult(
                status=HealthStatus.CRITICAL,
                response_time=time.time() - start_time,
                is_running=False,
                is_responsive=False,
                error_message=str(e),
                error_code="ERROR"
            )
    
    def get_connection_state(self, server_name: str) -> Optional[ConnectionState]:
        """Get connection state for a server"""
        return self._connection_states.get(server_name)
    
    def get_all_connection_states(self) -> Dict[str, ConnectionState]:
        """Get all connection states"""
        return self._connection_states.copy()
    
    def get_active_connections_count(self) -> int:
        """Get number of active connections"""
        return self._active_connections
    
    async def close_all_connections(self) -> None:
        """Close all connections"""
        logger.info("Closing all MCP connections")
        
        for server_name in list(self._connections.keys()):
            await self.disconnect_server(server_name)
        
        self._connections.clear()
        self._connection_states.clear()
        self._processes.clear()
        self._retry_counts.clear()
        self._last_activity.clear()
        self._active_connections = 0