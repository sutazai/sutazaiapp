"""
MCP Stdio Bridge - Proper MCP Server Integration via stdin/stdout
Replaces the broken TCP-based approach with correct stdio communication
"""
import asyncio
import json
import logging
import os
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class MCPStdioConfig:
    """Configuration for an MCP service using stdio transport"""
    name: str
    wrapper_script: str
    capabilities: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    auto_restart: bool = True
    max_retries: int = 3
    startup_timeout: int = 5  # Reduced from 30 since stdio is faster

class MCPStdioAdapter:
    """Adapter for MCP services using stdio transport"""
    
    def __init__(self, config: MCPStdioConfig):
        self.config = config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.retry_count = 0
        self.last_health_check = datetime.now()
        self._reader_task: Optional[asyncio.Task] = None
        self._running = False
        
    async def start(self) -> bool:
        """Start the MCP service process with stdio transport"""
        try:
            # Start MCP service process with proper stdio setup
            self.process = await asyncio.create_subprocess_exec(
                self.config.wrapper_script,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={
                    **os.environ,
                    "MCP_SERVICE_NAME": self.config.name,
                }
            )
            
            self._running = True
            
            # Start reader task for responses
            self._reader_task = asyncio.create_task(self._read_responses())
            
            # Send initialize request
            initialized = await self._initialize_mcp()
            
            if initialized:
                logger.info(f"MCP service {self.config.name} started successfully via stdio")
                return True
            else:
                logger.error(f"MCP service {self.config.name} failed initialization")
                await self.stop()
                return False
                
        except Exception as e:
            logger.error(f"Failed to start MCP service {self.config.name}: {e}")
            return False
    
    async def stop(self) -> bool:
        """Stop the MCP service process"""
        try:
            self._running = False
            
            # Cancel reader task
            if self._reader_task:
                self._reader_task.cancel()
                try:
                    await self._reader_task
                except asyncio.CancelledError:
                    pass
            
            # Terminate process
            if self.process:
                try:
                    self.process.terminate()
                    await asyncio.wait_for(self.process.wait(), timeout=5)
                except asyncio.TimeoutError:
                    self.process.kill()
                    await self.process.wait()
            
            logger.info(f"MCP service {self.config.name} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop MCP service {self.config.name}: {e}")
            return False
    
    async def restart(self) -> bool:
        """Restart the MCP service"""
        await self.stop()
        await asyncio.sleep(1)  # Brief pause before restart
        return await self.start()
    
    async def health_check(self) -> bool:
        """Check health of the MCP service"""
        try:
            # Check process status
            if not self.process or self.process.returncode is not None:
                logger.warning(f"MCP service {self.config.name} process died")
                if self.config.auto_restart and self.retry_count < self.config.max_retries:
                    self.retry_count += 1
                    return await self.restart()
                return False
            
            # For now, just check if process is running
            # More sophisticated health checks can be added later
            self.last_health_check = datetime.now()
            return True
            
        except Exception as e:
            logger.error(f"Health check failed for MCP service {self.config.name}: {e}")
            return False
    
    async def _initialize_mcp(self) -> bool:
        """Send initialization request to MCP server"""
        try:
            # For now, we'll skip the formal initialization as the servers
            # might not all implement the full MCP protocol yet
            # Just check if the process started successfully
            await asyncio.sleep(1)  # Give process time to start
            
            if self.process and self.process.returncode is None:
                logger.info(f"MCP service {self.config.name} process started")
                return True
            else:
                logger.error(f"MCP service {self.config.name} process failed to start")
                return False
            
        except Exception as e:
            logger.error(f"Failed to initialize MCP service {self.config.name}: {e}")
            return False
    
    async def _send_request(self, request: Dict[str, Any], timeout: float = 5.0) -> Optional[Dict[str, Any]]:
        """Send a request to the MCP server and wait for response"""
        if not self.process or not self.process.stdin:
            return None
        
        try:
            # Send request
            request_str = json.dumps(request) + "\n"
            self.process.stdin.write(request_str.encode())
            await self.process.stdin.drain()
            
            # Wait for response with timeout
            response_future = asyncio.create_future()
            self._pending_requests[request.get("id")] = response_future
            
            try:
                response = await asyncio.wait_for(response_future, timeout=timeout)
                return response
            except asyncio.TimeoutError:
                logger.warning(f"Request timeout for MCP service {self.config.name}")
                return None
            finally:
                self._pending_requests.pop(request.get("id"), None)
                
        except Exception as e:
            logger.error(f"Failed to send request to MCP service {self.config.name}: {e}")
            return None
    
    async def _send_notification(self, notification: Dict[str, Any]) -> None:
        """Send a notification to the MCP server (no response expected)"""
        if not self.process or not self.process.stdin:
            return
        
        try:
            notification_str = json.dumps(notification) + "\n"
            self.process.stdin.write(notification_str.encode())
            await self.process.stdin.drain()
        except Exception as e:
            logger.error(f"Failed to send notification to MCP service {self.config.name}: {e}")
    
    async def _read_responses(self) -> None:
        """Read responses from the MCP server stdout"""
        if not self.process or not self.process.stdout:
            return
        
        buffer = ""
        while self._running:
            try:
                # Read data from stdout
                data = await self.process.stdout.read(1024)
                if not data:
                    break
                
                buffer += data.decode()
                
                # Process complete JSON messages
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    if line.strip():
                        try:
                            message = json.loads(line)
                            await self._handle_message(message)
                        except json.JSONDecodeError:
                            logger.warning(f"Invalid JSON from MCP service {self.config.name}: {line}")
                            
            except Exception as e:
                logger.error(f"Error reading from MCP service {self.config.name}: {e}")
                break
    
    async def _handle_message(self, message: Dict[str, Any]) -> None:
        """Handle a message from the MCP server"""
        # Handle response to a request
        if "id" in message and message["id"] in self._pending_requests:
            future = self._pending_requests.get(message["id"])
            if future and not future.done():
                future.set_result(message)
        
        # Handle notifications/events
        elif "method" in message and not "id" in message:
            # This is a notification from the server
            logger.debug(f"Notification from MCP service {self.config.name}: {message.get('method')}")
    
    def __init__(self, config: MCPStdioConfig):
        self.config = config
        self.process: Optional[asyncio.subprocess.Process] = None
        self.retry_count = 0
        self.last_health_check = datetime.now()
        self._reader_task: Optional[asyncio.Task] = None
        self._running = False
        self._pending_requests: Dict[Any, asyncio.Future] = {}

class MCPStdioBridge:
    """Bridge for MCP servers using stdio transport"""
    
    def __init__(self):
        self.adapters: Dict[str, MCPStdioAdapter] = {}
        self.registry: Dict[str, Any] = self._load_mcp_registry()
        self._initialized = False
        
    def _load_mcp_registry(self) -> Dict[str, Any]:
        """Load MCP service registry from configuration"""
        return {
            "mcp_services": [
                {
                    "name": "postgres",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/postgres.sh",
                    "capabilities": ["sql", "database", "query"],
                    "metadata": {"database": "sutazai", "schema": "public"}
                },
                {
                    "name": "files",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/files.sh",
                    "capabilities": ["read", "write", "list", "delete"],
                    "metadata": {"root": "/opt/sutazaiapp"}
                },
                {
                    "name": "http_fetch",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/http_fetch.sh",
                    "capabilities": ["fetch", "post", "put", "delete"],
                    "metadata": {"timeout": 30}
                },
                {
                    "name": "ddg",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/ddg.sh",
                    "capabilities": ["search", "news", "images"],
                    "metadata": {"engine": "duckduckgo"}
                },
                {
                    "name": "github",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/github.sh",
                    "capabilities": ["repos", "issues", "pulls", "actions"],
                    "metadata": {"api_version": "v3"}
                },
                {
                    "name": "extended-memory",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/extended-memory.sh",
                    "capabilities": ["store", "retrieve", "search", "delete"],
                    "metadata": {"backend": "chromadb"}
                },
                {
                    "capabilities": ["screenshot", "pdf", "scrape", "navigate"],
                    "metadata": {"headless": True}
                },
                {
                    "name": "playwright-mcp",
                    "wrapper": "/opt/sutazaiapp/scripts/mcp/wrappers/playwright-mcp.sh",
                    "capabilities": ["screenshot", "pdf", "scrape", "navigate", "multi-browser"],
                    "metadata": {"browsers": ["chromium", "firefox", "webkit"]}
                }
            ]
        }
    
    async def initialize(self) -> Dict[str, Any]:
        """Initialize all MCP services"""
        if self._initialized:
            return {"status": "already_initialized", "services": list(self.adapters.keys())}
        
        started = []
        failed = []
        
        for service_config in self.registry.get("mcp_services", []):
            config = MCPStdioConfig(
                name=service_config["name"],
                wrapper_script=service_config["wrapper"],
                capabilities=service_config.get("capabilities", []),
                metadata=service_config.get("metadata", {})
            )
            
            adapter = MCPStdioAdapter(config)
            
            if await adapter.start():
                self.adapters[service_config["name"]] = adapter
                started.append(service_config["name"])
            else:
                failed.append(service_config["name"])
        
        self._initialized = True
        
        return {
            "status": "initialized",
            "started": started,
            "failed": failed,
            "total": len(started) + len(failed)
        }
    
    async def shutdown(self) -> bool:
        """Shutdown all MCP services"""
        success = True
        for adapter in self.adapters.values():
            if not await adapter.stop():
                success = False
        
        self.adapters.clear()
        self._initialized = False
        return success
    
    async def call_mcp_service(
        self, 
        service_name: str, 
        method: str, 
        params: Dict[str, Any]
    ) -> Any:
        """Call an MCP service method"""
        
        # Ensure service is registered
        if service_name not in self.adapters:
            raise ValueError(f"MCP service {service_name} not found")
        
        adapter = self.adapters[service_name]
        
        # Send request to MCP server
        request_id = f"{service_name}-{datetime.now().timestamp()}"
        response = await adapter._send_request({
            "jsonrpc": "2.0",
            "method": f"tools/{method}",
            "params": params,
            "id": request_id
        }, timeout=30.0)
        
        if response and "result" in response:
            return response["result"]
        elif response and "error" in response:
            raise Exception(f"MCP call failed: {response['error']}")
        else:
            raise Exception(f"MCP call failed: No response")
    
    async def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get status of an MCP service"""
        if service_name not in self.adapters:
            return {
                "service": service_name,
                "status": "not_found",
                "error": f"Service {service_name} not registered"
            }
        
        adapter = self.adapters[service_name]
        is_healthy = await adapter.health_check()
        
        return {
            "service": service_name,
            "status": "healthy" if is_healthy else "unhealthy",
            "process_running": adapter.process is not None and adapter.process.returncode is None,
            "last_health_check": adapter.last_health_check.isoformat()
        }
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart an MCP service"""
        if service_name not in self.adapters:
            return False
        
        return await self.adapters[service_name].restart()
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Health check all MCP services"""
        results = {}
        
        for name, adapter in self.adapters.items():
            healthy = await adapter.health_check()
            results[name] = {
                "healthy": healthy,
                "process_running": adapter.process is not None and adapter.process.returncode is None,
                "last_check": adapter.last_health_check.isoformat()
            }
        
        return results

# Global bridge instance
_mcp_stdio_bridge: Optional[MCPStdioBridge] = None

async def get_mcp_stdio_bridge() -> MCPStdioBridge:
    """Get or create MCP stdio bridge instance"""
    global _mcp_stdio_bridge
    
    if _mcp_stdio_bridge is None:
        _mcp_stdio_bridge = MCPStdioBridge()
    
    return _mcp_stdio_bridge