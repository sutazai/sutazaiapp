"""
MCP Protocol Translation Layer
Bridges STDIO-based MCP servers with HTTP/TCP mesh communication
Resolves protocol incompatibility and enables seamless integration
"""
import asyncio
import json
import logging
import subprocess
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import hashlib
import threading
from queue import Queue, Empty
import signal

logger = logging.getLogger(__name__)

@dataclass
class MCPRequest:
    """Represents a request to an MCP server"""
    id: str
    method: str
    params: Dict[str, Any]
    timeout: float = 30.0
    trace_id: Optional[str] = None
    
    def to_stdio_format(self) -> str:
        """Convert to MCP STDIO JSON-RPC format"""
        return json.dumps({
            "jsonrpc": "2.0",
            "id": self.id,
            "method": self.method,
            "params": self.params
        }) + "\n"

@dataclass
class MCPResponse:
    """Represents a response from an MCP server"""
    id: str
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    duration: float = 0.0
    
    def to_http_format(self) -> Dict[str, Any]:
        """Convert to HTTP response format"""
        if self.error:
            return {
                "status_code": 500,
                "body": {"error": self.error},
                "headers": {"X-Response-Time": str(self.duration)}
            }
        return {
            "status_code": 200,
            "body": self.result,
            "headers": {"X-Response-Time": str(self.duration)}
        }

class STDIOProcessManager:
    """Manages STDIO process lifecycle with resource isolation"""
    
    def __init__(self, wrapper_script: str, service_name: str, port: int):
        self.wrapper_script = wrapper_script
        self.service_name = service_name
        self.port = port
        self.process: Optional[subprocess.Popen] = None
        self.stdin_queue = Queue()
        self.stdout_queue = Queue()
        self.stderr_queue = Queue()
        self.reader_thread: Optional[threading.Thread] = None
        self.writer_thread: Optional[threading.Thread] = None
        self.running = False
        self.lock = threading.Lock()
        
    async def start(self) -> bool:
        """Start the STDIO process with proper isolation"""
        try:
            with self.lock:
                if self.running:
                    return True
                
                # Create isolated environment
                env = os.environ.copy()
                env.update({
                    "MCP_SERVICE_NAME": self.service_name,
                    "MCP_SERVICE_PORT": str(self.port),
                    "MCP_ISOLATION": "true",
                    "NODE_OPTIONS": "--max-old-space-size=512",  # Limit memory
                    "PYTHONUNBUFFERED": "1"
                })
                
                # Start process with resource limits
                self.process = subprocess.Popen(
                    [self.wrapper_script],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env=env,
                    bufsize=0,  # Unbuffered
                    preexec_fn=os.setsid if os.name != 'nt' else None  # Process group for cleanup
                )
                
                # Start reader/writer threads
                self.running = True
                self.reader_thread = threading.Thread(target=self._read_output, daemon=True)
                self.writer_thread = threading.Thread(target=self._write_input, daemon=True)
                self.reader_thread.start()
                self.writer_thread.start()
                
                # Wait for process to be ready
                await self._wait_for_ready()
                
                logger.info(f"✅ Started STDIO process for {self.service_name}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start STDIO process for {self.service_name}: {e}")
            self.running = False
            return False
    
    async def stop(self) -> bool:
        """Stop the STDIO process gracefully"""
        try:
            with self.lock:
                if not self.running:
                    return True
                
                self.running = False
                
                # Send shutdown signal
                if self.process:
                    # Try graceful shutdown first
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        # Force kill if needed
                        if os.name != 'nt':
                            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                        else:
                            self.process.kill()
                        self.process.wait()
                
                logger.info(f"✅ Stopped STDIO process for {self.service_name}")
                return True
                
        except Exception as e:
            logger.error(f"Error stopping STDIO process for {self.service_name}: {e}")
            return False
    
    def _read_output(self):
        """Read thread for STDOUT"""
        while self.running and self.process:
            try:
                line = self.process.stdout.readline()
                if line:
                    try:
                        # Try to parse as JSON-RPC response
                        data = json.loads(line.decode('utf-8'))
                        self.stdout_queue.put(data)
                    except json.JSONDecodeError:
                        # Store raw output
                        self.stdout_queue.put({"raw": line.decode('utf-8')})
                else:
                    if self.process.poll() is not None:
                        # Process ended
                        self.running = False
                        break
            except Exception as e:
                logger.debug(f"Read error for {self.service_name}: {e}")
    
    def _write_input(self):
        """Write thread for STDIN"""
        while self.running and self.process:
            try:
                # Get from queue with timeout
                data = self.stdin_queue.get(timeout=0.1)
                if data and self.process.stdin:
                    self.process.stdin.write(data.encode('utf-8'))
                    self.process.stdin.flush()
            except Empty:
                continue
            except Exception as e:
                logger.debug(f"Write error for {self.service_name}: {e}")
    
    async def send_request(self, request: MCPRequest) -> MCPResponse:
        """Send request and wait for response"""
        if not self.running:
            return MCPResponse(
                id=request.id,
                error={"code": -32603, "message": "Process not running"}
            )
        
        # Send request
        self.stdin_queue.put(request.to_stdio_format())
        
        # Wait for response with timeout
        start_time = asyncio.get_event_loop().time()
        timeout = request.timeout
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            try:
                # Check for response
                if not self.stdout_queue.empty():
                    response_data = self.stdout_queue.get_nowait()
                    
                    # Check if this is our response
                    if isinstance(response_data, dict) and response_data.get("id") == request.id:
                        duration = asyncio.get_event_loop().time() - start_time
                        
                        if "error" in response_data:
                            return MCPResponse(
                                id=request.id,
                                error=response_data["error"],
                                duration=duration
                            )
                        else:
                            return MCPResponse(
                                id=request.id,
                                result=response_data.get("result"),
                                duration=duration
                            )
                    else:
                        # Not our response, put it back
                        self.stdout_queue.put(response_data)
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error waiting for response: {e}")
        
        # Timeout
        return MCPResponse(
            id=request.id,
            error={"code": -32603, "message": "Request timeout"},
            duration=timeout
        )
    
    async def _wait_for_ready(self, timeout: float = 10.0):
        """Wait for process to be ready"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            if self.process and self.process.poll() is None:
                # Send a test request
                test_request = MCPRequest(
                    id=str(uuid.uuid4()),
                    method="initialize",
                    params={},
                    timeout=2.0
                )
                
                response = await self.send_request(test_request)
                if not response.error or response.error.get("code") != -32603:
                    # Process is responding
                    return True
            
            await asyncio.sleep(0.5)
        
        return False
    
    def is_healthy(self) -> bool:
        """Check if process is healthy"""
        return self.running and self.process and self.process.poll() is None

class MCPProtocolTranslator:
    """
    Translates between STDIO MCP protocol and HTTP/TCP mesh protocol
    Enables seamless communication between different protocol layers
    """
    
    def __init__(self):
        self.process_managers: Dict[str, STDIOProcessManager] = {}
        self.request_cache: Dict[str, MCPRequest] = {}
        self.response_cache: Dict[str, MCPResponse] = {}
        self.cache_ttl = timedelta(minutes=5)
        
    async def translate_http_to_stdio(
        self,
        service_name: str,
        http_request: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Translate HTTP request to STDIO MCP request and back
        
        Args:
            service_name: Name of the MCP service
            http_request: HTTP request with method, params, etc.
        
        Returns:
            HTTP response with status_code, body, headers
        """
        try:
            # Extract request details
            method = http_request.get("method", "")
            params = http_request.get("params", {})
            timeout = http_request.get("timeout", 30.0)
            trace_id = http_request.get("trace_id", str(uuid.uuid4()))
            
            # Create MCP request
            mcp_request = MCPRequest(
                id=trace_id,
                method=method,
                params=params,
                timeout=timeout,
                trace_id=trace_id
            )
            
            # Get process manager
            if service_name not in self.process_managers:
                return {
                    "status_code": 503,
                    "body": {"error": f"Service {service_name} not available"},
                    "headers": {}
                }
            
            manager = self.process_managers[service_name]
            
            # Check health
            if not manager.is_healthy():
                # Try to restart
                await manager.stop()
                if not await manager.start():
                    return {
                        "status_code": 503,
                        "body": {"error": f"Service {service_name} unhealthy"},
                        "headers": {}
                    }
            
            # Send request and get response
            mcp_response = await manager.send_request(mcp_request)
            
            # Convert to HTTP format
            return mcp_response.to_http_format()
            
        except Exception as e:
            logger.error(f"Translation error for {service_name}: {e}")
            return {
                "status_code": 500,
                "body": {"error": str(e)},
                "headers": {}
            }
    
    async def register_mcp_service(
        self,
        service_name: str,
        wrapper_script: str,
        port: int
    ) -> bool:
        """
        Register and start an MCP service with protocol translation
        
        Args:
            service_name: Name of the service
            wrapper_script: Path to wrapper script
            port: Port for the service
        
        Returns:
            True if successful
        """
        try:
            # Check if already registered
            if service_name in self.process_managers:
                logger.warning(f"Service {service_name} already registered")
                return True
            
            # Create process manager
            manager = STDIOProcessManager(wrapper_script, service_name, port)
            
            # Start the process
            if await manager.start():
                self.process_managers[service_name] = manager
                logger.info(f"✅ Registered MCP service {service_name} with protocol translation")
                return True
            else:
                logger.error(f"Failed to start MCP service {service_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error registering MCP service {service_name}: {e}")
            return False
    
    async def unregister_mcp_service(self, service_name: str) -> bool:
        """Unregister and stop an MCP service"""
        try:
            if service_name not in self.process_managers:
                return True
            
            manager = self.process_managers[service_name]
            success = await manager.stop()
            
            if success:
                del self.process_managers[service_name]
                logger.info(f"✅ Unregistered MCP service {service_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Error unregistering MCP service {service_name}: {e}")
            return False
    
    async def get_service_health(self, service_name: str) -> Dict[str, Any]:
        """Get health status of an MCP service"""
        if service_name not in self.process_managers:
            return {
                "service": service_name,
                "status": "not_found",
                "healthy": False
            }
        
        manager = self.process_managers[service_name]
        is_healthy = manager.is_healthy()
        
        return {
            "service": service_name,
            "status": "healthy" if is_healthy else "unhealthy",
            "healthy": is_healthy,
            "process_running": manager.process is not None and manager.process.poll() is None
        }
    
    async def health_check_all(self) -> Dict[str, Any]:
        """Health check all registered MCP services"""
        results = {}
        
        for service_name in self.process_managers:
            results[service_name] = await self.get_service_health(service_name)
        
        healthy_count = sum(1 for r in results.values() if r["healthy"])
        total = len(results)
        
        return {
            "services": results,
            "summary": {
                "total": total,
                "healthy": healthy_count,
                "unhealthy": total - healthy_count,
                "percentage_healthy": (healthy_count / total * 100) if total > 0 else 0
            }
        }
    
    async def shutdown_all(self):
        """Shutdown all MCP services"""
        for service_name in list(self.process_managers.keys()):
            await self.unregister_mcp_service(service_name)

# Global translator instance
_protocol_translator: Optional[MCPProtocolTranslator] = None

async def get_protocol_translator() -> MCPProtocolTranslator:
    """Get or create protocol translator instance"""
    global _protocol_translator
    
    if _protocol_translator is None:
        _protocol_translator = MCPProtocolTranslator()
    
    return _protocol_translator