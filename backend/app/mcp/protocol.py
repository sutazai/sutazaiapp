"""
Production-Grade MCP Protocol Implementation
Full JSON-RPC 2.0 compliance with MCP extensions
"""
import json
import asyncio
import logging
import uuid
from typing import Any, Dict, List, Optional, Union, Callable
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
import hashlib

logger = logging.getLogger(__name__)

class MCPVersion(Enum):
    """MCP Protocol versions"""
    V1_0_0 = "2024-11-05"
    LATEST = V1_0_0

class ErrorCode(Enum):
    """JSON-RPC 2.0 and MCP-specific error codes"""
    # JSON-RPC 2.0 standard errors
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603
    
    # MCP-specific errors
    PROTOCOL_ERROR = -32000
    CAPABILITY_NOT_SUPPORTED = -32001
    RESOURCE_NOT_FOUND = -32002
    TOOL_NOT_FOUND = -32003
    PROMPT_NOT_FOUND = -32004
    RATE_LIMIT_EXCEEDED = -32005
    AUTHENTICATION_REQUIRED = -32006
    AUTHORIZATION_FAILED = -32007
    TIMEOUT = -32008
    CANCELLED = -32009

@dataclass
class MCPError:
    """MCP Error representation"""
    code: int
    message: str
    data: Optional[Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"code": self.code, "message": self.message}
        if self.data is not None:
            result["data"] = self.data
        return result

@dataclass
class MCPRequest:
    """MCP Request message"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Union[Dict[str, Any], List[Any]]] = None
    id: Optional[Union[str, int]] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCPRequest":
        """Create request from dictionary"""
        return cls(
            jsonrpc=data.get("jsonrpc", "2.0"),
            method=data.get("method", ""),
            params=data.get("params"),
            id=data.get("id")
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            result["params"] = self.params
        if self.id is not None:
            result["id"] = self.id
        return result
    
    def is_notification(self) -> bool:
        """Check if this is a notification (no id)"""
        return self.id is None

@dataclass
class MCPResponse:
    """MCP Response message"""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[MCPError] = None
    id: Optional[Union[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"jsonrpc": self.jsonrpc}
        
        if self.error is not None:
            result["error"] = self.error.to_dict()
        elif self.result is not None:
            result["result"] = self.result
        else:
            result["result"] = None
            
        if self.id is not None:
            result["id"] = self.id
            
        return result
    
    @classmethod
    def success(cls, result: Any, request_id: Optional[Union[str, int]] = None) -> "MCPResponse":
        """Create success response"""
        return cls(result=result, id=request_id)
    
    @classmethod
    def error(cls, error: MCPError, request_id: Optional[Union[str, int]] = None) -> "MCPResponse":
        """Create error response"""
        return cls(error=error, id=request_id)

@dataclass
class MCPNotification:
    """MCP Notification message"""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Optional[Union[Dict[str, Any], List[Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"jsonrpc": self.jsonrpc, "method": self.method}
        if self.params is not None:
            result["params"] = self.params
        return result

@dataclass
class MCPCapabilities:
    """MCP Server capabilities"""
    tools: bool = True
    resources: bool = True
    prompts: bool = True
    logging: bool = True
    experimental: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "tools": {"listChanged": self.tools} if self.tools else {},
            "resources": {"subscribe": self.resources, "listChanged": self.resources} if self.resources else {},
            "prompts": {"listChanged": self.prompts} if self.prompts else {},
            "logging": {} if self.logging else {},
            "experimental": self.experimental
        }

@dataclass
class MCPServerInfo:
    """MCP Server information"""
    name: str
    version: str
    protocolVersion: str = MCPVersion.LATEST.value
    capabilities: MCPCapabilities = field(default_factory=MCPCapabilities)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "name": self.name,
            "version": self.version,
            "protocolVersion": self.protocolVersion,
            "capabilities": self.capabilities.to_dict()
        }

@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    inputSchema: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

@dataclass
class MCPResource:
    """MCP Resource definition"""
    uri: str
    name: str
    description: Optional[str] = None
    mimeType: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"uri": self.uri, "name": self.name}
        if self.description:
            result["description"] = self.description
        if self.mimeType:
            result["mimeType"] = self.mimeType
        return result

@dataclass
class MCPPrompt:
    """MCP Prompt definition"""
    name: str
    description: Optional[str] = None
    arguments: Optional[List[Dict[str, Any]]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        result = {"name": self.name}
        if self.description:
            result["description"] = self.description
        if self.arguments:
            result["arguments"] = self.arguments
        return result

class MCPProtocolHandler:
    """
    Production-grade MCP protocol handler
    Implements full JSON-RPC 2.0 with MCP extensions
    """
    
    def __init__(self, server_info: MCPServerInfo):
        self.server_info = server_info
        self.method_handlers: Dict[str, Callable] = {}
        self.notification_handlers: Dict[str, Callable] = {}
        self._setup_core_handlers()
        
    def _setup_core_handlers(self):
        """Setup core MCP method handlers"""
        self.register_method("initialize", self._handle_initialize)
        self.register_method("ping", self._handle_ping)
        self.register_method("tools/list", self._handle_tools_list)
        self.register_method("resources/list", self._handle_resources_list)
        self.register_method("prompts/list", self._handle_prompts_list)
        
    async def _handle_initialize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle initialize request"""
        client_info = params.get("clientInfo", {})
        logger.info(f"Client connected: {client_info.get('name', 'unknown')} v{client_info.get('version', 'unknown')}")
        
        return {
            "protocolVersion": self.server_info.protocolVersion,
            "serverInfo": self.server_info.to_dict()["serverInfo"] if "serverInfo" in self.server_info.to_dict() else {
                "name": self.server_info.name,
                "version": self.server_info.version
            },
            "capabilities": self.server_info.capabilities.to_dict()
        }
    
    async def _handle_ping(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle ping request"""
        return {"pong": True, "timestamp": datetime.now(timezone.utc).isoformat()}
    
    async def _handle_tools_list(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle tools/list request - to be overridden by implementations"""
        return {"tools": []}
    
    async def _handle_resources_list(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle resources/list request - to be overridden by implementations"""
        return {"resources": []}
    
    async def _handle_prompts_list(self, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Handle prompts/list request - to be overridden by implementations"""
        return {"prompts": []}
    
    def register_method(self, method: str, handler: Callable):
        """Register a method handler"""
        self.method_handlers[method] = handler
        
    def register_notification(self, method: str, handler: Callable):
        """Register a notification handler"""
        self.notification_handlers[method] = handler
    
    async def handle_message(self, message: Union[str, bytes, Dict[str, Any]]) -> Optional[Union[Dict[str, Any], List[Dict[str, Any]]]]:
        """
        Handle incoming MCP message
        Returns response(s) or None for notifications
        """
        try:
            # Parse message
            if isinstance(message, (str, bytes)):
                try:
                    data = json.loads(message)
                except json.JSONDecodeError as e:
                    return MCPResponse.error(
                        MCPError(ErrorCode.PARSE_ERROR.value, "Parse error", str(e))
                    ).to_dict()
            else:
                data = message
            
            # Handle batch requests
            if isinstance(data, list):
                if not data:
                    return MCPResponse.error(
                        MCPError(ErrorCode.INVALID_REQUEST.value, "Invalid Request", "Empty batch")
                    ).to_dict()
                    
                responses = []
                for item in data:
                    response = await self._handle_single_message(item)
                    if response is not None:
                        responses.append(response)
                        
                return responses if responses else None
            
            # Handle single request
            return await self._handle_single_message(data)
            
        except Exception as e:
            logger.error(f"Error handling message: {e}", exc_info=True)
            return MCPResponse.error(
                MCPError(ErrorCode.INTERNAL_ERROR.value, "Internal error", str(e))
            ).to_dict()
    
    async def _handle_single_message(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Handle a single MCP message"""
        try:
            # Validate JSON-RPC version
            if data.get("jsonrpc") != "2.0":
                return MCPResponse.error(
                    MCPError(ErrorCode.INVALID_REQUEST.value, "Invalid Request", "JSON-RPC 2.0 required"),
                    data.get("id")
                ).to_dict()
            
            # Check if it's a request or notification
            method = data.get("method")
            if not method:
                return MCPResponse.error(
                    MCPError(ErrorCode.INVALID_REQUEST.value, "Invalid Request", "Method required"),
                    data.get("id")
                ).to_dict()
            
            request_id = data.get("id")
            params = data.get("params")
            
            # Notification (no id)
            if request_id is None:
                handler = self.notification_handlers.get(method)
                if handler:
                    try:
                        await handler(params)
                    except Exception as e:
                        logger.error(f"Error handling notification {method}: {e}", exc_info=True)
                return None
            
            # Request (has id)
            handler = self.method_handlers.get(method)
            if not handler:
                return MCPResponse.error(
                    MCPError(ErrorCode.METHOD_NOT_FOUND.value, f"Method not found: {method}"),
                    request_id
                ).to_dict()
            
            # Execute handler
            try:
                result = await handler(params)
                return MCPResponse.success(result, request_id).to_dict()
            except Exception as e:
                logger.error(f"Error in method handler {method}: {e}", exc_info=True)
                return MCPResponse.error(
                    MCPError(ErrorCode.INTERNAL_ERROR.value, "Internal error", str(e)),
                    request_id
                ).to_dict()
                
        except Exception as e:
            logger.error(f"Error handling single message: {e}", exc_info=True)
            return MCPResponse.error(
                MCPError(ErrorCode.INTERNAL_ERROR.value, "Internal error", str(e)),
                data.get("id")
            ).to_dict()

class MCPBatchProcessor:
    """Process MCP requests in batches for efficiency"""
    
    def __init__(self, handler: MCPProtocolHandler, max_batch_size: int = 100):
        self.handler = handler
        self.max_batch_size = max_batch_size
        self.pending_requests: List[Tuple[Dict[str, Any], asyncio.Future]] = []
        self.processing = False
        
    async def add_request(self, request: Dict[str, Any]) -> Any:
        """Add request to batch and get result"""
        future = asyncio.Future()
        self.pending_requests.append((request, future))
        
        # Start processing if not already running
        if not self.processing:
            asyncio.create_task(self._process_batch())
            
        return await future
    
    async def _process_batch(self):
        """Process pending requests in batch"""
        if self.processing:
            return
            
        self.processing = True
        
        try:
            while self.pending_requests:
                # Get batch of requests
                batch = []
                futures = []
                
                for _ in range(min(self.max_batch_size, len(self.pending_requests))):
                    if not self.pending_requests:
                        break
                    request, future = self.pending_requests.pop(0)
                    batch.append(request)
                    futures.append(future)
                
                if not batch:
                    break
                
                # Process batch
                try:
                    if len(batch) == 1:
                        # Single request
                        response = await self.handler.handle_message(batch[0])
                        futures[0].set_result(response)
                    else:
                        # Batch request
                        responses = await self.handler.handle_message(batch)
                        if isinstance(responses, list):
                            for i, response in enumerate(responses):
                                if i < len(futures):
                                    futures[i].set_result(response)
                        else:
                            # Unexpected response format
                            for future in futures:
                                future.set_exception(Exception("Invalid batch response"))
                                
                except Exception as e:
                    # Set exception for all futures in batch
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)
                            
        finally:
            self.processing = False

class MCPRateLimiter:
    """Rate limiter for MCP operations"""
    
    def __init__(self, requests_per_second: float = 100):
        self.requests_per_second = requests_per_second
        self.min_interval = 1.0 / requests_per_second
        self.last_request_time: Dict[str, float] = {}
        self.request_counts: Dict[str, int] = {}
        
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if client has exceeded rate limit"""
        current_time = asyncio.get_event_loop().time()
        last_time = self.last_request_time.get(client_id, 0)
        
        if current_time - last_time < self.min_interval:
            return False
            
        self.last_request_time[client_id] = current_time
        
        # Track request count for monitoring
        self.request_counts[client_id] = self.request_counts.get(client_id, 0) + 1
        
        return True
    
    async def wait_if_needed(self, client_id: str):
        """Wait if rate limit would be exceeded"""
        current_time = asyncio.get_event_loop().time()
        last_time = self.last_request_time.get(client_id, 0)
        
        wait_time = self.min_interval - (current_time - last_time)
        if wait_time > 0:
            await asyncio.sleep(wait_time)
            
        self.last_request_time[client_id] = asyncio.get_event_loop().time()
        self.request_counts[client_id] = self.request_counts.get(client_id, 0) + 1

# Export main components
__all__ = [
    "MCPVersion",
    "ErrorCode",
    "MCPError",
    "MCPRequest",
    "MCPResponse",
    "MCPNotification",
    "MCPCapabilities",
    "MCPServerInfo",
    "MCPTool",
    "MCPResource",
    "MCPPrompt",
    "MCPProtocolHandler",
    "MCPBatchProcessor",
    "MCPRateLimiter"
]