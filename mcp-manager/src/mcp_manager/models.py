"""
Data models for MCP Management System

Comprehensive data structures for server configuration, status tracking,
and health monitoring with full validation and type safety.
"""

import json
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, validator, root_validator


class ServerType(str, Enum):
    """MCP Server implementation types"""
    PYTHON = "python"
    NODE = "node"
    GO = "go"
    RUST = "rust"
    SHELL = "shell"
    HTTP = "http"
    UNKNOWN = "unknown"


class ConnectionType(str, Enum):
    """MCP Connection types"""
    STDIO = "stdio"
    HTTP = "http"
    WEBSOCKET = "websocket"
    TCP = "tcp"
    UNIX_SOCKET = "unix_socket"


class ServerStatus(str, Enum):
    """MCP Server status states"""
    UNKNOWN = "unknown"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    STOPPED = "stopped"
    ERROR = "error"
    CRASHED = "crashed"
    UNREACHABLE = "unreachable"


class HealthStatus(str, Enum):
    """Health check status"""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    CRITICAL = "critical"


class MCPCapability(BaseModel):
    """Individual MCP capability/tool definition"""
    name: str = Field(..., description="Tool name")
    description: str = Field("", description="Tool description")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Parameter schema")
    required: List[str] = Field(default_factory=list, description="Required parameters")
    
    class Config:
        extra = "allow"


class ServerConfig(BaseModel):
    """Comprehensive MCP server configuration"""
    
    # Identity
    name: str = Field(..., description="Unique server name")
    description: str = Field("", description="Server description")
    version: str = Field("1.0.0", description="Server version")
    
    # Connection details
    command: str = Field(..., description="Command to start server")
    args: List[str] = Field(default_factory=list, description="Command arguments")
    connection_type: ConnectionType = Field(ConnectionType.STDIO, description="Connection method")
    
    # Server type and runtime
    server_type: ServerType = Field(ServerType.UNKNOWN, description="Implementation type")
    working_directory: Optional[Path] = Field(None, description="Working directory")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    # Networking (for HTTP/WebSocket servers)
    host: str = Field("localhost", description="Server host")
    port: Optional[int] = Field(None, description="Server port")
    base_url: Optional[str] = Field(None, description="Base URL for HTTP servers")
    
    # Timeouts and limits
    startup_timeout: float = Field(30.0, description="Startup timeout seconds", gt=0)
    request_timeout: float = Field(60.0, description="Request timeout seconds", gt=0)
    max_retries: int = Field(3, description="Maximum retry attempts", ge=0)
    retry_delay: float = Field(1.0, description="Retry delay seconds", ge=0)
    
    # Health monitoring
    health_check_interval: float = Field(30.0, description="Health check interval seconds", gt=0)
    health_check_timeout: float = Field(10.0, description="Health check timeout seconds", gt=0)
    failure_threshold: int = Field(3, description="Failures before marking unhealthy", gt=0)
    recovery_threshold: int = Field(2, description="Successes needed for recovery", gt=0)
    
    # Auto-restart and recovery
    auto_restart: bool = Field(True, description="Auto-restart on failure")
    restart_delay: float = Field(5.0, description="Delay before restart seconds", ge=0)
    max_restart_attempts: int = Field(5, description="Max restart attempts", ge=0)
    restart_window: float = Field(300.0, description="Restart window seconds", gt=0)
    
    # Security and validation
    requires_auth: bool = Field(False, description="Requires authentication")
    auth_token: Optional[str] = Field(None, description="Authentication token")
    validate_ssl: bool = Field(True, description="Validate SSL certificates")
    allowed_origins: List[str] = Field(default_factory=list, description="Allowed CORS origins")
    
    # Performance and resource limits
    max_concurrent_requests: int = Field(10, description="Max concurrent requests", gt=0)
    request_rate_limit: Optional[float] = Field(None, description="Requests per second limit")
    memory_limit_mb: Optional[int] = Field(None, description="Memory limit in MB")
    cpu_limit_percent: Optional[float] = Field(None, description="CPU limit percentage")
    
    # Metadata
    tags: List[str] = Field(default_factory=list, description="Server tags")
    priority: int = Field(100, description="Server priority (lower = higher priority)")
    enabled: bool = Field(True, description="Server enabled state")
    
    @validator("port")
    def validate_port(cls, v: Optional[int]) -> Optional[int]:
        if v is not None and (v < 1 or v > 65535):
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator("working_directory", pre=True)
    def convert_working_directory(cls, v: Union[str, Path, None]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v)
    
    @root_validator
    def validate_connection_config(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        connection_type = values.get("connection_type")
        port = values.get("port")
        base_url = values.get("base_url")
        
        if connection_type in [ConnectionType.HTTP, ConnectionType.WEBSOCKET]:
            if not port and not base_url:
                raise ValueError(f"{connection_type} requires either port or base_url")
        
        return values


class ServerMetrics(BaseModel):
    """Server performance and resource metrics"""
    
    # Request metrics
    total_requests: int = Field(0, description="Total requests processed")
    successful_requests: int = Field(0, description="Successful requests")
    failed_requests: int = Field(0, description="Failed requests")
    avg_response_time: float = Field(0.0, description="Average response time seconds")
    
    # Resource usage
    cpu_percent: float = Field(0.0, description="CPU usage percentage")
    memory_mb: float = Field(0.0, description="Memory usage in MB")
    memory_percent: float = Field(0.0, description="Memory usage percentage")
    
    # Connection metrics
    active_connections: int = Field(0, description="Active connections")
    total_connections: int = Field(0, description="Total connections made")
    connection_errors: int = Field(0, description="Connection errors")
    
    # Health metrics
    uptime_seconds: float = Field(0.0, description="Uptime in seconds")
    restart_count: int = Field(0, description="Number of restarts")
    last_restart: Optional[datetime] = Field(None, description="Last restart time")
    
    # Error tracking
    error_rate: float = Field(0.0, description="Error rate percentage")
    last_error: Optional[str] = Field(None, description="Last error message")
    last_error_time: Optional[datetime] = Field(None, description="Last error timestamp")


class HealthCheckResult(BaseModel):
    """Result of a health check operation"""
    
    status: HealthStatus = Field(..., description="Health status")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Check timestamp")
    response_time: float = Field(0.0, description="Response time in seconds")
    
    # Detailed status information
    is_running: bool = Field(False, description="Process is running")
    is_responsive: bool = Field(False, description="Server is responsive")
    capabilities_count: int = Field(0, description="Number of available capabilities")
    
    # Error information
    error_message: Optional[str] = Field(None, description="Error message if any")
    error_code: Optional[str] = Field(None, description="Error code")
    
    # Additional metadata
    server_version: Optional[str] = Field(None, description="Server version")
    server_info: Dict[str, Any] = Field(default_factory=dict, description="Additional server info")


class ServerState(BaseModel):
    """Complete server state information"""
    
    # Basic information
    config: ServerConfig = Field(..., description="Server configuration")
    status: ServerStatus = Field(ServerStatus.UNKNOWN, description="Current status")
    
    # Health and monitoring
    health: HealthCheckResult = Field(..., description="Latest health check")
    metrics: ServerMetrics = Field(default_factory=ServerMetrics, description="Performance metrics")
    
    # Capabilities
    capabilities: List[MCPCapability] = Field(default_factory=list, description="Available tools")
    last_capability_update: Optional[datetime] = Field(None, description="Last capability refresh")
    
    # Runtime information
    process_id: Optional[int] = Field(None, description="Process ID")
    start_time: Optional[datetime] = Field(None, description="Server start time")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    
    # Connection information
    connection_established: Optional[datetime] = Field(None, description="Connection established time")
    connection_failures: int = Field(0, description="Connection failure count")
    last_connection_error: Optional[str] = Field(None, description="Last connection error")
    
    @property
    def is_healthy(self) -> bool:
        """Check if server is in a healthy state"""
        return (
            self.status == ServerStatus.RUNNING and
            self.health.status in [HealthStatus.HEALTHY, HealthStatus.DEGRADED]
        )
    
    @property
    def uptime(self) -> timedelta:
        """Calculate server uptime"""
        if self.start_time:
            return datetime.utcnow() - self.start_time
        return timedelta(0)
    
    @property
    def needs_restart(self) -> bool:
        """Determine if server needs restart"""
        return (
            self.status in [ServerStatus.ERROR, ServerStatus.CRASHED] or
            self.health.status == HealthStatus.CRITICAL or
            self.connection_failures >= self.config.failure_threshold
        )


class MCPManagerConfig(BaseModel):
    """Configuration for the MCP Manager itself"""
    
    # Discovery settings
    config_directories: List[Path] = Field(
        default_factory=lambda: [Path("/opt/sutazaiapp/.mcp")],
        description="Directories to scan for server configs"
    )
    auto_discovery: bool = Field(True, description="Enable automatic server discovery")
    discovery_interval: float = Field(60.0, description="Discovery interval seconds", gt=0)
    
    # Global monitoring settings
    global_health_check_interval: float = Field(30.0, description="Global health check interval", gt=0)
    metrics_collection_interval: float = Field(15.0, description="Metrics collection interval", gt=0)
    
    # Recovery settings
    global_auto_recovery: bool = Field(True, description="Enable global auto-recovery")
    recovery_check_interval: float = Field(10.0, description="Recovery check interval", gt=0)
    
    # Performance limits
    max_concurrent_health_checks: int = Field(5, description="Max concurrent health checks", gt=0)
    max_startup_concurrency: int = Field(3, description="Max concurrent startups", gt=0)
    
    # Storage and logging
    state_file: Path = Field(Path("/tmp/mcp_manager_state.json"), description="State persistence file")
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field("INFO", description="Log level")
    log_file: Optional[Path] = Field(None, description="Log file path")
    
    # API settings
    api_enabled: bool = Field(True, description="Enable management API")
    api_host: str = Field("localhost", description="API host")
    api_port: int = Field(8080, description="API port")
    
    @validator("config_directories", pre=True)
    def convert_paths(cls, v: List[Union[str, Path]]) -> List[Path]:
        return [Path(p) if isinstance(p, str) else p for p in v]
    
    @validator("state_file", "log_file", pre=True)
    def convert_path(cls, v: Union[str, Path, None]) -> Optional[Path]:
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v


class ConnectionState(BaseModel):
    """State of a connection to an MCP server"""
    
    server_name: str = Field(..., description="Server name")
    connection_type: ConnectionType = Field(..., description="Connection type")
    
    # Connection status
    is_connected: bool = Field(False, description="Connection established")
    connection_time: Optional[datetime] = Field(None, description="Connection established timestamp")
    last_activity: Optional[datetime] = Field(None, description="Last activity timestamp")
    
    # Connection details
    transport_info: Dict[str, Any] = Field(default_factory=dict, description="Transport-specific info")
    
    # Performance tracking
    requests_sent: int = Field(0, description="Total requests sent")
    responses_received: int = Field(0, description="Total responses received")
    avg_latency: float = Field(0.0, description="Average request latency seconds")
    
    # Error tracking
    connection_errors: int = Field(0, description="Connection error count")
    last_error: Optional[str] = Field(None, description="Last connection error")
    last_error_time: Optional[datetime] = Field(None, description="Last error timestamp")


# Type aliases for convenience
ServerConfigDict = Dict[str, Any]
ServerStateDict = Dict[str, ServerState]
ConnectionStateDict = Dict[str, ConnectionState]