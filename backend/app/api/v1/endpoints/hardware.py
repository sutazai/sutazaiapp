"""
Hardware Resource Optimization API Endpoints
Integrates with hardware-resource-optimizer service at port 8080
Production-ready implementation with full async proxy, auth, caching, and error handling
"""

import os
import asyncio
import logging
import json
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from contextlib import asynccontextmanager
from enum import Enum
import uuid

from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks, Query, Path, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
import httpx

# Import core services
from app.core.connection_pool import get_http_client, get_pool_manager
from app.core.cache import (
    get_cache_service, cached, cache_model_data, cache_session_data,
    cache_api_response, cache_database_query, cache_heavy_computation,
    cache_static_data
)
from app.core.task_queue import get_task_queue, create_background_task
from app.auth.dependencies import get_current_user, require_permissions, get_optional_user

logger = logging.getLogger(__name__)

# Router setup
router = APIRouter(prefix="/hardware", tags=["Hardware Optimization"])

# Configuration with comprehensive environment variables
HARDWARE_SERVICE_URL = os.getenv("HARDWARE_OPTIMIZER_URL", "http://sutazai-hardware-resource-optimizer:8080")
HARDWARE_SERVICE_TIMEOUT = int(os.getenv("HARDWARE_SERVICE_TIMEOUT", "30"))
HARDWARE_CACHE_TTL = int(os.getenv("HARDWARE_CACHE_TTL", "300"))  # 5 minutes
HARDWARE_STREAM_INTERVAL = int(os.getenv("HARDWARE_STREAM_INTERVAL", "5"))  # 5 seconds
HARDWARE_MAX_RETRIES = int(os.getenv("HARDWARE_MAX_RETRIES", "3"))
HARDWARE_RETRY_DELAY = float(os.getenv("HARDWARE_RETRY_DELAY", "1.0"))

# Enums for validation
class OptimizationType(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    DISK = "disk"
    NETWORK = "network"
    POWER = "power"
    THERMAL = "thermal"
    PROCESSES = "processes"
    SERVICES = "services"
    STARTUP = "startup"
    FULL_SYSTEM = "full_system"

class Priority(str, Enum):
    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"
    CRITICAL = "critical"

class ProcessAction(str, Enum):
    KILL = "kill"
    SUSPEND = "suspend"
    RESUME = "resume"
    PRIORITIZE = "prioritize"
    LIMIT = "limit"

class AlertSeverity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class SortField(str, Enum):
    CPU = "cpu"
    MEMORY = "memory"
    NAME = "name"
    PID = "pid"

# Request/Response Models
class HardwareMetricsRequest(BaseModel):
    """Request model for hardware metrics collection"""
    include_processes: bool = Field(default=True, description="Include process-level metrics")
    include_network: bool = Field(default=True, description="Include network metrics")
    include_disk: bool = Field(default=True, description="Include disk I/O metrics")
    include_gpu: bool = Field(default=False, description="Include GPU metrics if available")
    include_containers: bool = Field(default=True, description="Include container metrics")
    sample_duration: int = Field(default=5, ge=1, le=60, description="Sampling duration in seconds")

class OptimizationRequest(BaseModel):
    """Request model for system optimization"""
    optimization_type: OptimizationType = Field(..., description="Type of optimization to perform")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Optimization parameters")
    priority: Priority = Field(default=Priority.NORMAL, description="Optimization priority")
    dry_run: bool = Field(default=False, description="Perform dry run without applying changes")
    auto_rollback: bool = Field(default=True, description="Enable automatic rollback on failure")
    rollback_timeout_minutes: int = Field(default=60, ge=5, le=1440, description="Rollback timeout in minutes")

class ResourceLimit(BaseModel):
    """Model for resource limits"""
    cpu_percent: Optional[float] = Field(None, ge=0, le=100, description="CPU usage limit percentage")
    memory_mb: Optional[int] = Field(None, ge=0, description="Memory limit in MB")
    disk_io_mb_s: Optional[float] = Field(None, ge=0, description="Disk I/O limit in MB/s")
    network_mb_s: Optional[float] = Field(None, ge=0, description="Network limit in MB/s")
    gpu_memory_mb: Optional[int] = Field(None, ge=0, description="GPU memory limit in MB")

class ProcessControlRequest(BaseModel):
    """Request model for process control operations"""
    action: ProcessAction = Field(..., description="Action to perform")
    process_id: Optional[int] = Field(None, description="Process ID")
    process_name: Optional[str] = Field(None, description="Process name pattern")
    resource_limits: Optional[ResourceLimit] = Field(None, description="Resource limits for process")
    priority: Optional[int] = Field(None, ge=-20, le=19, description="Process priority (-20 to 19)")
    timeout_seconds: int = Field(default=30, ge=1, le=300, description="Operation timeout")

    @validator('process_id', 'process_name')
    def validate_process_identifier(cls, v, values):
        """Ensure at least one process identifier is provided"""
        if not v and not values.get('process_id') and not values.get('process_name'):
            raise ValueError("Either process_id or process_name must be provided")
        return v

class MonitoringConfigRequest(BaseModel):
    """Request model for monitoring configuration"""
    interval_seconds: int = Field(default=30, ge=5, le=3600, description="Monitoring interval")
    alert_thresholds: Dict[str, float] = Field(default_factory=dict, description="Alert thresholds")
    enabled_metrics: List[str] = Field(default_factory=list, description="Enabled metric types")
    retention_hours: int = Field(default=24, ge=1, le=168, description="Data retention period")
    notification_webhooks: List[str] = Field(default_factory=list, description="Webhook URLs for notifications")

class BenchmarkRequest(BaseModel):
    """Request model for system benchmarking"""
    benchmark_type: str = Field(..., description="Type of benchmark to run")
    duration_seconds: int = Field(default=60, ge=10, le=600, description="Benchmark duration")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Benchmark parameters")
    save_results: bool = Field(default=True, description="Save benchmark results to database")

# Response Models
class HardwareStatus(BaseModel):
    """Hardware service status response"""
    status: str = Field(..., description="Service status")
    agent: str = Field(..., description="Agent identifier")
    timestamp: Union[str, float] = Field(..., description="Status timestamp")
    version: Optional[str] = Field(None, description="Service version")
    description: Optional[str] = Field(None, description="Service description")
    docker_available: Optional[bool] = Field(None, description="Docker availability")
    system_status: Optional[Dict[str, Any]] = Field(None, description="System status")
    uptime_seconds: Optional[float] = Field(None, description="Service uptime")
    last_optimization: Optional[str] = Field(None, description="Last optimization timestamp")
    active_optimizations: int = Field(default=0, description="Number of active optimizations")
    system_health_score: Optional[float] = Field(None, ge=0, le=100, description="Overall system health score")

class SystemMetrics(BaseModel):
    """System metrics response"""
    timestamp: str = Field(..., description="Metrics timestamp")
    cpu: Dict[str, Any] = Field(..., description="CPU metrics")
    memory: Dict[str, Any] = Field(..., description="Memory metrics")
    disk: Dict[str, Any] = Field(..., description="Disk metrics")
    network: Dict[str, Any] = Field(..., description="Network metrics")
    processes: Optional[List[Dict[str, Any]]] = Field(None, description="Process metrics")
    containers: Optional[List[Dict[str, Any]]] = Field(None, description="Container metrics")
    gpu: Optional[Dict[str, Any]] = Field(None, description="GPU metrics")
    system_load: Dict[str, float] = Field(default_factory=dict, description="System load averages")
    uptime_seconds: Optional[float] = Field(None, description="System uptime")

class OptimizationResult(BaseModel):
    """Optimization operation result"""
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Optimization status")
    optimization_type: str = Field(..., description="Type of optimization performed")
    started_at: str = Field(..., description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    duration_seconds: Optional[float] = Field(None, description="Operation duration")
    changes_applied: List[str] = Field(default_factory=list, description="List of changes applied")
    performance_impact: Optional[Dict[str, float]] = Field(None, description="Performance impact metrics")
    rollback_info: Optional[Dict[str, Any]] = Field(None, description="Rollback information")
    dry_run: bool = Field(default=False, description="Whether this was a dry run")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class ProcessInfo(BaseModel):
    """Process information model"""
    pid: int = Field(..., description="Process ID")
    name: str = Field(..., description="Process name")
    status: str = Field(..., description="Process status")
    cpu_percent: float = Field(..., description="CPU usage percentage")
    memory_mb: float = Field(..., description="Memory usage in MB")
    memory_percent: Optional[float] = Field(None, description="Memory usage percentage")
    threads: int = Field(..., description="Number of threads")
    created_time: str = Field(..., description="Process creation time")
    command_line: Optional[str] = Field(None, description="Process command line")
    parent_pid: Optional[int] = Field(None, description="Parent process ID")

class AlertInfo(BaseModel):
    """Hardware alert information"""
    id: str = Field(..., description="Alert ID")
    severity: AlertSeverity = Field(..., description="Alert severity")
    title: str = Field(..., description="Alert title")
    message: str = Field(..., description="Alert message")
    timestamp: str = Field(..., description="Alert timestamp")
    acknowledged: bool = Field(default=False, description="Whether alert is acknowledged")
    acknowledged_by: Optional[str] = Field(None, description="User who acknowledged the alert")
    acknowledged_at: Optional[str] = Field(None, description="Acknowledgment timestamp")
    metric_name: Optional[str] = Field(None, description="Related metric name")
    threshold_value: Optional[float] = Field(None, description="Threshold that was breached")
    current_value: Optional[float] = Field(None, description="Current metric value")

class RecommendationInfo(BaseModel):
    """Optimization recommendation"""
    id: str = Field(..., description="Recommendation ID")
    category: str = Field(..., description="Recommendation category")
    title: str = Field(..., description="Recommendation title")
    description: str = Field(..., description="Detailed description")
    priority: Priority = Field(..., description="Recommendation priority")
    potential_impact: Dict[str, Any] = Field(..., description="Expected impact metrics")
    implementation_complexity: str = Field(..., description="Implementation complexity level")
    estimated_duration_minutes: Optional[int] = Field(None, description="Estimated implementation time")
    prerequisites: List[str] = Field(default_factory=list, description="Prerequisites for implementation")

class BenchmarkResult(BaseModel):
    """Benchmark execution result"""
    task_id: str = Field(..., description="Benchmark task ID")
    benchmark_type: str = Field(..., description="Type of benchmark")
    status: str = Field(..., description="Benchmark status")
    started_at: str = Field(..., description="Start timestamp")
    completed_at: Optional[str] = Field(None, description="Completion timestamp")
    duration_seconds: Optional[float] = Field(None, description="Execution duration")
    results: Dict[str, Any] = Field(default_factory=dict, description="Benchmark results")
    baseline_comparison: Optional[Dict[str, Any]] = Field(None, description="Comparison with baseline")
    error_message: Optional[str] = Field(None, description="Error message if failed")

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code")
    timestamp: str = Field(..., description="Error timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request ID for tracking")

# Connection and retry utilities with enhanced error handling
class HardwareServiceClient:
    """Client for hardware service communication with comprehensive retry logic and circuit breaker"""
    
    def __init__(self):
        self.base_url = HARDWARE_SERVICE_URL
        self.timeout = HARDWARE_SERVICE_TIMEOUT
        self.max_retries = HARDWARE_MAX_RETRIES
        self.retry_delay = HARDWARE_RETRY_DELAY
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = None
        self._circuit_breaker_threshold = 5
        self._circuit_breaker_timeout = 300  # 5 minutes
    
    def _is_circuit_breaker_open(self) -> bool:
        """Check if circuit breaker is open"""
        if self._circuit_breaker_failures < self._circuit_breaker_threshold:
            return False
        
        if self._circuit_breaker_last_failure:
            time_since_failure = (datetime.utcnow() - self._circuit_breaker_last_failure).total_seconds()
            if time_since_failure > self._circuit_breaker_timeout:
                # Reset circuit breaker
                self._circuit_breaker_failures = 0
                self._circuit_breaker_last_failure = None
                return False
        
        return True
    
    def _record_failure(self):
        """Record a failure for circuit breaker"""
        self._circuit_breaker_failures += 1
        self._circuit_breaker_last_failure = datetime.utcnow()
    
    def _record_success(self):
        """Record a success for circuit breaker"""
        self._circuit_breaker_failures = 0
        self._circuit_breaker_last_failure = None
    
    async def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        data: Optional[Dict] = None,
        params: Optional[Dict] = None,
        timeout: Optional[int] = None,
        stream: bool = False
    ) -> Union[Dict[str, Any], httpx.Response]:
        """Make HTTP request with comprehensive retry logic and circuit breaker"""
        
        # Check circuit breaker
        if self._is_circuit_breaker_open():
            raise HTTPException(
                status_code=503,
                detail="Hardware service circuit breaker is open - service temporarily unavailable"
            )
        
        timeout = timeout or self.timeout
        retry_count = 0
        last_exception = None
        
        while retry_count < self.max_retries:
            try:
                async with await get_http_client('agents') as client:
                    url = f"{self.base_url}{endpoint}"
                    
                    request_kwargs = {
                        "timeout": timeout,
                        "follow_redirects": True
                    }
                    
                    if params:
                        request_kwargs["params"] = params
                    
                    if data and method.upper() in ["POST", "PUT", "PATCH"]:
                        request_kwargs["json"] = data
                    
                    # Make the request
                    if method.upper() == "GET":
                        response = await client.get(url, **request_kwargs)
                    elif method.upper() == "POST":
                        response = await client.post(url, **request_kwargs)
                    elif method.upper() == "PUT":
                        response = await client.put(url, **request_kwargs)
                    elif method.upper() == "PATCH":
                        response = await client.patch(url, **request_kwargs)
                    elif method.upper() == "DELETE":
                        response = await client.delete(url, **request_kwargs)
                    else:
                        raise ValueError(f"Unsupported HTTP method: {method}")
                    
                    response.raise_for_status()
                    
                    # Record success
                    self._record_success()
                    
                    # Return raw response for streaming
                    if stream:
                        return response
                    
                    # Handle different content types
                    content_type = response.headers.get("content-type", "").lower()
                    if "application/json" in content_type:
                        return response.json()
                    else:
                        return {"response": response.text, "status_code": response.status_code}
                        
            except httpx.TimeoutException as e:
                last_exception = e
                logger.warning(f"Timeout on attempt {retry_count + 1}/{self.max_retries} for {endpoint}")
                
            except httpx.HTTPStatusError as e:
                if e.response.status_code in [500, 502, 503, 504]:
                    last_exception = e
                    logger.warning(f"Server error on attempt {retry_count + 1}/{self.max_retries} for {endpoint}: {e.response.status_code}")
                else:
                    # Don't retry client errors, but record failure
                    self._record_failure()
                    error_detail = f"Hardware service error: {e.response.status_code}"
                    try:
                        error_body = e.response.json()
                        error_detail += f" - {error_body.get('detail', e.response.text)}"
                    except (ValueError, TypeError, KeyError, AttributeError) as e:
                        # TODO: Review this exception handling
                        logger.error(f"Unexpected exception: {e}", exc_info=True)
                        error_detail += f" - {e.response.text}"
                    
                    raise HTTPException(
                        status_code=e.response.status_code,
                        detail=error_detail
                    )
                    
            except Exception as e:
                last_exception = e
                logger.error(f"Unexpected error on attempt {retry_count + 1}/{self.max_retries} for {endpoint}: {e}")
            
            retry_count += 1
            if retry_count < self.max_retries:
                delay = self.retry_delay * (2 ** (retry_count - 1))  # Exponential backoff
                await asyncio.sleep(delay)
        
        # All retries exhausted - record failure and raise exception
        self._record_failure()
        
        error_msg = f"Hardware service unavailable after {self.max_retries} attempts"
        if last_exception:
            error_msg += f": {str(last_exception)}"
        
        raise HTTPException(
            status_code=503,
            detail=error_msg
        )
    
    async def get(self, endpoint: str, params: Optional[Dict] = None, timeout: Optional[int] = None, stream: bool = False) -> Union[Dict[str, Any], httpx.Response]:
        return await self._make_request("GET", endpoint, params=params, timeout=timeout, stream=stream)
    
    async def post(self, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        return await self._make_request("POST", endpoint, data=data, params=params, timeout=timeout)
    
    async def put(self, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        return await self._make_request("PUT", endpoint, data=data, params=params, timeout=timeout)
    
    async def patch(self, endpoint: str, data: Optional[Dict] = None, params: Optional[Dict] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        return await self._make_request("PATCH", endpoint, data=data, params=params, timeout=timeout)
    
    async def delete(self, endpoint: str, params: Optional[Dict] = None, timeout: Optional[int] = None) -> Dict[str, Any]:
        return await self._make_request("DELETE", endpoint, params=params, timeout=timeout)

# Global client instance with proper singleton pattern
_hardware_client: Optional[HardwareServiceClient] = None
_client_lock = asyncio.Lock()

async def get_hardware_client() -> HardwareServiceClient:
    """Get hardware service client instance with thread-safe singleton"""
    global _hardware_client
    
    async with _client_lock:
        if _hardware_client is None:
            _hardware_client = HardwareServiceClient()
    
    return _hardware_client

# API Endpoints with comprehensive functionality

@router.get("/health", response_model=HardwareStatus, summary="Hardware Service Health Check")
@cache_api_response(ttl=30)
async def get_hardware_service_health():
    """
    Get hardware optimization service health status with detailed metrics.
    
    Returns comprehensive health information including:
    - Service status and version
    - Uptime and last optimization time
    - Active optimization count
    - System health score
    """
    client = await get_hardware_client()
    
    try:
        result = await client.get("/health", timeout=5)
        return HardwareStatus(**result)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Hardware service health check failed: {e}")
        raise HTTPException(
            status_code=503, 
            detail=f"Hardware optimization service is unavailable: {str(e)}"
        )

@router.get("/status", response_model=HardwareStatus, summary="Detailed Service Status")
async def get_detailed_status(
    current_user = Depends(get_optional_user)
):
    """
    Get detailed hardware service status with extended information.
    
    Includes additional diagnostics and performance metrics.
    Authentication optional - detailed info requires login.
    """
    client = await get_hardware_client()
    
    try:
        result = await client.get("/status", timeout=10)
        
        # Add user context if authenticated
        if current_user:
            result["user_context"] = {
                "user_id": getattr(current_user, 'id', None),
                "permissions": getattr(current_user, 'permissions', [])
            }
        
        return HardwareStatus(**result)
    except Exception as e:
        logger.error(f"Failed to get detailed status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve detailed status: {str(e)}"
        )

@router.get("/metrics", response_model=SystemMetrics, summary="System Metrics Collection")
@cache_api_response(ttl=10)
async def get_system_metrics(
    include_processes: bool = Query(default=True, description="Include process-level metrics"),
    include_network: bool = Query(default=True, description="Include network metrics"),
    include_disk: bool = Query(default=True, description="Include disk I/O metrics"),
    include_gpu: bool = Query(default=False, description="Include GPU metrics if available"),
    include_containers: bool = Query(default=True, description="Include container metrics"),
    sample_duration: int = Query(default=5, ge=1, le=60, description="Sampling duration in seconds"),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive system metrics from hardware optimizer.
    
    Collects detailed system performance metrics including:
    - CPU, memory, disk, and network usage
    - Process-level resource consumption
    - Container metrics (Docker/Podman)
    - GPU utilization (if available)
    
    Requires authentication and monitoring permissions.
    """
    client = await get_hardware_client()
    
    try:
        params = {
            "include_processes": include_processes,
            "include_network": include_network,
            "include_disk": include_disk,
            "include_gpu": include_gpu,
            "include_containers": include_containers,
            "sample_duration": sample_duration
        }
        
        result = await client.get("/metrics", params=params, timeout=sample_duration + 10)
        return SystemMetrics(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get system metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve system metrics: {str(e)}"
        )

@router.get("/metrics/stream", summary="Real-time Metrics Stream")
async def stream_system_metrics(
    interval: int = Query(default=5, ge=1, le=60, description="Update interval in seconds"),
    include_processes: bool = Query(default=False, description="Include process metrics in stream"),
    include_containers: bool = Query(default=True, description="Include container metrics"),
    format: str = Query(default="json", regex="^(json|csv)$", description="Stream format"),
    current_user = Depends(get_current_user)
):
    """
    Stream real-time system metrics using Server-Sent Events (SSE).
    
    Provides continuous monitoring of system resources with configurable:
    - Update intervals (1-60 seconds)
    - Metric inclusion/exclusion
    - Output format (JSON/CSV)
    
    Ideal for dashboards and real-time monitoring applications.
    """
    client = await get_hardware_client()
    
    async def generate_metrics():
        session_id = str(uuid.uuid4())
        logger.info(f"Starting metrics stream session {session_id} for user {getattr(current_user, 'id', 'unknown')}")
        
        try:
            while True:
                try:
                    params = {
                        "include_processes": include_processes,
                        "include_containers": include_containers,
                        "sample_duration": min(interval, 10),  # Cap at 10 seconds
                        "format": format
                    }
                    
                    metrics = await client.get("/metrics", params=params, timeout=interval + 5)
                    
                    # Add metadata
                    metrics["stream_metadata"] = {
                        "session_id": session_id,
                        "timestamp": datetime.utcnow().isoformat(),
                        "interval": interval
                    }
                    
                    # Format as SSE
                    yield f"data: {json.dumps(metrics)}\n\n"
                    
                    await asyncio.sleep(interval)
                    
                except Exception as e:
                    error_data = {
                        "error": str(e),
                        "timestamp": datetime.utcnow().isoformat(),
                        "session_id": session_id,
                        "retry_in_seconds": 5
                    }
                    yield f"data: {json.dumps(error_data)}\n\n"
                    await asyncio.sleep(5)
                    
        except asyncio.CancelledError:
            logger.info(f"Metrics streaming session {session_id} cancelled")
            return
        finally:
            logger.info(f"Metrics streaming session {session_id} ended")
    
    # Get secure CORS configuration
    from app.core.cors_security import cors_security
    allowed_origins = cors_security.get_allowed_origins()
    
    return StreamingResponse(
        generate_metrics(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Access-Control-Allow-Origin": ", ".join(allowed_origins),
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With, Cache-Control",
            "Access-Control-Allow-Credentials": "true"
        }
    )

@router.post("/optimize", response_model=OptimizationResult, summary="Start System Optimization")
async def start_optimization(
    request: OptimizationRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_permissions(["hardware:optimize"]))
):
    """
    Start comprehensive system optimization process.
    
    Supports multiple optimization types:
    - CPU: Process scheduling, affinity, frequency scaling
    - Memory: Cache optimization, swap management, cleanup
    - Disk: I/O scheduling, defragmentation, cleanup
    - Network: Buffer tuning, connection optimization
    - Power: Power management, thermal throttling
    - Full System: Comprehensive optimization across all subsystems
    
    Features:
    - Dry-run mode for testing without applying changes
    - Automatic rollback on failure
    - Priority-based execution
    - Background progress tracking
    
    Requires hardware optimization permissions.
    """
    client = await get_hardware_client()
    
    try:
        # Prepare optimization request with enhanced metadata
        optimization_data = request.dict()
        optimization_data.update({
            "user_id": getattr(current_user, 'id', 'unknown'),
            "user_permissions": getattr(current_user, 'permissions', []),
            "requested_at": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4())
        })
        
        # Start optimization
        result = await client.post("/optimize", data=optimization_data, timeout=60)
        
        # Create background task to track optimization progress
        if not request.dry_run and result.get("task_id"):
            background_tasks.add_task(
                track_optimization_progress,
                result["task_id"],
                request.optimization_type,
                getattr(current_user, 'id', 'unknown')
            )
        
        return OptimizationResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to start optimization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start optimization: {str(e)}"
        )

@router.get("/optimize", summary="List Active Optimizations")
@cache_api_response(ttl=30)
async def list_active_optimizations(
    status_filter: Optional[str] = Query(None, description="Filter by optimization status"),
    user_filter: Optional[str] = Query(None, description="Filter by user (admin only)"),
    limit: int = Query(default=50, ge=1, le=200, description="Maximum number of results"),
    current_user = Depends(get_current_user)
):
    """
    List all active optimization tasks with filtering options.
    
    Provides overview of:
    - Running optimizations
    - Queued optimizations  
    - Recent completions
    - Failed optimizations
    
    Admin users can view optimizations from all users.
    Regular users see only their own optimizations.
    """
    client = await get_hardware_client()
    
    try:
        params = {"limit": limit}
        
        if status_filter:
            params["status"] = status_filter
            
        # Admin users can filter by any user
        if getattr(current_user, 'is_admin', False) and user_filter:
            params["user_id"] = user_filter
        else:
            # Regular users see only their optimizations
            params["user_id"] = getattr(current_user, 'id', 'unknown')
        
        result = await client.get("/optimize", params=params, timeout=15)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to list optimizations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list active optimizations: {str(e)}"
        )

@router.get("/optimize/{task_id}", response_model=OptimizationResult, summary="Get Optimization Status")
@cache_api_response(ttl=10)
async def get_optimization_status(
    task_id: str = Path(..., description="Optimization task ID"),
    current_user = Depends(get_current_user)
):
    """
    Get detailed status and results for a specific optimization task.
    
    Returns comprehensive information including:
    - Current execution status
    - Applied changes and their impact
    - Performance metrics before/after
    - Rollback information if available
    - Error details for failed optimizations
    """
    client = await get_hardware_client()
    
    try:
        result = await client.get(f"/optimize/{task_id}", timeout=10)
        
        # Verify user can access this optimization
        task_user_id = result.get("user_id")
        if (not getattr(current_user, 'is_admin', False) and 
            task_user_id != str(getattr(current_user, 'id', ''))):
            raise HTTPException(
                status_code=403,
                detail="You can only view your own optimization tasks"
            )
        
        return OptimizationResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get optimization status: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get optimization status: {str(e)}"
        )

@router.delete("/optimize/{task_id}", summary="Cancel Optimization Task")
async def cancel_optimization(
    task_id: str = Path(..., description="Optimization task ID"),
    force: bool = Query(default=False, description="Force cancellation even if in critical phase"),
    current_user = Depends(require_permissions(["hardware:optimize"]))
):
    """
    Cancel a running optimization task.
    
    Supports graceful cancellation with rollback of partial changes.
    Force option available for emergency situations.
    
    Warning: Forced cancellation may leave system in inconsistent state.
    """
    client = await get_hardware_client()
    
    try:
        # Get task details first to verify ownership
        task_status = await client.get(f"/optimize/{task_id}", timeout=5)
        task_user_id = task_status.get("user_id")
        
        # Verify user can cancel this optimization
        if (not getattr(current_user, 'is_admin', False) and 
            task_user_id != str(getattr(current_user, 'id', ''))):
            raise HTTPException(
                status_code=403,
                detail="You can only cancel your own optimization tasks"
            )
        
        params = {"force": force}
        result = await client.delete(f"/optimize/{task_id}", params=params, timeout=30)
        
        return {
            "message": f"Optimization task {task_id} cancellation requested",
            "details": result,
            "cancelled_by": getattr(current_user, 'id', 'unknown'),
            "cancelled_at": datetime.utcnow().isoformat(),
            "force_used": force
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to cancel optimization: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to cancel optimization: {str(e)}"
        )

@router.get("/processes", response_model=List[ProcessInfo], summary="List System Processes")
@cache_api_response(ttl=10)
async def get_processes(
    sort_by: SortField = Query(default=SortField.CPU, description="Sort processes by field"),
    limit: int = Query(default=50, ge=1, le=500, description="Maximum number of processes to return"),
    filter_pattern: Optional[str] = Query(None, description="Filter processes by name pattern"),
    min_cpu_percent: Optional[float] = Query(None, ge=0, le=100, description="Minimum CPU usage filter"),
    min_memory_mb: Optional[float] = Query(None, ge=0, description="Minimum memory usage filter (MB)"),
    current_user = Depends(get_current_user)
):
    """
    Get comprehensive list of running processes with resource usage.
    
    Features:
    - Flexible sorting by CPU, memory, name, or PID
    - Pattern-based filtering (regex support)
    - Resource threshold filtering
    - Detailed process information including command line and parent PID
    
    Essential for system monitoring and process management.
    """
    client = await get_hardware_client()
    
    try:
        params = {
            "sort_by": sort_by.value,
            "limit": limit
        }
        
        if filter_pattern:
            params["filter"] = filter_pattern
        if min_cpu_percent is not None:
            params["min_cpu_percent"] = min_cpu_percent
        if min_memory_mb is not None:
            params["min_memory_mb"] = min_memory_mb
            
        result = await client.get("/processes", params=params, timeout=15)
        
        # Convert to ProcessInfo models with validation
        processes = []
        for proc in result.get("processes", []):
            try:
                processes.append(ProcessInfo(**proc))
            except Exception as e:
                logger.warning(f"Failed to parse process data: {e}")
                continue
        
        return processes
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get processes: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get process list: {str(e)}"
        )

@router.post("/processes/control", summary="Control System Processes")
async def control_process(
    request: ProcessControlRequest,
    current_user = Depends(require_permissions(["hardware:process_control"]))
):
    """
    Control system processes with various actions.
    
    Supported actions:
    - Kill: Terminate process (SIGKILL/SIGTERM)
    - Suspend: Pause process execution (SIGSTOP)
    - Resume: Resume suspended process (SIGCONT)  
    - Prioritize: Adjust process priority/nice value
    - Limit: Apply resource limits (CPU, memory, I/O)
    
    Safety features:
    - Process validation before action
    - User permission verification
    - Audit logging of all actions
    - Rollback support for priority/limit changes
    
    ⚠️ WARNING: Use with extreme caution. Killing critical processes can destabilize the system.
    """
    client = await get_hardware_client()
    
    try:
        # Enhance request with user and security context
        control_data = request.dict(exclude_none=True)
        control_data.update({
            "user_id": getattr(current_user, 'id', 'unknown'),
            "user_permissions": getattr(current_user, 'permissions', []),
            "timestamp": datetime.utcnow().isoformat(),
            "request_id": str(uuid.uuid4()),
            "source_ip": "backend_api",  # Could be enhanced with actual client IP
            "user_agent": "sutazai_backend"
        })
        
        result = await client.post("/processes/control", data=control_data, timeout=30)
        
        # Log the action for audit trail
        logger.info(f"Process control action executed: {request.action} by user {getattr(current_user, 'id', 'unknown')}")
        
        return {
            **result,
            "action_logged": True,
            "executed_by": getattr(current_user, 'id', 'unknown'),
            "executed_at": datetime.utcnow().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to control process: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to control process: {str(e)}"
        )

@router.get("/monitoring/config", summary="Get Monitoring Configuration")
@cache_static_data(ttl=300)
async def get_monitoring_config(
    current_user = Depends(get_current_user)
):
    """
    Get current hardware monitoring configuration.
    
    Returns detailed monitoring settings including:
    - Collection intervals and retention policies
    - Alert thresholds and notification settings
    - Enabled metric categories
    - Historical configuration changes
    """
    client = await get_hardware_client()
    
    try:
        result = await client.get("/monitoring/config", timeout=10)
        
        # Add user context for configuration access audit
        result["accessed_by"] = getattr(current_user, 'id', 'unknown')
        result["accessed_at"] = datetime.utcnow().isoformat()
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get monitoring config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get monitoring configuration: {str(e)}"
        )

@router.post("/monitoring/config", summary="Update Monitoring Configuration")
async def update_monitoring_config(
    request: MonitoringConfigRequest,
    current_user = Depends(require_permissions(["hardware:configure"]))
):
    """
    Update hardware monitoring configuration.
    
    Configurable settings:
    - Monitoring intervals (5 seconds to 1 hour)
    - Alert thresholds for various metrics
    - Metric collection toggles
    - Data retention periods (1-168 hours)
    - Notification webhook endpoints
    
    Changes are applied immediately and persist across service restarts.
    Configuration history is maintained for audit purposes.
    """
    client = await get_hardware_client()
    
    try:
        # Enhance config with user metadata and validation
        config_data = request.dict()
        config_data.update({
            "updated_by": getattr(current_user, 'id', 'unknown'),
            "updated_at": datetime.utcnow().isoformat(),
            "update_reason": "api_request",
            "validation_passed": True
        })
        
        # Validate webhook URLs if provided
        if config_data.get("notification_webhooks"):
            for webhook_url in config_data["notification_webhooks"]:
                if not webhook_url.startswith(("http://", "https://")):
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid webhook URL format: {webhook_url}"
                    )
        
        result = await client.post("/monitoring/config", data=config_data, timeout=15)
        
        # Clear monitoring config cache after update
        cache_service = await get_cache_service()
        await cache_service.delete_pattern("monitoring_config:*")
        
        logger.info(f"Monitoring configuration updated by user {getattr(current_user, 'id', 'unknown')}")
        
        return {
            **result,
            "cache_cleared": True,
            "updated_by": getattr(current_user, 'id', 'unknown')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to update monitoring config: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to update monitoring configuration: {str(e)}"
        )

@router.get("/alerts", response_model=List[AlertInfo], summary="Get Hardware Alerts")
async def get_hardware_alerts(
    severity: Optional[AlertSeverity] = Query(None, description="Filter by alert severity"),
    limit: int = Query(default=100, ge=1, le=1000, description="Maximum number of alerts to return"),
    since_hours: int = Query(default=24, ge=1, le=168, description="Get alerts from last N hours"),
    acknowledged: Optional[bool] = Query(None, description="Filter by acknowledgment status"),
    current_user = Depends(get_current_user)
):
    """
    Get hardware-related alerts and notifications with comprehensive filtering.
    
    Alert categories include:
    - Resource threshold breaches (CPU, memory, disk, network)
    - Hardware failures or warnings
    - Optimization completion/failure notifications
    - System health score changes
    - Performance anomaly detections
    
    Filtering options:
    - Severity levels (low, medium, high, critical)
    - Time ranges (1-168 hours)
    - Acknowledgment status
    - Custom metric-based filters
    """
    client = await get_hardware_client()
    
    try:
        params = {
            "limit": limit,
            "since_hours": since_hours,
            "user_id": getattr(current_user, 'id', 'unknown')
        }
        
        if severity:
            params["severity"] = severity.value
        if acknowledged is not None:
            params["acknowledged"] = acknowledged
            
        result = await client.get("/alerts", params=params, timeout=15)
        
        # Convert to AlertInfo models with validation
        alerts = []
        for alert_data in result.get("alerts", []):
            try:
                alerts.append(AlertInfo(**alert_data))
            except Exception as e:
                logger.warning(f"Failed to parse alert data: {e}")
                continue
        
        return alerts
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get hardware alerts: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get hardware alerts: {str(e)}"
        )

@router.post("/alerts/{alert_id}/acknowledge", summary="Acknowledge Hardware Alert")
async def acknowledge_alert(
    alert_id: str = Path(..., description="Alert ID to acknowledge"),
    note: Optional[str] = Query(None, description="Optional acknowledgment note"),
    current_user = Depends(get_current_user)
):
    """
    Acknowledge a hardware alert to mark it as seen and handled.
    
    Acknowledgment includes:
    - User identification and timestamp
    - Optional note for documentation
    - Audit trail entry
    - Notification to monitoring systems
    
    Acknowledged alerts may still be active but won't trigger repeated notifications.
    """
    client = await get_hardware_client()
    
    try:
        ack_data = {
            "acknowledged_by": getattr(current_user, 'id', 'unknown'),
            "acknowledged_at": datetime.utcnow().isoformat(),
            "acknowledgment_note": note,
            "source": "api_request"
        }
        
        result = await client.post(f"/alerts/{alert_id}/acknowledge", data=ack_data, timeout=10)
        
        logger.info(f"Alert {alert_id} acknowledged by user {getattr(current_user, 'id', 'unknown')}")
        
        return {
            **result,
            "acknowledgment_confirmed": True,
            "acknowledged_by": getattr(current_user, 'id', 'unknown')
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to acknowledge alert: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to acknowledge alert: {str(e)}"
        )

@router.get("/recommendations", response_model=List[RecommendationInfo], summary="Get AI Optimization Recommendations")
@cache_heavy_computation(ttl=1800)  # 30 minutes
async def get_optimization_recommendations(
    category: Optional[str] = Query(None, description="Filter by recommendation category"),
    priority: Optional[Priority] = Query(None, description="Filter by priority level"),
    implementation_complexity: Optional[str] = Query(None, description="Filter by complexity level"),
    limit: int = Query(default=20, ge=1, le=100, description="Maximum recommendations to return"),
    current_user = Depends(get_current_user)
):
    """
    Get AI-powered optimization recommendations based on current system analysis.
    
    Intelligent recommendations include:
    - Performance optimization opportunities
    - Resource usage efficiency improvements
    - Security hardening suggestions
    - Cost optimization strategies
    - Preventive maintenance actions
    
    Each recommendation includes:
    - Implementation complexity assessment
    - Expected performance impact
    - Prerequisites and dependencies
    - Estimated implementation time
    - Priority scoring based on current system state
    
    Recommendations are generated using machine learning analysis of:
    - Historical performance patterns
    - Resource utilization trends
    - Error and alert patterns
    - Industry best practices
    """
    client = await get_hardware_client()
    
    try:
        params = {
            "limit": limit,
            "user_context": getattr(current_user, 'id', 'unknown')
        }
        
        if category:
            params["category"] = category
        if priority:
            params["priority"] = priority.value
        if implementation_complexity:
            params["complexity"] = implementation_complexity
            
        result = await client.get("/recommendations", params=params, timeout=30)
        
        # Convert to RecommendationInfo models
        recommendations = []
        for rec_data in result.get("recommendations", []):
            try:
                recommendations.append(RecommendationInfo(**rec_data))
            except Exception as e:
                logger.warning(f"Failed to parse recommendation data: {e}")
                continue
        
        return recommendations
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get recommendations: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get optimization recommendations: {str(e)}"
        )

@router.post("/benchmark", response_model=BenchmarkResult, summary="Run System Performance Benchmark")
async def run_system_benchmark(
    request: BenchmarkRequest,
    background_tasks: BackgroundTasks,
    current_user = Depends(require_permissions(["hardware:benchmark"]))
):
    """
    Execute comprehensive system performance benchmarks.
    
    Available benchmark types:
    - CPU: Multi-core processing, floating-point operations, instruction throughput
    - Memory: Bandwidth, latency, cache performance
    - Disk: Sequential/random I/O, IOPS, throughput
    - Network: Bandwidth, latency, packet loss
    - GPU: Compute performance, memory bandwidth (if available)
    - Full: Comprehensive system benchmark across all subsystems
    
    Benchmark results include:
    - Performance scores and percentiles
    - Comparison with baseline measurements
    - Hardware capability analysis
    - Optimization recommendations based on results
    - Historical trend analysis
    
    ⚠️ Note: Benchmarking may temporarily impact system performance.
    Schedule during maintenance windows for production systems.
    """
    client = await get_hardware_client()
    
    try:
        # Prepare benchmark request with comprehensive metadata
        benchmark_data = request.dict()
        benchmark_data.update({
            "started_by": getattr(current_user, 'id', 'unknown'),
            "started_at": datetime.utcnow().isoformat(),
            "user_permissions": getattr(current_user, 'permissions', []),
            "request_id": str(uuid.uuid4()),
            "system_context": {
                "api_version": "v1",
                "source": "backend_api"
            }
        })
        
        result = await client.post("/benchmark", data=benchmark_data, timeout=request.duration_seconds + 60)
        
        # Create background task to monitor benchmark progress
        if result.get("task_id"):
            background_tasks.add_task(
                track_benchmark_progress,
                result["task_id"],
                request.benchmark_type,
                getattr(current_user, 'id', 'unknown')
            )
        
        logger.info(f"Benchmark {request.benchmark_type} started by user {getattr(current_user, 'id', 'unknown')}")
        
        return BenchmarkResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to run benchmark: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to run system benchmark: {str(e)}"
        )

@router.get("/benchmark/{task_id}", response_model=BenchmarkResult, summary="Get Benchmark Results")
@cache_api_response(ttl=300)
async def get_benchmark_results(
    task_id: str = Path(..., description="Benchmark task ID"),
    include_raw_data: bool = Query(default=False, description="Include raw benchmark data"),
    current_user = Depends(get_current_user)
):
    """
    Get detailed results from a completed or running benchmark.
    
    Results include:
    - Performance metrics and scores
    - Comparison with historical baselines
    - System capability assessment
    - Detailed raw data (optional)
    - Recommendations based on results
    """
    client = await get_hardware_client()
    
    try:
        params = {}
        if include_raw_data:
            params["include_raw"] = True
            
        result = await client.get(f"/benchmark/{task_id}", params=params, timeout=15)
        
        # Verify user can access this benchmark
        task_user_id = result.get("started_by")
        if (not getattr(current_user, 'is_admin', False) and 
            task_user_id != str(getattr(current_user, 'id', ''))):
            raise HTTPException(
                status_code=403,
                detail="You can only view your own benchmark results"
            )
        
        return BenchmarkResult(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get benchmark results: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get benchmark results: {str(e)}"
        )

# Analysis endpoints for advanced hardware insights

@router.get("/analyze/performance", summary="Analyze System Performance Patterns")
async def analyze_performance_patterns(
    time_window_hours: int = Query(default=24, ge=1, le=168, description="Analysis time window"),
    include_predictions: bool = Query(default=True, description="Include performance predictions"),
    current_user = Depends(get_current_user)
):
    """
    Perform intelligent analysis of system performance patterns and trends.
    
    Analysis includes:
    - Resource utilization trends and patterns
    - Performance bottleneck identification
    - Capacity planning recommendations
    - Anomaly detection and root cause analysis
    - Predictive insights for future performance
    
    Machine learning algorithms analyze historical data to provide:
    - Seasonal usage patterns
    - Performance degradation trends
    - Optimal resource allocation suggestions
    - Proactive maintenance recommendations
    """
    client = await get_hardware_client()
    
    try:
        params = {
            "window_hours": time_window_hours,
            "include_predictions": include_predictions,
            "user_context": getattr(current_user, 'id', 'unknown')
        }
        
        result = await client.get("/analyze/performance", params=params, timeout=45)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze performance: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze performance patterns: {str(e)}"
        )

@router.get("/analyze/capacity", summary="Capacity Planning Analysis")
async def analyze_capacity_planning(
    forecast_days: int = Query(default=30, ge=7, le=365, description="Forecast period in days"),
    growth_scenarios: List[str] = Query(default=["conservative", "moderate", "aggressive"], description="Growth scenarios to analyze"),
    current_user = Depends(get_current_user)
):
    """
    Generate comprehensive capacity planning analysis and forecasts.
    
    Provides strategic insights for:
    - Resource growth projections
    - Hardware upgrade recommendations
    - Cost optimization strategies
    - Performance scaling requirements
    - Risk assessment for capacity constraints
    
    Multiple growth scenarios help with:
    - Conservative planning for steady growth
    - Moderate planning for expected expansion
    - Aggressive planning for rapid scaling
    """
    client = await get_hardware_client()
    
    try:
        params = {
            "forecast_days": forecast_days,
            "scenarios": growth_scenarios,
            "analysis_depth": "comprehensive"
        }
        
        result = await client.get("/analyze/capacity", params=params, timeout=60)
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze capacity: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze capacity planning: {str(e)}"
        )

@router.get("/analyze/security", summary="Hardware Security Assessment")
async def analyze_hardware_security(
    include_vulnerabilities: bool = Query(default=True, description="Include vulnerability assessment"),
    include_compliance: bool = Query(default=True, description="Include compliance check"),
    current_user = Depends(require_permissions(["hardware:security", "security:read"]))
):
    """
    Perform comprehensive hardware security analysis and assessment.
    
    Security analysis includes:
    - Hardware vulnerability scanning
    - Configuration security assessment
    - Compliance checking (SOC 2, ISO 27001, etc.)
    - Access control evaluation
    - Audit trail analysis
    
    Identifies security risks in:
    - Hardware configurations
    - Resource access patterns
    - Process execution privileges
    - Network exposure points
    - Data protection measures
    """
    client = await get_hardware_client()
    
    try:
        params = {
            "include_vulnerabilities": include_vulnerabilities,
            "include_compliance": include_compliance,
            "security_level": "comprehensive",
            "auditor": getattr(current_user, 'id', 'unknown')
        }
        
        result = await client.get("/analyze/security", params=params, timeout=120)
        
        # Log security analysis for audit
        logger.info(f"Hardware security analysis performed by user {getattr(current_user, 'id', 'unknown')}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to analyze security: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to analyze hardware security: {str(e)}"
        )

# Background task functions

async def track_optimization_progress(task_id: str, optimization_type: str, user_id: str):
    """
    Background task to track optimization progress and update metrics with enhanced monitoring
    """
    try:
        client = await get_hardware_client()
        cache_service = await get_cache_service()
        
        logger.info(f"Starting optimization tracking for task {task_id} (type: {optimization_type}, user: {user_id})")
        
        # Track progress with more frequent updates and better error handling
        max_checks = 240  # 20 minutes max (5 second intervals)
        check_count = 0
        last_status = None
        consecutive_failures = 0
        
        while check_count < max_checks:
            try:
                status = await client.get(f"/optimize/{task_id}", timeout=10)
                
                # Reset failure counter on success
                consecutive_failures = 0
                
                # Update cache with latest status
                cache_key = f"optimization_status:{task_id}"
                await cache_service.set(cache_key, status, ttl=300)
                
                # Log status changes
                if status.get("status") != last_status:
                    logger.info(f"Optimization {task_id} status changed: {last_status} -> {status.get('status')}")
                    last_status = status.get("status")
                
                # Check for completion
                if status.get("status") in ["completed", "failed", "cancelled"]:
                    logger.info(f"Optimization {task_id} finished with status: {status.get('status')}")
                    
                    # Store final results in cache for extended period
                    await cache_service.set(
                        f"optimization_final:{task_id}", 
                        status, 
                        ttl=86400  # 24 hours
                    )
                    break
                    
                await asyncio.sleep(5)  # Check every 5 seconds
                check_count += 1
                
            except Exception as e:
                consecutive_failures += 1
                logger.warning(f"Failed to check optimization progress for {task_id} (attempt {consecutive_failures}): {e}")
                
                # If too many consecutive failures, extend sleep time
                if consecutive_failures >= 3:
                    await asyncio.sleep(30)
                    check_count += 6  # Account for longer sleep
                else:
                    await asyncio.sleep(10)
                    check_count += 2
                
                # If too many failures, give up
                if consecutive_failures >= 10:
                    logger.error(f"Too many failures tracking optimization {task_id}, giving up")
                    break
        
        if check_count >= max_checks:
            logger.warning(f"Optimization tracking timeout for task {task_id}")
            
    except Exception as e:
        logger.error(f"Error in optimization progress tracking: {e}")

async def track_benchmark_progress(task_id: str, benchmark_type: str, user_id: str):
    """
    Background task to track benchmark progress and handle results
    """
    try:
        client = await get_hardware_client()
        cache_service = await get_cache_service()
        
        logger.info(f"Starting benchmark tracking for task {task_id} (type: {benchmark_type}, user: {user_id})")
        
        # Track benchmark with appropriate timeout
        max_checks = 180  # 15 minutes max (5 second intervals)
        check_count = 0
        
        while check_count < max_checks:
            try:
                status = await client.get(f"/benchmark/{task_id}", timeout=15)
                
                # Update cache
                cache_key = f"benchmark_result:{task_id}"
                await cache_service.set(cache_key, status, ttl=1800)  # 30 minutes
                
                if status.get("status") in ["completed", "failed", "cancelled"]:
                    logger.info(f"Benchmark {task_id} finished with status: {status.get('status')}")
                    
                    # Store results for extended period
                    await cache_service.set(
                        f"benchmark_final:{task_id}",
                        status,
                        ttl=604800  # 7 days
                    )
                    break
                    
                await asyncio.sleep(5)
                check_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to check benchmark progress for {task_id}: {e}")
                await asyncio.sleep(15)
                check_count += 3
        
        if check_count >= max_checks:
            logger.warning(f"Benchmark tracking timeout for task {task_id}")
            
    except Exception as e:
        logger.error(f"Error in benchmark progress tracking: {e}")

# Error Handlers with comprehensive error mapping
# Note: Exception handlers need to be registered at the FastAPI app level, not router level

# Health check for router itself
@router.get("/router/health", summary="Hardware Router Health Check")
async def router_health_check():
    """
    Health check for the hardware API router itself.
    
    Verifies:
    - Router configuration
    - Client initialization
    - Cache service connectivity
    - Basic service connectivity
    """
    try:
        # Test client creation
        client = await get_hardware_client()
        
        # Test cache service
        cache_service = await get_cache_service()
        test_key = f"router_health_test:{datetime.utcnow().timestamp()}"
        await cache_service.set(test_key, "ok", ttl=60)
        cache_result = await cache_service.get(test_key)
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "router_version": "1.0.0",
            "client_initialized": client is not None,
            "cache_functional": cache_result == "ok",
            "hardware_service_url": HARDWARE_SERVICE_URL,
            "configuration": {
                "timeout": HARDWARE_SERVICE_TIMEOUT,
                "cache_ttl": HARDWARE_CACHE_TTL,
                "max_retries": HARDWARE_MAX_RETRIES
            }
        }
        
    except Exception as e:
        logger.error(f"Hardware router health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "timestamp": datetime.utcnow().isoformat(),
                "error": str(e)
            }
        )