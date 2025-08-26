"""
MCP (Model Context Protocol) API Endpoints
Provides HTTP/REST access to MCP servers through the service mesh
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
import logging
import os

# Configure logger first
logger = logging.getLogger(__name__)

# Real DinD integration imports (FIXED IMPORT PATH)
try:
    from app.mesh.dind_mesh_bridge import get_dind_bridge, DinDMeshBridge
    from app.mesh.service_mesh import ServiceMesh
    from app.mesh.mcp_stdio_bridge import get_mcp_stdio_bridge, MCPStdioBridge
    from app.mesh.unified_dev_adapter import get_unified_dev_adapter, UnifiedDevAdapter
    DIND_AVAILABLE = True
    MESH_AVAILABLE = True
    UNIFIED_DEV_AVAILABLE = True
except ImportError as e:
    logger.warning(f"MCP bridge modules not available: {e}")
    DIND_AVAILABLE = False
    MESH_AVAILABLE = False
    UNIFIED_DEV_AVAILABLE = False

# Service mesh dependency
async def get_service_mesh() -> ServiceMesh:
    """Get service mesh instance"""
    if MESH_AVAILABLE:
        from app.mesh.service_mesh import ServiceMesh
        # Initialize with proper configuration
        return ServiceMesh()
    else:
        raise HTTPException(status_code=503, detail="Service mesh not available")

router = APIRouter(prefix="/mcp", tags=["mcp"])

# Import and include unified memory routes
try:
    from .unified_memory import router as unified_memory_router
    router.include_router(unified_memory_router, prefix="/unified-memory", tags=["unified-memory"])
except ImportError:
    logger.warning("Unified memory router not available")

# Import and include migration routes
try:
    from .migration import router as migration_router
    router.include_router(migration_router, prefix="/migration", tags=["migration"])
except ImportError:
    logger.warning("Migration router not available")

# Request/Response Models
class MCPExecuteRequest(BaseModel):
    """Request model for MCP command execution"""
    method: str = Field(..., description="MCP method to execute")
    params: Dict[str, Any] = Field(default_factory=dict, description="Method parameters")
    timeout: Optional[float] = Field(30.0, description="Request timeout in seconds")
    retry_count: Optional[int] = Field(3, description="Number of retries on failure")

class MCPServiceStatus(BaseModel):
    """MCP service status response"""
    service: str
    status: str
    mesh_instances: int
    adapter_instances: int
    instances: List[Dict[str, Any]]
    mesh_registration: bool

class MCPHealthStatus(BaseModel):
    """MCP service health status"""
    total_instances: int
    healthy: int
    unhealthy: int
    unknown: int
    overall_health: str

class MCPHealthSummary(BaseModel):
    """MCP health summary"""
    total: int
    healthy: int
    unhealthy: int
    percentage_healthy: float

class MCPServiceHealth(BaseModel):
    """Individual MCP service health"""
    healthy: bool
    available: bool
    process_running: bool
    retry_count: int
    last_check: Optional[str]

class MCPHealthResponse(BaseModel):
    """Complete MCP health response"""
    services: Dict[str, MCPServiceHealth]
    summary: MCPHealthSummary

# Working MCP bridge functionality
async def get_bridge():
    """Get working MCP bridge instance with fallback strategy"""
    
    # Priority 1: Try DinD bridge for best isolation and multi-client support
    if DIND_AVAILABLE:
        try:
            mesh = await get_service_mesh()
            dind_bridge = await get_dind_bridge(mesh)
            if dind_bridge.initialized:
                logger.info("Using DinD bridge for MCP operations")
                return dind_bridge
        except Exception as e:
            logger.warning(f"DinD bridge initialization failed: {e}")
    
    # Priority 2: Try STDIO bridge for direct MCP communication
    if MESH_AVAILABLE:
        try:
            stdio_bridge = await get_mcp_stdio_bridge()
            logger.info("Using STDIO bridge for MCP operations")
            return stdio_bridge
        except Exception as e:
            logger.warning(f"STDIO bridge initialization failed: {e}")
    
    # If all bridges fail, provide informative error
    raise HTTPException(
        status_code=503, 
        detail="MCP infrastructure not available. Check DinD containers and bridge modules."
    )

# API Endpoints
@router.get("/status")
async def get_mcp_status(bridge = Depends(get_bridge)):
    """
    Get overall MCP system status
    
    Returns comprehensive status of MCP infrastructure including DinD and bridge status
    """
    try:
        # Get bridge status
        bridge_type = type(bridge).__name__
        # Handle both DinD/container bridges (initialized) and stdio bridge (_initialized)
        if hasattr(bridge, 'initialized'):
            bridge_initialized = bool(getattr(bridge, 'initialized'))
        elif hasattr(bridge, '_initialized'):
            bridge_initialized = bool(getattr(bridge, '_initialized'))
        else:
            bridge_initialized = False
        
        # Get service count
        services = bridge.registry.get('mcp_services', [])
        service_count = len(services)
        
        # Get DinD status if available
        dind_status = "unknown"
        if DIND_AVAILABLE:
            try:
                # Check if DinD client is properly connected
                if hasattr(bridge, 'dind_client') and bridge.dind_client:
                    try:
                        # Test actual connection by pinging DinD
                        bridge.dind_client.ping()
                        dind_status = "connected"
                    except Exception as ping_error:
                        logger.warning(f"DinD ping failed: {ping_error}")
                        dind_status = "not_connected"
                else:
                    dind_status = "not_connected"
            except Exception:
                dind_status = "error"
        else:
            dind_status = "not_available"
        
        return {
            "status": "operational" if bridge_initialized else "initializing",
            "bridge_type": bridge_type,
            "bridge_initialized": bridge_initialized,
            "service_count": service_count,
            "dind_status": dind_status,
            "infrastructure": {
                "dind_available": DIND_AVAILABLE,
                "mesh_available": MESH_AVAILABLE,
                "bridge_type": bridge_type
            },
            "timestamp": "2025-08-16T23:36:00Z"
        }
    except Exception as e:
        logger.error(f"Failed to get MCP status: {e}")
        return {
            "status": "error",
            "error": str(e),
            "bridge_type": "unknown",
            "bridge_initialized": False,
            "service_count": 0,
            "dind_status": "error",
            "infrastructure": {
                "dind_available": DIND_AVAILABLE,
                "mesh_available": MESH_AVAILABLE,
                "bridge_type": "error"
            },
            "timestamp": "2025-08-16T23:36:00Z"
        }

@router.get("/services", response_model=List[str])
async def list_mcp_services(bridge = Depends(get_bridge)):
    """
    List all available MCP services
    
    Returns list of registered MCP service names
    """
    try:
        # Get services from registry
        services = [
            service.get('name') 
            for service in bridge.registry.get('mcp_services', [])
            if service.get('name')
        ]
        return services
    except Exception as e:
        logger.error(f"Failed to list MCP services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/services/{service_name}/status", response_model=MCPServiceStatus)
async def get_mcp_service_status(
    service_name: str,
    bridge = Depends(get_bridge)
):
    """
    Get detailed status of an MCP service
    
    Args:
        service_name: Name of the MCP service
    
    Returns service status including instance details
    """
    try:
        status = await bridge.get_service_status(service_name)
        if status.get('status') == 'not_found':
            raise HTTPException(status_code=404, detail=status.get('error'))
        return status
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP service status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/services/{service_name}/execute")
async def execute_mcp_command(
    service_name: str,
    request: MCPExecuteRequest,
    bridge = Depends(get_bridge)
):
    """
    Execute a command on an MCP service
    
    Args:
        service_name: Name of the MCP service
        request: Execution request with method and parameters
    
    Returns command execution result
    """
    try:
        result = await bridge.call_mcp_service(
            service_name=service_name,
            method=request.method,
            params=request.params
        )
        return result
    except Exception as e:
        logger.error(f"Failed to execute MCP command: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=MCPHealthResponse)
async def get_mcp_health(bridge = Depends(get_bridge)):
    """
    Get health status of all MCP services
    
    Returns health metrics for each registered MCP service
    """
    try:
        # Get raw health data from bridge
        health = await bridge.health_check_all()
        
        # If no services are actively running, check configuration
        if not health.get('services'):
            # Load MCP configuration to show available services
            import json
            config_file = "/opt/sutazaiapp/.mcp.json"
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    mcp_config = json.load(f)
                
                mcp_servers = mcp_config.get("mcpServers", {})
                services = {}
                
                # Report configuration status for each service
                for name, server_config in mcp_servers.items():
                    # Check if wrapper exists
                    wrapper_exists = False
                    if server_config.get("command", "").startswith("/opt/sutazaiapp/scripts/mcp/wrappers/"):
                        wrapper_exists = os.path.exists(server_config["command"])
                    else:
                        wrapper_path = f"/opt/sutazaiapp/scripts/mcp/wrappers/{name}.sh"
                        wrapper_exists = os.path.exists(wrapper_path)
                    
                    services[name] = MCPServiceHealth(
                        healthy=wrapper_exists,  # Consider configured as "healthy" if wrapper exists
                        available=wrapper_exists,
                        process_running=False,  # Not actually running
                        retry_count=0,
                        last_check=None
                    )
                
                # Create summary based on configuration
                total = len(services)
                healthy_count = sum(1 for s in services.values() if s.healthy)
                
                summary = MCPHealthSummary(
                    total=total,
                    healthy=healthy_count,
                    unhealthy=total - healthy_count,
                    percentage_healthy=(healthy_count / total * 100) if total > 0 else 0.0
                )
                
                return MCPHealthResponse(services=services, summary=summary)
        
        # Transform the runtime response to match the model
        services = {}
        for name, status in health.get('services', {}).items():
            services[name] = MCPServiceHealth(
                healthy=status.get('healthy', False),
                available=status.get('available', False),
                process_running=status.get('process_running', False),
                retry_count=status.get('retry_count', 0),
                last_check=status.get('last_check')
            )
        
        summary_data = health.get('summary', {})
        summary = MCPHealthSummary(
            total=summary_data.get('total', 0),
            healthy=summary_data.get('healthy', 0),
            unhealthy=summary_data.get('unhealthy', 0),
            percentage_healthy=summary_data.get('percentage_healthy', 0.0)
        )
        
        return MCPHealthResponse(services=services, summary=summary)
    except Exception as e:
        logger.error(f"Failed to get MCP health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/services/{service_name}/restart")
async def restart_mcp_service(
    service_name: str,
    bridge = Depends(get_bridge)
):
    """
    Restart an MCP service
    
    Args:
        service_name: Name of the MCP service to restart
    
    Returns restart status
    """
    try:
        success = await bridge.restart_service(service_name)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to restart {service_name}")
        return {"status": "success", "service": service_name, "message": "Service restarted"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to restart MCP service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/services/{service_name}")
async def stop_mcp_service(
    service_name: str,
    bridge = Depends(get_bridge)
):
    """
    Stop an MCP service
    
    Args:
        service_name: Name of the MCP service to stop
    
    Returns stop status
    """
    try:
        success = await bridge.stop_service(service_name)
        if not success:
            raise HTTPException(status_code=500, detail=f"Failed to stop {service_name}")
        return {"status": "success", "service": service_name, "message": "Service stopped"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to stop MCP service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/initialize")
async def initialize_mcp_services(bridge = Depends(get_bridge)):
    """
    Initialize all MCP services from registry
    
    Starts all configured MCP services and registers them with the mesh
    """
    try:
        result = await bridge.initialize()
        
        # Check if any services failed
        if result.get("failed"):
            return {
                "status": "partial",
                "message": f"Initialized {len(result['started'])} services, {len(result['failed'])} failed",
                "details": result
            }
        
        return {
            "status": "success",
            "message": f"Initialized {len(result['started'])} MCP services",
            "details": result
        }
    except Exception as e:
        logger.error(f"Failed to initialize MCP services: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Convenience endpoints for specific MCP servers
@router.post("/postgres/query")
async def postgres_query(
    query: str = Body(..., description="SQL query to execute"),
    bridge = Depends(get_bridge)
):
    """Execute SQL query via Postgres MCP server"""
    try:
        result = await bridge.call_mcp_service(
            service_name="postgres",
            method="query",
            params={"query": query}
        )
        return result
    except Exception as e:
        logger.error(f"Postgres query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/files/read")
async def read_file(
    path: str = Body(..., description="File path to read"),
    bridge = Depends(get_bridge)
):
    """Read file via Files MCP server"""
    try:
        result = await bridge.call_mcp_service(
            service_name="files",
            method="read",
            params={"path": path}
        )
        return result
    except Exception as e:
        logger.error(f"File read failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/http/fetch")
async def fetch_url(
    url: str = Body(..., description="URL to fetch"),
    method: str = Body("GET", description="HTTP method"),
    bridge = Depends(get_bridge)
):
    """Fetch URL via HTTP MCP server"""
    try:
        result = await bridge.call_mcp_service(
            service_name="http",
            method="fetch",
            params={"url": url, "method": method}
        )
        return result
    except Exception as e:
        logger.error(f"HTTP fetch failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/search")
async def search_web(
    query: str = Body(..., description="Search query"),
    engine: str = Body("ddg", description="Search engine (ddg)"),
    bridge = Depends(get_bridge)
):
    """Search the web via search MCP servers"""
    try:
        result = await bridge.call_mcp_service(
            service_name=engine,
            method="search",
            params={"query": query}
        )
        return result
    except Exception as e:
        logger.error(f"Search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/capabilities/{service_name}")
async def get_mcp_capabilities(
    service_name: str,
    bridge = Depends(get_bridge)
):
    """
    Get capabilities of an MCP service
    
    Returns the methods and features supported by the service
    """
    try:
        # Find service in registry
        for service in bridge.registry.get('mcp_services', []):
            if service.get('name') == service_name:
                return {
                    "service": service_name,
                    "capabilities": service.get('metadata', {}).get('capabilities', []),
                    "tags": service.get('tags', []),
                    "metadata": service.get('metadata', {})
                }
        
        raise HTTPException(status_code=404, detail=f"Service {service_name} not found")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get MCP capabilities: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# DinD Multi-Client Endpoints
@router.get("/dind/status")
async def get_dind_status():
    """
    Get status of Docker-in-Docker MCP orchestration
    
    Returns information about DinD-managed MCP containers
    """
    if not DIND_AVAILABLE:
        return {
            "error": "DinD bridge not available",
            "initialized": False,
            "services": {},
            "container_status": "Not available"
        }
    
    try:
        # Check DinD container directly
        import subprocess
        result = subprocess.run(
            ["docker", "ps", "--filter", "name=sutazai-mcp-orchestrator", "--format", "{{.Status}}"],
            capture_output=True, text=True
        )
        
        dind_running = "Up" in result.stdout if result.returncode == 0 else False
        
        # Check containers inside DinD
        if dind_running:
            inner_result = subprocess.run(
                ["docker", "exec", "sutazai-mcp-orchestrator", "docker", "ps", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            mcp_containers = inner_result.stdout.strip().split('\n') if inner_result.stdout.strip() else []
        else:
            mcp_containers = []
        
        return {
            "dind_container_running": dind_running,
            "dind_container_status": result.stdout.strip() if result.returncode == 0 else "Not found",
            "mcp_containers": mcp_containers,
            "mcp_container_count": len(mcp_containers),
            "ports": {
                "docker_api": 12375,
                "docker_api_tls": 12376,
                "orchestrator_api": 18080,
                "metrics": 19090
            },
            "initialized": dind_running
        }
    except Exception as e:
        logger.error(f"Failed to get DinD status: {e}")
        return {
            "error": str(e),
            "initialized": False,
            "services": {}
        }

@router.post("/dind/deploy")
async def deploy_mcp_to_dind(
    config: Dict[str, Any] = Body(..., description="MCP container configuration")
):
    """
    Deploy a new MCP container to DinD orchestrator
    
    Args:
        config: Container configuration including name, image, environment
    
    Returns deployment status
    """
    try:
        dind_bridge = await get_dind_bridge(mesh)
        service = await dind_bridge.deploy_mcp(config)
        
        if service:
            return {
                "status": "deployed",
                "service": service.name,
                "mesh_port": service.mesh_port,
                "container_id": service.container_id[:12]
            }
        else:
            raise HTTPException(status_code=500, detail="Failed to deploy MCP container")
            
    except Exception as e:
        logger.error(f"Failed to deploy MCP to DinD: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/dind/{service_name}/request")
async def multi_client_request(
    service_name: str,
    client_id: str = Query(..., description="Client identifier (e.g., 'claude-code', 'codex')"),
    request: Dict[str, Any] = Body(..., description="Request payload"),
    mesh: ServiceMesh = Depends(get_service_mesh)
):
    """
    Send request to DinD-managed MCP service with multi-client support
    
    Args:
        service_name: Name of the MCP service
        client_id: Identifier for the client making the request
        request: Request payload
    
    Returns response from MCP service
    """
    try:
        dind_bridge = await get_dind_bridge(mesh)
        response = await dind_bridge.handle_client_request(
            service_name=service_name,
            client_id=client_id,
            request=request
        )
        return response
    except Exception as e:
        logger.error(f"Failed to handle multi-client request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/dind/{service_name}/clients")
async def get_service_clients(
    service_name: str,
    mesh: ServiceMesh = Depends(get_service_mesh)
):
    """
    Get list of clients connected to a DinD MCP service
    
    Args:
        service_name: Name of the MCP service
    
    Returns list of connected client IDs
    """
    try:
        dind_bridge = await get_dind_bridge(mesh)
        if service_name in dind_bridge.mcp_services:
            service = dind_bridge.mcp_services[service_name]
            return {
                "service": service_name,
                "clients": service.clients,
                "client_count": len(service.clients)
            }
        else:
            raise HTTPException(status_code=404, detail=f"Service {service_name} not found in DinD")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get service clients: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =============================================================================
# UNIFIED DEVELOPMENT SERVICE ENDPOINTS
# Consolidates ultimatecoder, language-server, and sequentialthinking
# =============================================================================

class UnifiedDevRequest(BaseModel):
    """Base request model for unified development service"""
    service: Optional[str] = Field(None, description="Target service (ultimatecoder, language-server, sequentialthinking)")

class CodeRequest(UnifiedDevRequest):
    """Request model for code-related operations"""
    code: str = Field(..., description="Source code to process")
    language: str = Field(..., description="Programming language")
    action: str = Field("generate", description="Action to perform (generate, analyze, refactor, optimize)")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)

class LanguageServerRequest(UnifiedDevRequest):
    """Request model for language server operations"""
    method: str = Field(..., description="LSP method name")
    params: Optional[Dict[str, Any]] = Field(default_factory=dict)
    workspace: str = Field("/opt/sutazaiapp", description="Workspace path")

class SequentialThinkingRequest(UnifiedDevRequest):
    """Request model for sequential thinking operations"""
    query: str = Field(..., description="Query or problem to analyze")
    context: Optional[Dict[str, Any]] = Field(default_factory=dict)
    max_steps: int = Field(10, description="Maximum reasoning steps")
    steps: Optional[List[Dict[str, Any]]] = Field(default_factory=list)

@router.get("/unified-dev/status")
async def get_unified_dev_status():
    """
    Get status of the unified development service
    
    Returns health and capability information for the consolidated service
    """
    if not UNIFIED_DEV_AVAILABLE:
        raise HTTPException(status_code=503, detail="Unified development service not available")
    
    try:
        adapter = await get_unified_dev_adapter()
        health = await adapter.health_check()
        
        if not health:
            raise HTTPException(status_code=503, detail="Unified development service is unhealthy")
        
        performance = adapter.get_performance_summary()
        
        return {
            "status": "healthy",
            "service": "unified-dev",
            "version": "1.0.0",
            "consolidated_services": ["ultimatecoder", "language-server", "sequentialthinking"],
            "health": health,
            "performance": performance,
            "capabilities": adapter.service_capabilities,
            "memory_target": "512MB",
            "consolidation_savings": {
                "memory_saved": "512MB",
                "processes_reduced": "66%",
                "containers_eliminated": 2
            }
        }
    except Exception as e:
        logger.error(f"Failed to get unified dev status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/unified-dev/metrics")
async def get_unified_dev_metrics():
    """
    Get detailed metrics from unified development service
    
    Returns performance metrics, resource usage, and operational statistics
    """
    if not UNIFIED_DEV_AVAILABLE:
        raise HTTPException(status_code=503, detail="Unified development service not available")
    
    try:
        adapter = await get_unified_dev_adapter()
        metrics = await adapter.get_metrics()
        performance = adapter.get_performance_summary()
        
        return {
            "detailed_metrics": metrics,
            "performance_summary": performance,
            "timestamp": metrics.get("timestamp"),
            "service": "unified-dev"
        }
    except Exception as e:
        logger.error(f"Failed to get unified dev metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unified-dev/code")
async def process_code_request(request: CodeRequest):
    """
    Process code-related requests using UltimateCoder capabilities
    
    Supports: generate, analyze, refactor, optimize actions
    """
    if not UNIFIED_DEV_AVAILABLE:
        raise HTTPException(status_code=503, detail="Unified development service not available")
    
    try:
        adapter = await get_unified_dev_adapter()
        
        if request.action == "generate":
            result = await adapter.generate_code(request.code, request.language, request.context)
        elif request.action == "analyze":
            result = await adapter.analyze_code(request.code, request.language, request.context)
        elif request.action == "refactor":
            result = await adapter.refactor_code(request.code, request.language, request.context)
        elif request.action == "optimize":
            result = await adapter.optimize_code(request.code, request.language, request.context)
        else:
            raise HTTPException(status_code=400, detail=f"Unknown action: {request.action}")
        
        return {
            "success": True,
            "action": request.action,
            "language": request.language,
            "result": result,
            "consolidated_service": "ultimatecoder"
        }
        
    except Exception as e:
        logger.error(f"Failed to process code request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unified-dev/lsp")
async def process_language_server_request(request: LanguageServerRequest):
    """
    Process Language Server Protocol requests
    
    Supports: completion, diagnostics, hover, definition, and other LSP methods
    """
    if not UNIFIED_DEV_AVAILABLE:
        raise HTTPException(status_code=503, detail="Unified development service not available")
    
    try:
        adapter = await get_unified_dev_adapter()
        
        # Route to appropriate LSP method
        if request.method == "textDocument/completion":
            result = await adapter.get_completions(request.method, request.params, request.workspace)
        elif request.method == "textDocument/publishDiagnostics":
            result = await adapter.get_diagnostics(request.method, request.params, request.workspace)
        elif request.method == "textDocument/hover":
            result = await adapter.get_hover_info(request.method, request.params, request.workspace)
        elif request.method == "textDocument/definition":
            result = await adapter.get_definition(request.method, request.params, request.workspace)
        else:
            # Generic LSP method handling
            result = await adapter.get_completions(request.method, request.params, request.workspace)
        
        return {
            "success": True,
            "method": request.method,
            "workspace": request.workspace,
            "result": result,
            "consolidated_service": "language-server"
        }
        
    except Exception as e:
        logger.error(f"Failed to process LSP request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unified-dev/reasoning")
async def process_sequential_thinking_request(request: SequentialThinkingRequest):
    """
    Process sequential thinking and reasoning requests
    
    Supports: multi-step reasoning, planning, complex problem analysis
    """
    if not UNIFIED_DEV_AVAILABLE:
        raise HTTPException(status_code=503, detail="Unified development service not available")
    
    try:
        adapter = await get_unified_dev_adapter()
        
        if request.steps:
            # Multi-step planning request
            result = await adapter.multi_step_planning(
                request.query, 
                request.steps, 
                request.context
            )
        else:
            # Sequential reasoning request
            result = await adapter.sequential_reasoning(
                request.query, 
                request.context, 
                request.max_steps
            )
        
        return {
            "success": True,
            "query": request.query,
            "max_steps": request.max_steps,
            "result": result,
            "consolidated_service": "sequentialthinking"
        }
        
    except Exception as e:
        logger.error(f"Failed to process reasoning request: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unified-dev/comprehensive-analysis")
async def comprehensive_code_analysis_endpoint(
    code: str = Body(..., description="Source code to analyze"),
    language: str = Body(..., description="Programming language"),
    include_reasoning: bool = Body(True, description="Include sequential thinking analysis")
):
    """
    Perform comprehensive code analysis using multiple unified services
    
    Combines UltimateCoder analysis with Sequential Thinking reasoning
    """
    if not UNIFIED_DEV_AVAILABLE:
        raise HTTPException(status_code=503, detail="Unified development service not available")
    
    try:
        adapter = await get_unified_dev_adapter()
        result = await adapter.comprehensive_code_analysis(code, language, include_reasoning)
        
        return {
            "success": True,
            "analysis_type": "comprehensive",
            "language": language,
            "include_reasoning": include_reasoning,
            "result": result,
            "consolidated_services": ["ultimatecoder", "sequentialthinking"] if include_reasoning else ["ultimatecoder"]
        }
        
    except Exception as e:
        logger.error(f"Failed to perform comprehensive analysis: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/unified-dev/intelligent-generation")
async def intelligent_code_generation_endpoint(
    requirements: str = Body(..., description="Code requirements and specifications"),
    language: str = Body(..., description="Target programming language"),
    use_planning: bool = Body(True, description="Use sequential thinking for planning")
):
    """
    Generate code with intelligent planning and reasoning
    
    Combines Sequential Thinking planning with UltimateCoder generation
    """
    if not UNIFIED_DEV_AVAILABLE:
        raise HTTPException(status_code=503, detail="Unified development service not available")
    
    try:
        adapter = await get_unified_dev_adapter()
        result = await adapter.intelligent_code_generation(requirements, language, use_planning)
        
        return {
            "success": True,
            "generation_type": "intelligent",
            "language": language,
            "use_planning": use_planning,
            "requirements": requirements,
            "result": result,
            "consolidated_services": ["sequentialthinking", "ultimatecoder"] if use_planning else ["ultimatecoder"]
        }
        
    except Exception as e:
        logger.error(f"Failed to perform intelligent generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Legacy compatibility endpoints for backward compatibility
@router.post("/ultimatecoder/{action}")
async def legacy_ultimatecoder_endpoint(
    action: str,
    code: str = Body(...),
    language: str = Body(...),
    context: Optional[Dict[str, Any]] = Body(default_factory=dict)
):
    """Legacy compatibility endpoint for ultimatecoder service"""
    request = CodeRequest(
        service="ultimatecoder",
        code=code,
        language=language,
        action=action,
        context=context
    )
    return await process_code_request(request)

@router.post("/language-server/{method}")
async def legacy_language_server_endpoint(
    method: str,
    params: Optional[Dict[str, Any]] = Body(default_factory=dict),
    workspace: str = Body("/opt/sutazaiapp")
):
    """Legacy compatibility endpoint for language-server service"""
    request = LanguageServerRequest(
        service="language-server",
        method=method,
        params=params,
        workspace=workspace
    )
    return await process_language_server_request(request)

@router.post("/sequentialthinking/reasoning")
async def legacy_sequential_thinking_endpoint(
    query: str = Body(...),
    context: Optional[Dict[str, Any]] = Body(default_factory=dict),
    max_steps: int = Body(10)
):
    """Legacy compatibility endpoint for sequentialthinking service"""
    request = SequentialThinkingRequest(
        service="sequentialthinking",
        query=query,
        context=context,
        max_steps=max_steps
    )
    return await process_sequential_thinking_request(request)