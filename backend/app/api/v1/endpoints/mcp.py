"""
MCP (Model Context Protocol) API Endpoints
Provides HTTP/REST access to MCP servers through the service mesh
"""
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Depends, Query, Body
from pydantic import BaseModel, Field
import logging

from ....mesh.service_mesh import ServiceMesh, get_service_mesh
from ....mesh.mcp_bridge import get_mcp_bridge, MCPMeshBridge

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["mcp"])

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

# Dependency to get MCP bridge
async def get_bridge() -> MCPMeshBridge:
    """Get MCP bridge instance"""
    mesh = await get_service_mesh()
    return await get_mcp_bridge(mesh)

# API Endpoints
@router.get("/services", response_model=List[str])
async def list_mcp_services(bridge: MCPMeshBridge = Depends(get_bridge)):
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
    bridge: MCPMeshBridge = Depends(get_bridge)
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
    bridge: MCPMeshBridge = Depends(get_bridge)
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

@router.get("/health", response_model=Dict[str, MCPHealthStatus])
async def get_mcp_health(bridge: MCPMeshBridge = Depends(get_bridge)):
    """
    Get health status of all MCP services
    
    Returns health metrics for each registered MCP service
    """
    try:
        health = await bridge.health_check_all()
        return health
    except Exception as e:
        logger.error(f"Failed to get MCP health: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/services/{service_name}/restart")
async def restart_mcp_service(
    service_name: str,
    bridge: MCPMeshBridge = Depends(get_bridge)
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
    bridge: MCPMeshBridge = Depends(get_bridge)
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
async def initialize_mcp_services(bridge: MCPMeshBridge = Depends(get_bridge)):
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
    bridge: MCPMeshBridge = Depends(get_bridge)
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
    bridge: MCPMeshBridge = Depends(get_bridge)
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
    bridge: MCPMeshBridge = Depends(get_bridge)
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
    bridge: MCPMeshBridge = Depends(get_bridge)
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
    bridge: MCPMeshBridge = Depends(get_bridge)
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