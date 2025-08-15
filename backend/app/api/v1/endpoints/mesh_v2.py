"""
Service Mesh V2 API - Real service mesh with discovery, load balancing, and circuit breaking
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from fastapi import APIRouter, HTTPException, Query, BackgroundTasks
import logging

from app.mesh.service_mesh import (
    get_mesh, ServiceRequest, ServiceInstance, ServiceState,
    LoadBalancerStrategy, enqueue_task_mesh, tail_results_mesh
)

logger = logging.getLogger(__name__)
router = APIRouter()

class ServiceRegistrationRequest(BaseModel):
    """Request to register a service"""
    service_name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9-]+$")
    address: str = Field(..., min_length=1)
    port: int = Field(..., ge=1, le=65535)
    tags: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)

class ServiceRegistrationResponse(BaseModel):
    """Response from service registration"""
    service_id: str
    status: str
    message: str

class ServiceCallRequest(BaseModel):
    """Request to call a service"""
    service_name: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9-]+$")
    method: str = Field(default="GET")
    path: str = Field(default="/")
    headers: Dict[str, str] = Field(default_factory=dict)
    body: Optional[Any] = None
    timeout: float = Field(default=30.0, ge=1.0, le=300.0)
    retry_count: int = Field(default=3, ge=0, le=10)

class ServiceCallResponse(BaseModel):
    """Response from service call"""
    status_code: int
    headers: Dict[str, str]
    body: Any
    instance_id: str
    trace_id: str
    duration: float

class MeshTopologyResponse(BaseModel):
    """Service mesh topology information"""
    services: Dict[str, Any]
    total_instances: int
    healthy_instances: int
    circuit_breakers: Dict[str, str]

class EnqueueRequest(BaseModel):
    """Backward compatibility for existing enqueue API"""
    topic: str = Field(..., min_length=1, max_length=64, pattern=r"^[a-z0-9:_-]+$")
    task: Dict[str, Any]

class EnqueueResponse(BaseModel):
    """Response from enqueue operation"""
    id: str
    message: str = "Task enqueued through service mesh"

@router.post("/register", response_model=ServiceRegistrationResponse)
async def register_service(
    request: ServiceRegistrationRequest,
    background_tasks: BackgroundTasks
) -> ServiceRegistrationResponse:
    """Register a new service instance with the mesh"""
    try:
        mesh = await get_mesh()
        
        instance = await mesh.register_service(
            service_name=request.service_name,
            address=request.address,
            port=request.port,
            tags=request.tags,
            metadata=request.metadata
        )
        
        # Schedule health checks in background
        background_tasks.add_task(
            mesh.discovery.health_check,
            instance
        )
        
        return ServiceRegistrationResponse(
            service_id=instance.service_id,
            status="registered",
            message=f"Service {request.service_name} registered successfully"
        )
        
    except Exception as e:
        logger.error(f"Failed to register service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/deregister/{service_id}")
async def deregister_service(service_id: str) -> Dict[str, str]:
    """Deregister a service instance from the mesh"""
    try:
        mesh = await get_mesh()
        success = await mesh.discovery.deregister_service(service_id)
        
        if success:
            return {
                "status": "deregistered",
                "message": f"Service {service_id} deregistered successfully"
            }
        else:
            raise HTTPException(status_code=404, detail="Service not found")
            
    except Exception as e:
        logger.error(f"Failed to deregister service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/discover/{service_name}")
async def discover_service(
    service_name: str,
    use_cache: bool = Query(default=True)
) -> Dict[str, Any]:
    """Discover available instances of a service"""
    try:
        mesh = await get_mesh()
        instances = await mesh.discovery.discover_services(service_name, use_cache)
        
        return {
            "service_name": service_name,
            "instances": [
                {
                    "id": inst.service_id,
                    "address": f"{inst.address}:{inst.port}",
                    "state": inst.state.value,
                    "tags": inst.tags,
                    "metadata": inst.metadata
                }
                for inst in instances
            ],
            "total": len(instances),
            "healthy": len([i for i in instances if i.state == ServiceState.HEALTHY])
        }
        
    except Exception as e:
        logger.error(f"Failed to discover service: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/call", response_model=ServiceCallResponse)
async def call_service(request: ServiceCallRequest) -> ServiceCallResponse:
    """Call a service through the mesh with load balancing and circuit breaking"""
    try:
        mesh = await get_mesh()
        
        service_request = ServiceRequest(
            service_name=request.service_name,
            method=request.method,
            path=request.path,
            headers=request.headers,
            body=request.body,
            timeout=request.timeout,
            retry_count=request.retry_count
        )
        
        result = await mesh.call_service(service_request)
        
        return ServiceCallResponse(**result)
        
    except Exception as e:
        logger.error(f"Service call failed: {e}")
        raise HTTPException(status_code=503, detail=str(e))

@router.get("/topology", response_model=MeshTopologyResponse)
async def get_topology() -> MeshTopologyResponse:
    """Get current service mesh topology and health status"""
    try:
        mesh = await get_mesh()
        topology = await mesh.get_service_topology()
        
        return MeshTopologyResponse(**topology)
        
    except Exception as e:
        logger.error(f"Failed to get topology: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def mesh_health() -> Dict[str, Any]:
    """Service mesh health check"""
    try:
        mesh = await get_mesh()
        topology = await mesh.get_service_topology()
        
        # Calculate health score
        health_score = 1.0
        if topology["total_instances"] > 0:
            health_score = topology["healthy_instances"] / topology["total_instances"]
        
        # Count open circuit breakers
        open_breakers = sum(1 for state in topology["circuit_breakers"].values() if state == "open")
        
        status = "healthy"
        if health_score < 0.5:
            status = "unhealthy"
        elif health_score < 0.8 or open_breakers > 0:
            status = "degraded"
        
        return {
            "status": status,
            "health_score": health_score,
            "total_services": len(topology["services"]),
            "total_instances": topology["total_instances"],
            "healthy_instances": topology["healthy_instances"],
            "open_circuit_breakers": open_breakers,
            "consul_connected": mesh.discovery.consul_client is not None
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        }

@router.post("/configure/load-balancer")
async def configure_load_balancer(strategy: str = Query(..., pattern=r"^(round_robin|least_connections|weighted|random|ip_hash)$")) -> Dict[str, str]:
    """Configure load balancer strategy"""
    try:
        mesh = await get_mesh()
        mesh.load_balancer.strategy = LoadBalancerStrategy(strategy)
        
        return {
            "status": "configured",
            "strategy": strategy
        }
        
    except Exception as e:
        logger.error(f"Failed to configure load balancer: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/configure/circuit-breaker")
async def configure_circuit_breaker(
    failure_threshold: int = Query(default=5, ge=1, le=100),
    recovery_timeout: int = Query(default=60, ge=1, le=3600)
) -> Dict[str, Any]:
    """Configure circuit breaker parameters"""
    try:
        mesh = await get_mesh()
        mesh.circuit_breaker.failure_threshold = failure_threshold
        mesh.circuit_breaker.recovery_timeout = recovery_timeout
        
        return {
            "status": "configured",
            "failure_threshold": failure_threshold,
            "recovery_timeout": recovery_timeout
        }
        
    except Exception as e:
        logger.error(f"Failed to configure circuit breaker: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# Backward compatibility endpoints
@router.post("/enqueue", response_model=EnqueueResponse)
async def enqueue_task_compat(request: EnqueueRequest) -> EnqueueResponse:
    """Backward compatibility endpoint for task enqueueing"""
    try:
        task_id = await enqueue_task_mesh(request.topic, request.task)
        return EnqueueResponse(id=task_id)
        
    except Exception as e:
        logger.error(f"Failed to enqueue task: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/results")
async def get_results_compat(
    topic: str = Query(..., pattern=r"^[a-z0-9:_-]+$"),
    count: int = Query(10, ge=1, le=100)
) -> List[Dict[str, Any]]:
    """Backward compatibility endpoint for getting results"""
    try:
        items = await tail_results_mesh(topic, count)
        return [{"id": mid, "data": data} for (mid, data) in items]
        
    except Exception as e:
        logger.error(f"Failed to get results: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/metrics")
async def get_mesh_metrics() -> Dict[str, Any]:
    """Get service mesh metrics"""
    try:
        from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
        from fastapi.responses import Response
        
        metrics_data = generate_latest()
        return Response(content=metrics_data, media_type=CONTENT_TYPE_LATEST)
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))