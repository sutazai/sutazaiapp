"""
Edge Inference API - REST API for edge inference optimization system
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import asyncio
from datetime import datetime

from . import (
    initialize_edge_inference_system,
    shutdown_edge_inference_system,
    get_system_status,
    get_inference_proxy,
    get_model_cache,
    get_batch_processor,
    get_quantization_manager,
    get_memory_manager,
    get_intelligent_router,
    get_telemetry_system,
    get_failover_manager,
    get_deployment_manager,
    InferenceRequest as EdgeInferenceRequest,
    BatchRequest,
    RequestPriority,
    DeploymentConfig,
    EdgePlatform,
    QuantizationConfig,
    QuantizationType,
    QuantizationStrategy
)

app = FastAPI(
    title="SutazAI Edge Inference API",
    description="Advanced edge inference optimization and management API",
    version="1.0.0"
)

# Request/Response Models
class InferenceRequest(BaseModel):
    """API model for inference requests"""
    request_id: str
    model_name: str
    prompt: str
    parameters: Dict[str, Any] = Field(default_factory=dict)
    priority: int = Field(default=3, ge=1, le=10)
    timeout: float = Field(default=30.0, gt=0)
    client_location: Optional[str] = None
    context_length: int = Field(default=2048, gt=0)
    requires_gpu: bool = False

class InferenceResponse(BaseModel):
    """API model for inference responses"""
    request_id: str
    response: str
    node_id: str
    processing_time: float
    queue_time: float
    cached: bool = False
    tokens_generated: int = 0

class BatchInferenceRequest(BaseModel):
    """API model for batch inference requests"""
    requests: List[InferenceRequest]
    batch_strategy: str = "adaptive"
    max_batch_size: int = Field(default=8, ge=1, le=32)
    batch_timeout_ms: float = Field(default=100.0, gt=0)

class NodeRegistration(BaseModel):
    """API model for node registration"""
    node_id: str
    endpoint: str
    capabilities: Dict[str, Any] = Field(default_factory=dict)
    cpu_cores: int = Field(default=1, gt=0)
    memory_gb: float = Field(default=1.0, gt=0)
    models_supported: List[str] = Field(default_factory=list)
    location: Optional[str] = None
    max_concurrent: int = Field(default=10, gt=0)

class ModelQuantizationRequest(BaseModel):
    """API model for model quantization requests"""
    model_path: str
    output_path: str
    quantization_type: str = "int8"
    strategy: str = "balanced"
    target_accuracy_loss: float = Field(default=0.05, ge=0.0, le=1.0)
    compression_target: float = Field(default=0.25, ge=0.1, le=1.0)

class EdgeDeploymentRequest(BaseModel):
    """API model for edge deployment requests"""
    name: str
    platform: str
    image: str = "edge-inference:latest"
    replicas: int = Field(default=1, ge=1)
    cpu_request: str = "100m"
    cpu_limit: str = "500m"
    memory_request: str = "128Mi" 
    memory_limit: str = "512Mi"
    environment: Dict[str, str] = Field(default_factory=dict)
    external_access: bool = False
    hostname: Optional[str] = None

# Global system state
_system_initialized = False

@app.on_event("startup")
async def startup_event():
    """Initialize the edge inference system on startup"""
    global _system_initialized
    try:
        config = {
            'proxy': {'enable_batching': True, 'enable_caching': True},
            'telemetry': {'enable_database': True, 'enable_system_monitoring': True},
            'memory_manager': {'max_model_memory_gb': 8.0},
            'model_cache': {'max_cache_size_gb': 4.0},
            'router': {'routing_objective': 'multi_objective'}
        }
        
        status = await initialize_edge_inference_system(config)
        _system_initialized = True
        print(f"Edge inference system initialized: {status}")
    except Exception as e:
        print(f"Failed to initialize edge inference system: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Shutdown the edge inference system"""
    global _system_initialized
    if _system_initialized:
        try:
            status = await shutdown_edge_inference_system()
            print(f"Edge inference system shutdown: {status}")
        except Exception as e:
            print(f"Error during shutdown: {e}")
        finally:
            _system_initialized = False

def require_initialized():
    """Dependency to ensure system is initialized"""
    if not _system_initialized:
        raise HTTPException(status_code=503, detail="Edge inference system not initialized")

# Health and Status Endpoints
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.get("/status")
async def get_status(deps: None = Depends(require_initialized)):
    """Get system status"""
    status = get_system_status()
    
    # Add runtime statistics
    try:
        telemetry = get_telemetry_system()
        system_metrics = telemetry.get_system_metrics()
        inference_metrics = telemetry.get_inference_metrics()
        
        status.update({
            "system_metrics": system_metrics,
            "inference_metrics": inference_metrics,
            "initialized": _system_initialized
        })
    except Exception as e:
        status["metrics_error"] = str(e)
    
    return status

# Inference Endpoints
@app.post("/inference", response_model=InferenceResponse)
async def process_inference(
    request: InferenceRequest, 
    deps: None = Depends(require_initialized)
):
    """Process a single inference request"""
    try:
        proxy = get_inference_proxy()
        
        # Convert API request to internal request
        edge_request = EdgeInferenceRequest(
            request_id=request.request_id,
            model_name=request.model_name,
            prompt=request.prompt,
            parameters=request.parameters,
            priority=request.priority,
            timeout=request.timeout,
            created_at=datetime.now(),
            client_location=request.client_location,
            context_length=request.context_length,
            requires_gpu=request.requires_gpu
        )
        
        # Process request
        result = await proxy.process_request(edge_request)
        
        if not result:
            raise HTTPException(status_code=500, detail="Inference processing failed")
        
        return InferenceResponse(
            request_id=result.request_id,
            response=result.response,
            node_id=result.node_id,
            processing_time=result.processing_time,
            queue_time=result.queue_time,
            cached=result.cached,
            tokens_generated=result.tokens_generated
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

@app.post("/inference/batch")
async def process_batch_inference(
    request: BatchInferenceRequest,
    deps: None = Depends(require_initialized)
):
    """Process batch inference requests"""
    try:
        batch_processor = get_batch_processor()
        
        results = []
        for req in request.requests:
            # Convert to batch request
            batch_req = BatchRequest(
                request_id=req.request_id,
                prompt=req.prompt,
                model_name=req.model_name,
                parameters=req.parameters,
                priority=RequestPriority(min(max(req.priority, 1), 5)),
                timeout=req.timeout,
                created_at=datetime.now(),
                client_id=None,
                expected_tokens=100
            )
            
            # Add to batch processor
            result_text = await batch_processor.add_request(batch_req)
            
            results.append({
                "request_id": req.request_id,
                "response": result_text,
                "processing_time": 0.1,  # Would be actual from batch result
                "batch_processed": True
            })
        
        return {"results": results, "batch_size": len(results)}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch inference failed: {str(e)}")

# Node Management Endpoints
@app.post("/nodes/register")
async def register_node(
    node: NodeRegistration,
    deps: None = Depends(require_initialized)
):
    """Register an edge node"""
    try:
        from .proxy import EdgeNode
        from .intelligent_router import RoutingNode, NodeStatus
        
        # Register with proxy
        proxy = get_inference_proxy()
        edge_node = EdgeNode(
            node_id=node.node_id,
            endpoint=node.endpoint,
            capabilities=node.capabilities,
            cpu_cores=node.cpu_cores,
            memory_gb=node.memory_gb,
            models_loaded=set(node.models_supported),
            location=node.location,
            max_concurrent=node.max_concurrent
        )
        proxy.register_node(edge_node)
        
        # Register with router
        router = get_intelligent_router()
        routing_node = RoutingNode(
            node_id=node.node_id,
            endpoint=node.endpoint,
            status=NodeStatus.HEALTHY,
            current_load=0.0,
            avg_latency_ms=100.0,
            error_rate=0.0,
            throughput_rps=10.0,
            memory_usage=0.0,
            cpu_usage=0.0,
            model_capabilities=set(node.models_supported),
            hardware_score=float(node.cpu_cores * node.memory_gb),
            cost_per_request=0.001,
            location=node.location
        )
        router.register_node(routing_node)
        
        # Register with failover manager
        failover = get_failover_manager()
        failover.register_node(node.node_id, {
            "endpoint": node.endpoint,
            "capabilities": node.capabilities
        })
        
        return {"message": f"Node {node.node_id} registered successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Node registration failed: {str(e)}")

@app.get("/nodes")
async def list_nodes(deps: None = Depends(require_initialized)):
    """List all registered nodes"""
    try:
        router = get_intelligent_router()
        node_stats = router.get_node_stats()
        return {"nodes": node_stats}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list nodes: {str(e)}")

@app.delete("/nodes/{node_id}")
async def unregister_node(node_id: str, deps: None = Depends(require_initialized)):
    """Unregister a node"""
    try:
        proxy = get_inference_proxy()
        router = get_intelligent_router()
        failover = get_failover_manager()
        
        proxy.unregister_node(node_id)
        router.unregister_node(node_id)
        failover.unregister_node(node_id)
        
        return {"message": f"Node {node_id} unregistered successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Node unregistration failed: {str(e)}")

# Model Management Endpoints
@app.post("/models/quantize")
async def quantize_model(
    request: ModelQuantizationRequest,
    background_tasks: BackgroundTasks,
    deps: None = Depends(require_initialized)
):
    """Quantize a model for edge deployment"""
    try:
        quantization_manager = get_quantization_manager()
        
        # Convert API request to internal config
        config = QuantizationConfig(
            quantization_type=QuantizationType(request.quantization_type),
            strategy=QuantizationStrategy(request.strategy),
            target_accuracy_loss=request.target_accuracy_loss,
            compression_target=request.compression_target
        )
        
        # Start quantization in background
        async def quantize_task():
            result = await quantization_manager.quantizer.quantize_model(
                request.model_path,
                request.output_path,
                config
            )
            return result
        
        background_tasks.add_task(quantize_task)
        
        return {"message": "Model quantization started", "output_path": request.output_path}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quantization failed: {str(e)}")

@app.get("/models/cache")
async def get_cached_models(deps: None = Depends(require_initialized)):
    """Get cached models information"""
    try:
        model_cache = get_model_cache()
        cached_models = model_cache.get_cached_models()
        
        model_info = []
        for model in cached_models:
            model_info.append({
                "model_id": model.model_id,
                "model_name": model.model_name,
                "size_mb": model.size_mb,
                "last_accessed": model.last_accessed.isoformat(),
                "access_count": model.access_count,
                "quantized": model.quantized,
                "compression_ratio": model.compression_ratio
            })
        
        cache_stats = model_cache.get_stats()
        
        return {
            "models": model_info,
            "cache_stats": {
                "total_entries": cache_stats.total_models,
                "total_size_mb": cache_stats.total_size_mb,
                "hit_ratio": cache_stats.hit_ratio,
                "disk_usage_mb": cache_stats.disk_usage_mb
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get cached models: {str(e)}")

# Deployment Endpoints
@app.post("/deploy")
async def deploy_edge_system(
    request: EdgeDeploymentRequest,
    deps: None = Depends(require_initialized)
):
    """Deploy edge inference system"""
    try:
        deployment_manager = get_deployment_manager()
        
        # Convert API request to deployment config
        config = DeploymentConfig(
            name=request.name,
            platform=EdgePlatform(request.platform),
            replicas=request.replicas,
            resources={
                "image": request.image,
                "cpu_request": request.cpu_request,
                "cpu_limit": request.cpu_limit,
                "memory_request": request.memory_request,
                "memory_limit": request.memory_limit
            },
            environment=request.environment,
            health_check={
                "http_path": "/health",
                "port": 8000,
                "initial_delay": 30,
                "period": 10
            },
            networking={
                "external_access": request.external_access,
                "hostname": request.hostname
            }
        )
        
        # Start deployment
        job_id = await deployment_manager.deploy(config)
        
        return {"job_id": job_id, "message": f"Deployment started for {request.name}"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Deployment failed: {str(e)}")

@app.get("/deploy/{job_id}")
async def get_deployment_status(job_id: str, deps: None = Depends(require_initialized)):
    """Get deployment status"""
    try:
        deployment_manager = get_deployment_manager()
        job = await deployment_manager.get_deployment_status(job_id)
        
        if not job:
            raise HTTPException(status_code=404, detail="Deployment job not found")
        
        return {
            "job_id": job.job_id,
            "status": job.status.value,
            "created_at": job.created_at.isoformat(),
            "updated_at": job.updated_at.isoformat(),
            "logs": job.logs,
            "error_message": job.error_message,
            "endpoints": job.deployed_endpoints
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get deployment status: {str(e)}")

# Analytics and Monitoring Endpoints
@app.get("/metrics")
async def get_metrics(deps: None = Depends(require_initialized)):
    """Get system metrics"""
    try:
        telemetry = get_telemetry_system()
        
        # Get performance summary
        performance = telemetry.get_performance_summary(time_window_minutes=60)
        
        # Get system metrics
        system_metrics = telemetry.get_system_metrics()
        
        # Get inference metrics
        inference_metrics = telemetry.get_inference_metrics()
        
        return {
            "performance_summary": {
                "total_requests": performance.total_requests,
                "successful_requests": performance.successful_requests,
                "failed_requests": performance.failed_requests,
                "avg_latency_ms": performance.avg_latency_ms,
                "p95_latency_ms": performance.p95_latency_ms,
                "throughput_rps": performance.throughput_rps,
                "error_rate": performance.error_rate
            },
            "system_metrics": system_metrics,
            "inference_metrics": inference_metrics
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

@app.get("/alerts")
async def get_alerts(deps: None = Depends(require_initialized)):
    """Get active alerts"""
    try:
        telemetry = get_telemetry_system()
        alerts = telemetry.get_active_alerts()
        
        alert_data = []
        for alert in alerts:
            alert_data.append({
                "alert_id": alert.alert_id,
                "name": alert.name,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp.isoformat(),
                "node_id": alert.node_id,
                "metric_name": alert.metric_name,
                "threshold_value": alert.threshold_value,
                "current_value": alert.current_value
            })
        
        return {"alerts": alert_data}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)