"""
Service Mesh Module - Production-grade service mesh implementation

This module provides:
- Service discovery and registration
- Load balancing with multiple strategies
- Circuit breaking for fault tolerance
- Distributed tracing
- Real-time monitoring dashboard
- Kong API Gateway integration
- Consul service discovery

Version 2.0.0 - Complete rewrite from Redis queue to real service mesh
"""

from app.mesh.service_mesh import (
    ServiceMesh,
    ServiceInstance,
    ServiceState,
    ServiceRequest,
    LoadBalancerStrategy,
    ServiceDiscovery,
    LoadBalancer,
    CircuitBreakerManager,
    get_mesh
)

from app.mesh.distributed_tracing import (
    Tracer,
    Span,
    SpanContext,
    SpanType,
    SpanStatus,
    TraceCollector,
    TracingInterceptor,
    get_tracer
)

from app.mesh.mesh_dashboard import (
    MeshDashboard,
    ServiceMetrics,
    MeshMetrics,
    get_dashboard
)

# Backward compatibility
from app.mesh.redis_bus import (
    enqueue_task,
    tail_results,
    list_agents,
    register_agent,
    heartbeat_agent
)

__version__ = "2.0.0"

__all__ = [
    # Service Mesh Core
    "ServiceMesh",
    "ServiceInstance",
    "ServiceState",
    "ServiceRequest",
    "LoadBalancerStrategy",
    "ServiceDiscovery",
    "LoadBalancer",
    "CircuitBreakerManager",
    "get_mesh",
    
    # Distributed Tracing
    "Tracer",
    "Span",
    "SpanContext",
    "SpanType",
    "SpanStatus",
    "TraceCollector",
    "TracingInterceptor",
    "get_tracer",
    
    # Dashboard
    "MeshDashboard",
    "ServiceMetrics",
    "MeshMetrics",
    "get_dashboard",
    
    # Legacy/Compatibility
    "enqueue_task",
    "tail_results",
    "list_agents",
    "register_agent",
    "heartbeat_agent"
]