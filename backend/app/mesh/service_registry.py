"""
Service Registry - Actual implementation for registering real services
Ensures Rule 1 compliance by implementing real service discovery
"""
import os
import logging
from typing import Dict, List, Any
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

@dataclass
class ServiceDefinition:
    """Definition of a real service in the system"""
    name: str
    container_name: str
    internal_port: int
    external_port: int
    health_endpoint: str = "/health"
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

# Define ACTUAL services running in our system
REAL_SERVICES = [
    # Core Backend Services
    ServiceDefinition(
        name="backend-api",
        container_name="sutazai-backend",
        internal_port=8000,
        external_port=10010,
        health_endpoint="/health",
        tags=["api", "core", "fastapi"],
        metadata={"version": "2.0.0", "tier": "application"}
    ),
    
    # Frontend Service
    ServiceDefinition(
        name="frontend-ui",
        container_name="sutazai-frontend",
        internal_port=8501,
        external_port=10011,
        health_endpoint="/",
        tags=["ui", "streamlit", "frontend"],
        metadata={"version": "1.0.0", "tier": "presentation"}
    ),
    
    # AI Service
    ServiceDefinition(
        name="ollama-llm",
        container_name="sutazai-ollama",
        internal_port=11434,
        external_port=10104,
        health_endpoint="/api/version",
        tags=["ai", "llm", "ollama"],
        metadata={"model": "tinyllama", "tier": "ai"}
    ),
    
    # Database Services
    ServiceDefinition(
        name="postgres-db",
        container_name="sutazai-postgres",
        internal_port=5432,
        external_port=10000,
        health_endpoint=None,  # Use TCP health check
        tags=["database", "postgres", "sql"],
        metadata={"version": "16", "tier": "data"}
    ),
    
    ServiceDefinition(
        name="redis-cache",
        container_name="sutazai-redis",
        internal_port=6379,
        external_port=10001,
        health_endpoint=None,  # Use PING command
        tags=["cache", "redis", "nosql"],
        metadata={"version": "7.2", "tier": "cache"}
    ),
    
    ServiceDefinition(
        name="neo4j-graph",
        container_name="sutazai-neo4j",
        internal_port=7687,
        external_port=10002,
        health_endpoint=None,  # Use bolt protocol
        tags=["database", "graph", "neo4j"],
        metadata={"version": "5.17", "tier": "data"}
    ),
    
    # Vector Databases
    ServiceDefinition(
        name="chromadb-vector",
        container_name="sutazai-chromadb",
        internal_port=8000,
        external_port=10100,
        health_endpoint="/api/v1/heartbeat",
        tags=["vector", "database", "chromadb"],
        metadata={"tier": "vector"}
    ),
    
    ServiceDefinition(
        name="qdrant-vector",
        container_name="sutazai-qdrant",
        internal_port=6333,
        external_port=10101,
        health_endpoint="/health",
        tags=["vector", "database", "qdrant"],
        metadata={"tier": "vector"}
    ),
    
    ServiceDefinition(
        name="faiss-vector",
        container_name="sutazai-faiss",
        internal_port=8000,
        external_port=10103,
        health_endpoint="/health",
        tags=["vector", "database", "faiss"],
        metadata={"tier": "vector"}
    ),
    
    # Monitoring Services
    ServiceDefinition(
        name="prometheus-metrics",
        container_name="sutazai-prometheus",
        internal_port=9090,
        external_port=10200,
        health_endpoint="/-/healthy",
        tags=["monitoring", "metrics", "prometheus"],
        metadata={"tier": "monitoring"}
    ),
    
    ServiceDefinition(
        name="grafana-dashboards",
        container_name="sutazai-grafana",
        internal_port=3000,
        external_port=10201,
        health_endpoint="/api/health",
        tags=["monitoring", "visualization", "grafana"],
        metadata={"tier": "monitoring"}
    ),
    
    ServiceDefinition(
        name="loki-logs",
        container_name="sutazai-loki",
        internal_port=3100,
        external_port=10202,
        health_endpoint="/ready",
        tags=["monitoring", "logs", "loki"],
        metadata={"tier": "monitoring"}
    ),
    
    # Message Queue
    ServiceDefinition(
        name="rabbitmq-broker",
        container_name="sutazai-rabbitmq",
        internal_port=5672,
        external_port=10007,
        health_endpoint=None,  # Use management API
        tags=["messaging", "queue", "rabbitmq"],
        metadata={"tier": "messaging"}
    ),
    
    # API Gateway and Service Discovery (these ARE running!)
    ServiceDefinition(
        name="kong-gateway",
        container_name="sutazai-kong",
        internal_port=8000,
        external_port=10005,
        health_endpoint="/status",
        tags=["gateway", "proxy", "kong"],
        metadata={"tier": "infrastructure", "admin_port": 10015}
    ),
    
    ServiceDefinition(
        name="consul-discovery",
        container_name="sutazai-consul",
        internal_port=8500,
        external_port=10006,
        health_endpoint="/v1/agent/self",
        tags=["discovery", "consul", "registry"],
        metadata={"tier": "infrastructure"}
    ),
]

async def register_all_services(service_mesh):
    """
    Register all REAL running services with the service mesh
    This ensures Rule 1 compliance - Real Implementation Only
    """
    registered_count = 0
    failed_services = []
    
    # Detect if we're running in a container or on host
    is_container = os.path.exists("/.dockerenv")
    
    for service_def in REAL_SERVICES:
        if not service_def.enabled:
            logger.info(f"Skipping disabled service: {service_def.name}")
            continue
            
        try:
            # Use appropriate address based on environment
            # From host: use localhost with external port
            # From container: use container name with internal port
            if is_container:
                address = service_def.container_name
                port = service_def.internal_port
            else:
                address = "localhost"
                port = service_def.external_port
            
            # Register with the mesh using the actual ServiceMesh API
            instance = await service_mesh.register_service(
                service_name=service_def.name,
                address=address,
                port=port,
                tags=service_def.tags,
                metadata=service_def.metadata
            )
            
            logger.info(f"✅ Registered service: {service_def.name} at {address}:{port}")
            registered_count += 1
            
        except Exception as e:
            logger.error(f"❌ Failed to register {service_def.name}: {e}")
            failed_services.append(service_def.name)
    
    # Log summary
    logger.info(f"""
    Service Registration Summary:
    ============================
    Total Services: {len(REAL_SERVICES)}
    Successfully Registered: {registered_count}
    Failed: {len(failed_services)}
    """)
    
    if failed_services:
        logger.warning(f"Failed services: {', '.join(failed_services)}")
    
    return {
        "total": len(REAL_SERVICES),
        "registered": registered_count,
        "failed": failed_services
    }

def get_service_by_name(name: str) -> ServiceDefinition:
    """Get service definition by name"""
    for service in REAL_SERVICES:
        if service.name == name:
            return service
    return None

def get_services_by_tag(tag: str) -> List[ServiceDefinition]:
    """Get all services with a specific tag"""
    return [s for s in REAL_SERVICES if tag in s.tags]

def get_services_by_tier(tier: str) -> List[ServiceDefinition]:
    """Get all services in a specific tier"""
    return [s for s in REAL_SERVICES if s.metadata.get("tier") == tier]