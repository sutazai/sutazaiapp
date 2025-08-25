"""
Simplified Service Mesh Configuration
Removes unnecessary proxy layers and reduces latency
"""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ServiceConfig:
    """Configuration for a single service"""
    name: str
    host: str
    port: int
    health_check_path: str = "/health"
    timeout: float = 5.0  # Reduced from 30s
    retry_count: int = 3
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: int = 5  # Reduced from 60s

class SimplifiedServiceMesh:
    """
    Simplified service mesh that removes unnecessary proxy layers
    Direct service-to-service communication without Kong/Consul overhead
    """
    
    def __init__(self):
        self.services = self._initialize_services()
        self.direct_connections = {}
        
    def _initialize_services(self) -> Dict[str, ServiceConfig]:
        """Initialize service configurations with direct addresses"""
        return {
            # Core Services - Direct connections without proxy
            "postgres": ServiceConfig(
                name="postgres",
                host=os.getenv("POSTGRES_HOST", "postgres"),
                port=5432,
                health_check_path="/",
                timeout=5.0
            ),
            "redis": ServiceConfig(
                name="redis",
                host=os.getenv("REDIS_HOST", "redis"),
                port=6379,
                health_check_path="/",
                timeout=2.0  # Redis is fast
            ),
            "neo4j": ServiceConfig(
                name="neo4j",
                host=os.getenv("NEO4J_HOST", "neo4j"),
                port=7687,  # Bolt protocol
                health_check_path="/",
                timeout=5.0
            ),
            "rabbitmq": ServiceConfig(
                name="rabbitmq",
                host=os.getenv("RABBITMQ_HOST", "rabbitmq"),
                port=5672,
                health_check_path="/",
                timeout=5.0
            ),
            
            # Vector Databases - Direct connections
            "chromadb": ServiceConfig(
                name="chromadb",
                host=os.getenv("CHROMADB_HOST", "chromadb"),
                port=8000,
                health_check_path="/api/v1/heartbeat",
                timeout=5.0
            ),
            "qdrant": ServiceConfig(
                name="qdrant",
                host=os.getenv("QDRANT_HOST", "qdrant"),
                port=6333,
                health_check_path="/health",
                timeout=5.0
            ),
            "faiss": ServiceConfig(
                name="faiss",
                host=os.getenv("FAISS_HOST", "faiss"),
                port=8080,
                health_check_path="/health",
                timeout=5.0
            ),
            
            # AI Services - Direct connections
            "ollama": ServiceConfig(
                name="ollama",
                host=os.getenv("OLLAMA_HOST", "ollama"),
                port=11434,
                health_check_path="/api/tags",
                timeout=10.0  # AI models need more time
            ),
            
            # Monitoring - Direct connections
            "prometheus": ServiceConfig(
                name="prometheus",
                host=os.getenv("PROMETHEUS_HOST", "prometheus"),
                port=9090,
                health_check_path="/-/healthy",
                timeout=5.0
            ),
            "grafana": ServiceConfig(
                name="grafana",
                host=os.getenv("GRAFANA_HOST", "grafana"),
                port=3000,
                health_check_path="/api/health",
                timeout=5.0
            )
        }
    
    def get_service_url(self, service_name: str) -> str:
        """
        Get direct service URL without proxy layers
        This bypasses Kong and Consul for direct communication
        """
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        service = self.services[service_name]
        
        # Use container names for internal Docker network communication
        if self._is_docker_environment():
            return f"http://{service.host}:{service.port}"
        else:
            # Use localhost for local development
            return f"http://127.0.0.1:{self._get_exposed_port(service_name)}"
    
    def _is_docker_environment(self) -> bool:
        """Check if running inside Docker container"""
        return os.path.exists("/.dockerenv") or os.getenv("DOCKER_CONTAINER") == "true"
    
    def _get_exposed_port(self, service_name: str) -> int:
        """Get the exposed port for local development"""
        port_mapping = {
            "postgres": 10000,
            "redis": 10001,
            "neo4j": 10003,  # Bolt port
            "rabbitmq": 10007,
            "chromadb": 10100,
            "qdrant": 10101,
            "faiss": 10103,
            "ollama": 10104,
            "prometheus": 10200,
            "grafana": 10201
        }
        return port_mapping.get(service_name, self.services[service_name].port)
    
    def get_connection_config(self, service_name: str) -> Dict[str, Any]:
        """Get optimized connection configuration for a service"""
        if service_name not in self.services:
            raise ValueError(f"Unknown service: {service_name}")
        
        service = self.services[service_name]
        
        return {
            "url": self.get_service_url(service_name),
            "timeout": service.timeout,
            "retry_count": service.retry_count,
            "circuit_breaker": {
                "failure_threshold": service.circuit_breaker_threshold,
                "recovery_timeout": service.circuit_breaker_timeout
            },
            # Connection pooling settings for better performance
            "pool_settings": {
                "max_connections": 20,
                "min_connections": 5,
                "connection_timeout": 2.0,
                "idle_timeout": 300.0,
                "max_lifetime": 3600.0
            }
        }
    
    async def health_check(self, service_name: str) -> bool:
        """Direct health check without proxy overhead"""
        import httpx
        
        if service_name not in self.services:
            return False
        
        service = self.services[service_name]
        url = f"{self.get_service_url(service_name)}{service.health_check_path}"
        
        try:
            async with httpx.AsyncClient(timeout=service.timeout) as client:
                response = await client.get(url)
                return response.status_code < 400
        except Exception as e:
            logger.warning(f"Health check failed for {service_name}: {e}")
            return False
    
    def get_database_url(self, db_type: str = "postgres") -> str:
        """Get database connection URL with optimized settings"""
        if db_type == "postgres":
            service = self.services["postgres"]
            user = os.getenv("POSTGRES_USER", "sutazai")
            password = os.getenv("POSTGRES_PASSWORD", "sutazai123")
            database = os.getenv("POSTGRES_DB", "sutazai")
            
            if self._is_docker_environment():
                host = service.host
                port = service.port
            else:
                host = "127.0.0.1"
                port = self._get_exposed_port("postgres")
            
            # Optimized connection parameters
            return (
                f"postgresql+asyncpg://{user}:{password}@{host}:{port}/{database}"
                f"?server_settings=jit%3Doff"  # Disable JIT for consistent performance
                f"&pool_size=20"  # Connection pool size
                f"&max_overflow=10"  # Maximum overflow connections
                f"&pool_pre_ping=true"  # Verify connections before use
                f"&pool_recycle=3600"  # Recycle connections after 1 hour
            )
        
        elif db_type == "redis":
            service = self.services["redis"]
            if self._is_docker_environment():
                return f"redis://{service.host}:{service.port}/0"
            else:
                return f"redis://127.0.0.1:{self._get_exposed_port('redis')}/0"
        
        elif db_type == "neo4j":
            service = self.services["neo4j"]
            user = os.getenv("NEO4J_USER", "neo4j")
            password = os.getenv("NEO4J_PASSWORD", "neo4j123")
            
            if self._is_docker_environment():
                return f"bolt://{user}:{password}@{service.host}:{service.port}"
            else:
                return f"bolt://{user}:{password}@127.0.0.1:{self._get_exposed_port('neo4j')}"
        
        else:
            raise ValueError(f"Unknown database type: {db_type}")

# Global instance
simplified_mesh = SimplifiedServiceMesh()

# Export convenience functions
def get_service_url(service_name: str) -> str:
    """Get direct service URL"""
    return simplified_mesh.get_service_url(service_name)

def get_database_url(db_type: str = "postgres") -> str:
    """Get optimized database URL"""
    return simplified_mesh.get_database_url(db_type)

def get_connection_config(service_name: str) -> Dict[str, Any]:
    """Get optimized connection configuration"""
    return simplified_mesh.get_connection_config(service_name)