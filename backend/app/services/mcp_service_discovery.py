"""
MCP Service Discovery Service
Clean architecture implementation for discovering and managing MCP services
"""

import asyncio
import logging
from typing import Dict, List, Optional, Set, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import httpx
from enum import Enum

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    """MCP service status enumeration"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"
    DEPRECATED = "deprecated"
    MIGRATED = "migrated"

@dataclass
class ServiceInfo:
    """Information about an MCP service"""
    name: str
    url: str
    port: int
    status: ServiceStatus
    capabilities: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_health_check: Optional[datetime] = None
    health_check_count: int = 0
    error_count: int = 0
    replacement_service: Optional[str] = None

@dataclass
class DiscoveryConfig:
    """Configuration for service discovery"""
    health_check_interval: int = 30  # seconds
    health_check_timeout: float = 10.0  # seconds
    max_error_count: int = 3
    deprecated_services: Set[str] = field(default_factory=lambda: {
        "http", "puppeteer-mcp"
    })
    service_mappings: Dict[str, str] = field(default_factory=lambda: {
        "http": "http_fetch",
        "puppeteer-mcp": "playwright-mcp"
    })

class MCPServiceDiscovery:
    """Service for discovering and monitoring MCP services"""
    
    def __init__(self, config: DiscoveryConfig):
        self.config = config
        self.services: Dict[str, ServiceInfo] = {}
        self.running = False
        self._health_check_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start service discovery and health monitoring"""
        if self.running:
            logger.warning("Service discovery already running")
            return
        
        self.running = True
        logger.info("Starting MCP service discovery")
        
        # Initial service registration
        await self._register_known_services()
        
        # Start health check background task
        self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def stop(self):
        """Stop service discovery"""
        self.running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Stopped MCP service discovery")
    
    async def register_service(self, service_info: ServiceInfo):
        """Register a new MCP service"""
        service_name = service_info.name
        
        # Check if service is deprecated
        if service_name in self.config.deprecated_services:
            service_info.status = ServiceStatus.DEPRECATED
            if service_name in self.config.service_mappings:
                service_info.replacement_service = self.config.service_mappings[service_name]
            logger.warning(f"Registering deprecated service: {service_name}")
        
        self.services[service_name] = service_info
        logger.info(f"Registered MCP service: {service_name} at {service_info.url}")
        
        # Perform initial health check
        await self._check_service_health(service_info)
    
    async def get_service(self, name: str) -> Optional[ServiceInfo]:
        """Get service information by name"""
        service = self.services.get(name)
        
        # If requesting deprecated service, suggest replacement
        if service and service.status == ServiceStatus.DEPRECATED:
            logger.warning(f"Service {name} is deprecated. Use {service.replacement_service} instead")
        
        return service
    
    async def get_healthy_services(self) -> List[ServiceInfo]:
        """Get list of healthy services"""
        return [
            service for service in self.services.values()
            if service.status == ServiceStatus.HEALTHY
        ]
    
    async def get_services_by_capability(self, capability: str) -> List[ServiceInfo]:
        """Get services that have a specific capability"""
        return [
            service for service in self.services.values()
            if capability in service.capabilities and service.status == ServiceStatus.HEALTHY
        ]
    
    async def get_service_summary(self) -> Dict[str, Any]:
        """Get summary of all services"""
        total = len(self.services)
        healthy = len([s for s in self.services.values() if s.status == ServiceStatus.HEALTHY])
        unhealthy = len([s for s in self.services.values() if s.status == ServiceStatus.UNHEALTHY])
        deprecated = len([s for s in self.services.values() if s.status == ServiceStatus.DEPRECATED])
        
        return {
            "total_services": total,
            "healthy": healthy,
            "unhealthy": unhealthy,
            "deprecated": deprecated,
            "health_percentage": (healthy / total * 100) if total > 0 else 0,
            "services": {
                name: {
                    "status": service.status.value,
                    "url": service.url,
                    "capabilities": service.capabilities,
                    "last_check": service.last_health_check.isoformat() if service.last_health_check else None,
                    "replacement": service.replacement_service
                }
                for name, service in self.services.items()
            }
        }
    
    async def _register_known_services(self):
        """Register known MCP services"""
        known_services = [
            ServiceInfo(
                name="unified-memory",
                url="http://localhost:3009",
                port=3009,
                status=ServiceStatus.UNKNOWN,
                capabilities=["store", "retrieve", "search", "delete", "stats"],
                tags=["memory", "unified", "consolidated"],
                metadata={"version": "1.0.0", "consolidates": ["extended-memory", "memory-bank-mcp"]}
            ),
            ServiceInfo(
                name="postgres",
                url="http://localhost:11100",
                port=11100,
                status=ServiceStatus.UNKNOWN,
                capabilities=["sql", "transactions", "query"],
                tags=["database", "postgres"],
                metadata={"version": "1.0.0"}
            ),
            ServiceInfo(
                name="files",
                url="http://localhost:11101",
                port=11101,
                status=ServiceStatus.UNKNOWN,
                capabilities=["read", "write", "watch", "list"],
                tags=["filesystem", "io"],
                metadata={"version": "1.0.0"}
            ),
            ServiceInfo(
                name="context7",
                url="http://localhost:11102",
                port=11102,
                status=ServiceStatus.UNKNOWN,
                capabilities=["docs", "examples", "resolve"],
                tags=["context", "library", "documentation"],
                metadata={"version": "1.0.0"}
            ),
            ServiceInfo(
                name="ddg",
                url="http://localhost:11103",
                port=11103,
                status=ServiceStatus.UNKNOWN,
                capabilities=["search", "suggestions"],
                tags=["search", "web"],
                metadata={"version": "1.0.0"}
            ),
            # Deprecated services
            ServiceInfo(
                name="extended-memory",
                url="http://localhost:11104",
                port=11104,
                status=ServiceStatus.DEPRECATED,
                capabilities=["store", "retrieve", "search"],
                tags=["memory", "deprecated"],
                metadata={"version": "1.0.0", "deprecated": True},
                replacement_service="unified-memory"
            ),
            ServiceInfo(
                name="memory-bank-mcp",
                url="http://localhost:11105", 
                port=11105,
                status=ServiceStatus.DEPRECATED,
                capabilities=["store", "query"],
                tags=["memory", "deprecated"],
                metadata={"version": "1.0.0", "deprecated": True},
                replacement_service="unified-memory"
            )
        ]
        
        for service in known_services:
            await self.register_service(service)
    
    async def _health_check_loop(self):
        """Background task for periodic health checks"""
        while self.running:
            try:
                # Check all services
                tasks = [
                    self._check_service_health(service)
                    for service in self.services.values()
                    if service.status != ServiceStatus.DEPRECATED
                ]
                
                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)
                
                # Wait for next check
                await asyncio.sleep(self.config.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {str(e)}")
                await asyncio.sleep(5)  # Brief pause before retry
    
    async def _check_service_health(self, service: ServiceInfo):
        """Check health of a single service"""
        if service.status == ServiceStatus.DEPRECATED:
            return  # Skip health checks for deprecated services
        
        try:
            async with httpx.AsyncClient(timeout=self.config.health_check_timeout) as client:
                # Try health endpoint first
                try:
                    response = await client.get(f"{service.url}/health")
                    if response.status_code == 200:
                        service.status = ServiceStatus.HEALTHY
                        service.health_check_count += 1
                        service.last_health_check = datetime.now()
                        service.error_count = 0  # Reset error count on success
                        return
                except:
                    pass
                
                # Fallback: try root endpoint
                try:
                    response = await client.get(service.url)
                    if response.status_code in [200, 404]:  # 404 is ok, service is responding
                        service.status = ServiceStatus.HEALTHY
                        service.health_check_count += 1
                        service.last_health_check = datetime.now()
                        service.error_count = 0
                        return
                except:
                    pass
                
                # If we get here, service is not responding
                service.error_count += 1
                service.status = ServiceStatus.UNHEALTHY if service.error_count >= self.config.max_error_count else service.status
                
        except Exception as e:
            service.error_count += 1
            service.status = ServiceStatus.UNHEALTHY if service.error_count >= self.config.max_error_count else service.status
            logger.debug(f"Health check failed for {service.name}: {str(e)}")

# Global service discovery instance
_service_discovery: Optional[MCPServiceDiscovery] = None

async def get_service_discovery() -> MCPServiceDiscovery:
    """Get global service discovery instance"""
    global _service_discovery
    
    if _service_discovery is None:
        config = DiscoveryConfig()
        _service_discovery = MCPServiceDiscovery(config)
        await _service_discovery.start()
    
    return _service_discovery

async def shutdown_service_discovery():
    """Shutdown global service discovery"""
    global _service_discovery
    
    if _service_discovery:
        await _service_discovery.stop()
        _service_discovery = None