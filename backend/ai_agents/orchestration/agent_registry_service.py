"""
Advanced Agent Registry and Discovery Service
===========================================

This module implements a comprehensive agent registry and discovery system with
real-time health monitoring, capability tracking, and intelligent service discovery.

Key Features:
- Dynamic agent registration and deregistration
- Real-time health monitoring
- Capability-based service discovery
- Load balancing and failover
- Service mesh integration
- Performance metrics collection
- Auto-scaling recommendations
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set, Callable
from app.orchestration.event_utils import register_event_handler as _reg_handler
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum
import httpx
import redis.asyncio as redis
from collections import defaultdict
import uuid
import hashlib

from .multi_agent_workflow_system import AgentProfile, AgentCapability
from ..protocols.enhanced_agent_communication import (
    EnhancedAgentCommunication, Message, MessageFactory,
    CommunicationPattern, MessagePriority
)

logger = logging.getLogger(__name__)


class ServiceStatus(Enum):
    """Service status enumeration"""
    INITIALIZING = "initializing"
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNREACHABLE = "unreachable"
    MAINTENANCE = "maintenance"
    SHUTDOWN = "shutdown"


class DiscoveryMethod(Enum):
    """Service discovery methods"""
    PASSIVE = "passive"          # Wait for registration
    ACTIVE = "active"           # Actively probe for services
    HYBRID = "hybrid"           # Combination of both
    DNS_BASED = "dns_based"     # DNS-based discovery
    CONSUL = "consul"           # Consul integration
    KUBERNETES = "kubernetes"   # K8s service discovery


@dataclass
class ServiceEndpoint:
    """Service endpoint information"""
    id: str
    host: str
    port: int
    protocol: str = "http"
    path: str = "/"
    health_check_path: str = "/health"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}{self.path}"
    
    @property
    def health_url(self) -> str:
        return f"{self.protocol}://{self.host}:{self.port}{self.health_check_path}"


@dataclass
class ServiceInstance:
    """Complete service instance information"""
    id: str
    name: str
    version: str
    agent_type: str
    capabilities: Set[AgentCapability]
    endpoints: List[ServiceEndpoint]
    status: ServiceStatus = ServiceStatus.INITIALIZING
    health_score: float = 100.0
    last_heartbeat: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metrics
    response_time_ms: float = 0.0
    success_rate: float = 1.0
    current_load: int = 0
    max_load: int = 100
    
    # Resource information
    resource_usage: Dict[str, float] = field(default_factory=dict)
    resource_limits: Dict[str, float] = field(default_factory=dict)
    
    # Registration info
    registered_at: datetime = field(default_factory=datetime.now)
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        data = asdict(self)
        data['capabilities'] = [cap.value for cap in self.capabilities]
        data['tags'] = list(self.tags)
        data['status'] = self.status.value
        data['registered_at'] = self.registered_at.isoformat()
        if self.last_heartbeat:
            data['last_heartbeat'] = self.last_heartbeat.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ServiceInstance":
        """Create from dictionary"""
        data['capabilities'] = {AgentCapability(cap) for cap in data.get('capabilities', [])}
        data['tags'] = set(data.get('tags', []))
        data['status'] = ServiceStatus(data.get('status', 'initializing'))
        data['registered_at'] = datetime.fromisoformat(data['registered_at'])
        if data.get('last_heartbeat'):
            data['last_heartbeat'] = datetime.fromisoformat(data['last_heartbeat'])
        
        # Convert endpoints
        endpoints_data = data.pop('endpoints', [])
        endpoints = [ServiceEndpoint(**ep) for ep in endpoints_data]
        data['endpoints'] = endpoints
        
        return cls(**data)


@dataclass
class HealthCheckConfig:
    """Health check configuration"""
    interval_seconds: int = 30
    timeout_seconds: int = 5
    retries: int = 3
    failure_threshold: int = 3
    success_threshold: int = 2
    enabled: bool = True


@dataclass
class DiscoveryQuery:
    """Service discovery query"""
    capabilities: Optional[Set[AgentCapability]] = None
    tags: Optional[Set[str]] = None
    agent_type: Optional[str] = None
    min_health_score: float = 0.0
    max_load: Optional[int] = None
    exclude_instances: Set[str] = field(default_factory=set)
    prefer_local: bool = False
    load_balance: bool = True


class AgentRegistryService:
    """
    Advanced agent registry and discovery service
    """
    
    def __init__(self, 
                 redis_url: str = "redis://redis:6379",
                 communication_system: Optional[EnhancedAgentCommunication] = None):
        self.redis_url = redis_url
        self.redis_client: Optional[redis.Redis] = None
        self.communication_system = communication_system
        
        # Service registry
        self.services: Dict[str, ServiceInstance] = {}
        self.service_index: Dict[str, Set[str]] = defaultdict(set)  # capability -> service_ids
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)      # tag -> service_ids
        
        # Health monitoring
        self.health_config = HealthCheckConfig()
        self.health_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Discovery configuration
        self.discovery_method = DiscoveryMethod.HYBRID
        self.discovery_config = {
            "registration_ttl": 300,  # 5 minutes
            "heartbeat_interval": 30,  # 30 seconds
            "cleanup_interval": 60,   # 1 minute
            "max_health_history": 100
        }
        
        # Load balancing
        self.load_balancer_algorithms = {
            "round_robin": self._round_robin_select,
            "least_connections": self._least_connections_select,
            "weighted_random": self._weighted_random_select,
            "health_based": self._health_based_select
        }
        self.default_lb_algorithm = "health_based"
        
        # Event handlers
        self.event_handlers: Dict[str, List[Callable]] = defaultdict(list)
        
        # Statistics
        self.stats = {
            "registrations": 0,
            "deregistrations": 0,
            "health_checks": 0,
            "discoveries": 0,
            "failures": 0
        }
        
        self.running = False
        self._background_tasks: List[asyncio.Task] = []
        self._round_robin_counters: Dict[str, int] = defaultdict(int)
    
    async def initialize(self):
        """Initialize the registry service"""
        logger.info("Initializing Agent Registry Service...")
        
        # Connect to Redis
        self.redis_client = await redis.from_url(self.redis_url)
        
        # Load existing registrations from Redis
        await self._load_persistent_state()
        
        # Start background services
        self._background_tasks = [
            asyncio.create_task(self._health_monitor()),
            asyncio.create_task(self._cleanup_service()),
            asyncio.create_task(self._metrics_collector()),
            asyncio.create_task(self._discovery_announcer()),
            asyncio.create_task(self._load_monitor())
        ]
        
        # Register event handlers
        if self.communication_system:
            self._register_communication_handlers()
        
        self.running = True
        logger.info("Agent Registry Service initialized")
    
    async def shutdown(self):
        """Shutdown the registry service"""
        logger.info("Shutting down Agent Registry Service...")
        self.running = False
        
        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()
        
        # Save state to Redis
        await self._save_persistent_state()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("Agent Registry Service shutdown complete")
    
    # ==================== Service Registration ====================
    
    async def register_service(self, service: ServiceInstance) -> bool:
        """Register a new service instance"""
        try:
            # Validate service
            if not await self._validate_service(service):
                logger.error(f"Service validation failed for {service.id}")
                return False
            
            # Check for existing registration
            if service.id in self.services:
                logger.info(f"Updating existing service registration: {service.id}")
            else:
                logger.info(f"Registering new service: {service.id}")
                self.stats["registrations"] += 1
            
            # Update service status
            service.last_heartbeat = datetime.now()
            service.status = ServiceStatus.HEALTHY
            
            # Store in registry
            self.services[service.id] = service
            
            # Update indexes
            await self._update_indexes(service)
            
            # Persist to Redis
            await self._persist_service(service)
            
            # Trigger events
            await self._trigger_event("service_registered", service)
            
            # Start health monitoring for new service
            if service.id not in self.health_history:
                asyncio.create_task(self._start_health_monitoring(service.id))
            
            logger.info(f"Successfully registered service {service.name} ({service.id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {service.id}: {e}")
            self.stats["failures"] += 1
            return False
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service instance"""
        try:
            if service_id not in self.services:
                logger.warning(f"Service {service_id} not found for deregistration")
                return False
            
            service = self.services[service_id]
            
            # Remove from indexes
            await self._remove_from_indexes(service)
            
            # Remove from registry
            del self.services[service_id]
            
            # Clean up health history
            if service_id in self.health_history:
                del self.health_history[service_id]
            
            # Remove from Redis
            await self.redis_client.hdel("services", service_id)
            
            # Trigger events
            await self._trigger_event("service_deregistered", service)
            
            self.stats["deregistrations"] += 1
            logger.info(f"Successfully deregistered service {service.name} ({service_id})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            self.stats["failures"] += 1
            return False
    
    async def update_service_health(self, service_id: str, health_data: Dict[str, Any]) -> bool:
        """Update service health information"""
        if service_id not in self.services:
            logger.warning(f"Service {service_id} not found for health update")
            return False
        
        service = self.services[service_id]
        service.last_heartbeat = datetime.now()
        
        # Update health metrics
        if "health_score" in health_data:
            service.health_score = max(0, min(100, health_data["health_score"]))
        
        if "response_time_ms" in health_data:
            service.response_time_ms = health_data["response_time_ms"]
        
        if "current_load" in health_data:
            service.current_load = health_data["current_load"]
        
        if "resource_usage" in health_data:
            service.resource_usage.update(health_data["resource_usage"])
        
        # Update status based on health score
        if service.health_score >= 80:
            service.status = ServiceStatus.HEALTHY
        elif service.health_score >= 50:
            service.status = ServiceStatus.DEGRADED
        else:
            service.status = ServiceStatus.UNHEALTHY
        
        # Store health history
        health_record = {
            "timestamp": datetime.now().isoformat(),
            "health_score": service.health_score,
            "response_time_ms": service.response_time_ms,
            "status": service.status.value
        }
        
        history = self.health_history[service_id]
        history.append(health_record)
        
        # Limit history size
        if len(history) > self.discovery_config["max_health_history"]:
            history.pop(0)
        
        # Persist updated service
        await self._persist_service(service)
        
        return True
    
    # ==================== Service Discovery ====================
    
    async def discover_services(self, query: DiscoveryQuery) -> List[ServiceInstance]:
        """Discover services based on query criteria"""
        try:
            self.stats["discoveries"] += 1
            
            # Find matching services
            candidates = await self._find_matching_services(query)
            
            # Filter by health and load requirements
            filtered_candidates = []
            for service in candidates:
                if (service.health_score >= query.min_health_score and
                    (query.max_load is None or service.current_load <= query.max_load) and
                    service.id not in query.exclude_instances):
                    filtered_candidates.append(service)
            
            # Apply load balancing if requested
            if query.load_balance and len(filtered_candidates) > 1:
                filtered_candidates = await self._apply_load_balancing(
                    filtered_candidates, self.default_lb_algorithm
                )
            
            # Sort by preference (local first if requested)
            if query.prefer_local:
                filtered_candidates = await self._sort_by_locality(filtered_candidates)
            
            logger.debug(f"Discovery query returned {len(filtered_candidates)} services")
            return filtered_candidates
            
        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            self.stats["failures"] += 1
            return []
    
    async def get_service(self, service_id: str) -> Optional[ServiceInstance]:
        """Get a specific service by ID"""
        return self.services.get(service_id)
    
    async def get_services_by_capability(self, capability: AgentCapability) -> List[ServiceInstance]:
        """Get all services with a specific capability"""
        service_ids = self.service_index.get(capability.value, set())
        return [self.services[sid] for sid in service_ids if sid in self.services]
    
    async def get_services_by_tag(self, tag: str) -> List[ServiceInstance]:
        """Get all services with a specific tag"""
        service_ids = self.tag_index.get(tag, set())
        return [self.services[sid] for sid in service_ids if sid in self.services]
    
    async def get_healthy_services(self, min_health_score: float = 80.0) -> List[ServiceInstance]:
        """Get all healthy services"""
        return [
            service for service in self.services.values()
            if service.health_score >= min_health_score and 
               service.status in [ServiceStatus.HEALTHY, ServiceStatus.DEGRADED]
        ]
    
    # ==================== Load Balancing ====================
    
    async def _apply_load_balancing(self, 
                                  services: List[ServiceInstance],
                                  algorithm: str) -> List[ServiceInstance]:
        """Apply load balancing algorithm"""
        if algorithm in self.load_balancer_algorithms:
            return await self.load_balancer_algorithms[algorithm](services)
        return services
    
    async def _round_robin_select(self, services: List[ServiceInstance]) -> List[ServiceInstance]:
        """Round-robin load balancing"""
        if not services:
            return []
        
        key = "round_robin"
        current = self._round_robin_counters[key]
        self._round_robin_counters[key] = (current + 1) % len(services)
        
        # Rotate the list
        return services[current:] + services[:current]
    
    async def _least_connections_select(self, services: List[ServiceInstance]) -> List[ServiceInstance]:
        """Least connections load balancing"""
        return sorted(services, key=lambda s: s.current_load)
    
    async def _weighted_random_select(self, services: List[ServiceInstance]) -> List[ServiceInstance]:
        """Weighted random selection based on health scores"""
        if not services:
            return []
        
        # Calculate weights (higher health = higher weight)
        weights = [s.health_score for s in services]
        total_weight = sum(weights)
        
        if total_weight == 0:
            return services
        
        # Normalize weights
        weights = [w / total_weight for w in weights]
        
        # Sort by weight (descending)
        weighted_services = list(zip(services, weights))
        weighted_services.sort(key=lambda x: x[1], reverse=True)
        
        return [s for s, _ in weighted_services]
    
    async def _health_based_select(self, services: List[ServiceInstance]) -> List[ServiceInstance]:
        """Health-based selection with load consideration"""
        def score_service(service: ServiceInstance) -> float:
            # Combine health score and load (inverse)
            load_factor = 1.0 - (service.current_load / max(service.max_load, 1))
            return (service.health_score / 100) * 0.7 + load_factor * 0.3
        
        return sorted(services, key=score_service, reverse=True)
    
    # ==================== Background Services ====================
    
    async def _health_monitor(self):
        """Background health monitoring service"""
        while self.running:
            try:
                for service_id in list(self.services.keys()):
                    await self._check_service_health(service_id)
                
                await asyncio.sleep(self.health_config.interval_seconds)
                
            except Exception as e:
                logger.error(f"Health monitor error: {e}")
    
    async def _check_service_health(self, service_id: str):
        """Check health of a specific service"""
        if service_id not in self.services:
            return
        
        service = self.services[service_id]
        
        try:
            self.stats["health_checks"] += 1
            
            # Check each endpoint
            health_results = []
            for endpoint in service.endpoints:
                result = await self._probe_endpoint_health(endpoint)
                health_results.append(result)
            
            # Calculate overall health
            if health_results:
                avg_health = sum(r["health_score"] for r in health_results) / len(health_results)
                avg_response_time = sum(r["response_time_ms"] for r in health_results) / len(health_results)
                
                # Update service health
                await self.update_service_health(service_id, {
                    "health_score": avg_health,
                    "response_time_ms": avg_response_time
                })
            
            # Check heartbeat timeout
            if service.last_heartbeat:
                time_since_heartbeat = datetime.now() - service.last_heartbeat
                if time_since_heartbeat > timedelta(seconds=self.discovery_config["registration_ttl"]):
                    logger.warning(f"Service {service_id} heartbeat timeout")
                    service.status = ServiceStatus.UNREACHABLE
            
        except Exception as e:
            logger.error(f"Health check failed for service {service_id}: {e}")
            service.status = ServiceStatus.UNREACHABLE
            await self._trigger_event("service_unhealthy", service)
    
    async def _probe_endpoint_health(self, endpoint: ServiceEndpoint) -> Dict[str, Any]:
        """Probe endpoint health"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    endpoint.health_url,
                    timeout=self.health_config.timeout_seconds
                )
                
                response_time = (asyncio.get_event_loop().time() - start_time) * 1000
                
                if response.status_code == 200:
                    health_score = 100.0
                    try:
                        data = response.json()
                        health_score = data.get("health_score", 100.0)
                    except (IOError, OSError, FileNotFoundError) as e:
                        # Suppressed exception (was bare except)
                        logger.debug(f"Suppressed exception: {e}")
                        pass
                    
                    return {
                        "healthy": True,
                        "health_score": health_score,
                        "response_time_ms": response_time,
                        "status_code": response.status_code
                    }
                else:
                    return {
                        "healthy": False,
                        "health_score": 0.0,
                        "response_time_ms": response_time,
                        "status_code": response.status_code
                    }
        
        except Exception as e:
            response_time = (asyncio.get_event_loop().time() - start_time) * 1000
            logger.debug(f"Health probe failed for {endpoint.health_url}: {e}")
            return {
                "healthy": False,
                "health_score": 0.0,
                "response_time_ms": response_time,
                "error": str(e)
            }
    
    async def _cleanup_service(self):
        """Clean up stale services"""
        while self.running:
            try:
                current_time = datetime.now()
                stale_services = []
                
                for service_id, service in self.services.items():
                    if service.status == ServiceStatus.UNREACHABLE:
                        # Remove unreachable services after timeout
                        if (not service.last_heartbeat or 
                            current_time - service.last_heartbeat > 
                            timedelta(seconds=self.discovery_config["registration_ttl"] * 2)):
                            stale_services.append(service_id)
                
                # Remove stale services
                for service_id in stale_services:
                    logger.info(f"Removing stale service: {service_id}")
                    await self.deregister_service(service_id)
                
                await asyncio.sleep(self.discovery_config["cleanup_interval"])
                
            except Exception as e:
                logger.error(f"Cleanup service error: {e}")
    
    async def _metrics_collector(self):
        """Collect and report metrics"""
        while self.running:
            try:
                # Calculate aggregate metrics
                total_services = len(self.services)
                healthy_services = len([
                    s for s in self.services.values() 
                    if s.status == ServiceStatus.HEALTHY
                ])
                
                avg_health_score = 0.0
                if self.services:
                    avg_health_score = sum(s.health_score for s in self.services.values()) / len(self.services)
                
                # Store metrics in Redis
                metrics = {
                    "total_services": total_services,
                    "healthy_services": healthy_services,
                    "avg_health_score": avg_health_score,
                    "registrations": self.stats["registrations"],
                    "deregistrations": self.stats["deregistrations"],
                    "health_checks": self.stats["health_checks"],
                    "discoveries": self.stats["discoveries"],
                    "failures": self.stats["failures"],
                    "timestamp": datetime.now().isoformat()
                }
                
                await self.redis_client.hset("registry:metrics", mapping=metrics)
                
                # Log summary
                logger.info(f"Registry metrics - Services: {total_services}, "
                          f"Healthy: {healthy_services}, Avg Health: {avg_health_score:.1f}")
                
                await asyncio.sleep(60)  # Collect metrics every minute
                
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")
    
    async def _discovery_announcer(self):
        """Announce registry capabilities"""
        while self.running:
            try:
                if self.communication_system:
                    # Broadcast service availability
                    announcement = MessageFactory.create_status_update(
                        "agent_registry",
                        "available",
                        {
                            "total_services": len(self.services),
                            "capabilities": list(set().union(*[
                                s.capabilities for s in self.services.values()
                            ])),
                            "discovery_methods": [method.value for method in DiscoveryMethod]
                        }
                    )
                    await self.communication_system.send_message(announcement)
                
                await asyncio.sleep(300)  # Announce every 5 minutes
                
            except Exception as e:
                logger.error(f"Discovery announcer error: {e}")
    
    async def _load_monitor(self):
        """Monitor system load and suggest optimizations"""
        while self.running:
            try:
                # Analyze load distribution
                high_load_services = [
                    s for s in self.services.values()
                    if s.current_load > s.max_load * 0.8
                ]
                
                if high_load_services:
                    logger.warning(f"High load detected on {len(high_load_services)} services")
                    
                    # Suggest scaling
                    for service in high_load_services:
                        await self._trigger_event("high_load_detected", service)
                
                await asyncio.sleep(60)  # Monitor every minute
                
            except Exception as e:
                logger.error(f"Load monitor error: {e}")
    
    # ==================== Helper Methods ====================
    
    async def _validate_service(self, service: ServiceInstance) -> bool:
        """Validate service registration"""
        if not service.id or not service.name:
            return False
        
        if not service.endpoints:
            return False
        
        # Validate endpoints
        for endpoint in service.endpoints:
            if not endpoint.host or endpoint.port <= 0:
                return False
        
        return True
    
    async def _update_indexes(self, service: ServiceInstance):
        """Update service indexes"""
        # Capability index
        for capability in service.capabilities:
            self.service_index[capability.value].add(service.id)
        
        # Tag index
        for tag in service.tags:
            self.tag_index[tag].add(service.id)
    
    async def _remove_from_indexes(self, service: ServiceInstance):
        """Remove service from indexes"""
        # Capability index
        for capability in service.capabilities:
            self.service_index[capability.value].discard(service.id)
        
        # Tag index
        for tag in service.tags:
            self.tag_index[tag].discard(service.id)
    
    async def _find_matching_services(self, query: DiscoveryQuery) -> List[ServiceInstance]:
        """Find services matching query criteria"""
        matching_service_ids = set()
        
        # Find by capabilities
        if query.capabilities:
            for capability in query.capabilities:
                service_ids = self.service_index.get(capability.value, set())
                if not matching_service_ids:
                    matching_service_ids = service_ids.copy()
                else:
                    matching_service_ids = matching_service_ids.intersection(service_ids)
        else:
            matching_service_ids = set(self.services.keys())
        
        # Filter by tags
        if query.tags:
            tag_matches = set()
            for tag in query.tags:
                tag_matches.update(self.tag_index.get(tag, set()))
            matching_service_ids = matching_service_ids.intersection(tag_matches)
        
        # Filter by agent type
        if query.agent_type:
            type_matches = {
                sid for sid, service in self.services.items()
                if service.agent_type == query.agent_type
            }
            matching_service_ids = matching_service_ids.intersection(type_matches)
        
        # Return matching services
        return [self.services[sid] for sid in matching_service_ids if sid in self.services]
    
    async def _sort_by_locality(self, services: List[ServiceInstance]) -> List[ServiceInstance]:
        """Sort services by locality preference"""
        # Simple locality sorting - can be enhanced with actual network topology
        local_services = []
        remote_services = []
        
        for service in services:
            # Check if service is local (same host/container network)
            is_local = any(
                endpoint.host in ["localhost", "127.0.0.1"] or 
                endpoint.host.startswith("sutazai-")
                for endpoint in service.endpoints
            )
            
            if is_local:
                local_services.append(service)
            else:
                remote_services.append(service)
        
        return local_services + remote_services
    
    async def _persist_service(self, service: ServiceInstance):
        """Persist service to Redis"""
        await self.redis_client.hset(
            "services",
            service.id,
            json.dumps(service.to_dict())
        )
    
    async def _load_persistent_state(self):
        """Load services from Redis"""
        try:
            services_data = await self.redis_client.hgetall("services")
            
            for service_id, service_json in services_data.items():
                try:
                    service_data = json.loads(service_json)
                    service = ServiceInstance.from_dict(service_data)
                    
                    # Mark as potentially stale
                    service.status = ServiceStatus.UNREACHABLE
                    
                    self.services[service_id] = service
                    await self._update_indexes(service)
                    
                except Exception as e:
                    logger.error(f"Failed to load service {service_id}: {e}")
            
            logger.info(f"Loaded {len(self.services)} services from persistent storage")
            
        except Exception as e:
            logger.error(f"Failed to load persistent state: {e}")
    
    async def _save_persistent_state(self):
        """Save current state to Redis"""
        try:
            for service in self.services.values():
                await self._persist_service(service)
            
            logger.info("Saved registry state to persistent storage")
            
        except Exception as e:
            logger.error(f"Failed to save persistent state: {e}")
    
    async def _trigger_event(self, event_type: str, service: ServiceInstance):
        """Trigger event handlers"""
        for handler in self.event_handlers.get(event_type, []):
            try:
                await handler(service)
            except Exception as e:
                logger.error(f"Event handler error for {event_type}: {e}")
    
    def register_event_handler(self, event_type: str, handler: Callable):
        """Register an event handler (canonical util)"""
        _reg_handler(self.event_handlers, event_type, handler)
    
    async def _start_health_monitoring(self, service_id: str):
        """Start health monitoring for a service"""
        # Health monitoring is handled by the main health monitor loop
        pass
    
    def _register_communication_handlers(self):
        """Register communication system handlers"""
        if not self.communication_system:
            return
        
        async def handle_service_discovery_request(message: Message):
            """Handle service discovery requests"""
            query_data = message.payload.get("query", {})
            
            # Convert to DiscoveryQuery
            query = DiscoveryQuery(
                capabilities={AgentCapability(cap) for cap in query_data.get("capabilities", [])},
                tags=set(query_data.get("tags", [])),
                agent_type=query_data.get("agent_type"),
                min_health_score=query_data.get("min_health_score", 0.0),
                max_load=query_data.get("max_load")
            )
            
            # Perform discovery
            services = await self.discover_services(query)
            
            # Send response
            response = Message(
                sender_id="agent_registry",
                recipient_id=message.sender_id,
                message_type="service_discovery_response",
                pattern=CommunicationPattern.POINT_TO_POINT,
                correlation_id=message.correlation_id,
                payload={
                    "services": [service.to_dict() for service in services],
                    "count": len(services)
                }
            )
            
            await self.communication_system.send_message(response)
        
        self.communication_system.register_handler(
            "service_discovery_request",
            handle_service_discovery_request
        )
    
    # ==================== Public API Methods ====================
    
    def get_registry_stats(self) -> Dict[str, Any]:
        """Get registry statistics"""
        return {
            "total_services": len(self.services),
            "healthy_services": len([
                s for s in self.services.values() 
                if s.status == ServiceStatus.HEALTHY
            ]),
            "capabilities_available": list(set().union(*[
                s.capabilities for s in self.services.values()
            ])) if self.services else [],
            "stats": self.stats.copy(),
            "discovery_method": self.discovery_method.value
        }
    
    def get_service_health_history(self, service_id: str) -> List[Dict[str, Any]]:
        """Get health history for a service"""
        return self.health_history.get(service_id, [])


# ==================== Factory Functions ====================

def create_registry_service(redis_url: str = "redis://redis:6379",
                           communication_system: Optional[EnhancedAgentCommunication] = None) -> AgentRegistryService:
    """Factory function to create registry service"""
    registry = AgentRegistryService(redis_url, communication_system)
    
    # Configure for SutazAI environment
    registry.discovery_config.update({
        "registration_ttl": 180,  # 3 minutes for faster updates
        "heartbeat_interval": 20,  # More frequent heartbeats
        "cleanup_interval": 45    # More frequent cleanup
    })
    
    return registry


# ==================== Utility Functions ====================

def create_service_instance_from_agent_profile(profile: AgentProfile) -> ServiceInstance:
    """Convert AgentProfile to ServiceInstance"""
    endpoint = ServiceEndpoint(
        id=f"{profile.id}_endpoint",
        host=profile.url.replace("http://", "").replace("https://", ""),
        port=profile.port,
        protocol="http"
    )
    
    return ServiceInstance(
        id=profile.id,
        name=profile.name,
        version="1.0.0",
        agent_type=profile.type,
        capabilities=profile.capabilities,
        endpoints=[endpoint],
        metadata=profile.metadata,
        resource_limits=profile.resource_requirements,
        tags={profile.type, "sutazai"}
    )


# ==================== Example Usage ====================

async def example_registry_usage():
    """Example of using the agent registry service"""
    
    # Initialize registry
    registry = create_registry_service()
    await registry.initialize()
    
    # Create a sample service
    endpoint = ServiceEndpoint(
        id="test_endpoint",
        host="localhost",
        port=8080
    )
    
    service = ServiceInstance(
        id="test_service",
        name="Test Code Generator",
        version="1.0.0",
        agent_type="code_generator",
        capabilities={AgentCapability.CODE_GENERATION, AgentCapability.CODE_ANALYSIS},
        endpoints=[endpoint],
        tags={"test", "development"}
    )
    
    # Register service
    success = await registry.register_service(service)
    logger.info(f"Registration successful: {success}")
    
    # Discover services
    query = DiscoveryQuery(
        capabilities={AgentCapability.CODE_GENERATION},
        min_health_score=50.0
    )
    
    discovered = await registry.discover_services(query)
    logger.info(f"Discovered {len(discovered)} services")
    
    # Update health
    await registry.update_service_health("test_service", {
        "health_score": 95.0,
        "response_time_ms": 150.0,
        "current_load": 3
    })
    
    # Get stats
    stats = registry.get_registry_stats()
    logger.info(f"Registry stats: {stats}")
    
    await registry.shutdown()


if __name__ == "__main__":
    asyncio.run(example_registry_usage())
