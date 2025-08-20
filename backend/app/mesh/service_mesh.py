"""
Production-Grade Service Mesh Implementation
Replaces fake Redis queue with real service discovery, load balancing, and circuit breaking
"""
from __future__ import annotations

import asyncio
import json
import time
import random
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple, Callable
from enum import Enum
from dataclasses import dataclass, field
import httpx
import consul
import pybreaker
from prometheus_client import Counter, Histogram, Gauge
import hashlib

logger = logging.getLogger(__name__)

# Metrics for monitoring
service_discovery_counter = Counter('mesh_service_discovery_total', 'Service discovery operations', ['operation', 'status'])
load_balancer_counter = Counter('mesh_load_balancer_requests', 'Load balancer request distribution', ['service', 'instance'])
circuit_breaker_counter = Counter('mesh_circuit_breaker_trips', 'Circuit breaker state changes', ['service', 'state'])
request_duration = Histogram('mesh_request_duration_seconds', 'Request duration by service', ['service', 'method'])
active_services = Gauge('mesh_active_services', 'Number of active services')
health_check_gauge = Gauge('mesh_health_check_status', 'Service health check status', ['service', 'instance'])

class ServiceState(Enum):
    """Service instance states"""
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

class LoadBalancerStrategy(Enum):
    """Load balancing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED = "weighted"
    RANDOM = "random"
    IP_HASH = "ip_hash"

@dataclass
class ServiceInstance:
    """Represents a service instance in the mesh"""
    service_id: str
    service_name: str
    address: str
    port: int
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    state: ServiceState = ServiceState.UNKNOWN
    weight: int = 100
    connections: int = 0
    last_health_check: Optional[float] = None
    health_check_failures: int = 0
    circuit_breaker_state: str = "closed"
    
    @property
    def url(self) -> str:
        return f"http://{self.address}:{self.port}"
    
    def to_consul_format(self) -> Dict[str, Any]:
        """Convert to Consul service format - compatible with python-consul 1.1.0"""
        # Note: python-consul 1.1.0 does not support 'meta' parameter
        # Metadata is stored in tags instead for compatibility
        consul_format = {
            "service_id": self.service_id,
            "name": self.service_name,
            "address": self.address,
            "port": self.port,
            "tags": self.tags.copy(),  # Create copy to avoid modifying original
            "check": {
                "http": f"{self.url}/health",
                "interval": "10s",
                "timeout": "5s",
                "deregister_critical_service_after": "1m"
            }
        }
        
        # Add metadata as tags (workaround for python-consul 1.1.0 compatibility)
        if self.metadata:
            for key, value in self.metadata.items():
                consul_format["tags"].append(f"meta_{key}={value}")
        
        return consul_format

@dataclass
class ServiceRequest:
    """Represents a request to a service"""
    service_name: str
    method: str
    path: str
    headers: Dict[str, str] = field(default_factory=dict)
    body: Optional[Any] = None
    timeout: float = 30.0
    retry_count: int = 3
    trace_id: Optional[str] = None
    
    def __post_init__(self):
        if not self.trace_id:
            self.trace_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()

class CircuitBreakerManager:
    """Manages circuit breakers for services"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60, expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        self.breakers: Dict[str, pybreaker.CircuitBreaker] = {}
    
    def get_breaker(self, service_id: str) -> pybreaker.CircuitBreaker:
        """Get or create circuit breaker for service"""
        if service_id not in self.breakers:
            self.breakers[service_id] = pybreaker.CircuitBreaker(
                fail_max=self.failure_threshold,
                reset_timeout=self.recovery_timeout,
                exclude=[self.expected_exception],
                name=f"breaker_{service_id}"
            )
        return self.breakers[service_id]
    
    def is_open(self, service_id: str) -> bool:
        """Check if circuit breaker is open"""
        breaker = self.get_breaker(service_id)
        return breaker.current_state == 'open'
    
    def record_success(self, service_id: str):
        """Record successful request"""
        breaker = self.get_breaker(service_id)
        # Use call() with a lambda that returns success
        try:
            breaker.call(lambda: True)
        except Exception:
            pass  # Success already recorded
    
    def record_failure(self, service_id: str):
        """Record failed request"""
        breaker = self.get_breaker(service_id)
        # Force a failure by calling with an exception
        try:
            breaker.call(lambda: (_ for _ in ()).throw(Exception("Simulated failure")))
        except Exception:
            pass  # Failure recorded
        circuit_breaker_counter.labels(service=service_id, state='open').inc()

class ServiceDiscovery:
    """Service discovery using Consul"""
    
    def __init__(self, consul_host: str = "localhost", consul_port: int = 10006):
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.consul_client = None
        self.services_cache: Dict[str, List[ServiceInstance]] = {}
        self.cache_ttl = 30  # seconds
        self.last_cache_update: Dict[str, float] = {}
        
    async def connect(self):
        """Connect to Consul with improved error detection"""
        try:
            self.consul_client = consul.Consul(
                host=self.consul_host,
                port=self.consul_port
            )
            # CRITICAL FIX #4: Test connection with timeout
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((self.consul_host, self.consul_port))
            sock.close()
            
            if result != 0:
                raise Exception(f"Cannot connect to Consul at {self.consul_host}:{self.consul_port}")
            
            # Test API call
            leader = self.consul_client.status.leader()
            logger.info(f"✅ Connected to Consul (leader: {leader})")
            service_discovery_counter.labels(operation='connect', status='success').inc()
            
        except Exception as e:
            logger.error(f"❌ Consul connection failed: {e}")
            logger.warning("Continuing in degraded mode without service discovery")
            self.consul_client = None
            service_discovery_counter.labels(operation='connect', status='failure').inc()
    
    async def register_service(self, instance: ServiceInstance) -> bool:
        """Register service with Consul"""
        try:
            if not self.consul_client:
                await self.connect()
            
            if not self.consul_client:
                logger.warning("Consul not available, using local cache only")
                # Store in local cache for degraded operation
                if instance.service_name not in self.services_cache:
                    self.services_cache[instance.service_name] = []
                self.services_cache[instance.service_name].append(instance)
                return True
            
            service_data = instance.to_consul_format()
            # Use synchronous call in async context
            self.consul_client.agent.service.register(**service_data)
            
            service_discovery_counter.labels(operation='register', status='success').inc()
            logger.info(f"Registered service {instance.service_id} with Consul")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register service {instance.service_id}: {e}")
            service_discovery_counter.labels(operation='register', status='failure').inc()
            # Fall back to local cache
            if instance.service_name not in self.services_cache:
                self.services_cache[instance.service_name] = []
            self.services_cache[instance.service_name].append(instance)
            return True  # Return True for degraded operation
    
    async def deregister_service(self, service_id: str) -> bool:
        """Deregister service from Consul"""
        try:
            if not self.consul_client:
                await self.connect()
            
            if not self.consul_client:
                # Remove from local cache
                for service_name, instances in self.services_cache.items():
                    self.services_cache[service_name] = [i for i in instances if i.service_id != service_id]
                return True
            
            self.consul_client.agent.service.deregister(service_id)
            service_discovery_counter.labels(operation='deregister', status='success').inc()
            logger.info(f"Deregistered service {service_id} from Consul")
            
            # Also remove from cache
            for service_name, instances in self.services_cache.items():
                self.services_cache[service_name] = [i for i in instances if i.service_id != service_id]
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            service_discovery_counter.labels(operation='deregister', status='failure').inc()
            # Still remove from cache
            for service_name, instances in self.services_cache.items():
                self.services_cache[service_name] = [i for i in instances if i.service_id != service_id]
            return True
    
    async def discover_services(self, service_name: str, use_cache: bool = True) -> List[ServiceInstance]:
        """Discover service instances"""
        try:
            # Check cache first
            if use_cache and service_name in self.services_cache:
                cache_age = time.time() - self.last_cache_update.get(service_name, 0)
                if cache_age < self.cache_ttl:
                    return self.services_cache[service_name]
            
            if not self.consul_client:
                await self.connect()
            
            if not self.consul_client:
                # Return cached data or empty list
                return self.services_cache.get(service_name, [])
            
            # Query Consul for service instances - synchronous call
            _, services = self.consul_client.health.service(service_name, passing=True)
            
            instances = []
            for service in services:
                instance = ServiceInstance(
                    service_id=service['Service']['ID'],
                    service_name=service['Service']['Service'],
                    address=service['Service']['Address'] or service['Node']['Address'],
                    port=service['Service']['Port'],
                    tags=service['Service'].get('Tags', []),
                    metadata=service['Service'].get('Meta', {}),
                    state=ServiceState.HEALTHY
                )
                instances.append(instance)
            
            # Update cache
            self.services_cache[service_name] = instances
            self.last_cache_update[service_name] = time.time()
            
            active_services.set(len(instances))
            service_discovery_counter.labels(operation='discover', status='success').inc()
            
            return instances
            
        except Exception as e:
            logger.error(f"Failed to discover services for {service_name}: {e}")
            service_discovery_counter.labels(operation='discover', status='failure').inc()
            
            # Return cached data if available
            if service_name in self.services_cache:
                return self.services_cache[service_name]
            return []
    
    async def health_check(self, instance: ServiceInstance) -> ServiceState:
        """Perform health check on service instance"""
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{instance.url}/health")
                
                if response.status_code == 200:
                    instance.state = ServiceState.HEALTHY
                    instance.health_check_failures = 0
                    health_check_gauge.labels(service=instance.service_name, instance=instance.service_id).set(1)
                elif 200 < response.status_code < 500:
                    instance.state = ServiceState.DEGRADED
                    health_check_gauge.labels(service=instance.service_name, instance=instance.service_id).set(0.5)
                else:
                    instance.state = ServiceState.UNHEALTHY
                    instance.health_check_failures += 1
                    health_check_gauge.labels(service=instance.service_name, instance=instance.service_id).set(0)
                
                instance.last_health_check = time.time()
                return instance.state
                
        except Exception as e:
            logger.warning(f"Health check failed for {instance.service_id}: {e}")
            instance.state = ServiceState.UNHEALTHY
            instance.health_check_failures += 1
            health_check_gauge.labels(service=instance.service_name, instance=instance.service_id).set(0)
            return ServiceState.UNHEALTHY

class LoadBalancer:
    """Load balancer for service instances"""
    
    def __init__(self, strategy: LoadBalancerStrategy = LoadBalancerStrategy.ROUND_ROBIN):
        self.strategy = strategy
        self.round_robin_counters: Dict[str, int] = {}
        
    def select_instance(
        self, 
        instances: List[ServiceInstance], 
        service_name: str,
        client_ip: Optional[str] = None
    ) -> Optional[ServiceInstance]:
        """Select service instance based on strategy"""
        
        # Filter healthy instances
        healthy_instances = [i for i in instances if i.state == ServiceState.HEALTHY]
        
        if not healthy_instances:
            # Fall back to degraded if no healthy instances
            healthy_instances = [i for i in instances if i.state == ServiceState.DEGRADED]
        
        if not healthy_instances:
            return None
        
        selected = None
        
        if self.strategy == LoadBalancerStrategy.ROUND_ROBIN:
            # Round-robin selection
            if service_name not in self.round_robin_counters:
                self.round_robin_counters[service_name] = 0
            
            index = self.round_robin_counters[service_name] % len(healthy_instances)
            selected = healthy_instances[index]
            self.round_robin_counters[service_name] += 1
            
        elif self.strategy == LoadBalancerStrategy.LEAST_CONNECTIONS:
            # Select instance with least connections
            selected = min(healthy_instances, key=lambda x: x.connections)
            
        elif self.strategy == LoadBalancerStrategy.WEIGHTED:
            # Weighted random selection
            weights = [i.weight for i in healthy_instances]
            selected = random.choices(healthy_instances, weights=weights)[0]
            
        elif self.strategy == LoadBalancerStrategy.RANDOM:
            # Random selection
            selected = random.choice(healthy_instances)
            
        elif self.strategy == LoadBalancerStrategy.IP_HASH:
            # Consistent hashing based on client IP
            if client_ip:
                hash_value = int(hashlib.md5(client_ip.encode()).hexdigest(), 16)
                index = hash_value % len(healthy_instances)
                selected = healthy_instances[index]
            else:
                selected = random.choice(healthy_instances)
        
        if selected:
            load_balancer_counter.labels(service=service_name, instance=selected.service_id).inc()
            selected.connections += 1
        
        return selected

class ServiceMesh:
    """Main service mesh orchestrator"""
    
    def __init__(
        self,
        consul_host: str = "localhost",
        consul_port: int = 10006,
        kong_admin_url: str = "http://localhost:10015",
        load_balancer_strategy: LoadBalancerStrategy = LoadBalancerStrategy.ROUND_ROBIN
    ):
        self.discovery = ServiceDiscovery(consul_host, consul_port)
        self.load_balancer = LoadBalancer(load_balancer_strategy)
        self.circuit_breaker = CircuitBreakerManager()
        self.kong_admin_url = kong_admin_url
        self.request_interceptors: List[Callable] = []
        self.response_interceptors: List[Callable] = []
        
    async def initialize(self):
        """Initialize service mesh components"""
        await self.discovery.connect()
        await self._configure_kong_routes()
        logger.info("Service mesh initialized")
    
    async def _configure_kong_routes(self):
        """Configure Kong API Gateway routes"""
        try:
            async with httpx.AsyncClient() as client:
                # Get existing services from Kong
                response = await client.get(f"{self.kong_admin_url}/services")
                if response.status_code == 200:
                    services = response.json().get('data', [])
                    logger.info(f"Kong has {len(services)} configured services")
        except Exception as e:
            logger.error(f"Failed to configure Kong routes: {e}")
    
    async def register_service(
        self,
        service_name: str,
        address: str,
        port: int,
        tags: List[str] = None,
        metadata: Dict[str, Any] = None
    ) -> ServiceInstance:
        """Register a new service instance"""
        instance = ServiceInstance(
            service_id=f"{service_name}-{address}-{port}",
            service_name=service_name,
            address=address,
            port=port,
            tags=tags or [],
            metadata=metadata or {}
        )
        
        # Register with Consul
        await self.discovery.register_service(instance)
        
        # Configure Kong upstream
        await self._configure_kong_upstream(service_name, instance)
        
        return instance
    
    async def _configure_kong_upstream(self, service_name: str, instance: ServiceInstance):
        """Configure Kong upstream for service"""
        try:
            async with httpx.AsyncClient() as client:
                # Create or update upstream
                upstream_data = {
                    "name": f"{service_name}-upstream",
                    "algorithm": "round-robin",
                    "healthchecks": {
                        "active": {
                            "type": "http",
                            "http_path": "/health",
                            "healthy": {
                                "interval": 5,
                                "successes": 1
                            },
                            "unhealthy": {
                                "interval": 10,
                                "http_failures": 3
                            }
                        }
                    }
                }
                
                response = await client.put(
                    f"{self.kong_admin_url}/upstreams/{service_name}-upstream",
                    json=upstream_data
                )
                
                if response.status_code in [200, 201]:
                    # Add target to upstream
                    target_data = {
                        "target": f"{instance.address}:{instance.port}",
                        "weight": instance.weight
                    }
                    
                    await client.post(
                        f"{self.kong_admin_url}/upstreams/{service_name}-upstream/targets",
                        json=target_data
                    )
                    
                    logger.info(f"Configured Kong upstream for {service_name}")
                    
        except Exception as e:
            logger.error(f"Failed to configure Kong upstream: {e}")
    
    async def call_service(self, request: ServiceRequest) -> Dict[str, Any]:
        """Call a service through the mesh"""
        start_time = time.time()
        
        # Add trace headers
        if not request.trace_id:
            request.trace_id = hashlib.md5(f"{time.time()}{random.random()}".encode()).hexdigest()
        
        request.headers['X-Trace-Id'] = request.trace_id
        request.headers['X-Request-Start'] = str(start_time)
        
        try:
            # Discover service instances
            instances = await self.discovery.discover_services(request.service_name)
            
            if not instances:
                raise Exception(f"No instances available for service {request.service_name}")
            
            # Select instance using load balancer
            instance = self.load_balancer.select_instance(instances, request.service_name)
            
            if not instance:
                raise Exception(f"No healthy instances for service {request.service_name}")
            
            # Check circuit breaker
            if self.circuit_breaker.is_open(instance.service_id):
                logger.warning(f"Circuit breaker open for {instance.service_id}")
                # Try another instance
                instances.remove(instance)
                instance = self.load_balancer.select_instance(instances, request.service_name)
                
                if not instance:
                    raise Exception(f"All instances have open circuit breakers")
            
            # Apply request interceptors
            for interceptor in self.request_interceptors:
                request = await interceptor(request)
            
            # Make the actual request
            async with httpx.AsyncClient(timeout=request.timeout) as client:
                url = f"{instance.url}{request.path}"
                
                response = await client.request(
                    method=request.method,
                    url=url,
                    headers=request.headers,
                    json=request.body if request.body else None
                )
                
                # Record success
                self.circuit_breaker.record_success(instance.service_id)
                instance.connections -= 1
                
                # Apply response interceptors
                result = {
                    "status_code": response.status_code,
                    "headers": dict(response.headers),
                    "body": response.json() if response.headers.get('content-type', '').startswith('application/json') else response.text,
                    "instance_id": instance.service_id,
                    "trace_id": request.trace_id,
                    "duration": time.time() - start_time
                }
                
                for interceptor in self.response_interceptors:
                    result = await interceptor(result)
                
                # Record metrics
                request_duration.labels(service=request.service_name, method=request.method).observe(result['duration'])
                
                return result
                
        except Exception as e:
            # Record failure
            if 'instance' in locals():
                self.circuit_breaker.record_failure(instance.service_id)
                instance.connections -= 1
            
            logger.error(f"Service call failed: {e}")
            
            # Retry logic with exponential backoff
            if request.retry_count > 0:
                request.retry_count -= 1
                # Calculate exponential backoff with jitter
                retry_attempt = request.__dict__.get('_retry_attempt', 0)
                request.__dict__['_retry_attempt'] = retry_attempt + 1
                backoff_time = min(2 ** retry_attempt + random.uniform(0, 1), 30)  # Max 30 seconds
                
                logger.info(f"Retrying request to {request.service_name} ({request.retry_count} retries left) after {backoff_time:.1f}s")
                await asyncio.sleep(backoff_time)
                return await self.call_service(request)
            
            raise
    
    async def get_service_topology(self) -> Dict[str, Any]:
        """Get current service mesh topology"""
        topology = {
            "services": {},
            "total_instances": 0,
            "healthy_instances": 0,
            "circuit_breakers": {}
        }
        
        # Get all registered services from cache
        for service_name, instances in self.discovery.services_cache.items():
            service_info = {
                "instances": [],
                "total": len(instances),
                "healthy": 0
            }
            
            for instance in instances:
                instance_info = {
                    "id": instance.service_id,
                    "address": f"{instance.address}:{instance.port}",
                    "state": instance.state.value,
                    "connections": instance.connections,
                    "circuit_breaker": self.circuit_breaker.is_open(instance.service_id)
                }
                
                service_info["instances"].append(instance_info)
                
                if instance.state == ServiceState.HEALTHY:
                    service_info["healthy"] += 1
                    topology["healthy_instances"] += 1
                
                topology["total_instances"] += 1
            
            topology["services"][service_name] = service_info
        
        # Add circuit breaker states
        for service_id, breaker in self.circuit_breaker.breakers.items():
            topology["circuit_breakers"][service_id] = breaker.current_state
        
        return topology
    
    def add_request_interceptor(self, interceptor: Callable):
        """Add request interceptor"""
        self.request_interceptors.append(interceptor)
    
    def add_response_interceptor(self, interceptor: Callable):
        """Add response interceptor"""
        self.response_interceptors.append(interceptor)
    
    # Compatibility methods for existing API endpoints
    async def register_service_v2(self, service_id: str, service_info: Dict[str, Any]) -> Dict[str, Any]:
        """Register service - compatibility wrapper for main.py endpoints"""
        instance = ServiceInstance(
            service_id=service_id,
            service_name=service_info.get("service_name", service_id),
            address=service_info.get("address", "localhost"),
            port=service_info.get("port", 8080),
            tags=service_info.get("tags", []),
            metadata=service_info.get("metadata", {})
        )
        
        success = await self.discovery.register_service(instance)
        if success:
            await self._configure_kong_upstream(instance.service_name, instance)
            return {"id": service_id, "status": "registered", "instance": instance.to_consul_format()}
        else:
            raise Exception("Failed to register service with Consul")
    
    async def discover_services(self, service_name: str = None) -> List[Dict[str, Any]]:
        """Discover services - compatibility wrapper for main.py endpoints"""
        if service_name:
            instances = await self.discovery.discover_services(service_name)
            return [
                {
                    "id": instance.service_id,
                    "name": instance.service_name,
                    "address": instance.address,
                    "port": instance.port,
                    "tags": instance.tags,
                    "metadata": instance.metadata,
                    "state": instance.state.value
                }
                for instance in instances
            ]
        else:
            # Query Consul for ALL services when no specific service_name is provided
            all_services = []
            try:
                if not self.discovery.consul_client:
                    await self.discovery.connect()
                
                if self.discovery.consul_client:
                    # Get all services from Consul
                    consul_services = self.discovery.consul_client.agent.services()
                    
                    for service_id, service_info in consul_services.items():
                        all_services.append({
                            "id": service_id,
                            "name": service_info.get('Service', service_id),
                            "address": service_info.get('Address', 'localhost'),
                            "port": service_info.get('Port', 0),
                            "tags": service_info.get('Tags', []),
                            "metadata": service_info.get('Meta', {}),
                            "state": "healthy"  # Assume healthy if in Consul
                        })
                    
                    logger.info(f"Discovered {len(all_services)} services from Consul")
                else:
                    # Fallback to cache if Consul is not available
                    logger.warning("Consul not available, returning cached services")
                    for service_name, instances in self.discovery.services_cache.items():
                        for instance in instances:
                            all_services.append({
                                "id": instance.service_id,
                                "name": instance.service_name,
                                "address": instance.address,
                                "port": instance.port,
                                "tags": instance.tags,
                                "metadata": instance.metadata,
                                "state": instance.state.value
                            })
            except Exception as e:
                logger.error(f"Failed to query Consul for all services: {e}")
                # Fallback to cache on error
                for service_name, instances in self.discovery.services_cache.items():
                    for instance in instances:
                        all_services.append({
                            "id": instance.service_id,
                            "name": instance.service_name,
                            "address": instance.address,
                            "port": instance.port,
                            "tags": instance.tags,
                            "metadata": instance.metadata,
                            "state": instance.state.value
                        })
            
            return all_services
    
    async def enqueue_task(self, task_type: str, payload: Dict[str, Any], priority: int = 0) -> str:
        """Enqueue task via service mesh - compatibility wrapper"""
        # Generate unique task ID
        import uuid
        task_id = str(uuid.uuid4())
        
        # Try to route to appropriate service based on task type
        request = ServiceRequest(
            service_name=task_type,
            method="POST",
            path="/process",
            body={
                "task_id": task_id,
                "payload": payload,
                "priority": priority
            }
        )
        
        try:
            result = await self.call_service(request)
            # Store task status for later retrieval
            self._task_statuses = getattr(self, '_task_statuses', {})
            self._task_statuses[task_id] = {
                "status": "completed",
                "result": result,
                "task_type": task_type
            }
            return task_id
        except Exception as e:
            # Store failure status
            self._task_statuses = getattr(self, '_task_statuses', {})
            self._task_statuses[task_id] = {
                "status": "failed",
                "error": str(e),
                "task_type": task_type
            }
            logger.error(f"Task {task_id} failed: {e}")
            return task_id
    
    async def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status - compatibility wrapper"""
        task_statuses = getattr(self, '_task_statuses', {})
        return task_statuses.get(task_id)
    
    async def health_check(self) -> Dict[str, Any]:
        """Service mesh health check - compatibility wrapper"""
        try:
            topology = await self.get_service_topology()
            
            # Count healthy services
            total_services = topology["total_instances"]
            healthy_services = topology["healthy_instances"]
            
            status = "healthy" if healthy_services > 0 else "degraded"
            if total_services == 0:
                status = "no_services"
            
            return {
                "status": status,
                "services": topology["services"],
                "queue_stats": {
                    "pending_tasks": len(getattr(self, '_task_statuses', {})),
                    "total_services": total_services,
                    "healthy_services": healthy_services
                },
                "consul_connected": self.discovery.consul_client is not None,
                "circuit_breakers": topology["circuit_breakers"]
            }
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "services": {},
                "queue_stats": {}
            }
            
    async def shutdown(self):
        """Shutdown service mesh"""
        # Clean up connections - Consul client doesn't need explicit close for synchronous version
        if self.discovery.consul_client:
            logger.info("Shutting down ServiceMesh connections")
        
        # Clear task statuses
        self._task_statuses = {}

# Global mesh instance
_mesh: Optional[ServiceMesh] = None

async def get_mesh() -> ServiceMesh:
    """Get or create service mesh instance"""
    global _mesh
    if _mesh is None:
        _mesh = ServiceMesh()
        await _mesh.initialize()
    return _mesh

async def get_service_mesh() -> ServiceMesh:
    """Get or create service mesh instance - compatibility alias"""
    return await get_mesh()

# Compatibility layer for existing code
async def enqueue_task_mesh(topic: str, payload: Dict[str, Any]) -> str:
    """Compatibility wrapper for existing enqueue_task calls"""
    mesh = await get_mesh()
    
    # Convert to service request
    request = ServiceRequest(
        service_name=topic,
        method="POST",
        path="/process",
        body=payload
    )
    
    try:
        result = await mesh.call_service(request)
        return result.get("trace_id", "unknown")
    except Exception as e:
        logger.error(f"Failed to enqueue task through mesh: {e}")
        # Fall back to Redis implementation if mesh fails
        from app.mesh.redis_bus import enqueue_task
        return enqueue_task(topic, payload)

async def tail_results_mesh(topic: str, count: int = 10) -> List[Tuple[str, Dict[str, Any]]]:
    """Compatibility wrapper for existing tail_results calls"""
    # For now, keep using Redis for results storage
    from app.mesh.redis_bus import tail_results
    return tail_results(topic, count)