#!/usr/bin/env python3
"""
SutazAI v9 Inter-Container Communication System
Manages communication between isolated AI agent containers and core services
"""

import asyncio
import aioredis
import aiohttp
import json
import logging
import time
import uuid
from typing import Dict, Any, List, Optional, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import consul
import consul.aio
from concurrent.futures import ThreadPoolExecutor
import threading
import weakref
import inspect
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MessageType(Enum):
    REQUEST = "request"
    RESPONSE = "response"
    EVENT = "event"
    HEARTBEAT = "heartbeat"
    SHUTDOWN = "shutdown"

class ServiceStatus(Enum):
    STARTING = "starting"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    STOPPING = "stopping"
    STOPPED = "stopped"

@dataclass
class ServiceInfo:
    """Information about a registered service"""
    id: str
    name: str
    type: str
    host: str
    port: int
    status: ServiceStatus
    capabilities: List[str]
    health_endpoint: str
    metadata: Dict[str, Any]
    last_heartbeat: float
    registered_at: float

@dataclass
class Message:
    """Inter-service message structure"""
    id: str
    type: MessageType
    sender: str
    recipient: str
    method: str
    payload: Dict[str, Any]
    timestamp: float
    correlation_id: Optional[str] = None
    timeout: int = 30
    priority: int = 5

    def __post_init__(self):
        if not self.id:
            self.id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = time.time()

class CircuitBreaker:
    """Circuit breaker for fault tolerance"""
    
    def __init__(self, failure_threshold: int = 5, recovery_timeout: int = 60):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failure_count = 0
        self.last_failure_time = None
        self.state = "CLOSED"  # CLOSED, OPEN, HALF_OPEN
        self.lock = threading.Lock()

    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker protection"""
        with self.lock:
            if self.state == "OPEN":
                if time.time() - self.last_failure_time > self.recovery_timeout:
                    self.state = "HALF_OPEN"
                else:
                    raise Exception("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                if self.state == "HALF_OPEN":
                    self.reset()
                return result
            except Exception as e:
                self.record_failure()
                raise e

    def record_failure(self):
        """Record a failure"""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = "OPEN"
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")

    def reset(self):
        """Reset circuit breaker"""
        self.failure_count = 0
        self.state = "CLOSED"
        self.last_failure_time = None
        logger.info("Circuit breaker reset to CLOSED state")

class ServiceRegistry:
    """Service discovery and registration using Consul"""
    
    def __init__(self, consul_host: str = "consul", consul_port: int = 8500):
        self.consul_host = consul_host
        self.consul_port = consul_port
        self.consul_client = None
        self.services = {}
        self.callbacks = {}

    async def initialize(self):
        """Initialize Consul connection"""
        try:
            self.consul_client = consul.aio.Consul(host=self.consul_host, port=self.consul_port)
            logger.info(f"Connected to Consul at {self.consul_host}:{self.consul_port}")
        except Exception as e:
            logger.error(f"Failed to connect to Consul: {e}")
            # Fallback to in-memory registry
            self.consul_client = None

    async def register_service(self, service_info: ServiceInfo) -> bool:
        """Register a service"""
        try:
            if self.consul_client:
                await self.consul_client.agent.service.register(
                    name=service_info.name,
                    service_id=service_info.id,
                    address=service_info.host,
                    port=service_info.port,
                    tags=service_info.capabilities,
                    check=consul.Check.http(
                        f"http://{service_info.host}:{service_info.port}{service_info.health_endpoint}",
                        interval="30s"
                    ),
                    meta=service_info.metadata
                )
            
            # Store in local registry
            self.services[service_info.id] = service_info
            logger.info(f"Registered service: {service_info.name} ({service_info.id})")
            
            # Notify callbacks
            await self._notify_callbacks("service_registered", service_info)
            
            return True
        except Exception as e:
            logger.error(f"Failed to register service {service_info.name}: {e}")
            return False

    async def deregister_service(self, service_id: str) -> bool:
        """Deregister a service"""
        try:
            if self.consul_client:
                await self.consul_client.agent.service.deregister(service_id)
            
            service_info = self.services.pop(service_id, None)
            if service_info:
                logger.info(f"Deregistered service: {service_info.name} ({service_id})")
                await self._notify_callbacks("service_deregistered", service_info)
            
            return True
        except Exception as e:
            logger.error(f"Failed to deregister service {service_id}: {e}")
            return False

    async def discover_services(self, service_name: str = None) -> List[ServiceInfo]:
        """Discover available services"""
        try:
            if self.consul_client:
                if service_name:
                    services = await self.consul_client.health.service(service_name, passing=True)
                    return [self._consul_to_service_info(s) for s in services[1]]
                else:
                    services = await self.consul_client.catalog.services()
                    all_services = []
                    for name in services[1]:
                        service_list = await self.consul_client.health.service(name, passing=True)
                        all_services.extend([self._consul_to_service_info(s) for s in service_list[1]])
                    return all_services
            else:
                # Fallback to local registry
                if service_name:
                    return [s for s in self.services.values() if s.name == service_name]
                else:
                    return list(self.services.values())
        
        except Exception as e:
            logger.error(f"Service discovery failed: {e}")
            return []

    async def get_service_by_capability(self, capability: str) -> List[ServiceInfo]:
        """Find services with specific capability"""
        services = await self.discover_services()
        return [s for s in services if capability in s.capabilities]

    async def update_service_status(self, service_id: str, status: ServiceStatus):
        """Update service status"""
        if service_id in self.services:
            self.services[service_id].status = status
            self.services[service_id].last_heartbeat = time.time()

    def add_callback(self, event_type: str, callback: Callable):
        """Add callback for service events"""
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        self.callbacks[event_type].append(callback)

    async def _notify_callbacks(self, event_type: str, service_info: ServiceInfo):
        """Notify event callbacks"""
        callbacks = self.callbacks.get(event_type, [])
        for callback in callbacks:
            try:
                if inspect.iscoroutinefunction(callback):
                    await callback(service_info)
                else:
                    callback(service_info)
            except Exception as e:
                logger.error(f"Callback error: {e}")

    def _consul_to_service_info(self, consul_service: Dict) -> ServiceInfo:
        """Convert Consul service to ServiceInfo"""
        service = consul_service.get('Service', {})
        return ServiceInfo(
            id=service.get('ID'),
            name=service.get('Service'),
            type=service.get('Meta', {}).get('type', 'unknown'),
            host=service.get('Address'),
            port=service.get('Port'),
            status=ServiceStatus.READY,
            capabilities=service.get('Tags', []),
            health_endpoint='/health',
            metadata=service.get('Meta', {}),
            last_heartbeat=time.time(),
            registered_at=time.time()
        )

class MessageBroker:
    """Redis-based message broker for inter-service communication"""
    
    def __init__(self, redis_url: str = "redis://redis-primary:6379"):
        self.redis_url = redis_url
        self.redis_pool = None
        self.subscriptions = {}
        self.message_handlers = {}
        self.circuit_breakers = {}

    async def initialize(self):
        """Initialize Redis connection"""
        try:
            self.redis_pool = await aioredis.create_redis_pool(
                self.redis_url,
                minsize=5,
                maxsize=20,
                encoding='utf-8'
            )
            logger.info(f"Connected to Redis at {self.redis_url}")
        except Exception as e:
            logger.error(f"Failed to connect to Redis: {e}")
            raise

    async def publish_message(self, channel: str, message: Message) -> bool:
        """Publish message to a channel"""
        try:
            message_json = json.dumps(asdict(message))
            await self.redis_pool.publish(channel, message_json)
            logger.debug(f"Published message {message.id} to {channel}")
            return True
        except Exception as e:
            logger.error(f"Failed to publish message: {e}")
            return False

    async def subscribe_to_channel(self, channel: str, handler: Callable[[Message], Any]):
        """Subscribe to a channel"""
        try:
            if channel not in self.subscriptions:
                subscription = await self.redis_pool.subscribe(channel)
                self.subscriptions[channel] = subscription
                
                # Start listening task
                asyncio.create_task(self._listen_to_channel(channel, handler))
                
            self.message_handlers[channel] = handler
            logger.info(f"Subscribed to channel: {channel}")
        except Exception as e:
            logger.error(f"Failed to subscribe to {channel}: {e}")

    async def _listen_to_channel(self, channel: str, handler: Callable):
        """Listen to channel messages"""
        subscription = self.subscriptions[channel]
        
        async for message in subscription[0].iter():
            try:
                message_data = json.loads(message.decode())
                message_obj = Message(**message_data)
                
                # Handle message with circuit breaker
                circuit_breaker = self._get_circuit_breaker(channel)
                await circuit_breaker.call(handler, message_obj)
                
            except Exception as e:
                logger.error(f"Error handling message on {channel}: {e}")

    def _get_circuit_breaker(self, channel: str) -> CircuitBreaker:
        """Get or create circuit breaker for channel"""
        if channel not in self.circuit_breakers:
            self.circuit_breakers[channel] = CircuitBreaker()
        return self.circuit_breakers[channel]

    async def send_request(self, 
                          service_name: str, 
                          method: str, 
                          params: Dict[str, Any],
                          timeout: int = 30) -> Any:
        """Send request to a service and wait for response"""
        
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.REQUEST,
            sender="communicator",
            recipient=service_name,
            method=method,
            payload=params,
            timeout=timeout
        )
        
        # Create response channel
        response_channel = f"response:{message.id}"
        response_future = asyncio.Future()
        
        async def response_handler(response_msg: Message):
            if response_msg.correlation_id == message.id:
                response_future.set_result(response_msg.payload)
        
        # Subscribe to response channel
        await self.subscribe_to_channel(response_channel, response_handler)
        
        try:
            # Send request
            request_channel = f"service:{service_name}:requests"
            await self.publish_message(request_channel, message)
            
            # Wait for response
            result = await asyncio.wait_for(response_future, timeout=timeout)
            return result
            
        except asyncio.TimeoutError:
            logger.error(f"Request {message.id} timed out")
            raise
        finally:
            # Cleanup subscription
            if response_channel in self.subscriptions:
                await self.subscriptions[response_channel].unsubscribe()
                del self.subscriptions[response_channel]

    async def close(self):
        """Close Redis connections"""
        if self.redis_pool:
            self.redis_pool.close()
            await self.redis_pool.wait_closed()

class ContainerCommunicator:
    """Main container communication orchestrator"""
    
    def __init__(self, 
                 service_name: str,
                 service_type: str = "service",
                 host: str = "localhost",
                 port: int = 8000,
                 capabilities: List[str] = None,
                 redis_url: str = "redis://redis-primary:6379",
                 consul_host: str = "consul"):
        
        self.service_name = service_name
        self.service_type = service_type
        self.host = host
        self.port = port
        self.capabilities = capabilities or []
        
        # Components
        self.service_registry = ServiceRegistry(consul_host=consul_host)
        self.message_broker = MessageBroker(redis_url=redis_url)
        
        # Service info
        self.service_info = ServiceInfo(
            id=f"{service_name}-{uuid.uuid4().hex[:8]}",
            name=service_name,
            type=service_type,
            host=host,
            port=port,
            status=ServiceStatus.STARTING,
            capabilities=self.capabilities,
            health_endpoint="/health",
            metadata={"version": "9.0.0"},
            last_heartbeat=time.time(),
            registered_at=time.time()
        )
        
        # Request handlers
        self.request_handlers = {}
        
        # Background tasks
        self.background_tasks = set()

    async def initialize(self):
        """Initialize communication system"""
        try:
            # Initialize components
            await self.service_registry.initialize()
            await self.message_broker.initialize()
            
            # Register service
            await self.service_registry.register_service(self.service_info)
            
            # Subscribe to service requests
            request_channel = f"service:{self.service_name}:requests"
            await self.message_broker.subscribe_to_channel(
                request_channel, 
                self._handle_request
            )
            
            # Start heartbeat
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            self.background_tasks.add(heartbeat_task)
            
            # Update status
            self.service_info.status = ServiceStatus.READY
            await self.service_registry.update_service_status(
                self.service_info.id, 
                ServiceStatus.READY
            )
            
            logger.info(f"ContainerCommunicator initialized for {self.service_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize ContainerCommunicator: {e}")
            raise

    async def register_handler(self, method: str, handler: Callable):
        """Register request handler"""
        self.request_handlers[method] = handler
        logger.info(f"Registered handler for method: {method}")

    async def call_service(self, 
                          service_name: str, 
                          method: str, 
                          params: Dict[str, Any] = None,
                          timeout: int = 30) -> Any:
        """Call method on another service"""
        try:
            result = await self.message_broker.send_request(
                service_name=service_name,
                method=method,
                params=params or {},
                timeout=timeout
            )
            return result
        except Exception as e:
            logger.error(f"Service call failed: {service_name}.{method} - {e}")
            raise

    async def discover_services(self, capability: str = None) -> List[ServiceInfo]:
        """Discover services by capability"""
        if capability:
            return await self.service_registry.get_service_by_capability(capability)
        else:
            return await self.service_registry.discover_services()

    async def broadcast_event(self, event_type: str, data: Dict[str, Any]):
        """Broadcast event to all interested services"""
        message = Message(
            id=str(uuid.uuid4()),
            type=MessageType.EVENT,
            sender=self.service_name,
            recipient="broadcast",
            method=event_type,
            payload=data
        )
        
        # Publish to event channel
        event_channel = f"events:{event_type}"
        await self.message_broker.publish_message(event_channel, message)

    async def subscribe_to_events(self, event_type: str, handler: Callable):
        """Subscribe to events of specific type"""
        event_channel = f"events:{event_type}"
        await self.message_broker.subscribe_to_channel(event_channel, handler)

    async def _handle_request(self, message: Message):
        """Handle incoming request"""
        try:
            self.service_info.status = ServiceStatus.BUSY
            
            method = message.method
            handler = self.request_handlers.get(method)
            
            if not handler:
                raise ValueError(f"No handler for method: {method}")
            
            # Execute handler
            if inspect.iscoroutinefunction(handler):
                result = await handler(**message.payload)
            else:
                result = handler(**message.payload)
            
            # Send response
            response = Message(
                id=str(uuid.uuid4()),
                type=MessageType.RESPONSE,
                sender=self.service_name,
                recipient=message.sender,
                method=f"{method}_response",
                payload={"result": result, "success": True},
                correlation_id=message.id
            )
            
            response_channel = f"response:{message.id}"
            await self.message_broker.publish_message(response_channel, response)
            
        except Exception as e:
            logger.error(f"Request handling failed: {e}")
            
            # Send error response
            error_response = Message(
                id=str(uuid.uuid4()),
                type=MessageType.RESPONSE,
                sender=self.service_name,
                recipient=message.sender,
                method=f"{message.method}_response",
                payload={"error": str(e), "success": False},
                correlation_id=message.id
            )
            
            response_channel = f"response:{message.id}"
            await self.message_broker.publish_message(response_channel, error_response)
            
        finally:
            self.service_info.status = ServiceStatus.READY

    async def _heartbeat_loop(self):
        """Send periodic heartbeats"""
        while True:
            try:
                await asyncio.sleep(30)  # Heartbeat every 30 seconds
                
                # Update service status
                await self.service_registry.update_service_status(
                    self.service_info.id,
                    self.service_info.status
                )
                
                # Send heartbeat message
                heartbeat = Message(
                    id=str(uuid.uuid4()),
                    type=MessageType.HEARTBEAT,
                    sender=self.service_name,
                    recipient="system",
                    method="heartbeat",
                    payload={
                        "service_id": self.service_info.id,
                        "status": self.service_info.status.value,
                        "timestamp": time.time()
                    }
                )
                
                await self.message_broker.publish_message("system:heartbeats", heartbeat)
                
            except Exception as e:
                logger.error(f"Heartbeat failed: {e}")

    async def shutdown(self):
        """Shutdown communication system"""
        try:
            # Update status
            self.service_info.status = ServiceStatus.STOPPING
            await self.service_registry.update_service_status(
                self.service_info.id,
                ServiceStatus.STOPPING
            )
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Deregister service
            await self.service_registry.deregister_service(self.service_info.id)
            
            # Close connections
            await self.message_broker.close()
            
            logger.info(f"ContainerCommunicator shutdown complete for {self.service_name}")
            
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    @asynccontextmanager
    async def managed_connection(self):
        """Context manager for automatic initialization and cleanup"""
        try:
            await self.initialize()
            yield self
        finally:
            await self.shutdown()

# Health monitoring system
class HealthMonitor:
    """Monitor health of all registered services"""
    
    def __init__(self, communicator: ContainerCommunicator):
        self.communicator = communicator
        self.health_status = {}
        self.monitoring_interval = 60  # Check every minute

    async def start_monitoring(self):
        """Start health monitoring"""
        asyncio.create_task(self._monitoring_loop())

    async def _monitoring_loop(self):
        """Monitor service health continuously"""
        while True:
            try:
                await asyncio.sleep(self.monitoring_interval)
                await self._check_all_services()
            except Exception as e:
                logger.error(f"Health monitoring error: {e}")

    async def _check_all_services(self):
        """Check health of all services"""
        services = await self.communicator.discover_services()
        
        for service in services:
            try:
                # Make health check request
                async with aiohttp.ClientSession() as session:
                    url = f"http://{service.host}:{service.port}{service.health_endpoint}"
                    
                    async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as response:
                        if response.status == 200:
                            self.health_status[service.id] = {
                                "status": "healthy",
                                "last_check": time.time(),
                                "response_time": response.headers.get("X-Response-Time", "unknown")
                            }
                        else:
                            self.health_status[service.id] = {
                                "status": "unhealthy",
                                "last_check": time.time(),
                                "error": f"HTTP {response.status}"
                            }
                            
            except Exception as e:
                self.health_status[service.id] = {
                    "status": "unreachable",
                    "last_check": time.time(),
                    "error": str(e)
                }
                logger.warning(f"Health check failed for {service.name}: {e}")

    async def get_health_summary(self) -> Dict[str, Any]:
        """Get health summary of all services"""
        healthy = len([s for s in self.health_status.values() if s["status"] == "healthy"])
        total = len(self.health_status)
        
        return {
            "total_services": total,
            "healthy_services": healthy,
            "unhealthy_services": total - healthy,
            "health_percentage": (healthy / total * 100) if total > 0 else 0,
            "services": self.health_status
        }

# Example usage
async def example_service():
    """Example of how to use ContainerCommunicator"""
    
    # Initialize communicator
    communicator = ContainerCommunicator(
        service_name="example-service",
        service_type="ai_agent",
        capabilities=["text_processing", "analysis"]
    )
    
    # Register request handlers
    async def process_text(text: str, operation: str = "analyze") -> Dict[str, Any]:
        """Example text processing handler"""
        # Simulate processing
        await asyncio.sleep(0.1)
        
        return {
            "processed_text": f"Processed: {text}",
            "operation": operation,
            "word_count": len(text.split()),
            "timestamp": time.time()
        }
    
    await communicator.register_handler("process_text", process_text)
    
    # Use within context manager
    async with communicator.managed_connection():
        
        # Example: Call another service
        try:
            result = await communicator.call_service(
                service_name="document-processor",
                method="extract_text",
                params={"file_path": "/path/to/document.pdf"}
            )
            logger.info(f"Document processing result: {result}")
        except Exception as e:
            logger.error(f"Service call failed: {e}")
        
        # Example: Broadcast event
        await communicator.broadcast_event(
            event_type="task_completed",
            data={"task_id": "12345", "result": "success"}
        )
        
        # Example: Discover services
        code_generators = await communicator.discover_services(capability="code_generation")
        logger.info(f"Found {len(code_generators)} code generation services")
        
        # Keep service running
        await asyncio.sleep(3600)  # Run for 1 hour

if __name__ == "__main__":
    asyncio.run(example_service())