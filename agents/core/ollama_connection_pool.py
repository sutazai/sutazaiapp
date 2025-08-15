#!/usr/bin/env python3
"""
High-Performance Ollama Connection Pool Manager
Optimized for 174+ concurrent AI agent connections with intelligent queuing and failover.
"""

import asyncio
import aiohttp
import json
import time
import logging
import threading
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import psutil
import redis.asyncio as redis
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class OllamaInstance:
    """Represents an Ollama instance with health and performance metrics."""
    host: str
    port: int
    max_connections: int = 25  # Per instance limit
    current_connections: int = 0
    response_time_avg: float = 0.0
    error_count: int = 0
    last_health_check: float = field(default_factory=time.time)
    is_healthy: bool = True
    load_factor: float = 0.0  # 0.0 = no load, 1.0 = fully loaded

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def availability_score(self) -> float:
        """Calculate availability score for load balancing."""
        health_weight = 1.0 if self.is_healthy else 0.0
        load_weight = max(0.0, 1.0 - self.load_factor)
        response_weight = max(0.0, 1.0 - min(self.response_time_avg / 10.0, 1.0))
        
        return (health_weight * 0.5 + load_weight * 0.3 + response_weight * 0.2)

@dataclass
class RequestMetrics:
    """Track request performance metrics."""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    avg_response_time: float = 0.0
    queue_wait_time: float = 0.0
    peak_queue_size: int = 0
    current_queue_size: int = 0

class OllamaConnectionPool:
    """
    High-performance connection pool for Ollama with intelligent load balancing,
    circuit breaking, and automatic failover for 174+ concurrent connections.
    """
    
    def __init__(self, 
                 instances: List[Tuple[str, int]] = None,
                 max_connections_per_instance: int = 25,
                 max_queue_size: int = 500,
                 request_timeout: int = 120,
                 health_check_interval: int = 30,
                 circuit_breaker_threshold: int = 10,
                 redis_url: str = "redis://localhost:6379"):
        
        # Default to single local instance if none provided
        if instances is None:
            instances = [("localhost", 10104)]
        
        self.instances = [
            OllamaInstance(host, port, max_connections_per_instance)
            for host, port in instances
        ]
        
        self.max_connections_per_instance = max_connections_per_instance
        self.max_queue_size = max_queue_size
        self.request_timeout = request_timeout
        self.health_check_interval = health_check_interval
        self.circuit_breaker_threshold = circuit_breaker_threshold
        
        # Request queue and metrics
        self.request_queue = asyncio.Queue(maxsize=max_queue_size)
        self.metrics = RequestMetrics()
        
        # Connection management
        self.session_pools: Dict[str, aiohttp.ClientSession] = {}
        self.active_requests: Dict[str, int] = {}
        
        # Background tasks
        self.health_check_task: Optional[asyncio.Task] = None
        self.queue_processor_task: Optional[asyncio.Task] = None
        self.metrics_collector_task: Optional[asyncio.Task] = None
        
        # Redis for distributed coordination
        self.redis_client: Optional[redis.Redis] = None
        self.redis_url = redis_url
        
        # Thread-safe locks
        self.lock = asyncio.Lock()
        
        logger.info(f"Initialized Ollama connection pool with {len(self.instances)} instances")
        logger.info(f"Total capacity: {len(self.instances) * max_connections_per_instance} concurrent connections")

    async def initialize(self):
        """Initialize the connection pool and background tasks."""
        try:
            # Initialize Redis connection
            self.redis_client = redis.from_url(self.redis_url)
            await self.redis_client.ping()
            logger.info("Connected to Redis for coordination")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Running without coordination.")
            self.redis_client = None
        
        # Create aiohttp sessions for each instance
        connector = aiohttp.TCPConnector(
            limit=100,  # Total connection pool size
            limit_per_host=50,  # Per-host limit
            ttl_dns_cache=300,
            ttl_dns_cache_resolution=300,
            use_dns_cache=True,
            keepalive_timeout=300,
            enable_cleanup_closed=True
        )
        
        for instance in self.instances:
            session = aiohttp.ClientSession(
                connector=connector,
                timeout=aiohttp.ClientTimeout(total=self.request_timeout),
                headers={"Content-Type": "application/json"}
            )
            self.session_pools[instance.url] = session
            self.active_requests[instance.url] = 0
            
        # Start background tasks
        self.health_check_task = asyncio.create_task(self._health_check_loop())
        self.queue_processor_task = asyncio.create_task(self._queue_processor_loop())
        self.metrics_collector_task = asyncio.create_task(self._metrics_collector_loop())
        
        logger.info("Connection pool initialized successfully")

    async def shutdown(self):
        """Gracefully shutdown the connection pool."""
        logger.info("Shutting down connection pool...")
        
        # Cancel background tasks
        for task in [self.health_check_task, self.queue_processor_task, self.metrics_collector_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Close all sessions
        for session in self.session_pools.values():
            await session.close()
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
            
        logger.info("Connection pool shutdown complete")

    async def _select_best_instance(self) -> Optional[OllamaInstance]:
        """Select the best available instance based on load and health."""
        healthy_instances = [i for i in self.instances if i.is_healthy]
        
        if not healthy_instances:
            logger.error("No healthy Ollama instances available")
            return None
        
        # Sort by availability score (highest first)
        healthy_instances.sort(key=lambda x: x.availability_score, reverse=True)
        
        # Select instance with lowest current load among top candidates
        best_instance = healthy_instances[0]
        
        for instance in healthy_instances[:3]:  # Consider top 3 instances
            if (self.active_requests.get(instance.url, 0) < 
                self.active_requests.get(best_instance.url, 0)):
                best_instance = instance
        
        return best_instance

    async def generate(self, 
                      model: str, 
                      prompt: str, 
                      options: Dict[str, Any] = None,
                      priority: int = 5) -> Dict[str, Any]:
        """
        Generate text using Ollama with intelligent routing and queuing.
        
        Args:
            model: Model name (defaults to tinyllama per Rule 16)
            prompt: Input prompt
            options: Generation options
            priority: Request priority (1=high, 10=low)
        
        Returns:
            Generation response dict
        """
        if model is None:
            model = "tinyllama"  # Rule 16 compliance
            
        request_start = time.time()
        
        # Create request payload
        payload = {
            "model": model,
            "prompt": prompt,
            "options": options or {},
            "stream": False
        }
        
        try:
            # Queue the request with priority
            await self.request_queue.put((priority, request_start, payload))
            self.metrics.current_queue_size = self.request_queue.qsize()
            self.metrics.peak_queue_size = max(self.metrics.peak_queue_size, 
                                             self.metrics.current_queue_size)
            
            # Process the request
            response = await self._process_request(payload, request_start)
            
            self.metrics.successful_requests += 1
            return response
            
        except Exception as e:
            self.metrics.failed_requests += 1
            logger.error(f"Request failed: {e}")
            raise
        finally:
            self.metrics.total_requests += 1

    async def _process_request(self, payload: Dict[str, Any], request_start: float) -> Dict[str, Any]:
        """Process a request using the best available instance."""
        async with self.lock:
            instance = await self._select_best_instance()
            
        if not instance:
            raise Exception("No healthy Ollama instances available")
        
        session = self.session_pools[instance.url]
        
        # Track active request
        self.active_requests[instance.url] += 1
        instance.current_connections += 1
        
        try:
            async with session.post(f"{instance.url}/api/generate", 
                                  json=payload) as response:
                
                if response.status == 200:
                    result = await response.json()
                    
                    # Update instance metrics
                    response_time = time.time() - request_start
                    instance.response_time_avg = (
                        (instance.response_time_avg * 0.9) + (response_time * 0.1)
                    )
                    
                    return result
                else:
                    error_msg = await response.text()
                    raise Exception(f"HTTP {response.status}: {error_msg}")
                    
        except Exception as e:
            instance.error_count += 1
            
            # Circuit breaker logic
            if instance.error_count > self.circuit_breaker_threshold:
                instance.is_healthy = False
                logger.warning(f"Circuit breaker opened for {instance.url}")
                
            raise e
            
        finally:
            # Release connection tracking
            self.active_requests[instance.url] -= 1
            instance.current_connections -= 1
            instance.load_factor = (
                self.active_requests[instance.url] / instance.max_connections
            )

    async def _health_check_loop(self):
        """Background task for health checking instances."""
        while True:
            try:
                await asyncio.sleep(self.health_check_interval)
                await self._check_all_instances_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")

    async def _check_all_instances_health(self):
        """Check health of all instances."""
        tasks = []
        for instance in self.instances:
            tasks.append(self._check_instance_health(instance))
        
        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_instance_health(self, instance: OllamaInstance):
        """Check health of a single instance."""
        try:
            session = self.session_pools[instance.url]
            start_time = time.time()
            
            async with session.get(f"{instance.url}/api/tags", 
                                 timeout=aiohttp.ClientTimeout(total=10)) as response:
                
                if response.status == 200:
                    response_time = time.time() - start_time
                    instance.response_time_avg = (
                        (instance.response_time_avg * 0.8) + (response_time * 0.2)
                    )
                    instance.is_healthy = True
                    instance.error_count = max(0, instance.error_count - 1)
                    instance.last_health_check = time.time()
                else:
                    instance.is_healthy = False
                    
        except Exception as e:
            instance.is_healthy = False
            instance.error_count += 1
            logger.warning(f"Health check failed for {instance.url}: {e}")

    async def _queue_processor_loop(self):
        """Background task for processing the request queue."""
        while True:
            try:
                # Process queued requests
                if not self.request_queue.empty():
                    await self._process_queue_batch()
                else:
                    await asyncio.sleep(0.1)  # Small delay when queue is empty
                    
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Queue processor error: {e}")

    async def _process_queue_batch(self):
        """Process a batch of queued requests."""
        batch_size = min(10, self.request_queue.qsize())
        
        for _ in range(batch_size):
            try:
                priority, request_start, payload = await asyncio.wait_for(
                    self.request_queue.get(), timeout=1.0
                )
                
                queue_wait_time = time.time() - request_start
                self.metrics.queue_wait_time = (
                    (self.metrics.queue_wait_time * 0.9) + (queue_wait_time * 0.1)
                )
                
                # Process request asynchronously
                asyncio.create_task(self._process_request(payload, request_start))
                
            except asyncio.TimeoutError:
                break  # No more items in queue
            except Exception as e:
                logger.error(f"Error processing queue item: {e}")

    async def _metrics_collector_loop(self):
        """Background task for collecting and reporting metrics."""
        while True:
            try:
                await asyncio.sleep(60)  # Collect metrics every minute
                await self._collect_and_report_metrics()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collector error: {e}")

    async def _collect_and_report_metrics(self):
        """Collect and report performance metrics."""
        total_connections = sum(self.active_requests.values())
        total_capacity = len(self.instances) * self.max_connections_per_instance
        
        utilization = (total_connections / total_capacity) * 100 if total_capacity > 0 else 0
        
        metrics_data = {
            "timestamp": time.time(),
            "instances": len(self.instances),
            "healthy_instances": sum(1 for i in self.instances if i.is_healthy),
            "total_capacity": total_capacity,
            "active_connections": total_connections,
            "utilization_percent": utilization,
            "queue_size": self.metrics.current_queue_size,
            "peak_queue_size": self.metrics.peak_queue_size,
            "total_requests": self.metrics.total_requests,
            "success_rate": (self.metrics.successful_requests / max(1, self.metrics.total_requests)) * 100,
            "avg_response_time": self.metrics.avg_response_time,
            "avg_queue_wait_time": self.metrics.queue_wait_time
        }
        
        # Store metrics in Redis if available
        if self.redis_client:
            try:
                await self.redis_client.setex(
                    "ollama:pool:metrics", 
                    300,  # 5 minute TTL
                    json.dumps(metrics_data)
                )
            except Exception as e:
                logger.error(f"Failed to store metrics in Redis: {e}")
        
        # Log metrics
        logger.info(f"Pool Metrics - Utilization: {utilization:.1f}%, "
                   f"Active: {total_connections}/{total_capacity}, "
                   f"Queue: {self.metrics.current_queue_size}, "
                   f"Success Rate: {metrics_data['success_rate']:.1f}%")

    def get_status(self) -> Dict[str, Any]:
        """Get current pool status and metrics."""
        healthy_instances = [i for i in self.instances if i.is_healthy]
        
        return {
            "total_instances": len(self.instances),
            "healthy_instances": len(healthy_instances),
            "total_capacity": len(self.instances) * self.max_connections_per_instance,
            "active_connections": sum(self.active_requests.values()),
            "queue_size": self.metrics.current_queue_size,
            "peak_queue_size": self.metrics.peak_queue_size,
            "total_requests": self.metrics.total_requests,
            "successful_requests": self.metrics.successful_requests,
            "failed_requests": self.metrics.failed_requests,
            "success_rate": (self.metrics.successful_requests / max(1, self.metrics.total_requests)) * 100,
            "avg_response_time": self.metrics.avg_response_time,
            "avg_queue_wait_time": self.metrics.queue_wait_time,
            "instances": [
                {
                    "url": instance.url,
                    "healthy": instance.is_healthy,
                    "connections": self.active_requests.get(instance.url, 0),
                    "max_connections": instance.max_connections,
                    "load_factor": instance.load_factor,
                    "avg_response_time": instance.response_time_avg,
                    "error_count": instance.error_count,
                    "availability_score": instance.availability_score
                }
                for instance in self.instances
            ]
        }

# Global connection pool instance
_connection_pool: Optional[OllamaConnectionPool] = None

async def get_connection_pool() -> OllamaConnectionPool:
    """Get or create the global connection pool."""
    global _connection_pool
    
    if _connection_pool is None:
        # Configure instances based on available setup
        instances = [("localhost", 10104)]  # Start with single instance
        
        _connection_pool = OllamaConnectionPool(
            instances=instances,
            max_connections_per_instance=50,  # High concurrency per instance
            max_queue_size=500,
            request_timeout=120,
            health_check_interval=30
        )
        
        await _connection_pool.initialize()
    
    return _connection_pool

async def ollama_generate(model: str = "tinyllama", 
                         prompt: str = "", 
                         options: Dict[str, Any] = None,
                         priority: int = 5) -> Dict[str, Any]:
    """
    High-level interface for generating text with Ollama.
    Automatically handles connection pooling, load balancing, and queuing.
    """
    pool = await get_connection_pool()
    return await pool.generate(model, prompt, options, priority)

if __name__ == "__main__":
    async def test_pool():
        """Test the connection pool functionality."""
        pool = await get_connection_pool()
        
        # Test basic generation
        response = await pool.generate(
            model="tinyllama",
            prompt="Hello, how are you?",
            options={"max_tokens": 50}
        )
        
        logger.info("Response:", response)
        logger.info("Pool Status:", pool.get_status())
        
        await pool.shutdown()

    asyncio.run(test_pool())