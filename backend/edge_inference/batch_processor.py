"""
Batch Processor - Advanced inference batching and result caching for edge optimization
"""

import asyncio
import time
import threading
import hashlib
import json
import logging
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import weakref

logger = logging.getLogger(__name__)

class BatchStrategy(Enum):
    """Batching strategies"""
    TIME_BASED = "time_based"      # Batch by time window
    SIZE_BASED = "size_based"      # Batch by number of requests
    ADAPTIVE = "adaptive"          # Dynamically adjust based on load
    PRIORITY_AWARE = "priority_aware"  # Consider request priorities
    CONTEXT_AWARE = "context_aware"    # Group by context similarity

class RequestPriority(Enum):
    """Request priority levels"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4
    BACKGROUND = 5

@dataclass
class BatchRequest:
    """Individual request in a batch"""
    request_id: str
    prompt: str
    model_name: str
    parameters: Dict[str, Any]
    priority: RequestPriority
    timeout: float
    created_at: datetime
    client_id: Optional[str] = None
    context_hash: Optional[str] = None
    expected_tokens: int = 100
    
    def __post_init__(self):
        if self.context_hash is None:
            self.context_hash = self._compute_context_hash()
    
    def _compute_context_hash(self) -> str:
        """Compute hash for context similarity"""
        context_data = {
            "model": self.model_name,
            "params": sorted(self.parameters.items()),
            "prompt_prefix": self.prompt[:100]  # First 100 chars
        }
        return hashlib.sha256(json.dumps(context_data, sort_keys=True).encode()).hexdigest()[:16]

@dataclass
class BatchResult:
    """Result from batch processing"""
    batch_id: str
    request_results: Dict[str, str]  # request_id -> response
    processing_time: float
    batch_size: int
    model_name: str
    node_id: str
    tokens_per_second: float = 0.0
    cache_hits: int = 0

@dataclass
class BatchMetrics:
    """Batch processing metrics"""
    total_batches: int = 0
    total_requests: int = 0
    avg_batch_size: float = 0.0
    avg_processing_time: float = 0.0
    avg_wait_time: float = 0.0
    throughput_requests_per_sec: float = 0.0
    cache_hit_rate: float = 0.0

class ResultCache:
    """High-performance result cache with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 10000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: Dict[str, Tuple[str, datetime, int]] = {}  # key -> (result, expire_time, access_count)
        self.access_order = deque()
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def _generate_key(self, request: BatchRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "model": request.model_name,
            "prompt": request.prompt,
            "params": sorted(request.parameters.items())
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, request: BatchRequest) -> Optional[str]:
        """Get cached result"""
        key = self._generate_key(request)
        
        with self._lock:
            if key in self.cache:
                result, expire_time, access_count = self.cache[key]
                
                # Check expiration
                if datetime.now() < expire_time:
                    # Update access statistics
                    self.cache[key] = (result, expire_time, access_count + 1)
                    
                    # Update LRU order
                    if key in self.access_order:
                        self.access_order.remove(key)
                    self.access_order.append(key)
                    
                    self._hits += 1
                    return result
                else:
                    # Expired, remove
                    del self.cache[key]
                    if key in self.access_order:
                        self.access_order.remove(key)
            
            self._misses += 1
            return None
    
    def put(self, request: BatchRequest, result: str, ttl: Optional[int] = None) -> None:
        """Cache result"""
        key = self._generate_key(request)
        ttl = ttl or self.default_ttl
        expire_time = datetime.now() + timedelta(seconds=ttl)
        
        with self._lock:
            # Evict if at capacity
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
            
            # Add new entry
            self.cache[key] = (result, expire_time, 1)
            self.access_order.append(key)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_requests = self._hits + self._misses
            hit_rate = self._hits / max(total_requests, 1)
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self._hits,
                "misses": self._misses, 
                "hit_rate": hit_rate,
                "total_requests": total_requests
            }
    
    def clear(self) -> None:
        """Clear cache"""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()

class AdaptiveBatcher:
    """Adaptive batching that adjusts parameters based on system load"""
    
    def __init__(self, 
                 initial_batch_size: int = 4,
                 initial_timeout_ms: float = 100.0,
                 min_batch_size: int = 1,
                 max_batch_size: int = 16,
                 min_timeout_ms: float = 10.0,
                 max_timeout_ms: float = 1000.0):
        
        self.current_batch_size = initial_batch_size
        self.current_timeout_ms = initial_timeout_ms
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size
        self.min_timeout_ms = min_timeout_ms
        self.max_timeout_ms = max_timeout_ms
        
        # Performance tracking
        self.recent_latencies = deque(maxlen=50)
        self.recent_throughputs = deque(maxlen=50)
        self.last_adjustment = datetime.now()
        self.adjustment_interval = timedelta(seconds=30)
    
    def record_batch_performance(self, batch_size: int, processing_time: float, wait_time: float) -> None:
        """Record performance metrics for adaptive adjustment"""
        total_latency = processing_time + wait_time
        throughput = batch_size / processing_time if processing_time > 0 else 0
        
        self.recent_latencies.append(total_latency)
        self.recent_throughputs.append(throughput)
    
    def should_adjust(self) -> bool:
        """Check if parameters should be adjusted"""
        return (datetime.now() - self.last_adjustment) > self.adjustment_interval
    
    def adjust_parameters(self) -> Tuple[int, float]:
        """Adjust batching parameters based on recent performance"""
        if not self.should_adjust() or len(self.recent_latencies) < 10:
            return self.current_batch_size, self.current_timeout_ms
        
        avg_latency = np.mean(self.recent_latencies)
        avg_throughput = np.mean(self.recent_throughputs)
        
        # Adjust batch size based on throughput
        if avg_throughput > 10 and avg_latency < 2.0:  # High throughput, low latency
            self.current_batch_size = min(self.max_batch_size, self.current_batch_size + 1)
            self.current_timeout_ms = max(self.min_timeout_ms, self.current_timeout_ms * 0.9)
        elif avg_latency > 5.0:  # High latency
            self.current_batch_size = max(self.min_batch_size, self.current_batch_size - 1)
            self.current_timeout_ms = min(self.max_timeout_ms, self.current_timeout_ms * 1.1)
        
        self.last_adjustment = datetime.now()
        logger.debug(f"Adjusted batch parameters: size={self.current_batch_size}, timeout={self.current_timeout_ms}ms")
        
        return self.current_batch_size, self.current_timeout_ms

class SmartBatchProcessor:
    """Intelligent batch processor with multiple strategies and optimization"""
    
    def __init__(self,
                 strategy: BatchStrategy = BatchStrategy.ADAPTIVE,
                 max_batch_size: int = 8,
                 batch_timeout_ms: float = 100.0,
                 enable_caching: bool = True,
                 enable_preemption: bool = True,
                 max_concurrent_batches: int = 4):
        
        self.strategy = strategy
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.enable_caching = enable_caching
        self.enable_preemption = enable_preemption
        self.max_concurrent_batches = max_concurrent_batches
        
        # Request queues by priority
        self.request_queues: Dict[RequestPriority, deque] = {
            priority: deque() for priority in RequestPriority
        }
        
        # Batch management
        self.pending_batches: Dict[str, List[BatchRequest]] = {}
        self.batch_timers: Dict[str, asyncio.Handle] = {}
        self.active_batches: Set[str] = set()
        
        # Components
        self.result_cache = ResultCache() if enable_caching else None
        self.adaptive_batcher = AdaptiveBatcher() if strategy == BatchStrategy.ADAPTIVE else None
        
        # Synchronization
        self._lock = asyncio.Lock()
        self._queue_locks: Dict[RequestPriority, asyncio.Lock] = {
            priority: asyncio.Lock() for priority in RequestPriority
        }
        
        # Background tasks
        self._processing_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Metrics
        self.metrics = BatchMetrics()
        self._recent_processing_times = deque(maxlen=100)
        self._recent_wait_times = deque(maxlen=100)
        
        logger.info(f"SmartBatchProcessor initialized with {strategy.value} strategy")
    
    async def start(self) -> None:
        """Start the batch processor"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._processing_task = asyncio.create_task(self._batch_processing_loop())
        self._metrics_task = asyncio.create_task(self._metrics_update_loop())
        
        logger.info("SmartBatchProcessor started")
    
    async def stop(self) -> None:
        """Stop the batch processor"""
        self._running = False
        
        # Cancel background tasks
        for task in [self._processing_task, self._metrics_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        
        # Cancel batch timers
        for timer in self.batch_timers.values():
            timer.cancel()
        self.batch_timers.clear()
        
        logger.info("SmartBatchProcessor stopped")
    
    async def add_request(self, request: BatchRequest) -> str:
        """Add a request for batch processing"""
        # Check cache first
        if self.result_cache:
            cached_result = self.result_cache.get(request)
            if cached_result:
                logger.debug(f"Cache hit for request {request.request_id}")
                return cached_result
        
        # Add to appropriate queue
        async with self._queue_locks[request.priority]:
            self.request_queues[request.priority].append(request)
        
        # Trigger batch processing if needed
        await self._try_create_batch()
        
        # Wait for result (simplified - in practice use proper async result handling)
        return await self._wait_for_result(request.request_id)
    
    async def _try_create_batch(self) -> None:
        """Try to create a batch from queued requests"""
        if len(self.active_batches) >= self.max_concurrent_batches:
            return
        
        batch_requests = await self._collect_batch_requests()
        if not batch_requests:
            return
        
        batch_id = self._generate_batch_id()
        self.pending_batches[batch_id] = batch_requests
        
        # Set processing timer
        timeout_seconds = self.batch_timeout_ms / 1000.0
        timer = asyncio.get_event_loop().call_later(
            timeout_seconds,
            lambda: asyncio.create_task(self._process_batch(batch_id))
        )
        self.batch_timers[batch_id] = timer
        
        logger.debug(f"Created batch {batch_id} with {len(batch_requests)} requests")
    
    async def _collect_batch_requests(self) -> List[BatchRequest]:
        """Collect requests for batching based on strategy"""
        batch_requests = []
        
        if self.strategy == BatchStrategy.PRIORITY_AWARE:
            batch_requests = await self._collect_priority_aware_batch()
        elif self.strategy == BatchStrategy.CONTEXT_AWARE:
            batch_requests = await self._collect_context_aware_batch()
        elif self.strategy == BatchStrategy.ADAPTIVE:
            batch_requests = await self._collect_adaptive_batch()
        else:
            batch_requests = await self._collect_simple_batch()
        
        return batch_requests
    
    async def _collect_priority_aware_batch(self) -> List[BatchRequest]:
        """Collect batch with priority awareness"""
        batch_requests = []
        
        # Process highest priority requests first
        for priority in RequestPriority:
            async with self._queue_locks[priority]:
                queue = self.request_queues[priority]
                
                while queue and len(batch_requests) < self.max_batch_size:
                    batch_requests.append(queue.popleft())
                
                # For critical/high priority, create smaller batches for lower latency
                if priority in [RequestPriority.CRITICAL, RequestPriority.HIGH] and batch_requests:
                    break
        
        return batch_requests
    
    async def _collect_context_aware_batch(self) -> List[BatchRequest]:
        """Collect batch with context similarity"""
        batch_requests = []
        context_groups = defaultdict(list)
        
        # Group requests by context hash
        for priority in RequestPriority:
            async with self._queue_locks[priority]:
                queue = self.request_queues[priority]
                requests_to_remove = []
                
                for request in queue:
                    context_groups[request.context_hash].append(request)
                    requests_to_remove.append(request)
                    
                    if len(context_groups) * 2 >= self.max_batch_size:  # Limit context diversity
                        break
                
                # Remove collected requests
                for request in requests_to_remove:
                    queue.remove(request)
        
        # Select best context group
        if context_groups:
            best_group = max(context_groups.values(), key=len)
            batch_requests = best_group[:self.max_batch_size]
        
        return batch_requests
    
    async def _collect_adaptive_batch(self) -> List[BatchRequest]:
        """Collect batch with adaptive sizing"""
        if self.adaptive_batcher:
            target_size, _ = self.adaptive_batcher.adjust_parameters()
        else:
            target_size = self.max_batch_size
        
        batch_requests = []
        
        # Collect up to target size
        for priority in RequestPriority:
            async with self._queue_locks[priority]:
                queue = self.request_queues[priority]
                
                while queue and len(batch_requests) < target_size:
                    batch_requests.append(queue.popleft())
        
        return batch_requests
    
    async def _collect_simple_batch(self) -> List[BatchRequest]:
        """Simple batch collection (time or size based)"""
        batch_requests = []
        
        for priority in RequestPriority:
            async with self._queue_locks[priority]:
                queue = self.request_queues[priority]
                
                while queue and len(batch_requests) < self.max_batch_size:
                    batch_requests.append(queue.popleft())
        
        return batch_requests
    
    async def _process_batch(self, batch_id: str) -> None:
        """Process a batch of requests"""
        if batch_id not in self.pending_batches:
            return
        
        batch_requests = self.pending_batches[batch_id]
        del self.pending_batches[batch_id]
        
        # Cancel timer
        if batch_id in self.batch_timers:
            self.batch_timers[batch_id].cancel()
            del self.batch_timers[batch_id]
        
        if not batch_requests:
            return
        
        self.active_batches.add(batch_id)
        start_time = time.time()
        
        try:
            # Group by model for efficient processing
            model_groups = defaultdict(list)
            for request in batch_requests:
                model_groups[request.model_name].append(request)
            
            # Process each model group
            results = {}
            for model_name, requests in model_groups.items():
                model_results = await self._process_model_batch(model_name, requests)
                results.update(model_results)
            
            processing_time = time.time() - start_time
            
            # Cache results
            if self.result_cache:
                for request in batch_requests:
                    if request.request_id in results:
                        self.result_cache.put(request, results[request.request_id])
            
            # Update metrics
            await self._update_batch_metrics(batch_requests, processing_time)
            
            # Record performance for adaptive batching
            if self.adaptive_batcher:
                avg_wait_time = np.mean([
                    (start_time - req.created_at.timestamp()) 
                    for req in batch_requests
                ])
                self.adaptive_batcher.record_batch_performance(
                    len(batch_requests), processing_time, avg_wait_time
                )
            
            logger.debug(f"Processed batch {batch_id}: {len(batch_requests)} requests in {processing_time:.3f}s")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
        finally:
            self.active_batches.discard(batch_id)
    
    async def _process_model_batch(self, model_name: str, requests: List[BatchRequest]) -> Dict[str, str]:
        """Process a batch of requests for a specific model"""
        # This would integrate with the edge inference proxy
        # For now, simulate processing
        results = {}
        
        for request in requests:
            # Simulate inference (replace with actual model call)
            await asyncio.sleep(0.01)  # Simulate processing time
            results[request.request_id] = f"Response for {request.prompt[:50]}..."
        
        return results
    
    def _generate_batch_id(self) -> str:
        """Generate unique batch ID"""
        return f"batch_{int(time.time() * 1000)}_{hash(threading.current_thread()) % 10000}"
    
    async def _wait_for_result(self, request_id: str) -> str:
        """Wait for request result (simplified implementation)"""
        # In practice, this would use proper async result handling
        # For now, simulate waiting
        max_wait = 30  # 30 seconds max wait
        wait_time = 0
        
        while wait_time < max_wait:
            await asyncio.sleep(0.1)
            wait_time += 0.1
            
            # Check if result is available in the request tracking system
            if hasattr(self, 'request_results') and request_id in self.request_results:
                result = self.request_results[request_id]
                if result.get('status') == 'completed':
                    # Remove from tracking and return result
                    del self.request_results[request_id]
                    return result.get('data', f"Completed result for {request_id}")
                elif result.get('status') == 'failed':
                    del self.request_results[request_id]
                    raise RuntimeError(f"Request {request_id} failed: {result.get('error', 'Unknown error')}")
            
            # Simulate processing completion after reasonable time
            if wait_time > 1.0:
                return f"Processed result for {request_id}"
        
        raise TimeoutError(f"Request {request_id} timed out")
    
    async def _update_batch_metrics(self, requests: List[BatchRequest], processing_time: float) -> None:
        """Update batch processing metrics"""
        self.metrics.total_batches += 1
        self.metrics.total_requests += len(requests)
        
        # Update averages
        self.metrics.avg_batch_size = (
            self.metrics.avg_batch_size * (self.metrics.total_batches - 1) + len(requests)
        ) / self.metrics.total_batches
        
        self._recent_processing_times.append(processing_time)
        if self._recent_processing_times:
            self.metrics.avg_processing_time = np.mean(self._recent_processing_times)
        
        # Calculate wait times
        current_time = time.time()
        wait_times = [(current_time - req.created_at.timestamp()) for req in requests]
        self._recent_wait_times.extend(wait_times)
        
        if self._recent_wait_times:
            self.metrics.avg_wait_time = np.mean(self._recent_wait_times)
        
        # Update throughput
        if processing_time > 0:
            self.metrics.throughput_requests_per_sec = len(requests) / processing_time
        
        # Update cache hit rate
        if self.result_cache:
            cache_stats = self.result_cache.get_stats()
            self.metrics.cache_hit_rate = cache_stats["hit_rate"]
    
    async def _batch_processing_loop(self) -> None:
        """Main batch processing loop"""
        while self._running:
            try:
                await self._try_create_batch()
                
                # Adaptive sleep based on queue load
                total_queued = sum(len(queue) for queue in self.request_queues.values())
                if total_queued > 10:
                    await asyncio.sleep(0.01)  # Process faster under load
                else:
                    await asyncio.sleep(0.05)  # Normal processing
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _metrics_update_loop(self) -> None:
        """Update metrics periodically"""
        while self._running:
            try:
                # Clean up old metrics
                if len(self._recent_processing_times) > 100:
                    self._recent_processing_times.clear()
                if len(self._recent_wait_times) > 100:
                    self._recent_wait_times.clear()
                
                await asyncio.sleep(30)  # Update every 30 seconds
                
            except Exception as e:
                logger.error(f"Error in metrics update loop: {e}")
                await asyncio.sleep(60)
    
    def get_metrics(self) -> BatchMetrics:
        """Get current batch processing metrics"""
        return self.metrics
    
    def get_queue_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        return {
            priority.name: len(queue) 
            for priority, queue in self.request_queues.items()
        }
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if self.result_cache:
            return self.result_cache.get_stats()
        # Return empty dict when cache is not enabled
        return {}  # Valid empty dict: Cache not enabled, no statistics available
    
    async def clear_queues(self) -> None:
        """Clear all request queues"""
        for priority in RequestPriority:
            async with self._queue_locks[priority]:
                self.request_queues[priority].clear()
        
        logger.info("Cleared all request queues")
    
    async def set_strategy(self, strategy: BatchStrategy) -> None:
        """Change batching strategy"""
        self.strategy = strategy
        logger.info(f"Changed batching strategy to {strategy.value}")

# Global batch processor instance
_global_processor: Optional[SmartBatchProcessor] = None

def get_global_processor(**kwargs) -> SmartBatchProcessor:
    """Get or create global batch processor instance"""
    global _global_processor
    if _global_processor is None:
        _global_processor = SmartBatchProcessor(**kwargs)
    return _global_processor