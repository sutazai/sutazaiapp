#!/usr/bin/env python3
"""
Purpose: Batch processing optimization for efficient multi-agent inference
Usage: Optimizes request batching and parallel processing on CPU
Requirements: asyncio, numpy, multiprocessing
"""

import os
import asyncio
import logging
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import numpy as np
from collections import defaultdict, deque
import time
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import heapq
import json

logger = logging.getLogger('batch-processing-optimizer')


@dataclass
class InferenceRequest:
    """Single inference request"""
    request_id: str
    agent_id: str
    prompt: str
    priority: int = 5  # 1-10, higher is more important
    max_tokens: int = 512
    temperature: float = 0.7
    timestamp: datetime = field(default_factory=datetime.utcnow)
    deadline: Optional[datetime] = None
    
    def __lt__(self, other):
        # For priority queue comparison
        return self.priority > other.priority


@dataclass
class BatchConfig:
    """Configuration for batch processing"""
    max_batch_size: int = 8
    max_wait_time_ms: int = 50
    priority_threshold: int = 8  # High priority requests bypass batching
    dynamic_batching: bool = True
    cpu_cores: int = 12
    memory_limit_mb: int = 8192
    enable_request_merging: bool = True
    enable_response_caching: bool = True


@dataclass
class BatchMetrics:
    """Metrics for batch processing performance"""
    total_requests: int = 0
    total_batches: int = 0
    average_batch_size: float = 0.0
    average_wait_time_ms: float = 0.0
    cache_hit_rate: float = 0.0
    throughput_rps: float = 0.0
    cpu_utilization: float = 0.0
    memory_utilization: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0


class BatchProcessingOptimizer:
    """
    Optimizes batch processing for CPU-based inference
    
    Features:
    - Dynamic batching based on load
    - Priority-aware scheduling
    - Request merging for similar prompts
    - Response caching
    - CPU affinity optimization
    - Memory-aware batching
    """
    
    def __init__(self, config: BatchConfig):
        self.config = config
        self.metrics = BatchMetrics()
        
        # Request queues
        self.pending_requests: Dict[str, deque] = defaultdict(deque)  # Per agent
        self.priority_queue: List[InferenceRequest] = []  # High priority
        
        # Batch tracking
        self.active_batches: Dict[str, List[InferenceRequest]] = {}
        self.batch_timers: Dict[str, float] = {}
        
        # Caching
        self.response_cache: Dict[str, Tuple[str, datetime]] = {}
        self.cache_ttl = timedelta(minutes=5)
        
        # Processing pools
        self.thread_pool = ThreadPoolExecutor(max_workers=config.cpu_cores)
        self.process_pool = ProcessPoolExecutor(max_workers=config.cpu_cores // 2)
        
        # Performance tracking
        self.latency_history = deque(maxlen=1000)
        self.start_time = time.time()
        
        # Running state
        self._running = False
        self._batch_task = None
        
    async def start(self):
        """Start the batch processing optimizer"""
        self._running = True
        self._batch_task = asyncio.create_task(self._batch_formation_loop())
        logger.info("Batch processing optimizer started")
        
    async def stop(self):
        """Stop the batch processing optimizer"""
        self._running = False
        if self._batch_task:
            await self._batch_task
        self.thread_pool.shutdown()
        self.process_pool.shutdown()
        logger.info("Batch processing optimizer stopped")
        
    async def submit_request(self, request: InferenceRequest) -> str:
        """
        Submit a request for processing
        
        Args:
            request: The inference request
            
        Returns:
            Request ID for tracking
        """
        self.metrics.total_requests += 1
        
        # Check cache first
        if self.config.enable_response_caching:
            cache_key = self._get_cache_key(request)
            if cache_key in self.response_cache:
                response, timestamp = self.response_cache[cache_key]
                if datetime.utcnow() - timestamp < self.cache_ttl:
                    self.metrics.cache_hit_rate = self._update_rate(
                        self.metrics.cache_hit_rate, True
                    )
                    return response
        
        # High priority requests bypass batching
        if request.priority >= self.config.priority_threshold:
            heapq.heappush(self.priority_queue, request)
            asyncio.create_task(self._process_priority_request(request))
        else:
            # Add to agent-specific queue
            self.pending_requests[request.agent_id].append(request)
            
            # Check if we should form a batch immediately
            if len(self.pending_requests[request.agent_id]) >= self.config.max_batch_size:
                asyncio.create_task(self._form_batch(request.agent_id))
        
        return request.request_id
    
    async def _batch_formation_loop(self):
        """Main loop for forming batches"""
        while self._running:
            try:
                # Check each agent's queue
                for agent_id, queue in list(self.pending_requests.items()):
                    if not queue:
                        continue
                    
                    # Check if oldest request has waited too long
                    oldest_request = queue[0]
                    wait_time = (datetime.utcnow() - oldest_request.timestamp).total_seconds() * 1000
                    
                    if wait_time >= self.config.max_wait_time_ms:
                        await self._form_batch(agent_id)
                    elif len(queue) >= self.config.max_batch_size:
                        await self._form_batch(agent_id)
                
                # Small sleep to prevent busy waiting
                await asyncio.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Error in batch formation loop: {e}")
                
    async def _form_batch(self, agent_id: str):
        """Form a batch for processing"""
        queue = self.pending_requests[agent_id]
        if not queue:
            return
        
        # Determine batch size dynamically
        if self.config.dynamic_batching:
            batch_size = self._calculate_optimal_batch_size(agent_id)
        else:
            batch_size = self.config.max_batch_size
        
        # Extract requests for batch
        batch = []
        for _ in range(min(batch_size, len(queue))):
            if queue:
                batch.append(queue.popleft())
        
        if not batch:
            return
        
        # Apply request merging if enabled
        if self.config.enable_request_merging:
            batch = self._merge_similar_requests(batch)
        
        # Track batch
        batch_id = f"{agent_id}_{int(time.time() * 1000)}"
        self.active_batches[batch_id] = batch
        self.batch_timers[batch_id] = time.time()
        
        # Update metrics
        self.metrics.total_batches += 1
        self.metrics.average_batch_size = self._update_average(
            self.metrics.average_batch_size,
            len(batch),
            self.metrics.total_batches
        )
        
        # Process batch
        asyncio.create_task(self._process_batch(batch_id, agent_id, batch))
        
    def _calculate_optimal_batch_size(self, agent_id: str) -> int:
        """Calculate optimal batch size based on current load"""
        
        # Factors to consider:
        # 1. Current CPU utilization
        # 2. Memory usage
        # 3. Queue depth
        # 4. Request priorities
        
        queue_depth = len(self.pending_requests[agent_id])
        
        # Start with max batch size
        optimal_size = self.config.max_batch_size
        
        # Adjust based on CPU utilization
        if self.metrics.cpu_utilization > 80:
            optimal_size = max(2, optimal_size // 2)
        elif self.metrics.cpu_utilization < 40:
            optimal_size = min(optimal_size * 2, 16)
        
        # Adjust based on memory
        if self.metrics.memory_utilization > 80:
            optimal_size = max(2, optimal_size // 2)
        
        # Consider queue depth
        if queue_depth < optimal_size:
            # Don't wait for full batch if queue is small
            if queue_depth <= 2:
                optimal_size = queue_depth
            else:
                optimal_size = max(2, queue_depth // 2)
        
        return optimal_size
    
    def _merge_similar_requests(self, batch: List[InferenceRequest]) -> List[InferenceRequest]:
        """Merge similar requests to reduce redundant computation"""
        
        if len(batch) <= 1:
            return batch
        
        # Group by similarity (simplified - in practice use embeddings)
        groups = defaultdict(list)
        
        for request in batch:
            # Simple similarity based on prompt prefix
            key = request.prompt[:50] if len(request.prompt) > 50 else request.prompt
            groups[key].append(request)
        
        # Keep one representative from each group
        merged_batch = []
        for group_requests in groups.values():
            # Keep the highest priority request as representative
            representative = max(group_requests, key=lambda r: r.priority)
            merged_batch.append(representative)
            
            # Mark others as duplicates for later response copying
            for req in group_requests:
                if req != representative:
                    req.merged_with = representative.request_id
        
        return merged_batch
    
    async def _process_batch(self, batch_id: str, agent_id: str, batch: List[InferenceRequest]):
        """Process a batch of requests"""
        
        start_time = time.time()
        
        try:
            # Prepare batch input
            prompts = [req.prompt for req in batch]
            
            # Simulate batch inference
            # In practice, this would call the actual model
            results = await self._run_batch_inference(agent_id, prompts)
            
            # Process results
            for request, result in zip(batch, results):
                # Cache result
                if self.config.enable_response_caching:
                    cache_key = self._get_cache_key(request)
                    self.response_cache[cache_key] = (result, datetime.utcnow())
                
                # Update metrics
                latency = (time.time() - start_time) * 1000
                self.latency_history.append(latency)
                
                # Handle merged requests
                if hasattr(request, 'merged_with'):
                    # Copy result to merged requests
                    pass
            
            # Update batch metrics
            batch_time = time.time() - self.batch_timers[batch_id]
            avg_wait = sum((r.timestamp - datetime.utcnow()).total_seconds() 
                          for r in batch) / len(batch) * 1000
            
            self.metrics.average_wait_time_ms = self._update_average(
                self.metrics.average_wait_time_ms,
                abs(avg_wait),
                self.metrics.total_batches
            )
            
        except Exception as e:
            logger.error(f"Error processing batch {batch_id}: {e}")
            
        finally:
            # Cleanup
            del self.active_batches[batch_id]
            del self.batch_timers[batch_id]
    
    async def _process_priority_request(self, request: InferenceRequest):
        """Process high-priority request immediately"""
        
        start_time = time.time()
        
        try:
            # Process individually without batching
            result = await self._run_single_inference(request.agent_id, request.prompt)
            
            # Update metrics
            latency = (time.time() - start_time) * 1000
            self.latency_history.append(latency)
            
            logger.info(f"Processed priority request {request.request_id} in {latency:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error processing priority request: {e}")
    
    async def _run_batch_inference(self, agent_id: str, prompts: List[str]) -> List[str]:
        """Run batch inference (simulation)"""
        
        # Simulate CPU-optimized batch processing
        await asyncio.sleep(0.05 * len(prompts))  # Simulate processing time
        
        # Return dummy results
        return [f"Response for: {prompt[:30]}..." for prompt in prompts]
    
    async def _run_single_inference(self, agent_id: str, prompt: str) -> str:
        """Run single inference (simulation)"""
        
        await asyncio.sleep(0.1)  # Simulate processing time
        return f"Priority response for: {prompt[:30]}..."
    
    def _get_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        
        # Include relevant parameters in cache key
        key_parts = [
            request.agent_id,
            request.prompt,
            str(request.max_tokens),
            str(request.temperature)
        ]
        
        return hashlib.md5('|'.join(key_parts).encode()).hexdigest()
    
    def _update_average(self, current_avg: float, new_value: float, count: int) -> float:
        """Update running average"""
        return ((current_avg * (count - 1)) + new_value) / count if count > 0 else new_value
    
    def _update_rate(self, current_rate: float, hit: bool) -> float:
        """Update hit rate with exponential moving average"""
        alpha = 0.1  # Smoothing factor
        value = 1.0 if hit else 0.0
        return alpha * value + (1 - alpha) * current_rate
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics"""
        
        # Calculate percentiles
        if self.latency_history:
            latencies = sorted(self.latency_history)
            p95_idx = int(len(latencies) * 0.95)
            p99_idx = int(len(latencies) * 0.99)
            
            self.metrics.p95_latency_ms = latencies[p95_idx] if p95_idx < len(latencies) else 0
            self.metrics.p99_latency_ms = latencies[p99_idx] if p99_idx < len(latencies) else 0
        
        # Calculate throughput
        runtime_seconds = time.time() - self.start_time
        self.metrics.throughput_rps = self.metrics.total_requests / runtime_seconds if runtime_seconds > 0 else 0
        
        return {
            'total_requests': self.metrics.total_requests,
            'total_batches': self.metrics.total_batches,
            'average_batch_size': round(self.metrics.average_batch_size, 2),
            'average_wait_time_ms': round(self.metrics.average_wait_time_ms, 2),
            'cache_hit_rate': round(self.metrics.cache_hit_rate, 3),
            'throughput_rps': round(self.metrics.throughput_rps, 2),
            'p95_latency_ms': round(self.metrics.p95_latency_ms, 2),
            'p99_latency_ms': round(self.metrics.p99_latency_ms, 2),
            'active_batches': len(self.active_batches),
            'pending_requests': sum(len(q) for q in self.pending_requests.values())
        }
    
    def optimize_for_workload(self, workload_profile: Dict[str, Any]):
        """Adjust configuration based on workload characteristics"""
        
        avg_request_rate = workload_profile.get('avg_request_rate', 100)
        peak_request_rate = workload_profile.get('peak_request_rate', 500)
        latency_sla_ms = workload_profile.get('latency_sla_ms', 100)
        
        # Adjust batch size
        if avg_request_rate > 200:
            self.config.max_batch_size = min(16, self.config.max_batch_size * 2)
        elif avg_request_rate < 50:
            self.config.max_batch_size = max(4, self.config.max_batch_size // 2)
        
        # Adjust wait time
        if latency_sla_ms < 50:
            self.config.max_wait_time_ms = min(20, latency_sla_ms // 2)
        else:
            self.config.max_wait_time_ms = min(100, latency_sla_ms // 2)
        
        logger.info(f"Optimized config: batch_size={self.config.max_batch_size}, "
                   f"wait_time={self.config.max_wait_time_ms}ms")


async def main():
    """Demo batch processing optimization"""
    
    config = BatchConfig(
        max_batch_size=8,
        max_wait_time_ms=50,
        cpu_cores=12,
        dynamic_batching=True,
        enable_request_merging=True,
        enable_response_caching=True
    )
    
    optimizer = BatchProcessingOptimizer(config)
    await optimizer.start()
    
    # Simulate requests
    for i in range(100):
        request = InferenceRequest(
            request_id=f"req_{i}",
            agent_id=f"agent_{i % 10}",
            prompt=f"Test prompt {i % 20}",
            priority=np.random.randint(1, 10)
        )
        await optimizer.submit_request(request)
        await asyncio.sleep(0.01)  # Simulate request arrival pattern
    
    # Wait for processing
    await asyncio.sleep(2)
    
    # Get metrics
    metrics = optimizer.get_metrics()
    print(json.dumps(metrics, indent=2))
    
    await optimizer.stop()


if __name__ == "__main__":
    asyncio.run(main())