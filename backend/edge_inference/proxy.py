"""
Edge Inference Proxy - High-performance inference routing and optimization for edge deployment
"""

import asyncio
import time
import threading
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib
import json
from datetime import datetime, timedelta
import aiohttp
from collections import defaultdict, deque
import weakref
import psutil
import numpy as np

logger = logging.getLogger(__name__)

class RoutingStrategy(Enum):
    """Inference routing strategies"""
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"
    FASTEST_RESPONSE = "fastest_response"
    MODEL_AFFINITY = "model_affinity"
    RESOURCE_AWARE = "resource_aware"
    GEOGRAPHIC = "geographic"

class ModelState(Enum):
    """Model loading states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    LOADED = "loaded"
    WARMING = "warming"
    READY = "ready"
    ERROR = "error"

@dataclass
class EdgeNode:
    """Represents an edge inference node"""
    node_id: str
    endpoint: str
    capabilities: Dict[str, Any]
    current_load: float = 0.0
    avg_response_time: float = 0.0
    last_heartbeat: datetime = field(default_factory=datetime.now)
    health_score: float = 1.0
    location: Optional[str] = None
    cpu_cores: int = 0
    memory_gb: float = 0.0
    models_loaded: Set[str] = field(default_factory=set)
    max_concurrent: int = 10
    current_requests: int = 0

@dataclass
class InferenceRequest:
    """Inference request with metadata"""
    request_id: str
    model_name: str
    prompt: str
    parameters: Dict[str, Any]
    priority: int = 1
    timeout: float = 30.0
    created_at: datetime = field(default_factory=datetime.now)
    client_location: Optional[str] = None
    requires_gpu: bool = False
    context_length: int = 2048

@dataclass
class InferenceResult:
    """Inference result with metadata"""
    request_id: str
    response: str
    node_id: str
    processing_time: float
    queue_time: float
    model_load_time: float = 0.0
    tokens_generated: int = 0
    cached: bool = False

@dataclass
class ModelCache:
    """Model caching information"""
    model_name: str
    size_mb: float
    last_accessed: datetime
    access_count: int = 0
    load_time: float = 0.0
    quantized: bool = False
    compression_ratio: float = 1.0

class ResultCache:
    """LRU cache for inference results"""
    
    def __init__(self, max_size: int = 1000, ttl_seconds: int = 3600):
        self.max_size = max_size
        self.ttl_seconds = ttl_seconds
        self.cache: Dict[str, Tuple[str, datetime]] = {}
        self.access_order = deque()
        self._lock = threading.RLock()
    
    def _get_cache_key(self, request: InferenceRequest) -> str:
        """Generate cache key for request"""
        key_data = {
            "model": request.model_name,
            "prompt": request.prompt,
            "params": sorted(request.parameters.items())
        }
        return hashlib.sha256(json.dumps(key_data, sort_keys=True).encode()).hexdigest()
    
    def get(self, request: InferenceRequest) -> Optional[str]:
        """Get cached result if available and valid"""
        with self._lock:
            cache_key = self._get_cache_key(request)
            
            if cache_key in self.cache:
                result, timestamp = self.cache[cache_key]
                
                # Check TTL
                if datetime.now() - timestamp < timedelta(seconds=self.ttl_seconds):
                    # Update access order
                    if cache_key in self.access_order:
                        self.access_order.remove(cache_key)
                    self.access_order.append(cache_key)
                    return result
                else:
                    # Expired, remove
                    del self.cache[cache_key]
                    if cache_key in self.access_order:
                        self.access_order.remove(cache_key)
            
            return None
    
    def put(self, request: InferenceRequest, result: str) -> None:
        """Cache inference result"""
        with self._lock:
            cache_key = self._get_cache_key(request)
            
            # Evict oldest if at capacity
            while len(self.cache) >= self.max_size and self.access_order:
                oldest_key = self.access_order.popleft()
                if oldest_key in self.cache:
                    del self.cache[oldest_key]
            
            # Add new result
            self.cache[cache_key] = (result, datetime.now())
            self.access_order.append(cache_key)
    
    def clear(self) -> None:
        """Clear cache"""
        with self._lock:
            self.cache.clear()
            self.access_order.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        with self._lock:
            total_size = len(self.cache)
            expired_count = 0
            now = datetime.now()
            
            for _, timestamp in self.cache.values():
                if now - timestamp >= timedelta(seconds=self.ttl_seconds):
                    expired_count += 1
            
            return {
                "total_entries": total_size,
                "expired_entries": expired_count,
                "hit_ratio": getattr(self, '_hit_count', 0) / max(getattr(self, '_total_requests', 1), 1),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds
            }

class EdgeInferenceProxy:
    """Main edge inference proxy with intelligent routing and optimization"""
    
    def __init__(self, 
                 routing_strategy: RoutingStrategy = RoutingStrategy.RESOURCE_AWARE,
                 enable_batching: bool = True,
                 enable_caching: bool = True,
                 max_batch_size: int = 8,
                 batch_timeout: float = 0.1):
        
        self.routing_strategy = routing_strategy
        self.enable_batching = enable_batching
        self.enable_caching = enable_caching
        self.max_batch_size = max_batch_size
        self.batch_timeout = batch_timeout
        
        # Node management
        self.nodes: Dict[str, EdgeNode] = {}
        self.node_locks: Dict[str, asyncio.Lock] = {}
        
        # Request management
        self.pending_requests: Dict[str, InferenceRequest] = {}
        self.request_batches: Dict[str, List[InferenceRequest]] = defaultdict(list)
        self.batch_timers: Dict[str, asyncio.Handle] = {}
        
        # Caching
        self.result_cache = ResultCache() if enable_caching else None
        self.model_cache: Dict[str, ModelCache] = {}
        
        # Performance tracking
        self.metrics = {
            "total_requests": 0,
            "cache_hits": 0,
            "batch_requests": 0,
            "avg_response_time": 0.0,
            "node_failures": 0,
            "model_loads": 0
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._batch_processing_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"EdgeInferenceProxy initialized with {routing_strategy.value} routing")
    
    async def start(self) -> None:
        """Start the edge inference proxy"""
        if self._running:
            return
        
        self._running = True
        
        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._monitor_nodes())
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        if self.enable_batching:
            self._batch_processing_task = asyncio.create_task(self._batch_processing_loop())
        
        logger.info("Edge Inference Proxy started")
    
    async def stop(self) -> None:
        """Stop the edge inference proxy"""
        self._running = False
        
        # Cancel background tasks
        for task in [self._monitoring_task, self._batch_processing_task, self._health_check_task]:
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
        
        logger.info("Edge Inference Proxy stopped")
    
    def register_node(self, node: EdgeNode) -> None:
        """Register an edge inference node"""
        self.nodes[node.node_id] = node
        self.node_locks[node.node_id] = asyncio.Lock()
        logger.info(f"Registered edge node: {node.node_id} at {node.endpoint}")
    
    def unregister_node(self, node_id: str) -> None:
        """Unregister an edge inference node"""
        if node_id in self.nodes:
            del self.nodes[node_id]
            if node_id in self.node_locks:
                del self.node_locks[node_id]
            logger.info(f"Unregistered edge node: {node_id}")
    
    async def process_request(self, request: InferenceRequest) -> InferenceResult:
        """Main entry point for processing inference requests"""
        start_time = time.time()
        self.metrics["total_requests"] += 1
        
        try:
            # Check cache first
            if self.result_cache:
                cached_result = self.result_cache.get(request)
                if cached_result:
                    self.metrics["cache_hits"] += 1
                    return InferenceResult(
                        request_id=request.request_id,
                        response=cached_result,
                        node_id="cache",
                        processing_time=0.001,
                        queue_time=0.0,
                        cached=True
                    )
            
            # Handle batching if enabled
            if self.enable_batching and self._should_batch_request(request):
                return await self._add_to_batch(request)
            
            # Process immediately
            return await self._execute_request(request)
            
        except Exception as e:
            logger.error(f"Error processing request {request.request_id}: {e}")
            return InferenceResult(
                request_id=request.request_id,
                response=f"Error: {str(e)}",
                node_id="error",
                processing_time=time.time() - start_time,
                queue_time=0.0
            )
    
    def _should_batch_request(self, request: InferenceRequest) -> bool:
        """Determine if request should be batched"""
        # Don't batch high priority or urgent requests
        if request.priority > 5 or request.timeout < 2.0:
            return False
        
        # Don't batch very long prompts
        if len(request.prompt) > 2000:
            return False
        
        # Check if similar requests are pending
        batch_key = f"{request.model_name}_{request.priority}"
        return len(self.request_batches[batch_key]) < self.max_batch_size
    
    async def _add_to_batch(self, request: InferenceRequest) -> InferenceResult:
        """Add request to batch for processing"""
        batch_key = f"{request.model_name}_{request.priority}"
        self.request_batches[batch_key].append(request)
        
        # Set timer for batch processing if not already set
        if batch_key not in self.batch_timers:
            self.batch_timers[batch_key] = asyncio.get_event_loop().call_later(
                self.batch_timeout,
                lambda: asyncio.create_task(self._process_batch(batch_key))
            )
        
        # Wait for batch processing
        # In practice, you'd use a more sophisticated mechanism
        while request.request_id in [r.request_id for r in self.request_batches[batch_key]]:
            await asyncio.sleep(0.01)
        
        # Request should be processed by now
        return await self._get_result_for_request(request.request_id)
    
    async def _process_batch(self, batch_key: str) -> None:
        """Process a batch of requests"""
        if batch_key not in self.request_batches:
            return
        
        batch = self.request_batches[batch_key]
        if not batch:
            return
        
        self.metrics["batch_requests"] += len(batch)
        
        # Clear batch and timer
        del self.request_batches[batch_key]
        if batch_key in self.batch_timers:
            self.batch_timers[batch_key].cancel()
            del self.batch_timers[batch_key]
        
        # Process batch on optimal node
        model_name = batch[0].model_name
        node = await self._select_optimal_node(model_name, batch[0])
        
        if node:
            await self._execute_batch(batch, node)
        else:
            # Fallback to individual processing
            for request in batch:
                asyncio.create_task(self._execute_request(request))
    
    async def _execute_batch(self, batch: List[InferenceRequest], node: EdgeNode) -> None:
        """Execute a batch of requests on a node"""
        try:
            async with self.node_locks[node.node_id]:
                # Prepare batch request
                batch_prompts = [req.prompt for req in batch]
                
                # For simplicity, process sequentially (in production, use actual batch API)
                for request in batch:
                    result = await self._execute_single_request(request, node)
                    
                    # Cache result
                    if self.result_cache and result.response:
                        self.result_cache.put(request, result.response)
                    
        except Exception as e:
            logger.error(f"Batch execution failed on {node.node_id}: {e}")
            # Fallback to individual processing
            for request in batch:
                asyncio.create_task(self._execute_request(request))
    
    async def _execute_request(self, request: InferenceRequest) -> InferenceResult:
        """Execute a single inference request"""
        node = await self._select_optimal_node(request.model_name, request)
        
        if not node:
            raise Exception("No available nodes for inference")
        
        return await self._execute_single_request(request, node)
    
    async def _execute_single_request(self, request: InferenceRequest, node: EdgeNode) -> InferenceResult:
        """Execute request on specific node"""
        start_time = time.time()
        
        try:
            async with self.node_locks[node.node_id]:
                node.current_requests += 1
                
                # Ensure model is loaded
                model_load_time = 0.0
                if request.model_name not in node.models_loaded:
                    load_start = time.time()
                    await self._ensure_model_loaded(node, request.model_name)
                    model_load_time = time.time() - load_start
                    self.metrics["model_loads"] += 1
                
                # Execute inference
                response = await self._call_node_inference(node, request)
                
                processing_time = time.time() - start_time
                
                # Update node metrics
                node.avg_response_time = (node.avg_response_time * 0.8 + processing_time * 0.2)
                node.current_requests -= 1
                
                result = InferenceResult(
                    request_id=request.request_id,
                    response=response,
                    node_id=node.node_id,
                    processing_time=processing_time,
                    queue_time=0.0,
                    model_load_time=model_load_time,
                    tokens_generated=len(response.split()) if response else 0
                )
                
                # Cache result
                if self.result_cache and response:
                    self.result_cache.put(request, response)
                
                return result
                
        except Exception as e:
            logger.error(f"Request execution failed on {node.node_id}: {e}")
            node.current_requests = max(0, node.current_requests - 1)
            raise
    
    async def _select_optimal_node(self, model_name: str, request: InferenceRequest) -> Optional[EdgeNode]:
        """Select optimal node based on routing strategy"""
        available_nodes = [
            node for node in self.nodes.values()
            if (node.health_score > 0.5 and 
                node.current_requests < node.max_concurrent and
                self._node_can_handle_request(node, request))
        ]
        
        if not available_nodes:
            return None
        
        if self.routing_strategy == RoutingStrategy.ROUND_ROBIN:
            # Simple round-robin
            return min(available_nodes, key=lambda n: n.current_requests)
        
        elif self.routing_strategy == RoutingStrategy.LEAST_LOADED:
            return min(available_nodes, key=lambda n: n.current_load)
        
        elif self.routing_strategy == RoutingStrategy.FASTEST_RESPONSE:
            return min(available_nodes, key=lambda n: n.avg_response_time)
        
        elif self.routing_strategy == RoutingStrategy.MODEL_AFFINITY:
            # Prefer nodes that already have the model loaded
            with_model = [n for n in available_nodes if model_name in n.models_loaded]
            if with_model:
                return min(with_model, key=lambda n: n.current_load)
            return min(available_nodes, key=lambda n: n.current_load)
        
        elif self.routing_strategy == RoutingStrategy.RESOURCE_AWARE:
            return self._select_resource_aware_node(available_nodes, request)
        
        elif self.routing_strategy == RoutingStrategy.GEOGRAPHIC:
            return self._select_geographic_node(available_nodes, request)
        
        return available_nodes[0]
    
    def _node_can_handle_request(self, node: EdgeNode, request: InferenceRequest) -> bool:
        """Check if node can handle the request"""
        # Check GPU requirements
        if request.requires_gpu and not node.capabilities.get("gpu", False):
            return False
        
        # Check memory requirements (rough estimate)
        estimated_memory = request.context_length * 0.004  # 4MB per 1K context
        if estimated_memory > node.memory_gb * 0.8:  # Leave 20% headroom
            return False
        
        return True
    
    def _select_resource_aware_node(self, nodes: List[EdgeNode], request: InferenceRequest) -> EdgeNode:
        """Select node based on resource requirements and availability"""
        scores = []
        
        for node in nodes:
            score = 0.0
            
            # Load factor (lower is better)
            load_factor = 1.0 - (node.current_requests / node.max_concurrent)
            score += load_factor * 0.4
            
            # Response time factor (lower is better)
            response_factor = 1.0 / (1.0 + node.avg_response_time)
            score += response_factor * 0.3
            
            # Model affinity (higher if model already loaded)
            if request.model_name in node.models_loaded:
                score += 0.2
            
            # Health factor
            score += node.health_score * 0.1
            
            scores.append((score, node))
        
        return max(scores, key=lambda x: x[0])[1]
    
    def _select_geographic_node(self, nodes: List[EdgeNode], request: InferenceRequest) -> EdgeNode:
        """Select node based on geographic proximity"""
        if not request.client_location:
            return self._select_resource_aware_node(nodes, request)
        
        # For simplicity, just use lexicographic distance
        # In production, use proper geographic distance calculation
        def location_distance(node_loc: str, client_loc: str) -> float:
            if not node_loc:
                return float('inf')
            return abs(hash(node_loc) - hash(client_loc)) % 1000
        
        return min(nodes, key=lambda n: location_distance(n.location or "", request.client_location))
    
    async def _ensure_model_loaded(self, node: EdgeNode, model_name: str) -> None:
        """Ensure model is loaded on node"""
        if model_name in node.models_loaded:
            return
        
        try:
            # Call node to load model
            async with aiohttp.ClientSession() as session:
                load_data = {"model": model_name}
                async with session.post(
                    f"{node.endpoint}/api/load_model",
                    json=load_data,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        node.models_loaded.add(model_name)
                        logger.info(f"Loaded model {model_name} on node {node.node_id}")
                    else:
                        raise Exception(f"Failed to load model: HTTP {response.status}")
        
        except Exception as e:
            logger.error(f"Error loading model {model_name} on {node.node_id}: {e}")
            raise
    
    async def _call_node_inference(self, node: EdgeNode, request: InferenceRequest) -> str:
        """Call node for inference"""
        try:
            async with aiohttp.ClientSession() as session:
                inference_data = {
                    "model": request.model_name,
                    "prompt": request.prompt,
                    "stream": False,
                    "options": request.parameters
                }
                
                async with session.post(
                    f"{node.endpoint}/api/generate",
                    json=inference_data,
                    timeout=aiohttp.ClientTimeout(total=request.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("response", "")
                    else:
                        raise Exception(f"Inference failed: HTTP {response.status}")
        
        except Exception as e:
            logger.error(f"Error calling node {node.node_id}: {e}")
            raise
    
    async def _monitor_nodes(self) -> None:
        """Monitor node health and performance"""
        while self._running:
            try:
                for node in self.nodes.values():
                    await self._update_node_metrics(node)
                
                await asyncio.sleep(10)  # Monitor every 10 seconds
                
            except Exception as e:
                logger.error(f"Error in node monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _update_node_metrics(self, node: EdgeNode) -> None:
        """Update metrics for a node"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{node.endpoint}/api/health",
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as response:
                    if response.status == 200:
                        health_data = await response.json()
                        node.current_load = health_data.get("cpu_percent", 0.0) / 100.0
                        node.last_heartbeat = datetime.now()
                        node.health_score = min(1.0, 2.0 - node.current_load)  # Simple health score
                    else:
                        node.health_score *= 0.9  # Degrade health on failure
        
        except Exception:
            node.health_score *= 0.8  # More aggressive degradation on timeout
            if node.health_score < 0.1:
                logger.warning(f"Node {node.node_id} appears to be failing")
    
    async def _health_check_loop(self) -> None:
        """Periodic health checks and cleanup"""
        while self._running:
            try:
                # Remove unhealthy nodes
                unhealthy_nodes = [
                    node_id for node_id, node in self.nodes.items()
                    if node.health_score < 0.1 or 
                    (datetime.now() - node.last_heartbeat).total_seconds() > 60
                ]
                
                for node_id in unhealthy_nodes:
                    logger.warning(f"Removing unhealthy node: {node_id}")
                    self.unregister_node(node_id)
                    self.metrics["node_failures"] += 1
                
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(60)
    
    async def _batch_processing_loop(self) -> None:
        """Process batches that haven't been triggered by timeout"""
        while self._running:
            try:
                # Check for stale batches
                current_time = time.time()
                stale_batches = []
                
                for batch_key, requests in self.request_batches.items():
                    if requests and (current_time - requests[0].created_at.timestamp()) > self.batch_timeout * 2:
                        stale_batches.append(batch_key)
                
                for batch_key in stale_batches:
                    await self._process_batch(batch_key)
                
                await asyncio.sleep(self.batch_timeout / 2)
                
            except Exception as e:
                logger.error(f"Error in batch processing loop: {e}")
                await asyncio.sleep(1)
    
    async def _get_result_for_request(self, request_id: str) -> InferenceResult:
        """Get result for a specific request (placeholder)"""
        # In practice, you'd maintain a results store
        return InferenceResult(
            request_id=request_id,
            response="Batch processed",
            node_id="batch",
            processing_time=0.1,
            queue_time=0.05
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """Get proxy statistics"""
        node_stats = {
            "total_nodes": len(self.nodes),
            "healthy_nodes": len([n for n in self.nodes.values() if n.health_score > 0.5]),
            "total_requests_active": sum(n.current_requests for n in self.nodes.values()),
            "avg_node_load": np.mean([n.current_load for n in self.nodes.values()]) if self.nodes else 0.0
        }
        
        cache_stats = self.result_cache.get_stats() if self.result_cache else {}
        
        return {
            **self.metrics,
            **node_stats,
            "cache_stats": cache_stats,
            "routing_strategy": self.routing_strategy.value,
            "batching_enabled": self.enable_batching,
            "caching_enabled": self.enable_caching
        }
    
    def optimize_routing_strategy(self) -> None:
        """Dynamically optimize routing strategy based on performance"""
        # Analyze recent performance and adjust strategy
        # This is a placeholder for ML-based optimization
        if self.metrics["avg_response_time"] > 2.0:
            if self.routing_strategy != RoutingStrategy.FASTEST_RESPONSE:
                self.routing_strategy = RoutingStrategy.FASTEST_RESPONSE
                logger.info("Switched to FASTEST_RESPONSE routing due to high latency")