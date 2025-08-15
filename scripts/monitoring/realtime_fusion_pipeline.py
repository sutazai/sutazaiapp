#!/usr/bin/env python3
"""
Real-Time Multi-Modal Fusion Processing Pipeline

This module implements a high-performance, real-time processing pipeline for 
multi-modal fusion in the SutazAI platform. It handles streaming data from 
multiple modalities with low latency and high throughput.

Key Features:
- Stream processing with configurable buffering
- Asynchronous processing with priority queues
- Load balancing across fusion strategies
- Auto-scaling based on system load
- Integration with SutazAI's agent ecosystem
- Real-time monitoring and metrics

Performance Targets:
- < 100ms latency for simple fusion
- > 1000 requests/second throughput
- Support for 174 concurrent consumers (aligned with Ollama config)

Author: SutazAI Multi-Modal Fusion System
Version: 1.0.0
"""

import asyncio
import logging
import time
import json
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable, AsyncGenerator
from dataclasses import dataclass, field
from enum import Enum
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import uuid
from pathlib import Path
from collections import defaultdict, deque
import heapq
import aiohttp
import websockets
import signal
import psutil
import gc
from contextlib import asynccontextmanager

from .multi_modal_fusion_coordinator import (
    MultiModalFusionCoordinator, ModalityType, ModalityData, 
    FusionResult, FusionStrategy
)
from .unified_representation import (
    UnifiedRepresentationFramework, UnifiedRepresentation, RepresentationLevel
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ProcessingPriority(Enum):
    """Processing priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

class PipelineStage(Enum):
    """Pipeline processing stages"""
    INGESTION = "ingestion"
    PREPROCESSING = "preprocessing"
    FUSION = "fusion"
    POSTPROCESSING = "postprocessing"
    OUTPUT = "output"

@dataclass
class ProcessingRequest:
    """Request for real-time processing"""
    request_id: str
    modality_data: Dict[ModalityType, ModalityData]
    fusion_strategy: FusionStrategy = FusionStrategy.HYBRID
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    callback: Optional[Callable] = None
    timeout: float = 10.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    
    def __lt__(self, other):
        # For priority queue ordering (higher priority first, then by timestamp)
        if self.priority.value != other.priority.value:
            return self.priority.value > other.priority.value
        return self.timestamp < other.timestamp

@dataclass
class ProcessingResponse:
    """Response from real-time processing"""
    request_id: str
    fusion_result: Optional[FusionResult] = None
    unified_representation: Optional[UnifiedRepresentation] = None
    error: Optional[str] = None
    processing_latency: float = 0.0
    queue_wait_time: float = 0.0
    pipeline_metrics: Dict[str, float] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

@dataclass
class PipelineMetrics:
    """Real-time pipeline performance metrics"""
    requests_processed: int = 0
    requests_failed: int = 0
    average_latency: float = 0.0
    throughput_per_second: float = 0.0
    queue_sizes: Dict[str, int] = field(default_factory=dict)
    memory_usage_mb: float = 0.0
    cpu_usage_percent: float = 0.0
    active_workers: int = 0
    last_updated: float = field(default_factory=time.time)

class StreamBuffer:
    """Thread-safe circular buffer for streaming data"""
    
    def __init__(self, capacity: int = 1000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.lock = threading.RLock()
        self.condition = threading.Condition(self.lock)
    
    def put(self, item: Any, timeout: Optional[float] = None) -> bool:
        """Add item to buffer"""
        with self.condition:
            if len(self.buffer) >= self.capacity:
                if timeout is None:
                    return False
                
                # Wait for space with timeout
                start_time = time.time()
                while len(self.buffer) >= self.capacity:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        return False
                    self.condition.wait(remaining)
            
            self.buffer.append(item)
            self.condition.notify_all()
            return True
    
    def get(self, timeout: Optional[float] = None) -> Optional[Any]:
        """Get item from buffer"""
        with self.condition:
            if not self.buffer:
                if timeout is None:
                    return None
                
                # Wait for items with timeout
                start_time = time.time()
                while not self.buffer:
                    remaining = timeout - (time.time() - start_time)
                    if remaining <= 0:
                        return None
                    self.condition.wait(remaining)
            
            item = self.buffer.popleft()
            self.condition.notify_all()
            return item
    
    def get_batch(self, batch_size: int, timeout: Optional[float] = None) -> List[Any]:
        """Get multiple items from buffer"""
        batch = []
        end_time = time.time() + timeout if timeout else None
        
        while len(batch) < batch_size:
            remaining_timeout = None
            if end_time:
                remaining_timeout = max(0, end_time - time.time())
                if remaining_timeout <= 0:
                    break
            
            item = self.get(remaining_timeout)
            if item is None:
                break
            batch.append(item)
        
        return batch
    
    def size(self) -> int:
        """Get current buffer size"""
        with self.lock:
            return len(self.buffer)
    
    def clear(self):
        """Clear buffer"""
        with self.condition:
            self.buffer.clear()
            self.condition.notify_all()

class LoadBalancer:
    """Load balancer for distributing processing requests"""
    
    def __init__(self, strategies: List[str] = None):
        self.strategies = strategies or ['round_robin', 'least_connections', 'weighted_response_time']
        self.current_strategy = 'round_robin'
        self.worker_stats = defaultdict(lambda: {'connections': 0, 'response_times': deque(maxlen=100)})
        self.round_robin_index = 0
        self.lock = threading.Lock()
    
    def select_worker(self, available_workers: List[str]) -> str:
        """Select optimal worker based on current strategy"""
        if not available_workers:
            raise ValueError("No available workers")
        
        with self.lock:
            if self.current_strategy == 'round_robin':
                worker = available_workers[self.round_robin_index % len(available_workers)]
                self.round_robin_index += 1
                return worker
            
            elif self.current_strategy == 'least_connections':
                return min(available_workers, key=lambda w: self.worker_stats[w]['connections'])
            
            elif self.current_strategy == 'weighted_response_time':
                # Select worker with lowest average response time
                worker_avg_times = {}
                for worker in available_workers:
                    response_times = self.worker_stats[worker]['response_times']
                    avg_time = np.mean(response_times) if response_times else 0.0
                    worker_avg_times[worker] = avg_time
                
                return min(worker_avg_times.keys(), key=lambda w: worker_avg_times[w])
            
            else:
                return available_workers[0]  # Fallback
    
    def record_worker_usage(self, worker: str, response_time: float):
        """Record worker usage statistics"""
        with self.lock:
            self.worker_stats[worker]['connections'] += 1
            self.worker_stats[worker]['response_times'].append(response_time)
    
    def release_worker(self, worker: str):
        """Release worker connection"""
        with self.lock:
            if self.worker_stats[worker]['connections'] > 0:
                self.worker_stats[worker]['connections'] -= 1

class AutoScaler:
    """Auto-scaling system for pipeline workers"""
    
    def __init__(self, min_workers: int = 2, max_workers: int = 16):
        self.min_workers = min_workers
        self.max_workers = max_workers
        self.current_workers = min_workers
        self.scaling_history = deque(maxlen=100)
        self.last_scale_time = time.time()
        self.scale_cooldown = 30.0  # seconds
    
    def should_scale_up(self, metrics: PipelineMetrics) -> bool:
        """Determine if we should scale up workers"""
        # Check if cooldown period has passed
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Scale up conditions
        conditions = [
            metrics.cpu_usage_percent > 80,
            metrics.average_latency > 1.0,  # > 1 second latency
            sum(metrics.queue_sizes.values()) > 100  # Large queue backlog
        ]
        
        return any(conditions) and self.current_workers < self.max_workers
    
    def should_scale_down(self, metrics: PipelineMetrics) -> bool:
        """Determine if we should scale down workers"""
        # Check if cooldown period has passed
        if time.time() - self.last_scale_time < self.scale_cooldown:
            return False
        
        # Scale down conditions (all must be true)
        conditions = [
            metrics.cpu_usage_percent < 30,
            metrics.average_latency < 0.1,  # < 100ms latency
            sum(metrics.queue_sizes.values()) < 10  # Small queue
        ]
        
        return all(conditions) and self.current_workers > self.min_workers
    
    def scale_workers(self, target_workers: int) -> int:
        """Scale to target number of workers"""
        target_workers = max(self.min_workers, min(self.max_workers, target_workers))
        
        if target_workers != self.current_workers:
            self.scaling_history.append({
                'timestamp': time.time(),
                'from': self.current_workers,
                'to': target_workers,
                'reason': 'auto_scale'
            })
            
            self.current_workers = target_workers
            self.last_scale_time = time.time()
            
            logger.info(f"Auto-scaled workers from {self.scaling_history[-1]['from']} to {target_workers}")
        
        return self.current_workers

class RealTimeFusionPipeline:
    """
    High-performance real-time multi-modal fusion pipeline
    
    Integrates with SutazAI infrastructure:
    - Ollama for text processing (supports 174 concurrent consumers)
    - Jarvis for voice interface
    - Agent orchestration system
    - Vector databases for embeddings
    - Knowledge graph for semantic understanding
    """
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/config/fusion_config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Core components
        self.fusion_coordinator = MultiModalFusionCoordinator(config_path)
        self.representation_framework = UnifiedRepresentationFramework(config_path)
        
        # Pipeline stages
        self.ingestion_buffer = StreamBuffer(self.config.get('buffer_size', 1000))
        self.processing_queue = queue.PriorityQueue(maxsize=self.config.get('queue_size', 500))
        self.output_buffer = StreamBuffer(self.config.get('output_buffer_size', 500))
        
        # Workers and scaling
        self.thread_executor = ThreadPoolExecutor(
            max_workers=self.config.get('max_threads', 16),
            thread_name_prefix='fusion-worker'
        )
        self.process_executor = ProcessPoolExecutor(
            max_workers=self.config.get('max_processes', 8)
        )
        
        self.load_balancer = LoadBalancer()
        self.auto_scaler = AutoScaler(
            min_workers=self.config.get('min_workers', 4),
            max_workers=self.config.get('max_workers', 32)
        )
        
        # Pipeline state
        self.is_running = False
        self.worker_tasks = []
        self.metrics = PipelineMetrics()
        self.processing_callbacks = {}
        
        # Monitoring
        self.metrics_lock = threading.Lock()
        self.metrics_history = deque(maxlen=1000)
        
        # WebSocket connections for real-time updates
        self.websocket_connections = set()
        
        # Signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        logger.info("Real-Time Fusion Pipeline initialized")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load pipeline configuration"""
        default_config = {
            'buffer_size': 2000,
            'queue_size': 1000,
            'output_buffer_size': 1000,
            'max_threads': 16,
            'max_processes': 8,
            'min_workers': 4,
            'max_workers': 32,
            'batch_size': 16,
            'max_latency_ms': 100,
            'throughput_target': 1000,
            'enable_auto_scaling': True,
            'enable_load_balancing': True,
            'enable_metrics': True,
            'metrics_interval': 1.0,
            'websocket_port': 8765,
            'enable_websocket': True,
            'memory_limit_mb': 4096,
            'cpu_limit_percent': 90
        }
        
        if self.config_path.exists():
            try:
                import yaml
                with open(self.config_path) as f:
                    loaded_config = yaml.safe_load(f)
                default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Failed to load pipeline config: {e}")
        
        return default_config
    
    async def start_pipeline(self):
        """Start the real-time processing pipeline"""
        if self.is_running:
            logger.warning("Pipeline is already running")
            return
        
        logger.info("Starting Real-Time Fusion Pipeline")
        self.is_running = True
        
        # Start worker tasks
        num_workers = self.auto_scaler.current_workers
        
        for i in range(num_workers):
            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.worker_tasks.append(task)
        
        # Start monitoring tasks
        if self.config.get('enable_metrics', True):
            asyncio.create_task(self._metrics_loop())
        
        if self.config.get('enable_auto_scaling', True):
            asyncio.create_task(self._auto_scaling_loop())
        
        # Start WebSocket server for real-time updates
        if self.config.get('enable_websocket', True):
            asyncio.create_task(self._start_websocket_server())
        
        logger.info(f"Pipeline started with {num_workers} workers")
    
    async def stop_pipeline(self):
        """Stop the real-time processing pipeline"""
        if not self.is_running:
            return
        
        logger.info("Stopping Real-Time Fusion Pipeline")
        self.is_running = False
        
        # Cancel all worker tasks
        for task in self.worker_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Clear buffers
        self.ingestion_buffer.clear()
        self.output_buffer.clear()
        
        # Shutdown executors
        self.thread_executor.shutdown(wait=True)
        self.process_executor.shutdown(wait=True)
        
        logger.info("Pipeline stopped")
    
    async def process_request(self, request: ProcessingRequest) -> ProcessingResponse:
        """Process a single fusion request"""
        start_time = time.time()
        
        try:
            # Add to processing queue
            queue_start = time.time()
            self.processing_queue.put(request, timeout=request.timeout)
            queue_wait_time = time.time() - queue_start
            
            # Wait for processing to complete
            response = await self._wait_for_response(request.request_id, request.timeout)
            response.queue_wait_time = queue_wait_time
            
            return response
            
        except queue.Full:
            return ProcessingResponse(
                request_id=request.request_id,
                error="Processing queue is full",
                processing_latency=time.time() - start_time
            )
        except asyncio.TimeoutError:
            return ProcessingResponse(
                request_id=request.request_id,
                error="Processing timeout",
                processing_latency=time.time() - start_time
            )
        except Exception as e:
            logger.error(f"Request processing failed: {e}")
            return ProcessingResponse(
                request_id=request.request_id,
                error=str(e),
                processing_latency=time.time() - start_time
            )
    
    async def process_stream(self, 
                           modality_stream: AsyncGenerator[Dict[ModalityType, ModalityData], None],
                           fusion_strategy: FusionStrategy = FusionStrategy.HYBRID) -> AsyncGenerator[ProcessingResponse, None]:
        """Process streaming multi-modal data"""
        async for modality_data in modality_stream:
            request = ProcessingRequest(
                request_id=str(uuid.uuid4()),
                modality_data=modality_data,
                fusion_strategy=fusion_strategy,
                priority=ProcessingPriority.NORMAL
            )
            
            response = await self.process_request(request)
            yield response
    
    async def _worker_loop(self, worker_id: str):
        """Main worker processing loop"""
        logger.info(f"Worker {worker_id} started")
        
        while self.is_running:
            try:
                # Get request from queue with timeout
                try:
                    request = self.processing_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                
                # Record worker usage
                self.load_balancer.record_worker_usage(worker_id, 0.0)
                
                # Process the request
                start_time = time.time()
                response = await self._process_fusion_request(request, worker_id)
                processing_time = time.time() - start_time
                
                # Update worker stats
                self.load_balancer.record_worker_usage(worker_id, processing_time)
                self.load_balancer.release_worker(worker_id)
                
                # Store response for callback
                self.processing_callbacks[request.request_id] = response
                
                # Update metrics
                await self._update_metrics(processing_time, success=response.error is None)
                
                # Mark task as done
                self.processing_queue.task_done()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
                # Mark task as done even on error
                try:
                    self.processing_queue.task_done()
                except (IOError, OSError, FileNotFoundError) as e:
                    # Suppressed exception (was bare except)
                    logger.debug(f"Suppressed exception: {e}")
                    pass
        
        logger.info(f"Worker {worker_id} stopped")
    
    async def _process_fusion_request(self, request: ProcessingRequest, worker_id: str) -> ProcessingResponse:
        """Process individual fusion request"""
        start_time = time.time()
        
        try:
            # Stage 1: Preprocessing
            preprocessing_start = time.time()
            # Add any preprocessing logic here
            preprocessing_time = time.time() - preprocessing_start
            
            # Stage 2: Fusion
            fusion_start = time.time()
            fusion_result = await self.fusion_coordinator.process_multi_modal_input(
                modality_data=request.modality_data,
                fusion_strategy=request.fusion_strategy,
                session_id=request.request_id
            )
            fusion_time = time.time() - fusion_start
            
            # Stage 3: Unified Representation
            repr_start = time.time()
            unified_repr = await self.representation_framework.create_unified_representation(
                modality_data=request.modality_data,
                representation_level=RepresentationLevel.UNIFIED,
                session_id=request.request_id
            )
            repr_time = time.time() - repr_start
            
            # Stage 4: Postprocessing
            postprocessing_start = time.time()
            # Add any postprocessing logic here
            postprocessing_time = time.time() - postprocessing_start
            
            # Create response
            total_time = time.time() - start_time
            
            response = ProcessingResponse(
                request_id=request.request_id,
                fusion_result=fusion_result,
                unified_representation=unified_repr,
                processing_latency=total_time,
                pipeline_metrics={
                    'preprocessing_time': preprocessing_time,
                    'fusion_time': fusion_time,
                    'representation_time': repr_time,
                    'postprocessing_time': postprocessing_time,
                    'worker_id': worker_id
                }
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Fusion processing failed for request {request.request_id}: {e}")
            return ProcessingResponse(
                request_id=request.request_id,
                error=str(e),
                processing_latency=time.time() - start_time,
                pipeline_metrics={'worker_id': worker_id}
            )
    
    async def _wait_for_response(self, request_id: str, timeout: float) -> ProcessingResponse:
        """Wait for processing response"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if request_id in self.processing_callbacks:
                response = self.processing_callbacks.pop(request_id)
                return response
            
            await asyncio.sleep(0.01)  # 10ms polling interval
        
        raise asyncio.TimeoutError(f"No response received for request {request_id}")
    
    async def _metrics_loop(self):
        """Continuous metrics collection loop"""
        while self.is_running:
            try:
                await self._collect_metrics()
                await asyncio.sleep(self.config.get('metrics_interval', 1.0))
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5.0)
    
    async def _collect_metrics(self):
        """Collect current pipeline metrics"""
        with self.metrics_lock:
            # System metrics
            process = psutil.Process()
            memory_info = process.memory_info()
            
            self.metrics.memory_usage_mb = memory_info.rss / (1024 * 1024)
            self.metrics.cpu_usage_percent = process.cpu_percent()
            
            # Queue sizes
            self.metrics.queue_sizes = {
                'ingestion': self.ingestion_buffer.size(),
                'processing': self.processing_queue.qsize(),
                'output': self.output_buffer.size()
            }
            
            # Worker count
            self.metrics.active_workers = len([task for task in self.worker_tasks if not task.done()])
            
            # Update timestamp
            self.metrics.last_updated = time.time()
            
            # Store in history
            self.metrics_history.append({
                'timestamp': self.metrics.last_updated,
                'metrics': self.metrics.__dict__.copy()
            })
        
        # Send metrics to WebSocket clients
        if self.websocket_connections:
            metrics_json = json.dumps(self.metrics.__dict__, default=str)
            disconnected = set()
            
            for websocket in self.websocket_connections:
                try:
                    await websocket.send(metrics_json)
                except websockets.exceptions.ConnectionClosed:
                    disconnected.add(websocket)
            
            # Remove disconnected clients
            self.websocket_connections -= disconnected
    
    async def _update_metrics(self, processing_time: float, success: bool):
        """Update processing metrics"""
        with self.metrics_lock:
            if success:
                self.metrics.requests_processed += 1
            else:
                self.metrics.requests_failed += 1
            
            # Update average latency (exponential moving average)
            alpha = 0.1
            self.metrics.average_latency = (
                alpha * processing_time + 
                (1 - alpha) * self.metrics.average_latency
            )
            
            # Calculate throughput
            current_time = time.time()
            if hasattr(self, '_last_throughput_update'):
                time_diff = current_time - self._last_throughput_update
                if time_diff >= 1.0:  # Update every second
                    requests_in_period = self.metrics.requests_processed - getattr(self, '_last_request_count', 0)
                    self.metrics.throughput_per_second = requests_in_period / time_diff
                    self._last_request_count = self.metrics.requests_processed
                    self._last_throughput_update = current_time
            else:
                self._last_throughput_update = current_time
                self._last_request_count = self.metrics.requests_processed
    
    async def _auto_scaling_loop(self):
        """Auto-scaling management loop"""
        while self.is_running:
            try:
                if self.auto_scaler.should_scale_up(self.metrics):
                    new_worker_count = self.auto_scaler.scale_workers(self.auto_scaler.current_workers + 2)
                    if new_worker_count > len(self.worker_tasks):
                        # Add new workers
                        for i in range(len(self.worker_tasks), new_worker_count):
                            task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
                            self.worker_tasks.append(task)
                
                elif self.auto_scaler.should_scale_down(self.metrics):
                    new_worker_count = self.auto_scaler.scale_workers(self.auto_scaler.current_workers - 1)
                    if new_worker_count < len(self.worker_tasks):
                        # Cancel excess workers
                        for i in range(new_worker_count, len(self.worker_tasks)):
                            self.worker_tasks[i].cancel()
                        self.worker_tasks = self.worker_tasks[:new_worker_count]
                
                await asyncio.sleep(10.0)  # Check every 10 seconds
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Auto-scaling error: {e}")
                await asyncio.sleep(30.0)
    
    async def _start_websocket_server(self):
        """Start WebSocket server for real-time monitoring"""
        async def handle_websocket(websocket, path):
            self.websocket_connections.add(websocket)
            logger.info(f"WebSocket client connected: {websocket.remote_address}")
            
            try:
                # Send initial metrics
                metrics_json = json.dumps(self.metrics.__dict__, default=str)
                await websocket.send(metrics_json)
                
                # Keep connection alive
                await websocket.wait_closed()
            except websockets.exceptions.ConnectionClosed:
                pass
            finally:
                self.websocket_connections.discard(websocket)
                logger.info(f"WebSocket client disconnected: {websocket.remote_address}")
        
        try:
            port = self.config.get('websocket_port', 8765)
            server = await websockets.serve(handle_websocket, "0.0.0.0", port)
            logger.info(f"WebSocket server started on port {port}")
            
            # Keep server running
            await server.wait_closed()
        except Exception as e:
            logger.error(f"WebSocket server error: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        asyncio.create_task(self.stop_pipeline())
    
    def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status"""
        return {
            'is_running': self.is_running,
            'metrics': self.metrics.__dict__,
            'worker_count': len(self.worker_tasks),
            'active_workers': self.metrics.active_workers,
            'auto_scaler_config': {
                'min_workers': self.auto_scaler.min_workers,
                'max_workers': self.auto_scaler.max_workers,
                'current_workers': self.auto_scaler.current_workers
            },
            'queue_sizes': self.metrics.queue_sizes,
            'websocket_connections': len(self.websocket_connections)
        }
    
    def get_metrics_history(self, minutes: int = 60) -> List[Dict[str, Any]]:
        """Get metrics history for specified time period"""
        cutoff_time = time.time() - (minutes * 60)
        
        with self.metrics_lock:
            filtered_history = [
                entry for entry in self.metrics_history
                if entry['timestamp'] >= cutoff_time
            ]
        
        return filtered_history
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform pipeline health check"""
        health_status = {
            'status': 'healthy',
            'timestamp': time.time(),
            'checks': {}
        }
        
        # Check if pipeline is running
        health_status['checks']['pipeline_running'] = self.is_running
        
        # Check memory usage
        memory_ok = self.metrics.memory_usage_mb < self.config.get('memory_limit_mb', 4096)
        health_status['checks']['memory_usage'] = {
            'status': 'ok' if memory_ok else 'warning',
            'current_mb': self.metrics.memory_usage_mb,
            'limit_mb': self.config.get('memory_limit_mb', 4096)
        }
        
        # Check CPU usage
        cpu_ok = self.metrics.cpu_usage_percent < self.config.get('cpu_limit_percent', 90)
        health_status['checks']['cpu_usage'] = {
            'status': 'ok' if cpu_ok else 'warning',
            'current_percent': self.metrics.cpu_usage_percent,
            'limit_percent': self.config.get('cpu_limit_percent', 90)
        }
        
        # Check queue sizes
        total_queue_size = sum(self.metrics.queue_sizes.values())
        queue_ok = total_queue_size < 1000
        health_status['checks']['queue_sizes'] = {
            'status': 'ok' if queue_ok else 'warning',
            'total_size': total_queue_size,
            'details': self.metrics.queue_sizes
        }
        
        # Check worker health
        workers_ok = self.metrics.active_workers >= self.auto_scaler.min_workers
        health_status['checks']['workers'] = {
            'status': 'ok' if workers_ok else 'error',
            'active_workers': self.metrics.active_workers,
            'min_workers': self.auto_scaler.min_workers
        }
        
        # Overall status
        if not self.is_running or not workers_ok:
            health_status['status'] = 'error'
        elif not memory_ok or not cpu_ok or not queue_ok:
            health_status['status'] = 'warning'
        
        return health_status

# Example usage and testing
async def main():
    """Example usage of Real-Time Fusion Pipeline"""
    
    # Initialize pipeline
    pipeline = RealTimeFusionPipeline()
    
    try:
        # Start pipeline
        await pipeline.start_pipeline()
        
        logger.info("Pipeline started. Processing sample requests...")
        
        # Process sample requests
        for i in range(10):
            # Create sample multi-modal data
            text_data = ModalityData(
                modality_type=ModalityType.TEXT,
                data=f"Sample text request {i}",
                timestamp=time.time(),
                confidence=0.9
            )
            
            voice_data = ModalityData(
                modality_type=ModalityType.VOICE,
                data=f"[audio_sample_{i}]",
                timestamp=time.time(),
                confidence=0.8
            )
            
            # Create processing request
            request = ProcessingRequest(
                request_id=f"test_request_{i}",
                modality_data={
                    ModalityType.TEXT: text_data,
                    ModalityType.VOICE: voice_data
                },
                fusion_strategy=FusionStrategy.HYBRID,
                priority=ProcessingPriority.NORMAL
            )
            
            # Process request
            response = await pipeline.process_request(request)
            
            logger.error(f"Request {i}: "
                  f"Latency: {response.processing_latency:.3f}s, "
                  f"Success: {response.error is None}")
        
        # Get pipeline status
        status = pipeline.get_pipeline_status()
        logger.info(f"\nPipeline Status:")
        logger.info(f"  Running: {status['is_running']}")
        logger.info(f"  Workers: {status['active_workers']}/{status['worker_count']}")
        logger.info(f"  Throughput: {status['metrics']['throughput_per_second']:.1f} req/s")
        logger.info(f"  Avg Latency: {status['metrics']['average_latency']:.3f}s")
        logger.info(f"  Memory Usage: {status['metrics']['memory_usage_mb']:.1f} MB")
        
        # Health check
        health = await pipeline.health_check()
        logger.info(f"\nHealth Status: {health['status']}")
        
        # Let pipeline run for a few seconds
        await asyncio.sleep(5)
        
    finally:
        # Stop pipeline
        await pipeline.stop_pipeline()
        logger.info("Pipeline stopped")

if __name__ == "__main__":
    asyncio.run(main())