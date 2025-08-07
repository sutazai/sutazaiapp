#!/usr/bin/env python3
"""
Request Queue Management for SutazAI System
Manages concurrent request limits and queuing for resource-constrained environments

Features:
- Configurable concurrency limits and queue sizes
- Priority-based request handling
- Request timeout and retry mechanisms
- Fair scheduling and load balancing
- Comprehensive metrics and monitoring
- Optimized for limited hardware resources
"""

import asyncio
import time
import logging
from typing import Any, Callable, Optional, Dict, List, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from datetime import datetime, timedelta
import heapq
import uuid
from contextlib import asynccontextmanager
import weakref


logger = logging.getLogger(__name__)


class RequestPriority(IntEnum):
    """Request priority levels (lower values = higher priority)"""
    CRITICAL = 0
    HIGH = 1
    NORMAL = 2
    LOW = 3
    BACKGROUND = 4


class RequestStatus(Enum):
    """Request status tracking"""
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class QueuedRequest:
    """Represents a queued request"""
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: RequestPriority
    timeout: Optional[float]
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: RequestStatus = RequestStatus.QUEUED
    result: Any = None
    error: Optional[Exception] = None
    retry_count: int = 0
    max_retries: int = 0
    
    def __lt__(self, other):
        """For priority queue ordering"""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.created_at < other.created_at
    
    @property
    def queue_time(self) -> float:
        """Time spent in queue"""
        if self.started_at:
            return (self.started_at - self.created_at).total_seconds()
        return (datetime.utcnow() - self.created_at).total_seconds()
    
    @property
    def processing_time(self) -> Optional[float]:
        """Time spent processing"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            return (datetime.utcnow() - self.started_at).total_seconds()
        return None
    
    @property
    def total_time(self) -> float:
        """Total time from creation to completion"""
        end_time = self.completed_at or datetime.utcnow()
        return (end_time - self.created_at).total_seconds()


@dataclass
class QueueMetrics:
    """Queue performance metrics"""
    total_requests: int = 0
    completed_requests: int = 0
    failed_requests: int = 0
    timeout_requests: int = 0
    cancelled_requests: int = 0
    current_queue_size: int = 0
    max_queue_size: int = 0
    current_active: int = 0
    max_concurrent: int = 0
    average_queue_time: float = 0.0
    average_processing_time: float = 0.0
    total_queue_time: float = 0.0
    total_processing_time: float = 0.0
    throughput_per_second: float = 0.0
    last_reset: datetime = field(default_factory=datetime.utcnow)


class RequestTimeoutError(Exception):
    """Raised when a request times out"""
    def __init__(self, request_id: str, timeout: float):
        self.request_id = request_id
        self.timeout = timeout
        super().__init__(f"Request {request_id} timed out after {timeout}s")


class QueueFullError(Exception):
    """Raised when queue is full"""
    def __init__(self, max_size: int):
        self.max_size = max_size
        super().__init__(f"Queue is full (max size: {max_size})")


class RequestQueue:
    """
    Advanced request queue with priority handling and concurrency control
    
    Designed for resource-constrained environments with focus on:
    - Fair request scheduling
    - Configurable concurrency limits
    - Priority-based processing
    - Comprehensive monitoring and metrics
    """
    
    def __init__(self,
                 max_queue_size: int = 100,
                 max_concurrent: int = 3,
                 timeout: float = 300.0,
                 max_retries: int = 2,
                 name: str = "request_queue",
                 enable_background: bool = True):
        """
        Initialize request queue
        
        Args:
            max_queue_size: Maximum number of queued requests
            max_concurrent: Maximum concurrent requests
            timeout: Default timeout for requests (seconds)
            max_retries: Default retry count for failed requests
            name: Queue name for logging and metrics
        """
        self.max_queue_size = max_queue_size
        self.max_concurrent = max_concurrent
        self.default_timeout = timeout
        self.default_max_retries = max_retries
        self.name = name
        
        # Queue and tracking
        self._queue: List[QueuedRequest] = []
        self._active_requests: Dict[str, QueuedRequest] = {}
        self._completed_requests: Dict[str, QueuedRequest] = {}
        self._request_futures: Dict[str, asyncio.Future] = {}
        
        # Synchronization
        self._queue_lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._shutdown_event = asyncio.Event()
        
        # Metrics
        self.metrics = QueueMetrics(max_concurrent=max_concurrent)
        
        # Background tasks
        self._processor_task: Optional[asyncio.Task] = None
        self._metrics_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Configuration
        self._cleanup_interval = 300  # 5 minutes
        self._metrics_interval = 60   # 1 minute
        self._max_completed_history = 1000
        
        # Start background processing if enabled
        self._background_enabled = enable_background
        if self._background_enabled:
            self._start_background_tasks()
        
        logger.info(f"Request queue '{name}' initialized: max_queue={max_queue_size}, max_concurrent={max_concurrent}")
    
    def _start_background_tasks(self):
        """Start background processing tasks"""
        self._processor_task = asyncio.create_task(self._process_queue())
        self._metrics_task = asyncio.create_task(self._update_metrics())
        self._cleanup_task = asyncio.create_task(self._cleanup_completed())
    
    async def submit(self,
                    func: Callable,
                    *args,
                    priority: RequestPriority = RequestPriority.NORMAL,
                    timeout: Optional[float] = None,
                    max_retries: Optional[int] = None,
                    request_id: Optional[str] = None,
                    **kwargs) -> str:
        """
        Submit a request to the queue
        
        Args:
            func: Function to execute
            *args: Positional arguments for func
            priority: Request priority
            timeout: Request timeout (uses default if None)
            max_retries: Max retry attempts (uses default if None)
            request_id: Custom request ID (generated if None)
            **kwargs: Keyword arguments for func
            
        Returns:
            Request ID for tracking
            
        Raises:
            QueueFullError: When queue is at capacity
        """
        async with self._queue_lock:
            # Check queue capacity
            if len(self._queue) >= self.max_queue_size:
                raise QueueFullError(self.max_queue_size)
            
            # Generate request ID if not provided
            if not request_id:
                request_id = str(uuid.uuid4())
            
            # Create request
            request = QueuedRequest(
                id=request_id,
                func=func,
                args=args,
                kwargs=kwargs,
                priority=priority,
                timeout=timeout or self.default_timeout,
                max_retries=max_retries or self.default_max_retries
            )
            
            # Add to queue (heapq maintains priority order)
            heapq.heappush(self._queue, request)
            
            # Create future for result
            self._request_futures[request_id] = asyncio.Future()
            
            # Update metrics
            self.metrics.total_requests += 1
            self.metrics.current_queue_size = len(self._queue)
            self.metrics.max_queue_size = max(self.metrics.max_queue_size, len(self._queue))
            
            logger.debug(f"Request {request_id} queued with priority {priority.name}")
            
            return request_id
    
    async def get_result(self, request_id: str, timeout: Optional[float] = None) -> Any:
        """
        Get result of a submitted request
        
        Args:
            request_id: Request ID returned by submit()
            timeout: Timeout for waiting for result
            
        Returns:
            Request result
            
        Raises:
            asyncio.TimeoutError: If timeout exceeded
            Exception: Original exception from request processing
        """
        if request_id not in self._request_futures:
            raise ValueError(f"Unknown request ID: {request_id}")
        
        future = self._request_futures[request_id]
        
        try:
            if timeout:
                result = await asyncio.wait_for(future, timeout=timeout)
            else:
                result = await future
            
            return result
            
        except asyncio.TimeoutError:
            # Cancel the request if it's still queued
            await self._cancel_request(request_id)
            raise
        
        finally:
            # Clean up future
            self._request_futures.pop(request_id, None)
    
    async def submit_and_wait(self,
                             func: Callable,
                             *args,
                             priority: RequestPriority = RequestPriority.NORMAL,
                             timeout: Optional[float] = None,
                             max_retries: Optional[int] = None,
                             **kwargs) -> Any:
        """
        Submit request and wait for result (convenience method)
        
        Returns:
            Request result
        """
        request_id = await self.submit(
            func, *args,
            priority=priority,
            timeout=timeout,
            max_retries=max_retries,
            **kwargs
        )
        
        return await self.get_result(request_id, timeout)
    
    async def _cancel_request(self, request_id: str):
        """Cancel a queued or active request"""
        async with self._queue_lock:
            # Try to remove from queue
            for i, request in enumerate(self._queue):
                if request.id == request_id:
                    request.status = RequestStatus.CANCELLED
                    self._queue.pop(i)
                    heapq.heapify(self._queue)  # Restore heap property
                    self.metrics.cancelled_requests += 1
                    self.metrics.current_queue_size = len(self._queue)
                    
                    # Complete the future with cancellation
                    future = self._request_futures.get(request_id)
                    if future and not future.done():
                        future.cancel()
                    
                    logger.debug(f"Request {request_id} cancelled from queue")
                    return
            
            # Check if it's currently active
            if request_id in self._active_requests:
                request = self._active_requests[request_id]
                request.status = RequestStatus.CANCELLED
                self.metrics.cancelled_requests += 1
                
                # The processing task will handle cleanup
                logger.debug(f"Request {request_id} marked for cancellation")
    
    async def _process_queue(self):
        """Background task to process queued requests"""
        while not self._shutdown_event.is_set():
            try:
                # Get next request from queue
                request = None
                async with self._queue_lock:
                    if self._queue:
                        request = heapq.heappop(self._queue)
                        self.metrics.current_queue_size = len(self._queue)
                
                if request:
                    # Process request with concurrency control
                    asyncio.create_task(self._handle_request(request))
                else:
                    # No requests, wait a bit
                    await asyncio.sleep(0.1)
                    
            except Exception as e:
                logger.error(f"Error in queue processor: {e}")
                await asyncio.sleep(1)
    
    async def _handle_request(self, request: QueuedRequest):
        """Handle individual request with concurrency control"""
        # Acquire semaphore for concurrency control
        async with self._semaphore:
            if request.status == RequestStatus.CANCELLED:
                return
            
            # Move to active requests
            async with self._queue_lock:
                self._active_requests[request.id] = request
                self.metrics.current_active = len(self._active_requests)
            
            request.status = RequestStatus.PROCESSING
            request.started_at = datetime.utcnow()
            
            try:
                # Execute request with timeout
                if request.timeout:
                    result = await asyncio.wait_for(
                        self._execute_request(request),
                        timeout=request.timeout
                    )
                else:
                    result = await self._execute_request(request)
                
                # Mark as completed
                request.status = RequestStatus.COMPLETED
                request.result = result
                request.completed_at = datetime.utcnow()
                
                # Complete the future
                future = self._request_futures.get(request.id)
                if future and not future.done():
                    future.set_result(result)
                
                self.metrics.completed_requests += 1
                
            except asyncio.TimeoutError:
                request.status = RequestStatus.TIMEOUT
                request.error = RequestTimeoutError(request.id, request.timeout)
                request.completed_at = datetime.utcnow()
                
                # Complete future with timeout error
                future = self._request_futures.get(request.id)
                if future and not future.done():
                    future.set_exception(request.error)
                
                self.metrics.timeout_requests += 1
                logger.warning(f"Request {request.id} timed out after {request.timeout}s")
                
            except Exception as e:
                request.status = RequestStatus.FAILED
                request.error = e
                request.completed_at = datetime.utcnow()
                
                # Check if we should retry
                if request.retry_count < request.max_retries:
                    request.retry_count += 1
                    request.status = RequestStatus.QUEUED
                    request.started_at = None
                    request.completed_at = None
                    
                    # Re-queue for retry
                    async with self._queue_lock:
                        heapq.heappush(self._queue, request)
                        self.metrics.current_queue_size = len(self._queue)
                    
                    logger.info(f"Retrying request {request.id} (attempt {request.retry_count}/{request.max_retries})")
                    return
                
                # Complete future with error
                future = self._request_futures.get(request.id)
                if future and not future.done():
                    future.set_exception(e)
                
                self.metrics.failed_requests += 1
                logger.error(f"Request {request.id} failed: {e}")
                
            finally:
                # Move to completed requests
                async with self._queue_lock:
                    self._active_requests.pop(request.id, None)
                    self._completed_requests[request.id] = request
                    self.metrics.current_active = len(self._active_requests)
    
    async def _execute_request(self, request: QueuedRequest) -> Any:
        """Execute the actual request function"""
        import inspect
        
        if inspect.iscoroutinefunction(request.func):
            return await request.func(*request.args, **request.kwargs)
        else:
            # Run sync function in executor to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, request.func, *request.args, **request.kwargs)
    
    async def _update_metrics(self):
        """Background task to update metrics"""
        while not self._shutdown_event.is_set():
            try:
                # Calculate throughput
                now = datetime.utcnow()
                time_since_reset = (now - self.metrics.last_reset).total_seconds()
                
                if time_since_reset > 0:
                    self.metrics.throughput_per_second = self.metrics.completed_requests / time_since_reset
                
                # Calculate average times
                if self.metrics.completed_requests > 0:
                    self.metrics.average_queue_time = (
                        self.metrics.total_queue_time / self.metrics.completed_requests
                    )
                    self.metrics.average_processing_time = (
                        self.metrics.total_processing_time / self.metrics.completed_requests
                    )
                
                # Update queue time totals
                for request in self._completed_requests.values():
                    if hasattr(request, '_metrics_processed'):
                        continue
                    
                    if request.status == RequestStatus.COMPLETED:
                        self.metrics.total_queue_time += request.queue_time
                        if request.processing_time:
                            self.metrics.total_processing_time += request.processing_time
                    
                    request._metrics_processed = True
                
            except Exception as e:
                logger.error(f"Error updating metrics: {e}")
            
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._metrics_interval
                )
                break
            except asyncio.TimeoutError:
                continue
    
    async def _cleanup_completed(self):
        """Background task to cleanup old completed requests"""
        while not self._shutdown_event.is_set():
            try:
                async with self._queue_lock:
                    if len(self._completed_requests) > self._max_completed_history:
                        # Sort by completion time and keep only recent ones
                        sorted_requests = sorted(
                            self._completed_requests.items(),
                            key=lambda x: x[1].completed_at or datetime.min,
                            reverse=True
                        )
                        
                        # Keep only the most recent requests
                        to_keep = dict(sorted_requests[:self._max_completed_history])
                        removed_count = len(self._completed_requests) - len(to_keep)
                        
                        self._completed_requests = to_keep
                        
                        if removed_count > 0:
                            logger.debug(f"Cleaned up {removed_count} old completed requests")
                
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
            
            try:
                await asyncio.wait_for(
                    self._shutdown_event.wait(),
                    timeout=self._cleanup_interval
                )
                break
            except asyncio.TimeoutError:
                continue
    
    def get_stats(self) -> Dict[str, Any]:
        """Get queue statistics"""
        success_rate = 0.0
        if self.metrics.total_requests > 0:
            success_rate = self.metrics.completed_requests / self.metrics.total_requests
        
        return {
            "name": self.name,
            "max_queue_size": self.max_queue_size,
            "max_concurrent": self.max_concurrent,
            "current_queue_size": self.metrics.current_queue_size,
            "current_active": self.metrics.current_active,
            "total_requests": self.metrics.total_requests,
            "completed_requests": self.metrics.completed_requests,
            "failed_requests": self.metrics.failed_requests,
            "timeout_requests": self.metrics.timeout_requests,
            "cancelled_requests": self.metrics.cancelled_requests,
            "success_rate": success_rate,
            "average_queue_time": self.metrics.average_queue_time,
            "average_processing_time": self.metrics.average_processing_time,
            "throughput_per_second": self.metrics.throughput_per_second,
            "queue_utilization": len(self._queue) / self.max_queue_size,
            "concurrency_utilization": len(self._active_requests) / self.max_concurrent
        }
    
    def get_request_status(self, request_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific request"""
        # Check active requests
        if request_id in self._active_requests:
            request = self._active_requests[request_id]
        # Check completed requests
        elif request_id in self._completed_requests:
            request = self._completed_requests[request_id]
        # Check queued requests
        else:
            for req in self._queue:
                if req.id == request_id:
                    request = req
                    break
            else:
                return None
        
        return {
            "id": request.id,
            "status": request.status.value,
            "priority": request.priority.name,
            "created_at": request.created_at.isoformat(),
            "started_at": request.started_at.isoformat() if request.started_at else None,
            "completed_at": request.completed_at.isoformat() if request.completed_at else None,
            "queue_time": request.queue_time,
            "processing_time": request.processing_time,
            "total_time": request.total_time,
            "retry_count": request.retry_count,
            "max_retries": request.max_retries,
            "error": str(request.error) if request.error else None
        }
    
    async def cancel_request(self, request_id: str) -> bool:
        """Cancel a specific request"""
        await self._cancel_request(request_id)
        return True
    
    async def clear_queue(self):
        """Clear all queued requests"""
        async with self._queue_lock:
            cancelled_count = len(self._queue)
            
            # Cancel all queued requests
            for request in self._queue:
                request.status = RequestStatus.CANCELLED
                future = self._request_futures.get(request.id)
                if future and not future.done():
                    future.cancel()
            
            self._queue.clear()
            self.metrics.current_queue_size = 0
            self.metrics.cancelled_requests += cancelled_count
            
            logger.info(f"Cleared {cancelled_count} queued requests")
    
    async def wait_for_completion(self, timeout: Optional[float] = None):
        """Wait for all queued and active requests to complete"""
        start_time = time.time()
        
        while True:
            async with self._queue_lock:
                if not self._queue and not self._active_requests:
                    break
            
            if timeout and (time.time() - start_time) > timeout:
                raise asyncio.TimeoutError("Timeout waiting for queue completion")
            
            await asyncio.sleep(0.1)
    
    async def close(self):
        """Close the request queue and cleanup resources"""
        logger.info(f"Closing request queue '{self.name}'...")
        
        # Signal shutdown
        self._shutdown_event.set()
        
        # Ask background tasks to finish gracefully, then force-cancel if needed
        background_tasks = [self._processor_task, self._metrics_task, self._cleanup_task] if self._background_enabled else []
        for task in background_tasks:
            if not task:
                continue
            if task.done():
                continue
            try:
                # Give the task a moment to exit after shutdown event is set
                await asyncio.wait_for(task, timeout=1.0)
            except asyncio.TimeoutError:
                try:
                    task.cancel()
                    await task
                except asyncio.CancelledError:
                    pass
                except Exception as e:
                    logger.debug(f"Background task forced shutdown error: {e}")
        
        # Cancel all pending futures
        for future in list(self._request_futures.values()):
            if not future.done():
                future.cancel()
        self._request_futures.clear()
        
        # Clear queues
        await self.clear_queue()
        
        logger.info(f"Request queue '{self.name}' closed")
    
    def __len__(self) -> int:
        """Get current queue size"""
        return len(self._queue)
    
    def __bool__(self) -> bool:
        """Check if queue has requests"""
        return len(self._queue) > 0 or len(self._active_requests) > 0
    
    def __repr__(self) -> str:
        return (f"RequestQueue(name='{self.name}', queued={len(self._queue)}, "
                f"active={len(self._active_requests)}, max_concurrent={self.max_concurrent})")


# Context manager for temporary queue
@asynccontextmanager
async def temporary_queue(max_concurrent: int = 3, **kwargs):
    """Create a temporary request queue for limited use"""
    queue = RequestQueue(max_concurrent=max_concurrent, **kwargs)
    try:
        yield queue
    finally:
        await queue.close()


# Factory functions for common use cases
def create_ollama_queue(max_concurrent: int = 2) -> RequestQueue:
    """Create request queue optimized for Ollama requests"""
    return RequestQueue(
        max_queue_size=50,  # Conservative for limited hardware
        max_concurrent=max_concurrent,
        timeout=300.0,      # 5 minutes for model generation
        max_retries=2,
        name="ollama_request_queue"
    )


def create_api_queue(max_concurrent: int = 5) -> RequestQueue:
    """Create request queue optimized for API requests"""
    return RequestQueue(
        max_queue_size=100,
        max_concurrent=max_concurrent,
        timeout=30.0,       # 30 seconds for API calls
        max_retries=3,
        name="api_request_queue"
    )
