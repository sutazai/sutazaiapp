"""
Async and Concurrency Optimization for SutazAI
Advanced async patterns and concurrency management
"""

import asyncio
import logging
import time
import threading
from typing import Dict, List, Any, Optional, Callable, Awaitable
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import weakref
from functools import wraps

logger = logging.getLogger(__name__)

@dataclass
class ConcurrencyConfig:
    """Concurrency configuration"""
    max_concurrent_tasks: int = 100
    task_timeout: float = 30.0
    semaphore_limit: int = 50
    rate_limit_per_second: float = 100.0
    batch_size: int = 10
    queue_size: int = 1000

class AsyncOptimizer:
    """Advanced async and concurrency optimization"""
    
    def __init__(self, config: ConcurrencyConfig = None):
        self.config = config or ConcurrencyConfig()
        
        # Concurrency controls
        self.semaphore = asyncio.Semaphore(self.config.semaphore_limit)
        self.rate_limiter = asyncio.Semaphore(int(self.config.rate_limit_per_second))
        
        # Task management
        self.active_tasks = set()
        self.task_queue = asyncio.Queue(maxsize=self.config.queue_size)
        self.task_results = {}
        
        # Statistics
        self.stats = {
            "tasks_executed": 0,
            "tasks_completed": 0,
            "tasks_failed": 0,
            "total_execution_time": 0.0,
            "average_execution_time": 0.0,
            "concurrent_tasks_peak": 0
        }
        
        # Background workers
        self.workers_active = False
        self.worker_tasks = []
    
    async def initialize(self):
        """Initialize async optimizer"""
        logger.info("🔄 Initializing Async Optimizer")
        
        # Start background workers
        self.workers_active = True
        for i in range(min(10, self.config.max_concurrent_tasks // 10)):
            worker_task = asyncio.create_task(self._worker_loop(f"worker-{i}"))
            self.worker_tasks.append(worker_task)
        
        # Start rate limiter reset task
        asyncio.create_task(self._rate_limiter_reset())
        
        # Start statistics collector
        asyncio.create_task(self._stats_collector())
        
        logger.info("✅ Async Optimizer initialized")
    
    async def execute_with_concurrency_control(self, coro: Awaitable, task_id: str = None) -> Any:
        """Execute coroutine with concurrency control"""
        if task_id is None:
            task_id = f"task-{int(time.time() * 1000000)}"
        
        async with self.semaphore:
            async with self.rate_limiter:
                start_time = time.time()
                
                try:
                    # Track active task
                    task = asyncio.current_task()
                    self.active_tasks.add(task)
                    
                    # Update peak concurrent tasks
                    current_concurrent = len(self.active_tasks)
                    if current_concurrent > self.stats["concurrent_tasks_peak"]:
                        self.stats["concurrent_tasks_peak"] = current_concurrent
                    
                    # Execute with timeout
                    result = await asyncio.wait_for(coro, timeout=self.config.task_timeout)
                    
                    # Update statistics
                    execution_time = time.time() - start_time
                    self.stats["tasks_executed"] += 1
                    self.stats["tasks_completed"] += 1
                    self.stats["total_execution_time"] += execution_time
                    self.stats["average_execution_time"] = (
                        self.stats["total_execution_time"] / self.stats["tasks_executed"]
                    )
                    
                    # Store result
                    self.task_results[task_id] = {
                        "result": result,
                        "execution_time": execution_time,
                        "status": "completed"
                    }
                    
                    return result
                    
                except asyncio.TimeoutError:
                    self.stats["tasks_failed"] += 1
                    error_result = {
                        "error": "Task timeout",
                        "execution_time": time.time() - start_time,
                        "status": "timeout"
                    }
                    self.task_results[task_id] = error_result
                    raise
                    
                except Exception as e:
                    self.stats["tasks_failed"] += 1
                    error_result = {
                        "error": str(e),
                        "execution_time": time.time() - start_time,
                        "status": "failed"
                    }
                    self.task_results[task_id] = error_result
                    raise
                    
                finally:
                    # Remove from active tasks
                    if task in self.active_tasks:
                        self.active_tasks.remove(task)
    
    async def execute_batch(self, coroutines: List[Awaitable], batch_size: int = None) -> List[Any]:
        """Execute multiple coroutines in batches"""
        if batch_size is None:
            batch_size = self.config.batch_size
        
        results = []
        
        # Process in batches
        for i in range(0, len(coroutines), batch_size):
            batch = coroutines[i:i + batch_size]
            
            # Execute batch concurrently
            batch_tasks = [
                self.execute_with_concurrency_control(coro, f"batch-{i}-{j}")
                for j, coro in enumerate(batch)
            ]
            
            try:
                batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                results.extend(batch_results)
            except Exception as e:
                logger.error(f"Batch execution failed: {e}")
                results.extend([e] * len(batch))
        
        return results
    
    async def submit_task(self, coro: Awaitable, priority: int = 0) -> str:
        """Submit task to queue for background processing"""
        task_id = f"queued-{int(time.time() * 1000000)}"
        
        task_item = {
            "task_id": task_id,
            "coro": coro,
            "priority": priority,
            "submitted_at": time.time()
        }
        
        try:
            await self.task_queue.put(task_item)
            return task_id
        except asyncio.QueueFull:
            raise RuntimeError("Task queue is full")
    
    async def _worker_loop(self, worker_name: str):
        """Background worker loop"""
        logger.info(f"Started worker: {worker_name}")
        
        while self.workers_active:
            try:
                # Get task from queue with timeout
                task_item = await asyncio.wait_for(
                    self.task_queue.get(),
                    timeout=1.0
                )
                
                # Execute task
                await self.execute_with_concurrency_control(
                    task_item["coro"],
                    task_item["task_id"]
                )
                
                # Mark task as done
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                # No tasks in queue, continue
                continue
            except Exception as e:
                logger.error(f"Worker {worker_name} error: {e}")
                await asyncio.sleep(1)
    
    async def _rate_limiter_reset(self):
        """Reset rate limiter periodically"""
        while self.workers_active:
            await asyncio.sleep(1.0)
            
            # Release all rate limiter permits
            for _ in range(int(self.config.rate_limit_per_second)):
                try:
                    self.rate_limiter.release()
                except ValueError:
                    # Already at maximum
                    break
    
    async def _stats_collector(self):
        """Collect and log statistics"""
        while self.workers_active:
            await asyncio.sleep(60)  # Log stats every minute
            
            logger.info(f"Async Stats: {self.get_performance_stats()}")
    
    async def wait_for_all_tasks(self):
        """Wait for all queued tasks to complete"""
        await self.task_queue.join()
    
    def get_task_result(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get result of a completed task"""
        return self.task_results.get(task_id)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        queue_size = self.task_queue.qsize()
        
        return {
            **self.stats,
            "active_tasks": len(self.active_tasks),
            "queued_tasks": queue_size,
            "queue_utilization": (queue_size / self.config.queue_size) * 100,
            "success_rate": (
                self.stats["tasks_completed"] / max(1, self.stats["tasks_executed"]) * 100
            ),
            "workers_active": len(self.worker_tasks)
        }
    
    async def shutdown(self):
        """Shutdown async optimizer"""
        logger.info("🛑 Shutting down Async Optimizer")
        
        # Stop workers
        self.workers_active = False
        
        # Wait for current tasks to complete
        if self.worker_tasks:
            await asyncio.gather(*self.worker_tasks, return_exceptions=True)
        
        # Cancel any remaining active tasks
        for task in self.active_tasks.copy():
            if not task.done():
                task.cancel()
        
        logger.info("✅ Async Optimizer shutdown complete")

# Optimization decorators
def with_concurrency_control(max_concurrent: int = 10):
    """Decorator to add concurrency control to async functions"""
    semaphore = asyncio.Semaphore(max_concurrent)
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            async with semaphore:
                return await func(*args, **kwargs)
        return wrapper
    return decorator

def with_timeout(timeout_seconds: float):
    """Decorator to add timeout to async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await asyncio.wait_for(
                func(*args, **kwargs),
                timeout=timeout_seconds
            )
        return wrapper
    return decorator

def with_retry(max_retries: int = 3, delay: float = 1.0):
    """Decorator to add retry logic to async functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_retries:
                        await asyncio.sleep(delay * (2 ** attempt))  # Exponential backoff
                    else:
                        raise last_exception
        return wrapper
    return decorator

def batch_processor(batch_size: int = 10, timeout: float = 1.0):
    """Decorator to process items in batches"""
    def decorator(func):
        @wraps(func)
        async def wrapper(items: List[Any], *args, **kwargs):
            results = []
            
            for i in range(0, len(items), batch_size):
                batch = items[i:i + batch_size]
                
                try:
                    batch_result = await asyncio.wait_for(
                        func(batch, *args, **kwargs),
                        timeout=timeout
                    )
                    results.extend(batch_result if isinstance(batch_result, list) else [batch_result])
                except asyncio.TimeoutError:
                    logger.warning(f"Batch {i//batch_size} timed out")
                    results.extend([None] * len(batch))
                except Exception as e:
                    logger.error(f"Batch {i//batch_size} failed: {e}")
                    results.extend([e] * len(batch))
            
            return results
        return wrapper
    return decorator

# Global async optimizer instance
async_optimizer = AsyncOptimizer()
