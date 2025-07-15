"""
CPU and Memory Optimization for SutazAI
Advanced resource management and optimization
"""

import gc
import os
import psutil
import threading
import multiprocessing
import asyncio
import logging
import time
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass
from functools import wraps, lru_cache
import weakref
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

logger = logging.getLogger(__name__)

@dataclass
class ResourceLimits:
    """Resource usage limits"""
    max_memory_mb: int = 2048
    max_cpu_percent: float = 80.0
    max_threads: int = 50
    max_processes: int = None
    
    def __post_init__(self):
        if self.max_processes is None:
            self.max_processes = multiprocessing.cpu_count()

class CPUMemoryOptimizer:
    """CPU and memory optimization manager"""
    
    def __init__(self, limits: ResourceLimits = None):
        self.limits = limits or ResourceLimits()
        self.memory_usage_history = deque(maxlen=100)
        self.cpu_usage_history = deque(maxlen=100)
        self.gc_stats = {"collections": 0, "freed_objects": 0}
        self.optimization_active = False
        
        # Thread and process pools
        self.thread_pool = None
        self.process_pool = None
        
        # Memory cleanup callbacks
        self.cleanup_callbacks = []
        
    async def initialize(self):
        """Initialize CPU and memory optimizer"""
        logger.info("🔄 Initializing CPU and Memory Optimizer")
        
        # Configure garbage collection
        self._configure_garbage_collection()
        
        # Initialize thread and process pools
        self._initialize_pools()
        
        # Start monitoring
        self.optimization_active = True
        asyncio.create_task(self._monitoring_loop())
        
        # Start periodic optimization
        asyncio.create_task(self._optimization_loop())
        
        logger.info("✅ CPU and Memory Optimizer initialized")
    
    def _configure_garbage_collection(self):
        """Configure garbage collection for optimal performance"""
        try:
            # Set garbage collection thresholds
            gc.set_threshold(700, 10, 10)  # More aggressive GC
            
            # Enable automatic garbage collection
            gc.enable()
            
            logger.info("✅ Garbage collection configured")
        except Exception as e:
            logger.warning(f"GC configuration failed: {e}")
    
    def _initialize_pools(self):
        """Initialize thread and process pools"""
        try:
            # Thread pool for I/O bound tasks
            self.thread_pool = ThreadPoolExecutor(
                max_workers=self.limits.max_threads,
                thread_name_prefix="sutazai"
            )
            
            # Process pool for CPU bound tasks
            self.process_pool = ProcessPoolExecutor(
                max_workers=self.limits.max_processes
            )
            
            logger.info(f"✅ Resource pools initialized (threads: {self.limits.max_threads}, processes: {self.limits.max_processes})")
        except Exception as e:
            logger.error(f"Pool initialization failed: {e}")
    
    async def _monitoring_loop(self):
        """Monitor resource usage"""
        while self.optimization_active:
            try:
                # Get current resource usage
                process = psutil.Process()
                memory_info = process.memory_info()
                cpu_percent = process.cpu_percent()
                
                memory_mb = memory_info.rss / 1024 / 1024
                
                # Store in history
                self.memory_usage_history.append({
                    "timestamp": time.time(),
                    "memory_mb": memory_mb,
                    "memory_percent": process.memory_percent()
                })
                
                self.cpu_usage_history.append({
                    "timestamp": time.time(),
                    "cpu_percent": cpu_percent
                })
                
                # Check for resource limits
                await self._check_resource_limits(memory_mb, cpu_percent)
                
                await asyncio.sleep(5)  # Monitor every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(5)
    
    async def _check_resource_limits(self, memory_mb: float, cpu_percent: float):
        """Check if resource limits are exceeded"""
        if memory_mb > self.limits.max_memory_mb:
            logger.warning(f"Memory limit exceeded: {memory_mb:.1f}MB > {self.limits.max_memory_mb}MB")
            await self._emergency_memory_cleanup()
        
        if cpu_percent > self.limits.max_cpu_percent:
            logger.warning(f"CPU limit exceeded: {cpu_percent:.1f}% > {self.limits.max_cpu_percent}%")
            await self._reduce_cpu_load()
    
    async def _emergency_memory_cleanup(self):
        """Emergency memory cleanup"""
        try:
            logger.info("🧹 Starting emergency memory cleanup")
            
            # Force garbage collection
            collected = gc.collect()
            self.gc_stats["collections"] += 1
            self.gc_stats["freed_objects"] += collected
            
            # Call registered cleanup callbacks
            for callback in self.cleanup_callbacks:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback()
                    else:
                        callback()
                except Exception as e:
                    logger.warning(f"Cleanup callback failed: {e}")
            
            logger.info(f"✅ Emergency cleanup completed, freed {collected} objects")
            
        except Exception as e:
            logger.error(f"Emergency cleanup failed: {e}")
    
    async def _reduce_cpu_load(self):
        """Reduce CPU load"""
        try:
            logger.info("⏱️ Reducing CPU load")
            
            # Add small delays to reduce CPU pressure
            await asyncio.sleep(0.1)
            
            # Yield control to other tasks
            await asyncio.sleep(0)
            
        except Exception as e:
            logger.error(f"CPU load reduction failed: {e}")
    
    async def _optimization_loop(self):
        """Periodic optimization loop"""
        while self.optimization_active:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Perform periodic optimizations
                await self._periodic_optimization()
                
            except Exception as e:
                logger.error(f"Optimization loop error: {e}")
                await asyncio.sleep(300)
    
    async def _periodic_optimization(self):
        """Perform periodic optimizations"""
        try:
            # Garbage collection
            collected = gc.collect()
            if collected > 0:
                self.gc_stats["collections"] += 1
                self.gc_stats["freed_objects"] += collected
                logger.info(f"Periodic GC freed {collected} objects")
            
            # Memory optimization
            await self._optimize_memory_usage()
            
            # CPU optimization
            await self._optimize_cpu_usage()
            
        except Exception as e:
            logger.error(f"Periodic optimization failed: {e}")
    
    async def _optimize_memory_usage(self):
        """Optimize memory usage"""
        try:
            # Clear weak references
            gc.collect()
            
            # Optimize process memory if available
            process = psutil.Process()
            try:
                if hasattr(process, "memory_maps"):
                    # Force memory map cleanup on Linux
                    pass
            except Exception:
                pass
            
        except Exception as e:
            logger.warning(f"Memory optimization failed: {e}")
    
    async def _optimize_cpu_usage(self):
        """Optimize CPU usage"""
        try:
            # Set process priority to normal
            process = psutil.Process()
            try:
                if process.nice() < 0:
                    process.nice(0)  # Reset to normal priority
            except Exception:
                pass
            
        except Exception as e:
            logger.warning(f"CPU optimization failed: {e}")
    
    def register_cleanup_callback(self, callback: Callable):
        """Register a cleanup callback for memory pressure"""
        self.cleanup_callbacks.append(callback)
    
    async def execute_cpu_bound_task(self, func: Callable, *args, **kwargs):
        """Execute CPU-bound task in process pool"""
        if self.process_pool:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.process_pool, func, *args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    async def execute_io_bound_task(self, func: Callable, *args, **kwargs):
        """Execute I/O-bound task in thread pool"""
        if self.thread_pool:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(self.thread_pool, func, *args, **kwargs)
        else:
            return func(*args, **kwargs)
    
    def get_resource_stats(self) -> Dict[str, Any]:
        """Get current resource statistics"""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                "memory": {
                    "rss_mb": memory_info.rss / 1024 / 1024,
                    "vms_mb": memory_info.vms / 1024 / 1024,
                    "percent": process.memory_percent(),
                    "available_mb": psutil.virtual_memory().available / 1024 / 1024
                },
                "cpu": {
                    "percent": process.cpu_percent(),
                    "num_threads": process.num_threads(),
                    "num_fds": process.num_fds() if hasattr(process, "num_fds") else 0
                },
                "gc_stats": self.gc_stats.copy(),
                "limits": {
                    "max_memory_mb": self.limits.max_memory_mb,
                    "max_cpu_percent": self.limits.max_cpu_percent,
                    "max_threads": self.limits.max_threads,
                    "max_processes": self.limits.max_processes
                },
                "pools": {
                    "thread_pool_active": self.thread_pool is not None,
                    "process_pool_active": self.process_pool is not None
                }
            }
        except Exception as e:
            logger.error(f"Failed to get resource stats: {e}")
            return {}
    
    def get_performance_trends(self) -> Dict[str, Any]:
        """Get performance trends"""
        if not self.memory_usage_history or not self.cpu_usage_history:
            return {}
        
        # Calculate trends
        recent_memory = [entry["memory_mb"] for entry in list(self.memory_usage_history)[-10:]]
        recent_cpu = [entry["cpu_percent"] for entry in list(self.cpu_usage_history)[-10:]]
        
        return {
            "memory_trend": {
                "current": recent_memory[-1] if recent_memory else 0,
                "average": sum(recent_memory) / len(recent_memory) if recent_memory else 0,
                "max": max(recent_memory) if recent_memory else 0,
                "trend": "increasing" if len(recent_memory) >= 2 and recent_memory[-1] > recent_memory[0] else "stable"
            },
            "cpu_trend": {
                "current": recent_cpu[-1] if recent_cpu else 0,
                "average": sum(recent_cpu) / len(recent_cpu) if recent_cpu else 0,
                "max": max(recent_cpu) if recent_cpu else 0,
                "trend": "increasing" if len(recent_cpu) >= 2 and recent_cpu[-1] > recent_cpu[0] else "stable"
            }
        }
    
    async def shutdown(self):
        """Shutdown optimizer and cleanup resources"""
        self.optimization_active = False
        
        if self.thread_pool:
            self.thread_pool.shutdown(wait=True)
        
        if self.process_pool:
            self.process_pool.shutdown(wait=True)
        
        logger.info("🛑 CPU and Memory Optimizer shutdown")

# Performance decorators
def optimize_memory(func):
    """Decorator to optimize memory usage of functions"""
    @wraps(func)
    async def async_wrapper(*args, **kwargs):
        # Clear cache before execution if function has one
        if hasattr(func, "cache_clear"):
            func.cache_clear()
        
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            # Force garbage collection after execution
            gc.collect()
    
    @wraps(func)
    def sync_wrapper(*args, **kwargs):
        if hasattr(func, "cache_clear"):
            func.cache_clear()
        
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            gc.collect()
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper

def cpu_bound(max_workers: int = None):
    """Decorator for CPU-bound functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            optimizer = cpu_memory_optimizer
            return await optimizer.execute_cpu_bound_task(func, *args, **kwargs)
        return wrapper
    return decorator

def io_bound(max_workers: int = None):
    """Decorator for I/O-bound functions"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            optimizer = cpu_memory_optimizer
            return await optimizer.execute_io_bound_task(func, *args, **kwargs)
        return wrapper
    return decorator

# Global optimizer instance
cpu_memory_optimizer = CPUMemoryOptimizer()
