"""
Edge Memory Manager - Dynamic model loading and intelligent memory management for edge inference
"""

import asyncio
import threading
import time
import os
import gc
import mmap
import logging
import weakref
from typing import Dict, List, Optional, Any, Tuple, Set, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import psutil
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

class LoadStrategy(Enum):
    """Model loading strategies"""
    LAZY = "lazy"                  # Load on first use
    EAGER = "eager"                # Load immediately
    PREDICTIVE = "predictive"      # Load based on usage patterns
    ADAPTIVE = "adaptive"          # Adjust strategy based on performance
    MEMORY_AWARE = "memory_aware"  # Load based on memory availability

class MemoryPool(Enum):
    """Memory pool types"""
    MODEL_WEIGHTS = "model_weights"
    KV_CACHE = "kv_cache"
    ACTIVATIONS = "activations"
    BUFFERS = "buffers"
    SHARED = "shared"

@dataclass
class ModelLoadInfo:
    """Information about a loaded model"""
    model_id: str
    model_path: str
    memory_usage_mb: float
    load_time_ms: float
    last_accessed: datetime
    access_count: int = 0
    reference_count: int = 0
    memory_mapped: bool = False
    compressed: bool = False
    shared_handle: Optional[Any] = None
    pool_assignments: Dict[MemoryPool, int] = field(default_factory=dict)

@dataclass
class MemoryStats:
    """Memory usage statistics"""
    total_memory_mb: float
    used_memory_mb: float
    available_memory_mb: float
    model_memory_mb: float
    cache_memory_mb: float
    buffer_memory_mb: float
    shared_memory_mb: float
    fragmentation_ratio: float
    gc_collections: int = 0

class MemoryMonitor:
    """Monitors system memory usage and patterns"""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.memory_history = deque(maxlen=300)  # 5 minutes at 1s intervals
        self.memory_pressure_threshold = 0.85  # 85% memory usage
        self.low_memory_threshold = 0.95       # 95% memory usage
        self._monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self) -> None:
        """Start memory monitoring"""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Memory monitoring started")
    
    async def stop_monitoring(self) -> None:
        """Stop memory monitoring"""
        self._monitoring = False
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Memory monitoring stopped")
    
    async def _monitoring_loop(self) -> None:
        """Memory monitoring loop"""
        while self._monitoring:
            try:
                stats = self._collect_memory_stats()
                self.memory_history.append(stats)
                
                # Check for memory pressure
                if stats.used_memory_mb / stats.total_memory_mb > self.memory_pressure_threshold:
                    await self._handle_memory_pressure(stats)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                logger.error(f"Memory monitoring error: {e}")
                await asyncio.sleep(5)
    
    def _collect_memory_stats(self) -> MemoryStats:
        """Collect current memory statistics"""
        memory = psutil.virtual_memory()
        process = psutil.Process()
        process_memory = process.memory_info()
        
        return MemoryStats(
            total_memory_mb=memory.total / (1024 * 1024),
            used_memory_mb=memory.used / (1024 * 1024),
            available_memory_mb=memory.available / (1024 * 1024),
            model_memory_mb=0.0,  # Will be updated by ModelMemoryManager
            cache_memory_mb=0.0,
            buffer_memory_mb=0.0,
            shared_memory_mb=0.0,
            fragmentation_ratio=self._estimate_fragmentation()
        )
    
    def _estimate_fragmentation(self) -> float:
        """Estimate memory fragmentation ratio"""
        # Simple estimation based on available vs free memory
        memory = psutil.virtual_memory()
        if memory.free > 0:
            return min(1.0, (memory.available - memory.free) / memory.available)
        return 0.0
    
    async def _handle_memory_pressure(self, stats: MemoryStats) -> None:
        """Handle memory pressure situation"""
        logger.warning(f"Memory pressure detected: {stats.used_memory_mb:.1f}MB / {stats.total_memory_mb:.1f}MB")
        
        # Trigger garbage collection
        gc.collect()
        
        # Notify memory manager (would be implemented with callbacks)
        # This is where we'd trigger model eviction, cache clearing, etc.
    
    def get_memory_pressure_level(self) -> float:
        """Get current memory pressure level (0.0 to 1.0)"""
        if not self.memory_history:
            return 0.0
        
        latest_stats = self.memory_history[-1]
        return latest_stats.used_memory_mb / latest_stats.total_memory_mb
    
    def predict_memory_trend(self, seconds_ahead: int = 60) -> float:
        """Predict memory usage trend"""
        if len(self.memory_history) < 10:
            return self.get_memory_pressure_level()
        
        # Simple linear trend prediction
        recent_usage = [stats.used_memory_mb for stats in list(self.memory_history)[-10:]]
        if len(recent_usage) >= 2:
            trend = (recent_usage[-1] - recent_usage[0]) / len(recent_usage)
            predicted = recent_usage[-1] + (trend * seconds_ahead)
            total_memory = self.memory_history[-1].total_memory_mb
            return min(1.0, max(0.0, predicted / total_memory))
        
        return self.get_memory_pressure_level()

class ModelMemoryPool:
    """Memory pool for efficient model memory management"""
    
    def __init__(self, 
                 pool_type: MemoryPool,
                 max_size_mb: float,
                 alignment: int = 64):
        self.pool_type = pool_type
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.alignment = alignment
        
        self.allocated_blocks: Dict[int, Tuple[int, int]] = {}  # block_id -> (offset, size)
        self.free_blocks: List[Tuple[int, int]] = [(0, self.max_size_bytes)]  # (offset, size)
        self.used_bytes = 0
        self._next_block_id = 1
        self._lock = threading.RLock()
        
        logger.debug(f"Created memory pool {pool_type.value}: {max_size_mb}MB")
    
    def allocate(self, size_bytes: int) -> Optional[int]:
        """Allocate memory block"""
        # Align size
        aligned_size = ((size_bytes + self.alignment - 1) // self.alignment) * self.alignment
        
        with self._lock:
            # Find suitable free block
            for i, (offset, block_size) in enumerate(self.free_blocks):
                if block_size >= aligned_size:
                    # Allocate from this block
                    block_id = self._next_block_id
                    self._next_block_id += 1
                    
                    self.allocated_blocks[block_id] = (offset, aligned_size)
                    self.used_bytes += aligned_size
                    
                    # Update free blocks
                    remaining_size = block_size - aligned_size
                    if remaining_size > 0:
                        self.free_blocks[i] = (offset + aligned_size, remaining_size)
                    else:
                        del self.free_blocks[i]
                    
                    logger.debug(f"Allocated {aligned_size} bytes in pool {self.pool_type.value}")
                    return block_id
            
            return None  # No suitable block found
    
    def deallocate(self, block_id: int) -> bool:
        """Deallocate memory block"""
        with self._lock:
            if block_id not in self.allocated_blocks:
                return False
            
            offset, size = self.allocated_blocks[block_id]
            del self.allocated_blocks[block_id]
            self.used_bytes -= size
            
            # Add to free blocks and merge adjacent blocks
            self._add_free_block(offset, size)
            
            logger.debug(f"Deallocated {size} bytes in pool {self.pool_type.value}")
            return True
    
    def _add_free_block(self, offset: int, size: int) -> None:
        """Add free block and merge with adjacent blocks"""
        # Insert in sorted order
        inserted = False
        for i, (block_offset, block_size) in enumerate(self.free_blocks):
            if offset < block_offset:
                self.free_blocks.insert(i, (offset, size))
                inserted = True
                break
        
        if not inserted:
            self.free_blocks.append((offset, size))
        
        # Merge adjacent blocks
        self._merge_free_blocks()
    
    def _merge_free_blocks(self) -> None:
        """Merge adjacent free blocks"""
        if len(self.free_blocks) <= 1:
            return
        
        merged = []
        current_offset, current_size = self.free_blocks[0]
        
        for offset, size in self.free_blocks[1:]:
            if current_offset + current_size == offset:
                # Adjacent blocks, merge
                current_size += size
            else:
                # Non-adjacent, add current block
                merged.append((current_offset, current_size))
                current_offset, current_size = offset, size
        
        merged.append((current_offset, current_size))
        self.free_blocks = merged
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        with self._lock:
            fragmentation = 0.0
            if len(self.free_blocks) > 1:
                total_free = sum(size for _, size in self.free_blocks)
                largest_free = max(size for _, size in self.free_blocks) if self.free_blocks else 0
                if total_free > 0:
                    fragmentation = 1.0 - (largest_free / total_free)
            
            return {
                "pool_type": self.pool_type.value,
                "max_size_mb": self.max_size_bytes / (1024 * 1024),
                "used_mb": self.used_bytes / (1024 * 1024),
                "free_mb": (self.max_size_bytes - self.used_bytes) / (1024 * 1024),
                "utilization": self.used_bytes / self.max_size_bytes,
                "fragmentation": fragmentation,
                "allocated_blocks": len(self.allocated_blocks),
                "free_blocks": len(self.free_blocks)
            }

class DynamicModelLoader:
    """Dynamic model loader with intelligent preloading"""
    
    def __init__(self, 
                 memory_manager: 'ModelMemoryManager',
                 load_strategy: LoadStrategy = LoadStrategy.ADAPTIVE):
        self.memory_manager = memory_manager
        self.load_strategy = load_strategy
        
        self.loading_tasks: Dict[str, asyncio.Task] = {}
        self.load_queue = asyncio.PriorityQueue()
        self.usage_patterns: Dict[str, List[datetime]] = defaultdict(list)
        self.prediction_weights = {"hourly": 0.4, "daily": 0.3, "trend": 0.3}
        
        self._loader_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the dynamic loader"""
        if self._running:
            return
        
        self._running = True
        self._loader_task = asyncio.create_task(self._loader_loop())
        logger.info("DynamicModelLoader started")
    
    async def stop(self) -> None:
        """Stop the dynamic loader"""
        self._running = False
        
        if self._loader_task and not self._loader_task.done():
            self._loader_task.cancel()
            try:
                await self._loader_task
            except asyncio.CancelledError:
                pass
        
        # Cancel loading tasks
        for task in self.loading_tasks.values():
            task.cancel()
        
        logger.info("DynamicModelLoader stopped")
    
    async def request_model_load(self, 
                                model_id: str, 
                                model_path: str,
                                priority: int = 5,
                                preload: bool = False) -> Optional[ModelLoadInfo]:
        """Request model loading"""
        # Check if already loaded
        if self.memory_manager.is_model_loaded(model_id):
            model_info = self.memory_manager.get_model_info(model_id)
            if model_info:
                self._record_usage(model_id)
                return model_info
        
        # Check if already loading
        if model_id in self.loading_tasks:
            try:
                return await self.loading_tasks[model_id]
            except Exception as e:
                logger.error(f"Model loading failed: {e}")
                return None
        
        # Add to queue or load immediately
        if preload or self.load_strategy == LoadStrategy.EAGER:
            return await self._load_model_immediately(model_id, model_path)
        else:
            await self.load_queue.put((priority, time.time(), model_id, model_path))
            return None
    
    async def _load_model_immediately(self, model_id: str, model_path: str) -> Optional[ModelLoadInfo]:
        """Load model immediately"""
        if model_id in self.loading_tasks:
            return await self.loading_tasks[model_id]
        
        self.loading_tasks[model_id] = asyncio.create_task(
            self.memory_manager.load_model(model_id, model_path)
        )
        
        try:
            result = await self.loading_tasks[model_id]
            self._record_usage(model_id)
            return result
        finally:
            if model_id in self.loading_tasks:
                del self.loading_tasks[model_id]
    
    async def _loader_loop(self) -> None:
        """Main loader loop"""
        while self._running:
            try:
                # Process predictive loading
                if self.load_strategy in [LoadStrategy.PREDICTIVE, LoadStrategy.ADAPTIVE]:
                    await self._process_predictive_loading()
                
                # Process queued loads
                try:
                    priority, queued_time, model_id, model_path = await asyncio.wait_for(
                        self.load_queue.get(), timeout=1.0
                    )
                    
                    # Check if model is still needed (not too old)
                    if time.time() - queued_time < 30:  # 30 second timeout
                        await self._load_model_immediately(model_id, model_path)
                    
                except asyncio.TimeoutError:
                    pass  # No queued items, continue
                
            except Exception as e:
                logger.error(f"Loader loop error: {e}")
                await asyncio.sleep(1)
    
    async def _process_predictive_loading(self) -> None:
        """Process predictive model loading"""
        try:
            predictions = self._predict_model_usage()
            
            for model_id, probability in predictions.items():
                if probability > 0.7 and not self.memory_manager.is_model_loaded(model_id):
                    # High probability of usage, consider preloading
                    if self.memory_manager.can_load_model(model_id):
                        logger.info(f"Predictively loading model {model_id} (probability: {probability:.2f})")
                        # Add to high priority queue
                        await self.load_queue.put((1, time.time(), model_id, ""))
            
        except Exception as e:
            logger.error(f"Predictive loading error: {e}")
    
    def _predict_model_usage(self) -> Dict[str, float]:
        """Predict model usage probability"""
        predictions = {}
        current_time = datetime.now()
        
        for model_id, usage_times in self.usage_patterns.items():
            if not usage_times:
                continue
            
            # Remove old usage records (older than 7 days)
            cutoff = current_time - timedelta(days=7)
            recent_usage = [t for t in usage_times if t > cutoff]
            self.usage_patterns[model_id] = recent_usage
            
            if len(recent_usage) < 2:
                predictions[model_id] = 0.1  # Low probability
                continue
            
            # Hourly pattern
            current_hour = current_time.hour
            hourly_usage = len([t for t in recent_usage if t.hour == current_hour])
            hourly_prob = min(1.0, hourly_usage / max(len(recent_usage), 1))
            
            # Daily pattern
            current_weekday = current_time.weekday()
            daily_usage = len([t for t in recent_usage if t.weekday() == current_weekday])
            daily_prob = min(1.0, daily_usage / max(len(recent_usage), 1))
            
            # Recent trend
            if len(recent_usage) >= 5:
                recent_5 = recent_usage[-5:]
                avg_interval = sum((recent_5[i] - recent_5[i-1]).total_seconds() 
                                 for i in range(1, len(recent_5))) / (len(recent_5) - 1)
                time_since_last = (current_time - recent_usage[-1]).total_seconds()
                trend_prob = max(0.0, 1.0 - (time_since_last / avg_interval))
            else:
                trend_prob = 0.0
            
            # Weighted combination
            combined_prob = (
                hourly_prob * self.prediction_weights["hourly"] +
                daily_prob * self.prediction_weights["daily"] +
                trend_prob * self.prediction_weights["trend"]
            )
            
            predictions[model_id] = combined_prob
        
        return predictions
    
    def _record_usage(self, model_id: str) -> None:
        """Record model usage for pattern learning"""
        self.usage_patterns[model_id].append(datetime.now())
        
        # Keep only recent usage (last 1000 entries per model)
        if len(self.usage_patterns[model_id]) > 1000:
            self.usage_patterns[model_id] = self.usage_patterns[model_id][-1000:]

class ModelMemoryManager:
    """Main memory manager for dynamic model loading and memory optimization"""
    
    def __init__(self,
                 max_model_memory_gb: float = 8.0,
                 enable_memory_pools: bool = True,
                 enable_memory_mapping: bool = True,
                 enable_compression: bool = True):
        
        self.max_model_memory_bytes = int(max_model_memory_gb * 1024 * 1024 * 1024)
        self.enable_memory_pools = enable_memory_pools
        self.enable_memory_mapping = enable_memory_mapping
        self.enable_compression = enable_compression
        
        # Loaded models
        self.loaded_models: Dict[str, ModelLoadInfo] = {}
        self.model_locks: Dict[str, asyncio.Lock] = {}
        
        # Memory pools
        self.memory_pools: Dict[MemoryPool, ModelMemoryPool] = {}
        if enable_memory_pools:
            self._initialize_memory_pools(max_model_memory_gb)
        
        # Components
        self.memory_monitor = MemoryMonitor()
        self.dynamic_loader = DynamicModelLoader(self)
        
        # Memory mapping
        self.memory_maps: Dict[str, mmap.mmap] = {}
        
        # Statistics
        self.stats = MemoryStats(
            total_memory_mb=0.0,
            used_memory_mb=0.0,
            available_memory_mb=0.0,
            model_memory_mb=0.0,
            cache_memory_mb=0.0,
            buffer_memory_mb=0.0,
            shared_memory_mb=0.0,
            fragmentation_ratio=0.0
        )
        
        self._lock = asyncio.Lock()
        self._running = False
        
        logger.info(f"ModelMemoryManager initialized: {max_model_memory_gb}GB")
    
    def _initialize_memory_pools(self, total_memory_gb: float) -> None:
        """Initialize memory pools"""
        # Distribute memory across pools
        pool_allocations = {
            MemoryPool.MODEL_WEIGHTS: total_memory_gb * 0.60,  # 60% for model weights
            MemoryPool.KV_CACHE: total_memory_gb * 0.25,       # 25% for KV cache
            MemoryPool.ACTIVATIONS: total_memory_gb * 0.10,    # 10% for activations
            MemoryPool.BUFFERS: total_memory_gb * 0.05         # 5% for buffers
        }
        
        for pool_type, size_gb in pool_allocations.items():
            self.memory_pools[pool_type] = ModelMemoryPool(pool_type, size_gb)
    
    async def start(self) -> None:
        """Start the memory manager"""
        if self._running:
            return
        
        self._running = True
        
        # Start components
        await self.memory_monitor.start_monitoring()
        await self.dynamic_loader.start()
        
        logger.info("ModelMemoryManager started")
    
    async def stop(self) -> None:
        """Stop the memory manager"""
        self._running = False
        
        # Stop components
        await self.dynamic_loader.stop()
        await self.memory_monitor.stop_monitoring()
        
        # Clean up memory maps
        for mmap_obj in self.memory_maps.values():
            mmap_obj.close()
        self.memory_maps.clear()
        
        # Clean up loaded models
        for model_id in list(self.loaded_models.keys()):
            await self.unload_model(model_id)
        
        logger.info("ModelMemoryManager stopped")
    
    async def load_model(self, model_id: str, model_path: str) -> Optional[ModelLoadInfo]:
        """Load a model into memory"""
        async with self._lock:
            if model_id in self.loaded_models:
                model_info = self.loaded_models[model_id]
                model_info.access_count += 1
                model_info.last_accessed = datetime.now()
                return model_info
            
            # Create lock for this model
            if model_id not in self.model_locks:
                self.model_locks[model_id] = asyncio.Lock()
        
        async with self.model_locks[model_id]:
            try:
                start_time = time.time()
                
                # Check memory availability
                if not await self._ensure_memory_available(model_path):
                    logger.warning(f"Insufficient memory to load model {model_id}")
                    return None
                
                # Allocate memory
                memory_blocks = await self._allocate_model_memory(model_id, model_path)
                if not memory_blocks:
                    logger.error(f"Failed to allocate memory for model {model_id}")
                    return None
                
                # Load model (this would integrate with actual model loading)
                model_size = os.path.getsize(model_path)
                
                # Set up memory mapping if enabled
                memory_mapped = False
                if self.enable_memory_mapping:
                    try:
                        with open(model_path, 'rb') as f:
                            mmap_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                            self.memory_maps[model_id] = mmap_obj
                            memory_mapped = True
                    except Exception as e:
                        logger.warning(f"Memory mapping failed for {model_id}: {e}")
                
                load_time = (time.time() - start_time) * 1000
                
                # Create model info
                model_info = ModelLoadInfo(
                    model_id=model_id,
                    model_path=model_path,
                    memory_usage_mb=model_size / (1024 * 1024),
                    load_time_ms=load_time,
                    last_accessed=datetime.now(),
                    access_count=1,
                    memory_mapped=memory_mapped,
                    pool_assignments=memory_blocks
                )
                
                self.loaded_models[model_id] = model_info
                
                # Update statistics
                await self._update_memory_stats()
                
                logger.info(f"Loaded model {model_id}: {model_info.memory_usage_mb:.1f}MB in {load_time:.1f}ms")
                return model_info
                
            except Exception as e:
                logger.error(f"Failed to load model {model_id}: {e}")
                return None
    
    async def unload_model(self, model_id: str) -> bool:
        """Unload a model from memory"""
        async with self._lock:
            if model_id not in self.loaded_models:
                return False
            
            model_info = self.loaded_models[model_id]
            
            # Check if model is still being used
            if model_info.reference_count > 0:
                logger.warning(f"Cannot unload model {model_id}: still in use")
                return False
            
            # Deallocate memory blocks
            for pool_type, block_id in model_info.pool_assignments.items():
                if pool_type in self.memory_pools:
                    self.memory_pools[pool_type].deallocate(block_id)
            
            # Close memory mapping
            if model_id in self.memory_maps:
                self.memory_maps[model_id].close()
                del self.memory_maps[model_id]
            
            # Remove from loaded models
            del self.loaded_models[model_id]
            
            # Clean up lock
            if model_id in self.model_locks:
                del self.model_locks[model_id]
            
            # Update statistics
            await self._update_memory_stats()
            
            logger.info(f"Unloaded model {model_id}")
            return True
    
    async def _ensure_memory_available(self, model_path: str) -> bool:
        """Ensure sufficient memory is available"""
        model_size = os.path.getsize(model_path)
        current_usage = sum(info.memory_usage_mb for info in self.loaded_models.values()) * 1024 * 1024
        
        if current_usage + model_size > self.max_model_memory_bytes:
            # Try to free memory by unloading least recently used models
            await self._free_memory_for_model(model_size)
        
        # Check again
        current_usage = sum(info.memory_usage_mb for info in self.loaded_models.values()) * 1024 * 1024
        return current_usage + model_size <= self.max_model_memory_bytes
    
    async def _free_memory_for_model(self, required_bytes: int) -> None:
        """Free memory by unloading models"""
        # Sort models by LRU
        models_by_lru = sorted(
            [(model_id, info) for model_id, info in self.loaded_models.items()],
            key=lambda x: x[1].last_accessed
        )
        
        freed_bytes = 0
        for model_id, model_info in models_by_lru:
            if model_info.reference_count == 0:  # Only unload unused models
                if await self.unload_model(model_id):
                    freed_bytes += model_info.memory_usage_mb * 1024 * 1024
                    if freed_bytes >= required_bytes:
                        break
    
    async def _allocate_model_memory(self, model_id: str, model_path: str) -> Dict[MemoryPool, int]:
        """Allocate memory blocks for model"""
        if not self.enable_memory_pools:
            return {}
        
        model_size = os.path.getsize(model_path)
        allocations = {}
        
        # Allocate from model weights pool
        weights_pool = self.memory_pools.get(MemoryPool.MODEL_WEIGHTS)
        if weights_pool:
            block_id = weights_pool.allocate(model_size)
            if block_id:
                allocations[MemoryPool.MODEL_WEIGHTS] = block_id
        
        return allocations
    
    async def _update_memory_stats(self) -> None:
        """Update memory statistics"""
        # System memory
        memory = psutil.virtual_memory()
        self.stats.total_memory_mb = memory.total / (1024 * 1024)
        self.stats.used_memory_mb = memory.used / (1024 * 1024)
        self.stats.available_memory_mb = memory.available / (1024 * 1024)
        
        # Model memory
        self.stats.model_memory_mb = sum(
            info.memory_usage_mb for info in self.loaded_models.values()
        )
        
        # Pool statistics
        if self.memory_pools:
            for pool in self.memory_pools.values():
                pool_stats = pool.get_stats()
                if pool.pool_type == MemoryPool.KV_CACHE:
                    self.stats.cache_memory_mb += pool_stats["used_mb"]
                elif pool.pool_type == MemoryPool.BUFFERS:
                    self.stats.buffer_memory_mb += pool_stats["used_mb"]
    
    def is_model_loaded(self, model_id: str) -> bool:
        """Check if model is loaded"""
        return model_id in self.loaded_models
    
    def get_model_info(self, model_id: str) -> Optional[ModelLoadInfo]:
        """Get information about a loaded model"""
        return self.loaded_models.get(model_id)
    
    def can_load_model(self, model_id: str) -> bool:
        """Check if model can be loaded given current memory constraints"""
        if model_id in self.loaded_models:
            return True
        
        # Check memory pressure
        memory_pressure = self.memory_monitor.get_memory_pressure_level()
        return memory_pressure < 0.8  # Don't load if memory pressure > 80%
    
    def get_memory_stats(self) -> MemoryStats:
        """Get current memory statistics"""
        return self.stats
    
    def get_loaded_models(self) -> List[ModelLoadInfo]:
        """Get list of loaded models"""
        return list(self.loaded_models.values())
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get memory pool statistics"""
        if not self.memory_pools:
            return {}
        
        return {
            pool_type.value: pool.get_stats()
            for pool_type, pool in self.memory_pools.items()
        }

# Global memory manager instance
_global_manager: Optional[ModelMemoryManager] = None

def get_global_manager(**kwargs) -> ModelMemoryManager:
    """Get or create global memory manager instance"""
    global _global_manager
    if _global_manager is None:
        _global_manager = ModelMemoryManager(**kwargs)
    return _global_manager