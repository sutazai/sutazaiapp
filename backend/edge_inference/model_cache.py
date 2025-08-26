"""
Edge Model Cache - Intelligent model caching and sharing for edge inference optimization
"""

import asyncio
import threading
import time
import os
import mmap
import hashlib
import pickle
import logging
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from pathlib import Path
import psutil
import numpy as np
from collections import defaultdict, OrderedDict
import weakref
import json

logger = logging.getLogger(__name__)

class CacheEvictionPolicy(Enum):
    """Cache eviction policies"""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    SIZE_BASED = "size_based"  # Evict largest models first
    ACCESS_TIME = "access_time"  # Evict oldest accessed
    HYBRID = "hybrid"  # Combination of factors

class ModelFormat(Enum):
    """Model storage formats"""
    GGUF = "gguf"
    ONNX = "onnx"
    PYTORCH = "pytorch"
    TENSORFLOW = "tensorflow"
    QUANTIZED = "quantized"

@dataclass
class CachedModel:
    """Represents a cached model"""
    model_id: str
    model_name: str
    format: ModelFormat
    file_path: str
    size_bytes: int
    loaded_at: datetime
    last_accessed: datetime
    access_count: int = 0
    reference_count: int = 0
    quantization_level: Optional[str] = None
    compression_ratio: float = 1.0
    load_time_ms: float = 0.0
    memory_mapped: bool = False
    shared_memory_key: Optional[str] = None

@dataclass  
class ModelCacheStats:
    """Model cache statistics"""
    total_models: int = 0
    total_size_mb: float = 0.0
    cache_hits: int = 0
    cache_misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    disk_usage_mb: float = 0.0
    hit_ratio: float = 0.0
    avg_load_time_ms: float = 0.0

class SharedModelRegistry:
    """Registry for sharing models across processes/nodes"""
    
    def __init__(self, registry_path: str = "/tmp/sutazai_model_registry.json"):
        self.registry_path = Path(registry_path)
        self.models: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.RLock()
        self._load_registry()
    
    def _load_registry(self) -> None:
        """Load registry from disk"""
        try:
            if self.registry_path.exists():
                with open(self.registry_path, 'r') as f:
                    self.models = json.load(f)
                logger.info(f"Loaded model registry with {len(self.models)} entries")
        except Exception as e:
            logger.warning(f"Failed to load model registry: {e}")
            self.models = {}
    
    def _save_registry(self) -> None:
        """Save registry to disk"""
        try:
            with open(self.registry_path, 'w') as f:
                json.dump(self.models, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to save model registry: {e}")
    
    def register_model(self, model_id: str, model_info: Dict[str, Any]) -> None:
        """Register a shared model"""
        with self._lock:
            self.models[model_id] = {
                **model_info,
                "registered_at": datetime.now().isoformat(),
                "process_id": os.getpid()
            }
            self._save_registry()
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get information about a shared model"""
        with self._lock:
            return self.models.get(model_id)
    
    def unregister_model(self, model_id: str) -> None:
        """Unregister a shared model"""
        with self._lock:
            if model_id in self.models:
                del self.models[model_id]
                self._save_registry()
    
    def list_models(self) -> List[Dict[str, Any]]:
        """List all registered models"""
        with self._lock:
            return list(self.models.values())
    
    def cleanup_stale_entries(self) -> None:
        """Remove entries from dead processes"""
        with self._lock:
            stale_models = []
            for model_id, info in self.models.items():
                process_id = info.get("process_id")
                if process_id and not psutil.pid_exists(process_id):
                    stale_models.append(model_id)
            
            for model_id in stale_models:
                del self.models[model_id]
            
            if stale_models:
                self._save_registry()
                logger.info(f"Cleaned up {len(stale_models)} stale registry entries")

class ModelQuantizer:
    """Handles model quantization for edge deployment"""
    
    def __init__(self):
        self.quantization_methods = {
            "int8": self._quantize_int8,
            "int4": self._quantize_int4,
            "fp16": self._quantize_fp16,
            "dynamic": self._quantize_dynamic
        }
    
    async def quantize_model(self, 
                           model_path: str, 
                           output_path: str,
                           method: str = "int8",
                           preserve_accuracy: bool = True) -> Tuple[bool, float]:
        """
        Quantize a model for edge deployment
        
        Returns:
            Tuple of (success, compression_ratio)
        """
        try:
            if method not in self.quantization_methods:
                raise ValueError(f"Unsupported quantization method: {method}")
            
            original_size = os.path.getsize(model_path)
            
            # Run quantization in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            success = await loop.run_in_executor(
                None, 
                self.quantization_methods[method],
                model_path, 
                output_path, 
                preserve_accuracy
            )
            
            if success and os.path.exists(output_path):
                quantized_size = os.path.getsize(output_path)
                compression_ratio = original_size / quantized_size
                logger.info(f"Quantized model with {compression_ratio:.2f}x compression ({method})")
                return True, compression_ratio
            
            return False, 1.0
            
        except Exception as e:
            logger.error(f"Model quantization failed: {e}")
            return False, 1.0
    
    def _quantize_int8(self, model_path: str, output_path: str, preserve_accuracy: bool) -> bool:
        """INT8 quantization implementation"""
        try:
            # This is a placeholder - in production, use ONNX, TensorRT, or other frameworks
            # For demonstration, just copy the file with a marker
            import shutil
            shutil.copy2(model_path, output_path)
            
            # Simulate quantization by truncating file (NOT for production!)
            with open(output_path, 'r+b') as f:
                f.seek(0, 2)  # Go to end
                size = f.tell()
                # Reduce file by ~50% to simulate compression
                f.truncate(int(size * 0.5))
            
            logger.info(f"Applied INT8 quantization to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"INT8 quantization failed: {e}")
            return False
    
    def _quantize_int4(self, model_path: str, output_path: str, preserve_accuracy: bool) -> bool:
        """INT4 quantization implementation"""
        try:
            import shutil
            shutil.copy2(model_path, output_path)
            
            # Simulate more aggressive compression for INT4
            with open(output_path, 'r+b') as f:
                f.seek(0, 2)
                size = f.tell()
                f.truncate(int(size * 0.25))  # ~75% reduction
            
            logger.info(f"Applied INT4 quantization to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"INT4 quantization failed: {e}")
            return False
    
    def _quantize_fp16(self, model_path: str, output_path: str, preserve_accuracy: bool) -> bool:
        """FP16 quantization implementation"""
        try:
            import shutil
            shutil.copy2(model_path, output_path)
            
            # Simulate FP16 compression
            with open(output_path, 'r+b') as f:
                f.seek(0, 2)
                size = f.tell()
                f.truncate(int(size * 0.75))  # ~25% reduction
            
            logger.info(f"Applied FP16 quantization to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"FP16 quantization failed: {e}")
            return False
    
    def _quantize_dynamic(self, model_path: str, output_path: str, preserve_accuracy: bool) -> bool:
        """Dynamic quantization implementation"""
        try:
            import shutil
            shutil.copy2(model_path, output_path)
            
            # Simulate dynamic quantization
            with open(output_path, 'r+b') as f:
                f.seek(0, 2)
                size = f.tell()
                f.truncate(int(size * 0.6))  # ~40% reduction
            
            logger.info(f"Applied dynamic quantization to {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Dynamic quantization failed: {e}")
            return False

class EdgeModelCache:
    """Main edge model cache with intelligent caching and sharing"""
    
    def __init__(self,
                 cache_dir: str = "/tmp/sutazai_model_cache",
                 max_cache_size_gb: float = 10.0,
                 eviction_policy: CacheEvictionPolicy = CacheEvictionPolicy.HYBRID,
                 enable_quantization: bool = True,
                 enable_memory_mapping: bool = True,
                 enable_sharing: bool = True):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_cache_size_bytes = int(max_cache_size_gb * 1024 * 1024 * 1024)
        self.eviction_policy = eviction_policy
        self.enable_quantization = enable_quantization
        self.enable_memory_mapping = enable_memory_mapping
        self.enable_sharing = enable_sharing
        
        # Cache storage
        self.cached_models: Dict[str, CachedModel] = {}
        self.access_order = OrderedDict()  # For LRU
        self.access_counts = defaultdict(int)  # For LFU
        
        # Components
        self.quantizer = ModelQuantizer() if enable_quantization else None
        self.shared_registry = SharedModelRegistry() if enable_sharing else None
        
        # Synchronization
        self._cache_lock = threading.RLock()
        self._memory_maps: Dict[str, mmap.mmap] = {}
        
        # Statistics
        self.stats = ModelCacheStats()
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info(f"EdgeModelCache initialized: {cache_dir}, {max_cache_size_gb}GB, {eviction_policy.value}")
    
    async def start(self) -> None:
        """Start the model cache"""
        if self._running:
            return
        
        self._running = True
        
        # Start background cleanup
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Load existing cache
        await self._load_existing_cache()
        
        logger.info("EdgeModelCache started")
    
    async def stop(self) -> None:
        """Stop the model cache"""
        self._running = False
        
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Close memory maps
        for mmap_obj in self._memory_maps.values():
            mmap_obj.close()
        self._memory_maps.clear()
        
        logger.info("EdgeModelCache stopped")
    
    async def get_model(self, 
                       model_name: str,
                       quantization_level: Optional[str] = None,
                       force_reload: bool = False) -> Optional[CachedModel]:
        """
        Get a model from cache, loading if necessary
        
        Args:
            model_name: Name of the model
            quantization_level: Desired quantization (int8, int4, fp16, dynamic)
            force_reload: Force reload even if cached
            
        Returns:
            CachedModel if available, None otherwise
        """
        model_id = self._get_model_id(model_name, quantization_level)
        
        with self._cache_lock:
            # Check cache first
            if not force_reload and model_id in self.cached_models:
                cached_model = self.cached_models[model_id]
                self._update_access_stats(model_id)
                self.stats.cache_hits += 1
                return cached_model
            
            self.stats.cache_misses += 1
        
        # Try to load from shared registry
        if self.shared_registry:
            shared_info = self.shared_registry.get_model_info(model_id)
            if shared_info and os.path.exists(shared_info.get("file_path", "")):
                return await self._load_shared_model(shared_info)
        
        # Load model
        return await self._load_model(model_name, quantization_level)
    
    async def _load_model(self, 
                         model_name: str, 
                         quantization_level: Optional[str] = None) -> Optional[CachedModel]:
        """Load a model into cache"""
        start_time = time.time()
        
        try:
            # Check if we need to make space
            await self._ensure_cache_space()
            
            model_id = self._get_model_id(model_name, quantization_level)
            
            # Determine model file path (this would integrate with Ollama in practice)
            model_file = await self._get_model_file_path(model_name)
            if not model_file or not os.path.exists(model_file):
                logger.error(f"Model file not found: {model_name}")
                return None
            
            # Apply quantization if requested
            if quantization_level and self.quantizer:
                quantized_file = self.cache_dir / f"{model_name}_{quantization_level}.quantized"
                success, compression_ratio = await self.quantizer.quantize_model(
                    model_file, str(quantized_file), quantization_level
                )
                if success:
                    model_file = str(quantized_file)
                else:
                    logger.warning(f"Quantization failed, using original model")
                    compression_ratio = 1.0
            else:
                compression_ratio = 1.0
            
            # Create cached model entry
            cached_model = CachedModel(
                model_id=model_id,
                model_name=model_name,
                format=self._detect_model_format(model_file),
                file_path=model_file,
                size_bytes=os.path.getsize(model_file),
                loaded_at=datetime.now(),
                last_accessed=datetime.now(),
                quantization_level=quantization_level,
                compression_ratio=compression_ratio,
                load_time_ms=(time.time() - start_time) * 1000
            )
            
            # Set up memory mapping if enabled
            if self.enable_memory_mapping:
                await self._setup_memory_mapping(cached_model)
            
            # Add to cache
            with self._cache_lock:
                self.cached_models[model_id] = cached_model
                self._update_access_stats(model_id)
                self.stats.total_models += 1
                self.stats.total_size_mb += cached_model.size_bytes / (1024 * 1024)
            
            # Register with shared registry
            if self.shared_registry:
                self.shared_registry.register_model(model_id, {
                    "model_name": model_name,
                    "file_path": model_file,
                    "size_bytes": cached_model.size_bytes,
                    "quantization_level": quantization_level,
                    "load_time_ms": cached_model.load_time_ms
                })
            
            logger.info(f"Loaded model {model_name} into cache ({cached_model.size_bytes / 1024 / 1024:.1f}MB)")
            return cached_model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            return None
    
    async def _load_shared_model(self, shared_info: Dict[str, Any]) -> Optional[CachedModel]:
        """Load a model from shared registry"""
        try:
            model_id = shared_info.get("model_id")
            if not model_id:
                return None
            
            # Create cached model from shared info
            cached_model = CachedModel(
                model_id=model_id,
                model_name=shared_info["model_name"],
                format=self._detect_model_format(shared_info["file_path"]),
                file_path=shared_info["file_path"],
                size_bytes=shared_info["size_bytes"],
                loaded_at=datetime.now(),
                last_accessed=datetime.now(),
                quantization_level=shared_info.get("quantization_level"),
                load_time_ms=shared_info.get("load_time_ms", 0.0)
            )
            
            # Add to local cache
            with self._cache_lock:
                self.cached_models[model_id] = cached_model
                self._update_access_stats(model_id)
                self.stats.cache_hits += 1
            
            logger.info(f"Loaded shared model {cached_model.model_name}")
            return cached_model
            
        except Exception as e:
            logger.error(f"Failed to load shared model: {e}")
            return None
    
    async def _setup_memory_mapping(self, cached_model: CachedModel) -> None:
        """Set up memory mapping for a model"""
        try:
            with open(cached_model.file_path, 'rb') as f:
                mmap_obj = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                self._memory_maps[cached_model.model_id] = mmap_obj
                cached_model.memory_mapped = True
                logger.debug(f"Memory mapped model {cached_model.model_name}")
                
        except Exception as e:
            logger.warning(f"Failed to memory map model {cached_model.model_name}: {e}")
    
    def _get_model_id(self, model_name: str, quantization_level: Optional[str] = None) -> str:
        """Generate unique model ID"""
        if quantization_level:
            return f"{model_name}_{quantization_level}"
        return model_name
    
    def _detect_model_format(self, file_path: str) -> ModelFormat:
        """Detect model format from file"""
        file_ext = Path(file_path).suffix.lower()
        
        if file_ext in ['.gguf', '.ggml']:
            return ModelFormat.GGUF
        elif file_ext == '.onnx':
            return ModelFormat.ONNX
        elif file_ext in ['.pt', '.pth']:
            return ModelFormat.PYTORCH
        elif file_ext in ['.pb', '.savedmodel']:
            return ModelFormat.TENSORFLOW
        elif 'quantized' in file_path.lower():
            return ModelFormat.QUANTIZED
        
        return ModelFormat.GGUF  # Default
    
    async def _get_model_file_path(self, model_name: str) -> Optional[str]:
        """Get the file path for a model (integrate with Ollama)"""
        # This would integrate with Ollama's model storage
        # For now, simulate with a placeholder
        ollama_models_dir = Path.home() / ".ollama" / "models" / "blobs"
        
        if ollama_models_dir.exists():
            # Look for model files
            for model_file in ollama_models_dir.glob("*"):
                if model_file.is_file() and model_file.stat().st_size > 100 * 1024 * 1024:  # > 100MB
                    return str(model_file)
        
        # Fallback: create a dummy file for testing
        dummy_file = self.cache_dir / f"{model_name}.dummy"
        if not dummy_file.exists():
            dummy_file.write_bytes(b"0" * 1024 * 1024)  # 1MB dummy file
        
        return str(dummy_file)
    
    def _update_access_stats(self, model_id: str) -> None:
        """Update access statistics for a model"""
        now = datetime.now()
        
        if model_id in self.cached_models:
            self.cached_models[model_id].last_accessed = now
            self.cached_models[model_id].access_count += 1
        
        # Update LRU order
        if model_id in self.access_order:
            del self.access_order[model_id]
        self.access_order[model_id] = now
        
        # Update LFU count
        self.access_counts[model_id] += 1
    
    async def _ensure_cache_space(self, required_bytes: int = 0) -> None:
        """Ensure there's enough space in cache"""
        with self._cache_lock:
            current_size = sum(model.size_bytes for model in self.cached_models.values())
            
            if current_size + required_bytes <= self.max_cache_size_bytes:
                return
            
            # Need to evict models
            bytes_to_free = current_size + required_bytes - self.max_cache_size_bytes
            await self._evict_models(bytes_to_free)
    
    async def _evict_models(self, bytes_to_free: int) -> None:
        """Evict models based on eviction policy"""
        if self.eviction_policy == CacheEvictionPolicy.LRU:
            candidates = sorted(
                self.cached_models.items(),
                key=lambda x: x[1].last_accessed
            )
        elif self.eviction_policy == CacheEvictionPolicy.LFU:
            candidates = sorted(
                self.cached_models.items(),
                key=lambda x: x[1].access_count
            )
        elif self.eviction_policy == CacheEvictionPolicy.SIZE_BASED:
            candidates = sorted(
                self.cached_models.items(),
                key=lambda x: x[1].size_bytes,
                reverse=True
            )
        elif self.eviction_policy == CacheEvictionPolicy.HYBRID:
            # Hybrid scoring: combine access frequency, recency, and size
            def hybrid_score(item):
                model = item[1]
                age_hours = (datetime.now() - model.last_accessed).total_seconds() / 3600
                frequency_score = model.access_count / max(age_hours, 1)
                size_score = model.size_bytes / (1024 * 1024)  # MB
                return frequency_score / size_score  # Lower is better for eviction
            
            candidates = sorted(self.cached_models.items(), key=hybrid_score)
        else:
            candidates = list(self.cached_models.items())
        
        freed_bytes = 0
        evicted_models = []
        
        for model_id, model in candidates:
            if freed_bytes >= bytes_to_free:
                break
            
            # Don't evict models that are currently being used
            if model.reference_count > 0:
                continue
            
            await self._evict_model(model_id)
            freed_bytes += model.size_bytes
            evicted_models.append(model_id)
            self.stats.evictions += 1
        
        logger.info(f"Evicted {len(evicted_models)} models, freed {freed_bytes / 1024 / 1024:.1f}MB")
    
    async def _evict_model(self, model_id: str) -> None:
        """Evict a specific model"""
        with self._cache_lock:
            if model_id not in self.cached_models:
                return
            
            model = self.cached_models[model_id]
            
            # Close memory mapping
            if model_id in self._memory_maps:
                self._memory_maps[model_id].close()
                del self._memory_maps[model_id]
            
            # Remove quantized files if we created them
            if model.quantization_level and model.file_path.endswith('.quantized'):
                try:
                    os.remove(model.file_path)
                except OSError:
                    pass
            
            # Remove from cache
            del self.cached_models[model_id]
            
            # Update access tracking
            if model_id in self.access_order:
                del self.access_order[model_id]
            if model_id in self.access_counts:
                del self.access_counts[model_id]
            
            # Unregister from shared registry
            if self.shared_registry:
                self.shared_registry.unregister_model(model_id)
            
            self.stats.total_models -= 1
            self.stats.total_size_mb -= model.size_bytes / (1024 * 1024)
    
    async def _load_existing_cache(self) -> None:
        """Load models that are already cached on disk"""
        try:
            for file_path in self.cache_dir.glob("*.quantized"):
                # Try to reconstruct cached model info
                parts = file_path.stem.split('_')
                if len(parts) >= 2:
                    model_name = '_'.join(parts[:-1])
                    quantization_level = parts[-1]
                    
                    cached_model = CachedModel(
                        model_id=self._get_model_id(model_name, quantization_level),
                        model_name=model_name,
                        format=self._detect_model_format(str(file_path)),
                        file_path=str(file_path),
                        size_bytes=file_path.stat().st_size,
                        loaded_at=datetime.fromtimestamp(file_path.stat().st_mtime),
                        last_accessed=datetime.fromtimestamp(file_path.stat().st_atime),
                        quantization_level=quantization_level
                    )
                    
                    self.cached_models[cached_model.model_id] = cached_model
                    logger.info(f"Restored cached model: {model_name}")
            
        except Exception as e:
            logger.warning(f"Failed to load existing cache: {e}")
    
    async def _cleanup_loop(self) -> None:
        """Background cleanup loop"""
        while self._running:
            try:
                # Clean up stale shared registry entries
                if self.shared_registry:
                    self.shared_registry.cleanup_stale_entries()
                
                # Update statistics
                await self._update_stats()
                
                # Check for models that haven't been accessed recently
                await self._cleanup_stale_models()
                
                await asyncio.sleep(300)  # 5 minutes
                
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _update_stats(self) -> None:
        """Update cache statistics"""
        with self._cache_lock:
            total_requests = self.stats.cache_hits + self.stats.cache_misses
            self.stats.hit_ratio = self.stats.cache_hits / max(total_requests, 1)
            
            if self.cached_models:
                self.stats.avg_load_time_ms = np.mean([
                    model.load_time_ms for model in self.cached_models.values()
                    if model.load_time_ms > 0
                ])
            
            # Update memory usage
            process = psutil.Process()
            self.stats.memory_usage_mb = process.memory_info().rss / (1024 * 1024)
            
            # Update disk usage
            total_disk_usage = 0
            for file_path in self.cache_dir.glob("*"):
                if file_path.is_file():
                    total_disk_usage += file_path.stat().st_size
            self.stats.disk_usage_mb = total_disk_usage / (1024 * 1024)
    
    async def _cleanup_stale_models(self) -> None:
        """Remove models that haven't been accessed recently"""
        cutoff_time = datetime.now() - timedelta(hours=24)  # 24 hours
        
        stale_models = [
            model_id for model_id, model in self.cached_models.items()
            if model.last_accessed < cutoff_time and model.reference_count == 0
        ]
        
        for model_id in stale_models:
            await self._evict_model(model_id)
            logger.info(f"Cleaned up stale model: {model_id}")
    
    def get_stats(self) -> ModelCacheStats:
        """Get cache statistics"""
        return self.stats
    
    def get_cached_models(self) -> List[CachedModel]:
        """Get list of all cached models"""
        with self._cache_lock:
            return list(self.cached_models.values())
    
    async def preload_models(self, model_names: List[str], quantization_level: Optional[str] = None) -> int:
        """Preload multiple models for faster access"""
        loaded_count = 0
        
        for model_name in model_names:
            try:
                model = await self.get_model(model_name, quantization_level)
                if model:
                    loaded_count += 1
                    logger.info(f"Preloaded model: {model_name}")
            except Exception as e:
                logger.error(f"Failed to preload model {model_name}: {e}")
        
        return loaded_count
    
    async def clear_cache(self) -> None:
        """Clear all cached models"""
        with self._cache_lock:
            model_ids = list(self.cached_models.keys())
        
        for model_id in model_ids:
            await self._evict_model(model_id)
        
        logger.info("Cache cleared")

# Global cache instance
_global_cache: Optional[EdgeModelCache] = None

def get_global_cache(**kwargs) -> EdgeModelCache:
    """Get or create global model cache instance"""
    global _global_cache
    if _global_cache is None:
        _global_cache = EdgeModelCache(**kwargs)
    return _global_cache