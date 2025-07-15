"""
Advanced Caching System for SutazAI
Multi-tier caching with memory, Redis, and file-based caching
"""

import asyncio
import json
import time
import hashlib
import logging
import pickle
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import threading

logger = logging.getLogger(__name__)

class CacheType(str, Enum):
    MEMORY = "memory"
    REDIS = "redis"
    FILE = "file"
    HYBRID = "hybrid"

@dataclass
class CacheConfig:
    """Cache configuration"""
    default_ttl: int = 3600  # 1 hour
    max_memory_size: int = 100 * 1024 * 1024  # 100MB
    redis_url: str = "redis://localhost:6379/0"
    file_cache_dir: str = "/opt/sutazaiapp/cache"
    compression_enabled: bool = True

class CacheEntry:
    """Cache entry with metadata"""
    
    def __init__(self, value: Any, ttl: int = 3600):
        self.value = value
        self.created_at = time.time()
        self.ttl = ttl
        self.access_count = 0
        self.last_accessed = time.time()
    
    def is_expired(self) -> bool:
        """Check if cache entry is expired"""
        return time.time() - self.created_at > self.ttl
    
    def touch(self):
        """Update access statistics"""
        self.access_count += 1
        self.last_accessed = time.time()

class AdvancedCacheManager:
    """Advanced multi-tier cache manager"""
    
    def __init__(self, config: CacheConfig = None):
        self.config = config or CacheConfig()
        self.memory_cache = {}
        self.memory_size = 0
        self.cache_lock = threading.RLock()
        self.redis_client = None
        self.file_cache_dir = Path(self.config.file_cache_dir)
        self.file_cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Performance statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "memory_usage": 0,
            "operations": 0
        }
    
    async def initialize(self):
        """Initialize cache manager"""
        logger.info("🔄 Initializing Advanced Cache Manager")
        
        # Try to initialize Redis
        await self._initialize_redis()
        
        # Load persistent cache entries
        await self._load_persistent_cache()
        
        # Start cleanup task
        asyncio.create_task(self._cleanup_task())
        
        logger.info("✅ Cache manager initialized")
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            import redis.asyncio as redis
            self.redis_client = redis.from_url(self.config.redis_url)
            await self.redis_client.ping()
            logger.info("✅ Redis cache connected")
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}")
            self.redis_client = None
    
    async def _load_persistent_cache(self):
        """Load persistent cache entries from disk"""
        try:
            persistent_cache_file = self.file_cache_dir / "persistent_cache.json"
            if persistent_cache_file.exists():
                with open(persistent_cache_file, 'r') as f:
                    data = json.load(f)
                
                for key, entry_data in data.items():
                    if time.time() - entry_data['created_at'] < entry_data['ttl']:
                        entry = CacheEntry(
                            value=entry_data['value'],
                            ttl=entry_data['ttl']
                        )
                        entry.created_at = entry_data['created_at']
                        self.memory_cache[key] = entry
                
                logger.info(f"Loaded {len(data)} persistent cache entries")
        except Exception as e:
            logger.warning(f"Failed to load persistent cache: {e}")
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache"""
        self.stats["operations"] += 1
        
        # Check memory cache first
        with self.cache_lock:
            if key in self.memory_cache:
                entry = self.memory_cache[key]
                if not entry.is_expired():
                    entry.touch()
                    self.stats["hits"] += 1
                    return entry.value
                else:
                    # Remove expired entry
                    del self.memory_cache[key]
                    self._update_memory_size()
        
        # Check Redis cache
        if self.redis_client:
            try:
                value = await self.redis_client.get(key)
                if value is not None:
                    # Deserialize and store in memory cache
                    deserialized_value = pickle.loads(value)
                    await self.set(key, deserialized_value, ttl=self.config.default_ttl)
                    self.stats["hits"] += 1
                    return deserialized_value
            except Exception as e:
                logger.warning(f"Redis get failed: {e}")
        
        # Check file cache
        file_cache_path = self.file_cache_dir / f"{self._hash_key(key)}.cache"
        if file_cache_path.exists():
            try:
                with open(file_cache_path, 'rb') as f:
                    cache_data = pickle.load(f)
                
                if time.time() - cache_data['created_at'] < cache_data['ttl']:
                    # Load into memory cache
                    await self.set(key, cache_data['value'], ttl=cache_data['ttl'])
                    self.stats["hits"] += 1
                    return cache_data['value']
                else:
                    # Remove expired file
                    file_cache_path.unlink()
            except Exception as e:
                logger.warning(f"File cache read failed: {e}")
        
        self.stats["misses"] += 1
        return default
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in cache"""
        if ttl is None:
            ttl = self.config.default_ttl
        
        self.stats["operations"] += 1
        
        # Store in memory cache
        with self.cache_lock:
            entry = CacheEntry(value, ttl)
            self.memory_cache[key] = entry
            self._update_memory_size()
            self._evict_if_needed()
        
        # Store in Redis cache
        if self.redis_client:
            try:
                serialized_value = pickle.dumps(value)
                await self.redis_client.setex(key, ttl, serialized_value)
            except Exception as e:
                logger.warning(f"Redis set failed: {e}")
        
        # Store in file cache for large values
        if self._get_value_size(value) > 1024:  # > 1KB
            try:
                file_cache_path = self.file_cache_dir / f"{self._hash_key(key)}.cache"
                cache_data = {
                    'value': value,
                    'created_at': time.time(),
                    'ttl': ttl
                }
                
                with open(file_cache_path, 'wb') as f:
                    pickle.dump(cache_data, f)
            except Exception as e:
                logger.warning(f"File cache write failed: {e}")
        
        return True
    
    async def delete(self, key: str) -> bool:
        """Delete key from all cache layers"""
        deleted = False
        
        # Delete from memory cache
        with self.cache_lock:
            if key in self.memory_cache:
                del self.memory_cache[key]
                self._update_memory_size()
                deleted = True
        
        # Delete from Redis cache
        if self.redis_client:
            try:
                await self.redis_client.delete(key)
                deleted = True
            except Exception as e:
                logger.warning(f"Redis delete failed: {e}")
        
        # Delete from file cache
        file_cache_path = self.file_cache_dir / f"{self._hash_key(key)}.cache"
        if file_cache_path.exists():
            try:
                file_cache_path.unlink()
                deleted = True
            except Exception as e:
                logger.warning(f"File cache delete failed: {e}")
        
        return deleted
    
    async def clear(self):
        """Clear all cache layers"""
        # Clear memory cache
        with self.cache_lock:
            self.memory_cache.clear()
            self.memory_size = 0
        
        # Clear Redis cache
        if self.redis_client:
            try:
                await self.redis_client.flushdb()
            except Exception as e:
                logger.warning(f"Redis clear failed: {e}")
        
        # Clear file cache
        try:
            for cache_file in self.file_cache_dir.glob("*.cache"):
                cache_file.unlink()
        except Exception as e:
            logger.warning(f"File cache clear failed: {e}")
    
    def _hash_key(self, key: str) -> str:
        """Create hash for key"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _get_value_size(self, value: Any) -> int:
        """Estimate size of value in bytes"""
        try:
            return len(pickle.dumps(value))
        except:
            return len(str(value).encode())
    
    def _update_memory_size(self):
        """Update memory usage statistics"""
        total_size = 0
        for entry in self.memory_cache.values():
            total_size += self._get_value_size(entry.value)
        
        self.memory_size = total_size
        self.stats["memory_usage"] = total_size
    
    def _evict_if_needed(self):
        """Evict entries if memory limit exceeded"""
        if self.memory_size <= self.config.max_memory_size:
            return
        
        # Sort by last access time and evict oldest
        sorted_entries = sorted(
            self.memory_cache.items(),
            key=lambda x: x[1].last_accessed
        )
        
        while self.memory_size > self.config.max_memory_size * 0.8:
            if not sorted_entries:
                break
            
            key, entry = sorted_entries.pop(0)
            del self.memory_cache[key]
            self.stats["evictions"] += 1
        
        self._update_memory_size()
    
    async def _cleanup_task(self):
        """Periodic cleanup task"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Clean expired memory cache entries
                with self.cache_lock:
                    expired_keys = [
                        key for key, entry in self.memory_cache.items()
                        if entry.is_expired()
                    ]
                    
                    for key in expired_keys:
                        del self.memory_cache[key]
                    
                    if expired_keys:
                        self._update_memory_size()
                        logger.info(f"Cleaned {len(expired_keys)} expired cache entries")
                
                # Clean expired file cache
                for cache_file in self.file_cache_dir.glob("*.cache"):
                    try:
                        if cache_file.stat().st_mtime < time.time() - self.config.default_ttl:
                            cache_file.unlink()
                    except Exception:
                        pass
                
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = (self.stats["hits"] / max(1, total_requests)) * 100
        
        return {
            **self.stats,
            "hit_rate_percent": hit_rate,
            "total_requests": total_requests,
            "memory_cache_size": len(self.memory_cache),
            "memory_usage_mb": self.memory_size / (1024 * 1024),
            "redis_available": self.redis_client is not None
        }

# Global cache manager instance
cache_manager = AdvancedCacheManager()

# Decorators for easy caching
def cached(ttl: int = 3600, key_prefix: str = ""):
    """Decorator for caching function results"""
    def decorator(func):
        async def async_wrapper(*args, **kwargs):
            # Generate cache key
            key_parts = [key_prefix, func.__name__]
            key_parts.extend(str(arg) for arg in args)
            key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
            cache_key = ":".join(key_parts)
            
            # Try to get from cache
            result = await cache_manager.get(cache_key)
            if result is not None:
                return result
            
            # Execute function and cache result
            result = await func(*args, **kwargs)
            await cache_manager.set(cache_key, result, ttl)
            return result
        
        def sync_wrapper(*args, **kwargs):
            # For sync functions, use asyncio.run
            async def async_func():
                return await async_wrapper(*args, **kwargs)
            
            return asyncio.run(async_func())
        
        return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
    
    return decorator
