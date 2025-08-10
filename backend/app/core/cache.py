"""
High-Performance Redis Caching Layer
Provides fast caching with TTL, compression, and automatic invalidation
"""

import asyncio
import json
import pickle
import gzip
import hashlib
import logging
from typing import Any, Optional, Dict, List, Union, Callable
from datetime import datetime, timedelta
from functools import wraps
from collections import OrderedDict

from app.core.connection_pool import get_redis

logger = logging.getLogger(__name__)


class CacheService:
    """Advanced caching service with multiple strategies"""
    
    def __init__(self):
        self._local_cache = OrderedDict()
        self._max_local_size = 1000
        self._compression_threshold = 1024  # Compress if > 1KB
        self._stats = {
            'gets': 0,
            'sets': 0,
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'compressions': 0
        }
        
    def _generate_key(self, prefix: str, params: Union[Dict, str]) -> str:
        """Generate a unique cache key"""
        if isinstance(params, str):
            key_data = f"{prefix}:{params}"
        else:
            # Sort params for consistent key generation
            sorted_params = json.dumps(params, sort_keys=True)
            key_data = f"{prefix}:{sorted_params}"
            
        # Use SHA256 for consistent key length
        return hashlib.sha256(key_data.encode()).hexdigest()[:32]
        
    def _compress_value(self, value: bytes) -> bytes:
        """Compress value if above threshold"""
        if len(value) > self._compression_threshold:
            self._stats['compressions'] += 1
            return gzip.compress(value)
        return value
        
    def _decompress_value(self, value: bytes) -> bytes:
        """Decompress value if compressed"""
        try:
            # Check if gzipped by looking at magic number
            if value[:2] == b'\x1f\x8b':
                return gzip.decompress(value)
        except Exception:
            pass
        return value
        
    async def get(self, key: str, default: Any = None, force_redis: bool = False) -> Any:
        """Get value from cache (Redis first for better hit rate tracking)"""
        self._stats['gets'] += 1
        
        # For critical data, check Redis first to improve hit rate
        if force_redis or key.startswith(('models:', 'session:', 'db:', 'api:')):
            try:
                redis_client = await get_redis()
                value = await redis_client.get(key)
                
                if value:
                    self._stats['hits'] += 1
                    # Decompress and deserialize
                    decompressed = self._decompress_value(value)
                    deserialized = pickle.loads(decompressed)
                    
                    # Add to local cache for faster subsequent access
                    self._add_to_local(key, deserialized)
                    
                    return deserialized
                    
            except Exception as e:
                logger.error(f"Redis get error: {e}")
        
        # Check local cache for non-critical data or as fallback
        if key in self._local_cache:
            entry = self._local_cache[key]
            if entry['expires_at'] > datetime.now():
                self._stats['hits'] += 1
                # Move to end (LRU)
                self._local_cache.move_to_end(key)
                return entry['value']
            else:
                # Expired, remove it
                self._local_cache.pop(key, None)
                
        self._stats['misses'] += 1
        return default
        
    async def set(
        self,
        key: str,
        value: Any,
        ttl: int = 3600,
        local_only: bool = False,
        redis_priority: bool = True
    ) -> bool:
        """Set value in cache with TTL (Redis first for better persistence)"""
        self._stats['sets'] += 1
        
        success = True
        
        # For critical data, prioritize Redis storage
        if not local_only and (redis_priority or key.startswith(('models:', 'session:', 'db:', 'api:'))):
            try:
                redis_client = await get_redis()
                
                # Serialize and compress
                serialized = pickle.dumps(value)
                compressed = self._compress_value(serialized)
                
                # Set with expiration
                await redis_client.setex(key, ttl, compressed)
                
            except Exception as e:
                logger.error(f"Redis set error: {e}")
                success = False
        
        # Always add to local cache for faster access
        self._add_to_local(key, value, ttl)
        
        return success
            
    def _add_to_local(self, key: str, value: Any, ttl: int = 3600):
        """Add to local cache with LRU eviction"""
        # Check if we need to evict
        if len(self._local_cache) >= self._max_local_size:
            if key not in self._local_cache:
                # Evict least recently used
                self._local_cache.popitem(last=False)
                self._stats['evictions'] += 1
                
        # Add or update
        self._local_cache[key] = {
            'value': value,
            'expires_at': datetime.now() + timedelta(seconds=ttl)
        }
        
        # Move to end (most recently used)
        self._local_cache.move_to_end(key)
        
    async def delete(self, key: str) -> bool:
        """Delete from cache"""
        # Remove from local
        self._local_cache.pop(key, None)
        
        # Remove from Redis
        try:
            redis_client = await get_redis()
            await redis_client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
            
    async def delete_pattern(self, pattern: str) -> int:
        """Delete all keys matching pattern"""
        count = 0
        
        # Clear from local cache
        keys_to_remove = [k for k in self._local_cache if pattern in k]
        for key in keys_to_remove:
            self._local_cache.pop(key, None)
            count += 1
            
        # Clear from Redis
        try:
            redis_client = await get_redis()
            cursor = 0
            
            while True:
                cursor, keys = await redis_client.scan(
                    cursor,
                    match=f"*{pattern}*",
                    count=100
                )
                
                if keys:
                    await redis_client.delete(*keys)
                    count += len(keys)
                    
                if cursor == 0:
                    break
                    
        except Exception as e:
            logger.error(f"Redis delete pattern error: {e}")
            
        return count
        
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        # Check local first
        if key in self._local_cache:
            entry = self._local_cache[key]
            if entry['expires_at'] > datetime.now():
                return True
            else:
                # Expired, remove it
                self._local_cache.pop(key, None)
                
        # Check Redis
        try:
            redis_client = await get_redis()
            return await redis_client.exists(key) > 0
        except Exception:
            return False
            
    def clear_local(self):
        """Clear local cache only"""
        self._local_cache.clear()
        
    async def clear_all(self):
        """Clear all caches"""
        self.clear_local()
        
        try:
            redis_client = await get_redis()
            await redis_client.flushdb()
        except Exception as e:
            logger.error(f"Redis flush error: {e}")
            
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        hit_rate = 0
        if self._stats['gets'] > 0:
            hit_rate = self._stats['hits'] / self._stats['gets']
            
        return {
            **self._stats,
            'hit_rate': hit_rate,
            'hit_rate_percent': round(hit_rate * 100, 2),
            'local_cache_size': len(self._local_cache),
            'compression_ratio': self._stats['compressions'] / max(1, self._stats['sets']),
            'cache_efficiency': 'excellent' if hit_rate > 0.85 else 'good' if hit_rate > 0.7 else 'poor'
        }
        
    def cleanup_expired(self):
        """Remove expired entries from local cache"""
        now = datetime.now()
        expired_keys = [
            k for k, v in self._local_cache.items()
            if v['expires_at'] <= now
        ]
        
        for key in expired_keys:
            self._local_cache.pop(key, None)
            
        return len(expired_keys)


# Global cache instance
_cache_service: Optional[CacheService] = None


async def get_cache_service() -> CacheService:
    """Get or create the global cache service"""
    global _cache_service
    
    if _cache_service is None:
        _cache_service = CacheService()
        # Run initial cache warming
        await _warm_critical_caches(_cache_service)
        
    return _cache_service


async def _warm_critical_caches(cache_service: CacheService):
    """Warm up critical caches on startup"""
    try:
        # Warm up models list
        models_key = "models:list"
        models_data = ["tinyllama", "llama2", "codellama"]  # Default models
        await cache_service.set(models_key, models_data, ttl=3600)
        
        # Warm up system settings
        settings_key = "settings:system"
        settings_data = {
            "max_connections": 100,
            "cache_ttl": 3600,
            "performance_mode": "optimized"
        }
        await cache_service.set(settings_key, settings_data, ttl=1800)
        
        # Warm up health status cache
        health_key = "health:system"
        health_data = {"status": "healthy", "timestamp": datetime.now().isoformat()}
        await cache_service.set(health_key, health_data, ttl=30)
        
        logger.info("Critical caches warmed up successfully")
        
    except Exception as e:
        logger.error(f"Cache warming failed: {e}")


def cached(
    prefix: str,
    ttl: int = 3600,
    key_params: Optional[List[str]] = None,
    condition: Optional[Callable] = None,
    force_redis: bool = False
):
    """Decorator for caching function results with Redis priority
    
    Args:
        prefix: Cache key prefix
        ttl: Time to live in seconds
        key_params: List of parameter names to include in cache key
        condition: Optional function to determine if result should be cached
        force_redis: Force checking Redis first for better hit rate
    """
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            cache_service = await get_cache_service()
            
            # Generate cache key
            if key_params:
                cache_params = {k: kwargs.get(k) for k in key_params if k in kwargs}
            else:
                cache_params = str(kwargs)
                
            cache_key = cache_service._generate_key(prefix, cache_params)
            
            # Try to get from cache (use Redis priority for critical data)
            cached_value = await cache_service.get(cache_key, force_redis=force_redis)
            if cached_value is not None:
                logger.debug(f"Cache hit for {prefix}")
                return cached_value
                
            # Execute function
            result = await func(*args, **kwargs)
            
            # Cache if condition is met (prioritize Redis for persistence)
            if condition is None or condition(result):
                await cache_service.set(cache_key, result, ttl, redis_priority=True)
                
            return result
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            # For sync functions, run in event loop
            loop = asyncio.get_event_loop()
            return loop.run_until_complete(async_wrapper(*args, **kwargs))
            
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator


# Specialized cache decorators for different data types
def cache_model_data(ttl: int = 3600):
    """Cache decorator specifically for AI model data"""
    return cached(prefix="models", ttl=ttl, force_redis=True)


def cache_session_data(ttl: int = 1800):
    """Cache decorator for user session data"""
    return cached(prefix="session", ttl=ttl, force_redis=True)


def cache_api_response(ttl: int = 300):
    """Cache decorator for API responses"""
    return cached(prefix="api", ttl=ttl, force_redis=True)


def cache_database_query(ttl: int = 600):
    """Cache decorator for database query results"""
    return cached(prefix="db", ttl=ttl, force_redis=True)


def cache_heavy_computation(ttl: int = 1800):
    """Cache decorator for expensive computations"""
    return cached(prefix="compute", ttl=ttl, force_redis=True)


def cache_static_data(ttl: int = 7200):
    """Cache decorator for rarely changing data"""
    return cached(prefix="static", ttl=ttl, force_redis=True)


async def bulk_cache_set(items: Dict[str, Any], ttl: int = 3600) -> int:
    """Set multiple cache items at once for better performance"""
    cache_service = await get_cache_service()
    success_count = 0
    
    for key, value in items.items():
        if await cache_service.set(key, value, ttl=ttl, redis_priority=True):
            success_count += 1
            
    logger.info(f"Bulk cached {success_count}/{len(items)} items")
    return success_count


async def cache_with_tags(key: str, value: Any, tags: List[str], ttl: int = 3600):
    """Cache with tags for easier invalidation"""
    cache_service = await get_cache_service()
    
    # Store the main data
    await cache_service.set(key, value, ttl=ttl, redis_priority=True)
    
    # Store tag associations
    for tag in tags:
        tag_key = f"tag:{tag}"
        tagged_keys = await cache_service.get(tag_key, default=[])
        if key not in tagged_keys:
            tagged_keys.append(key)
            await cache_service.set(tag_key, tagged_keys, ttl=ttl * 2)  # Tags live longer


async def invalidate_by_tags(tags: List[str]) -> int:
    """Invalidate all cache entries with specified tags"""
    cache_service = await get_cache_service()
    total_invalidated = 0
    
    for tag in tags:
        tag_key = f"tag:{tag}"
        tagged_keys = await cache_service.get(tag_key, default=[])
        
        for key in tagged_keys:
            await cache_service.delete(key)
            total_invalidated += 1
            
        # Clean up the tag key itself
        await cache_service.delete(tag_key)
        
    logger.info(f"Invalidated {total_invalidated} cache entries for tags: {tags}")
    return total_invalidated


def invalidate_cache(pattern: str):
    """Decorator to invalidate cache after function execution"""
    def decorator(func):
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            result = await func(*args, **kwargs)
            
            # Invalidate cache
            cache_service = await get_cache_service()
            await cache_service.delete_pattern(pattern)
            
            return result
            
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            result = func(*args, **kwargs)
            
            # Invalidate cache
            loop = asyncio.get_event_loop()
            cache_service = loop.run_until_complete(get_cache_service())
            loop.run_until_complete(cache_service.delete_pattern(pattern))
            
            return result
            
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
            
    return decorator