# backend/app/services/cache_manager.py
from typing import Any, Optional, Dict, List
import json
import asyncio
from datetime import datetime, timedelta
import hashlib
from functools import wraps
import sys

import redis.asyncio as redis
from app.core.config import settings

class CacheManager:
    """Multi-layer intelligent caching system"""
    
    def __init__(self):
        # L1 Cache: In-memory (process-local)
        self.l1_cache: Dict[str, Any] = {}
        self.l1_max_size = 1000
        self.l1_ttl = timedelta(minutes=5)
        
        # L2 Cache: Redis (distributed)
        self.redis_client = None
        self.l2_ttl = timedelta(hours=1)
        
        # L3 Cache: Database (persistent)
        self.l3_ttl = timedelta(days=7)
        
        # Cache statistics
        self.stats = {
            "l1_hits": 0,
            "l1_misses": 0,
            "l2_hits": 0,
            "l2_misses": 0,
            "l3_hits": 0,
            "l3_misses": 0
        }
        
    async def initialize(self):
        """Initialize cache connections"""
        self.redis_client = await redis.from_url(
            settings.REDIS_URL,
            encoding="utf-8",
            decode_responses=True
        )
        
    def generate_cache_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate consistent cache key from parameters"""
        # Sort parameters for consistency
        sorted_params = sorted(params.items())
        param_str = json.dumps(sorted_params, sort_keys=True)
        
        # Create hash
        param_hash = hashlib.md5(param_str.encode()).hexdigest()
        
        return f"{prefix}:{param_hash}"
    
    async def get(
        self,
        key: str,
        fetch_func: Optional[callable] = None,
        ttl: Optional[timedelta] = None
    ) -> Any:
        """Get value from cache with multi-layer lookup"""
        
        # Check L1 cache
        if key in self.l1_cache:
            if (datetime.utcnow() - self.l1_cache[key]["timestamp"]) < self.l1_ttl:
                self.stats["l1_hits"] += 1
                return self.l1_cache[key]["value"]
            else:
                del self.l1_cache[key]
        
        self.stats["l1_misses"] += 1
        
        # Check L2 cache (Redis)
        if self.redis_client:
            value = await self.redis_client.get(key)
            if value:
                self.stats["l2_hits"] += 1
                deserialized_value = json.loads(value)
                self._set_l1(key, deserialized_value)
                return deserialized_value
        
        self.stats["l2_misses"] += 1
        
        # Check L3 cache (Database)
        db_value = await self._get_from_database(key)
        if db_value:
            self.stats["l3_hits"] += 1
            await self._set_l2(key, db_value, self.l2_ttl)
            self._set_l1(key, db_value)
            return db_value
        
        self.stats["l3_misses"] += 1
        
        # Cache miss - fetch from source
        if fetch_func:
            value = await fetch_func()
            await self.set(key, value, ttl)
            return value
        
        return None
    
    async def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[timedelta] = None
    ):
        """Set value in all cache layers"""
        self._set_l1(key, value)
        await self._set_l2(key, value, ttl or self.l2_ttl)
        await self._set_l3(key, value, ttl or self.l3_ttl)
    
    def _set_l1(self, key: str, value: Any):
        """Set value in L1 cache with LRU eviction"""
        if len(self.l1_cache) >= self.l1_max_size:
            try:
                oldest_key = min(
                    self.l1_cache.keys(),
                    key=lambda k: self.l1_cache[k]["timestamp"]
                )
                del self.l1_cache[oldest_key]
            except ValueError: # Cache is empty
                pass
        
        self.l1_cache[key] = {
            "value": value,
            "timestamp": datetime.utcnow()
        }
    
    async def _set_l2(self, key: str, value: Any, ttl: timedelta):
        """Set value in Redis cache"""
        if self.redis_client:
            await self.redis_client.set(
                key,
                json.dumps(value),
                ex=ttl
            )
    
    async def _set_l3(self, key: str, value: Any, ttl: timedelta):
        """Set value in database cache (placeholder)"""
        pass
    
    async def _get_from_database(self, key: str) -> Optional[Any]:
        """Get value from database cache (placeholder)"""
        return None

    async def invalidate(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        keys_to_remove = [k for k in self.l1_cache if pattern in k]
        for key in keys_to_remove:
            if key in self.l1_cache:
                del self.l1_cache[key]
        
        if self.redis_client:
            async for key in self.redis_client.scan_iter(f"*{pattern}*"):
                await self.redis_client.delete(key)
        
        await self._invalidate_database_cache(pattern)

    async def _invalidate_database_cache(self, pattern: str):
        """Invalidate database cache entries (placeholder)"""
        pass
    
    def cache_result(
        self,
        prefix: str,
        ttl: Optional[timedelta] = None,
        key_params: Optional[List[str]] = None
    ):
        """Decorator for caching function results"""
        def decorator(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                if key_params:
                    cache_params = {p: kwargs.get(p) for p in key_params if kwargs.get(p) is not None}
                else:
                    cache_params = kwargs
                
                all_params = {**cache_params, 'args': args}
                cache_key = self.generate_cache_key(prefix, all_params)
                
                cached = await self.get(cache_key)
                if cached is not None:
                    return cached
                
                result = await func(*args, **kwargs)
                await self.set(cache_key, result, ttl)
                return result
            
            return wrapper
        return decorator
    
    async def _warm_single_pattern(self, pattern: Dict[str, Any]):
        """Warms a single cache pattern (placeholder)"""
        pass

    async def warm_cache(self, patterns: List[Dict[str, Any]]):
        """Pre-warm cache with common queries"""
        tasks = [self._warm_single_pattern(p) for p in patterns]
        await asyncio.gather(*tasks)
    
    def _calculate_memory_usage(self) -> int:
        """Calculate L1 cache memory usage in bytes (approximation)"""
        return sum(sys.getsizeof(k) + sys.getsizeof(v) for k, v in self.l1_cache.items())

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        l1_reqs = self.stats["l1_hits"] + self.stats["l1_misses"]
        total_hits = self.stats["l1_hits"] + self.stats["l2_hits"] + self.stats["l3_hits"]
        hit_rate = total_hits / l1_reqs if l1_reqs > 0 else 0
        
        return {
            "statistics": self.stats,
            "hit_rate": hit_rate,
            "l1_size": len(self.l1_cache),
            "l1_memory_usage_bytes": self._calculate_memory_usage()
        }
