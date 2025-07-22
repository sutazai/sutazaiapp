"""
Enterprise-Grade Performance Optimization Module
"""

import asyncio
import time
import functools
from typing import Dict, Any, Optional, Callable, List, Union
from collections import defaultdict, OrderedDict
import redis.asyncio as redis
from datetime import datetime, timedelta
import json
import logging
import psutil
import numpy as np
from contextlib import asynccontextmanager
import pickle
import hashlib
import gzip

from app.core.config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Advanced caching system with TTL, LRU, and compression"""
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.REDIS_URL
        self._redis_client = None
        self._local_cache = OrderedDict()
        self._cache_stats = defaultdict(int)
        self.max_local_size = 1000
        self.compression_threshold = 1024  # Compress if > 1KB
        
    async def get_redis(self) -> redis.Redis:
        """Get Redis client with connection pooling"""
        if not self._redis_client:
            self._redis_client = await redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=False,
                max_connections=50,
                socket_keepalive=True,
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 2,  # TCP_KEEPINTVL
                    3: 3,  # TCP_KEEPCNT
                }
            )
        return self._redis_client
    
    def _generate_key(self, prefix: str, params: Dict[str, Any]) -> str:
        """Generate cache key from parameters"""
        # Sort params for consistent keys
        sorted_params = json.dumps(params, sort_keys=True)
        hash_digest = hashlib.md5(sorted_params.encode()).hexdigest()
        return f"{prefix}:{hash_digest}"
    
    def _compress_value(self, value: bytes) -> bytes:
        """Compress value if above threshold"""
        if len(value) > self.compression_threshold:
            return b"COMPRESSED:" + gzip.compress(value)
        return value
    
    def _decompress_value(self, value: bytes) -> bytes:
        """Decompress value if compressed"""
        if value.startswith(b"COMPRESSED:"):
            return gzip.decompress(value[11:])
        return value
    
    async def get(self, key: str, default: Any = None) -> Any:
        """Get value from cache (local first, then Redis)"""
        # Check local cache
        if key in self._local_cache:
            self._cache_stats["local_hits"] += 1
            # Move to end (LRU)
            self._local_cache.move_to_end(key)
            return self._local_cache[key]
        
        # Check Redis
        try:
            client = await self.get_redis()
            value = await client.get(key)
            if value:
                self._cache_stats["redis_hits"] += 1
                decompressed = self._decompress_value(value)
                result = pickle.loads(decompressed)
                
                # Add to local cache
                self._add_to_local_cache(key, result)
                return result
        except Exception as e:
            logger.error(f"Redis get error: {e}")
        
        self._cache_stats["misses"] += 1
        return default
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Set value in cache with TTL"""
        try:
            # Add to local cache
            self._add_to_local_cache(key, value)
            
            # Add to Redis
            client = await self.get_redis()
            serialized = pickle.dumps(value)
            compressed = self._compress_value(serialized)
            
            await client.setex(key, ttl, compressed)
            self._cache_stats["sets"] += 1
            return True
        except Exception as e:
            logger.error(f"Redis set error: {e}")
            return False
    
    def _add_to_local_cache(self, key: str, value: Any):
        """Add to local cache with LRU eviction"""
        if key in self._local_cache:
            self._local_cache.move_to_end(key)
        else:
            self._local_cache[key] = value
            if len(self._local_cache) > self.max_local_size:
                # Remove least recently used
                self._local_cache.popitem(last=False)
    
    async def delete(self, key: str) -> bool:
        """Delete from cache"""
        # Remove from local
        self._local_cache.pop(key, None)
        
        # Remove from Redis
        try:
            client = await self.get_redis()
            await client.delete(key)
            return True
        except Exception as e:
            logger.error(f"Redis delete error: {e}")
            return False
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        count = 0
        try:
            client = await self.get_redis()
            cursor = 0
            while True:
                cursor, keys = await client.scan(cursor, match=pattern, count=100)
                if keys:
                    await client.delete(*keys)
                    count += len(keys)
                    # Also remove from local cache
                    for key in keys:
                        self._local_cache.pop(key.decode() if isinstance(key, bytes) else key, None)
                if cursor == 0:
                    break
        except Exception as e:
            logger.error(f"Redis clear pattern error: {e}")
        return count
    
    def get_stats(self) -> Dict[str, int]:
        """Get cache statistics"""
        total_requests = sum(self._cache_stats.values())
        hit_rate = 0
        if total_requests > 0:
            hits = self._cache_stats["local_hits"] + self._cache_stats["redis_hits"]
            hit_rate = hits / total_requests
        
        return {
            **self._cache_stats,
            "local_cache_size": len(self._local_cache),
            "hit_rate": hit_rate
        }


class PerformanceOptimizer:
    """Central performance optimization system"""
    
    def __init__(self):
        self.cache_manager = CacheManager()
        self._request_timings = defaultdict(list)
        self._db_query_cache = {}
        self._connection_pools = {}
        self.max_timing_history = 1000
        
    def cache(self, prefix: str, ttl: int = 3600, key_params: List[str] = None):
        """Decorator for caching function results"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                # Generate cache key
                if key_params:
                    cache_params = {k: kwargs.get(k) for k in key_params if k in kwargs}
                else:
                    cache_params = kwargs
                
                cache_key = self.cache_manager._generate_key(prefix, cache_params)
                
                # Try to get from cache
                cached_result = await self.cache_manager.get(cache_key)
                if cached_result is not None:
                    return cached_result
                
                # Execute function
                result = await func(*args, **kwargs)
                
                # Cache result
                await self.cache_manager.set(cache_key, result, ttl)
                
                return result
            return wrapper
        return decorator
    
    def track_timing(self, operation: str):
        """Decorator to track operation timing"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self._request_timings[operation].append(duration)
                    
                    # Keep only recent timings
                    if len(self._request_timings[operation]) > self.max_timing_history:
                        self._request_timings[operation] = self._request_timings[operation][-self.max_timing_history:]
                    
                    # Log slow operations
                    if duration > 1.0:
                        logger.warning(f"Slow operation {operation}: {duration:.2f}s")
            return wrapper
        return decorator
    
    async def batch_process(self, items: List[Any], processor: Callable, batch_size: int = 50, max_concurrent: int = 5):
        """Process items in batches with concurrency control"""
        results = []
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def process_batch(batch):
            async with semaphore:
                return await processor(batch)
        
        # Create batches
        batches = [items[i:i + batch_size] for i in range(0, len(items), batch_size)]
        
        # Process batches concurrently
        batch_results = await asyncio.gather(*[process_batch(batch) for batch in batches])
        
        # Flatten results
        for batch_result in batch_results:
            results.extend(batch_result)
        
        return results
    
    @asynccontextmanager
    async def connection_pool(self, service: str, factory: Callable, max_size: int = 10):
        """Manage connection pools for external services"""
        if service not in self._connection_pools:
            self._connection_pools[service] = asyncio.Queue(maxsize=max_size)
            
            # Pre-populate pool
            for _ in range(max_size):
                conn = await factory()
                await self._connection_pools[service].put(conn)
        
        pool = self._connection_pools[service]
        conn = await pool.get()
        try:
            yield conn
        finally:
            await pool.put(conn)
    
    def optimize_query(self, query: str) -> str:
        """Optimize database queries"""
        # Add query optimization logic
        optimized = query
        
        # Add LIMIT if not present for SELECT
        if query.upper().startswith("SELECT") and "LIMIT" not in query.upper():
            optimized += " LIMIT 1000"
        
        # Add index hints
        # This would be more sophisticated in production
        
        return optimized
    
    async def profile_memory(self) -> Dict[str, Any]:
        """Profile memory usage"""
        process = psutil.Process()
        memory_info = process.memory_info()
        
        return {
            "rss": memory_info.rss / 1024 / 1024,  # MB
            "vms": memory_info.vms / 1024 / 1024,  # MB
            "percent": process.memory_percent(),
            "available": psutil.virtual_memory().available / 1024 / 1024  # MB
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get comprehensive performance metrics"""
        metrics = {
            "cache_stats": self.cache_manager.get_stats(),
            "operation_timings": {},
            "memory": asyncio.run(self.profile_memory())
        }
        
        # Calculate timing statistics
        for operation, timings in self._request_timings.items():
            if timings:
                metrics["operation_timings"][operation] = {
                    "count": len(timings),
                    "avg": np.mean(timings),
                    "min": np.min(timings),
                    "max": np.max(timings),
                    "p50": np.percentile(timings, 50),
                    "p95": np.percentile(timings, 95),
                    "p99": np.percentile(timings, 99)
                }
        
        return metrics


class QueryOptimizer:
    """Database query optimization"""
    
    def __init__(self):
        self.query_plans = {}
        self.slow_query_threshold = 1.0  # seconds
        
    async def analyze_query(self, query: str, execution_time: float) -> Dict[str, Any]:
        """Analyze query performance"""
        analysis = {
            "query": query,
            "execution_time": execution_time,
            "is_slow": execution_time > self.slow_query_threshold,
            "suggestions": []
        }
        
        # Check for common issues
        upper_query = query.upper()
        
        if "SELECT *" in upper_query:
            analysis["suggestions"].append("Avoid SELECT *, specify needed columns")
        
        if "JOIN" in upper_query and "INDEX" not in upper_query:
            analysis["suggestions"].append("Ensure JOIN columns are indexed")
        
        if upper_query.count("JOIN") > 3:
            analysis["suggestions"].append("Consider breaking complex JOINs into smaller queries")
        
        if "LIKE '%'" in upper_query:
            analysis["suggestions"].append("Leading wildcard in LIKE prevents index usage")
        
        if "OR" in upper_query:
            analysis["suggestions"].append("OR conditions may prevent index usage, consider UNION")
        
        return analysis
    
    def create_index_suggestion(self, table: str, columns: List[str]) -> str:
        """Generate index creation SQL"""
        index_name = f"idx_{table}_{'_'.join(columns)}"
        columns_str = ", ".join(columns)
        return f"CREATE INDEX {index_name} ON {table} ({columns_str});"


class LoadBalancer:
    """Simple load balancer for distributing requests"""
    
    def __init__(self):
        self.backends = []
        self.current_index = 0
        self.health_checks = {}
        
    def add_backend(self, url: str, weight: int = 1):
        """Add backend server"""
        for _ in range(weight):
            self.backends.append(url)
    
    def get_next_backend(self) -> Optional[str]:
        """Get next backend using round-robin"""
        if not self.backends:
            return None
        
        backend = self.backends[self.current_index]
        self.current_index = (self.current_index + 1) % len(self.backends)
        
        # Skip unhealthy backends
        if not self.health_checks.get(backend, True):
            return self.get_next_backend()
        
        return backend
    
    async def health_check(self, backend: str) -> bool:
        """Check backend health"""
        try:
            # Implement actual health check
            # For now, always return True
            self.health_checks[backend] = True
            return True
        except Exception:
            self.health_checks[backend] = False
            return False


class RateLimiter:
    """Token bucket rate limiter"""
    
    def __init__(self, rate: int = 100, burst: int = 200):
        self.rate = rate  # tokens per second
        self.burst = burst  # max tokens
        self.tokens = burst
        self.last_update = time.time()
        self.lock = asyncio.Lock()
    
    async def acquire(self, tokens: int = 1) -> bool:
        """Acquire tokens, return True if allowed"""
        async with self.lock:
            now = time.time()
            elapsed = now - self.last_update
            self.last_update = now
            
            # Add new tokens
            self.tokens = min(self.burst, self.tokens + elapsed * self.rate)
            
            if self.tokens >= tokens:
                self.tokens -= tokens
                return True
            return False
    
    def get_wait_time(self, tokens: int = 1) -> float:
        """Get time to wait for tokens"""
        if self.tokens >= tokens:
            return 0
        needed = tokens - self.tokens
        return needed / self.rate


# Global instances
performance_optimizer = PerformanceOptimizer()
query_optimizer = QueryOptimizer()
load_balancer = LoadBalancer()
rate_limiter = RateLimiter()


# Convenience decorators
cache = performance_optimizer.cache
track_timing = performance_optimizer.track_timing