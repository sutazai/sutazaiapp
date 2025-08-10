"""
High-Performance Caching Layer for SutazAI Frontend
Implements intelligent client-side caching with TTL and dependency invalidation
"""

import streamlit as st
import asyncio
import hashlib
import time
from functools import wraps
import logging

logger = logging.getLogger(__name__)

class PerformanceCache:
    """High-performance caching system with TTL and smart invalidation"""
    
    def __init__(self):
        self.default_ttl = 300  # 5 minutes default
        self.max_cache_size = 1000  # Prevent memory bloat
        
    def _get_cache_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate deterministic cache key"""
        key_data = f"{prefix}:{args}:{sorted(kwargs.items())}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def _init_cache_storage(self):
        """Initialize cache storage in session state"""
        if 'performance_cache' not in st.session_state:
            st.session_state.performance_cache = {}
        if 'cache_metadata' not in st.session_state:
            st.session_state.cache_metadata = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get cached value with TTL check"""
        self._init_cache_storage()
        
        if key not in st.session_state.performance_cache:
            return default
            
        metadata = st.session_state.cache_metadata.get(key, {})
        ttl = metadata.get('ttl', 0)
        cached_time = metadata.get('cached_time', 0)
        
        # Check TTL expiration
        if time.time() - cached_time > ttl:
            self._evict(key)
            return default
            
        return st.session_state.performance_cache[key]
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set cached value with TTL"""
        self._init_cache_storage()
        
        if ttl is None:
            ttl = self.default_ttl
            
        # Enforce cache size limit
        if len(st.session_state.performance_cache) >= self.max_cache_size:
            self._evict_oldest()
        
        st.session_state.performance_cache[key] = value
        st.session_state.cache_metadata[key] = {
            'ttl': ttl,
            'cached_time': time.time(),
            'access_count': 1
        }
    
    def _evict(self, key: str) -> None:
        """Remove specific cache entry"""
        if key in st.session_state.performance_cache:
            del st.session_state.performance_cache[key]
        if key in st.session_state.cache_metadata:
            del st.session_state.cache_metadata[key]
    
    def _evict_oldest(self) -> None:
        """Evict oldest cache entries (LRU-style)"""
        if not st.session_state.cache_metadata:
            return
            
        # Find oldest entry
        oldest_key = min(
            st.session_state.cache_metadata.keys(),
            key=lambda k: st.session_state.cache_metadata[k]['cached_time']
        )
        self._evict(oldest_key)
    
    def clear_expired(self) -> int:
        """Clear all expired entries, return count cleared"""
        self._init_cache_storage()
        current_time = time.time()
        expired_keys = []
        
        for key, metadata in st.session_state.cache_metadata.items():
            if current_time - metadata['cached_time'] > metadata['ttl']:
                expired_keys.append(key)
        
        for key in expired_keys:
            self._evict(key)
            
        return len(expired_keys)
    
    def clear_all(self) -> None:
        """Clear entire cache"""
        if 'performance_cache' in st.session_state:
            st.session_state.performance_cache.clear()
        if 'cache_metadata' in st.session_state:
            st.session_state.cache_metadata.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        self._init_cache_storage()
        
        total_entries = len(st.session_state.performance_cache)
        total_size = sum(
            len(str(v)) for v in st.session_state.performance_cache.values()
        )
        
        return {
            'total_entries': total_entries,
            'estimated_size_bytes': total_size,
            'cache_utilization': f"{(total_entries / self.max_cache_size) * 100:.1f}%"
        }

# Global cache instance
cache = PerformanceCache()

def cached_api_call(ttl: int = 300, key_prefix: str = "api"):
    """Decorator for caching API calls with TTL"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache._get_cache_key(f"{key_prefix}:{func.__name__}", *args, **kwargs)
            
            # Try to get from cache
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                logger.debug(f"Cache HIT for {func.__name__}")
                return cached_result
            
            # Execute function and cache result
            logger.debug(f"Cache MISS for {func.__name__}")
            result = await func(*args, **kwargs)
            
            if result is not None:  # Only cache successful results
                cache.set(cache_key, result, ttl)
            
            return result
        
        # Also provide sync wrapper
        def sync_wrapper(*args, **kwargs):
            return asyncio.run(wrapper(*args, **kwargs))
        
        wrapper.sync = sync_wrapper
        return wrapper
    return decorator

@cached_api_call(ttl=60, key_prefix="health")
async def cached_health_check(url: str) -> Dict[str, Any]:
    """Cached health check with 1-minute TTL"""
    from .api_client import call_api
    return await call_api("/health", timeout=2.0)

@cached_api_call(ttl=180, key_prefix="system")
async def cached_system_metrics() -> Dict[str, Any]:
    """Cached system metrics with 3-minute TTL"""
    from .api_client import call_api
    return await call_api("/metrics", timeout=5.0)

def memoize_component(ttl: int = 300):
    """Decorator for memoizing expensive component computations"""
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_key = cache._get_cache_key(f"component:{func.__name__}", *args, **kwargs)
            
            cached_result = cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            result = func(*args, **kwargs)
            cache.set(cache_key, result, ttl)
            return result
        
        return wrapper
    return decorator

class SmartRefresh:
    """Intelligent refresh system that minimizes unnecessary API calls"""
    
    @staticmethod
    def should_refresh(key: str, interval: int = 30) -> bool:
        """Determine if data should be refreshed based on interval"""
        last_refresh_key = f"last_refresh:{key}"
        last_refresh = st.session_state.get(last_refresh_key, 0)
        current_time = time.time()
        
        if current_time - last_refresh >= interval:
            st.session_state[last_refresh_key] = current_time
            return True
        return False
    
    @staticmethod
    def mark_refreshed(key: str):
        """Mark data as refreshed"""
        st.session_state[f"last_refresh:{key}"] = time.time()

# Export main components
__all__ = ['cache', 'cached_api_call', 'cached_health_check', 'cached_system_metrics', 
           'memoize_component', 'SmartRefresh']