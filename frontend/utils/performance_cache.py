"""
Performance Cache for SutazAI Frontend
Advanced caching system with smart refresh and memory management
"""

import time
import logging
from typing import Dict, Any, Optional
from collections import OrderedDict

logger = logging.getLogger(__name__)

class PerformanceCache:
    """High-performance cache with LRU eviction and TTL support"""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 300):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache = OrderedDict()
        self._timestamps = {}
        self._access_times = {}
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_entries": 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache if not expired"""
        if key in self._cache:
            # Check if expired
            if time.time() - self._timestamps[key] > self.default_ttl:
                self.remove(key)
                self._stats["misses"] += 1
                return None
            
            # Update access time for LRU
            self._access_times[key] = time.time()
            self._cache.move_to_end(key)
            self._stats["hits"] += 1
            return self._cache[key]
        
        self._stats["misses"] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        """Set value in cache with optional TTL"""
        ttl = ttl or self.default_ttl
        
        # Remove existing key if present
        if key in self._cache:
            self.remove(key)
        
        # Check if we need to evict
        if len(self._cache) >= self.max_size:
            self._evict_lru()
        
        # Add new entry
        self._cache[key] = value
        self._timestamps[key] = time.time()
        self._access_times[key] = time.time()
        self._stats["total_entries"] += 1
    
    def remove(self, key: str) -> None:
        """Remove key from cache"""
        if key in self._cache:
            del self._cache[key]
            del self._timestamps[key]
            del self._access_times[key]
    
    def clear_expired(self) -> int:
        """Clear expired entries and return count removed"""
        current_time = time.time()
        expired_keys = [
            key for key, timestamp in self._timestamps.items()
            if current_time - timestamp > self.default_ttl
        ]
        
        for key in expired_keys:
            self.remove(key)
        
        return len(expired_keys)
    
    def clear_all(self) -> None:
        """Clear all cache entries"""
        self._cache.clear()
        self._timestamps.clear()
        self._access_times.clear()
        self._stats = {
            "hits": 0,
            "misses": 0,
            "evictions": 0,
            "total_entries": 0
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        total_requests = self._stats["hits"] + self._stats["misses"]
        hit_rate = (self._stats["hits"] / total_requests) if total_requests > 0 else 0
        
        return {
            "total_entries": len(self._cache),
            "max_size": self.max_size,
            "cache_utilization": f"{(len(self._cache) / self.max_size) * 100:.1f}%",
            "hit_rate": hit_rate,
            "hits": self._stats["hits"],
            "misses": self._stats["misses"],
            "evictions": self._stats["evictions"],
            "estimated_size_bytes": len(str(self._cache))
        }
    
    def _evict_lru(self) -> None:
        """Evict least recently used item"""
        if self._cache:
            # Find LRU item
            lru_key = min(self._access_times.keys(), key=lambda k: self._access_times[k])
            self.remove(lru_key)
            self._stats["evictions"] += 1

class SmartRefresh:
    """Smart refresh manager to prevent unnecessary updates"""
    
    _last_refresh_times = {}
    
    @staticmethod
    def should_refresh(operation: str, interval: int) -> bool:
        """Check if operation should refresh based on interval"""
        current_time = time.time()
        last_refresh = SmartRefresh._last_refresh_times.get(operation, 0)
        return current_time - last_refresh >= interval
    
    @staticmethod
    def mark_refreshed(operation: str) -> None:
        """Mark operation as refreshed"""
        SmartRefresh._last_refresh_times[operation] = time.time()

class SmartPreloader:
    """Smart component preloader for better performance"""
    
    _preload_cache = {}
    
    @staticmethod
    def preload_for_page() -> None:
        """Preload components based on current page context"""
        # Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test implementation - would preload relevant components
        current_time = time.time()
        SmartPreloader._preload_cache["last_preload"] = current_time
        
        # In real implementation, would preload:
        # - API data for current page
        # - Related components
        # - Common resources
        logger.debug("Smart preloader executed")
    
    @staticmethod
    def get_preload_stats() -> Dict[str, Any]:
        """Get preloader statistics"""
        return {
            "last_preload": SmartPreloader._preload_cache.get("last_preload", 0),
            "preload_cache_size": len(SmartPreloader._preload_cache)
        }

# Global cache instance
cache = PerformanceCache(max_size=500, default_ttl=300)  # 5 minute TTL

# Export main components
__all__ = [
    'cache',
    'SmartRefresh',
    'SmartPreloader',
    'PerformanceCache'
]