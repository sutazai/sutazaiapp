#!/usr/bin/env python3
"""
SutazAI Cache Manager
Redis-based caching system
"""

import json
import logging
from typing import Any, Optional, Dict, List, Union
from datetime import datetime, timedelta

import aioredis
from .config import settings

logger = logging.getLogger(__name__)


class CacheManager:
    """Redis-based cache manager"""
    
    def __init__(self):
        self.redis_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Redis connection"""
        if self._initialized:
            return
        
        try:
            self.redis_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=100,
                retry_on_timeout=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                health_check_interval=30,
            )
            
            # Test connection
            await self.redis_client.ping()
            
            self._initialized = True
            logger.info("Cache manager initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize cache manager: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown cache manager"""
        if self.redis_client:
            await self.redis_client.close()
            self._initialized = False
            logger.info("Cache manager shutdown")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        if not self._initialized:
            await self.initialize()
        
        try:
            value = await self.redis_client.get(key)
            if value is None:
                return None
            
            # Try to decode JSON, fallback to string
            try:
                return json.loads(value)
            except (json.JSONDecodeError, TypeError):
                return value
                
        except Exception as e:
            logger.error(f"Cache get error for key {key}: {e}")
            return None
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        if not self._initialized:
            await self.initialize()
        
        try:
            # Serialize value to JSON if not string
            if isinstance(value, (dict, list, tuple, int, float, bool)):
                value = json.dumps(value)
            
            ttl = ttl or settings.CACHE_TTL
            result = await self.redis_client.set(key, value, ex=ttl)
            return result is True
            
        except Exception as e:
            logger.error(f"Cache set error for key {key}: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete key from cache"""
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await self.redis_client.delete(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Cache delete error for key {key}: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in cache"""
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await self.redis_client.exists(key)
            return result > 0
            
        except Exception as e:
            logger.error(f"Cache exists error for key {key}: {e}")
            return False
    
    async def increment(self, key: str, amount: int = 1) -> int:
        """Increment value in cache"""
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await self.redis_client.incrby(key, amount)
            return result
            
        except Exception as e:
            logger.error(f"Cache increment error for key {key}: {e}")
            return 0
    
    async def expire(self, key: str, ttl: int) -> bool:
        """Set expiration time for key"""
        if not self._initialized:
            await self.initialize()
        
        try:
            result = await self.redis_client.expire(key, ttl)
            return result is True
            
        except Exception as e:
            logger.error(f"Cache expire error for key {key}: {e}")
            return False
    
    async def keys(self, pattern: str = "*") -> List[str]:
        """Get keys matching pattern"""
        if not self._initialized:
            await self.initialize()
        
        try:
            keys = await self.redis_client.keys(pattern)
            return keys
            
        except Exception as e:
            logger.error(f"Cache keys error for pattern {pattern}: {e}")
            return []
    
    async def clear_pattern(self, pattern: str) -> int:
        """Clear all keys matching pattern"""
        if not self._initialized:
            await self.initialize()
        
        try:
            keys = await self.keys(pattern)
            if keys:
                result = await self.redis_client.delete(*keys)
                return result
            return 0
            
        except Exception as e:
            logger.error(f"Cache clear pattern error for {pattern}: {e}")
            return 0
    
    async def clear_all(self) -> bool:
        """Clear all cache"""
        if not self._initialized:
            await self.initialize()
        
        try:
            await self.redis_client.flushdb()
            return True
            
        except Exception as e:
            logger.error(f"Cache clear all error: {e}")
            return False
    
    async def health_check(self) -> Dict[str, Any]:
        """Health check for cache"""
        if not self._initialized:
            return {"status": "uninitialized", "error": "Cache not initialized"}
        
        try:
            # Test basic operations
            test_key = "health_check_test"
            test_value = {"timestamp": datetime.now().isoformat()}
            
            await self.set(test_key, test_value, ttl=5)
            retrieved = await self.get(test_key)
            await self.delete(test_key)
            
            if retrieved == test_value:
                return {"status": "healthy", "redis_connected": True}
            else:
                return {"status": "unhealthy", "error": "Cache read/write test failed"}
                
        except Exception as e:
            logger.error(f"Cache health check error: {e}")
            return {"status": "unhealthy", "error": str(e)}
    
    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics"""
        if not self._initialized:
            return {"error": "Cache not initialized"}
        
        try:
            info = await self.redis_client.info()
            return {
                "connected_clients": info.get("connected_clients", 0),
                "used_memory": info.get("used_memory", 0),
                "used_memory_human": info.get("used_memory_human", "0B"),
                "keyspace_hits": info.get("keyspace_hits", 0),
                "keyspace_misses": info.get("keyspace_misses", 0),
                "total_commands_processed": info.get("total_commands_processed", 0),
                "uptime_in_seconds": info.get("uptime_in_seconds", 0)
            }
            
        except Exception as e:
            logger.error(f"Cache stats error: {e}")
            return {"error": str(e)}