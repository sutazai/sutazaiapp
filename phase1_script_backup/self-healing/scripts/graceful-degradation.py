#!/usr/bin/env python3
"""
Graceful Degradation System for SutazAI
Manages feature flags and fallback mechanisms
"""

import os
import json
import logging
import redis
import yaml
from typing import Dict, Any, Optional, Callable
from datetime import datetime, timedelta
from functools import wraps, lru_cache
import asyncio

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureFlag:
    """
    Represents a feature that can be toggled and has fallback behavior
    """
    
    def __init__(self, 
                 name: str,
                 enabled: bool = True,
                 fallback: str = "disabled",
                 cache_ttl: int = 300,
                 config: Dict[str, Any] = None):
        self.name = name
        self.enabled = enabled
        self.fallback = fallback
        self.cache_ttl = cache_ttl
        self.config = config or {}
        self.last_check = datetime.now()
        self.cached_value = None
        
    def is_enabled(self) -> bool:
        """Check if feature is enabled"""
        return self.enabled
    
    def get_fallback_config(self) -> Dict[str, Any]:
        """Get fallback configuration"""
        return {
            "fallback": self.fallback,
            "cache_ttl": self.cache_ttl,
            **self.config
        }


class CacheStrategy:
    """
    Implements caching strategies for graceful degradation
    """
    
    def __init__(self, 
                 strategy_type: str = "lru",
                 size: int = 1000,
                 ttl: int = 300,
                 redis_client: Optional[redis.Redis] = None):
        self.strategy_type = strategy_type
        self.size = size
        self.ttl = ttl
        self.redis_client = redis_client
        self._local_cache = {}
        self._cache_order = []
        
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        # Try Redis first
        if self.redis_client:
            try:
                value = self.redis_client.get(f"cache:{key}")
                if value:
                    return json.loads(value)
            except Exception as e:
                logger.warning(f"Redis cache get failed: {e}")
        
        # Fall back to local cache
        if key in self._local_cache:
            value, expiry = self._local_cache[key]
            if datetime.now() < expiry:
                # Update order for LRU
                if self.strategy_type == "lru" and key in self._cache_order:
                    self._cache_order.remove(key)
                    self._cache_order.append(key)
                return value
            else:
                # Expired
                del self._local_cache[key]
                if key in self._cache_order:
                    self._cache_order.remove(key)
        
        return None
    
    def set(self, key: str, value: Any):
        """Set value in cache"""
        expiry = datetime.now() + timedelta(seconds=self.ttl)
        
        # Save to Redis
        if self.redis_client:
            try:
                self.redis_client.setex(
                    f"cache:{key}",
                    self.ttl,
                    json.dumps(value)
                )
            except Exception as e:
                logger.warning(f"Redis cache set failed: {e}")
        
        # Save to local cache
        self._local_cache[key] = (value, expiry)
        
        # Manage cache size
        if self.strategy_type == "lru":
            if key in self._cache_order:
                self._cache_order.remove(key)
            self._cache_order.append(key)
            
            # Evict if necessary
            while len(self._local_cache) > self.size:
                oldest_key = self._cache_order.pop(0)
                del self._local_cache[oldest_key]
        
        elif self.strategy_type == "lfu":
            # Simple LFU implementation (could be enhanced)
            if len(self._local_cache) > self.size:
                # Remove random item for simplicity
                # In production, track access frequency
                remove_key = next(iter(self._local_cache))
                del self._local_cache[remove_key]
    
    def clear(self):
        """Clear all cache"""
        self._local_cache.clear()
        self._cache_order.clear()
        
        if self.redis_client:
            try:
                # Clear Redis cache keys
                for key in self.redis_client.scan_iter("cache:*"):
                    self.redis_client.delete(key)
            except Exception as e:
                logger.warning(f"Redis cache clear failed: {e}")


class GracefulDegradationManager:
    """
    Manages graceful degradation for the entire system
    """
    
    def __init__(self, config_path: str = "/opt/sutazaiapp/self-healing/config/self-healing-config.yaml"):
        self.config_path = config_path
        self.feature_flags: Dict[str, FeatureFlag] = {}
        self.cache_strategies: Dict[str, CacheStrategy] = {}
        self.redis_client = None
        self._load_config()
        self._init_redis()
        self._init_feature_flags()
        self._init_cache_strategies()
        
    def _load_config(self):
        """Load configuration from YAML"""
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)
            self.degradation_config = self.config.get('graceful_degradation', {})
            
    def _init_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = redis.Redis(
                host='redis',
                port=6379,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info("Connected to Redis for graceful degradation")
        except Exception as e:
            logger.warning(f"Failed to connect to Redis: {e}")
            self.redis_client = None
            
    def _init_feature_flags(self):
        """Initialize feature flags from config"""
        flags_config = self.degradation_config.get('feature_flags', {})
        
        for flag_name, flag_config in flags_config.items():
            flag = FeatureFlag(
                name=flag_name,
                enabled=flag_config.get('enabled', True),
                fallback=flag_config.get('fallback', 'disabled'),
                cache_ttl=self._parse_duration(flag_config.get('cache_ttl', 300)),
                config=flag_config
            )
            self.feature_flags[flag_name] = flag
            logger.info(f"Initialized feature flag: {flag_name}")
            
    def _init_cache_strategies(self):
        """Initialize cache strategies from config"""
        cache_config = self.degradation_config.get('cache_strategies', {})
        
        for strategy_name, strategy_config in cache_config.items():
            strategy = CacheStrategy(
                strategy_type=strategy_config.get('type', 'lru'),
                size=strategy_config.get('size', 1000),
                ttl=strategy_config.get('ttl', 300),
                redis_client=self.redis_client
            )
            self.cache_strategies[strategy_name] = strategy
            logger.info(f"Initialized cache strategy: {strategy_name}")
    
    def _parse_duration(self, duration: Any) -> int:
        """Parse duration to seconds"""
        if isinstance(duration, int):
            return duration
        elif isinstance(duration, str):
            if duration.endswith('s'):
                return int(duration[:-1])
            elif duration.endswith('m'):
                return int(duration[:-1]) * 60
            elif duration.endswith('h'):
                return int(duration[:-1]) * 3600
        return 300  # Default 5 minutes
    
    def is_feature_enabled(self, feature_name: str) -> bool:
        """Check if a feature is enabled"""
        flag = self.feature_flags.get(feature_name)
        if flag:
            # Check Redis for dynamic updates
            if self.redis_client:
                try:
                    redis_value = self.redis_client.get(f"feature_flag:{feature_name}")
                    if redis_value:
                        flag.enabled = redis_value.lower() == 'true'
                except Exception as e:
                    logger.warning(f"Failed to check Redis for feature flag: {e}")
            
            return flag.is_enabled()
        return True  # Default to enabled if not configured
    
    def get_fallback_strategy(self, feature_name: str) -> str:
        """Get fallback strategy for a feature"""
        flag = self.feature_flags.get(feature_name)
        if flag:
            return flag.fallback
        return "disabled"
    
    def get_cache_strategy(self, strategy_name: str) -> Optional[CacheStrategy]:
        """Get a cache strategy"""
        return self.cache_strategies.get(strategy_name, self.cache_strategies.get('default'))
    
    def toggle_feature(self, feature_name: str, enabled: bool):
        """Toggle a feature flag"""
        flag = self.feature_flags.get(feature_name)
        if flag:
            flag.enabled = enabled
            # Update Redis
            if self.redis_client:
                try:
                    self.redis_client.setex(
                        f"feature_flag:{feature_name}",
                        3600,  # 1 hour TTL
                        str(enabled)
                    )
                except Exception as e:
                    logger.error(f"Failed to update feature flag in Redis: {e}")
            logger.info(f"Feature {feature_name} set to {enabled}")
    
    def get_all_features(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all feature flags"""
        return {
            name: {
                "enabled": flag.is_enabled(),
                "fallback": flag.fallback,
                "config": flag.config
            }
            for name, flag in self.feature_flags.items()
        }
    
    def cache_with_fallback(self, 
                           key: str, 
                           compute_func: Callable,
                           cache_strategy: str = "default",
                           fallback_value: Any = None):
        """
        Get value from cache or compute and cache it
        Falls back to fallback_value if compute fails
        """
        strategy = self.get_cache_strategy(cache_strategy)
        if not strategy:
            strategy = CacheStrategy()  # Default strategy
        
        # Try to get from cache
        cached_value = strategy.get(key)
        if cached_value is not None:
            logger.debug(f"Cache hit for {key}")
            return cached_value
        
        # Compute value
        try:
            value = compute_func()
            strategy.set(key, value)
            return value
        except Exception as e:
            logger.error(f"Failed to compute value for {key}: {e}")
            if fallback_value is not None:
                return fallback_value
            raise


# Decorators for easy graceful degradation
def feature_flag(feature_name: str, fallback_func: Optional[Callable] = None):
    """
    Decorator to protect a function with a feature flag
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = GracefulDegradationManager()
            
            if manager.is_feature_enabled(feature_name):
                return func(*args, **kwargs)
            else:
                logger.info(f"Feature {feature_name} is disabled, using fallback")
                if fallback_func:
                    return fallback_func(*args, **kwargs)
                else:
                    fallback_strategy = manager.get_fallback_strategy(feature_name)
                    if fallback_strategy == "cache":
                        # Try to return cached result
                        cache_key = f"{feature_name}:{func.__name__}:{str(args)}"
                        strategy = manager.get_cache_strategy(feature_name)
                        if strategy:
                            cached = strategy.get(cache_key)
                            if cached:
                                return cached
                    elif fallback_strategy == "basic":
                        # Return basic version if available
                        basic_func_name = f"{func.__name__}_basic"
                        if hasattr(func.__module__, basic_func_name):
                            basic_func = getattr(func.__module__, basic_func_name)
                            return basic_func(*args, **kwargs)
                    
                    # Default: return None or raise exception
                    return None
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = GracefulDegradationManager()
            
            if manager.is_feature_enabled(feature_name):
                return await func(*args, **kwargs)
            else:
                logger.info(f"Feature {feature_name} is disabled, using fallback")
                if fallback_func:
                    if asyncio.iscoroutinefunction(fallback_func):
                        return await fallback_func(*args, **kwargs)
                    else:
                        return fallback_func(*args, **kwargs)
                else:
                    # Similar fallback logic for async
                    return None
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


def cache_on_failure(cache_strategy: str = "default", ttl: int = 300):
    """
    Decorator to cache function results and use cache on failure
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            manager = GracefulDegradationManager()
            cache_key = f"{func.__module__}.{func.__name__}:{str(args)}:{str(kwargs)}"
            
            def compute():
                return func(*args, **kwargs)
            
            return manager.cache_with_fallback(
                key=cache_key,
                compute_func=compute,
                cache_strategy=cache_strategy
            )
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            manager = GracefulDegradationManager()
            cache_key = f"{func.__module__}.{func.__name__}:{str(args)}:{str(kwargs)}"
            
            async def compute():
                return await func(*args, **kwargs)
            
            # For async, we need a slightly different approach
            strategy = manager.get_cache_strategy(cache_strategy)
            if strategy:
                cached = strategy.get(cache_key)
                if cached:
                    return cached
            
            try:
                result = await func(*args, **kwargs)
                if strategy:
                    strategy.set(cache_key, result)
                return result
            except Exception as e:
                logger.error(f"Function failed, checking cache: {e}")
                if strategy:
                    cached = strategy.get(cache_key)
                    if cached:
                        return cached
                raise
        
        # Return appropriate wrapper
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return wrapper
    
    return decorator


if __name__ == "__main__":
    # Example usage
    manager = GracefulDegradationManager()
    
    # Print all feature flags
    print("Feature Flags Status:")
    for name, status in manager.get_all_features().items():
        print(f"\n{name}:")
        for key, value in status.items():
            print(f"  {key}: {value}")
    
    # Example of toggling a feature
    manager.toggle_feature("ai_suggestions", False)
    print(f"\nAI Suggestions enabled: {manager.is_feature_enabled('ai_suggestions')}")