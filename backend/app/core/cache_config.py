"""
Optimized Cache Configuration for 80%+ Hit Rates
"""
import os
from typing import Any, Dict, Optional

class CacheConfig:
    """High-performance cache configuration"""
    
    # Redis connection settings
    REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:10001/0")
    REDIS_MAX_CONNECTIONS = 50
    REDIS_DECODE_RESPONSES = True
    REDIS_SOCKET_TIMEOUT = 5
    REDIS_SOCKET_CONNECT_TIMEOUT = 5
    REDIS_SOCKET_KEEPALIVE = True
    REDIS_SOCKET_KEEPALIVE_OPTIONS = {
        1: 3,   # TCP_KEEPIDLE
        2: 3,   # TCP_KEEPINTVL  
        3: 3,   # TCP_KEEPCNT
    }
    
    # Cache TTL settings (in seconds)
    DEFAULT_TTL = 3600              # 1 hour default
    SHORT_TTL = 300                 # 5 minutes for volatile data
    MEDIUM_TTL = 1800               # 30 minutes for semi-static data
    LONG_TTL = 86400                # 24 hours for static data
    
    # Cache key patterns
    CACHE_KEY_PREFIX = "sutazai:"
    USER_CACHE_PREFIX = f"{CACHE_KEY_PREFIX}user:"
    API_CACHE_PREFIX = f"{CACHE_KEY_PREFIX}api:"
    MODEL_CACHE_PREFIX = f"{CACHE_KEY_PREFIX}model:"
    AGENT_CACHE_PREFIX = f"{CACHE_KEY_PREFIX}agent:"
    
    # Performance settings
    ENABLE_COMPRESSION = True       # Compress large values
    COMPRESSION_THRESHOLD = 1024    # Compress if > 1KB
    ENABLE_CACHE_WARMING = True     # Pre-warm cache on startup
    ENABLE_REQUEST_COALESCING = True # Coalesce duplicate requests
    MAX_CACHE_SIZE_MB = 512         # Maximum cache size
    
    # Cache warming patterns
    WARM_CACHE_PATTERNS = [
        "api:models:*",
        "api:agents:*",
        "api:health",
        "model:tinyllama:*"
    ]
    
    # Monitoring
    ENABLE_CACHE_METRICS = True
    METRICS_SAMPLE_RATE = 0.1      # Sample 10% of operations
    
    @classmethod
    def get_ttl_for_key(cls, key: str) -> int:
        """Get appropriate TTL based on key pattern"""
        if "health" in key or "status" in key:
            return cls.SHORT_TTL
        elif "model" in key or "agent" in key:
            return cls.MEDIUM_TTL
        elif "config" in key or "static" in key:
            return cls.LONG_TTL
        return cls.DEFAULT_TTL
