"""
Redis Connection Manager - Breaking Circular Dependency
========================================================
This module provides Redis connections WITHOUT importing cache.py
to break the circular dependency that causes backend deadlock.
"""

import os
import socket
import logging
from typing import Optional
import redis.asyncio as redis

logger = logging.getLogger(__name__)

# Global Redis client instance
_redis_client: Optional[redis.Redis] = None
_redis_pool: Optional[redis.ConnectionPool] = None

async def get_redis_connection() -> redis.Redis:
    """
    Get Redis connection without circular imports.
    This is the PRIMARY Redis connection function that should be used everywhere.
    """
    global _redis_client, _redis_pool
    
    if _redis_client is None:
        # Determine environment
        is_container = os.path.exists("/.dockerenv")
        
        # Use proper hostnames based on environment
        redis_host = os.getenv("REDIS_HOST", 
                               "sutazai-redis" if is_container else "localhost")
        redis_port = int(os.getenv("REDIS_PORT", 
                                   "6379" if is_container else "10001"))
        
        logger.info(f"Initializing Redis connection to {redis_host}:{redis_port}")
        
        # Create connection pool with optimized settings
        redis_config = {
            'host': redis_host,
            'port': redis_port,
            'db': 0,
            'max_connections': 50,
            'socket_keepalive': True,
            'socket_keepalive_options': {
                socket.TCP_KEEPIDLE: 1,
                socket.TCP_KEEPINTVL: 3,
                socket.TCP_KEEPCNT: 5,
            },
            'decode_responses': False,
            'socket_connect_timeout': 5,
            'socket_timeout': 5,
            'retry_on_timeout': True,
            'retry_on_error': [ConnectionError, TimeoutError],
            'health_check_interval': 30
        }
        
        # Add password if configured
        redis_password = os.getenv('REDIS_PASSWORD')
        if redis_password and redis_password.strip():
            redis_config['password'] = redis_password
            logger.info("Redis configured with authentication")
        
        try:
            _redis_pool = redis.ConnectionPool(**redis_config)
            _redis_client = redis.Redis(connection_pool=_redis_pool)
            
            # Test connection
            await _redis_client.ping()
            logger.info("✅ Redis connection established successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            # Return a dummy client that won't break the application
            _redis_client = None
            raise
    
    return _redis_client

async def close_redis_connection():
    """Close Redis connection pool properly"""
    global _redis_client, _redis_pool
    
    if _redis_client:
        await _redis_client.close()
        _redis_client = None
    
    if _redis_pool:
        await _redis_pool.disconnect()
        _redis_pool = None
    
    logger.info("Redis connections closed")

# Backward compatibility alias
get_redis = get_redis_connection