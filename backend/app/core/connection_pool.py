"""
High-Performance Connection Pooling Manager
Handles HTTP, Database, and Redis connections with proper pooling
"""

import os
import asyncio
import logging
import socket
from typing import Optional, Dict, Any
from contextlib import asynccontextmanager
import httpx
import asyncpg
import redis.asyncio as redis
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class ConnectionPoolManager:
    """Centralized connection pooling for all external services"""
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            self._http_clients: Dict[str, httpx.AsyncClient] = {}
            self._db_pool: Optional[asyncpg.Pool] = None
            self._redis_pool: Optional[redis.ConnectionPool] = None
            self._redis_client: Optional[redis.Redis] = None
            self._stats = {
                'http_requests': 0,
                'db_queries': 0,
                'redis_operations': 0,
                'connection_errors': 0,
                'pool_exhaustion': 0
            }
            
    async def initialize(self, config: Dict[str, Any]):
        """Initialize all connection pools"""
        try:
            # Initialize Redis pool
            self._redis_pool = redis.ConnectionPool(
                host=config.get('redis_host', 'sutazai-redis'),
                port=config.get('redis_port', 6379),
                db=0,
                max_connections=50,
                socket_keepalive=True,
                socket_keepalive_options={
                    socket.TCP_KEEPIDLE: 1,  # TCP_KEEPIDLE
                    socket.TCP_KEEPINTVL: 3,  # TCP_KEEPINTVL
                    socket.TCP_KEEPCNT: 5,  # TCP_KEEPCNT
                },
                decode_responses=False
            )
            self._redis_client = redis.Redis(connection_pool=self._redis_pool)
            
            # Initialize PostgreSQL pool
            self._db_pool = await asyncpg.create_pool(
                host=config.get('db_host', 'sutazai-postgres'),
                port=config.get('db_port', 5432),
                user=config.get('db_user', 'sutazai'),
                password=config.get('db_password', 'sutazai'),
                database=config.get('db_name', 'sutazai'),
                min_size=10,
                max_size=20,
                max_queries=50000,
                max_inactive_connection_lifetime=300,
                command_timeout=60
            )
            
            # Initialize HTTP clients for different services
            self._initialize_http_clients(config)
            
            logger.info("Connection pools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pools: {e}")
            raise
            
    def _initialize_http_clients(self, config: Dict[str, Any]):
        """Initialize HTTP clients with connection pooling"""
        
        # Default limits for connection pooling
        limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=100,
            keepalive_expiry=30
        )
        
        # Ollama client with longer timeout for LLM operations
        self._http_clients['ollama'] = httpx.AsyncClient(
            base_url=config.get('ollama_url', 'http://sutazai-ollama:11434'),
            limits=limits,
            timeout=httpx.Timeout(
                connect=5.0,
                read=30.0,  # Reduced from 120s
                write=10.0,
                pool=5.0
            ),
            http2=False  # Disabled until h2 package is installed
        )
        
        # Agent services client with standard timeout
        self._http_clients['agents'] = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(
                connect=2.0,
                read=10.0,
                write=5.0,
                pool=2.0
            ),
            http2=False  # Disabled until h2 package is installed
        )
        
        # External API client
        self._http_clients['external'] = httpx.AsyncClient(
            limits=limits,
            timeout=httpx.Timeout(
                connect=5.0,
                read=30.0,
                write=10.0,
                pool=5.0
            ),
            http2=False,  # Disabled until h2 package is installed
            follow_redirects=True
        )
        
    @asynccontextmanager
    async def get_http_client(self, service: str = 'default'):
        """Get HTTP client for specific service"""
        if service not in self._http_clients:
            service = 'agents'  # Default to agents client
            
        client = self._http_clients[service]
        self._stats['http_requests'] += 1
        
        try:
            yield client
        except httpx.PoolTimeout:
            self._stats['pool_exhaustion'] += 1
            logger.warning(f"HTTP connection pool exhausted for {service}")
            raise
        except Exception as e:
            self._stats['connection_errors'] += 1
            logger.error(f"HTTP request error for {service}: {e}")
            raise
            
    @asynccontextmanager
    async def get_db_connection(self):
        """Get database connection from pool"""
        if not self._db_pool:
            raise RuntimeError("Database pool not initialized")
            
        self._stats['db_queries'] += 1
        
        try:
            async with self._db_pool.acquire() as connection:
                yield connection
        except asyncpg.TooManyConnectionsError:
            self._stats['pool_exhaustion'] += 1
            logger.warning("Database connection pool exhausted")
            raise
        except Exception as e:
            self._stats['connection_errors'] += 1
            logger.error(f"Database connection error: {e}")
            raise
            
    def get_redis_client(self) -> redis.Redis:
        """Get Redis client (uses internal connection pool)"""
        if not self._redis_client:
            raise RuntimeError("Redis client not initialized")
            
        self._stats['redis_operations'] += 1
        return self._redis_client
        
    async def execute_db_query(self, query: str, *args, fetch_one: bool = False):
        """Execute database query with automatic connection management"""
        async with self.get_db_connection() as conn:
            if fetch_one:
                return await conn.fetchrow(query, *args)
            return await conn.fetch(query, *args)
            
    async def execute_db_command(self, query: str, *args):
        """Execute database command (INSERT, UPDATE, DELETE)"""
        async with self.get_db_connection() as conn:
            return await conn.execute(query, *args)
            
    async def health_check(self) -> Dict[str, Any]:
        """Check health of all connection pools"""
        health = {
            'status': 'healthy',
            'pools': {},
            'stats': self._stats
        }
        
        # Check Redis
        try:
            await self._redis_client.ping()
            health['pools']['redis'] = 'healthy'
        except Exception as e:
            health['pools']['redis'] = f'unhealthy: {e}'
            health['status'] = 'degraded'
            
        # Check Database
        try:
            await self.execute_db_query("SELECT 1", fetch_one=True)
            health['pools']['database'] = 'healthy'
        except Exception as e:
            health['pools']['database'] = f'unhealthy: {e}'
            health['status'] = 'degraded'
            
        # Check HTTP clients
        for name, client in self._http_clients.items():
            try:
                # Just check if client is configured
                health['pools'][f'http_{name}'] = 'configured'
            except Exception as e:
                health['pools'][f'http_{name}'] = f'error: {e}'
                
        return health
        
    def get_stats(self) -> Dict[str, int]:
        """Get connection pool statistics"""
        stats = self._stats.copy()
        
        # Add pool-specific stats
        if self._db_pool:
            stats['db_pool_size'] = self._db_pool.get_size()
            stats['db_pool_free'] = self._db_pool.get_idle_size()
            
        return stats
        
    async def close(self):
        """Close all connection pools"""
        # Close HTTP clients
        for client in self._http_clients.values():
            await client.aclose()
        self._http_clients.clear()
        
        # Close database pool
        if self._db_pool:
            await self._db_pool.close()
            self._db_pool = None
            
        # Close Redis
        if self._redis_client:
            await self._redis_client.close()
            self._redis_client = None
            
        logger.info("All connection pools closed")


# Global singleton instance
_pool_manager: Optional[ConnectionPoolManager] = None


async def get_pool_manager() -> ConnectionPoolManager:
    """Get or create the global connection pool manager"""
    global _pool_manager
    
    if _pool_manager is None:
        async with ConnectionPoolManager._lock:
            if _pool_manager is None:
                _pool_manager = ConnectionPoolManager()
                # Initialize with default config
                await _pool_manager.initialize({
                    'redis_host': os.getenv('REDIS_HOST', 'redis'),
                    'redis_port': int(os.getenv('REDIS_PORT', '6379')),
                    'db_host': os.getenv('POSTGRES_HOST', 'postgres'),
                    'db_port': int(os.getenv('POSTGRES_PORT', '5432')),
                    'db_user': os.getenv('POSTGRES_USER', 'sutazai'),
                    # Security: do not use hardcoded fallback for passwords.
                    # Require POSTGRES_PASSWORD to come from the environment.
                    'db_password': os.getenv('POSTGRES_PASSWORD'),
                    'db_name': os.getenv('POSTGRES_DB', 'sutazai'),
                    'ollama_url': os.getenv('OLLAMA_URL', 'http://ollama:11434')
                })
                
    return _pool_manager


# Convenience functions
async def get_http_client(service: str = 'default'):
    """Quick access to HTTP client"""
    manager = await get_pool_manager()
    return manager.get_http_client(service)


async def get_redis() -> redis.Redis:
    """Quick access to Redis client"""
    manager = await get_pool_manager()
    return manager.get_redis_client()


async def execute_query(query: str, *args, fetch_one: bool = False):
    """Quick database query execution"""
    manager = await get_pool_manager()
    return await manager.execute_db_query(query, *args, fetch_one=fetch_one)
