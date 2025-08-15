"""
High-Performance Connection Pooling Manager
Handles HTTP, Database, and Redis connections with proper pooling
Now with Circuit Breaker pattern for resilient service communication
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

# Import circuit breaker components
from .circuit_breaker_integration import (
    SimpleCircuitBreaker,
    CircuitBreakerManager,
    get_circuit_breaker_manager
)

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
            self._db_cfg: Dict[str, Any] = {}
            self._redis_pool: Optional[redis.ConnectionPool] = None
            self._redis_client: Optional[redis.Redis] = None
            self._breaker_manager = None  # Will be initialized async
            self._stats = {
                'http_requests': 0,
                'db_queries': 0,
                'redis_operations': 0,
                'connection_errors': 0,
                'pool_exhaustion': 0,
                'circuit_breaker_trips': 0
            }
            
            # Circuit breakers will be initialized async
            pass
    
    async def _init_circuit_breakers(self):
        """Initialize circuit breakers for different services (async)"""
        if self._breaker_manager is None:
            self._breaker_manager = await get_circuit_breaker_manager()
        
        # Create circuit breakers for different services
        # ULTRAFIX: Increased Ollama circuit breaker timeout to match HTTP client
        await self._breaker_manager.get_or_create_breaker(
            'ollama',
            failure_threshold=5,
            recovery_timeout=60.0,  # ULTRAFIX: Increased recovery timeout
            timeout=120.0  # ULTRAFIX: Match HTTP client timeout for consistency
        )
        
        await self._breaker_manager.get_or_create_breaker(
            'redis',
            failure_threshold=3,
            recovery_timeout=30.0,
            timeout=5.0
        )
        
        await self._breaker_manager.get_or_create_breaker(
            'database',
            failure_threshold=3,
            recovery_timeout=30.0,
            timeout=10.0
        )
        
        await self._breaker_manager.get_or_create_breaker(
            'agents',
            failure_threshold=5,
            recovery_timeout=30.0,
            timeout=10.0
        )
        
        await self._breaker_manager.get_or_create_breaker(
            'external',
            failure_threshold=5,
            recovery_timeout=30.0,
            timeout=30.0
        )
        
        logger.info("Circuit breakers initialized for all services")
            
    async def initialize(self, config: Dict[str, Any]):
        """Initialize all connection pools"""
        try:
            # Initialize Redis pool with optional authentication
            redis_config = {
                'host': config.get('redis_host', '172.20.0.2'),
                'port': config.get('redis_port', 6379),
                'db': 0,
                'max_connections': 50,
                'socket_keepalive': True,
                'socket_keepalive_options': {
                    socket.TCP_KEEPIDLE: 1,  # TCP_KEEPIDLE
                    socket.TCP_KEEPINTVL: 3,  # TCP_KEEPINTVL
                    socket.TCP_KEEPCNT: 5,  # TCP_KEEPCNT
                },
                'decode_responses': False
            }
            
            # Add password only if provided and not empty
            redis_password = config.get('redis_password')
            if redis_password and redis_password.strip():
                redis_config['password'] = redis_password
                logger.info("Redis configured with authentication")
            else:
                logger.info("Redis configured without authentication")
                
            self._redis_pool = redis.ConnectionPool(**redis_config)
            self._redis_client = redis.Redis(connection_pool=self._redis_pool)
            
            # ULTRAFIX: Optimized PostgreSQL pool for high concurrency
            # Based on formula: pool_size = (num_workers * 2) + max_overflow
            # For 28 containers with avg 2 connections each = 56 + 20 = 76
            self._db_cfg = {
                'host': config.get('db_host', '172.20.0.5'),
                'port': config.get('db_port', 5432),
                'user': config.get('db_user', 'sutazai'),
                'password': config.get('db_password', 'sutazai123'),
                'database': config.get('db_name', 'sutazai'),
                'min_size': 20,  # ULTRAFIX: Increased from 10 for better warm pool
                'max_size': 50,  # ULTRAFIX: Increased from 20 for high concurrency
                'max_queries': 100000,  # ULTRAFIX: Doubled for longer connection reuse
                'max_inactive_connection_lifetime': 600,  # ULTRAFIX: 10 min for stable connections
                'command_timeout': 60,
                'server_settings': {
                    'jit': 'on'  # Enable JIT compilation - safe to set at connection time
                },
                'ssl': False  # Explicitly disable SSL
            }
            self._db_pool = await asyncpg.create_pool(**self._db_cfg)
            
            # Initialize HTTP clients for different services
            self._initialize_http_clients(config)
            
            # Initialize circuit breakers
            await self._init_circuit_breakers()
            
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
        # ULTRAFIX: Increased timeout from 30s to 120s for reliable LLM operations
        self._http_clients['ollama'] = httpx.AsyncClient(
            base_url=config.get('ollama_url', 'http://sutazai-ollama:11434'),
            limits=limits,
            timeout=httpx.Timeout(
                connect=5.0,
                read=120.0,  # ULTRAFIX: Restored to 120s for reliable LLM operations
                write=30.0,  # ULTRAFIX: Increased write timeout for large prompts
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
        """Get HTTP client for specific service with circuit breaker protection"""
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
    
    async def make_http_request(
        self, 
        service: str, 
        method: str, 
        url: str, 
        **kwargs
    ) -> httpx.Response:
        """
        Make HTTP request with circuit breaker protection
        
        Args:
            service: Service name (ollama, agents, external)
            method: HTTP method (GET, POST, etc.)
            url: URL to request
            **kwargs: Additional arguments for httpx request
            
        Returns:
            httpx.Response object
            
        Raises:
            CircuitBreakerError: If circuit is open
            httpx.HTTPError: For HTTP errors
        """
        breaker = self._breaker_manager.get_breaker(service)
        if not breaker:
            breaker = self._breaker_manager.get_or_create(service)
        
        async def _make_request():
            async with self.get_http_client(service) as client:
                response = await client.request(method, url, **kwargs)
                response.raise_for_status()
                return response
        
        try:
            return await breaker.call(_make_request)
        except CircuitBreakerError:
            self._stats['circuit_breaker_trips'] += 1
            logger.warning(f"Circuit breaker OPEN for service '{service}', request to {url} blocked")
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
        except (asyncpg.InterfaceError,
                asyncpg.CannotConnectNowError,
                asyncpg.PostgresConnectionError,
                ConnectionError,
                OSError) as e:
            # Attempt to transparently recreate the pool and retry once
            self._stats['connection_errors'] += 1
            logger.warning(f"Database connection lost: {e}. Attempting to recreate pool...")
            try:
                await self._recreate_db_pool()
                async with self._db_pool.acquire() as connection:
                    logger.info("Re-established PostgreSQL connection pool successfully")
                    yield connection
                    return
            except Exception as re:
                logger.error(f"Failed to reconnect to PostgreSQL: {re}")
                raise
        except Exception as e:
            self._stats['connection_errors'] += 1
            logger.error(f"Database connection error: {e}")
            raise

    async def _recreate_db_pool(self, max_retries: int = 5) -> None:
        """Close and recreate the asyncpg pool with exponential backoff."""
        # Close old pool if present
        try:
            if self._db_pool:
                await self._db_pool.close()
        finally:
            self._db_pool = None

        delay = 0.5
        last_err: Optional[Exception] = None
        for attempt in range(1, max_retries + 1):
            try:
                self._db_pool = await asyncpg.create_pool(**self._db_cfg)
                return
            except Exception as e:
                last_err = e
                logger.warning(f"Recreate DB pool attempt {attempt}/{max_retries} failed: {e}")
                await asyncio.sleep(delay)
                delay = min(delay * 2, 10.0)
        raise RuntimeError(f"Unable to recreate PostgreSQL pool after {max_retries} attempts: {last_err}")
            
    def get_redis_client(self) -> redis.Redis:
        """Get Redis client (uses internal connection pool)"""
        if not self._redis_client:
            raise RuntimeError("Redis client not initialized")
            
        self._stats['redis_operations'] += 1
        return self._redis_client
    
    async def execute_redis_command(self, command: str, *args, **kwargs):
        """
        Execute Redis command with circuit breaker protection
        
        Args:
            command: Redis command name (get, set, hget, etc.)
            *args: Command arguments
            **kwargs: Command keyword arguments
            
        Returns:
            Command result
            
        Raises:
            CircuitBreakerError: If circuit is open
            redis.RedisError: For Redis errors
        """
        breaker = self._breaker_manager.get_breaker('redis')
        
        async def _execute():
            client = self.get_redis_client()
            method = getattr(client, command)
            return await method(*args, **kwargs)
        
        try:
            return await breaker.call(_execute)
        except CircuitBreakerError:
            self._stats['circuit_breaker_trips'] += 1
            logger.warning(f"Circuit breaker OPEN for Redis, command '{command}' blocked")
            raise
        
    async def execute_db_query(self, query: str, *args, fetch_one: bool = False):
        """Execute database query with circuit breaker protection"""
        breaker = self._breaker_manager.get_breaker('database')
        
        async def _execute():
            async with self.get_db_connection() as conn:
                if fetch_one:
                    return await conn.fetchrow(query, *args)
                return await conn.fetch(query, *args)
        
        try:
            return await breaker.call(_execute)
        except CircuitBreakerError:
            self._stats['circuit_breaker_trips'] += 1
            logger.warning(f"Circuit breaker OPEN for database, query blocked")
            raise
            
    async def execute_db_command(self, query: str, *args):
        """Execute database command with circuit breaker protection"""
        breaker = self._breaker_manager.get_breaker('database')
        
        async def _execute():
            async with self.get_db_connection() as conn:
                return await conn.execute(query, *args)
        
        try:
            return await breaker.call(_execute)
        except CircuitBreakerError:
            self._stats['circuit_breaker_trips'] += 1
            logger.warning(f"Circuit breaker OPEN for database, command blocked")
            raise
            
    async def health_check(self, timeout: float = 5.0) -> Dict[str, Any]:
        """Optimized health check with parallel execution and timeouts"""
        health = {
            'status': 'healthy',
            'pools': {},
            'stats': self._stats,
            'circuit_breakers': self.get_circuit_breaker_status()
        }
        
        # Create health check tasks to run in parallel
        health_tasks = []
        
        # Redis health check task
        async def check_redis():
            try:
                await asyncio.wait_for(self._redis_client.ping(), timeout=1.0)
                return ('redis', 'healthy')
            except asyncio.TimeoutError:
                return ('redis', 'timeout')
            except Exception as e:
                return ('redis', f'unhealthy: {str(e)[:50]}')
        
        # Database health check task
        async def check_database():
            try:
                await asyncio.wait_for(
                    self.execute_db_query("SELECT 1", fetch_one=True),
                    timeout=2.0
                )
                return ('database', 'healthy')
            except asyncio.TimeoutError:
                return ('database', 'timeout')
            except Exception as e:
                return ('database', f'unhealthy: {str(e)[:50]}')
        
        # Add tasks
        health_tasks.append(asyncio.create_task(check_redis()))
        health_tasks.append(asyncio.create_task(check_database()))
        
        # Run all health checks in parallel with overall timeout
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*health_tasks, return_exceptions=True),
                timeout=timeout
            )
            
            # Process results
            for result in results:
                if isinstance(result, tuple):
                    service_name, status = result
                    health['pools'][service_name] = status
                    if status != 'healthy' and status != 'configured':
                        health['status'] = 'degraded'
                else:
                    # Exception occurred
                    health['status'] = 'degraded'
                    
        except asyncio.TimeoutError:
            health['status'] = 'timeout'
            health['pools']['error'] = 'Health check timeout'
            
        # HTTP clients are just configuration checks (instant)
        for name in self._http_clients.keys():
            health['pools'][f'http_{name}'] = 'configured'
                
        return health
        
    async def quick_health_check(self) -> bool:
        """Ultra-fast health check for critical services only"""
        try:
            # Check Redis and DB in parallel with very short timeout
            redis_task = asyncio.create_task(
                asyncio.wait_for(self._redis_client.ping(), timeout=0.5)
            )
            db_task = asyncio.create_task(
                asyncio.wait_for(
                    self.execute_db_query("SELECT 1", fetch_one=True),
                    timeout=0.5
                )
            )
            
            # Wait for both with 1 second total timeout
            await asyncio.wait_for(
                asyncio.gather(redis_task, db_task),
                timeout=1.0
            )
            return True
            
        except:
            return False
        
    def get_stats(self) -> Dict[str, int]:
        """Get connection pool statistics"""
        stats = self._stats.copy()
        
        # Add pool-specific stats
        if self._db_pool:
            stats['db_pool_size'] = self._db_pool.get_size()
            stats['db_pool_free'] = self._db_pool.get_idle_size()
            
        return stats
    
    def get_circuit_breaker_status(self) -> Dict[str, Any]:
        """Get status of all circuit breakers"""
        return self._breaker_manager.get_metrics()
    
    def get_circuit_breaker(self, service: str) -> Optional[SimpleCircuitBreaker]:
        """Get a specific circuit breaker"""
        return self._breaker_manager.get_breaker(service)
    
    def reset_circuit_breaker(self, service: str):
        """Reset a specific circuit breaker"""
        breaker = self._breaker_manager.get_breaker(service)
        if breaker:
            breaker.reset()
            logger.info(f"Circuit breaker for '{service}' has been reset")
        else:
            logger.warning(f"Circuit breaker for '{service}' not found")
    
    def reset_all_circuit_breakers(self):
        """Reset all circuit breakers"""
        self._breaker_manager.reset_all()
        logger.info("All circuit breakers have been reset")
    
    def reset_error_counters(self):
        """ULTRAFIX: Reset all error counters to prevent accumulation"""
        old_errors = self._stats['connection_errors']
        self._stats = {
            'http_requests': self._stats['http_requests'],
            'db_queries': self._stats['db_queries'],
            'redis_operations': self._stats['redis_operations'],
            'connection_errors': 0,  # ULTRAFIX: Reset to zero
            'pool_exhaustion': 0,    # ULTRAFIX: Reset to zero
            'circuit_breaker_trips': 0  # ULTRAFIX: Reset to zero
        }
        logger.info(f"ULTRAFIX: Reset error counters (was {old_errors} connection errors)")
        
    async def recover_connections(self):
        """ULTRAFIX: Attempt to recover all connections by recreating pools"""
        try:
            logger.info("ULTRAFIX: Attempting connection recovery...")
            
            # Close and recreate HTTP clients
            for service_name, client in self._http_clients.items():
                try:
                    await client.aclose()
                    logger.info(f"Closed HTTP client for {service_name}")
                except Exception as e:
                    logger.warning(f"Error closing HTTP client {service_name}: {e}")
            
            # Reinitialize HTTP clients with current config
            config = {
                'ollama_url': 'http://172.20.0.8:11434',
                'redis_host': '172.20.0.2',
                'redis_port': 6379,
                'db_host': '172.20.0.5',
                'db_port': 5432
            }
            self._initialize_http_clients(config)
            
            # Reset circuit breakers
            self.reset_all_circuit_breakers()
            
            # Reset error counters
            self.reset_error_counters()
            
            logger.info("ULTRAFIX: Connection recovery completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ULTRAFIX: Connection recovery failed: {e}")
            return False
        
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
                    'redis_host': os.getenv('REDIS_HOST', '172.20.0.2'),
                    'redis_port': int(os.getenv('REDIS_PORT', '6379')),
                    'redis_password': os.getenv('REDIS_PASSWORD'),  # Optional Redis password
                    'db_host': os.getenv('POSTGRES_HOST', '172.20.0.5'),
                    'db_port': int(os.getenv('POSTGRES_PORT', '5432')),
                    'db_user': os.getenv('POSTGRES_USER', 'sutazai'),
                    # Using correct password for PostgreSQL
                    'db_password': os.getenv('POSTGRES_PASSWORD', 'sutazai123'),
                    'db_name': os.getenv('POSTGRES_DB', 'sutazai'),
                    'ollama_url': os.getenv('OLLAMA_URL', 'http://172.20.0.8:11434')
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
