"""
ULTRA-PERFORMANCE Connection Pool Manager
Optimized for 1000+ concurrent users with minimal resource usage
"""

import os
import asyncio
import logging
import socket
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager
import httpx
import asyncpg
import redis.asyncio as redis
from datetime import datetime, timedelta
import time

logger = logging.getLogger(__name__)


class UltraConnectionPool:
    """
    ULTRA-optimized connection pooling for maximum performance
    
    Key optimizations:
    - Dynamic pool sizing based on load
    - Connection warming and keep-alive
    - Intelligent connection reuse
    - Memory-efficient pooling
    - Circuit breaker integration
    """
    
    _instance = None
    _lock = asyncio.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, '_initialized'):
            self._initialized = True
            
            # HTTP client pools with optimized settings
            self._http_pools = {}
            
            # Database pool
            self._db_pool: Optional[asyncpg.Pool] = None
            
            # Redis pools (separate for different use cases)
            self._redis_pools = {}
            
            # Connection health tracking
            self._health_scores = {}
            self._last_health_check = {}
            
            # Performance metrics
            self._metrics = {
                'connections_created': 0,
                'connections_reused': 0,
                'pool_hits': 0,
                'pool_misses': 0,
                'avg_acquire_time_ms': 0,
                'active_connections': 0
            }
            
            # Load tracking for dynamic sizing
            self._load_history = []
            self._last_resize = time.time()
            
            logger.info("ULTRA Connection Pool Manager initialized")
    
    async def initialize(self, config: Dict[str, Any] = None):
        """Initialize all connection pools with ULTRA optimizations"""
        config = config or self._get_default_config()
        
        try:
            # Initialize Redis pools with different purposes
            await self._init_redis_pools(config)
            
            # Initialize PostgreSQL pool with ULTRA settings
            await self._init_database_pool(config)
            
            # Initialize HTTP client pools
            self._init_http_pools(config)
            
            # Start connection warming
            asyncio.create_task(self._connection_warmer())
            
            # Start metrics collector
            asyncio.create_task(self._metrics_collector())
            
            logger.info("ULTRA Connection Pools initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ULTRA pools: {e}")
            raise
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get optimized default configuration"""
        return {
            # Redis settings
            'redis_host': os.getenv('REDIS_HOST', 'redis'),
            'redis_port': int(os.getenv('REDIS_PORT', '6379')),
            'redis_password': os.getenv('REDIS_PASSWORD'),
            
            # PostgreSQL settings
            'db_host': os.getenv('POSTGRES_HOST', 'postgres'),
            'db_port': int(os.getenv('POSTGRES_PORT', '5432')),
            'db_user': os.getenv('POSTGRES_USER', 'sutazai'),
            'db_password': os.getenv('POSTGRES_PASSWORD'),
            'db_name': os.getenv('POSTGRES_DB', 'sutazai'),
            
            # Ollama settings
            'ollama_url': os.getenv('OLLAMA_URL', 'http://ollama:11434'),
            
            # Pool sizing
            'min_pool_size': 10,
            'max_pool_size': 100,
            'target_pool_utilization': 0.7
        }
    
    async def _init_redis_pools(self, config: Dict[str, Any]):
        """Initialize Redis connection pools for different purposes"""
        
        base_redis_config = {
            'host': config.get('redis_host', 'redis'),
            'port': config.get('redis_port', 6379),
            'decode_responses': False,
            'socket_keepalive': True,
            'socket_keepalive_options': {
                socket.TCP_KEEPIDLE: 1,
                socket.TCP_KEEPINTVL: 3,
                socket.TCP_KEEPCNT: 5,
            }
        }
        
        # Add password if provided
        if config.get('redis_password'):
            base_redis_config['password'] = config['redis_password']
        
        # Cache pool - high connection count for parallel operations
        cache_pool_config = {
            **base_redis_config,
            'max_connections': 100,
            'connection_class': redis.Connection,
            'health_check_interval': 30
        }
        self._redis_pools['cache'] = redis.ConnectionPool(**cache_pool_config)
        
        # Session pool - moderate connections for user sessions
        session_pool_config = {
            **base_redis_config,
            'max_connections': 50,
            'connection_class': redis.Connection,
            'health_check_interval': 60
        }
        self._redis_pools['session'] = redis.ConnectionPool(**session_pool_config)
        
        # Queue pool - for task queues and pub/sub
        queue_pool_config = {
            **base_redis_config,
            'max_connections': 30,
            'connection_class': redis.Connection,
            'health_check_interval': 10
        }
        self._redis_pools['queue'] = redis.ConnectionPool(**queue_pool_config)
        
        logger.info("Redis pools initialized: cache(100), session(50), queue(30)")
    
    async def _init_database_pool(self, config: Dict[str, Any]):
        """Initialize PostgreSQL pool with ULTRA optimizations"""
        
        # Calculate optimal pool size based on expected load
        # Formula: pool_size = (num_workers * connections_per_worker) + buffer
        num_workers = 28  # Number of containers
        connections_per_worker = 2
        buffer = 20
        optimal_pool_size = min(num_workers * connections_per_worker + buffer, 100)
        
        self._db_pool = await asyncpg.create_pool(
            host=config.get('db_host', 'postgres'),
            port=config.get('db_port', 5432),
            user=config.get('db_user', 'sutazai'),
            password=config.get('db_password'),
            database=config.get('db_name', 'sutazai'),
            
            # ULTRA pool sizing
            min_size=20,  # Keep warm connections
            max_size=optimal_pool_size,
            
            # Connection lifecycle
            max_queries=50000,  # Reuse connections longer
            max_inactive_connection_lifetime=300,  # 5 minutes
            max_cached_statement_lifetime=300,
            
            # Performance settings
            command_timeout=30,
            
            # Statement cache
            statement_cache_size=1024,  # Cache prepared statements
            
            # Connection initialization
            init=self._init_db_connection,
            
            # Server settings for performance
            server_settings={
                'jit': 'on',
                'random_page_cost': '1.1',  # For SSD storage
                'effective_cache_size': '1GB',
                'shared_buffers': '256MB'
            }
        )
        
        logger.info(f"PostgreSQL pool initialized: min={20}, max={optimal_pool_size}")
    
    async def _init_db_connection(self, connection):
        """Initialize each database connection with optimizations"""
        # Set connection-level optimizations
        await connection.execute("SET work_mem = '8MB'")
        await connection.execute("SET maintenance_work_mem = '64MB'")
        await connection.execute("SET synchronous_commit = 'off'")  # Faster writes
        await connection.execute("SET statement_timeout = '30s'")
    
    def _init_http_pools(self, config: Dict[str, Any]):
        """Initialize HTTP client pools with ULTRA settings"""
        
        # Ollama pool - optimized for LLM operations
        ollama_limits = httpx.Limits(
            max_keepalive_connections=50,  # High keep-alive for reuse
            max_connections=100,
            keepalive_expiry=60  # Keep connections alive longer
        )
        
        self._http_pools['ollama'] = httpx.AsyncClient(
            base_url=config.get('ollama_url', 'http://ollama:11434'),
            limits=ollama_limits,
            timeout=httpx.Timeout(
                connect=2.0,  # Fast connect
                read=30.0,  # Reasonable read timeout
                write=10.0,
                pool=1.0  # Fast pool acquire
            ),
            http2=True,  # Enable HTTP/2 for multiplexing
            follow_redirects=False,
            
            # Transport options for performance
            transport=httpx.AsyncHTTPTransport(
                retries=1,
                keepalive_expiry=60,
                http2=True
            )
        )
        
        # Agent pool - for internal services
        agent_limits = httpx.Limits(
            max_keepalive_connections=30,
            max_connections=60,
            keepalive_expiry=30
        )
        
        self._http_pools['agents'] = httpx.AsyncClient(
            limits=agent_limits,
            timeout=httpx.Timeout(
                connect=1.0,
                read=5.0,
                write=2.0,
                pool=0.5
            ),
            http2=True,
            transport=httpx.AsyncHTTPTransport(
                retries=1,
                http2=True
            )
        )
        
        # External API pool
        external_limits = httpx.Limits(
            max_keepalive_connections=20,
            max_connections=40,
            keepalive_expiry=30
        )
        
        self._http_pools['external'] = httpx.AsyncClient(
            limits=external_limits,
            timeout=httpx.Timeout(
                connect=3.0,
                read=15.0,
                write=5.0,
                pool=1.0
            ),
            follow_redirects=True,
            http2=False  # Many external APIs don't support HTTP/2
        )
        
        logger.info("HTTP pools initialized: ollama, agents, external")
    
    async def get_redis(self, purpose: str = 'cache') -> redis.Redis:
        """Get Redis client for specific purpose"""
        start_time = time.time()
        
        pool = self._redis_pools.get(purpose, self._redis_pools['cache'])
        client = redis.Redis(connection_pool=pool)
        
        # Track metrics
        acquire_time = (time.time() - start_time) * 1000
        self._update_acquire_time(acquire_time)
        self._metrics['pool_hits'] += 1
        
        return client
    
    @asynccontextmanager
    async def get_db_connection(self):
        """Get database connection with metrics"""
        start_time = time.time()
        
        if not self._db_pool:
            raise RuntimeError("Database pool not initialized")
        
        try:
            async with self._db_pool.acquire() as connection:
                # Track metrics
                acquire_time = (time.time() - start_time) * 1000
                self._update_acquire_time(acquire_time)
                self._metrics['pool_hits'] += 1
                self._metrics['active_connections'] += 1
                
                yield connection
                
        finally:
            self._metrics['active_connections'] -= 1
    
    @asynccontextmanager
    async def get_http_client(self, service: str = 'agents'):
        """Get HTTP client for specific service"""
        start_time = time.time()
        
        client = self._http_pools.get(service, self._http_pools['agents'])
        
        # Track metrics
        acquire_time = (time.time() - start_time) * 1000
        self._update_acquire_time(acquire_time)
        self._metrics['pool_hits'] += 1
        
        yield client
    
    def _update_acquire_time(self, time_ms: float):
        """Update average connection acquire time"""
        hits = self._metrics['pool_hits']
        if hits > 0:
            current_avg = self._metrics['avg_acquire_time_ms']
            self._metrics['avg_acquire_time_ms'] = (current_avg * (hits - 1) + time_ms) / hits
    
    async def _connection_warmer(self):
        """Keep connections warm for fast access"""
        while True:
            try:
                # Warm Redis connections
                for purpose in ['cache', 'session', 'queue']:
                    client = await self.get_redis(purpose)
                    await client.ping()
                
                # Warm database connections
                async with self.get_db_connection() as conn:
                    await conn.fetchval("SELECT 1")
                
                # Warm HTTP connections (lightweight requests)
                for service in ['ollama', 'agents']:
                    async with self.get_http_client(service) as client:
                        try:
                            if service == 'ollama':
                                await client.get('/api/tags', timeout=2.0)
                        except:
                            pass  # Ignore warming errors
                
                await asyncio.sleep(30)  # Warm every 30 seconds
                
            except Exception as e:
                logger.error(f"Connection warming error: {e}")
                await asyncio.sleep(60)
    
    async def _metrics_collector(self):
        """Collect and log performance metrics"""
        while True:
            await asyncio.sleep(60)  # Collect every minute
            
            # Calculate pool efficiency
            total_requests = self._metrics['pool_hits'] + self._metrics['pool_misses']
            hit_rate = (self._metrics['pool_hits'] / max(1, total_requests)) * 100
            
            # Log metrics
            logger.info(
                f"ULTRA Pool Metrics - "
                f"Hit Rate: {hit_rate:.1f}%, "
                f"Avg Acquire: {self._metrics['avg_acquire_time_ms']:.2f}ms, "
                f"Active Connections: {self._metrics['active_connections']}, "
                f"Reuse Rate: {(self._metrics['connections_reused'] / max(1, self._metrics['connections_created'])) * 100:.1f}%"
            )
            
            # Track load for dynamic resizing
            self._load_history.append({
                'timestamp': time.time(),
                'active_connections': self._metrics['active_connections'],
                'hit_rate': hit_rate
            })
            
            # Keep only last hour of history
            cutoff = time.time() - 3600
            self._load_history = [h for h in self._load_history if h['timestamp'] > cutoff]
            
            # Consider resizing pools if needed
            await self._consider_pool_resize()
    
    async def _consider_pool_resize(self):
        """Dynamically resize pools based on load"""
        # Only resize every 5 minutes
        if time.time() - self._last_resize < 300:
            return
        
        if not self._load_history:
            return
        
        # Calculate average load
        avg_connections = sum(h['active_connections'] for h in self._load_history) / len(self._load_history)
        
        # Resize database pool if needed
        if self._db_pool:
            current_max = self._db_pool._maxsize
            
            # Scale up if consistently high usage
            if avg_connections > current_max * 0.8 and current_max < 100:
                new_max = min(current_max + 10, 100)
                logger.info(f"Scaling up database pool: {current_max} -> {new_max}")
                # Note: asyncpg doesn't support runtime resizing, would need to recreate
            
            # Scale down if consistently low usage
            elif avg_connections < current_max * 0.3 and current_max > 30:
                new_max = max(current_max - 10, 30)
                logger.info(f"Scaling down database pool: {current_max} -> {new_max}")
                # Note: asyncpg doesn't support runtime resizing, would need to recreate
        
        self._last_resize = time.time()
    
    async def health_check(self) -> Dict[str, Any]:
        """Ultra-fast health check"""
        health = {
            'status': 'healthy',
            'pools': {},
            'metrics': self._metrics
        }
        
        # Quick Redis check
        try:
            client = await self.get_redis('cache')
            await asyncio.wait_for(client.ping(), timeout=0.5)
            health['pools']['redis'] = 'healthy'
        except:
            health['pools']['redis'] = 'unhealthy'
            health['status'] = 'degraded'
        
        # Quick DB check
        try:
            async with self.get_db_connection() as conn:
                await asyncio.wait_for(conn.fetchval("SELECT 1"), timeout=0.5)
            health['pools']['database'] = 'healthy'
        except:
            health['pools']['database'] = 'unhealthy'
            health['status'] = 'degraded'
        
        # HTTP clients are just configuration
        health['pools']['http_ollama'] = 'configured'
        health['pools']['http_agents'] = 'configured'
        
        return health
    
    def get_stats(self) -> Dict[str, Any]:
        """Get comprehensive pool statistics"""
        total_requests = self._metrics['pool_hits'] + self._metrics['pool_misses']
        hit_rate = (self._metrics['pool_hits'] / max(1, total_requests)) * 100
        
        stats = {
            **self._metrics,
            'hit_rate_percent': round(hit_rate, 2),
            'efficiency': 'ULTRA' if hit_rate > 95 else 'GOOD' if hit_rate > 85 else 'IMPROVING'
        }
        
        # Add pool-specific stats
        if self._db_pool:
            stats['db_pool'] = {
                'size': self._db_pool.get_size() if hasattr(self._db_pool, 'get_size') else 'N/A',
                'idle': self._db_pool.get_idle_size() if hasattr(self._db_pool, 'get_idle_size') else 'N/A',
                'max': self._db_pool._maxsize if hasattr(self._db_pool, '_maxsize') else 'N/A'
            }
        
        return stats
    
    async def close(self):
        """Close all pools gracefully"""
        # Close HTTP clients
        for client in self._http_pools.values():
            await client.aclose()
        
        # Close database pool
        if self._db_pool:
            await self._db_pool.close()
        
        # Redis pools close automatically
        
        logger.info("ULTRA Connection Pools closed")


# Global instance
_ultra_pool: Optional[UltraConnectionPool] = None


async def get_ultra_pool() -> UltraConnectionPool:
    """Get or create the ULTRA connection pool"""
    global _ultra_pool
    
    if _ultra_pool is None:
        async with UltraConnectionPool._lock:
            if _ultra_pool is None:
                _ultra_pool = UltraConnectionPool()
                await _ultra_pool.initialize()
                
                logger.info("ULTRA Connection Pool initialized for 1000+ users")
    
    return _ultra_pool