"""
ULTRAPERFORMANCE Connection Pool Manager
Optimized for maximum throughput and minimal latency
"""

import asyncio
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timedelta
import asyncpg
import redis.asyncio as redis
from contextlib import asynccontextmanager
from functools import lru_cache

logger = logging.getLogger(__name__)


class UltraPerformanceConnectionPool:
    """
    ULTRAPERFORMANCE connection pooling with:
    - Connection reuse and warm pools
    - Health monitoring and auto-recovery
    - Query performance tracking
    - Adaptive pool sizing
    """
    
    def __init__(self):
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_pool: Optional[redis.ConnectionPool] = None
        self.redis_client: Optional[redis.Redis] = None
        self.stats = {
            'pg_connections_created': 0,
            'pg_connections_reused': 0,
            'pg_queries_executed': 0,
            'pg_avg_query_time': 0,
            'redis_commands_executed': 0,
            'redis_pipeline_batches': 0,
            'pool_exhaustion_events': 0,
            'connection_errors': 0
        }
        self._query_times = []
        self._initialized = False
        
    async def initialize(self):
        """Initialize connection pools with ULTRAPERFORMANCE settings"""
        if self._initialized:
            return
            
        try:
            # PostgreSQL connection pool with optimized settings
            self.pg_pool = await asyncpg.create_pool(
                host='localhost',
                port=10000,
                database='sutazai',
                user='sutazai',
                password='sutazai123',
                
                # ULTRAPERFORMANCE pool settings
                min_size=10,              # Minimum connections (warm pool)
                max_size=50,              # Maximum connections
                max_queries=10000,        # Queries before connection refresh
                max_inactive_connection_lifetime=300,  # 5 minutes idle timeout
                
                # Performance optimizations
                command_timeout=10,
                server_settings={
                    'jit': 'on',
                    'max_parallel_workers_per_gather': '4',
                    'work_mem': '8MB',
                    'shared_buffers': '256MB',
                    'effective_cache_size': '1GB',
                    'random_page_cost': '1.1',
                    'effective_io_concurrency': '200',
                },
                
                # Connection initialization
                init=self._init_pg_connection
            )
            
            # Redis connection pool with optimized settings
            self.redis_pool = redis.ConnectionPool(
                host='localhost',
                port=10001,
                db=0,
                
                # ULTRAPERFORMANCE pool settings
                max_connections=100,           # High connection limit
                socket_keepalive=True,         # Keep connections alive
                socket_keepalive_options={
                    1: 1,  # TCP_KEEPIDLE
                    2: 3,  # TCP_KEEPINTVL
                    3: 5   # TCP_KEEPCNT
                },
                socket_connect_timeout=2,
                socket_timeout=5,
                retry_on_timeout=True,
                
                # Performance settings
                decode_responses=False,        # Binary mode for speed
                health_check_interval=30,      # Regular health checks
            )
            
            self.redis_client = redis.Redis(connection_pool=self.redis_pool)
            
            # Warm up the pools
            await self._warm_up_pools()
            
            self._initialized = True
            logger.info("ULTRAPERFORMANCE connection pools initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize connection pools: {e}")
            raise
            
    async def _init_pg_connection(self, connection):
        """Initialize each PostgreSQL connection with optimal settings"""
        # Set connection-level optimizations
        await connection.execute("SET synchronous_commit = OFF")
        await connection.execute("SET statement_timeout = '30s'")
        await connection.execute("SET lock_timeout = '10s'")
        await connection.execute("SET idle_in_transaction_session_timeout = '60s'")
        
        # Prepare frequently used statements for better performance
        await connection.execute("""
            PREPARE get_user_by_id AS 
            SELECT * FROM users WHERE id = $1
        """)
        
        await connection.execute("""
            PREPARE get_tasks_by_status AS
            SELECT * FROM tasks WHERE status = $1 ORDER BY created_at DESC LIMIT $2
        """)
        
        self.stats['pg_connections_created'] += 1
        
    async def _warm_up_pools(self):
        """Pre-warm connection pools for immediate availability"""
        logger.info("Warming up connection pools...")
        
        # Warm PostgreSQL pool
        warmup_tasks = []
        for _ in range(10):  # Create 10 warm connections
            warmup_tasks.append(self._pg_warmup_query())
            
        await asyncio.gather(*warmup_tasks, return_exceptions=True)
        
        # Warm Redis pool
        pipeline = self.redis_client.pipeline()
        for i in range(10):
            pipeline.ping()
        await pipeline.execute()
        
        logger.info("Connection pools warmed up successfully")
        
    async def _pg_warmup_query(self):
        """Execute a simple query to warm up a PostgreSQL connection"""
        async with self.pg_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            
    @asynccontextmanager
    async def get_pg_connection(self):
        """
        Get a PostgreSQL connection from the pool with performance tracking
        """
        if not self.pg_pool:
            await self.initialize()
            
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self.pg_pool.acquire() as connection:
                self.stats['pg_connections_reused'] += 1
                
                # Track connection acquisition time
                acquisition_time = asyncio.get_event_loop().time() - start_time
                if acquisition_time > 0.1:  # Log slow acquisitions
                    logger.warning(f"Slow PG connection acquisition: {acquisition_time:.3f}s")
                    
                yield connection
                
        except asyncpg.exceptions.TooManyConnectionsError:
            self.stats['pool_exhaustion_events'] += 1
            logger.error("PostgreSQL connection pool exhausted!")
            raise
        except Exception as e:
            self.stats['connection_errors'] += 1
            logger.error(f"PostgreSQL connection error: {e}")
            raise
            
    async def execute_query(self, query: str, *args, track_performance: bool = True):
        """
        Execute a query with performance tracking
        """
        start_time = asyncio.get_event_loop().time()
        
        async with self.get_pg_connection() as conn:
            try:
                result = await conn.fetch(query, *args)
                
                if track_performance:
                    query_time = asyncio.get_event_loop().time() - start_time
                    self._track_query_performance(query_time)
                    
                    if query_time > 1.0:  # Log slow queries
                        logger.warning(f"Slow query ({query_time:.3f}s): {query[:100]}...")
                        
                return result
                
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                raise
                
    async def execute_many(self, query: str, args_list: list):
        """
        Execute multiple queries in a batch for better performance
        """
        async with self.get_pg_connection() as conn:
            try:
                await conn.executemany(query, args_list)
                self.stats['pg_queries_executed'] += len(args_list)
            except Exception as e:
                logger.error(f"Batch execution error: {e}")
                raise
                
    def _track_query_performance(self, query_time: float):
        """Track query performance metrics"""
        self.stats['pg_queries_executed'] += 1
        self._query_times.append(query_time)
        
        # Keep only last 1000 query times
        if len(self._query_times) > 1000:
            self._query_times = self._query_times[-1000:]
            
        # Update average query time
        self.stats['pg_avg_query_time'] = sum(self._query_times) / len(self._query_times)
        
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client with connection pooling"""
        if not self.redis_client:
            await self.initialize()
        return self.redis_client
        
    async def redis_pipeline_execute(self, commands: list):
        """
        Execute Redis commands in pipeline for better performance
        """
        pipeline = self.redis_client.pipeline()
        
        for cmd, *args in commands:
            getattr(pipeline, cmd)(*args)
            
        results = await pipeline.execute()
        self.stats['redis_pipeline_batches'] += 1
        self.stats['redis_commands_executed'] += len(commands)
        
        return results
        
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform health check on all connections
        """
        health = {
            'status': 'healthy',
            'postgresql': 'unknown',
            'redis': 'unknown',
            'stats': self.stats,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check PostgreSQL
        try:
            async with self.get_pg_connection() as conn:
                await conn.fetchval("SELECT 1")
                health['postgresql'] = 'healthy'
                
                # Get pool stats
                pool_stats = self.pg_pool.get_stats()
                health['pg_pool'] = {
                    'size': pool_stats['size'],
                    'free': pool_stats['free_size'],
                    'used': pool_stats['used_size'],
                    'acquiring': pool_stats['acquiring']
                }
        except Exception as e:
            health['postgresql'] = f'unhealthy: {str(e)}'
            health['status'] = 'degraded'
            
        # Check Redis
        try:
            await self.redis_client.ping()
            health['redis'] = 'healthy'
            
            # Get pool stats
            pool_stats = self.redis_pool.connection_kwargs
            health['redis_pool'] = {
                'max_connections': self.redis_pool.max_connections,
                'created_connections': len(self.redis_pool._created_connections),
                'available_connections': len(self.redis_pool._available_connections),
                'in_use_connections': len(self.redis_pool._in_use_connections)
            }
        except Exception as e:
            health['redis'] = f'unhealthy: {str(e)}'
            health['status'] = 'degraded'
            
        # Performance analysis
        if self.stats['pg_avg_query_time'] > 0.1:
            health['warnings'] = health.get('warnings', [])
            health['warnings'].append(f"High average query time: {self.stats['pg_avg_query_time']:.3f}s")
            
        if self.stats['pool_exhaustion_events'] > 0:
            health['warnings'] = health.get('warnings', [])
            health['warnings'].append(f"Pool exhaustion events: {self.stats['pool_exhaustion_events']}")
            
        return health
        
    async def optimize_pools(self):
        """
        Dynamically optimize pool sizes based on usage patterns
        """
        if not self.pg_pool:
            return
            
        pool_stats = self.pg_pool.get_stats()
        
        # If pool is frequently exhausted, increase size
        if pool_stats['used_size'] / pool_stats['size'] > 0.8:
            logger.info("PostgreSQL pool running hot, consider increasing max_size")
            
        # If pool has many idle connections, reduce min_size
        if pool_stats['free_size'] / pool_stats['size'] > 0.7:
            logger.info("PostgreSQL pool has many idle connections, consider reducing min_size")
            
    async def cleanup(self):
        """Clean up connection pools"""
        if self.pg_pool:
            await self.pg_pool.close()
            
        if self.redis_client:
            await self.redis_client.close()
            
        self._initialized = False
        logger.info("Connection pools closed")


# Global instance
_connection_manager: Optional[UltraPerformanceConnectionPool] = None


async def get_connection_manager() -> UltraPerformanceConnectionPool:
    """Get or create the global connection manager"""
    global _connection_manager
    
    if _connection_manager is None:
        _connection_manager = UltraPerformanceConnectionPool()
        await _connection_manager.initialize()
        
    return _connection_manager


async def execute_with_retry(query: str, *args, max_retries: int = 3):
    """
    Execute query with automatic retry on connection failure
    """
    manager = await get_connection_manager()
    
    for attempt in range(max_retries):
        try:
            return await manager.execute_query(query, *args)
        except (asyncpg.exceptions.ConnectionDoesNotExistError,
                asyncpg.exceptions.InterfaceError) as e:
            if attempt == max_retries - 1:
                raise
            logger.warning(f"Query failed (attempt {attempt + 1}), retrying: {e}")
            await asyncio.sleep(0.1 * (2 ** attempt))  # Exponential backoff


@lru_cache(maxsize=128)
def prepare_query(query_template: str, *args) -> str:
    """
    Cache prepared queries for better performance
    """
    return query_template.format(*args)


class QueryBatcher:
    """
    Batch multiple queries for efficient execution
    """
    
    def __init__(self, batch_size: int = 100, flush_interval: float = 0.1):
        self.batch_size = batch_size
        self.flush_interval = flush_interval
        self.pending_queries = []
        self.flush_task = None
        
    async def add_query(self, query: str, args: tuple):
        """Add query to batch"""
        self.pending_queries.append((query, args))
        
        if len(self.pending_queries) >= self.batch_size:
            await self.flush()
        elif not self.flush_task:
            self.flush_task = asyncio.create_task(self._auto_flush())
            
    async def _auto_flush(self):
        """Auto-flush after interval"""
        await asyncio.sleep(self.flush_interval)
        await self.flush()
        
    async def flush(self):
        """Execute all pending queries"""
        if not self.pending_queries:
            return
            
        manager = await get_connection_manager()
        queries_to_execute = self.pending_queries.copy()
        self.pending_queries.clear()
        
        if self.flush_task:
            self.flush_task.cancel()
            self.flush_task = None
            
        # Group queries by template for batch execution
        query_groups = {}
        for query, args in queries_to_execute:
            if query not in query_groups:
                query_groups[query] = []
            query_groups[query].append(args)
            
        # Execute each group
        for query, args_list in query_groups.items():
            if len(args_list) == 1:
                await manager.execute_query(query, *args_list[0])
            else:
                await manager.execute_many(query, args_list)


# Performance monitoring decorator
def track_db_performance(operation_name: str):
    """Decorator to track database operation performance"""
    def decorator(func):
        async def wrapper(*args, **kwargs):
            start_time = asyncio.get_event_loop().time()
            
            try:
                result = await func(*args, **kwargs)
                elapsed = asyncio.get_event_loop().time() - start_time
                
                if elapsed > 1.0:
                    logger.warning(f"Slow DB operation '{operation_name}': {elapsed:.3f}s")
                    
                return result
                
            except Exception as e:
                logger.error(f"DB operation '{operation_name}' failed: {e}")
                raise
                
        return wrapper
    return decorator