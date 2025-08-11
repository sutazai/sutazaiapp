"""
ULTRAPERFORMANCE: Database Query Optimization and Performance Enhancement
Implements query optimization, connection pooling, and caching strategies
"""

import asyncio
import time
import logging
from typing import Any, Dict, List, Optional, Callable
from functools import wraps
import asyncpg
import psutil
from sqlalchemy import create_engine, text, Index
from sqlalchemy.pool import QueuePool
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class DatabaseOptimizer:
    """
    Database performance optimizer with:
    - Connection pooling
    - Query optimization
    - Automatic indexing
    - Query result caching
    - Batch operations
    """
    
    def __init__(self, database_url: str):
        self.database_url = database_url
        
        # Async connection pool
        self.async_pool: Optional[asyncpg.Pool] = None
        
        # Sync connection pool for migrations
        self.sync_engine = create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=20,
            max_overflow=40,
            pool_pre_ping=True,
            pool_recycle=3600
        )
        
        # Query performance tracking
        self.query_stats = {}
        self.slow_query_threshold = 0.1  # 100ms
        
    async def initialize(self):
        """Initialize async connection pool and optimize database"""
        # Create async pool
        self.async_pool = await asyncpg.create_pool(
            self.database_url,
            min_size=10,
            max_size=50,
            max_queries=1000,
            max_inactive_connection_lifetime=300,
            command_timeout=10
        )
        
        # Run initial optimizations
        await self.optimize_database()
        
        logger.info("Database optimizer initialized with connection pooling")
    
    async def optimize_database(self):
        """Apply database optimizations"""
        async with self.async_pool.acquire() as conn:
            # Enable query statistics
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")
            
            # Optimize PostgreSQL settings
            optimizations = [
                "SET random_page_cost = 1.1",  # For SSD storage
                "SET effective_cache_size = '4GB'",
                "SET shared_buffers = '1GB'",
                "SET maintenance_work_mem = '256MB'",
                "SET checkpoint_completion_target = 0.9",
                "SET wal_buffers = '16MB'",
                "SET default_statistics_target = 100",
                "SET effective_io_concurrency = 200"  # For SSD
            ]
            
            for optimization in optimizations:
                try:
                    await conn.execute(optimization)
                except Exception as e:
                    logger.warning(f"Could not apply optimization {optimization}: {e}")
            
            # Create essential indexes
            await self.create_indexes(conn)
            
            # Analyze tables for query planner
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            for table in tables:
                await conn.execute(f"ANALYZE {table['tablename']}")
    
    async def create_indexes(self, conn: asyncpg.Connection):
        """Create performance-critical indexes"""
        indexes = [
            # User-related indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email ON users(email)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_created_at ON users(created_at DESC)",
            
            # Session indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_token ON sessions(token)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at)",
            
            # Task indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_status ON tasks(status)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_created_at ON tasks(created_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_user_id_status ON tasks(user_id, status)",
            
            # Agent activity indexes
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_activities_agent_id ON agent_activities(agent_id)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_agent_activities_timestamp ON agent_activities(timestamp DESC)",
            
            # Composite indexes for common queries
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_tasks_user_status_created ON tasks(user_id, status, created_at DESC)",
            "CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_users_email_status ON users(email, status) WHERE status = 'active'"
        ]
        
        for index_sql in indexes:
            try:
                await conn.execute(index_sql)
                logger.info(f"Created index: {index_sql.split('idx_')[1].split(' ')[0]}")
            except Exception as e:
                if "already exists" not in str(e):
                    logger.error(f"Failed to create index: {e}")
    
    @asynccontextmanager
    async def acquire_connection(self):
        """Acquire a database connection from the pool"""
        async with self.async_pool.acquire() as conn:
            yield conn
    
    async def execute_query(self, query: str, *args, track_performance: bool = True) -> List[Dict]:
        """Execute a query with performance tracking"""
        start_time = time.perf_counter()
        
        async with self.acquire_connection() as conn:
            try:
                # Execute query
                results = await conn.fetch(query, *args)
                
                # Track performance
                if track_performance:
                    elapsed = time.perf_counter() - start_time
                    self._track_query_performance(query, elapsed)
                    
                    if elapsed > self.slow_query_threshold:
                        logger.warning(f"Slow query detected ({elapsed:.3f}s): {query[:100]}")
                
                # Convert to dict
                return [dict(record) for record in results]
                
            except Exception as e:
                logger.error(f"Query execution error: {e}")
                raise
    
    async def execute_many(self, query: str, args_list: List[tuple]) -> int:
        """Execute multiple queries in a batch for better performance"""
        async with self.acquire_connection() as conn:
            # Use prepared statement for better performance
            stmt = await conn.prepare(query)
            
            # Execute in batch
            async with conn.transaction():
                count = 0
                for args in args_list:
                    await stmt.fetch(*args)
                    count += 1
                
                return count
    
    async def bulk_insert(self, table: str, records: List[Dict]) -> int:
        """Perform bulk insert using COPY for maximum performance"""
        if not records:
            return 0
        
        # Get column names from first record
        columns = list(records[0].keys())
        
        async with self.acquire_connection() as conn:
            # Use COPY for bulk insert (fastest method)
            result = await conn.copy_records_to_table(
                table,
                records=[(r[col] for col in columns) for r in records],
                columns=columns
            )
            
            return len(records)
    
    def _track_query_performance(self, query: str, elapsed: float):
        """Track query performance statistics"""
        # Normalize query for grouping
        normalized = self._normalize_query(query)
        
        if normalized not in self.query_stats:
            self.query_stats[normalized] = {
                "count": 0,
                "total_time": 0,
                "min_time": float('inf'),
                "max_time": 0,
                "avg_time": 0
            }
        
        stats = self.query_stats[normalized]
        stats["count"] += 1
        stats["total_time"] += elapsed
        stats["min_time"] = min(stats["min_time"], elapsed)
        stats["max_time"] = max(stats["max_time"], elapsed)
        stats["avg_time"] = stats["total_time"] / stats["count"]
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for statistics grouping"""
        # Remove values and whitespace for grouping similar queries
        import re
        normalized = re.sub(r'\s+', ' ', query)
        normalized = re.sub(r"'[^']*'", '?', normalized)
        normalized = re.sub(r'\d+', '?', normalized)
        return normalized[:100]  # Truncate for storage
    
    async def get_slow_queries(self) -> List[Dict]:
        """Get slow queries from pg_stat_statements"""
        async with self.acquire_connection() as conn:
            slow_queries = await conn.fetch("""
                SELECT 
                    query,
                    calls,
                    total_exec_time,
                    mean_exec_time,
                    stddev_exec_time,
                    min_exec_time,
                    max_exec_time
                FROM pg_stat_statements
                WHERE mean_exec_time > 100  -- Queries slower than 100ms
                ORDER BY mean_exec_time DESC
                LIMIT 20
            """)
            
            return [dict(q) for q in slow_queries]
    
    async def analyze_query_plan(self, query: str, *args) -> Dict:
        """Analyze query execution plan"""
        async with self.acquire_connection() as conn:
            # Get query plan
            plan = await conn.fetch(f"EXPLAIN (ANALYZE, BUFFERS, FORMAT JSON) {query}", *args)
            
            return {
                "query": query,
                "plan": plan[0]["QUERY PLAN"][0],
                "recommendations": self._analyze_plan(plan[0]["QUERY PLAN"][0])
            }
    
    def _analyze_plan(self, plan: Dict) -> List[str]:
        """Analyze query plan and provide recommendations"""
        recommendations = []
        
        # Check for sequential scans on large tables
        if "Seq Scan" in str(plan):
            recommendations.append("Consider adding an index to avoid sequential scan")
        
        # Check for high cost
        if plan.get("Total Cost", 0) > 1000:
            recommendations.append("Query has high cost - consider optimization")
        
        # Check for missing indexes
        if "Filter" in str(plan) and "Index" not in str(plan):
            recommendations.append("Filter without index detected - consider adding index")
        
        return recommendations
    
    async def vacuum_analyze(self):
        """Run VACUUM ANALYZE for maintenance"""
        async with self.acquire_connection() as conn:
            tables = await conn.fetch(
                "SELECT tablename FROM pg_tables WHERE schemaname = 'public'"
            )
            
            for table in tables:
                table_name = table["tablename"]
                await conn.execute(f"VACUUM ANALYZE {table_name}")
                logger.info(f"Vacuum analyzed table: {table_name}")
    
    def get_performance_stats(self) -> Dict:
        """Get database performance statistics"""
        return {
            "pool_stats": {
                "size": self.async_pool.get_size() if self.async_pool else 0,
                "free_connections": self.async_pool.get_idle_size() if self.async_pool else 0,
                "used_connections": self.async_pool.get_size() - self.async_pool.get_idle_size() if self.async_pool else 0
            },
            "query_stats": {
                query: {
                    "count": stats["count"],
                    "avg_time_ms": stats["avg_time"] * 1000,
                    "max_time_ms": stats["max_time"] * 1000
                }
                for query, stats in sorted(
                    self.query_stats.items(),
                    key=lambda x: x[1]["avg_time"],
                    reverse=True
                )[:10]  # Top 10 slowest queries
            }
        }
    
    async def cleanup(self):
        """Cleanup database connections"""
        if self.async_pool:
            await self.async_pool.close()


# Query result caching decorator
def cached_query(ttl: int = 300, key_prefix: str = "query"):
    """Decorator for caching database query results"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            # Import here to avoid circular dependency
            from app.core.cache import get_cache_service
            
            # Generate cache key
            import hashlib
            import json
            
            key_data = json.dumps({
                "func": func.__name__,
                "args": args,
                "kwargs": kwargs
            }, sort_keys=True, default=str)
            
            cache_key = f"{key_prefix}:{hashlib.sha256(key_data.encode()).hexdigest()[:16]}"
            
            # Check cache
            cache = await get_cache_service()
            cached_result = await cache.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Execute query
            result = await func(*args, **kwargs)
            
            # Cache result
            await cache.set(cache_key, result, ttl=ttl)
            
            return result
        
        return wrapper
    return decorator


# Global optimizer instance
_db_optimizer: Optional[DatabaseOptimizer] = None


async def get_database_optimizer() -> DatabaseOptimizer:
    """Get the database optimizer instance"""
    global _db_optimizer
    
    if _db_optimizer is None:
        database_url = "postgresql://sutazai:sutazai123@sutazai-postgres:5432/sutazai"
        _db_optimizer = DatabaseOptimizer(database_url)
        await _db_optimizer.initialize()
    
    return _db_optimizer