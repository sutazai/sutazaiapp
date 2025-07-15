"""
Optimized Database Manager for SutazAI
High-performance database operations with connection pooling
"""

import asyncio
import asyncpg
import sqlite3
import logging
import json
import time
from typing import Dict, List, Any, Optional, Union
from contextlib import asynccontextmanager
from dataclasses import dataclass
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    """Database configuration"""
    db_type: str = "sqlite"  # sqlite, postgresql
    host: str = "localhost"
    port: int = 5432
    database: str = "sutazaidb"
    username: str = "sutazai"
    password: str = ""
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30

class OptimizedDatabaseManager:
    """High-performance database manager"""
    
    def __init__(self, config: DatabaseConfig = None):
        self.config = config or DatabaseConfig()
        self.pool = None
        self.sqlite_connections = {}
        self.connection_lock = threading.Lock()
        self.query_cache = {}
        self.performance_stats = {
            "queries_executed": 0,
            "cache_hits": 0,
            "average_query_time": 0.0,
            "total_query_time": 0.0
        }
    
    async def initialize(self):
        """Initialize database manager"""
        logger.info("🔄 Initializing Optimized Database Manager")
        
        if self.config.db_type == "postgresql":
            await self._initialize_postgresql()
        elif self.config.db_type == "sqlite":
            await self._initialize_sqlite()
        
        # Create indexes for performance
        await self._create_performance_indexes()
        
        logger.info("✅ Database manager initialized")
    
    async def _initialize_postgresql(self):
        """Initialize PostgreSQL connection pool"""
        try:
            self.pool = await asyncpg.create_pool(
                host=self.config.host,
                port=self.config.port,
                user=self.config.username,
                password=self.config.password,
                database=self.config.database,
                min_size=5,
                max_size=self.config.pool_size,
                command_timeout=self.config.pool_timeout
            )
            logger.info("✅ PostgreSQL connection pool created")
        except Exception as e:
            logger.error(f"PostgreSQL initialization failed: {e}")
            # Fallback to SQLite
            self.config.db_type = "sqlite"
            await self._initialize_sqlite()
    
    async def _initialize_sqlite(self):
        """Initialize SQLite with optimizations"""
        db_path = Path("/opt/sutazaiapp/data/sutazai_optimized.db")
        db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create optimized SQLite connection
        conn = sqlite3.connect(str(db_path), check_same_thread=False)
        
        # Apply SQLite optimizations
        optimizations = [
            "PRAGMA journal_mode = WAL;",
            "PRAGMA synchronous = NORMAL;",
            "PRAGMA cache_size = 10000;",
            "PRAGMA temp_store = MEMORY;",
            "PRAGMA mmap_size = 268435456;",  # 256MB
            "PRAGMA optimize;",
        ]
        
        for optimization in optimizations:
            conn.execute(optimization)
        
        conn.commit()
        self.sqlite_connections[threading.current_thread().ident] = conn
        
        logger.info("✅ SQLite database optimized")
    
    async def _create_performance_indexes(self):
        """Create indexes for better performance"""
        indexes = [
            "CREATE INDEX IF NOT EXISTS idx_users_email ON users(email);",
            "CREATE INDEX IF NOT EXISTS idx_sessions_token ON sessions(session_token);",
            "CREATE INDEX IF NOT EXISTS idx_ai_interactions_user_id ON ai_interactions(user_id);",
            "CREATE INDEX IF NOT EXISTS idx_ai_interactions_created_at ON ai_interactions(created_at);",
            "CREATE INDEX IF NOT EXISTS idx_ai_interactions_model ON ai_interactions(model_used);",
        ]
        
        for index_sql in indexes:
            try:
                await self.execute(index_sql)
            except Exception as e:
                logger.warning(f"Index creation failed: {e}")
    
    def _get_sqlite_connection(self):
        """Get thread-local SQLite connection"""
        thread_id = threading.current_thread().ident
        
        if thread_id not in self.sqlite_connections:
            with self.connection_lock:
                if thread_id not in self.sqlite_connections:
                    db_path = Path("/opt/sutazaiapp/data/sutazai_optimized.db")
                    conn = sqlite3.connect(str(db_path), check_same_thread=False)
                    conn.row_factory = sqlite3.Row
                    self.sqlite_connections[thread_id] = conn
        
        return self.sqlite_connections[thread_id]
    
    async def execute(self, query: str, params: tuple = None) -> Any:
        """Execute database query with performance tracking"""
        start_time = time.time()
        
        try:
            # Check cache first for SELECT queries
            cache_key = f"{query}:{params}" if params else query
            if query.strip().upper().startswith("SELECT") and cache_key in self.query_cache:
                self.performance_stats["cache_hits"] += 1
                return self.query_cache[cache_key]
            
            if self.config.db_type == "postgresql" and self.pool:
                result = await self._execute_postgresql(query, params)
            else:
                result = await self._execute_sqlite(query, params)
            
            # Cache SELECT results
            if query.strip().upper().startswith("SELECT"):
                self.query_cache[cache_key] = result
                # Limit cache size
                if len(self.query_cache) > 1000:
                    # Remove oldest entries
                    for _ in range(100):
                        self.query_cache.pop(next(iter(self.query_cache)))
            
            # Update performance stats
            query_time = time.time() - start_time
            self.performance_stats["queries_executed"] += 1
            self.performance_stats["total_query_time"] += query_time
            self.performance_stats["average_query_time"] = (
                self.performance_stats["total_query_time"] / 
                self.performance_stats["queries_executed"]
            )
            
            return result
            
        except Exception as e:
            logger.error(f"Database query failed: {e}")
            raise
    
    async def _execute_postgresql(self, query: str, params: tuple = None):
        """Execute PostgreSQL query"""
        async with self.pool.acquire() as conn:
            if params:
                return await conn.fetch(query, *params)
            else:
                return await conn.fetch(query)
    
    async def _execute_sqlite(self, query: str, params: tuple = None):
        """Execute SQLite query"""
        def _execute():
            conn = self._get_sqlite_connection()
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if query.strip().upper().startswith("SELECT"):
                return cursor.fetchall()
            else:
                conn.commit()
                return cursor.rowcount
        
        # Run in thread pool for async compatibility
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _execute)
    
    async def execute_many(self, query: str, params_list: List[tuple]) -> int:
        """Execute query with multiple parameter sets"""
        if self.config.db_type == "postgresql" and self.pool:
            async with self.pool.acquire() as conn:
                return await conn.executemany(query, params_list)
        else:
            def _execute_many():
                conn = self._get_sqlite_connection()
                cursor = conn.cursor()
                cursor.executemany(query, params_list)
                conn.commit()
                return cursor.rowcount
            
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, _execute_many)
    
    @asynccontextmanager
    async def transaction(self):
        """Database transaction context manager"""
        if self.config.db_type == "postgresql" and self.pool:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    yield conn
        else:
            conn = self._get_sqlite_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                raise
    
    async def optimize_database(self):
        """Run database optimization"""
        try:
            if self.config.db_type == "sqlite":
                await self.execute("VACUUM;")
                await self.execute("ANALYZE;")
            elif self.config.db_type == "postgresql":
                await self.execute("VACUUM ANALYZE;")
            
            logger.info("✅ Database optimization completed")
        except Exception as e:
            logger.error(f"Database optimization failed: {e}")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get database performance statistics"""
        return {
            **self.performance_stats,
            "cache_size": len(self.query_cache),
            "cache_hit_rate": (
                self.performance_stats["cache_hits"] / 
                max(1, self.performance_stats["queries_executed"])
            ) * 100,
            "database_type": self.config.db_type
        }
    
    async def close(self):
        """Close database connections"""
        if self.pool:
            await self.pool.close()
        
        for conn in self.sqlite_connections.values():
            conn.close()
        
        self.sqlite_connections.clear()

# Global database manager instance
db_manager = OptimizedDatabaseManager()
