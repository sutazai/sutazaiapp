"""
Database Management for SutazAI
===============================

Async database operations with connection pooling and health monitoring.
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List
from contextlib import asynccontextmanager

import asyncpg
import redis.asyncio as redis
try:
    from .config import Settings, get_database_config, get_redis_config
    from .utils import setup_logging, retry_async
except ImportError:
    from config import Settings, get_database_config, get_redis_config
    from utils import setup_logging, retry_async

logger = setup_logging(__name__)


class DatabaseManager:
    """Manages database connections and operations"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.pg_pool: Optional[asyncpg.Pool] = None
        self.redis_client: Optional[redis.Redis] = None
        
    async def initialize(self):
        """Initialize database connections"""
        logger.info("Initializing database connections...")
        
        try:
            # Initialize PostgreSQL connection pool
            await self.init_postgres()
            
            # Initialize Redis connection
            await self.init_redis()
            
            logger.info("✅ Database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize database: {e}")
            raise
    
    async def init_postgres(self):
        """Initialize PostgreSQL connection pool"""
        try:
            db_config = get_database_config()
            
            self.pg_pool = await asyncpg.create_pool(
                dsn=db_config["url"],
                min_size=5,
                max_size=20,
                command_timeout=30,
                server_settings={
                    'application_name': 'sutazai_backend',
                    'timezone': 'UTC'
                }
            )
            
            # Test connection
            async with self.pg_pool.acquire() as conn:
                await conn.execute("SELECT 1")
            
            logger.info("✅ PostgreSQL connection pool created")
            
        except Exception as e:
            logger.error(f"❌ Failed to create PostgreSQL pool: {e}")
            raise
    
    async def init_redis(self):
        """Initialize Redis connection"""
        try:
            redis_config = get_redis_config()
            
            self.redis_client = redis.from_url(
                redis_config["url"],
                decode_responses=redis_config["decode_responses"],
                socket_timeout=redis_config["socket_timeout"],
                socket_connect_timeout=redis_config["socket_connect_timeout"]
            )
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("✅ Redis connection established")
            
        except Exception as e:
            logger.error(f"❌ Failed to connect to Redis: {e}")
            raise
    
    async def close(self):
        """Close database connections"""
        logger.info("Closing database connections...")
        
        try:
            if self.pg_pool:
                await self.pg_pool.close()
                logger.info("✅ PostgreSQL pool closed")
            
            if self.redis_client:
                await self.redis_client.close()
                logger.info("✅ Redis connection closed")
                
        except Exception as e:
            logger.error(f"❌ Error closing database connections: {e}")
    
    @asynccontextmanager
    async def get_postgres_connection(self):
        """Get PostgreSQL connection from pool"""
        if not self.pg_pool:
            raise Exception("PostgreSQL pool not initialized")
        
        async with self.pg_pool.acquire() as conn:
            yield conn
    
    async def get_redis_client(self) -> redis.Redis:
        """Get Redis client"""
        if not self.redis_client:
            raise Exception("Redis client not initialized")
        return self.redis_client
    
    @retry_async(max_retries=3)
    async def execute_query(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute PostgreSQL query and return results"""
        async with self.get_postgres_connection() as conn:
            rows = await conn.fetch(query, *args)
            return [dict(row) for row in rows]
    
    @retry_async(max_retries=3)
    async def execute_command(self, command: str, *args) -> str:
        """Execute PostgreSQL command and return status"""
        async with self.get_postgres_connection() as conn:
            return await conn.execute(command, *args)
    
    async def cache_set(self, key: str, value: str, expire: int = 3600):
        """Set cache value in Redis"""
        redis_client = await self.get_redis_client()
        await redis_client.setex(key, expire, value)
    
    async def cache_get(self, key: str) -> Optional[str]:
        """Get cache value from Redis"""
        redis_client = await self.get_redis_client()
        return await redis_client.get(key)
    
    async def cache_delete(self, key: str):
        """Delete cache value from Redis"""
        redis_client = await self.get_redis_client()
        await redis_client.delete(key)
    
    async def health_check(self) -> Dict[str, Any]:
        """Check health of database connections"""
        health = {
            "postgres": {"status": "unknown", "response_time": 0},
            "redis": {"status": "unknown", "response_time": 0}
        }
        
        # Check PostgreSQL
        try:
            import time
            start_time = time.time()
            
            async with self.get_postgres_connection() as conn:
                await conn.execute("SELECT 1")
            
            health["postgres"] = {
                "status": "healthy",
                "response_time": time.time() - start_time
            }
        except Exception as e:
            health["postgres"] = {
                "status": "unhealthy",
                "error": str(e),
                "response_time": 0
            }
        
        # Check Redis
        try:
            start_time = time.time()
            redis_client = await self.get_redis_client()
            await redis_client.ping()
            
            health["redis"] = {
                "status": "healthy",
                "response_time": time.time() - start_time
            }
        except Exception as e:
            health["redis"] = {
                "status": "unhealthy",
                "error": str(e),
                "response_time": 0
            }
        
        return health