"""
Database configuration and session management for SutazAI Backend
Provides async SQLAlchemy setup with connection pooling
"""

import os
import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.pool import StaticPool
from .secure_config import config

logger = logging.getLogger(__name__)

# Create declarative base for models
Base = declarative_base()

# Database configuration from secure config
DATABASE_URL = os.getenv("DATABASE_URL", config.database_url)

# Create async engine with connection pooling
engine = create_async_engine(
    DATABASE_URL,
    # Connection pool settings for high performance
    pool_size=20,  # Number of connections to maintain
    max_overflow=30,  # Additional connections when pool is exhausted
    pool_timeout=30,  # Seconds to wait for connection
    pool_recycle=3600,  # Recycle connections every hour
    pool_pre_ping=True,  # Validate connections before use
    # Pool class automatically handled by SQLAlchemy async engine (AsyncAdaptedQueuePool)
    # Echo SQL queries in development
    echo=os.getenv("DATABASE_ECHO", "false").lower() == "true",
    # Connection arguments
    connect_args={
        "server_settings": {
            "jit": "off",  # Disable JIT for faster connection
        }
    }
)

# Create async session maker
async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Keep objects accessible after commit
    autoflush=True,  # Auto flush changes
    autocommit=False  # Manual commit control
)


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session
    
    Yields:
        AsyncSession: Database session for use in FastAPI dependencies
        
    Usage:
        @app.get("/users")
        async def get_users(db: AsyncSession = Depends(get_db)):
            # Use db session here
    """
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


@asynccontextmanager
async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """
    Context manager for getting database session
    
    Yields:
        AsyncSession: Database session for use in async context managers
        
    Usage:
        async with get_db_session() as db:
            # Use db session here
    """
    async with async_session() as session:
        try:
            yield session
        except Exception as e:
            logger.error(f"Database session error: {e}")
            await session.rollback()
            raise
        finally:
            await session.close()


async def create_tables():
    """Create all database tables"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    logger.info("Database tables created successfully")


async def drop_tables():
    """Drop all database tables (use with caution)"""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    logger.warning("All database tables dropped")


async def check_database_connection() -> bool:
    """
    Check database connectivity
    
    Returns:
        bool: True if database is accessible, False otherwise
    """
    try:
        async with get_db_session() as db:
            # Simple query to check connection
            await db.execute("SELECT 1")
            return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


async def get_database_stats() -> dict:
    """
    Get database connection statistics
    
    Returns:
        dict: Database statistics including connection pool status
    """
    try:
        pool = engine.pool
        return {
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid(),
            "total_connections": pool.size() + pool.overflow(),
            "available_connections": pool.checkedin(),
            "active_connections": pool.checkedout()
        }
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        return {"error": str(e)}


async def init_database():
    """Initialize database with tables and initial data"""
    try:
        # Create tables if they don't exist
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        
        logger.info("Database initialization completed successfully")
        return True
        
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        return False


async def close_database():
    """Close database connections"""
    try:
        await engine.dispose()
        logger.info("Database connections closed")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")


# Database health check function
async def database_health_check() -> dict:
    """
    Comprehensive database health check
    
    Returns:
        dict: Health status and metrics
    """
    try:
        start_time = time.time()
        
        # Check basic connectivity
        is_connected = await check_database_connection()
        
        response_time = (time.time() - start_time) * 1000
        
        if is_connected:
            # Get connection pool stats
            stats = await get_database_stats()
            
            return {
                "status": "healthy",
                "response_time_ms": round(response_time, 2),
                "connected": True,
                "pool_stats": stats
            }
        else:
            return {
                "status": "unhealthy", 
                "response_time_ms": round(response_time, 2),
                "connected": False,
                "error": "Connection failed"
            }
            
    except Exception as e:
        return {
            "status": "error",
            "connected": False,
            "error": str(e)
        }


# Import time for health check
import time