"""
Database connection management with async SQLAlchemy
Following best practices for connection pooling and async operations
"""

from sqlalchemy.ext.asyncio import (
    AsyncSession,
    create_async_engine,
    async_sessionmaker,
    AsyncEngine
)
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from typing import AsyncGenerator, Dict, Any
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

# Initialize Prometheus metrics at module level to avoid duplication
try:
    from prometheus_client import Gauge
    DB_POOL_SIZE = Gauge('db_pool_size', 'Database connection pool size')
    DB_POOL_CHECKED_IN = Gauge('db_pool_checked_in', 'Database connections checked in')
    DB_POOL_CHECKED_OUT = Gauge('db_pool_checked_out', 'Database connections checked out')
    DB_POOL_OVERFLOW = Gauge('db_pool_overflow', 'Database pool overflow connections')
    METRICS_AVAILABLE = True
except ImportError:
    METRICS_AVAILABLE = False

# Create async engine with connection pooling
# Production-ready configuration with timeouts and health checks
engine: AsyncEngine = create_async_engine(
    settings.DATABASE_URL,
    echo=settings.DEBUG,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_timeout=settings.DB_POOL_TIMEOUT,
    pool_recycle=settings.DB_POOL_RECYCLE,
    pool_pre_ping=True,  # Verify connections before using
    connect_args={
        "timeout": 30,  # Connection timeout
        "command_timeout": 60,  # Query execution timeout
        "server_settings": {
            "application_name": "sutazai_backend",
            "jit": "off"  # Disable JIT for more predictable performance
        }
    },
    future=True
)

# Create async session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,  # Prevent session expiration after commit
    autocommit=False,
    autoflush=False
)

# Create base class for models
Base = declarative_base()


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """
    Dependency for getting database session
    Yields an async session and ensures proper cleanup
    Production-ready with comprehensive error handling and logging
    """
    session = None
    try:
        session = async_session_maker()
        yield session
        await session.commit()
    except Exception as e:
        if session:
            await session.rollback()
        logger.error(f"Database session error: {e}", exc_info=True, extra={
            "error_type": type(e).__name__,
            "session_active": session is not None
        })
        raise
    finally:
        if session:
            await session.close()


async def init_db() -> None:
    """Initialize database tables"""
    try:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
            logger.info("Database tables initialized")
    except Exception as e:
        logger.warning(f"Database initialization skipped: {e}")


async def get_pool_status() -> Dict[str, Any]:
    """Get current connection pool status for monitoring"""
    pool = engine.pool
    status = {
        "size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "total_connections": pool.size() + pool.overflow(),
        "available": pool.checkedin()
    }
    
    # Update Prometheus metrics if available
    if METRICS_AVAILABLE:
        DB_POOL_SIZE.set(status["size"])
        DB_POOL_CHECKED_IN.set(status["checked_in"])
        DB_POOL_CHECKED_OUT.set(status["checked_out"])
        DB_POOL_OVERFLOW.set(status["overflow"])
    
    return status


async def close_db() -> None:
    """Close database connections gracefully"""
    try:
        logger.info("Closing database connection pool...")
        pool_status = await get_pool_status()
        logger.info(f"Final pool status: {pool_status}")
        await engine.dispose()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.error(f"Error during DB close: {e}", exc_info=True)
