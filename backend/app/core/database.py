"""
Optimized Database Configuration with Connection Pooling
"""

from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import declarative_base
from sqlalchemy.pool import NullPool, QueuePool
from sqlalchemy import event, text
import logging
from typing import AsyncGenerator
from contextlib import asynccontextmanager
from collections import defaultdict
import asyncio

from app.core.config import settings

logger = logging.getLogger(__name__)

# Database URL
DATABASE_URL = settings.DATABASE_URL
if "postgresql://" in DATABASE_URL:
    DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

# Create async engine with optimized settings
engine = create_async_engine(
    DATABASE_URL,
    echo=False,  # Disable SQL logging for performance
    pool_size=20,  # Connection pool size
    max_overflow=40,  # Maximum overflow connections
    pool_timeout=30,  # Timeout for getting connection
    pool_recycle=3600,  # Recycle connections after 1 hour
    pool_pre_ping=True,  # Verify connections before use
    connect_args={
        "server_settings": {
            "application_name": "sutazai_backend",
            "jit": "off"  # Disable JIT for more predictable performance
        },
        "command_timeout": 60,
    } if "postgresql" in DATABASE_URL else {}
)

# Session factory
async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False
)

# Base class for models
Base = declarative_base()

# Database session dependency
async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """Get database session with automatic cleanup"""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

async def init_db_pool():
    """Initialize database connection pool"""
    logger.info("Initializing database connection pool...")
    
    # Test connection
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            logger.info("Database connection established")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

async def check_db_health() -> bool:
    """Check database health"""
    try:
        async with engine.begin() as conn:
            result = await conn.execute(text("SELECT 1"))
            return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False