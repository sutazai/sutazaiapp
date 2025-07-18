#!/usr/bin/env python3
"""
SutazAI Database Manager
Handles PostgreSQL, Redis, and MongoDB connections
"""

import asyncio
import logging
from typing import Optional, Dict, Any, List, Generator
from contextlib import asynccontextmanager

import asyncpg
import aioredis
from motor.motor_asyncio import AsyncIOMotorClient
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool
from sqlmodel import create_engine, Session, SQLModel

from .config import settings

logger = logging.getLogger(__name__)

# SQLAlchemy Base
Base = declarative_base()

# Legacy sync database for compatibility
DATABASE_URL = settings.DATABASE_URL
engine = create_engine(DATABASE_URL, echo=settings.SQLALCHEMY_ECHO)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=Session)

def get_db() -> Generator[Session, None, None]:
    """Dependency to get DB session (legacy sync version)."""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
        db.rollback()
    finally:
        db.close()

def init_db():
    """Initialize database tables."""
    try:
        logger.info("Initializing database...")
        SQLModel.metadata.create_all(engine)
        logger.info("Database initialized successfully.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        raise


class DatabaseManager:
    """Manages all database connections"""
    
    def __init__(self):
        self.postgres_engine = None
        self.postgres_session_factory = None
        self.redis_client = None
        self.mongodb_client = None
        self.mongodb_db = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize all database connections"""
        if self._initialized:
            return
        
        try:
            logger.info("Initializing database connections...")
            
            # Initialize PostgreSQL
            await self._initialize_postgres()
            
            # Initialize Redis
            await self._initialize_redis()
            
            # Initialize MongoDB
            await self._initialize_mongodb()
            
            self._initialized = True
            logger.info("All database connections initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database connections: {e}")
            raise
    
    async def _initialize_postgres(self):
        """Initialize PostgreSQL connection"""
        try:
            # Create async engine
            self.postgres_engine = create_async_engine(
                settings.DATABASE_URL,
                echo=settings.SQLALCHEMY_ECHO,
                pool_size=20,
                max_overflow=30,
                pool_timeout=30,
                pool_recycle=3600,
                poolclass=StaticPool,
            )
            
            # Create session factory
            self.postgres_session_factory = async_sessionmaker(
                self.postgres_engine,
                class_=AsyncSession,
                expire_on_commit=False
            )
            
            # Test connection
            async with self.postgres_engine.begin() as conn:
                await conn.execute("SELECT 1")
            
            logger.info("PostgreSQL connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize PostgreSQL: {e}")
            raise
    
    async def _initialize_redis(self):
        """Initialize Redis connection"""
        try:
            self.redis_client = aioredis.from_url(
                settings.REDIS_URL,
                encoding="utf-8",
                decode_responses=True,
                max_connections=100,
                retry_on_timeout=True,
                socket_timeout=5,
                socket_connect_timeout=5,
                health_check_interval=30,
            )
            
            # Test connection
            await self.redis_client.ping()
            
            logger.info("Redis connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            raise
    
    async def _initialize_mongodb(self):
        """Initialize MongoDB connection"""
        try:
            self.mongodb_client = AsyncIOMotorClient(
                settings.MONGODB_URL,
                maxPoolSize=50,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                serverSelectionTimeoutMS=5000,
                socketTimeoutMS=20000,
                connectTimeoutMS=10000,
            )
            
            # Get database
            db_name = settings.MONGODB_URL.split("/")[-1].split("?")[0]
            self.mongodb_db = self.mongodb_client[db_name]
            
            # Test connection
            await self.mongodb_client.admin.command('ping')
            
            logger.info("MongoDB connection established")
            
        except Exception as e:
            logger.error(f"Failed to initialize MongoDB: {e}")
            raise
    
    async def shutdown(self):
        """Shutdown all database connections"""
        try:
            logger.info("Shutting down database connections...")
            
            if self.postgres_engine:
                await self.postgres_engine.dispose()
                logger.info("PostgreSQL connection closed")
            
            if self.redis_client:
                await self.redis_client.close()
                logger.info("Redis connection closed")
            
            if self.mongodb_client:
                self.mongodb_client.close()
                logger.info("MongoDB connection closed")
            
            self._initialized = False
            logger.info("All database connections closed")
            
        except Exception as e:
            logger.error(f"Error during database shutdown: {e}")
    
    @asynccontextmanager
    async def get_postgres_session(self):
        """Get PostgreSQL session"""
        if not self._initialized:
            await self.initialize()
        
        async with self.postgres_session_factory() as session:
            try:
                yield session
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    
    async def get_redis_client(self):
        """Get Redis client"""
        if not self._initialized:
            await self.initialize()
        return self.redis_client
    
    async def get_mongodb_database(self):
        """Get MongoDB database"""
        if not self._initialized:
            await self.initialize()
        return self.mongodb_db
    
    async def get_mongodb_collection(self, collection_name: str):
        """Get MongoDB collection"""
        db = await self.get_mongodb_database()
        return db[collection_name]
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of all database connections"""
        health = {
            "postgres": False,
            "redis": False,
            "mongodb": False
        }
        
        try:
            if self.postgres_engine:
                async with self.postgres_engine.begin() as conn:
                    await conn.execute("SELECT 1")
                health["postgres"] = True
        except Exception as e:
            logger.warning(f"PostgreSQL health check failed: {e}")
        
        try:
            if self.redis_client:
                await self.redis_client.ping()
                health["redis"] = True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
        
        try:
            if self.mongodb_client:
                await self.mongodb_client.admin.command('ping')
                health["mongodb"] = True
        except Exception as e:
            logger.warning(f"MongoDB health check failed: {e}")
        
        return health 