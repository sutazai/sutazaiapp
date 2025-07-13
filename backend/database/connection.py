#!/usr/bin/env python3
"""
Database Connection Management for SutazAI
Handles SQLAlchemy engine creation and session management
"""

import os
from typing import Generator
from sqlalchemy import create_engine, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import StaticPool
from backend.config import Config
from loguru import logger

# Global variables
engine: Engine = None
SessionLocal: sessionmaker = None


def get_database_url() -> str:
    """Get database URL from configuration"""
    config = Config()
    
    # Check for environment variable first
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        return database_url
    
    # Use configuration
    db_config = getattr(config, 'database', None)
    if not db_config:
        # Default to SQLite if no database config
        return "sqlite:///data/sutazai.db"
    
    if db_config.get('type') == "postgresql":
        return (
            f"postgresql://{db_config.get('username', 'sutazai')}:{db_config.get('password', '')}"
            f"@{db_config.get('host', 'localhost')}:{db_config.get('port', 5432)}/{db_config.get('name', 'sutazai')}"
        )
    elif db_config.get('type') == "sqlite":
        db_path = db_config.get('path', 'data/sutazai.db')
        return f"sqlite:///{db_path}"
    else:
        # Default to SQLite
        return "sqlite:///data/sutazai.db"


def create_database_engine() -> Engine:
    """Create and configure database engine"""
    database_url = get_database_url()
    logger.info(f"Connecting to database: {database_url.split('@')[-1] if '@' in database_url else database_url}")
    
    # Engine configuration based on database type
    if database_url.startswith("sqlite"):
        from sqlalchemy import create_engine as sa_create_engine
        engine = sa_create_engine(
            database_url,
            poolclass=StaticPool,
            connect_args={
                "check_same_thread": False,
                "timeout": 30
            },
            echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
        )
    else:
        # PostgreSQL configuration
        from sqlalchemy import create_engine as sa_create_engine
        engine = sa_create_engine(
            database_url,
            pool_size=10,
            max_overflow=20,
            pool_pre_ping=True,
            pool_recycle=300,
            echo=os.getenv("SQL_DEBUG", "false").lower() == "true"
        )
    
    return engine


def init_database() -> None:
    """Initialize database connection"""
    global engine, SessionLocal
    
    if engine is None:
        engine = create_database_engine()
        SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
        logger.info("Database connection initialized")


def get_session() -> Generator[Session, None, None]:
    """Dependency to get database session"""
    if SessionLocal is None:
        init_database()
    
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def create_tables() -> None:
    """Create all database tables"""
    from .models import Base
    
    if engine is None:
        init_database()
    
    Base.metadata.create_all(bind=engine)
    logger.info("Database tables created/verified")


def drop_tables() -> None:
    """Drop all database tables (use with caution)"""
    from .models import Base
    
    if engine is None:
        init_database()
    
    Base.metadata.drop_all(bind=engine)
    logger.warning("All database tables dropped")


def check_database_connection() -> bool:
    """Check if database connection is working"""
    try:
        if engine is None:
            init_database()
        
        with engine.connect() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        return False


# Database will be initialized when needed