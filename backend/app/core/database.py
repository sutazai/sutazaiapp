"""
Optimized Database Configuration with Async-Compatible Connection Pooling
"""
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import NullPool
import os

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://sutazai:sutazai@localhost:10000/sutazai")

# Use NullPool for async compatibility (connection pooling handled at application level)
engine = create_engine(
    DATABASE_URL,
    poolclass=NullPool,     # Required for async compatibility
    pool_pre_ping=True,     # Test connections before using
    echo=False,             # Disable SQL logging for performance
    connect_args={
        "connect_timeout": 10,
        "options": "-c statement_timeout=30000"  # 30 second statement timeout
    }
)

# Thread-safe session factory
SessionLocal = scoped_session(
    sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False  # Don't expire objects after commit
    )
)

Base = declarative_base()

def get_db():
    """Get database session with proper cleanup"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def init_db():
    """Initialize database with tables"""
    Base.metadata.create_all(bind=engine)

def close_db():
    """Close all database connections"""
    SessionLocal.remove()
    engine.dispose()
