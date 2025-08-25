"""
Secure Database Configuration with Proper Connection Pooling
Production-ready setup with environment-based credentials and connection optimization
"""
from sqlalchemy import create_engine, event
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, scoped_session
from sqlalchemy.pool import QueuePool
import os
import logging
from urllib.parse import quote_plus

logger = logging.getLogger(__name__)

# Secure credential handling from environment variables
def get_database_url() -> str:
    """Build database URL from environment variables with security best practices"""
    
    # Required environment variables for security
    db_user = os.getenv("POSTGRES_USER", "sutazai")
    db_password = os.getenv("POSTGRES_PASSWORD")
    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "10000")
    db_name = os.getenv("POSTGRES_DB", "sutazai")
    
    # Fallback for development only - log warning
    if not db_password:
        logger.warning("POSTGRES_PASSWORD not set - using fallback for development only")
        db_password = "sutazai123"
    
    # URL-encode password to handle special characters securely
    encoded_password = quote_plus(db_password)
    
    return f"postgresql://{db_user}:{encoded_password}@{db_host}:{db_port}/{db_name}"

DATABASE_URL = get_database_url()

# Production-ready connection pooling with QueuePool
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,           # Proper connection pooling for production
    pool_size=20,                  # Base pool size for high concurrency
    max_overflow=30,               # Allow up to 50 total connections (20+30)
    pool_pre_ping=True,            # Test connections before using
    pool_recycle=3600,             # Recycle connections every hour
    pool_timeout=30,               # Timeout waiting for connection
    echo=False,                    # Disable SQL logging for performance
    connect_args={
        "connect_timeout": 10,
        "server_settings": {
            "statement_timeout": "30000",      # 30 second statement timeout
            "lock_timeout": "10000",           # 10 second lock timeout
            "idle_in_transaction_session_timeout": "300000"  # 5 minute idle timeout
        }
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
    """Get database session with proper cleanup and error handling"""
    db = None
    try:
        db = SessionLocal()
        yield db
    except Exception as e:
        if db:
            db.rollback()
        logger.error(f"Database session error: {e}")
        raise
    finally:
        if db:
            db.close()

def init_db():
    """Initialize database with tables"""
    Base.metadata.create_all(bind=engine)

def close_db():
    """Close all database connections and cleanup pool"""
    try:
        SessionLocal.remove()
        engine.dispose()
        logger.info("Database connections closed successfully")
    except Exception as e:
        logger.error(f"Error closing database connections: {e}")

# Connection pool monitoring
@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    """Set connection-level settings for PostgreSQL"""
    # This event is for PostgreSQL-specific optimizations
    pass

@event.listens_for(engine, "checkout")
def receive_checkout(dbapi_connection, connection_record, connection_proxy):
    """Log connection checkout for monitoring"""
    logger.debug("Database connection checked out from pool")

@event.listens_for(engine, "checkin") 
def receive_checkin(dbapi_connection, connection_record):
    """Log connection checkin for monitoring"""
    logger.debug("Database connection returned to pool")

def get_pool_status() -> dict:
    """Get current connection pool status for monitoring"""
    pool = engine.pool
    return {
        "pool_size": pool.size(),
        "checked_in": pool.checkedin(),
        "checked_out": pool.checkedout(),
        "overflow": pool.overflow(),
        "invalid": pool.invalidated(),
    }
