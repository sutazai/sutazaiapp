from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from .config import get_settings
import logging

# Get application settings
settings = get_settings()

# Set up logging
logger = logging.getLogger("database")

try:
    # Create database engine with appropriate pool settings based on database type
    if "postgresql" in settings.SQLALCHEMY_DATABASE_URL:
        # PostgreSQL-specific settings
        engine = create_engine(
            settings.SQLALCHEMY_DATABASE_URL,
            pool_pre_ping=True,  # Detect stale connections
            pool_recycle=3600,  # Recycle connections after 1 hour
            pool_size=10,  # Default pool size
            max_overflow=20,  # Allow up to 20 extra connections when needed
            pool_timeout=30,  # Timeout after 30 seconds waiting for connection
            connect_args={
                "connect_timeout": 10,  # Timeout after 10 seconds connecting to database
                "application_name": "SutazAI",  # Identify app in PostgreSQL logs
            },
        )
    elif "sqlite" in settings.SQLALCHEMY_DATABASE_URL:
        # SQLite-specific settings
        engine = create_engine(
            settings.SQLALCHEMY_DATABASE_URL,
            pool_pre_ping=True,  # Detect stale connections
            pool_recycle=3600,  # Recycle connections after 1 hour
            # SQLite connection pooling is less critical, but still optimized
            connect_args={"check_same_thread": False},  # Allow multi-threaded access
        )
    else:
        # Default settings for other database types
        engine = create_engine(
            settings.SQLALCHEMY_DATABASE_URL, pool_pre_ping=True, pool_recycle=3600
        )

    # Create session factory with expire_on_commit=False to prevent detached object errors
    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine,
        expire_on_commit=False,  # Prevent detached object errors with multi-process environments
    )

    Base = declarative_base()
    logger.info("Database connection established successfully")
except Exception as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    # Create fallback SQLite database for development/testing
    import os

    sqlite_path = os.path.join(
        os.path.dirname(__file__), "..", "..", "data", "fallback.db"
    )
    os.makedirs(os.path.dirname(sqlite_path), exist_ok=True)
    logger.warning(f"Using fallback SQLite database at {sqlite_path}")
    engine = create_engine(f"sqlite:///{sqlite_path}")
    SessionLocal = sessionmaker(
        autocommit=False, autoflush=False, bind=engine, expire_on_commit=False
    )
    Base = declarative_base()


# Function to dispose engine connections before forking new processes
def dispose_engine():
    """
    Dispose all connections in the engine pool.
    Call this before forking processes to prevent connection sharing issues.
    """
    try:
        engine.dispose()
        import gc

        gc.collect()  # Force garbage collection to ensure all connections are properly closed
        logger.info("Database engine connections disposed before forking")
        return True
    except Exception as e:
        logger.error(f"Error disposing database engine: {str(e)}")
        return False


# Dependency to get DB session with better error handling
def get_db():
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}")
    finally:
        db.close()


async def init_db():
    """
    Initialize the database.

    Creates all tables if they don't exist.
    """
    try:
        # Import all models to ensure they're registered with Base
        # Import order matters - import base_models first as it contains shared models

        # Then import the individual model files
        try:
            pass
        except Exception as e:
            logger.warning(f"Some model imports failed, but continuing: {str(e)}")

        # Create tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")

        # Try to execute a simple query to verify connection
        db = next(get_db())
        db.execute("SELECT 1")
        logger.info("Database connection verified")

        return True
    except Exception as e:
        logger.error(f"Error initializing database: {str(e)}")
        return False
