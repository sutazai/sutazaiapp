from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import logging
import os
from pydantic_settings import BaseSettings, SettingsConfigDict

# Configure logging
logger = logging.getLogger(__name__)

# Create settings configuration for database connection
class DatabaseSettings(BaseSettings):
    database_url: str = os.getenv(
        'DATABASE_URL', 
        'postgresql://sutazai:sutazai_password@sutazai-postgres:5432/sutazai_db'
    )
    debug: bool = os.getenv('DEBUG', 'False').lower() == 'true'

    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

settings = DatabaseSettings()

# Create SQLAlchemy engine
engine = create_engine(
    settings.database_url,
    pool_size=20,
    max_overflow=30,
    pool_timeout=300,
    pool_pre_ping=True,
    connect_args={
        "keepalives": 1,
        "keepalives_idle": 30,
        "keepalives_interval": 10,
        "keepalives_count": 5
    }
)

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
    expire_on_commit=False
)

# Create declarative base
Base = declarative_base()

def get_db():
    """
    Database dependency with proper resource cleanup
    """
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database error: {str(e)}")
        db.rollback()
        raise
    finally:
        db.close()