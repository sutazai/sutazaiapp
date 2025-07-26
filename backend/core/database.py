import logging
from typing import Generator # Import Generator

from sqlmodel import create_engine, Session, SQLModel
from sqlalchemy.orm import sessionmaker

from core.config import get_settings

settings = get_settings()

DATABASE_URL = str(settings.DATABASE_URI)

# Create engine
engine = create_engine(DATABASE_URL, echo=settings.SQLALCHEMY_ECHO)

# Create session local
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, class_=Session) # Use sqlmodel.Session

logger = logging.getLogger("database")

def get_db() -> Generator[Session, None, None]:
    """Dependency to get DB session."""
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {str(e)}") # Use logger
        db.rollback() # Rollback on error
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