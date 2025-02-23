import hashlib
import logging
import re
import secrets
import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship, validates
from sqlalchemy.pool import QueuePool

from backend.config.database import settings

# Create declarative base
Base = declarative_base()

class BaseModel:
    """
    Base model with common fields and methods for enhanced tracking.
    """
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    is_active = Column(Boolean, default=True)

    @classmethod
    def generate_secure_id(cls):
        """
        Generate a cryptographically secure unique identifier.
        
        Returns:
            str: Secure unique identifier.
        """
        return secrets.token_urlsafe(16)

class SutazAiInteractionLog(BaseModel, Base):
    """
    Enhanced database model for tracking SutazAi interactions with advanced security and performance features.
    """
    __tablename__ = "interaction_logs"
    __table_args__ = (
        UniqueConstraint('user_id', 'timestamp', name='_user_timestamp_uc'),
        {'comment': 'SutazAi comprehensive interaction tracking'}
    )

    user_id = Column(String(50), nullable=False, index=True)
    endpoint = Column(String(100), nullable=False)
    request_hash = Column(String(64), nullable=True)  # SHA-256 hash
    request = Column(JSONB, nullable=True)  # Supports complex JSON
    response = Column(JSONB, nullable=True)
    model_used = Column(String(100), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Performance and security tracking
    request_size = Column(Integer, nullable=True)
    response_size = Column(Integer, nullable=True)
    processing_time = Column(Float, nullable=True)  # seconds
    is_successful = Column(Boolean, default=True)
    error_type = Column(String(100), nullable=True)

    @validates('request')
    def validate_request(self, key, request):
        """
        Validate and sanitize request data.
        
        Args:
            request (dict): Request data to validate.
        
        Returns:
            dict: Validated and sanitized request data.
        """
        if request and isinstance(request, dict):
            # Remove potentially sensitive information
            sanitized_request = {
                k: v for k, v in request.items() 
                if not re.search(r'(password|token|secret)', k, re.IGNORECASE)
            }
            return sanitized_request
        return request

    def generate_request_hash(self):
        """
        Generate a secure hash of the request for tracking and deduplication.
        
        Returns:
            str: SHA-256 hash of the request.
        """
        if self.request:
            request_str = json.dumps(self.request, sort_keys=True)
            return hashlib.sha256(request_str.encode()).hexdigest()
        return None

    def __repr__(self):
        return f"<SutazAiInteractionLog(id={self.id}, user_id={self.user_id}, endpoint={self.endpoint})>"

class SutazAiModelLog(BaseModel, Base):
    """
    Advanced database model for tracking model performance, usage, and health metrics.
    """
    __tablename__ = "model_logs"
    __table_args__ = (
        UniqueConstraint('model_name', 'timestamp', name='_model_timestamp_uc'),
        {'comment': 'SutazAi comprehensive model performance tracking'}
    )

    model_name = Column(String(100), nullable=False, index=True)
    model_version = Column(String(50), nullable=True)
    timestamp = Column(DateTime(timezone=True), server_default=func.now(), nullable=False)
    
    # Performance metrics
    inference_time = Column(Float, nullable=True)  # seconds
    tokens_generated = Column(Integer, nullable=True)
    context_length = Column(Integer, nullable=True)
    
    # Error and health tracking
    error_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    error_rate = Column(Float, nullable=True)
    
    # Resource utilization
    memory_usage = Column(Float, nullable=True)  # MB
    cpu_usage = Column(Float, nullable=True)  # Percentage
    
    # Model configuration snapshot
    model_config = Column(JSONB, nullable=True)
    
    def calculate_error_rate(self):
        """
        Calculate the error rate for the model.
        
        Returns:
            float: Error rate percentage.
        """
        total_calls = self.success_count + self.error_count
        return (self.error_count / total_calls * 100) if total_calls > 0 else 0.0

    def __repr__(self):
        return f"<SutazAiModelLog(id={self.id}, model_name={self.model_name})>"

class SystemConfiguration(Base):
    __tablename__ = 'system_configs'
    
    id = Column(Integer, primary_key=True)
    config_name = Column(String(50), nullable=False)  # Fixed length specification
    config_value = Column(String(200), nullable=False)
    
    @validates('config_value')
    def validate_config_value(self, key, value):
        if not isinstance(value, str):
            raise TypeError("Config value must be string")
        return value

    def __repr__(self) -> str:  # Added return type hint
        return f"<SystemConfiguration {self.config_name}>"

class SutazAiUser(Base):
    __tablename__ = 'sutazai_users'  # updated table name

    id = Column(Integer, primary_key=True)
    username = Column(String, unique=True, nullable=False)
    email = Column(String, unique=True, nullable=False)
    # other fields...

    def __init__(self, username, email):
        self.username = username
        self.email = email

    def __repr__(self):
        return f"<SutazAiUser(username={self.username}, email={self.email})>"

# Database engine configuration with enhanced security and performance
engine = create_engine(
    settings.database_url,
    poolclass=QueuePool,
    pool_size=10,  # Increased pool size
    max_overflow=20,  # More flexible connection handling
    pool_timeout=45,  # Extended timeout
    pool_recycle=3600,  # Recycle connections every hour
    pool_pre_ping=True,  # Test connection health before use
    echo=settings.debug  # Logging based on debug setting
)

def create_tables(drop_existing: bool = False):
    """
    Create database tables with optional existing table drop.
    
    Args:
        drop_existing (bool): Whether to drop existing tables before creation.
    """
    if drop_existing:
        Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)

def get_session():
    """
    Create a database session factory.
    
    Returns:
        Session: SQLAlchemy session factory.
    """
    from sqlalchemy.orm import sessionmaker
    return sessionmaker(bind=engine)

# Optional: Utility functions for database management
def validate_database_connection():
    """
    Validate database connection and configuration.
    
    Returns:
        bool: Whether the database connection is valid.
    """
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        logging.error(f"Database connection validation failed: {e}")
        return False

if __name__ == '__main__':
    # Perform database validation on direct script execution
    if validate_database_connection():
        print("Database connection is valid.")
        create_tables()
    else:
        print("Database connection failed.")