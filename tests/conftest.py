"""Common test fixtures and configuration."""
import os
import pytest
from pathlib import Path
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

# Import your FastAPI app and database models here
# from backend.main import app
# from backend.database import Base

@pytest.fixture(scope="session")
def test_db():
    """Create a test database."""
    SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
    engine = create_engine(
        SQLALCHEMY_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    
    # Create test database
    # Base.metadata.create_all(bind=engine)
    
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    
    try:
        yield engine
    finally:
        # Clean up
        os.remove("./test.db")

@pytest.fixture
def test_client():
    """Create a test client for FastAPI app."""
    # with TestClient(app) as client:
    #     yield client
    pass

@pytest.fixture
def test_data_dir():
    """Provide path to test data directory."""
    return Path(__file__).parent / "test_data"