"""
Pytest Configuration and Fixtures for Backend Tests
Provides async database fixtures, test client, and shared test utilities
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy import event, text
import asyncio
import os
from typing import AsyncGenerator

# Set environment variables for testing before importing app
os.environ.setdefault("POSTGRES_PASSWORD", "sutazai_secure_2024")
os.environ.setdefault("RABBITMQ_PASSWORD", "sutazai_secure_2024")
os.environ.setdefault("NEO4J_PASSWORD", "sutazai_secure_2024")
os.environ.setdefault("CHROMADB_TOKEN", "sutazai-secure-token-2024")

from app.main import app
from app.core.database import Base, get_db
from app.core.config import settings

# Test database URL (separate from production)
TEST_DATABASE_URL = settings.DATABASE_URL.replace("/jarvis_ai", "/jarvis_ai_test")

# Create test engine with proper configuration
test_engine = create_async_engine(
    TEST_DATABASE_URL,
    echo=False,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10
)

# Create session factory
test_session_maker = async_sessionmaker(
    test_engine,
    class_=AsyncSession,
    expire_on_commit=False,
    autoflush=False
)


async def get_test_db() -> AsyncGenerator[AsyncSession, None]:
    """Override database dependency for tests"""
    async with test_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


# Event loop fixture removed - pytest-asyncio handles this automatically
# with asyncio_mode=auto and asyncio_default_fixture_loop_scope=function


@pytest_asyncio.fixture(scope="function")
async def db_session() -> AsyncGenerator[AsyncSession, None]:
    """Create fresh database for each test with proper cleanup"""
    # Create all tables
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    
    # Override dependency
    app.dependency_overrides[get_db] = get_test_db
    
    # Provide session for test
    async with test_session_maker() as session:
        try:
            yield session
        finally:
            # Ensure session is properly closed before cleanup
            await session.close()
    
    # Cleanup
    app.dependency_overrides.clear()
    
    # Drop all tables after test
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    
    # Dispose engine connections to prevent event loop issues
    await test_engine.dispose()


@pytest_asyncio.fixture
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """HTTP client for API requests with proper async transport"""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        try:
            yield ac
        finally:
            # Ensure client is properly closed
            await ac.aclose()


@pytest.fixture
def anyio_backend():
    """Configure anyio backend for pytest-asyncio"""
    return "asyncio"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers and settings"""
    config.addinivalue_line(
        "markers", "asyncio: mark test as an async test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
