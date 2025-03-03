"""
Tests for the main backend application.
"""

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from backend.backend_main import app


@pytest.fixture
def test_client() -> TestClient:
    """
    Fixture that provides a synchronous test client for the FastAPI app.
    
    Returns:
        TestClient: A test client for testing synchronous endpoints.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client() -> AsyncClient:
    """
    Fixture that provides an async test client for the FastAPI app.
    
    Returns:
        AsyncClient: An async test client for testing asynchronous endpoints.
    """
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


def test_health_check_sync(test_client: TestClient) -> None:
    """
    Test the health check endpoint using the synchronous client.
    
    Args:
        test_client: The test client fixture.
    """
    response = test_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "0.1.0"}


@pytest.mark.asyncio
async def test_health_check_async(async_client: AsyncClient) -> None:
    """
    Test the health check endpoint using the asynchronous client.
    
    Args:
        async_client: The async test client fixture.
    """
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "0.1.0"}


def test_docs_endpoints(test_client: TestClient) -> None:
    """
    Test that the API documentation endpoints are available in debug mode.
    
    Args:
        test_client: The test client fixture.
    """
    # Test Swagger UI
    swagger_response = test_client.get("/docs")
    assert swagger_response.status_code == 200
    
    # Test ReDoc
    redoc_response = test_client.get("/redoc")
    assert redoc_response.status_code == 200

