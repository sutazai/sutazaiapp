"""
Test module for the SutazAI backend main application.

This module contains tests for the main FastAPI application endpoints.
"""

from collections.abc import AsyncGenerator, Generator

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from backend.backend_main import app


@pytest.fixture
def test_client() -> Generator[TestClient, None, None]:
    """
    Fixture that provides a synchronous test client for the FastAPI app.

    Yields:
        TestClient: A test client for testing synchronous endpoints.
    """
    with TestClient(app) as client:
        yield client


@pytest.fixture
async def async_client() -> AsyncGenerator[AsyncClient, None]:
    """
    Fixture that provides an async test client for the FastAPI app.

    Yields:
        AsyncClient: An async test client for testing asynchronous endpoints.
    """
    async with AsyncClient(base_url="http://test") as client:
        # We need to manually dispatch requests to the app
        client.transport.app = app  # type: ignore
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
async def test_health_check(async_client: AsyncClient) -> None:
    """
    Test the health check endpoint using the asynchronous client.

    Args:
        async_client: The async test client fixture.

    Raises:
        AssertionError: If any test condition fails.
    """
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "0.1.0"}


@pytest.mark.asyncio
async def test_docs_endpoint(async_client: AsyncClient) -> None:
    """
    Test the documentation endpoints.

    Verifies:
    - Swagger UI is accessible
    - ReDoc is accessible

    Args:
        async_client: The async test client fixture.

    Raises:
        AssertionError: If documentation endpoints are not accessible.
    """
    swagger_response = await async_client.get("/docs")
    redoc_response = await async_client.get("/redoc")

    assert swagger_response.status_code == 200
    assert redoc_response.status_code == 200
