"""
Test module for the SutazAI backend main application.

This module contains tests for the main FastAPI application endpoints.
"""

from typing import AsyncGenerator

import pytest
from httpx import AsyncClient

from backend.backend_main import app  # Fixed import path


@pytest.fixture
async def client() -> AsyncGenerator[AsyncClient, None]:
    """
    Fixture that provides an async test client for the FastAPI app.

    Yields:
        AsyncClient: An async test client configured for testing.
    """
    async with AsyncClient(app=app, base_url="http://test") as test_client:
        yield test_client


@pytest.mark.asyncio
async def test_health_check(client: AsyncClient) -> None:
    """
    Test the health check endpoint.

    Verifies:
    - Correct status code
    - Correct response structure
    - Expected health status and version

    Args:
        client: The async test client fixture.

    Raises:
        AssertionError: If any test condition fails.
    """
    response = await client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy", "version": "0.1.0"}


@pytest.mark.asyncio
async def test_docs_endpoint(client: AsyncClient) -> None:
    """
    Test the documentation endpoints.

    Verifies:
    - Swagger UI is accessible
    - ReDoc is accessible

    Args:
        client: The async test client fixture.

    Raises:
        AssertionError: If documentation endpoints are not accessible.
    """
    swagger_response = await client.get("/docs")
    redoc_response = await client.get("/redoc")

    assert swagger_response.status_code == 200
    assert redoc_response.status_code == 200
