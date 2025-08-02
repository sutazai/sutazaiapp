"""
Integration tests for API endpoints
"""
import pytest
import httpx
import asyncio
from typing import Dict, Any

BASE_URL = "http://localhost:8000"

@pytest.fixture
async def client():
    """Create async HTTP client"""
    async with httpx.AsyncClient(base_url=BASE_URL) as client:
        yield client

@pytest.fixture
async def auth_headers(client):
    """Get authentication headers"""
    # Login to get token
    response = await client.post(
        "/api/v1/security/login",
        json={"username": "admin", "password": "password"}
    )
    if response.status_code == 200:
        token = response.json()["access_token"]
        return {"Authorization": f"Bearer {token}"}
    return {}

@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check endpoint"""
    response = await client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"

@pytest.mark.asyncio
async def test_system_status(client):
    """Test system status endpoint"""
    response = await client.get("/api/v1/system/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "operational"
    assert "services" in data

@pytest.mark.asyncio
async def test_coordinator_think(client, auth_headers):
    """Test automation coordinator thinking endpoint"""
    payload = {
        "input_data": {"text": "What is the meaning of life?"},
        "reasoning_type": "strategic"
    }
    
    response = await client.post(
        "/api/v1/coordinator/think",
        json=payload,
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "thought_id" in data
    assert "result" in data
    assert data["reasoning_type"] == "strategic"

@pytest.mark.asyncio
async def test_coordinator_status(client, auth_headers):
    """Test coordinator status endpoint"""
    response = await client.get(
        "/api/v1/coordinator/status",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "active_thoughts" in data
    assert "memory_usage" in data

@pytest.mark.asyncio
async def test_models_list(client, auth_headers):
    """Test models listing endpoint"""
    response = await client.get(
        "/api/v1/models/",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "models" in data
    assert isinstance(data["models"], list)

@pytest.mark.asyncio
async def test_agents_status(client, auth_headers):
    """Test agents status endpoint"""
    response = await client.get(
        "/api/v1/agents/status",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "agents" in data
    assert isinstance(data["agents"], list)

@pytest.mark.asyncio
async def test_security_login(client):
    """Test login endpoint"""
    response = await client.post(
        "/api/v1/security/login",
        json={"username": "admin", "password": "password"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert "refresh_token" in data
    assert data["token_type"] == "bearer"

@pytest.mark.asyncio
async def test_security_login_invalid(client):
    """Test login with invalid credentials"""
    response = await client.post(
        "/api/v1/security/login",
        json={"username": "invalid", "password": "wrong"}
    )
    
    assert response.status_code == 401

@pytest.mark.asyncio
async def test_feedback_status(client, auth_headers):
    """Test feedback loop status"""
    response = await client.get(
        "/api/v1/feedback/status",
        headers=auth_headers
    )
    
    assert response.status_code == 200
    data = response.json()
    assert "is_running" in data
    assert "metrics_collected" in data

@pytest.mark.asyncio
async def test_rate_limiting(client):
    """Test rate limiting"""
    # Send many requests quickly
    tasks = []
    for _ in range(150):  # Exceed rate limit
        tasks.append(client.get("/api/v1/system/status"))
    
    responses = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Check that some requests were rate limited
    status_codes = [r.status_code for r in responses if hasattr(r, 'status_code')]
    assert 429 in status_codes  # Too Many Requests

@pytest.mark.asyncio
async def test_cors_headers(client):
    """Test CORS headers"""
    response = await client.options(
        "/api/v1/coordinator/think",
        headers={"Origin": "http://localhost:3000"}
    )
    
    assert "Access-Control-Allow-Origin" in response.headers
    assert "Access-Control-Allow-Methods" in response.headers
    assert "Access-Control-Allow-Headers" in response.headers

@pytest.mark.asyncio
async def test_security_headers(client):
    """Test security headers in responses"""
    response = await client.get("/health")
    
    security_headers = [
        "X-Content-Type-Options",
        "X-Frame-Options",
        "X-XSS-Protection",
        "Strict-Transport-Security"
    ]
    
    for header in security_headers:
        assert header in response.headers

@pytest.mark.asyncio
async def test_concurrent_requests(client, auth_headers):
    """Test handling concurrent requests"""
    tasks = []
    for i in range(10):
        payload = {
            "input_data": {"text": f"Concurrent request {i}"},
            "reasoning_type": "deductive"
        }
        tasks.append(client.post(
            "/api/v1/coordinator/think",
            json=payload,
            headers=auth_headers
        ))
    
    responses = await asyncio.gather(*tasks)
    
    # All requests should succeed
    assert all(r.status_code == 200 for r in responses)
    
    # All should have unique thought IDs
    thought_ids = [r.json()["thought_id"] for r in responses]
    assert len(thought_ids) == len(set(thought_ids))

@pytest.mark.asyncio
async def test_error_handling(client, auth_headers):
    """Test API error handling"""
    # Test with invalid payload
    response = await client.post(
        "/api/v1/coordinator/think",
        json={"invalid": "payload"},
        headers=auth_headers
    )
    
    assert response.status_code == 422  # Unprocessable Entity
    assert "detail" in response.json()

@pytest.mark.asyncio
async def test_encryption_endpoints(client, auth_headers):
    """Test encryption/decryption endpoints"""
    # Encrypt data
    encrypt_response = await client.post(
        "/api/v1/security/encrypt",
        json={"data": "sensitive information"},
        headers=auth_headers
    )
    
    assert encrypt_response.status_code == 200
    encrypted = encrypt_response.json()["encrypted"]
    
    # Decrypt data
    decrypt_response = await client.post(
        "/api/v1/security/decrypt",
        json={"encrypted": encrypted},
        headers=auth_headers
    )
    
    assert decrypt_response.status_code == 200
    assert decrypt_response.json()["decrypted"] == "sensitive information"