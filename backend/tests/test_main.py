import pytest
from httpx import Client
from backend.main import app  # Import your FastAPI app

@pytest.fixture
def client():
    with Client(app=app, base_url="http://test") as client:
        yield client

def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "SutazAI Backend is running"} 