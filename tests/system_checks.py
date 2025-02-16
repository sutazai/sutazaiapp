import requests
import pytest

@pytest.fixture(scope="module")
def api_client():
    return requests.Session()

def test_api_health(api_client):
    response = api_client.get("http://localhost:8000/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_model_integration(api_client):
    test_prompt = "Explain SutazAi computing in simple terms"
    response = api_client.post(
        "http://localhost:8000/api/v1/chat",
        json={"message": test_prompt, "model": "deepseek-33b"}
    )
    assert response.status_code == 200
    assert len(response.json()["response"]) > 100

def test_vector_db_connection():
    from data.vector_init import initialize_chroma
    collection = initialize_chroma()
    assert collection.count() == 0 