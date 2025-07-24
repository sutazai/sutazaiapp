import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock

from backend.main import app

client = TestClient(app)


# Mock the vector store for testing
@pytest.fixture(scope="module")
def mock_vector_service():
    with patch("backend.services.vector_store.vector_service.VectorStore") as mock:
        # Create a mock instance
        instance = MagicMock()

        # Setup search_documents to return mock results
        mock_results = [
            {
                "document_id": "doc1",
                "chunk_id": "chunk1",
                "text": "This is a test document about AI",
                "score": 0.95,
                "metadata": {"title": "Test Document 1", "page": 1},
            },
            {
                "document_id": "doc2",
                "chunk_id": "chunk2",
                "text": "Another test document about vector search",
                "score": 0.85,
                "metadata": {"title": "Test Document 2", "page": 1},
            },
        ]
        instance.search_documents.return_value = mock_results

        # Setup add_document to return success
        instance.add_document.return_value = {
            "document_id": "test_doc_id",
            "chunks_stored": 5,
        }

        # Return the mock instance when VectorStore is instantiated
        mock.return_value = instance
        yield instance


@pytest.mark.usefixtures("mock_vector_service")
@pytest.mark.skip(reason="Vector search endpoints not implemented yet.")
def test_vector_search():
    """Test the vector search endpoint."""
    response = client.post(
        "/vector/search", json={"query": "AI and vector search", "limit": 5}
    )

    assert response.status_code == 200
    data = response.json()
    assert isinstance(data, list)
    assert len(data) > 0
    # Verify structure of returned items
    assert "document_id" in data[0]
    assert "text" in data[0]
    assert "score" in data[0]


@pytest.mark.usefixtures("mock_vector_service")
@pytest.mark.skip(reason="Vector search endpoints not implemented yet.")
def test_vector_add_document():
    """Test adding a document to the vector store."""
    # Create sample document data
    document_data = {
        "document_id": "test123",
        "title": "Test Document",
        "text": "This is a test document for vector storage.",
        "metadata": {"author": "Test Author", "date": "2023-01-01"},
    }

    response = client.post("/vector/add", json=document_data)

    assert response.status_code == 200
    data = response.json()
    assert "document_id" in data
    assert "chunks_stored" in data
