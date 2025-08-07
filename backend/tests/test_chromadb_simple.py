#!/usr/bin/env python3
"""
Simple ChromaDB test using HTTP API directly
"""

import requests
import json
import time
import socket
import pytest


def _is_port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False

def test_chromadb():
    base_url = "http://localhost:8001/api/v1"
    if not _is_port_open("localhost", 8001):
        pytest.skip("ChromaDB not running on localhost:8001; skipping external integration test")
    
    print("Testing ChromaDB HTTP API...")
    
    # Test 1: Heartbeat
    response = requests.get(f"{base_url}/heartbeat")
    print(f"Heartbeat: {response.status_code}")
    
    # Test 2: Version
    response = requests.get(f"{base_url}/version")
    print(f"Version: {response.json()}")
    
    # Test 3: List collections
    response = requests.get(f"{base_url}/collections")
    print(f"Collections: {response.json()}")
    
    # Test 4: Create collection
    collection_name = f"test_collection_{int(time.time())}"
    collection_data = {
        "name": collection_name,
        "metadata": {"description": "Test collection"}
    }
    
    response = requests.post(f"{base_url}/collections", json=collection_data)
    print(f"Create collection: {response.status_code}")
    if response.status_code == 200:
        print(f"Collection created: {response.json()}")
    
    # Test 5: Add documents
    documents_data = {
        "ids": ["doc1", "doc2"],
        "documents": ["Document 1 content", "Document 2 content"],
        "metadatas": [{"type": "test"}, {"type": "test"}]
    }
    
    response = requests.post(f"{base_url}/collections/{collection_name}/add", json=documents_data)
    print(f"Add documents: {response.status_code}")
    
    # Test 6: Query documents
    query_data = {
        "query_texts": ["Document content"],
        "n_results": 5
    }
    
    response = requests.post(f"{base_url}/collections/{collection_name}/query", json=query_data)
    print(f"Query documents: {response.status_code}")
    if response.status_code == 200:
        print(f"Query results: {len(response.json().get('documents', [[]])[0])} documents found")
    
    # Cleanup: Delete collection
    response = requests.delete(f"{base_url}/collections/{collection_name}")
    print(f"Delete collection: {response.status_code}")
    
    print("ChromaDB test completed!")

if __name__ == "__main__":
    test_chromadb()
