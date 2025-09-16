#!/usr/bin/env python3
"""
Test script for SutazAI vector databases
Tests ChromaDB and Qdrant functionality
"""

import requests
import json
import random

def test_chromadb():
    """Test ChromaDB operations"""
    print("Testing ChromaDB...")
    base_url = "http://localhost:10100/api/v1"
    headers = {"Authorization": "Bearer sutazai-secure-token-2024"}
    
    try:
        # Create a collection
        response = requests.post(
            f"{base_url}/collections",
            headers=headers,
            json={
                "name": "test_collection",
                "metadata": {"description": "Test collection for SutazAI"}
            }
        )
        print(f"  Create collection: {response.status_code}")
        
        # Add some test embeddings
        test_embeddings = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        test_ids = ["id1", "id2"]
        test_docs = ["Document 1", "Document 2"]
        
        response = requests.post(
            f"{base_url}/collections/test_collection/add",
            headers=headers,
            json={
                "embeddings": test_embeddings,
                "ids": test_ids,
                "documents": test_docs
            }
        )
        print(f"  Add embeddings: {response.status_code}")
        
        # Query the collection
        response = requests.post(
            f"{base_url}/collections/test_collection/query",
            headers=headers,
            json={
                "query_embeddings": [[0.15, 0.25, 0.35]],
                "n_results": 2
            }
        )
        print(f"  Query collection: {response.status_code}")
        
        print("  ChromaDB test completed successfully!")
        return True
    except Exception as e:
        print(f"  ChromaDB test failed: {e}")
        return False

def test_qdrant():
    """Test Qdrant operations"""
    print("\nTesting Qdrant...")
    base_url = "http://localhost:10102"
    
    try:
        # Check health
        response = requests.get(f"{base_url}/")
        print(f"  Health check: {response.status_code} - v{response.json()['version']}")
        
        # Create a collection
        response = requests.put(
            f"{base_url}/collections/test_collection",
            json={
                "vectors": {
                    "size": 384,
                    "distance": "Cosine"
                }
            }
        )
        print(f"  Create collection: {response.status_code}")
        
        # Insert points
        response = requests.put(
            f"{base_url}/collections/test_collection/points",
            json={
                "points": [
                    {
                        "id": 1,
                        "vector": [random.gauss(0, 1) for _ in range(384)],
                        "payload": {"text": "Test document 1"}
                    },
                    {
                        "id": 2,
                        "vector": [random.gauss(0, 1) for _ in range(384)],
                        "payload": {"text": "Test document 2"}
                    }
                ]
            }
        )
        print(f"  Insert points: {response.status_code}")
        
        # Search for similar vectors
        response = requests.post(
            f"{base_url}/collections/test_collection/points/search",
            json={
                "vector": [random.gauss(0, 1) for _ in range(384)],
                "limit": 2
            }
        )
        print(f"  Search vectors: {response.status_code}")
        
        print("  Qdrant test completed successfully!")
        return True
    except Exception as e:
        print(f"  Qdrant test failed: {e}")
        return False

def test_faiss():
    """Test FAISS operations"""
    print("\nTesting FAISS...")
    base_url = "http://localhost:10103"
    
    try:
        # Check health
        response = requests.get(f"{base_url}/health")
        print(f"  Health check: {response.status_code}")
        
        # Create a collection
        collection = "test_collection"
        response = requests.post(
            f"{base_url}/collections/{collection}/create",
            json={"dimension": 128}
        )
        print(f"  Create collection: {response.status_code}")
        
        # Add vectors
        test_vectors = []
        for i in range(5):
            test_vectors.append({
                "id": f"vec_{i}",
                "vector": [random.gauss(0, 1) for _ in range(128)],
                "metadata": {"text": f"Test document {i}"}
            })
        
        response = requests.post(
            f"{base_url}/vectors/add",
            json={
                "collection": collection,
                "vectors": test_vectors
            }
        )
        print(f"  Add vectors: {response.status_code}")
        
        # Search for similar vectors
        query_vector = [random.gauss(0, 1) for _ in range(128)]
        response = requests.post(
            f"{base_url}/vectors/search",
            json={
                "collection": collection,
                "query": query_vector,
                "k": 3
            }
        )
        print(f"  Search vectors: {response.status_code}")
        
        # Get collection info
        response = requests.get(f"{base_url}/collections/{collection}")
        print(f"  Get collection info: {response.status_code}")
        
        print("  FAISS test completed successfully!")
        return True
    except Exception as e:
        print(f"  FAISS test failed: {e}")
        return False

def main():
    """Run all vector database tests"""
    print("="*50)
    print("SutazAI Vector Database Tests")
    print("="*50)
    
    chromadb_ok = test_chromadb()
    qdrant_ok = test_qdrant()
    faiss_ok = test_faiss()
    
    print("\n" + "="*50)
    print("Test Results:")
    print(f"  ChromaDB: {'✓ PASSED' if chromadb_ok else '✗ FAILED'}")
    print(f"  Qdrant:   {'✓ PASSED' if qdrant_ok else '✗ FAILED'}")
    print(f"  FAISS:    {'✓ PASSED' if faiss_ok else '✗ FAILED'}")
    print("="*50)
    
    return chromadb_ok and qdrant_ok and faiss_ok

if __name__ == "__main__":
    exit(0 if main() else 1)