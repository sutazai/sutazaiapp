#!/usr/bin/env python3
"""
Vector Database Health Check
Tests all deployed vector databases
"""
import httpx
import asyncio
from datetime import datetime

async def test_chromadb():
    """Test ChromaDB connection"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test heartbeat endpoint
            response = await client.get("http://localhost:10100/api/v1/heartbeat")
            if response.status_code == 200:
                print(f"✅ ChromaDB: HEALTHY (heartbeat received)")
                return True
            else:
                print(f"⚠️  ChromaDB: Status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ ChromaDB: FAILED - {e}")
        return False

async def test_qdrant():
    """Test Qdrant connection"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test collections endpoint
            response = await client.get("http://localhost:10102/collections")
            if response.status_code == 200:
                print(f"✅ Qdrant: HEALTHY (REST API responding)")
                return True
            else:
                print(f"⚠️  Qdrant: Status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ Qdrant: FAILED - {e}")
        return False

async def test_faiss():
    """Test FAISS service connection"""
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            # Test health endpoint
            response = await client.get("http://localhost:10103/health")
            if response.status_code == 200:
                data = response.json()
                print(f"✅ FAISS: HEALTHY - {data}")
                return True
            else:
                print(f"⚠️  FAISS: Status {response.status_code}")
                return False
    except Exception as e:
        print(f"❌ FAISS: FAILED - {e}")
        return False

async def main():
    """Run all tests"""
    print("=" * 60)
    print("SutazAI Vector Database Health Check")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print("=" * 60)
    print()
    
    results = await asyncio.gather(
        test_chromadb(),
        test_qdrant(),
        test_faiss()
    )
    
    print()
    print("=" * 60)
    total = len(results)
    passed = sum(results)
    print(f"Results: {passed}/{total} services healthy")
    print("=" * 60)
    
    return all(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
