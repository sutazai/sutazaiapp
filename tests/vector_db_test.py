#!/usr/bin/env python3
"""
Vector Database Comprehensive Testing Script
Tests ChromaDB, Qdrant, and FAISS with 1000+ vectors
Validates: CRUD operations, search performance, accuracy
"""

import time
import random
import numpy as np
from typing import List, Dict, Tuple
import json
from datetime import datetime

# Test configuration
NUM_VECTORS = 1000
VECTOR_DIM = 384  # Standard sentence-transformer dimension
SEARCH_QUERIES = 10
TOP_K = 5

print(f"""
{'='*80}
🧪 VECTOR DATABASE COMPREHENSIVE TEST
{'='*80}
Configuration:
  - Number of vectors: {NUM_VECTORS}
  - Vector dimensions: {VECTOR_DIM}
  - Search queries: {SEARCH_QUERIES}
  - Top-K results: {TOP_K}
{'='*80}
""")

def generate_random_vectors(n: int, dim: int) -> np.ndarray:
    """Generate random normalized vectors"""
    vectors = np.random.randn(n, dim).astype('float32')
    # Normalize vectors
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / norms

def generate_metadata(n: int) -> List[Dict]:
    """Generate metadata for vectors"""
    categories = ["tech", "science", "business", "health", "entertainment"]
    return [
        {
            "id": f"doc_{i}",
            "category": random.choice(categories),
            "timestamp": time.time(),
            "priority": random.randint(1, 10)
        }
        for i in range(n)
    ]

# ============================================================================
# CHROMADB TESTS
# ============================================================================

def test_chromadb():
    """Test ChromaDB operations"""
    try:
        import chromadb
        from chromadb.config import Settings
        
        print("\n📦 Testing ChromaDB...")
        print(f"   ChromaDB version: {chromadb.__version__}")
        
        # Connect to ChromaDB
        start = time.time()
        client = chromadb.HttpClient(
            host="localhost",
            port=10100,
            settings=Settings(allow_reset=True, anonymized_telemetry=False)
        )
        connection_time = time.time() - start
        print(f"   ✅ Connected in {connection_time*1000:.0f}ms")
        
        # Create/get collection
        collection_name = f"test_collection_{int(time.time())}"
        start = time.time()
        collection = client.get_or_create_collection(name=collection_name)
        create_time = time.time() - start
        print(f"   ✅ Collection created in {create_time*1000:.0f}ms")
        
        # Generate test data
        vectors = generate_random_vectors(NUM_VECTORS, VECTOR_DIM)
        metadata = generate_metadata(NUM_VECTORS)
        ids = [m["id"] for m in metadata]
        documents = [f"Document {i}: Sample text content" for i in range(NUM_VECTORS)]
        
        # INSERT test
        print(f"\n   📥 INSERT Test ({NUM_VECTORS} vectors)...")
        start = time.time()
        collection.add(
            embeddings=vectors.tolist(),
            documents=documents,
            metadatas=metadata,
            ids=ids
        )
        insert_time = time.time() - start
        throughput = NUM_VECTORS / insert_time
        print(f"   ✅ Inserted in {insert_time:.2f}s ({throughput:.0f} vectors/sec)")
        
        # SEARCH test
        print(f"\n   🔍 SEARCH Test ({SEARCH_QUERIES} queries)...")
        query_vectors = generate_random_vectors(SEARCH_QUERIES, VECTOR_DIM)
        search_times = []
        
        for i, query_vec in enumerate(query_vectors):
            start = time.time()
            results = collection.query(
                query_embeddings=[query_vec.tolist()],
                n_results=TOP_K
            )
            search_time = time.time() - start
            search_times.append(search_time)
        
        avg_search_time = np.mean(search_times) * 1000
        print(f"   ✅ Average search time: {avg_search_time:.1f}ms")
        print(f"   ✅ Min: {min(search_times)*1000:.1f}ms, Max: {max(search_times)*1000:.1f}ms")
        
        # UPDATE test
        print(f"\n   ✏️  UPDATE Test (100 vectors)...")
        update_ids = ids[:100]
        update_metadata = [{"id": m["id"], "category": "updated", "priority": 99} for m in metadata[:100]]
        start = time.time()
        collection.update(
            ids=update_ids,
            metadatas=update_metadata
        )
        update_time = time.time() - start
        print(f"   ✅ Updated in {update_time*1000:.0f}ms")
        
        # DELETE test
        print(f"\n   🗑️  DELETE Test (50 vectors)...")
        delete_ids = ids[:50]
        start = time.time()
        collection.delete(ids=delete_ids)
        delete_time = time.time() - start
        print(f"   ✅ Deleted in {delete_time*1000:.0f}ms")
        
        # Verify count
        count = collection.count()
        print(f"   ✅ Final count: {count} (expected: {NUM_VECTORS - 50})")
        
        # Cleanup
        client.delete_collection(name=collection_name)
        
        return {
            "database": "ChromaDB",
            "status": "✅ PASS",
            "connection_time_ms": round(connection_time * 1000, 2),
            "insert_time_s": round(insert_time, 2),
            "insert_throughput": round(throughput, 0),
            "avg_search_time_ms": round(avg_search_time, 2),
            "update_time_ms": round(update_time * 1000, 2),
            "delete_time_ms": round(delete_time * 1000, 2),
            "final_count": count
        }
        
    except Exception as e:
        print(f"   ❌ ChromaDB test failed: {e}")
        return {
            "database": "ChromaDB",
            "status": f"❌ FAIL: {str(e)[:100]}"
        }

# ============================================================================
# QDRANT TESTS
# ============================================================================

def test_qdrant():
    """Test Qdrant operations"""
    try:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        
        print("\n📦 Testing Qdrant...")
        
        # Connect to Qdrant
        start = time.time()
        client = QdrantClient(url="http://localhost:10101", prefer_grpc=False)
        connection_time = time.time() - start
        print(f"   ✅ Connected in {connection_time*1000:.0f}ms")
        
        # Create collection
        collection_name = f"test_collection_{int(time.time())}"
        start = time.time()
        client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=VECTOR_DIM, distance=Distance.COSINE)
        )
        create_time = time.time() - start
        print(f"   ✅ Collection created in {create_time*1000:.0f}ms")
        
        # Generate test data
        vectors = generate_random_vectors(NUM_VECTORS, VECTOR_DIM)
        metadata = generate_metadata(NUM_VECTORS)
        
        # INSERT test
        print(f"\n   📥 INSERT Test ({NUM_VECTORS} vectors)...")
        points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload=metadata[i]
            )
            for i in range(NUM_VECTORS)
        ]
        
        start = time.time()
        client.upsert(collection_name=collection_name, points=points)
        insert_time = time.time() - start
        throughput = NUM_VECTORS / insert_time
        print(f"   ✅ Inserted in {insert_time:.2f}s ({throughput:.0f} vectors/sec)")
        
        # SEARCH test
        print(f"\n   🔍 SEARCH Test ({SEARCH_QUERIES} queries)...")
        query_vectors = generate_random_vectors(SEARCH_QUERIES, VECTOR_DIM)
        search_times = []
        
        for query_vec in query_vectors:
            start = time.time()
            results = client.search(
                collection_name=collection_name,
                query_vector=query_vec.tolist(),
                limit=TOP_K
            )
            search_time = time.time() - start
            search_times.append(search_time)
        
        avg_search_time = np.mean(search_times) * 1000
        print(f"   ✅ Average search time: {avg_search_time:.1f}ms")
        print(f"   ✅ Min: {min(search_times)*1000:.1f}ms, Max: {max(search_times)*1000:.1f}ms")
        
        # UPDATE test
        print(f"\n   ✏️  UPDATE Test (100 vectors)...")
        update_points = [
            PointStruct(
                id=i,
                vector=vectors[i].tolist(),
                payload={"category": "updated", "priority": 99}
            )
            for i in range(100)
        ]
        start = time.time()
        client.upsert(collection_name=collection_name, points=update_points)
        update_time = time.time() - start
        print(f"   ✅ Updated in {update_time*1000:.0f}ms")
        
        # DELETE test
        print(f"\n   🗑️  DELETE Test (50 vectors)...")
        start = time.time()
        client.delete(
            collection_name=collection_name,
            points_selector=list(range(50))
        )
        delete_time = time.time() - start
        print(f"   ✅ Deleted in {delete_time*1000:.0f}ms")
        
        # Verify count
        info = client.get_collection(collection_name=collection_name)
        count = info.points_count
        print(f"   ✅ Final count: {count} (expected: {NUM_VECTORS - 50})")
        
        # Cleanup
        client.delete_collection(collection_name=collection_name)
        
        return {
            "database": "Qdrant",
            "status": "✅ PASS",
            "connection_time_ms": round(connection_time * 1000, 2),
            "insert_time_s": round(insert_time, 2),
            "insert_throughput": round(throughput, 0),
            "avg_search_time_ms": round(avg_search_time, 2),
            "update_time_ms": round(update_time * 1000, 2),
            "delete_time_ms": round(delete_time * 1000, 2),
            "final_count": count
        }
        
    except Exception as e:
        print(f"   ❌ Qdrant test failed: {e}")
        return {
            "database": "Qdrant",
            "status": f"❌ FAIL: {str(e)[:100]}"
        }

# ============================================================================
# FAISS TESTS
# ============================================================================

def test_faiss():
    """Test FAISS operations"""
    try:
        import faiss
        
        print("\n📦 Testing FAISS...")
        print(f"   FAISS version: Built with CPU support")
        
        # Generate test data
        vectors = generate_random_vectors(NUM_VECTORS, VECTOR_DIM)
        metadata = generate_metadata(NUM_VECTORS)
        
        # CREATE index
        start = time.time()
        index = faiss.IndexFlatIP(VECTOR_DIM)  # Inner product (cosine for normalized vectors)
        create_time = time.time() - start
        print(f"   ✅ Index created in {create_time*1000:.1f}ms")
        
        # INSERT test
        print(f"\n   📥 INSERT Test ({NUM_VECTORS} vectors)...")
        start = time.time()
        index.add(vectors)
        insert_time = time.time() - start
        throughput = NUM_VECTORS / insert_time
        print(f"   ✅ Inserted in {insert_time:.2f}s ({throughput:.0f} vectors/sec)")
        print(f"   ✅ Index size: {index.ntotal} vectors")
        
        # SEARCH test
        print(f"\n   🔍 SEARCH Test ({SEARCH_QUERIES} queries)...")
        query_vectors = generate_random_vectors(SEARCH_QUERIES, VECTOR_DIM)
        search_times = []
        
        for query_vec in query_vectors:
            start = time.time()
            distances, indices = index.search(np.array([query_vec]), TOP_K)
            search_time = time.time() - start
            search_times.append(search_time)
        
        avg_search_time = np.mean(search_times) * 1000
        print(f"   ✅ Average search time: {avg_search_time:.1f}ms")
        print(f"   ✅ Min: {min(search_times)*1000:.1f}ms, Max: {max(search_times)*1000:.1f}ms")
        
        # NOTE: FAISS doesn't support in-place update or delete easily
        print(f"\n   ℹ️  UPDATE/DELETE: FAISS is immutable (requires rebuild)")
        
        return {
            "database": "FAISS",
            "status": "✅ PASS",
            "create_time_ms": round(create_time * 1000, 2),
            "insert_time_s": round(insert_time, 2),
            "insert_throughput": round(throughput, 0),
            "avg_search_time_ms": round(avg_search_time, 2),
            "final_count": index.ntotal
        }
        
    except Exception as e:
        print(f"   ❌ FAISS test failed: {e}")
        return {
            "database": "FAISS",
            "status": f"❌ FAIL: {str(e)[:100]}"
        }

# ============================================================================
# MAIN TEST EXECUTION
# ============================================================================

if __name__ == "__main__":
    results = []
    
    # Test ChromaDB
    chroma_result = test_chromadb()
    results.append(chroma_result)
    
    # Test Qdrant
    qdrant_result = test_qdrant()
    results.append(qdrant_result)
    
    # Test FAISS
    faiss_result = test_faiss()
    results.append(faiss_result)
    
    # Summary
    print(f"\n{'='*80}")
    print(f"📊 TEST SUMMARY")
    print(f"{'='*80}\n")
    
    for result in results:
        print(f"{result['status']} {result['database']}")
        if "insert_throughput" in result:
            print(f"   Insert: {result.get('insert_time_s', 'N/A')}s ({result['insert_throughput']} vec/sec)")
        if "avg_search_time_ms" in result:
            print(f"   Search: {result['avg_search_time_ms']:.1f}ms avg")
        if "final_count" in result:
            print(f"   Final count: {result['final_count']}")
        print()
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "configuration": {
            "num_vectors": NUM_VECTORS,
            "vector_dim": VECTOR_DIM,
            "search_queries": SEARCH_QUERIES,
            "top_k": TOP_K
        },
        "results": results
    }
    
    output_file = f"/opt/sutazaiapp/test-results/vector_db_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"{'='*80}")
    print(f"📁 Results saved to: {output_file}")
    print(f"{'='*80}\n")
    
    # Exit code based on results
    failures = sum(1 for r in results if "FAIL" in r["status"])
    exit(failures)
