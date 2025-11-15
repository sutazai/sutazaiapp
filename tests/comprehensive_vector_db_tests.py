#!/usr/bin/env python3
"""
Comprehensive Vector Database Testing Suite - Phase 6
Tests ChromaDB, Qdrant, and FAISS operations with performance metrics

Created: 2025-11-15 19:00:00 UTC
Execution ID: phase6_vector_db_comprehensive_test
"""

import asyncio
import time
import json
import numpy as np
import requests
import chromadb
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import statistics

# Test Configuration
CHROMADB_URL = "http://localhost:10100"
CHROMADB_TOKEN = "sutazai-secure-token-2024"
QDRANT_URL = "http://localhost:10102"
FAISS_URL = "http://localhost:10103"

TEST_START_TIME = datetime.now(timezone.utc)
EXECUTION_ID = f"phase6_{TEST_START_TIME.strftime('%Y%m%d_%H%M%S')}"

@dataclass
class PerformanceMetrics:
    """Performance metrics for vector operations"""
    operation: str
    database: str
    vector_count: int
    dimension: int
    duration_ms: float
    throughput_vps: float  # vectors per second
    memory_mb: float = 0.0
    success: bool = True
    error: str = ""
    timestamp: str = ""

class VectorDBTester:
    """Comprehensive vector database testing framework"""
    
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
        self.test_results = {}
        # Initialize ChromaDB client
        self.chromadb_client = chromadb.HttpClient(
            host="localhost",
            port=10100,
            headers={"X-Chroma-Token": CHROMADB_TOKEN}
        )
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        timestamp = datetime.now(timezone.utc).isoformat()
        print(f"[{timestamp}] {level}: {message}")
        
    def generate_random_vectors(self, count: int, dimension: int) -> np.ndarray:
        """Generate random normalized vectors"""
        vectors = np.random.randn(count, dimension).astype(np.float32)
        # Normalize vectors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return (vectors / norms).tolist()
    
    def record_metric(self, operation: str, database: str, vector_count: int, 
                     dimension: int, duration_ms: float, success: bool = True, 
                     error: str = ""):
        """Record performance metric"""
        throughput = (vector_count / duration_ms) * 1000 if duration_ms > 0 else 0
        metric = PerformanceMetrics(
            operation=operation,
            database=database,
            vector_count=vector_count,
            dimension=dimension,
            duration_ms=duration_ms,
            throughput_vps=throughput,
            success=success,
            error=error,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        self.metrics.append(metric)
        return metric
    
    # ==================== ChromaDB Tests ====================
    
    def test_chromadb_collection_creation(self) -> bool:
        """Test ChromaDB collection creation with Python client"""
        self.log("Testing ChromaDB Collection Creation", "TEST")
        
        try:
            start_time = time.time()
            
            # Delete collection if exists (cleanup)
            try:
                self.chromadb_client.delete_collection("test_collection")
            except:
                pass
            
            # Create collection
            collection = self.chromadb_client.create_collection(
                name="test_collection",
                metadata={"test": "phase6"}
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            self.log(f"✅ ChromaDB collection created: {duration_ms:.2f}ms", "SUCCESS")
            self.record_metric("create_collection", "ChromaDB", 1, 0, duration_ms)
            return True
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"❌ ChromaDB collection creation failed: {e}", "ERROR")
            self.record_metric("create_collection", "ChromaDB", 1, 0, duration_ms, False, str(e))
            return False
    
    def test_chromadb_embedding_insertion(self, count: int, dimension: int = 384) -> bool:
        """Test ChromaDB embedding insertion with performance metrics"""
        self.log(f"Testing ChromaDB Embedding Insertion ({count} vectors, {dimension}D)", "TEST")
        
        try:
            start_time = time.time()
            
            # Get collection
            collection = self.chromadb_client.get_collection("test_collection")
            
            # Generate test vectors
            vectors = self.generate_random_vectors(count, dimension)
            ids = [f"vec_{i}" for i in range(count)]
            metadatas = [{"index": i, "type": "test"} for i in range(count)]
            
            # Insert embeddings
            collection.add(
                ids=ids,
                embeddings=vectors,
                metadatas=metadatas
            )
            
            duration_ms = (time.time() - start_time) * 1000
            throughput = (count / duration_ms) * 1000
            self.log(f"✅ Inserted {count} vectors in {duration_ms:.2f}ms ({throughput:.2f} vec/s)", "SUCCESS")
            self.record_metric("insert_embeddings", "ChromaDB", count, dimension, duration_ms)
            return True
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"❌ ChromaDB embedding insertion failed: {e}", "ERROR")
            self.record_metric("insert_embeddings", "ChromaDB", count, dimension, duration_ms, False, str(e))
            return False
    
    def test_chromadb_similarity_search(self, k: int = 10, dimension: int = 384) -> bool:
        """Test ChromaDB similarity search performance"""
        self.log(f"Testing ChromaDB Similarity Search (k={k})", "TEST")
        
        try:
            start_time = time.time()
            
            # Get collection
            collection = self.chromadb_client.get_collection("test_collection")
            
            # Generate query vector
            query_vector = self.generate_random_vectors(1, dimension)[0]
            
            # Perform search
            results = collection.query(
                query_embeddings=[query_vector],
                n_results=k
            )
            
            duration_ms = (time.time() - start_time) * 1000
            num_results = len(results.get("ids", [[]])[0])
            self.log(f"✅ Search completed in {duration_ms:.2f}ms, found {num_results} results", "SUCCESS")
            self.record_metric("similarity_search", "ChromaDB", k, dimension, duration_ms)
            return True
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"❌ ChromaDB similarity search failed: {e}", "ERROR")
            self.record_metric("similarity_search", "ChromaDB", k, dimension, duration_ms, False, str(e))
            return False
    
    def test_chromadb_update_delete(self) -> bool:
        """Test ChromaDB update and delete operations"""
        self.log("Testing ChromaDB Update/Delete Operations", "TEST")
        
        try:
            start_time = time.time()
            
            # Get collection
            collection = self.chromadb_client.get_collection("test_collection")
            
            # Update operation
            new_vector = self.generate_random_vectors(1, 384)[0]
            collection.update(
                ids=["vec_0"],
                embeddings=[new_vector],
                metadatas=[{"index": 0, "type": "updated"}]
            )
            
            # Delete operation
            collection.delete(ids=["vec_1", "vec_2"])
            
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"✅ Update/Delete operations completed in {duration_ms:.2f}ms", "SUCCESS")
            self.record_metric("update_delete", "ChromaDB", 3, 384, duration_ms)
            return True
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"❌ ChromaDB update/delete failed: {e}", "ERROR")
            self.record_metric("update_delete", "ChromaDB", 3, 384, duration_ms, False, str(e))
            return False
    
    # ==================== Qdrant Tests ====================
    
    def test_qdrant_collection_creation(self, dimension: int = 384) -> bool:
        """Test Qdrant collection creation"""
        self.log(f"Testing Qdrant Collection Creation ({dimension}D)", "TEST")
        
        try:
            start_time = time.time()
            
            # Delete collection if exists (cleanup)
            try:
                requests.delete(f"{QDRANT_URL}/collections/test_collection")
            except:
                pass
            
            # Create collection
            collection_config = {
                "vectors": {
                    "size": dimension,
                    "distance": "Cosine"
                }
            }
            
            response = requests.put(
                f"{QDRANT_URL}/collections/test_collection",
                json=collection_config
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code in [200, 201]:
                self.log(f"✅ Qdrant collection created in {duration_ms:.2f}ms", "SUCCESS")
                self.record_metric("create_collection", "Qdrant", 1, dimension, duration_ms)
                return True
            else:
                raise Exception(f"Failed to create collection: {response.status_code} - {response.text}")
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"❌ Qdrant collection creation failed: {e}", "ERROR")
            self.record_metric("create_collection", "Qdrant", 1, dimension, duration_ms, False, str(e))
            return False
    
    def test_qdrant_point_insertion(self, count: int, dimension: int = 384) -> bool:
        """Test Qdrant point insertion with performance metrics"""
        self.log(f"Testing Qdrant Point Insertion ({count} points, {dimension}D)", "TEST")
        
        try:
            start_time = time.time()
            
            # Generate test points
            vectors = self.generate_random_vectors(count, dimension)
            
            points = [
                {
                    "id": i,
                    "vector": vector,
                    "payload": {
                        "index": i,
                        "type": "test",
                        "text": f"Document {i}"
                    }
                }
                for i, vector in enumerate(vectors)
            ]
            
            # Insert points
            response = requests.put(
                f"{QDRANT_URL}/collections/test_collection/points",
                json={"points": points}
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code in [200, 201]:
                throughput = (count / duration_ms) * 1000
                self.log(f"✅ Inserted {count} points in {duration_ms:.2f}ms ({throughput:.2f} pts/s)", "SUCCESS")
                self.record_metric("insert_points", "Qdrant", count, dimension, duration_ms)
                return True
            else:
                raise Exception(f"Failed to insert points: {response.status_code} - {response.text}")
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"❌ Qdrant point insertion failed: {e}", "ERROR")
            self.record_metric("insert_points", "Qdrant", count, dimension, duration_ms, False, str(e))
            return False
    
    def test_qdrant_search_performance(self, k: int = 10, dimension: int = 384) -> bool:
        """Test Qdrant search performance"""
        self.log(f"Testing Qdrant Search Performance (k={k})", "TEST")
        
        try:
            start_time = time.time()
            
            # Generate query vector
            query_vector = self.generate_random_vectors(1, dimension)[0]
            
            # Perform search
            search_payload = {
                "vector": query_vector,
                "limit": k,
                "with_payload": True
            }
            
            response = requests.post(
                f"{QDRANT_URL}/collections/test_collection/points/search",
                json=search_payload
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                results = response.json().get("result", [])
                self.log(f"✅ Search completed in {duration_ms:.2f}ms, found {len(results)} results", "SUCCESS")
                self.record_metric("search", "Qdrant", k, dimension, duration_ms)
                return True
            else:
                raise Exception(f"Search failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"❌ Qdrant search failed: {e}", "ERROR")
            self.record_metric("search", "Qdrant", k, dimension, duration_ms, False, str(e))
            return False
    
    def test_qdrant_filtering(self, dimension: int = 384) -> bool:
        """Test Qdrant filtering capabilities"""
        self.log("Testing Qdrant Filtering Capabilities", "TEST")
        
        try:
            start_time = time.time()
            
            # Generate query vector
            query_vector = self.generate_random_vectors(1, dimension)[0]
            
            # Search with filter
            search_payload = {
                "vector": query_vector,
                "limit": 10,
                "filter": {
                    "must": [
                        {"key": "type", "match": {"value": "test"}}
                    ]
                },
                "with_payload": True
            }
            
            response = requests.post(
                f"{QDRANT_URL}/collections/test_collection/points/search",
                json=search_payload
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                results = response.json().get("result", [])
                self.log(f"✅ Filtered search completed in {duration_ms:.2f}ms, found {len(results)} results", "SUCCESS")
                self.record_metric("filtered_search", "Qdrant", 10, dimension, duration_ms)
                return True
            else:
                raise Exception(f"Filtered search failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"❌ Qdrant filtering failed: {e}", "ERROR")
            self.record_metric("filtered_search", "Qdrant", 10, dimension, duration_ms, False, str(e))
            return False
    
    # ==================== FAISS Tests ====================
    
    def test_faiss_index_creation(self, dimension: int = 768) -> bool:
        """Test FAISS index creation"""
        self.log(f"Testing FAISS Index Creation ({dimension}D)", "TEST")
        
        try:
            start_time = time.time()
            
            # Delete index if exists (cleanup)
            try:
                requests.delete(f"{FAISS_URL}/collections/test_collection")
            except:
                pass
            
            # Create index
            index_config = {
                "collection": "test_collection",
                "dimension": dimension,
                "index_type": "Flat"
            }
            
            response = requests.post(
                f"{FAISS_URL}/index/create",
                json=index_config
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code in [200, 201]:
                self.log(f"✅ FAISS index created in {duration_ms:.2f}ms", "SUCCESS")
                self.record_metric("create_index", "FAISS", 1, dimension, duration_ms)
                return True
            else:
                raise Exception(f"Failed to create index: {response.status_code} - {response.text}")
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"❌ FAISS index creation failed: {e}", "ERROR")
            self.record_metric("create_index", "FAISS", 1, dimension, duration_ms, False, str(e))
            return False
    
    def test_faiss_vector_operations(self, count: int, dimension: int = 768) -> bool:
        """Test FAISS vector operations"""
        self.log(f"Testing FAISS Vector Operations ({count} vectors, {dimension}D)", "TEST")
        
        try:
            start_time = time.time()
            
            # Generate test vectors
            vectors = self.generate_random_vectors(count, dimension)
            
            vector_data = [
                {
                    "id": f"vec_{i}",
                    "vector": vector,
                    "metadata": {"index": i, "type": "test"}
                }
                for i, vector in enumerate(vectors)
            ]
            
            # Add vectors
            response = requests.post(
                f"{FAISS_URL}/vectors/add?collection=test_collection",
                json=vector_data
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code in [200, 201]:
                throughput = (count / duration_ms) * 1000
                self.log(f"✅ Added {count} vectors in {duration_ms:.2f}ms ({throughput:.2f} vec/s)", "SUCCESS")
                self.record_metric("add_vectors", "FAISS", count, dimension, duration_ms)
                return True
            else:
                raise Exception(f"Failed to add vectors: {response.status_code} - {response.text}")
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"❌ FAISS vector operations failed: {e}", "ERROR")
            self.record_metric("add_vectors", "FAISS", count, dimension, duration_ms, False, str(e))
            return False
    
    def test_faiss_search_performance(self, k: int = 10, dimension: int = 768) -> bool:
        """Test FAISS search performance"""
        self.log(f"Testing FAISS Search Performance (k={k})", "TEST")
        
        try:
            start_time = time.time()
            
            # Generate query vector
            query_vector = self.generate_random_vectors(1, dimension)[0]
            
            # Perform search
            search_payload = {
                "collection": "test_collection",
                "vector": query_vector,
                "k": k
            }
            
            response = requests.post(
                f"{FAISS_URL}/search",
                json=search_payload
            )
            
            duration_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                results = response.json().get("results", [])
                self.log(f"✅ Search completed in {duration_ms:.2f}ms, found {len(results)} results", "SUCCESS")
                self.record_metric("search", "FAISS", k, dimension, duration_ms)
                return True
            else:
                raise Exception(f"Search failed: {response.status_code} - {response.text}")
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            self.log(f"❌ FAISS search failed: {e}", "ERROR")
            self.record_metric("search", "FAISS", k, dimension, duration_ms, False, str(e))
            return False
    
    # ==================== Performance Comparison ====================
    
    def run_performance_comparison(self):
        """Run comprehensive performance comparison across all databases"""
        self.log("=" * 80, "INFO")
        self.log("PERFORMANCE COMPARISON - ALL VECTOR DATABASES", "INFO")
        self.log("=" * 80, "INFO")
        
        # Test configurations
        dimensions = [384, 768]
        vector_counts = [100, 1000]
        search_k_values = [1, 10, 100]
        
        for dim in dimensions:
            self.log(f"\n{'=' * 40}", "INFO")
            self.log(f"Testing with dimension: {dim}", "INFO")
            self.log(f"{'=' * 40}", "INFO")
            
            # ChromaDB tests
            self.log("\n--- ChromaDB Tests ---", "INFO")
            self.test_chromadb_collection_creation()
            for count in vector_counts:
                self.test_chromadb_embedding_insertion(count, dim)
            for k in search_k_values:
                self.test_chromadb_similarity_search(k, dim)
            
            # Qdrant tests
            self.log("\n--- Qdrant Tests ---", "INFO")
            self.test_qdrant_collection_creation(dim)
            for count in vector_counts:
                self.test_qdrant_point_insertion(count, dim)
            for k in search_k_values:
                self.test_qdrant_search_performance(k, dim)
            
            # FAISS tests (only test native dimension)
            if dim == 768:
                self.log("\n--- FAISS Tests ---", "INFO")
                self.test_faiss_index_creation(dim)
                for count in vector_counts:
                    self.test_faiss_vector_operations(count, dim)
                for k in search_k_values:
                    self.test_faiss_search_performance(k, dim)
    
    def generate_performance_report(self) -> str:
        """Generate comprehensive performance report"""
        report_lines = []
        report_lines.append("=" * 100)
        report_lines.append("PHASE 6: VECTOR DATABASE PERFORMANCE REPORT")
        report_lines.append(f"Generated: {datetime.now(timezone.utc).isoformat()}")
        report_lines.append(f"Execution ID: {EXECUTION_ID}")
        report_lines.append("=" * 100)
        
        # Group metrics by database and operation
        db_metrics = {}
        for metric in self.metrics:
            db = metric.database
            op = metric.operation
            if db not in db_metrics:
                db_metrics[db] = {}
            if op not in db_metrics[db]:
                db_metrics[db][op] = []
            db_metrics[db][op].append(metric)
        
        # Summary statistics
        report_lines.append("\n## SUMMARY STATISTICS")
        report_lines.append("-" * 100)
        
        for db, operations in sorted(db_metrics.items()):
            report_lines.append(f"\n### {db}")
            report_lines.append(f"{'Operation':<30} {'Count':<8} {'Avg Duration (ms)':<20} {'Avg Throughput (vec/s)':<25} {'Success Rate':<15}")
            report_lines.append("-" * 100)
            
            for op, metrics_list in sorted(operations.items()):
                if not metrics_list:
                    continue
                    
                durations = [m.duration_ms for m in metrics_list if m.success]
                throughputs = [m.throughput_vps for m in metrics_list if m.success and m.throughput_vps > 0]
                success_count = sum(1 for m in metrics_list if m.success)
                total_count = len(metrics_list)
                
                avg_duration = statistics.mean(durations) if durations else 0
                avg_throughput = statistics.mean(throughputs) if throughputs else 0
                success_rate = (success_count / total_count) * 100 if total_count > 0 else 0
                
                report_lines.append(
                    f"{op:<30} {total_count:<8} {avg_duration:<20.2f} {avg_throughput:<25.2f} {success_rate:<15.1f}%"
                )
        
        # Detailed metrics
        report_lines.append("\n\n## DETAILED METRICS")
        report_lines.append("-" * 100)
        report_lines.append(f"{'Timestamp':<30} {'Database':<12} {'Operation':<25} {'Vectors':<10} {'Dim':<8} {'Duration (ms)':<15} {'Throughput':<20} {'Status':<10}")
        report_lines.append("-" * 100)
        
        for metric in self.metrics:
            status = "✅ OK" if metric.success else "❌ FAIL"
            report_lines.append(
                f"{metric.timestamp[11:19]:<30} {metric.database:<12} {metric.operation:<25} "
                f"{metric.vector_count:<10} {metric.dimension:<8} {metric.duration_ms:<15.2f} "
                f"{metric.throughput_vps:<20.2f} {status:<10}"
            )
        
        # Performance comparison table
        report_lines.append("\n\n## PERFORMANCE COMPARISON")
        report_lines.append("-" * 100)
        
        # Insertion performance
        report_lines.append("\n### Insertion Performance (vectors/second)")
        report_lines.append(f"{'Database':<15} {'100 vectors':<20} {'1000 vectors':<20}")
        report_lines.append("-" * 60)
        
        for db in sorted(db_metrics.keys()):
            throughputs = {}
            for op in ["insert_embeddings", "insert_points", "add_vectors"]:
                if op in db_metrics[db]:
                    for metric in db_metrics[db][op]:
                        if metric.success and metric.throughput_vps > 0:
                            count = metric.vector_count
                            if count not in throughputs:
                                throughputs[count] = []
                            throughputs[count].append(metric.throughput_vps)
            
            avg_100 = statistics.mean(throughputs.get(100, [0])) if 100 in throughputs else 0
            avg_1000 = statistics.mean(throughputs.get(1000, [0])) if 1000 in throughputs else 0
            
            report_lines.append(f"{db:<15} {avg_100:<20.2f} {avg_1000:<20.2f}")
        
        # Search performance
        report_lines.append("\n### Search Performance (milliseconds)")
        report_lines.append(f"{'Database':<15} {'k=1':<15} {'k=10':<15} {'k=100':<15}")
        report_lines.append("-" * 60)
        
        for db in sorted(db_metrics.keys()):
            latencies = {}
            for op in ["similarity_search", "search"]:
                if op in db_metrics[db]:
                    for metric in db_metrics[db][op]:
                        if metric.success:
                            k = metric.vector_count
                            if k not in latencies:
                                latencies[k] = []
                            latencies[k].append(metric.duration_ms)
            
            avg_k1 = statistics.mean(latencies.get(1, [0])) if 1 in latencies else 0
            avg_k10 = statistics.mean(latencies.get(10, [0])) if 10 in latencies else 0
            avg_k100 = statistics.mean(latencies.get(100, [0])) if 100 in latencies else 0
            
            report_lines.append(f"{db:<15} {avg_k1:<15.2f} {avg_k10:<15.2f} {avg_k100:<15.2f}")
        
        # Recommendations
        report_lines.append("\n\n## RECOMMENDATIONS")
        report_lines.append("-" * 100)
        report_lines.append("Based on performance testing:")
        report_lines.append("")
        report_lines.append("1. **ChromaDB**: Best for applications requiring built-in document storage and metadata management")
        report_lines.append("   - Excellent v2 API support")
        report_lines.append("   - Good for RAG applications with document retrieval")
        report_lines.append("")
        report_lines.append("2. **Qdrant**: Best for high-performance vector search with complex filtering")
        report_lines.append("   - Superior filtering capabilities")
        report_lines.append("   - Excellent for multi-tenant applications")
        report_lines.append("   - Good balance of speed and features")
        report_lines.append("")
        report_lines.append("3. **FAISS**: Best for pure vector similarity search at scale")
        report_lines.append("   - Fastest search performance")
        report_lines.append("   - Lower memory footprint")
        report_lines.append("   - Ideal for high-throughput applications")
        report_lines.append("")
        
        report_lines.append("\n" + "=" * 100)
        report_lines.append("END OF REPORT")
        report_lines.append("=" * 100)
        
        return "\n".join(report_lines)

def main():
    """Main test execution"""
    print("=" * 100)
    print("PHASE 6: COMPREHENSIVE VECTOR DATABASE TESTING")
    print(f"Started: {TEST_START_TIME.isoformat()}")
    print(f"Execution ID: {EXECUTION_ID}")
    print("=" * 100)
    
    tester = VectorDBTester()
    
    try:
        # Run comprehensive tests
        tester.log("\n🚀 Starting comprehensive vector database tests...", "INFO")
        
        # Individual database tests
        tester.log("\n" + "=" * 80, "INFO")
        tester.log("CHROMADB TESTS", "INFO")
        tester.log("=" * 80, "INFO")
        tester.test_chromadb_collection_creation()
        tester.test_chromadb_embedding_insertion(100, 384)
        tester.test_chromadb_embedding_insertion(1000, 384)
        tester.test_chromadb_similarity_search(10, 384)
        tester.test_chromadb_update_delete()
        
        tester.log("\n" + "=" * 80, "INFO")
        tester.log("QDRANT TESTS", "INFO")
        tester.log("=" * 80, "INFO")
        tester.test_qdrant_collection_creation(384)
        tester.test_qdrant_point_insertion(100, 384)
        tester.test_qdrant_point_insertion(1000, 384)
        tester.test_qdrant_search_performance(10, 384)
        tester.test_qdrant_filtering(384)
        
        tester.log("\n" + "=" * 80, "INFO")
        tester.log("FAISS TESTS", "INFO")
        tester.log("=" * 80, "INFO")
        tester.test_faiss_index_creation(768)
        tester.test_faiss_vector_operations(100, 768)
        tester.test_faiss_vector_operations(1000, 768)
        tester.test_faiss_search_performance(10, 768)
        
        # Performance comparison
        tester.run_performance_comparison()
        
        # Generate report
        tester.log("\n📊 Generating performance report...", "INFO")
        report = tester.generate_performance_report()
        
        # Save report
        report_filename = f"VECTOR_DB_PERFORMANCE_REPORT_{TEST_START_TIME.strftime('%Y%m%d_%H%M%S')}.txt"
        report_path = f"/opt/sutazaiapp/{report_filename}"
        
        with open(report_path, 'w') as f:
            f.write(report)
        
        tester.log(f"✅ Report saved to: {report_path}", "SUCCESS")
        
        # Print report to console
        print("\n" + report)
        
        # Save metrics as JSON
        metrics_filename = f"vector_db_metrics_{TEST_START_TIME.strftime('%Y%m%d_%H%M%S')}.json"
        metrics_path = f"/opt/sutazaiapp/{metrics_filename}"
        
        with open(metrics_path, 'w') as f:
            json.dump([asdict(m) for m in tester.metrics], f, indent=2)
        
        tester.log(f"✅ Metrics saved to: {metrics_path}", "SUCCESS")
        
        # Test summary
        total_tests = len(tester.metrics)
        successful_tests = sum(1 for m in tester.metrics if m.success)
        failed_tests = total_tests - successful_tests
        
        print("\n" + "=" * 100)
        print("TEST SUMMARY")
        print("=" * 100)
        print(f"Total Tests: {total_tests}")
        print(f"Successful: {successful_tests} ({(successful_tests/total_tests)*100:.1f}%)")
        print(f"Failed: {failed_tests} ({(failed_tests/total_tests)*100:.1f}%)")
        print("=" * 100)
        
        end_time = datetime.now(timezone.utc)
        duration = (end_time - TEST_START_TIME).total_seconds()
        print(f"\nCompleted: {end_time.isoformat()}")
        print(f"Total Duration: {duration:.2f} seconds")
        print("=" * 100)
        
        return 0 if failed_tests == 0 else 1
        
    except Exception as e:
        tester.log(f"❌ Fatal error during testing: {e}", "ERROR")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())
