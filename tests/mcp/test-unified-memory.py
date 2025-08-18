#!/usr/bin/env python3
"""
Unified Memory Service Test Suite
Tests consolidated memory service functionality
"""

import asyncio
import json
import time
from typing import Dict, Any
import httpx
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UnifiedMemoryTester:
    """Test suite for unified memory service"""
    
    def __init__(self, base_url: str = "http://localhost:3009"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
    
    async def test_store_memory(self) -> bool:
        """Test memory storage functionality"""
        test_data = {
            "key": "test_unified_store",
            "content": "This is a test memory for the unified service",
            "namespace": "test",
            "tags": ["test", "unified", "memory"],
            "importance_level": 8,
            "ttl": 3600
        }
        
        try:
            response = await self.client.post(f"{self.base_url}/memory/store", json=test_data)
            result = response.json()
            
            assert response.status_code == 200
            assert result["success"] is True
            assert "context_id" in result
            
            logger.info("✅ Store memory test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Store memory test failed: {e}")
            return False
    
    async def test_retrieve_memory(self) -> bool:
        """Test memory retrieval functionality"""
        try:
            response = await self.client.get(
                f"{self.base_url}/memory/retrieve/test_unified_store",
                params={"namespace": "test"}
            )
            result = response.json()
            
            assert response.status_code == 200
            assert result["success"] is True
            assert result["data"]["key"] == "test_unified_store"
            assert result["data"]["namespace"] == "test"
            assert "unified" in result["data"]["tags"]
            
            logger.info("✅ Retrieve memory test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Retrieve memory test failed: {e}")
            return False
    
    async def test_search_memory(self) -> bool:
        """Test memory search functionality"""
        try:
            response = await self.client.get(
                f"{self.base_url}/memory/search",
                params={"query": "unified", "namespace": "test", "limit": 5}
            )
            result = response.json()
            
            assert response.status_code == 200
            assert result["success"] is True
            assert len(result["data"]["results"]) > 0
            assert any("unified" in str(r).lower() for r in result["data"]["results"])
            
            logger.info("✅ Search memory test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Search memory test failed: {e}")
            return False
    
    async def test_memory_stats(self) -> bool:
        """Test memory statistics functionality"""
        try:
            response = await self.client.get(f"{self.base_url}/memory/stats")
            result = response.json()
            
            assert response.status_code == 200
            assert result["success"] is True
            assert "total_memories" in result["data"]
            assert "namespaces" in result["data"]
            assert result["data"]["total_memories"] > 0
            
            logger.info("✅ Memory stats test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Memory stats test failed: {e}")
            return False
    
    async def test_delete_memory(self) -> bool:
        """Test memory deletion functionality"""
        try:
            response = await self.client.delete(
                f"{self.base_url}/memory/delete/test_unified_store",
                params={"namespace": "test"}
            )
            result = response.json()
            
            assert response.status_code == 200
            assert result["success"] is True
            assert result["data"]["deleted"] is True
            
            logger.info("✅ Delete memory test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Delete memory test failed: {e}")
            return False
    
    async def test_health_check(self) -> bool:
        """Test service health check"""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            result = response.json()
            
            assert response.status_code == 200
            assert result["status"] == "healthy"
            assert result["service"] == "unified-memory"
            
            logger.info("✅ Health check test passed")
            return True
        except Exception as e:
            logger.error(f"❌ Health check test failed: {e}")
            return False
    
    async def test_performance(self) -> Dict[str, Any]:
        """Test performance characteristics"""
        results = {
            "store_latency": [],
            "retrieve_latency": [],
            "search_latency": []
        }
        
        # Test multiple operations for latency measurement
        for i in range(10):
            # Store performance
            start_time = time.time()
            await self.client.post(f"{self.base_url}/memory/store", json={
                "key": f"perf_test_{i}",
                "content": f"Performance test content {i}",
                "namespace": "performance",
                "importance_level": 5
            })
            results["store_latency"].append((time.time() - start_time) * 1000)
            
            # Retrieve performance
            start_time = time.time()
            await self.client.get(
                f"{self.base_url}/memory/retrieve/perf_test_{i}",
                params={"namespace": "performance"}
            )
            results["retrieve_latency"].append((time.time() - start_time) * 1000)
            
            # Search performance
            start_time = time.time()
            await self.client.get(
                f"{self.base_url}/memory/search",
                params={"query": f"test {i}", "namespace": "performance"}
            )
            results["search_latency"].append((time.time() - start_time) * 1000)
        
        # Calculate averages
        avg_results = {
            "avg_store_latency_ms": sum(results["store_latency"]) / len(results["store_latency"]),
            "avg_retrieve_latency_ms": sum(results["retrieve_latency"]) / len(results["retrieve_latency"]),
            "avg_search_latency_ms": sum(results["search_latency"]) / len(results["search_latency"]),
            "max_store_latency_ms": max(results["store_latency"]),
            "max_retrieve_latency_ms": max(results["retrieve_latency"]),
            "max_search_latency_ms": max(results["search_latency"])
        }
        
        logger.info(f"📊 Performance Results: {avg_results}")
        return avg_results
    
    async def run_all_tests(self) -> Dict[str, Any]:
        """Run complete test suite"""
        logger.info("🚀 Starting Unified Memory Service Test Suite")
        
        test_methods = [
            ("Health Check", self.test_health_check),
            ("Store Memory", self.test_store_memory),
            ("Retrieve Memory", self.test_retrieve_memory),
            ("Search Memory", self.test_search_memory),
            ("Memory Stats", self.test_memory_stats),
            ("Delete Memory", self.test_delete_memory),
        ]
        
        results = {
            "total_tests": len(test_methods),
            "passed": 0,
            "failed": 0,
            "test_details": []
        }
        
        for test_name, test_method in test_methods:
            logger.info(f"Running test: {test_name}")
            passed = await test_method()
            
            results["test_details"].append({
                "name": test_name,
                "passed": passed
            })
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
        
        # Run performance tests
        logger.info("Running performance tests...")
        performance_results = await self.test_performance()
        results["performance"] = performance_results
        
        # Calculate success rate
        results["success_rate"] = (results["passed"] / results["total_tests"]) * 100
        
        # Test summary
        logger.info(f"""
🎯 Test Suite Complete!
Total Tests: {results['total_tests']}
Passed: {results['passed']}
Failed: {results['failed']}
Success Rate: {results['success_rate']:.1f}%

Performance Summary:
- Average Store Latency: {performance_results['avg_store_latency_ms']:.1f}ms
- Average Retrieve Latency: {performance_results['avg_retrieve_latency_ms']:.1f}ms
- Average Search Latency: {performance_results['avg_search_latency_ms']:.1f}ms
        """)
        
        return results
    
    async def cleanup(self):
        """Cleanup test data and close client"""
        try:
            # Clean up performance test data
            for i in range(10):
                await self.client.delete(
                    f"{self.base_url}/memory/delete/perf_test_{i}",
                    params={"namespace": "performance"}
                )
        except:
            pass  # Ignore cleanup errors
        
        await self.client.aclose()

async def main():
    """Main test runner"""
    tester = UnifiedMemoryTester()
    
    try:
        results = await tester.run_all_tests()
        
        # Save results to file
        with open("/tmp/unified-memory-test-results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        # Exit with appropriate code
        if results["success_rate"] == 100:
            logger.info("🎉 All tests passed!")
            exit(0)
        else:
            logger.error(f"❌ {results['failed']} tests failed")
            exit(1)
    
    finally:
        await tester.cleanup()

if __name__ == "__main__":
    asyncio.run(main())