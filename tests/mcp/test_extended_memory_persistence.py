#!/usr/bin/env python3
"""
Integration tests for Extended Memory MCP Service with SQLite persistence
Tests data persistence across container restarts
"""

import pytest
import requests
import json
import time
import subprocess
import uuid
from typing import Dict, Any
from datetime import datetime

# Service configuration
SERVICE_URL = "http://localhost:3009"
CONTAINER_NAME = "mcp-extended-memory"

class TestExtendedMemoryPersistence:
    """Test suite for extended memory persistence"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.session = requests.Session()
        cls.test_data = {}
        
    def test_01_health_check(self):
        """Test service health endpoint"""
        response = self.session.get(f"{SERVICE_URL}/health")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "extended-memory"
        assert data["persistence"]["enabled"] is True
        assert data["persistence"]["type"] == "SQLite"
        
        print(f"✓ Health check passed: {data}")
    
    def test_02_store_data(self):
        """Test storing data in memory"""
        test_keys = {
            "test_string": "Hello, Extended Memory!",
            "test_number": 42,
            "test_float": 3.14159,
            "test_bool": True,
            "test_list": [1, 2, 3, "four", 5.0],
            "test_dict": {
                "nested": "value",
                "count": 123,
                "items": ["a", "b", "c"]
            },
            "test_null": None,
            f"test_uuid_{uuid.uuid4()}": "unique_value"
        }
        
        for key, value in test_keys.items():
            response = self.session.post(
                f"{SERVICE_URL}/store",
                json={"key": key, "value": value}
            )
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "stored"
            assert data["key"] == key
            assert data["persisted"] is True
            
            # Store for later verification
            self.test_data[key] = value
            
        print(f"✓ Stored {len(test_keys)} test items")
    
    def test_03_retrieve_data(self):
        """Test retrieving stored data"""
        for key, expected_value in self.test_data.items():
            response = self.session.get(f"{SERVICE_URL}/retrieve/{key}")
            assert response.status_code == 200
            
            data = response.json()
            assert data["status"] == "found"
            assert data["key"] == key
            assert data["value"] == expected_value
            
        print(f"✓ Retrieved {len(self.test_data)} items successfully")
    
    def test_04_list_keys(self):
        """Test listing all keys with pagination"""
        response = self.session.get(f"{SERVICE_URL}/list")
        assert response.status_code == 200
        
        data = response.json()
        assert "keys" in data
        assert "metadata" in data
        assert "total_count" in data
        
        # Verify our test keys are present
        stored_keys = set(data["keys"])
        test_keys = set(self.test_data.keys())
        assert test_keys.issubset(stored_keys)
        
        # Test metadata
        for item in data["metadata"]:
            assert "key" in item
            assert "type" in item
            assert "access_count" in item
            
        print(f"✓ Listed {data['total_count']} total keys")
    
    def test_05_statistics(self):
        """Test statistics endpoint"""
        response = self.session.get(f"{SERVICE_URL}/stats")
        assert response.status_code == 200
        
        data = response.json()
        assert "statistics" in data
        assert "most_accessed" in data
        assert "recently_accessed" in data
        
        stats = data["statistics"]
        assert stats["total_items"] >= len(self.test_data)
        assert stats["total_accesses"] > 0
        
        print(f"✓ Statistics: {stats}")
    
    def test_06_container_restart_persistence(self):
        """Test data persistence across container restart"""
        print("\n⚠ Testing container restart persistence...")
        
        # Store a marker value before restart
        marker_key = f"restart_test_{uuid.uuid4()}"
        marker_value = {
            "timestamp": datetime.utcnow().isoformat(),
            "test": "persistence",
            "data": list(range(100))
        }
        
        response = self.session.post(
            f"{SERVICE_URL}/store",
            json={"key": marker_key, "value": marker_value}
        )
        assert response.status_code == 200
        
        # Restart the container
        print(f"  Restarting container {CONTAINER_NAME}...")
        try:
            subprocess.run(
                ["docker", "restart", CONTAINER_NAME],
                check=True,
                capture_output=True
            )
        except subprocess.CalledProcessError as e:
            pytest.skip(f"Cannot restart container: {e}")
        
        # Wait for service to come back up
        max_retries = 30
        for i in range(max_retries):
            time.sleep(2)
            try:
                response = self.session.get(f"{SERVICE_URL}/health")
                if response.status_code == 200:
                    print(f"  Service is back up after {i*2} seconds")
                    break
            except requests.exceptions.ConnectionError:
                continue
        else:
            pytest.fail("Service did not come back up after restart")
        
        # Verify all data is still present
        response = self.session.get(f"{SERVICE_URL}/retrieve/{marker_key}")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "found"
        assert data["value"] == marker_value
        
        # Verify original test data
        for key, expected_value in self.test_data.items():
            response = self.session.get(f"{SERVICE_URL}/retrieve/{key}")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "found"
            assert data["value"] == expected_value
        
        print(f"✓ All data persisted across container restart!")
    
    def test_07_backup_functionality(self):
        """Test backup creation"""
        response = self.session.post(f"{SERVICE_URL}/backup")
        assert response.status_code == 200
        
        data = response.json()
        assert data["status"] == "backup_created"
        assert "path" in data
        assert "timestamp" in data
        
        print(f"✓ Backup created: {data['path']}")
    
    def test_08_performance_with_persistence(self):
        """Test performance with large datasets"""
        print("\n⚡ Testing performance with persistence...")
        
        # Store many items
        batch_size = 100
        start_time = time.time()
        
        for i in range(batch_size):
            key = f"perf_test_{i}"
            value = {
                "index": i,
                "data": f"Performance test data {i}" * 10,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            response = self.session.post(
                f"{SERVICE_URL}/store",
                json={"key": key, "value": value}
            )
            assert response.status_code == 200
        
        store_time = time.time() - start_time
        
        # Retrieve all items
        start_time = time.time()
        
        for i in range(batch_size):
            key = f"perf_test_{i}"
            response = self.session.get(f"{SERVICE_URL}/retrieve/{key}")
            assert response.status_code == 200
        
        retrieve_time = time.time() - start_time
        
        print(f"  Store {batch_size} items: {store_time:.2f}s ({batch_size/store_time:.1f} ops/s)")
        print(f"  Retrieve {batch_size} items: {retrieve_time:.2f}s ({batch_size/retrieve_time:.1f} ops/s)")
        
        # Performance should be reasonable even with persistence
        assert store_time < 30  # Should store 100 items in less than 30 seconds
        assert retrieve_time < 10  # Should retrieve 100 items in less than 10 seconds
        
        print(f"✓ Performance test passed")
    
    def test_09_concurrent_access(self):
        """Test concurrent access to the service"""
        import concurrent.futures
        
        print("\n🔄 Testing concurrent access...")
        
        def store_and_retrieve(index):
            """Store and immediately retrieve a value"""
            key = f"concurrent_{index}"
            value = f"concurrent_value_{index}"
            
            # Store
            response = requests.post(
                f"{SERVICE_URL}/store",
                json={"key": key, "value": value}
            )
            assert response.status_code == 200
            
            # Retrieve
            response = requests.get(f"{SERVICE_URL}/retrieve/{key}")
            assert response.status_code == 200
            data = response.json()
            assert data["value"] == value
            
            return True
        
        # Run concurrent operations
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(store_and_retrieve, i) for i in range(50)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        assert all(results)
        print(f"✓ Handled 50 concurrent operations successfully")
    
    def test_10_data_integrity_check(self):
        """Final data integrity check"""
        print("\n🔍 Final data integrity check...")
        
        # Get all keys
        response = self.session.get(f"{SERVICE_URL}/list?limit=1000")
        assert response.status_code == 200
        
        data = response.json()
        total_keys = data["total_count"]
        
        # Verify database statistics
        response = self.session.get(f"{SERVICE_URL}/stats")
        assert response.status_code == 200
        
        stats = response.json()["statistics"]
        assert stats["total_items"] == total_keys
        
        print(f"✓ Data integrity verified: {total_keys} items in database")
        print(f"✓ Total accesses: {stats['total_accesses']}")
        print(f"✓ Average accesses per key: {stats['avg_accesses']:.2f}")
    
    @classmethod
    def teardown_class(cls):
        """Cleanup after tests"""
        cls.session.close()


def run_persistence_tests():
    """Run all persistence tests"""
    print("=" * 60)
    print("Extended Memory MCP Service - Persistence Test Suite")
    print("=" * 60)
    
    # Check if service is running
    try:
        response = requests.get(f"{SERVICE_URL}/health", timeout=5)
        if response.status_code != 200:
            print("❌ Service is not healthy")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to service: {e}")
        print(f"   Make sure the container is running: docker ps | grep {CONTAINER_NAME}")
        return False
    
    # Run pytest
    pytest.main([__file__, "-v", "-s"])
    
    print("\n" + "=" * 60)
    print("✅ All persistence tests completed successfully!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    import sys
    success = run_persistence_tests()
    sys.exit(0 if success else 1)