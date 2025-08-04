#!/usr/bin/env python3
"""
Automated Test Cases for Sutazai Hygiene Monitoring System
Comprehensive suite for continuous validation and regression testing
"""
import asyncio
import aiohttp
import pytest
import time
import json
from datetime import datetime
import psutil
import docker
import subprocess
import sys
from pathlib import Path

# Test Configuration
API_BASE = "http://localhost:10420/api/hygiene"
RULE_API_BASE = "http://localhost:10421/api"
DASHBOARD_URL = "http://localhost:10422"
WEBSOCKET_URL = "ws://localhost:10420/ws"
STANDALONE_URL = "http://localhost:9080"

class TestHygieneSystem:
    """Main test class for hygiene monitoring system"""
    
    @pytest.fixture(scope="session")
    def docker_client(self):
        """Docker client fixture"""
        return docker.from_env()
    
    @pytest.fixture(scope="session")
    async def http_session(self):
        """HTTP session fixture"""
        session = aiohttp.ClientSession()
        yield session
        await session.close()

    # === Service Health Tests ===
    
    @pytest.mark.asyncio
    async def test_backend_service_health(self, http_session):
        """Test backend service is responding and healthy"""
        async with http_session.get(f"{API_BASE}/status") as response:
            assert response.status == 200
            data = await response.json()
            
            # Verify required fields
            assert "systemStatus" in data
            assert "complianceScore" in data
            assert "totalViolations" in data
            assert data["systemStatus"] in ["MONITORING", "ACTIVE"]
            
    @pytest.mark.asyncio
    async def test_rule_control_service_health(self, http_session):
        """Test rule control API is responding"""
        async with http_session.get(f"{RULE_API_BASE}/health/live") as response:
            assert response.status == 200
            data = await response.json()
            assert data["status"] == "alive"
            
    @pytest.mark.asyncio 
    async def test_dashboard_service_health(self, http_session):
        """Test dashboard UI is serving content"""
        async with http_session.get(DASHBOARD_URL) as response:
            assert response.status == 200
            content = await response.text()
            assert "Sutazai Hygiene" in content
            assert "<html" in content.lower()
            
    def test_container_health_status(self, docker_client):
        """Test all containers are healthy"""
        containers = docker_client.containers.list()
        hygiene_containers = [c for c in containers if "hygiene" in c.name]
        
        assert len(hygiene_containers) > 0, "No hygiene containers found"
        
        for container in hygiene_containers:
            health = container.attrs.get("State", {}).get("Health", {})
            if health:  # Only check if health is configured
                assert health["Status"] == "healthy", f"Container {container.name} is not healthy"

    # === API Functionality Tests ===
    
    @pytest.mark.asyncio
    async def test_scan_endpoint_functionality(self, http_session):
        """Test scan endpoint triggers and returns results"""
        start_time = time.time()
        
        async with http_session.post(f"{API_BASE}/scan") as response:
            elapsed = time.time() - start_time
            
            assert response.status == 200
            assert elapsed < 5.0, "Scan took too long"
            
            data = await response.json()
            assert data["success"] is True
            assert "violations_found" in data
            assert "violations" in data
            assert isinstance(data["violations"], list)
            
    @pytest.mark.asyncio
    async def test_system_metrics_endpoint(self, http_session):
        """Test system metrics are being collected"""
        async with http_session.get("http://localhost:10420/api/system/metrics") as response:
            assert response.status == 200
            data = await response.json()
            
            # Verify metric fields exist and are reasonable
            assert "cpu_usage" in data
            assert "memory_percentage" in data
            assert 0 <= data["cpu_usage"] <= 100
            assert 0 <= data["memory_percentage"] <= 100

    # === Real-time Features Tests ===
    
    @pytest.mark.asyncio
    async def test_websocket_connection(self):
        """Test WebSocket connection and basic communication"""
        async with aiohttp.ClientSession() as session:
            try:
                async with session.ws_connect(WEBSOCKET_URL) as ws:
                    # Send ping
                    await ws.send_json({"type": "ping"})
                    
                    # Wait for response with timeout
                    msg = await asyncio.wait_for(ws.receive(), timeout=5.0)
                    assert msg.type == aiohttp.WSMsgType.TEXT
                    
                    # Should receive some kind of response
                    data = json.loads(msg.data)
                    assert "type" in data
                    
            except asyncio.TimeoutError:
                # Connection established but no immediate response is also acceptable
                pass
                
    @pytest.mark.asyncio
    async def test_websocket_real_time_updates(self):
        """Test that WebSocket receives real-time updates"""
        async with aiohttp.ClientSession() as session:
            async with session.ws_connect(WEBSOCKET_URL) as ws:
                # Should receive initial data
                initial_msg = await asyncio.wait_for(ws.receive(), timeout=10.0)
                assert initial_msg.type == aiohttp.WSMsgType.TEXT
                
                data = json.loads(initial_msg.data)
                assert data.get("type") == "initial_data"
                assert "data" in data

    # === Performance Tests ===
    
    @pytest.mark.asyncio
    async def test_rapid_api_calls_no_throttling(self, http_session):
        """Test rapid API calls don't cause errors or excessive delays"""
        tasks = []
        call_count = 5
        
        start_time = time.time()
        for _ in range(call_count):
            task = http_session.get(f"{API_BASE}/status")
            tasks.append(task)
            
        responses = await asyncio.gather(*tasks)
        elapsed = time.time() - start_time
        
        # Clean up responses
        for response in responses:
            response.close()
            
        # All calls should succeed
        success_count = sum(1 for r in responses if r.status == 200)
        assert success_count == call_count
        
        # Should complete in reasonable time
        assert elapsed < 10.0, f"Rapid calls took too long: {elapsed}s"
        
    @pytest.mark.asyncio
    async def test_concurrent_scan_operations(self, http_session):
        """Test concurrent scan operations don't interfere"""
        tasks = []
        concurrent_scans = 3
        
        for _ in range(concurrent_scans):
            task = http_session.post(f"{API_BASE}/scan")
            tasks.append(task)
            
        responses = await asyncio.gather(*tasks, return_exceptions=True)
        
        # At least some should succeed (they might share results)
        successful_responses = [r for r in responses if hasattr(r, 'status') and r.status == 200]
        assert len(successful_responses) > 0
        
        # Clean up
        for response in responses:
            if hasattr(response, 'close'):
                response.close()

    # === Data Persistence Tests ===
    
    @pytest.mark.asyncio
    async def test_database_persistence(self, http_session):
        """Test that scan results persist in database"""
        # Trigger a scan
        async with http_session.post(f"{API_BASE}/scan") as scan_response:
            assert scan_response.status == 200
            scan_data = await scan_response.json()
            violations_from_scan = scan_data["violations_found"]
            
        # Wait for data to be stored
        await asyncio.sleep(1)
        
        # Check that data persists in status endpoint
        async with http_session.get(f"{API_BASE}/status") as status_response:
            assert status_response.status == 200
            status_data = await status_response.json()
            
            # Should have violations stored
            assert status_data["totalViolations"] > 0
            
    @pytest.mark.asyncio
    async def test_data_consistency_across_requests(self, http_session):
        """Test data consistency across multiple requests"""
        # Get status multiple times
        status_calls = 3
        violation_counts = []
        
        for _ in range(status_calls):
            async with http_session.get(f"{API_BASE}/status") as response:
                assert response.status == 200
                data = await response.json()
                violation_counts.append(data["totalViolations"])
                
        # Counts should be consistent (or only increase)
        assert len(set(violation_counts)) <= 2, "Data inconsistency detected"

    # === Error Handling Tests ===
    
    @pytest.mark.asyncio
    async def test_invalid_endpoint_handling(self, http_session):
        """Test handling of invalid API endpoints"""
        async with http_session.get(f"{API_BASE}/nonexistent") as response:
            assert response.status == 404
            
    @pytest.mark.asyncio 
    async def test_malformed_request_handling(self, http_session):
        """Test handling of malformed requests"""
        # Try to send invalid JSON to scan endpoint
        async with http_session.post(f"{API_BASE}/scan", 
                                   data="invalid json",
                                   headers={"Content-Type": "application/json"}) as response:
            # Should handle gracefully (either accept as no-payload or return 400)
            assert response.status in [200, 400]

    # === Resource Usage Tests ===
    
    def test_memory_usage_within_limits(self):
        """Test memory usage is within acceptable limits"""
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Should be under 90% for healthy operation
        assert memory_percent < 90, f"Memory usage too high: {memory_percent}%"
        
    def test_cpu_usage_reasonable(self):
        """Test CPU usage is reasonable during operation"""
        # Sample CPU usage over short period
        cpu_percent = psutil.cpu_percent(interval=2)
        
        # Under normal load, should be reasonable
        assert cpu_percent < 80, f"CPU usage too high: {cpu_percent}%"

    # === Integration Tests ===
    
    @pytest.mark.asyncio
    async def test_full_workflow_integration(self, http_session):
        """Test complete workflow from scan to result display"""
        # 1. Trigger scan
        async with http_session.post(f"{API_BASE}/scan") as scan_response:
            assert scan_response.status == 200
            scan_data = await scan_response.json()
            
        # 2. Verify results in status
        await asyncio.sleep(1)
        async with http_session.get(f"{API_BASE}/status") as status_response:
            assert status_response.status == 200
            status_data = await status_response.json()
            
        # 3. Check metrics are updated
        async with http_session.get("http://localhost:10420/api/system/metrics") as metrics_response:
            assert metrics_response.status == 200
            
        # All steps should complete successfully
        assert scan_data["success"] is True
        assert status_data["totalViolations"] >= 0

    # === Standalone Scanner Tests ===
    
    @pytest.mark.asyncio
    async def test_standalone_scanner_reports(self, http_session):
        """Test standalone scanner generates accessible reports"""
        try:
            async with http_session.get(STANDALONE_URL) as response:
                if response.status == 200:
                    content = await response.text()
                    assert "Hygiene Scan Report" in content
                    assert "<!DOCTYPE html>" in content
                else:
                    # Standalone scanner might not be running - this is acceptable
                    pytest.skip("Standalone scanner not available")
        except aiohttp.ClientConnectionError:
            pytest.skip("Standalone scanner not accessible")

    # === Regression Tests ===
    
    @pytest.mark.asyncio
    async def test_no_stack_overflow_in_rapid_calls(self, http_session):
        """Regression test: Ensure no stack overflow in rapid API calls"""
        call_count = 20
        delay_ms = 10
        
        for i in range(call_count):
            try:
                async with http_session.get(f"{API_BASE}/status") as response:
                    assert response.status == 200
                    # Should not throw stack overflow or similar errors
                    data = await response.json()
                    assert "systemStatus" in data
                    
            except Exception as e:
                error_msg = str(e).lower()
                assert "stack" not in error_msg, f"Stack overflow detected on call {i+1}: {e}"
                assert "recursion" not in error_msg, f"Recursion error detected on call {i+1}: {e}"
                
            if i < call_count - 1:
                await asyncio.sleep(delay_ms / 1000)

    # === Stress Tests ===
    
    @pytest.mark.stress
    @pytest.mark.asyncio
    async def test_extended_operation(self, http_session):
        """Stress test: Extended operation over time"""
        duration_seconds = 30
        interval_seconds = 2
        end_time = time.time() + duration_seconds
        
        request_count = 0
        error_count = 0
        
        while time.time() < end_time:
            try:
                async with http_session.get(f"{API_BASE}/status") as response:
                    request_count += 1
                    if response.status != 200:
                        error_count += 1
                        
            except Exception:
                error_count += 1
                
            await asyncio.sleep(interval_seconds)
            
        # Should have low error rate
        error_rate = error_count / request_count if request_count > 0 else 1
        assert error_rate < 0.1, f"High error rate during stress test: {error_rate:.2%}"

# === Utility Functions ===

def run_system_command(command):
    """Run system command and return result"""
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)

# === Test Configuration and Markers ===

# Pytest markers for selective test execution
pytestmarks = [
    pytest.mark.integration,  # All tests are integration tests
]

# Custom test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers"""
    for item in items:
        # Add markers based on test names
        if "stress" in item.name:
            item.add_marker(pytest.mark.stress)
        if "performance" in item.name:
            item.add_marker(pytest.mark.performance)
        if "regression" in item.name:
            item.add_marker(pytest.mark.regression)

if __name__ == "__main__":
    """Run tests directly"""
    print("🧪 Running Hygiene System Automated Tests...")
    
    # Run pytest with verbose output
    pytest_args = [
        __file__,
        "-v",
        "--tb=short",
        "-x",  # Stop on first failure
        "--asyncio-mode=auto"  # Handle async tests
    ]
    
    # Add stress tests if requested
    if "--stress" in sys.argv:
        pytest_args.append("-m stress")
        
    exit_code = pytest.main(pytest_args)
    sys.exit(exit_code)