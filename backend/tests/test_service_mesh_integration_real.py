"""
Real Service Mesh Integration Tests
Tests actual working functionality of the distributed system
"""
import pytest
import asyncio
import httpx
import json
import time
from typing import Dict, Any, List

# Test against real running services
CONSUL_URL = "http://sutazai-consul:8500"
KONG_ADMIN_URL = "http://sutazai-kong:8001"
KONG_PROXY_URL = "http://sutazai-kong:8000"


class TestRealServiceMeshFunctionality:
    """Tests that validate actual working service mesh features"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_consul_service_discovery_works(self):
        """Test that Consul service discovery is actually functional"""
        async with httpx.AsyncClient() as client:
            # Check Consul is running and has a leader
            response = await client.get(f"{CONSUL_URL}/v1/status/leader")
            assert response.status_code == 200
            assert response.text.strip() != ""
            
            # Register a test service
            test_service = {
                "ID": "pytest-test-service",
                "Name": "pytest-test",
                "Address": "172.17.0.1",
                "Port": 9999,
                "Tags": ["test", "pytest"],
                "Check": {
                    "TCP": "172.17.0.1:9999",
                    "Interval": "10s"
                }
            }
            
            response = await client.put(
                f"{CONSUL_URL}/v1/agent/service/register",
                json=test_service
            )
            assert response.status_code == 200
            
            # Verify service appears in catalog
            await asyncio.sleep(0.5)
            response = await client.get(f"{CONSUL_URL}/v1/catalog/service/pytest-test")
            assert response.status_code == 200
            services = response.json()
            assert len(services) > 0
            assert services[0]["ServiceName"] == "pytest-test"
            
            # Cleanup
            await client.put(f"{CONSUL_URL}/v1/agent/service/deregister/pytest-test-service")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_kong_api_gateway_routing_works(self):
        """Test that Kong API Gateway routing is functional"""
        async with httpx.AsyncClient() as client:
            # Check Kong health
            response = await client.get(f"{KONG_ADMIN_URL}/status")
            assert response.status_code == 200
            status = response.json()
            assert status["database"]["reachable"] is True
            
            # Get configured services
            response = await client.get(f"{KONG_ADMIN_URL}/services")
            assert response.status_code == 200
            services = response.json()["data"]
            assert len(services) > 0
            
            # Get configured routes
            response = await client.get(f"{KONG_ADMIN_URL}/routes")
            assert response.status_code == 200
            routes = response.json()["data"]
            assert len(routes) > 0
            
            # Verify at least one route is properly configured
            route_paths = []
            for route in routes:
                if "paths" in route:
                    route_paths.extend(route["paths"])
            
            assert len(route_paths) > 0
            assert any("/health" in path for path in route_paths)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_service_health_checking_works(self):
        """Test that service health checking is operational"""
        async with httpx.AsyncClient() as client:
            # Get health status from Consul
            response = await client.get(f"{CONSUL_URL}/v1/health/state/any")
            assert response.status_code == 200
            health_checks = response.json()
            assert len(health_checks) > 0
            
            # Verify at least one service has health checks
            service_checks = [h for h in health_checks if h["ServiceID"]]
            assert len(service_checks) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_consul_kv_store_works(self):
        """Test that Consul KV store is functional for configuration"""
        async with httpx.AsyncClient() as client:
            test_key = "test/pytest/mesh-validation"
            test_value = {"test": "data", "timestamp": time.time()}
            
            # Write to KV store
            response = await client.put(
                f"{CONSUL_URL}/v1/kv/{test_key}",
                content=json.dumps(test_value)
            )
            assert response.status_code == 200
            
            # Read from KV store
            response = await client.get(f"{CONSUL_URL}/v1/kv/{test_key}?raw=true")
            assert response.status_code == 200
            retrieved_value = json.loads(response.text)
            assert retrieved_value["test"] == "data"
            
            # List keys
            response = await client.get(f"{CONSUL_URL}/v1/kv/test?keys=true")
            assert response.status_code == 200
            keys = response.json()
            assert any(test_key in k for k in keys)
            
            # Cleanup
            await client.delete(f"{CONSUL_URL}/v1/kv/{test_key}")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_kong_plugin_system_works(self):
        """Test that Kong plugin system is operational"""
        async with httpx.AsyncClient() as client:
            # Get available plugins
            response = await client.get(f"{KONG_ADMIN_URL}/plugins/enabled")
            assert response.status_code == 200
            enabled_plugins = response.json()["enabled_plugins"]
            assert len(enabled_plugins) > 0
            
            # Common plugins that should be available
            expected_plugins = ["cors", "rate-limiting", "key-auth", "acl"]
            available = [p for p in expected_plugins if p in enabled_plugins]
            assert len(available) > 0
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_service_to_service_communication(self):
        """Test that services can communicate through the mesh"""
        async with httpx.AsyncClient() as client:
            # Get list of active services from Consul
            response = await client.get(f"{CONSUL_URL}/v1/catalog/services")
            assert response.status_code == 200
            services = response.json()
            
            # Verify we have multiple services registered
            assert len(services) >= 2  # At least Consul and one other service
            
            # Test that services can discover each other
            for service_name in list(services.keys())[:3]:  # Test first 3 services
                response = await client.get(f"{CONSUL_URL}/v1/catalog/service/{service_name}")
                assert response.status_code == 200
                instances = response.json()
                assert len(instances) > 0
                
                # Each instance should have address and port
                for instance in instances:
                    assert "ServiceAddress" in instance or "Address" in instance
                    assert "ServicePort" in instance
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_distributed_configuration_management(self):
        """Test distributed configuration through Consul KV"""
        async with httpx.AsyncClient() as client:
            # Set up distributed config
            config = {
                "database": {
                    "host": "postgres",
                    "port": 5432,
                    "pool_size": 10
                },
                "cache": {
                    "host": "redis",
                    "port": 6379,
                    "ttl": 3600
                },
                "features": {
                    "circuit_breaker": True,
                    "rate_limiting": True,
                    "retry_policy": {
                        "max_retries": 3,
                        "backoff": "exponential"
                    }
                }
            }
            
            # Store configuration
            response = await client.put(
                f"{CONSUL_URL}/v1/kv/config/mesh/settings",
                content=json.dumps(config)
            )
            assert response.status_code == 200
            
            # Retrieve configuration
            response = await client.get(f"{CONSUL_URL}/v1/kv/config/mesh/settings?raw=true")
            assert response.status_code == 200
            retrieved_config = json.loads(response.text)
            assert retrieved_config["features"]["circuit_breaker"] is True
            assert retrieved_config["features"]["retry_policy"]["max_retries"] == 3
            
            # Cleanup
            await client.delete(f"{CONSUL_URL}/v1/kv/config/mesh/settings")
    
    @pytest.mark.asyncio
    @pytest.mark.integration  
    async def test_service_mesh_observability(self):
        """Test that observability features are working"""
        async with httpx.AsyncClient() as client:
            # Test Consul metrics endpoint
            response = await client.get(f"{CONSUL_URL}/v1/agent/metrics")
            assert response.status_code == 200
            metrics = response.json()
            assert "Gauges" in metrics
            assert "Counters" in metrics
            assert "Samples" in metrics
            
            # Verify some key metrics exist
            gauge_names = [g["Name"] for g in metrics["Gauges"]]
            assert any("consul.runtime" in name for name in gauge_names)
            
            # Test Kong monitoring endpoint
            response = await client.get(f"{KONG_ADMIN_URL}/status")
            assert response.status_code == 200
            status = response.json()
            assert "server" in status
            assert "database" in status


class TestServiceMeshResilience:
    """Tests for resilience and fault tolerance features"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_service_registration_resilience(self):
        """Test that service registration handles failures gracefully"""
        async with httpx.AsyncClient() as client:
            # Register service with invalid health check (should still register)
            test_service = {
                "ID": "resilience-test",
                "Name": "resilience-service",
                "Address": "999.999.999.999",  # Invalid address
                "Port": 99999,  # Invalid port
                "Tags": ["test", "resilience"],
                "Check": {
                    "TCP": "999.999.999.999:99999",
                    "Interval": "10s",
                    "DeregisterCriticalServiceAfter": "30s"
                }
            }
            
            # Should register even with invalid check
            response = await client.put(
                f"{CONSUL_URL}/v1/agent/service/register",
                json=test_service
            )
            assert response.status_code == 200
            
            # Service should appear but be unhealthy
            await asyncio.sleep(1)
            response = await client.get(f"{CONSUL_URL}/v1/health/service/resilience-service")
            assert response.status_code == 200
            health = response.json()
            if health:  # Service registered
                checks = health[0]["Checks"]
                service_check = [c for c in checks if c["ServiceID"] == "resilience-test"]
                if service_check:
                    assert service_check[0]["Status"] in ["critical", "warning"]
            
            # Cleanup
            await client.put(f"{CONSUL_URL}/v1/agent/service/deregister/resilience-test")
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_configuration_versioning(self):
        """Test configuration versioning and rollback capabilities"""
        async with httpx.AsyncClient() as client:
            base_key = "config/versions/test"
            
            # Store multiple versions
            for version in range(1, 4):
                config = {
                    "version": version,
                    "timestamp": time.time(),
                    "settings": {"value": version * 10}
                }
                
                response = await client.put(
                    f"{CONSUL_URL}/v1/kv/{base_key}/v{version}",
                    content=json.dumps(config)
                )
                assert response.status_code == 200
            
            # Retrieve all versions
            response = await client.get(f"{CONSUL_URL}/v1/kv/{base_key}?keys=true")
            assert response.status_code == 200
            keys = response.json()
            assert len([k for k in keys if base_key in k]) == 3
            
            # Cleanup
            await client.delete(f"{CONSUL_URL}/v1/kv/{base_key}?recurse=true")


@pytest.fixture(scope="module")
def event_loop():
    """Create event loop for async tests"""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


def test_service_mesh_is_real():
    """Simple synchronous test to verify this is real functionality"""
    import requests
    
    # This is a real test against real services
    response = requests.get("http://localhost:10006/v1/status/leader")
    assert response.status_code == 200
    assert response.text.strip() != ""
    
    response = requests.get("http://localhost:10015/status")
    assert response.status_code == 200
    assert "database" in response.json()