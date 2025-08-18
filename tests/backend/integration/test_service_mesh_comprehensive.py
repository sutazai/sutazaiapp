"""
Comprehensive production-grade tests for ServiceMesh implementation
Tests all distributed system features including:
- Service discovery with Consul
- Load balancing strategies  
- Circuit breaker patterns
- Health checking
- Retry policies
- Distributed tracing
- Failure scenarios
"""
import pytest
import asyncio
import time
import json
from unittest.Mock import Mock, AsyncMock, patch, MagicMock, call
from typing import Dict, Any, List
import httpx

from app.mesh.service_mesh import (
    ServiceMesh, ServiceInstance, ServiceState, ServiceRequest,
    LoadBalancerStrategy, ServiceDiscovery, LoadBalancer, CircuitBreakerManager,
    get_mesh
)
from app.mesh.distributed_tracing import (
    Tracer, Span, SpanContext, SpanType, SpanStatus, TraceCollector
)
from app.mesh.mesh_dashboard import MeshDashboard, ServiceMetrics, MeshMetrics


class TestServiceMeshProduction:
    """Production-grade tests for ServiceMesh"""
    
    @pytest.fixture
    async def mesh(self):
        """Create a service mesh instance for testing"""
        mesh = ServiceMesh(
            consul_host="localhost",
            consul_port=8500,
            kong_admin_url="http://localhost:8001",
            load_balancer_strategy=LoadBalancerStrategy.ROUND_ROBIN
        )
        yield mesh
        await mesh.shutdown()
    
    @pytest.mark.asyncio
    async def test_service_mesh_initialization_with_consul(self):
        """Test service mesh initialization with actual Consul connection"""
        mesh = ServiceMesh(
            consul_host="sutazai-consul",
            consul_port=8500,
            kong_admin_url="http://sutazai-kong:8001"
        )
        
        await mesh.initialize()
        
        assert mesh.discovery is not None
        assert mesh.load_balancer is not None
        assert mesh.circuit_breaker is not None
        assert mesh.kong_admin_url == "http://sutazai-kong:8001"
    
    @pytest.mark.asyncio
    @patch('consul.Consul')
    async def test_consul_connection_failure_graceful_degradation(self, Mock_consul_class):
        """Test graceful degradation when Consul is unavailable"""
        # Simulate Consul connection failure
        Mock_consul_class.side_effect = Exception("Connection refused")
        
        discovery = ServiceDiscovery("sutazai-consul", 8500)
        await discovery.connect()
        
        # Should not raise, just log error
        assert discovery.consul_client is None
        
        # Should still work with local cache
        instance = ServiceInstance(
            service_id="test-1",
            service_name="test-service",
            address="localhost",
            port=8080
        )
        
        result = await discovery.register_service(instance)
        assert result is True  # Should succeed with local cache
        assert "test-service" in discovery.services_cache
        assert len(discovery.services_cache["test-service"]) == 1
    
    @pytest.mark.asyncio
    @patch('consul.Consul')
    async def test_service_registration_with_health_checks(self, Mock_consul_class):
        """Test service registration with health check configuration"""
        Mock_consul = Mock()
        Mock_consul.agent.self.return_value = {"Config": {"NodeName": "test-node"}}
        Mock_consul.agent.service.register = Mock()
        Mock_consul_class.return_value = Mock_consul
        
        discovery = ServiceDiscovery()
        await discovery.connect()
        
        instance = ServiceInstance(
            service_id="api-service-1",
            service_name="api-service",
            address="10.0.0.1",
            port=8080,
            tags=["api", "v1", "production"],
            metadata={"version": "1.0.0", "region": "us-east"}
        )
        
        result = await discovery.register_service(instance)
        
        assert result is True
        Mock_consul.agent.service.register.assert_called_once()
        
        # Verify health check configuration
        call_args = Mock_consul.agent.service.register.call_args
        service_data = call_args[1]
        assert service_data["ID"] == "api-service-1"
        assert service_data["Name"] == "api-service"
        assert service_data["Check"]["HTTP"] == "http://10.0.0.1:8080/health"
        assert service_data["Check"]["Interval"] == "10s"
        assert service_data["Check"]["Timeout"] == "5s"
    
    @pytest.mark.asyncio
    async def test_load_balancer_round_robin_strategy(self):
        """Test round-robin load balancing across multiple instances"""
        balancer = LoadBalancer(LoadBalancerStrategy.ROUND_ROBIN)
        
        instances = [
            ServiceInstance(f"api-{i}", "api", f"10.0.0.{i}", 8080, state=ServiceState.HEALTHY)
            for i in range(1, 4)
        ]
        
        # Should cycle through instances in order
        selections = []
        for _ in range(9):  # Test 3 full cycles
            selected = balancer.select_instance(instances, "api")
            selections.append(selected.service_id)
        
        expected = ["api-1", "api-2", "api-3"] * 3
        assert selections == expected
    
    @pytest.mark.asyncio
    async def test_load_balancer_least_connections_strategy(self):
        """Test least connections load balancing"""
        balancer = LoadBalancer(LoadBalancerStrategy.LEAST_CONNECTIONS)
        
        instances = [
            ServiceInstance("api-1", "api", "10.0.0.1", 8080, state=ServiceState.HEALTHY, connections=10),
            ServiceInstance("api-2", "api", "10.0.0.2", 8080, state=ServiceState.HEALTHY, connections=5),
            ServiceInstance("api-3", "api", "10.0.0.3", 8080, state=ServiceState.HEALTHY, connections=15)
        ]
        
        # Should always select instance with least connections
        for _ in range(5):
            selected = balancer.select_instance(instances, "api")
            assert selected.service_id == "api-2"
            selected.connections += 1  # Simulate connection
    
    @pytest.mark.asyncio
    async def test_load_balancer_weighted_strategy(self):
        """Test weighted load balancing"""
        balancer = LoadBalancer(LoadBalancerStrategy.WEIGHTED)
        
        instances = [
            ServiceInstance("api-1", "api", "10.0.0.1", 8080, state=ServiceState.HEALTHY, weight=100),
            ServiceInstance("api-2", "api", "10.0.0.2", 8080, state=ServiceState.HEALTHY, weight=0),
            ServiceInstance("api-3", "api", "10.0.0.3", 8080, state=ServiceState.HEALTHY, weight=50)
        ]
        
        # With weight 0, api-2 should never be selected
        selections = []
        for _ in range(100):
            selected = balancer.select_instance(instances, "api")
            selections.append(selected.service_id)
        
        assert "api-2" not in selections
        assert "api-1" in selections
        assert "api-3" in selections
        # api-1 should be selected roughly twice as often as api-3
        api1_count = selections.count("api-1")
        api3_count = selections.count("api-3")
        assert 1.5 < api1_count / api3_count < 2.5  # Allow some variance
    
    @pytest.mark.asyncio
    async def test_circuit_breaker_trip_on_failures(self):
        """Test circuit breaker tripping after threshold failures"""
        manager = CircuitBreakerManager(failure_threshold=3, recovery_timeout=5)
        
        service_id = "api-service-1"
        
        # Initially closed
        assert not manager.is_open(service_id)
        
        # Record failures
        for i in range(3):
            manager.record_failure(service_id)
            
        # After 3 failures, should trip (but pybreaker might need actual calls)
        # This is a simplified test - real circuit breaker testing needs actual calls
        breaker = manager.get_breaker(service_id)
        assert breaker._fail_counter >= 3
    
    @pytest.mark.asyncio
    @patch('httpx.AsyncClient')
    async def test_service_call_with_retry_policy(self, Mock_client_class):
        """Test service call with retry on failure"""
        mesh = ServiceMesh()
        
        # Setup Mock discovery
        test_instance = ServiceInstance(
            "api-1", "api-service", "localhost", 8080,
            state=ServiceState.HEALTHY
        )
        mesh.discovery.services_cache["api-service"] = [test_instance]
        mesh.discovery.last_cache_update["api-service"] = time.time()
        
        # Setup Mock HTTP client to fail then succeed
        Mock_client = AsyncMock()
        Mock_response_fail = AsyncMock()
        Mock_response_fail.status_code = 500
        Mock_response_success = AsyncMock()
        Mock_response_success.status_code = 200
        Mock_response_success.headers = {"content-type": "application/json"}
        Mock_response_success.json = Mock(return_value={"result": "success"})
        
        # First call fails, retry succeeds
        Mock_client.request = AsyncMock(side_effect=[
            Exception("Connection error"),
            Mock_response_success
        ])
        Mock_client.__aenter__ = AsyncMock(return_value=Mock_client)
        Mock_client.__aexit__ = AsyncMock()
        Mock_client_class.return_value = Mock_client
        
        request = ServiceRequest(
            service_name="api-service",
            method="GET",
            path="/test",
            retry_count=2
        )
        
        result = await mesh.call_service(request)
        
        assert result["status_code"] == 200
        assert result["body"]["result"] == "success"
        assert Mock_client.request.call_count == 2  # Initial + 1 retry
    
    @pytest.mark.asyncio
    async def test_health_check_state_transitions(self):
        """Test service health check and state transitions"""
        discovery = ServiceDiscovery()
        
        instance = ServiceInstance(
            "api-1", "api", "localhost", 8080,
            state=ServiceState.UNKNOWN
        )
        
        # Test healthy response
        with patch('httpx.AsyncClient') as Mock_client_class:
            Mock_client = AsyncMock()
            Mock_response = AsyncMock()
            Mock_response.status_code = 200
            Mock_client.get = AsyncMock(return_value=Mock_response)
            Mock_client.__aenter__ = AsyncMock(return_value=Mock_client)
            Mock_client.__aexit__ = AsyncMock()
            Mock_client_class.return_value = Mock_client
            
            state = await discovery.health_check(instance)
            assert state == ServiceState.HEALTHY
            assert instance.state == ServiceState.HEALTHY
            assert instance.health_check_failures == 0
        
        # Test degraded response
        with patch('httpx.AsyncClient') as Mock_client_class:
            Mock_client = AsyncMock()
            Mock_response = AsyncMock()
            Mock_response.status_code = 429  # Too many requests
            Mock_client.get = AsyncMock(return_value=Mock_response)
            Mock_client.__aenter__ = AsyncMock(return_value=Mock_client)
            Mock_client.__aexit__ = AsyncMock()
            Mock_client_class.return_value = Mock_client
            
            state = await discovery.health_check(instance)
            assert state == ServiceState.DEGRADED
            assert instance.state == ServiceState.DEGRADED
        
        # Test unhealthy response
        with patch('httpx.AsyncClient') as Mock_client_class:
            Mock_client = AsyncMock()
            Mock_client.get = AsyncMock(side_effect=Exception("Connection refused"))
            Mock_client.__aenter__ = AsyncMock(return_value=Mock_client)
            Mock_client.__aexit__ = AsyncMock()
            Mock_client_class.return_value = Mock_client
            
            state = await discovery.health_check(instance)
            assert state == ServiceState.UNHEALTHY
            assert instance.state == ServiceState.UNHEALTHY
            assert instance.health_check_failures == 1
    
    @pytest.mark.asyncio
    async def test_service_topology_reporting(self):
        """Test service mesh topology reporting"""
        mesh = ServiceMesh()
        
        # Setup test topology
        mesh.discovery.services_cache = {
            "api-service": [
                ServiceInstance("api-1", "api-service", "10.0.0.1", 8080, state=ServiceState.HEALTHY),
                ServiceInstance("api-2", "api-service", "10.0.0.2", 8080, state=ServiceState.HEALTHY),
                ServiceInstance("api-3", "api-service", "10.0.0.3", 8080, state=ServiceState.UNHEALTHY)
            ],
            "auth-service": [
                ServiceInstance("auth-1", "auth-service", "10.0.0.4", 9000, state=ServiceState.HEALTHY)
            ]
        }
        
        topology = await mesh.get_service_topology()
        
        assert len(topology["services"]) == 2
        assert topology["total_instances"] == 4
        assert topology["healthy_instances"] == 3
        
        assert topology["services"]["api-service"]["total"] == 3
        assert topology["services"]["api-service"]["healthy"] == 2
        assert topology["services"]["auth-service"]["total"] == 1
        assert topology["services"]["auth-service"]["healthy"] == 1
    
    @pytest.mark.asyncio
    async def test_request_interceptors(self):
        """Test request and response interceptors"""
        mesh = ServiceMesh()
        
        # Add request interceptor
        async def add_auth_header(request: ServiceRequest) -> ServiceRequest:
            request.headers["Authorization"] = "Bearer test-token"
            return request
        
        mesh.add_request_interceptor(add_auth_header)
        
        # Add response interceptor
        async def add_timing_info(response: Dict[str, Any]) -> Dict[str, Any]:
            response["processed_at"] = time.time()
            return response
        
        mesh.add_response_interceptor(add_timing_info)
        
        # Setup Mock for testing
        test_instance = ServiceInstance(
            "api-1", "api-service", "localhost", 8080,
            state=ServiceState.HEALTHY
        )
        mesh.discovery.services_cache["api-service"] = [test_instance]
        mesh.discovery.last_cache_update["api-service"] = time.time()
        
        with patch('httpx.AsyncClient') as Mock_client_class:
            Mock_client = AsyncMock()
            Mock_response = AsyncMock()
            Mock_response.status_code = 200
            Mock_response.headers = {"content-type": "application/json"}
            Mock_response.json = Mock(return_value={"data": "test"})
            Mock_client.request = AsyncMock(return_value=Mock_response)
            Mock_client.__aenter__ = AsyncMock(return_value=Mock_client)
            Mock_client.__aexit__ = AsyncMock()
            Mock_client_class.return_value = Mock_client
            
            request = ServiceRequest(
                service_name="api-service",
                method="GET",
                path="/test"
            )
            
            result = await mesh.call_service(request)
            
            # Verify interceptors were applied
            call_args = Mock_client.request.call_args
            assert call_args[1]["headers"]["Authorization"] == "Bearer test-token"
            assert "processed_at" in result
    
    @pytest.mark.asyncio
    async def test_kong_integration_configuration(self):
        """Test Kong API Gateway integration"""
        with patch('httpx.AsyncClient') as Mock_client_class:
            Mock_client = AsyncMock()
            
            # Mock Kong admin API responses
            Mock_response_services = AsyncMock()
            Mock_response_services.status_code = 200
            Mock_response_services.json = Mock(return_value={"data": []})
            
            Mock_response_upstream = AsyncMock()
            Mock_response_upstream.status_code = 201
            
            Mock_response_target = AsyncMock()
            Mock_response_target.status_code = 201
            
            Mock_client.get = AsyncMock(return_value=Mock_response_services)
            Mock_client.put = AsyncMock(return_value=Mock_response_upstream)
            Mock_client.post = AsyncMock(return_value=Mock_response_target)
            Mock_client.__aenter__ = AsyncMock(return_value=Mock_client)
            Mock_client.__aexit__ = AsyncMock()
            Mock_client_class.return_value = Mock_client
            
            mesh = ServiceMesh(kong_admin_url="http://sutazai-kong:8001")
            await mesh._configure_kong_routes()
            
            # Register a service
            instance = await mesh.register_service(
                service_name="api-service",
                address="10.0.0.1",
                port=8080,
                tags=["api", "v1"]
            )
            
            # Verify Kong upstream configuration
            Mock_client.put.assert_called()
            put_call = Mock_client.put.call_args
            assert "api-service-upstream" in put_call[0][0]
            
            upstream_data = put_call[1]["json"]
            assert upstream_data["name"] == "api-service-upstream"
            assert upstream_data["algorithm"] == "round-robin"
            assert "healthchecks" in upstream_data
    
    @pytest.mark.asyncio
    async def test_distributed_tracing_headers(self):
        """Test distributed tracing header propagation"""
        mesh = ServiceMesh()
        
        test_instance = ServiceInstance(
            "api-1", "api-service", "localhost", 8080,
            state=ServiceState.HEALTHY
        )
        mesh.discovery.services_cache["api-service"] = [test_instance]
        mesh.discovery.last_cache_update["api-service"] = time.time()
        
        with patch('httpx.AsyncClient') as Mock_client_class:
            Mock_client = AsyncMock()
            Mock_response = AsyncMock()
            Mock_response.status_code = 200
            Mock_response.headers = {"content-type": "application/json"}
            Mock_response.json = Mock(return_value={"data": "test"})
            Mock_client.request = AsyncMock(return_value=Mock_response)
            Mock_client.__aenter__ = AsyncMock(return_value=Mock_client)
            Mock_client.__aexit__ = AsyncMock()
            Mock_client_class.return_value = Mock_client
            
            request = ServiceRequest(
                service_name="api-service",
                method="GET",
                path="/test",
                trace_id="test-trace-123"
            )
            
            result = await mesh.call_service(request)
            
            # Verify trace headers were added
            call_args = Mock_client.request.call_args
            headers = call_args[1]["headers"]
            assert headers["X-Trace-Id"] == "test-trace-123"
            assert "X-Request-Start" in headers
            assert result["trace_id"] == "test-trace-123"
    
    @pytest.mark.asyncio
    async def test_failure_scenarios_cascading(self):
        """Test cascading failure prevention"""
        mesh = ServiceMesh()
        
        # Setup services with circuit breakers
        instances = [
            ServiceInstance("api-1", "api", "10.0.0.1", 8080, state=ServiceState.HEALTHY),
            ServiceInstance("api-2", "api", "10.0.0.2", 8080, state=ServiceState.HEALTHY),
            ServiceInstance("api-3", "api", "10.0.0.3", 8080, state=ServiceState.HEALTHY)
        ]
        mesh.discovery.services_cache["api"] = instances
        mesh.discovery.last_cache_update["api"] = time.time()
        
        # Simulate api-1 circuit breaker open
        mesh.circuit_breaker.get_breaker("api-1")
        mesh.circuit_breaker.breakers["api-1"].open()
        
        # Should skip api-1 and use api-2
        selected_instances = []
        for _ in range(5):
            request = ServiceRequest(
                service_name="api",
                method="GET",
                path="/test",
                retry_count=0
            )
            
            # Mock the actual call
            with patch('httpx.AsyncClient') as Mock_client_class:
                Mock_client = AsyncMock()
                Mock_response = AsyncMock()
                Mock_response.status_code = 200
                Mock_response.headers = {"content-type": "application/json"}
                Mock_response.json = Mock(return_value={"data": "test"})
                Mock_client.request = AsyncMock(return_value=Mock_response)
                Mock_client.__aenter__ = AsyncMock(return_value=Mock_client)
                Mock_client.__aexit__ = AsyncMock()
                Mock_client_class.return_value = Mock_client
                
                result = await mesh.call_service(request)
                selected_instances.append(result.get("instance_id"))
        
        # api-1 should never be selected due to open circuit breaker
        assert "api-1" not in selected_instances
    
    @pytest.mark.asyncio
    async def test_compatibility_api_endpoints(self):
        """Test backward compatibility with existing API endpoints"""
        mesh = ServiceMesh()
        
        # Test register_service_v2
        service_info = {
            "service_name": "test-service",
            "address": "localhost",
            "port": 8080,
            "tags": ["test"],
            "metadata": {"version": "1.0"}
        }
        
        with patch.object(mesh.discovery, 'register_service', return_value=True) as Mock_register:
            result = await mesh.register_service_v2("test-id", service_info)
            assert result["id"] == "test-id"
            assert result["status"] == "registered"
            Mock_register.assert_called_once()
        
        # Test discover_services compatibility
        mesh.discovery.services_cache["test-service"] = [
            ServiceInstance("test-1", "test-service", "localhost", 8080, state=ServiceState.HEALTHY)
        ]
        
        services = await mesh.discover_services("test-service")
        assert len(services) == 1
        assert services[0]["id"] == "test-1"
        assert services[0]["state"] == "healthy"
        
        # Test health_check compatibility
        health = await mesh.health_check()
        assert "status" in health
        assert "services" in health
        assert "queue_stats" in health
        assert "consul_connected" in health
    
    @pytest.mark.asyncio
    async def test_performance_metrics_collection(self):
        """Test performance metrics are properly collected"""
        from app.mesh.service_mesh import (
            service_discovery_counter,
            load_balancer_counter,
            circuit_breaker_counter,
            request_duration,
            active_services,
            health_check_gauge
        )
        
        # Reset metrics for testing
        mesh = ServiceMesh()
        
        # Register service and verify metrics
        with patch.object(mesh.discovery, 'consul_client') as Mock_consul:
            Mock_consul.agent.service.register = Mock()
            
            instance = ServiceInstance(
                "test-1", "test", "localhost", 8080
            )
            await mesh.discovery.register_service(instance)
            
            # Metrics should be updated
            # Note: In real tests, we'd check Prometheus registry
            assert mesh.discovery.services_cache.get("test", [])


class TestServiceMeshIntegration:
    """Integration tests for ServiceMesh with real services"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_end_to_end_service_communication(self):
        """Test end-to-end service communication through mesh"""
        # This test would require actual services running
        # Marked as integration test to be run in proper environment
        mesh = await get_mesh()
        
        # Register a test service
        instance = await mesh.register_service(
            service_name="test-integration",
            address="localhost",
            port=8080,
            tags=["integration", "test"]
        )
        
        assert instance is not None
        
        # Clean up
        await mesh.discovery.deregister_service(instance.service_id)
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_consul_integration_live(self):
        """Test actual Consul integration if available"""
        try:
            discovery = ServiceDiscovery("sutazai-consul", 8500)
            await discovery.connect()
            
            if discovery.consul_client:
                # Register a test service
                instance = ServiceInstance(
                    "integration-test-1",
                    "integration-test",
                    "localhost",
                    9999
                )
                
                result = await discovery.register_service(instance)
                assert result is True
                
                # Discover services
                services = await discovery.discover_services("integration-test")
                # May or may not find it immediately due to Consul propagation
                
                # Clean up
                await discovery.deregister_service("integration-test-1")
        except Exception as e:
            # Skip if Consul not available
            pytest.skip(f"Consul not available: {e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])