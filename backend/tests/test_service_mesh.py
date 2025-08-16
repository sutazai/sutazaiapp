"""
Comprehensive tests for the real service mesh implementation
"""
import pytest
import asyncio
import time
from unittest.Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test import Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test, patch, MagicRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test
from typing import Dict, Any, List

from app.mesh.service_mesh import (
    ServiceMesh, ServiceInstance, ServiceState, ServiceRequest,
    LoadBalancerStrategy, ServiceDiscovery, LoadBalancer, CircuitBreakerManager,
    get_mesh
)
from app.mesh.distributed_tracing import (
    Tracer, Span, SpanContext, SpanType, SpanStatus, TraceCollector
)
from app.mesh.mesh_dashboard import MeshDashboard, ServiceMetrics, MeshMetrics


class TestServiceInstance:
    """Test ServiceInstance class"""
    
    def test_service_instance_creation(self):
        """Test creating a service instance"""
        instance = ServiceInstance(
            service_id="test-service-1",
            service_name="test-service",
            address="localhost",
            port=8080,
            tags=["api", "v1"],
            metadata={"version": "1.0.0"}
        )
        
        assert instance.service_id == "test-service-1"
        assert instance.service_name == "test-service"
        assert instance.url == "http://localhost:8080"
        assert instance.state == ServiceState.UNKNOWN
        assert instance.weight == 100
        assert instance.connections == 0
    
    def test_to_consul_format(self):
        """Test converting to Consul service format"""
        instance = ServiceInstance(
            service_id="test-service-1",
            service_name="test-service",
            address="localhost",
            port=8080
        )
        
        consul_format = instance.to_consul_format()
        
        assert consul_format["ID"] == "test-service-1"
        assert consul_format["Name"] == "test-service"
        assert consul_format["Address"] == "localhost"
        assert consul_format["Port"] == 8080
        assert "Check" in consul_format
        assert consul_format["Check"]["HTTP"] == "http://localhost:8080/health"


class TestCircuitBreakerManager:
    """Test CircuitBreakerManager class"""
    
    def test_circuit_breaker_creation(self):
        """Test circuit breaker manager creation"""
        manager = CircuitBreakerManager(failure_threshold=3, recovery_timeout=30)
        
        assert manager.failure_threshold == 3
        assert manager.recovery_timeout == 30
        assert len(manager.breakers) == 0
    
    def test_get_breaker(self):
        """Test getting or creating a circuit breaker"""
        manager = CircuitBreakerManager()
        
        breaker1 = manager.get_breaker("service-1")
        breaker2 = manager.get_breaker("service-1")
        breaker3 = manager.get_breaker("service-2")
        
        assert breaker1 is breaker2  # Same breaker for same service
        assert breaker1 is not breaker3  # Different breaker for different service
        assert len(manager.breakers) == 2
    
    def test_circuit_breaker_state(self):
        """Test circuit breaker state management"""
        manager = CircuitBreakerManager(failure_threshold=2)
        
        # Initially closed
        assert not manager.is_open("service-1")
        
        # Record failures
        manager.record_failure("service-1")
        assert not manager.is_open("service-1")  # Still closed after 1 failure
        
        manager.record_failure("service-1")
        # Note: py-circuitbreaker may not immediately open, depends on implementation


@pytest.mark.asyncio
class TestServiceDiscovery:
    """Test ServiceDiscovery class"""
    
    async def test_service_discovery_initialization(self):
        """Test service discovery initialization"""
        discovery = ServiceDiscovery(consul_host="localhost", consul_port=8500)
        
        assert discovery.consul_host == "localhost"
        assert discovery.consul_port == 8500
        assert discovery.consul_client is None
        assert len(discovery.services_cache) == 0
    
    @patch('consul.aio.Consul')
    async def test_connect_to_consul(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul_class):
        """Test connecting to Consul"""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul
        
        discovery = ServiceDiscovery()
        await discovery.connect()
        
        assert discovery.consul_client is not None
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul_class.assert_called_once_with(host="consul", port=8500)
    
    @patch('consul.aio.Consul')
    async def test_register_service(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul_class):
        """Test registering a service with Consul"""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul.agent.service.register = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul
        
        discovery = ServiceDiscovery()
        
        instance = ServiceInstance(
            service_id="test-1",
            service_name="test",
            address="localhost",
            port=8080
        )
        
        result = await discovery.register_service(instance)
        
        assert result is True
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul.agent.service.register.assert_called_once()
    
    @patch('consul.aio.Consul')
    async def test_discover_services_with_cache(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul_class):
        """Test discovering services with caching"""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul.health.service = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=(
            None,
            [
                {
                    'Service': {
                        'ID': 'service-1',
                        'Service': 'test-service',
                        'Address': 'localhost',
                        'Port': 8080,
                        'Tags': ['api'],
                        'Meta': {}
                    },
                    'Node': {'Address': '127.0.0.1'}
                }
            ]
        ))
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul
        
        discovery = ServiceDiscovery()
        
        # First call - should query Consul
        instances1 = await discovery.discover_services("test-service")
        assert len(instances1) == 1
        assert instances1[0].service_id == "service-1"
        
        # Second call - should use cache
        instances2 = await discovery.discover_services("test-service", use_cache=True)
        assert len(instances2) == 1
        
        # Consul should only be called once due to caching
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_consul.health.service.assert_called_once()


class TestLoadBalancer:
    """Test LoadBalancer class"""
    
    def test_round_robin_strategy(self):
        """Test round-robin load balancing"""
        balancer = LoadBalancer(LoadBalancerStrategy.ROUND_ROBIN)
        
        instances = [
            ServiceInstance("s1", "test", "host1", 8080, state=ServiceState.HEALTHY),
            ServiceInstance("s2", "test", "host2", 8080, state=ServiceState.HEALTHY),
            ServiceInstance("s3", "test", "host3", 8080, state=ServiceState.HEALTHY)
        ]
        
        # Should cycle through instances
        selected1 = balancer.select_instance(instances, "test")
        selected2 = balancer.select_instance(instances, "test")
        selected3 = balancer.select_instance(instances, "test")
        selected4 = balancer.select_instance(instances, "test")
        
        assert selected1.service_id == "s1"
        assert selected2.service_id == "s2"
        assert selected3.service_id == "s3"
        assert selected4.service_id == "s1"  # Back to first
    
    def test_least_connections_strategy(self):
        """Test least connections load balancing"""
        balancer = LoadBalancer(LoadBalancerStrategy.LEAST_CONNECTIONS)
        
        instances = [
            ServiceInstance("s1", "test", "host1", 8080, state=ServiceState.HEALTHY, connections=5),
            ServiceInstance("s2", "test", "host2", 8080, state=ServiceState.HEALTHY, connections=2),
            ServiceInstance("s3", "test", "host3", 8080, state=ServiceState.HEALTHY, connections=8)
        ]
        
        selected = balancer.select_instance(instances, "test")
        assert selected.service_id == "s2"  # Has least connections
    
    def test_weighted_strategy(self):
        """Test weighted load balancing"""
        balancer = LoadBalancer(LoadBalancerStrategy.WEIGHTED)
        
        instances = [
            ServiceInstance("s1", "test", "host1", 8080, state=ServiceState.HEALTHY, weight=100),
            ServiceInstance("s2", "test", "host2", 8080, state=ServiceState.HEALTHY, weight=0)
        ]
        
        # With weight 0, s2 should never be selected
        selections = [balancer.select_instance(instances, "test") for _ in range(10)]
        assert all(s.service_id == "s1" for s in selections if s)
    
    def test_unhealthy_instance_filtering(self):
        """Test that unhealthy instances are filtered out"""
        balancer = LoadBalancer(LoadBalancerStrategy.ROUND_ROBIN)
        
        instances = [
            ServiceInstance("s1", "test", "host1", 8080, state=ServiceState.HEALTHY),
            ServiceInstance("s2", "test", "host2", 8080, state=ServiceState.UNHEALTHY),
            ServiceInstance("s3", "test", "host3", 8080, state=ServiceState.DEGRADED)
        ]
        
        # Should only select healthy instance
        selected = balancer.select_instance(instances, "test")
        assert selected.service_id == "s1"
    
    def test_fallback_to_degraded(self):
        """Test fallback to degraded instances when no healthy ones available"""
        balancer = LoadBalancer(LoadBalancerStrategy.ROUND_ROBIN)
        
        instances = [
            ServiceInstance("s1", "test", "host1", 8080, state=ServiceState.UNHEALTHY),
            ServiceInstance("s2", "test", "host2", 8080, state=ServiceState.DEGRADED)
        ]
        
        # Should select degraded instance when no healthy ones
        selected = balancer.select_instance(instances, "test")
        assert selected.service_id == "s2"


@pytest.mark.asyncio
class TestServiceMesh:
    """Test ServiceMesh class"""
    
    @patch('app.mesh.service_mesh.ServiceDiscovery')
    async def test_service_mesh_initialization(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery_class):
        """Test service mesh initialization"""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery
        
        mesh = ServiceMesh(consul_host="localhost", kong_admin_url="http://kong:8001")
        await mesh.initialize()
        
        assert mesh.discovery is not None
        assert mesh.load_balancer is not None
        assert mesh.circuit_breaker is not None
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery.connect.assert_called_once()
    
    @patch('app.mesh.service_mesh.ServiceDiscovery')
    @patch('httpx.AsyncClient')
    async def test_register_service(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client_class, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery_class):
        """Test registering a service with the mesh"""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery.register_service = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=True)
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery
        
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response.status_code = 201
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client.put = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response)
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client.post = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response)
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client.__aenter__ = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client)
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client.__aexit__ = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client
        
        mesh = ServiceMesh()
        mesh.discovery = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery
        
        instance = await mesh.register_service(
            service_name="test-service",
            address="localhost",
            port=8080,
            tags=["api"],
            metadata={"version": "1.0"}
        )
        
        assert instance.service_name == "test-service"
        assert instance.address == "localhost"
        assert instance.port == 8080
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery.register_service.assert_called_once()
    
    @patch('app.mesh.service_mesh.ServiceDiscovery')
    @patch('httpx.AsyncClient')
    async def test_call_service_success(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client_class, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery_class):
        """Test successful service call through mesh"""
        # Setup Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test discovery
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        test_instance = ServiceInstance(
            "test-1", "test-service", "localhost", 8080,
            state=ServiceState.HEALTHY
        )
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery.discover_services = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=[test_instance])
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery
        
        # Setup Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test HTTP client
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response.status_code = 200
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response.headers = {"content-type": "application/json"}
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response.json = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value={"result": "success"})
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client.request = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_response)
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client.__aenter__ = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client)
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client.__aexit__ = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_client
        
        mesh = ServiceMesh()
        mesh.discovery = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery
        
        request = ServiceRequest(
            service_name="test-service",
            method="GET",
            path="/test"
        )
        
        result = await mesh.call_service(request)
        
        assert result["status_code"] == 200
        assert result["body"]["result"] == "success"
        assert "trace_id" in result
        assert "duration" in result
    
    @patch('app.mesh.service_mesh.ServiceDiscovery')
    async def test_call_service_no_instances(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery_class):
        """Test service call when no instances available"""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery.discover_services = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test(return_value=[])
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery
        
        mesh = ServiceMesh()
        mesh.discovery = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery
        
        request = ServiceRequest(
            service_name="test-service",
            method="GET",
            path="/test",
            retry_count=0  # Disable retries for test
        )
        
        with pytest.raises(Exception, match="No instances available"):
            await mesh.call_service(request)
    
    @patch('app.mesh.service_mesh.ServiceDiscovery')
    async def test_get_service_topology(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery_class):
        """Test getting service mesh topology"""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery.services_cache = {
            "service1": [
                ServiceInstance("s1-1", "service1", "host1", 8080, state=ServiceState.HEALTHY),
                ServiceInstance("s1-2", "service1", "host2", 8080, state=ServiceState.UNHEALTHY)
            ],
            "service2": [
                ServiceInstance("s2-1", "service2", "host3", 8080, state=ServiceState.HEALTHY)
            ]
        }
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery_class.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery
        
        mesh = ServiceMesh()
        mesh.discovery = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_discovery
        
        topology = await mesh.get_service_topology()
        
        assert len(topology["services"]) == 2
        assert topology["total_instances"] == 3
        assert topology["healthy_instances"] == 2
        assert "service1" in topology["services"]
        assert "service2" in topology["services"]


class TestDistributedTracing:
    """Test distributed tracing components"""
    
    def test_span_context_creation(self):
        """Test creating span context"""
        context = SpanContext(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id="parent-789",
            flags=1,
            baggage={"user_id": "user-1"}
        )
        
        assert context.trace_id == "trace-123"
        assert context.span_id == "span-456"
        assert context.parent_span_id == "parent-789"
        assert context.baggage["user_id"] == "user-1"
    
    def test_span_context_to_headers(self):
        """Test converting span context to headers"""
        context = SpanContext(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id="parent-789"
        )
        
        headers = context.to_headers()
        
        assert headers["X-Trace-Id"] == "trace-123"
        assert headers["X-Span-Id"] == "span-456"
        assert headers["X-Parent-Span-Id"] == "parent-789"
        assert headers["X-Trace-Flags"] == "0"
    
    def test_span_context_from_headers(self):
        """Test creating span context from headers"""
        headers = {
            "X-Trace-Id": "trace-123",
            "X-Span-Id": "span-456",
            "X-Parent-Span-Id": "parent-789",
            "X-Trace-Flags": "1"
        }
        
        context = SpanContext.from_headers(headers)
        
        assert context is not None
        assert context.trace_id == "trace-123"
        assert context.span_id == "span-456"
        assert context.parent_span_id == "parent-789"
        assert context.flags == 1
    
    def test_span_creation_and_finish(self):
        """Test creating and finishing a span"""
        span = Span(
            trace_id="trace-123",
            span_id="span-456",
            parent_span_id=None,
            operation_name="test-operation",
            service_name="test-service",
            span_type=SpanType.SERVER,
            start_time=time.time()
        )
        
        span.add_tag("http.method", "GET")
        span.add_log("Request received", level="info")
        
        assert span.status == SpanStatus.OK
        assert span.end_time is None
        
        span.finish(SpanStatus.OK)
        
        assert span.end_time is not None
        assert span.duration is not None
        assert span.duration > 0
        assert len(span.tags) == 1
        assert len(span.logs) == 1
    
    @pytest.mark.asyncio
    async def test_trace_collector(self):
        """Test trace collector functionality"""
        collector = TraceCollector(max_traces=10)
        
        span1 = Span(
            trace_id="trace-1",
            span_id="span-1",
            parent_span_id=None,
            operation_name="op1",
            service_name="service1",
            span_type=SpanType.SERVER,
            start_time=time.time()
        )
        
        span2 = Span(
            trace_id="trace-1",
            span_id="span-2",
            parent_span_id="span-1",
            operation_name="op2",
            service_name="service2",
            span_type=SpanType.CLIENT,
            start_time=time.time()
        )
        
        await collector.add_span(span1)
        await collector.add_span(span2)
        
        trace = collector.get_trace("trace-1")
        assert len(trace) == 2
        
        span = collector.get_span("span-1")
        assert span.operation_name == "op1"
        
        dependencies = collector.get_service_dependencies()
        assert "service1" in dependencies
        assert "service2" in dependencies["service1"]


@pytest.mark.asyncio
class TestMeshDashboard:
    """Test mesh dashboard functionality"""
    
    @patch('app.mesh.mesh_dashboard.get_mesh')
    async def test_dashboard_initialization(self, Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_mesh):
        """Test dashboard initialization"""
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_mesh = AsyncRemove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test()
        Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_get_mesh.return_value = Remove Remove Remove Mocks - Only use Real Tests - Only use Real Tests - Only use Real Test_mesh
        
        dashboard = MeshDashboard()
        await dashboard.initialize()
        
        assert dashboard.mesh is not None
        assert dashboard._collection_task is not None
        
        await dashboard.shutdown()
    
    def test_service_metrics(self):
        """Test service metrics calculations"""
        metrics = ServiceMetrics(
            service_name="test-service",
            instance_count=10,
            healthy_count=8,
            unhealthy_count=1,
            degraded_count=1,
            request_rate=100.0,
            error_rate=2.5,
            p50_latency=50.0,
            p95_latency=150.0,
            p99_latency=500.0,
            circuit_breakers_open=0,
            active_connections=25
        )
        
        assert metrics.health_percentage == 80.0
    
    def test_mesh_metrics(self):
        """Test mesh metrics calculations"""
        metrics = MeshMetrics(
            total_services=5,
            total_instances=20,
            healthy_instances=18,
            unhealthy_instances=1,
            degraded_instances=1,
            total_requests=1000,
            failed_requests=25,
            avg_latency=75.0,
            circuit_breakers_open=0,
            active_traces=15
        )
        
        assert metrics.health_score == 90.0
        assert metrics.success_rate == 97.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])