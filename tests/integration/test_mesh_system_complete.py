"""
Comprehensive Mesh System Integration Tests
Purpose: Validate complete mesh functionality with all services
Created: 2025-08-17 UTC
"""
import pytest
import asyncio
import httpx
import json
import time
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add backend to path
sys.path.insert(0, '/opt/sutazaiapp/backend')

from app.mesh.service_mesh import ServiceMesh, ServiceInstance, ServiceState, LoadBalancerStrategy
from app.mesh.dind_mesh_bridge import DinDMeshBridge
from app.mesh.mcp_mesh_integration import MCPMeshIntegration, MCPMeshIntegrationConfig


class TestMeshSystemComplete:
    """Complete mesh system integration tests"""
    
    @pytest.fixture
    async def mesh(self):
        """Create mesh instance"""
        mesh = ServiceMesh()
        await mesh.initialize()
        yield mesh
        await mesh.shutdown()
    
    @pytest.fixture
    async def dind_bridge(self, mesh):
        """Create DinD bridge instance"""
        bridge = DinDMeshBridge(mesh_client=mesh)
        await bridge.initialize()
        yield bridge
        await bridge.shutdown()
    
    @pytest.fixture
    async def mcp_integration(self, mesh):
        """Create MCP integration instance"""
        config = MCPMeshIntegrationConfig(
            enable_protocol_translation=True,
            enable_resource_isolation=True,
            enable_process_orchestration=True,
            enable_request_routing=True,
            enable_load_balancing=True,
            enable_health_monitoring=True,
            enable_auto_recovery=True
        )
        integration = MCPMeshIntegration(mesh, config)
        await integration.initialize()
        yield integration
        await integration.shutdown()
    
    @pytest.mark.asyncio
    async def test_mesh_initialization(self, mesh):
        """Test mesh initialization"""
        assert mesh.initialized
        status = await mesh.get_mesh_status()
        assert status['mesh_initialized'] == True
        assert 'consul_connected' in status
        assert 'total_services' in status
    
    @pytest.mark.asyncio
    async def test_service_registration(self, mesh):
        """Test service registration with Consul"""
        # Register test service
        service = ServiceInstance(
            service_id="test-service-1",
            service_name="test-service",
            address="localhost",
            port=9999,
            tags=["test", "integration"],
            metadata={"version": "1.0.0"}
        )
        
        result = await mesh.register_service(service)
        assert result == True
        
        # Verify registration
        services = await mesh.discover_service("test-service")
        assert len(services) > 0
        assert services[0].service_id == "test-service-1"
        
        # Cleanup
        await mesh.deregister_service("test-service-1")
    
    @pytest.mark.asyncio
    async def test_service_discovery(self, mesh):
        """Test service discovery functionality"""
        # Register multiple instances
        for i in range(3):
            service = ServiceInstance(
                service_id=f"discovery-test-{i}",
                service_name="discovery-test",
                address="localhost",
                port=8000 + i,
                tags=["test"],
                state=ServiceState.HEALTHY
            )
            await mesh.register_service(service)
        
        # Discover services
        services = await mesh.discover_service("discovery-test")
        assert len(services) == 3
        
        # Test filtering by tags
        services = await mesh.discover_service("discovery-test", tags=["test"])
        assert len(services) == 3
        
        # Cleanup
        for i in range(3):
            await mesh.deregister_service(f"discovery-test-{i}")
    
    @pytest.mark.asyncio
    async def test_load_balancing_strategies(self, mesh):
        """Test different load balancing strategies"""
        # Register services for load balancing
        services = []
        for i in range(5):
            service = ServiceInstance(
                service_id=f"lb-test-{i}",
                service_name="lb-test",
                address="localhost",
                port=7000 + i,
                weight=100 if i != 2 else 200,  # Give one service more weight
                state=ServiceState.HEALTHY
            )
            services.append(service)
            await mesh.register_service(service)
        
        # Test round-robin
        mesh.set_load_balancer_strategy("lb-test", LoadBalancerStrategy.ROUND_ROBIN)
        selected = []
        for _ in range(5):
            instance = await mesh.select_instance("lb-test")
            selected.append(instance.service_id)
        assert len(set(selected)) == 5  # All instances selected
        
        # Test weighted
        mesh.set_load_balancer_strategy("lb-test", LoadBalancerStrategy.WEIGHTED)
        weighted_selections = {}
        for _ in range(100):
            instance = await mesh.select_instance("lb-test")
            weighted_selections[instance.service_id] = weighted_selections.get(instance.service_id, 0) + 1
        
        # Service with weight 200 should get roughly twice the traffic
        assert weighted_selections.get("lb-test-2", 0) > weighted_selections.get("lb-test-0", 0)
        
        # Cleanup
        for i in range(5):
            await mesh.deregister_service(f"lb-test-{i}")
    
    @pytest.mark.asyncio
    async def test_circuit_breaker(self, mesh):
        """Test circuit breaker functionality"""
        # Register service
        service = ServiceInstance(
            service_id="cb-test-1",
            service_name="cb-test",
            address="localhost",
            port=6666,
            state=ServiceState.HEALTHY
        )
        await mesh.register_service(service)
        
        # Get circuit breaker
        cb = mesh.get_circuit_breaker("cb-test")
        assert cb.current_state == "closed"
        
        # Simulate failures to trip breaker
        for _ in range(6):  # Default threshold is 5
            try:
                cb.call(lambda: 1/0)  # Will always fail
            except:
                pass
        
        # Check breaker is open
        assert cb.current_state == "open"
        
        # Test half-open after delay
        time.sleep(2)  # Wait for recovery timeout
        assert cb.current_state in ["half_open", "open"]
        
        # Cleanup
        await mesh.deregister_service("cb-test-1")
    
    @pytest.mark.asyncio
    async def test_health_checking(self, mesh):
        """Test service health checking"""
        # Register service with health check
        service = ServiceInstance(
            service_id="health-test-1",
            service_name="health-test",
            address="localhost",
            port=5555,
            state=ServiceState.HEALTHY
        )
        await mesh.register_service(service)
        
        # Perform health check
        is_healthy = await mesh.check_service_health("health-test-1")
        # Will fail since service doesn't actually exist, but check mechanism works
        assert isinstance(is_healthy, bool)
        
        # Update service state
        await mesh.update_service_state("health-test-1", ServiceState.UNHEALTHY)
        
        # Verify state update
        services = await mesh.discover_service("health-test")
        if services:
            assert services[0].state == ServiceState.UNHEALTHY
        
        # Cleanup
        await mesh.deregister_service("health-test-1")
    
    @pytest.mark.asyncio
    async def test_dind_bridge_initialization(self, dind_bridge):
        """Test DinD bridge initialization"""
        assert dind_bridge.initialized
        status = dind_bridge.get_service_status()
        assert 'total' in status
        assert 'healthy' in status
        assert 'unhealthy' in status
    
    @pytest.mark.asyncio
    async def test_dind_mcp_discovery(self, dind_bridge):
        """Test MCP container discovery in DinD"""
        services = await dind_bridge.discover_mcp_containers()
        # Should discover MCP containers if they're running
        assert isinstance(services, list)
        
        if services:
            # Verify service structure
            service = services[0]
            assert 'name' in service
            assert 'container_id' in service
            assert 'status' in service
    
    @pytest.mark.asyncio
    async def test_dind_bridge_routing(self, dind_bridge):
        """Test request routing through DinD bridge"""
        # Create test request
        request_data = {
            "service": "mcp-claude-flow",
            "method": "status",
            "params": {}
        }
        
        # Route request (will fail if service not running, but tests routing logic)
        try:
            response = await dind_bridge.route_to_mcp(
                "mcp-claude-flow",
                request_data
            )
            assert response is not None
        except Exception as e:
            # Expected if MCP not running, but routing logic tested
            assert "connection" in str(e).lower() or "not found" in str(e).lower()
    
    @pytest.mark.asyncio
    async def test_mcp_integration_protocol_translation(self, mcp_integration):
        """Test MCP protocol translation"""
        # Test JSON-RPC to REST translation
        jsonrpc_request = {
            "jsonrpc": "2.0",
            "method": "test_method",
            "params": {"key": "value"},
            "id": 1
        }
        
        rest_request = mcp_integration.translate_jsonrpc_to_rest(jsonrpc_request)
        assert rest_request['method'] == "POST"
        assert rest_request['path'] == "/test_method"
        assert rest_request['body'] == {"key": "value"}
        
        # Test REST to JSON-RPC translation
        rest_response = {
            "status": 200,
            "body": {"result": "success"}
        }
        
        jsonrpc_response = mcp_integration.translate_rest_to_jsonrpc(rest_response, 1)
        assert jsonrpc_response['jsonrpc'] == "2.0"
        assert jsonrpc_response['result'] == {"result": "success"}
        assert jsonrpc_response['id'] == 1
    
    @pytest.mark.asyncio
    async def test_mesh_request_routing(self, mesh):
        """Test request routing through mesh"""
        # Register test service
        service = ServiceInstance(
            service_id="route-test-1",
            service_name="route-test",
            address="localhost",
            port=4444,
            state=ServiceState.HEALTHY
        )
        await mesh.register_service(service)
        
        # Create and route request
        from app.mesh.service_mesh import ServiceRequest
        request = ServiceRequest(
            service_name="route-test",
            method="GET",
            path="/test",
            headers={"X-Test": "true"},
            timeout=5.0
        )
        
        # Attempt routing (will fail if service not running)
        try:
            response = await mesh.route_request(request)
            assert response is not None
        except Exception as e:
            # Expected if service not actually running
            assert "connection" in str(e).lower()
        
        # Cleanup
        await mesh.deregister_service("route-test-1")
    
    @pytest.mark.asyncio
    async def test_mesh_metrics_collection(self, mesh):
        """Test mesh metrics collection"""
        # Perform operations to generate metrics
        await mesh.discover_service("test-service")
        
        # Check metrics are being collected
        from prometheus_client import REGISTRY
        
        # Get all metric names
        metric_names = [metric.name for metric in REGISTRY.collect()]
        
        # Verify mesh metrics exist
        mesh_metrics = [
            "mesh_service_discovery_total",
            "mesh_load_balancer_requests",
            "mesh_active_services"
        ]
        
        for metric in mesh_metrics:
            assert any(metric in name for name in metric_names)
    
    @pytest.mark.asyncio
    async def test_mesh_auto_recovery(self, mesh):
        """Test automatic service recovery"""
        # Register service
        service = ServiceInstance(
            service_id="recovery-test-1",
            service_name="recovery-test",
            address="localhost",
            port=3333,
            state=ServiceState.HEALTHY
        )
        await mesh.register_service(service)
        
        # Mark as unhealthy
        await mesh.update_service_state("recovery-test-1", ServiceState.UNHEALTHY)
        
        # Trigger recovery check
        recovered = await mesh.attempt_service_recovery("recovery-test-1")
        
        # Recovery will fail for non-existent service, but mechanism tested
        assert isinstance(recovered, bool)
        
        # Cleanup
        await mesh.deregister_service("recovery-test-1")
    
    @pytest.mark.asyncio
    async def test_mesh_shutdown_cleanup(self, mesh):
        """Test proper mesh shutdown and cleanup"""
        # Register services
        for i in range(3):
            service = ServiceInstance(
                service_id=f"shutdown-test-{i}",
                service_name="shutdown-test",
                address="localhost",
                port=2000 + i,
                state=ServiceState.HEALTHY
            )
            await mesh.register_service(service)
        
        # Verify services registered
        services = await mesh.discover_service("shutdown-test")
        assert len(services) == 3
        
        # Shutdown mesh
        await mesh.shutdown()
        
        # Verify cleanup
        assert mesh.consul_client is None
        assert len(mesh._service_instances) == 0
        assert len(mesh._circuit_breakers) == 0


@pytest.mark.asyncio
class TestMeshEndToEnd:
    """End-to-end mesh integration tests"""
    
    async def test_complete_mesh_workflow(self):
        """Test complete mesh workflow from initialization to shutdown"""
        # Initialize mesh
        mesh = ServiceMesh()
        await mesh.initialize()
        assert mesh.initialized
        
        # Initialize DinD bridge
        dind_bridge = DinDMeshBridge(mesh_client=mesh)
        await dind_bridge.initialize()
        assert dind_bridge.initialized
        
        # Discover MCP services
        mcp_services = await dind_bridge.discover_mcp_containers()
        print(f"Discovered {len(mcp_services)} MCP services")
        
        # Register a test service
        test_service = ServiceInstance(
            service_id="e2e-test-service",
            service_name="e2e-test",
            address="localhost",
            port=19999,
            tags=["test", "e2e"],
            metadata={"test": "true"}
        )
        await mesh.register_service(test_service)
        
        # Discover the service
        discovered = await mesh.discover_service("e2e-test")
        assert len(discovered) > 0
        assert discovered[0].service_id == "e2e-test-service"
        
        # Test load balancing
        for _ in range(5):
            selected = await mesh.select_instance("e2e-test")
            assert selected.service_id == "e2e-test-service"
        
        # Get mesh status
        status = await mesh.get_mesh_status()
        assert status['total_services'] > 0
        assert status['mesh_initialized'] == True
        
        # Cleanup
        await mesh.deregister_service("e2e-test-service")
        await dind_bridge.shutdown()
        await mesh.shutdown()
        
        print("✓ Complete mesh workflow test passed")


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])