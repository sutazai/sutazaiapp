"""
Comprehensive Test Suite for MCP-Mesh Integration
Tests the complete integration of MCP servers with the service mesh
"""
import asyncio
import pytest
import json
from unittest.Mock import Mock, AsyncMock, patch, MagicMock
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.mesh.mcp_adapter import (
    MCPServiceAdapter, MCPProcess, MCPServerConfig, 
    MCPServerType, create_mcp_adapter
)
from app.mesh.mcp_bridge import MCPMeshBridge, get_mcp_bridge
from app.mesh.mcp_load_balancer import MCPLoadBalancer, MCPInstanceMetrics
from app.mesh.service_mesh import ServiceMesh, ServiceInstance, ServiceState, ServiceRequest
from app.api.v1.endpoints.mcp import MCPExecuteRequest

class TestMCPAdapter:
    """Test MCP Service Adapter functionality"""
    
    @pytest.fixture
    def Mock_config(self):
        """Create Mock MCP server configuration"""
        return MCPServerConfig(
            name="test-server",
            server_type=MCPServerType.NPX,
            wrapper_path="/opt/sutazaiapp/scripts/mcp/wrappers/test.sh",
            instances=2,
            port_range=[11100, 11110],
            tags=["test", "mcp"],
            metadata={"version": "1.0.0"}
        )
    
    @pytest.fixture
    def adapter(self, Mock_config):
        """Create MCP adapter instance"""
        return MCPServiceAdapter(Mock_config)
    
    def test_adapter_creation(self, adapter, Mock_config):
        """Test adapter is created with correct configuration"""
        assert adapter.config == Mock_config
        assert adapter.instances == {}
        assert not adapter.running
        assert adapter.app is not None
    
    def test_adapter_routes_setup(self, adapter):
        """Test that HTTP routes are properly configured"""
        # Get route paths from the app
        routes = [route.path for route in adapter.app.routes]
        
        assert "/health" in routes
        assert "/execute" in routes
        assert "/instances" in routes
    
    @pytest.mark.asyncio
    async def test_mcp_process_creation(self, Mock_config):
        """Test MCP process instance creation"""
        process = MCPProcess(Mock_config, instance_id=0)
        
        assert process.instance_id == 0
        assert process.config == Mock_config
        assert process.process is None
        assert process.health_status == "unknown"
        assert process.request_count == 0
        assert process.error_count == 0
    
    @pytest.mark.asyncio
    @patch('subprocess.Popen')
    async def test_mcp_process_start(self, Mock_popen, Mock_config):
        """Test starting an MCP process"""
        # Mock the subprocess
        Mock_proc = MagicMock()
        Mock_proc.poll.return_value = None  # Process is running
        Mock_popen.return_value = Mock_proc
        
        process = MCPProcess(Mock_config, instance_id=0)
        success = await process.start(port=11100)
        
        assert success
        assert process.port == 11100
        assert process.process == Mock_proc
        assert process.start_time is not None
        Mock_popen.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_health_check_endpoint(self, adapter):
        """Test the health check endpoint"""
        from fastapi.testclient import TestClient
        
        client = TestClient(adapter.app)
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert "service" in data
        assert "status" in data
        assert "instances" in data
        assert data["service"] == "test-server"

class TestMCPBridge:
    """Test MCP-Mesh Bridge functionality"""
    
    @pytest.fixture
    def Mock_mesh(self):
        """Create Mock service mesh"""
        mesh = AsyncMock(spec=ServiceMesh)
        mesh.register_service = AsyncMock(return_value=True)
        mesh.discover_services = AsyncMock(return_value=[])
        mesh.deregister_service = AsyncMock(return_value=True)
        mesh.call_service = AsyncMock(return_value={"result": "success"})
        return mesh
    
    @pytest.fixture
    def bridge(self, Mock_mesh, tmp_path):
        """Create MCP bridge with test registry"""
        # Create test registry file
        registry_path = tmp_path / "test_registry.yaml"
        registry_content = """
mcp_services:
  - name: test-service
    service_name: mcp-test-service
    instances: 1
    port_range: [11100, 11100]
    tags: ["test"]
    metadata:
      version: "1.0.0"
global_config:
  default_timeout: 30.0
"""
        registry_path.write_text(registry_content)
        
        return MCPMeshBridge(Mock_mesh, str(registry_path))
    
    def test_bridge_creation(self, bridge):
        """Test bridge is created correctly"""
        assert bridge.mesh is not None
        assert bridge.registry is not None
        assert len(bridge.registry.get('mcp_services', [])) == 1
        assert bridge.adapters == {}
        assert not bridge.running
    
    @pytest.mark.asyncio
    async def test_bridge_initialization(self, bridge, Mock_mesh):
        """Test bridge initialization process"""
        with patch('app.mesh.mcp_bridge.create_mcp_adapter') as Mock_create:
            # Mock adapter
            Mock_adapter = AsyncMock()
            Mock_adapter.start = AsyncMock(return_value=[11100])
            Mock_adapter.get_app = Mock(return_value=Mock())
            Mock_adapter.instances = {0: Mock(health_status="healthy")}
            Mock_adapter.running = True
            Mock_create.return_value = Mock_adapter
            
            with patch('app.mesh.mcp_bridge.uvicorn.Server'):
                results = await bridge.initialize()
        
        assert results["started"] == ["test-service"]
        assert results["failed"] == []
        assert results["registered"] == ["mcp-test-service"]
        assert bridge.running
    
    @pytest.mark.asyncio
    async def test_service_status(self, bridge):
        """Test getting service status"""
        # Add Mock adapter
        Mock_adapter = Mock()
        Mock_adapter.running = True
        Mock_adapter.instances = {
            0: Mock(
                service_id="test-0",
                port=11100,
                health_status="healthy",
                request_count=10,
                error_count=1
            )
        }
        bridge.adapters["test-service"] = Mock_adapter
        
        status = await bridge.get_service_status("test-service")
        
        assert status["service"] == "test-service"
        assert status["status"] == "active"
        assert len(status["instances"]) == 1
        assert status["instances"][0]["requests"] == 10
    
    @pytest.mark.asyncio
    async def test_call_mcp_service(self, bridge, Mock_mesh):
        """Test calling MCP service through bridge"""
        result = await bridge.call_mcp_service(
            service_name="test-service",
            method="test_method",
            params={"key": "value"}
        )
        
        assert result == {"result": "success"}
        Mock_mesh.call_service.assert_called_once()

class TestMCPLoadBalancer:
    """Test MCP-specific load balancing"""
    
    @pytest.fixture
    def load_balancer(self):
        """Create load balancer instance"""
        return MCPLoadBalancer()
    
    @pytest.fixture
    def Mock_instances(self):
        """Create Mock service instances"""
        return [
            ServiceInstance(
                service_id="inst-1",
                service_name="mcp-test",
                address="localhost",
                port=11100,
                state=ServiceState.HEALTHY,
                weight=100,
                metadata={"capabilities": ["cap1", "cap2"]}
            ),
            ServiceInstance(
                service_id="inst-2",
                service_name="mcp-test",
                address="localhost",
                port=11101,
                state=ServiceState.HEALTHY,
                weight=100,
                metadata={"capabilities": ["cap2", "cap3"]}
            ),
            ServiceInstance(
                service_id="inst-3",
                service_name="mcp-test",
                address="localhost",
                port=11102,
                state=ServiceState.UNHEALTHY,
                weight=100,
                metadata={"capabilities": ["cap1", "cap3"]}
            )
        ]
    
    def test_load_balancer_creation(self, load_balancer):
        """Test load balancer is created correctly"""
        assert load_balancer.instance_metrics == {}
        assert load_balancer.service_capabilities == {}
        assert load_balancer.sticky_sessions == {}
    
    def test_select_healthy_instances(self, load_balancer, Mock_instances):
        """Test that only healthy instances are selected"""
        selected = load_balancer.select_instance(
            Mock_instances, 
            "mcp-test"
        )
        
        assert selected is not None
        assert selected.state == ServiceState.HEALTHY
        assert selected.service_id in ["inst-1", "inst-2"]
    
    def test_capability_matching(self, load_balancer, Mock_instances):
        """Test capability-based selection"""
        context = {"required_capabilities": ["cap1", "cap2"]}
        
        selected = load_balancer.select_instance(
            Mock_instances,
            "mcp-language-server",  # Triggers capability strategy
            context
        )
        
        assert selected is not None
        # inst-1 has both cap1 and cap2, should be preferred
        assert selected.service_id == "inst-1"
    
    def test_metrics_update(self, load_balancer):
        """Test updating instance metrics"""
        load_balancer.update_metrics(
            instance_id="test-1",
            response_time=0.5,
            success=True,
            resource_usage={"cpu": 0.3, "memory": 0.4}
        )
        
        metrics = load_balancer.get_instance_stats("test-1")
        
        assert metrics["total_requests"] == 1
        assert metrics["response_time_avg"] == 0.5
        assert metrics["error_rate"] == 0.0
        assert metrics["cpu_usage"] == 0.3
        assert metrics["memory_usage"] == 0.4
        assert metrics["consecutive_errors"] == 0
    
    def test_error_tracking(self, load_balancer):
        """Test error rate tracking"""
        # Simulate some requests
        for _ in range(5):
            load_balancer.update_metrics("test-2", response_time=0.2, success=True)
        
        for _ in range(2):
            load_balancer.update_metrics("test-2", response_time=0.5, success=False)
        
        metrics = load_balancer.get_instance_stats("test-2")
        
        assert metrics["total_requests"] == 7
        assert metrics["consecutive_errors"] == 0  # Last was error but then reset
        assert 0.2 < metrics["error_rate"] < 0.3  # ~28% error rate

class TestMCPAPIEndpoints:
    """Test MCP API endpoints"""
    
    @pytest.fixture
    def Mock_bridge(self):
        """Create Mock MCP bridge"""
        bridge = AsyncMock(spec=MCPMeshBridge)
        bridge.registry = {
            "mcp_services": [
                {"name": "test-service", "metadata": {"capabilities": ["test"]}}
            ]
        }
        bridge.get_service_status = AsyncMock(return_value={
            "service": "test-service",
            "status": "active",
            "mesh_instances": 1,
            "adapter_instances": 1,
            "instances": [],
            "mesh_registration": True
        })
        bridge.call_mcp_service = AsyncMock(return_value={"result": "success"})
        bridge.health_check_all = AsyncMock(return_value={
            "test-service": {
                "total_instances": 1,
                "healthy": 1,
                "unhealthy": 0,
                "unknown": 0,
                "overall_health": "healthy"
            }
        })
        return bridge
    
    @pytest.mark.asyncio
    async def test_list_services_endpoint(self, Mock_bridge):
        """Test listing MCP services"""
        from app.api.v1.endpoints.mcp import list_mcp_services
        
        with patch('app.api.v1.endpoints.mcp.get_bridge', return_value=Mock_bridge):
            services = await list_mcp_services(bridge=Mock_bridge)
        
        assert services == ["test-service"]
    
    @pytest.mark.asyncio
    async def test_execute_command_endpoint(self, Mock_bridge):
        """Test executing MCP command"""
        from app.api.v1.endpoints.mcp import execute_mcp_command
        
        request = MCPExecuteRequest(
            method="test_method",
            params={"key": "value"}
        )
        
        with patch('app.api.v1.endpoints.mcp.get_bridge', return_value=Mock_bridge):
            result = await execute_mcp_command(
                service_name="test-service",
                request=request,
                bridge=Mock_bridge
            )
        
        assert result == {"result": "success"}
        Mock_bridge.call_mcp_service.assert_called_once_with(
            service_name="test-service",
            method="test_method",
            params={"key": "value"}
        )
    
    @pytest.mark.asyncio
    async def test_health_endpoint(self, Mock_bridge):
        """Test health check endpoint"""
        from app.api.v1.endpoints.mcp import get_mcp_health
        
        with patch('app.api.v1.endpoints.mcp.get_bridge', return_value=Mock_bridge):
            health = await get_mcp_health(bridge=Mock_bridge)
        
        assert "test-service" in health
        assert health["test-service"]["overall_health"] == "healthy"

class TestIntegration:
    """End-to-end integration tests"""
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_mcp_mesh_flow(self):
        """Test complete flow from MCP registration to service call"""
        # This would be an integration test with real components
        # For now, we'll Mock the key components
        
        with patch('app.mesh.service_mesh.get_service_mesh') as Mock_get_mesh:
            Mock_mesh = AsyncMock(spec=ServiceMesh)
            Mock_mesh.register_service = AsyncMock(return_value=True)
            Mock_mesh.discover_services = AsyncMock(return_value=[
                {
                    "service_id": "mcp-test-0",
                    "address": "localhost",
                    "port": 11100,
                    "state": "healthy"
                }
            ])
            Mock_mesh.call_service = AsyncMock(return_value={"result": "test"})
            Mock_get_mesh.return_value = Mock_mesh
            
            # Test the flow
            from app.mesh.mcp_bridge import get_mcp_bridge
            
            bridge = await get_mcp_bridge(Mock_mesh)
            assert bridge is not None
            
            # Simulate service call
            result = await bridge.call_mcp_service(
                service_name="test",
                method="execute",
                params={}
            )
            
            assert result == {"result": "test"}

# Test runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--asyncio-mode=auto"])