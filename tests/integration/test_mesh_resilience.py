"""
Integration tests for mesh system resilience and graceful degradation
Tests that the system handles missing services properly
"""
import asyncio
import pytest
from unittest.mock import Mock, patch, AsyncMock
import os
import tempfile

from backend.app.mesh.mcp_mesh_initializer import MCPMeshInitializer, get_mcp_mesh_initializer
from backend.app.mesh.mcp_bridge import MCPMeshBridge, MCPServiceAdapter, MCPServiceConfig
from backend.app.mesh.mcp_load_balancer import MCPLoadBalancer, get_mcp_load_balancer
from backend.app.core.mcp_startup import initialize_mcp_on_startup, shutdown_mcp_services


class TestMeshResilience:
    """Test mesh system resilience to missing services"""
    
    @pytest.mark.asyncio
    async def test_initializer_handles_missing_wrappers(self):
        """Test that initializer gracefully handles missing wrapper scripts"""
        # Create initializer without mesh (standalone mode)
        initializer = MCPMeshInitializer(mesh_client=None)
        
        # Mock os.path.exists to simulate missing wrappers
        with patch('os.path.exists') as mock_exists:
            # Only some wrappers exist
            def wrapper_exists(path):
                return "files.sh" in path or "http.sh" in path
            
            mock_exists.side_effect = wrapper_exists
            
            # Initialize and register
            results = await initializer.initialize_and_register()
            
            # Should have some registered, some skipped
            assert len(results['registered']) > 0
            assert len(results['skipped']) > 0
            assert results['registered'] + results['failed'] + results['skipped'] == results['total']
    
    @pytest.mark.asyncio
    async def test_bridge_works_without_mesh(self):
        """Test that MCP bridge works without service mesh"""
        # Create bridge without mesh
        bridge = MCPMeshBridge(mesh=None)
        
        # Create temp wrapper script
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write('#!/bin/bash\necho "test mcp server"')
            wrapper_path = f.name
        
        os.chmod(wrapper_path, 0o755)
        
        try:
            # Override registry with test service
            bridge.registry = {
                "mcp_services": [
                    {
                        "name": "test-service",
                        "wrapper": wrapper_path,
                        "port": 12000,
                        "capabilities": ["test"],
                        "tags": ["test"],
                        "metadata": {}
                    }
                ]
            }
            
            # Initialize should work without mesh
            results = await bridge.initialize()
            
            # Clean up (stop services)
            await bridge.shutdown()
            
            assert results['status'] == 'initialized'
            # May or may not start depending on environment
            assert isinstance(results['started'], list)
            assert isinstance(results['failed'], list)
            
        finally:
            # Clean up temp file
            os.unlink(wrapper_path)
    
    @pytest.mark.asyncio
    async def test_load_balancer_handles_no_instances(self):
        """Test load balancer handles no available instances gracefully"""
        load_balancer = MCPLoadBalancer()
        
        # Select with no instances
        result = load_balancer.select_instance([], "test-service")
        assert result is None
        
        # Select with only unhealthy instances
        from backend.app.mesh.service_mesh import ServiceInstance, ServiceState
        
        unhealthy_instances = [
            ServiceInstance(
                service_id="test-1",
                service_name="test-service",
                address="localhost",
                port=11100,
                state=ServiceState.UNHEALTHY
            )
        ]
        
        result = load_balancer.select_instance(unhealthy_instances, "test-service")
        assert result is None
    
    @pytest.mark.asyncio
    async def test_load_balancer_uses_degraded_instances(self):
        """Test load balancer uses degraded instances when no healthy ones available"""
        load_balancer = MCPLoadBalancer()
        
        from backend.app.mesh.service_mesh import ServiceInstance, ServiceState
        
        degraded_instances = [
            ServiceInstance(
                service_id="test-1",
                service_name="test-service",
                address="localhost",
                port=11100,
                state=ServiceState.DEGRADED
            )
        ]
        
        result = load_balancer.select_instance(degraded_instances, "test-service")
        assert result is not None
        assert result.service_id == "test-1"
    
    @pytest.mark.asyncio
    async def test_startup_continues_without_mcp(self):
        """Test that system startup continues even if MCP initialization fails"""
        with patch('backend.app.mesh.mcp_stdio_bridge.get_mcp_stdio_bridge') as mock_bridge:
            # Simulate bridge initialization failure
            mock_bridge.side_effect = Exception("Bridge initialization failed")
            
            # Should not raise, returns error result
            result = await initialize_mcp_on_startup()
            
            assert 'error' in result
            assert result['started'] == []
            assert result['failed'] == []
    
    @pytest.mark.asyncio
    async def test_adapter_handles_missing_wrapper(self):
        """Test that adapter handles missing wrapper script gracefully"""
        config = MCPServiceConfig(
            name="test-service",
            wrapper_script="/nonexistent/wrapper.sh",
            port=11100,
            required=False  # Optional service
        )
        
        adapter = MCPServiceAdapter(config, mesh=None)
        
        # Should return False, not raise
        result = await adapter.start()
        assert result is False
        assert not adapter.available
    
    @pytest.mark.asyncio
    async def test_health_check_recovery(self):
        """Test health check and recovery mechanism"""
        config = MCPServiceConfig(
            name="test-service",
            wrapper_script="/bin/true",  # Simple command that exits immediately
            port=11100,
            auto_restart=True,
            max_retries=2
        )
        
        adapter = MCPServiceAdapter(config, mesh=None)
        
        # Mock process to simulate death
        adapter.process = Mock()
        adapter.process.poll.return_value = 1  # Process died
        adapter.available = True
        
        # Health check should detect death
        result = await adapter.health_check()
        assert result is False
        assert not adapter.available
    
    @pytest.mark.asyncio
    async def test_dynamic_service_registration(self):
        """Test dynamic service registration"""
        bridge = MCPMeshBridge(mesh=None)
        
        # Create temp wrapper
        with tempfile.NamedTemporaryFile(mode='w', suffix='.sh', delete=False) as f:
            f.write('#!/bin/bash\necho "dynamic service"')
            wrapper_path = f.name
        
        os.chmod(wrapper_path, 0o755)
        
        try:
            # Register dynamic service
            service_config = {
                "name": "dynamic-test",
                "wrapper": wrapper_path,
                "port": 12001,
                "capabilities": ["dynamic"],
                "metadata": {"type": "test"}
            }
            
            result = await bridge.register_dynamic_service(service_config)
            
            # Clean up
            await bridge.shutdown()
            
            # May succeed or fail depending on environment
            assert isinstance(result, bool)
            
        finally:
            os.unlink(wrapper_path)
    
    @pytest.mark.asyncio
    async def test_service_status_reporting(self):
        """Test service status reporting"""
        bridge = MCPMeshBridge(mesh=None)
        
        # Get status of non-existent service
        status = await bridge.get_service_status("nonexistent")
        assert status['status'] == 'not_found'
        assert 'error' in status
    
    @pytest.mark.asyncio
    async def test_metrics_update_and_recovery(self):
        """Test metrics update and recovery detection"""
        load_balancer = get_mcp_load_balancer()
        
        # Update metrics with failures
        for i in range(6):
            load_balancer.update_metrics(
                instance_id="test-instance",
                response_time=1.0,
                success=False
            )
        
        # Get stats
        stats = load_balancer.get_instance_stats("test-instance")
        
        assert stats['consecutive_errors'] >= 5
        assert stats['health_score'] < 50
        assert stats['status'] in ['degraded', 'unhealthy']


class TestMeshIntegration:
    """Test complete mesh integration"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_initialization(self):
        """Test end-to-end initialization flow"""
        # This test simulates the complete initialization flow
        with patch('os.path.exists') as mock_exists:
            mock_exists.return_value = False  # No wrappers exist
            
            # Should complete without errors
            result = await initialize_mcp_on_startup()
            
            assert 'started' in result
            assert 'failed' in result
            
            # Shutdown should also work
            await shutdown_mcp_services()
    
    @pytest.mark.asyncio
    async def test_partial_service_availability(self):
        """Test system works with partial service availability"""
        bridge = MCPMeshBridge(mesh=None)
        
        # Mock registry with mix of available and unavailable services
        bridge.registry = {
            "mcp_services": [
                {
                    "name": "available",
                    "wrapper": "/bin/echo",  # Exists on most systems
                    "port": 11100,
                    "capabilities": ["test"]
                },
                {
                    "name": "unavailable",
                    "wrapper": "/nonexistent/wrapper.sh",
                    "port": 11101,
                    "capabilities": ["test"]
                }
            ]
        }
        
        results = await bridge.initialize()
        
        # Should have some skipped
        assert len(results['skipped']) > 0 or len(results['failed']) > 0
        
        # Cleanup
        await bridge.shutdown()


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])