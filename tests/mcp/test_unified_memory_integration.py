#!/usr/bin/env python3
"""
TDD Test Suite for Unified Memory Service Backend Integration
Tests unified memory service integration replacing extended-memory and memory-bank-mcp
"""

import pytest
import asyncio
import httpx
from unittest. import , patch, Async
from fastapi.testclient import TestClient
import json

# These tests should fail initially (RED phase)
class TestUnifiedMemoryIntegration:
    """Test suite for unified memory service backend integration"""
    
    @pytest.fixture
    async def client(self):
        """Test client for backend API"""
        # This will fail until we implement the routing
        from app.main import app
        return TestClient(app)
    
    @pytest.fixture
    def memory_request_payload(self):
        """Standard memory request for testing"""
        return {
            "key": "test_key",
            "content": "test content",
            "namespace": "test",
            "tags": ["test", "integration"],
            "importance_level": 8,
            "ttl": 3600
        }
    
    def test_unified_memory_store_endpoint_exists(self, client):
        """RED: Test that unified memory store endpoint exists"""
        # This should fail initially
        response = client.post(
            "/api/v1/mcp/unified-memory/store",
            json={
                "key": "test",
                "content": "test content",
                "namespace": "default"
            }
        )
        assert response.status_code != 404, "Unified memory store endpoint should exist"
    
    def test_unified_memory_retrieve_endpoint_exists(self, client):
        """RED: Test that unified memory retrieve endpoint exists"""
        # This should fail initially
        response = client.get("/api/v1/mcp/unified-memory/retrieve/test")
        assert response.status_code != 404, "Unified memory retrieve endpoint should exist"
    
    def test_extended_memory_routes_deprecated(self, client):
        """RED: Test that extended-memory routes are deprecated/redirected"""
        # This should fail until we implement deprecation
        response = client.post("/api/v1/mcp/extended-memory/save_context")
        assert response.status_code in [301, 302, 410], "Extended memory should be deprecated"
    
    def test_memory_bank_routes_deprecated(self, client):
        """RED: Test that memory-bank-mcp routes are deprecated/redirected"""
        # This should fail until we implement deprecation
        response = client.get("/api/v1/mcp/memory-bank-mcp/contexts")
        assert response.status_code in [301, 302, 410], "Memory bank should be deprecated"
    
    def test_unified_memory_service_health_check(self, client):
        """RED: Test unified memory service health via backend"""
        response = client.get("/api/v1/mcp/unified-memory/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "unified-memory"
    
    def test_unified_memory_store_functionality(self, client, memory_request_payload):
        """RED: Test storing memory via unified service"""
        response = client.post(
            "/api/v1/mcp/unified-memory/store",
            json=memory_request_payload
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "context_id" in data
    
    def test_unified_memory_retrieve_functionality(self, client, memory_request_payload):
        """RED: Test retrieving memory via unified service"""
        # First store something
        store_response = client.post(
            "/api/v1/mcp/unified-memory/store",
            json=memory_request_payload
        )
        assert store_response.status_code == 200
        
        # Then retrieve it
        response = client.get(
            f"/api/v1/mcp/unified-memory/retrieve/{memory_request_payload['key']}",
            params={"namespace": memory_request_payload["namespace"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["key"] == memory_request_payload["key"]
    
    def test_unified_memory_search_functionality(self, client):
        """RED: Test search functionality via unified service"""
        response = client.get(
            "/api/v1/mcp/unified-memory/search",
            params={"query": "test", "namespace": "test", "limit": 5}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "results" in data["data"]
    
    def test_unified_memory_stats_endpoint(self, client):
        """RED: Test stats endpoint via unified service"""
        response = client.get("/api/v1/mcp/unified-memory/stats")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "total_memories" in data["data"]
    
    def test_unified_memory_delete_functionality(self, client, memory_request_payload):
        """RED: Test delete functionality via unified service"""
        # First store something
        store_response = client.post(
            "/api/v1/mcp/unified-memory/store",
            json=memory_request_payload
        )
        assert store_response.status_code == 200
        
        # Then delete it
        response = client.delete(
            f"/api/v1/mcp/unified-memory/delete/{memory_request_payload['key']}",
            params={"namespace": memory_request_payload["namespace"]}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["data"]["deleted"] is True


class TestMCPServiceConfiguration:
    """Test MCP service configuration updates"""
    
    def test_mcp_adapter_includes_unified_memory(self):
        """RED: Test that MCP adapter includes unified-memory configuration"""
        from app.mesh.mcp_adapter import MCP_SERVERS
        
        assert "unified-memory" in MCP_SERVERS, "unified-memory should be in MCP_SERVERS"
        
        config = MCP_SERVERS["unified-memory"]
        assert config.name == "unified-memory"
        assert config.server_type.value in ["docker", "python"]
        assert "memory" in config.tags
    
    def test_mcp_adapter_excludes_deprecated_services(self):
        """RED: Test that deprecated memory services are removed from config"""
        from app.mesh.mcp_adapter import MCP_SERVERS
        
        assert "extended-memory" not in MCP_SERVERS, "extended-memory should be removed"
        assert "memory-bank-mcp" not in MCP_SERVERS, "memory-bank-mcp should be removed"
    
    def test_unified_memory_service_discovery(self):
        """RED: Test service discovery includes unified memory"""
        # This will fail until we update service discovery
        from app.mesh.mcp_stdio_bridge import MCPStdioBridge
        
        bridge = MCPStdioBridge()
        services = bridge.get_available_services()
        
        assert "unified-memory" in services, "unified-memory should be discoverable"
        assert "extended-memory" not in services, "extended-memory should not be discoverable"
        assert "memory-bank-mcp" not in services, "memory-bank-mcp should not be discoverable"


class TestBackwardCompatibility:
    """Test backward compatibility and migration support"""
    
    def test_extended_memory_migration_endpoint(self, client):
        """RED: Test migration endpoint for extended-memory data"""
        response = client.post("/api/v1/mcp/migration/extended-memory-to-unified")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "migrated_count" in data
    
    def test_memory_bank_migration_endpoint(self, client):
        """RED: Test migration endpoint for memory-bank-mcp data"""
        response = client.post("/api/v1/mcp/migration/memory-bank-to-unified")
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "migrated_count" in data
    
    def test_legacy_endpoint_deprecation_warnings(self, client):
        """RED: Test that legacy endpoints return deprecation warnings"""
        response = client.get("/api/v1/mcp/extended-memory/load_contexts")
        
        # Should either redirect or return deprecation warning
        if response.status_code == 200:
            # If still working, should have deprecation header
            assert "X-Deprecated" in response.headers
            assert "unified-memory" in response.headers.get("X-Deprecated", "")


class TestPerformanceRequirements:
    """Test performance requirements for unified service"""
    
    @pytest.mark.asyncio
    async def test_store_latency_requirement(self, client):
        """RED: Test that store operations meet latency requirements (<50ms)"""
        import time
        
        start_time = time.time()
        response = client.post(
            "/api/v1/mcp/unified-memory/store",
            json={
                "key": "perf_test",
                "content": "performance test content",
                "namespace": "performance"
            }
        )
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        assert response.status_code == 200
        assert latency < 50, f"Store latency {latency}ms should be <50ms"
    
    @pytest.mark.asyncio
    async def test_retrieve_latency_requirement(self, client):
        """RED: Test that retrieve operations meet latency requirements (<10ms)"""
        import time
        
        # First store something
        client.post(
            "/api/v1/mcp/unified-memory/store",
            json={
                "key": "perf_retrieve_test",
                "content": "retrieve performance test",
                "namespace": "performance"
            }
        )
        
        # Then test retrieve latency
        start_time = time.time()
        response = client.get(
            "/api/v1/mcp/unified-memory/retrieve/perf_retrieve_test",
            params={"namespace": "performance"}
        )
        latency = (time.time() - start_time) * 1000  # Convert to ms
        
        assert response.status_code == 200
        assert latency < 10, f"Retrieve latency {latency}ms should be <10ms"


# Run tests to see failures (RED phase)
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])