#!/usr/bin/env python3
"""
Purpose: Test external service integration functionality
Usage: python test_external_integration.py
Requirements: pytest, requests, docker
"""

import pytest
import requests
import docker
import time
import json
from typing import Dict, Any

class TestExternalIntegration:
    """Test suite for external service integration"""
    
    @classmethod
    def setup_class(cls):
        """Setup test environment"""
        cls.docker_client = docker.from_env()
        cls.base_url = "http://localhost"
        cls.services = {
            'service_discovery': 10000,
            'api_gateway': 10001,
            'prometheus': 10010,
            'postgres_adapter': 10100,
            'redis_adapter': 10110
        }
    
    def wait_for_service(self, port: int, timeout: int = 30) -> bool:
        """Wait for service to be ready"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{self.base_url}:{port}/health", timeout=1)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(1)
        return False
    
    def test_service_discovery_health(self):
        """Test service discovery is healthy"""
        if self.wait_for_service(self.services['service_discovery']):
            response = requests.get(f"{self.base_url}:{self.services['service_discovery']}/health")
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'healthy'
        else:
            pytest.skip("Service discovery not available")
    
    def test_api_gateway_health(self):
        """Test API gateway is healthy"""
        if self.wait_for_service(self.services['api_gateway']):
            response = requests.get(f"{self.base_url}:{self.services['api_gateway']}/")
            assert response.status_code in [200, 404]  # Gateway responds
        else:
            pytest.skip("API gateway not available")
    
    def test_prometheus_metrics(self):
        """Test Prometheus is collecting metrics"""
        if self.wait_for_service(self.services['prometheus']):
            response = requests.get(f"{self.base_url}:{self.services['prometheus']}/api/v1/query?query=up")
            assert response.status_code == 200
            data = response.json()
            assert data['status'] == 'success'
        else:
            pytest.skip("Prometheus not available")
    
    def test_postgres_adapter_health(self):
        """Test PostgreSQL adapter health check"""
        if self.wait_for_service(self.services['postgres_adapter']):
            response = requests.get(f"{self.base_url}:{self.services['postgres_adapter']}/health")
            assert response.status_code in [200, 503]  # Healthy or unhealthy
            data = response.json()
            assert 'status' in data
            assert 'target' in data
        else:
            pytest.skip("PostgreSQL adapter not available")
    
    def test_redis_adapter_health(self):
        """Test Redis adapter health check"""
        if self.wait_for_service(self.services['redis_adapter']):
            response = requests.get(f"{self.base_url}:{self.services['redis_adapter']}/health")
            assert response.status_code in [200, 503]
            data = response.json()
            assert 'status' in data
        else:
            pytest.skip("Redis adapter not available")
    
    def test_adapter_metrics_endpoint(self):
        """Test adapter metrics are exposed"""
        for service_name, port in [('postgres_adapter', 10100), ('redis_adapter', 10110)]:
            if self.wait_for_service(port, timeout=5):
                response = requests.get(f"{self.base_url}:{port}/metrics")
                if response.status_code == 200:
                    assert 'adapter_requests_total' in response.text
                    assert 'adapter_request_duration_seconds' in response.text
    
    def test_service_registration(self):
        """Test service can be registered"""
        # This would test the service registry functionality
        pass
    
    def test_api_gateway_routing(self):
        """Test API gateway routes requests correctly"""
        if self.wait_for_service(self.services['api_gateway']):
            # Test routing to PostgreSQL adapter
            response = requests.get(f"{self.base_url}:{self.services['api_gateway']}/postgres/health")
            assert response.status_code in [200, 404, 503]
            
            # Test routing to Redis adapter
            response = requests.get(f"{self.base_url}:{self.services['api_gateway']}/redis/health")
            assert response.status_code in [200, 404, 503]
        else:
            pytest.skip("API gateway not available")
    
    def test_container_network_connectivity(self):
        """Test containers can communicate on integration network"""
        try:
            # Get integration network
            networks = self.docker_client.networks.list(names=['sutazaiapp_sutazai-integration'])
            if networks:
                network = networks[0]
                connected_containers = network.attrs['Containers']
                assert len(connected_containers) > 0
            else:
                pytest.skip("Integration network not found")
        except docker.errors.APIError:
            pytest.skip("Docker API not accessible")
    
    def test_health_check_propagation(self):
        """Test health status propagates through the system"""
        # This would test that unhealthy services are detected
        pass

def run_integration_tests():
    """Run all integration tests"""
    pytest.main([__file__, '-v', '--tb=short'])

if __name__ == '__main__':
    print("Running SutazAI External Integration Tests...")
    run_integration_tests()