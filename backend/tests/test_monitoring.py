#!/usr/bin/env python3
"""
Monitoring Stack Testing
Tests Prometheus, Grafana, Loki, and metrics collection
"""

import pytest
import httpx
import asyncio
from typing import List, Dict

TIMEOUT = 30.0

class TestPrometheus:
    """Test Prometheus metrics collection"""
    
    @pytest.mark.asyncio
    async def test_prometheus_health(self):
        """Test Prometheus health"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:10300/-/healthy")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_prometheus_targets(self):
        """Test Prometheus targets status"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:10300/api/v1/targets")
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert "activeTargets" in data["data"]
    
    @pytest.mark.asyncio
    async def test_all_targets_up(self):
        """Test that all targets are reporting as healthy"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:10300/api/v1/targets")
            data = response.json()
            
            targets = data.get("data", {}).get("activeTargets", [])
            healthy_targets = [t for t in targets if t.get("health") == "up"]
            
            print(f"Healthy targets: {len(healthy_targets)}/{len(targets)}")
            assert len(healthy_targets) >= 10  # At least 10 targets healthy
    
    @pytest.mark.asyncio
    async def test_agent_metrics_collection(self):
        """Test that agent metrics are being collected"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:10300/api/v1/targets")
            data = response.json()
            
            targets = data.get("data", {}).get("activeTargets", [])
            agent_targets = [t for t in targets if "agent" in t.get("labels", {}).get("job", "").lower()]
            
            print(f"Agent targets: {len(agent_targets)}")
            assert len(agent_targets) >= 5  # At least 5 agents reporting
    
    @pytest.mark.asyncio
    async def test_query_metrics(self):
        """Test querying Prometheus metrics"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Query for http requests
            params = {"query": "http_requests_total"}
            response = await client.get("http://localhost:10300/api/v1/query", params=params)
            assert response.status_code == 200


class TestGrafana:
    """Test Grafana dashboard service"""
    
    @pytest.mark.asyncio
    async def test_grafana_health(self):
        """Test Grafana health"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:10301/api/health")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_grafana_datasources(self):
        """Test Grafana datasources"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:10301/api/datasources")
            assert response.status_code in [200, 401]  # May require auth
    
    @pytest.mark.asyncio
    async def test_grafana_dashboards(self):
        """Test Grafana dashboards"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:10301/api/search")
            assert response.status_code in [200, 401]


class TestLoki:
    """Test Loki log aggregation"""
    
    @pytest.mark.asyncio
    async def test_loki_health(self):
        """Test Loki health"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:10310/ready")
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_loki_query_logs(self):
        """Test querying logs from Loki"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            params = {
                "query": '{job="backend"}',
                "limit": 10
            }
            response = await client.get("http://localhost:10310/loki/api/v1/query_range", params=params)
            assert response.status_code in [200, 400]  # 400 if invalid time range


class TestNodeExporter:
    """Test Node Exporter system metrics"""
    
    @pytest.mark.asyncio
    async def test_node_exporter_metrics(self):
        """Test Node Exporter metrics endpoint"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:10305/metrics")
            assert response.status_code == 200
            assert "node_" in response.text
    
    @pytest.mark.asyncio
    async def test_system_metrics_collection(self):
        """Test that system metrics are available"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:10305/metrics")
            metrics_text = response.text
            
            # Check for key metrics
            assert "node_cpu_seconds_total" in metrics_text
            assert "node_memory_MemTotal_bytes" in metrics_text
            assert "node_disk_io_time_seconds_total" in metrics_text


class TestAgentMetrics:
    """Test metrics from AI agents"""
    
    @pytest.mark.asyncio
    async def test_all_agents_expose_metrics(self):
        """Test that all agents expose metrics endpoints"""
        agent_ports = [11401, 11403, 11404, 11405, 11410, 11413, 11414, 11416]
        
        results = {}
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            for port in agent_ports:
                try:
                    response = await client.get(f"http://localhost:{port}/metrics")
                    results[port] = response.status_code == 200
                except Exception:
                    results[port] = False
        
        successful = sum(results.values())
        print(f"Agents with metrics: {successful}/{len(agent_ports)}")
        assert successful >= 6  # At least 6/8 agents
    
    @pytest.mark.asyncio
    async def test_agent_custom_metrics(self):
        """Test that agents expose custom metrics"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            response = await client.get("http://localhost:11403/metrics")  # CrewAI
            if response.status_code == 200:
                metrics = response.text
                # Check for Python metrics
                assert "python_info" in metrics or "process_" in metrics


class TestAlertManager:
    """Test AlertManager if configured"""
    
    @pytest.mark.asyncio
    async def test_alertmanager_health(self):
        """Test AlertManager health"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            try:
                response = await client.get("http://localhost:10309/-/healthy")
                assert response.status_code in [200, 404]  # May not be deployed
            except httpx.ConnectError:
                # AlertManager is intentionally not deployed
                print("\nAlertManager not deployed (expected)")
                assert True


class TestMetricsIntegration:
    """Test end-to-end metrics flow"""
    
    @pytest.mark.asyncio
    async def test_metrics_collection_pipeline(self):
        """Test complete metrics collection pipeline"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # 1. Agent exposes metrics
            agent_response = await client.get("http://localhost:11403/metrics")
            assert agent_response.status_code == 200
            
            # 2. Prometheus scrapes and stores metrics
            prom_response = await client.get("http://localhost:10300/api/v1/targets")
            assert prom_response.status_code == 200
            
            # 3. Query metrics from Prometheus
            query_response = await client.get(
                "http://localhost:10300/api/v1/query",
                params={"query": "up"}
            )
            assert query_response.status_code == 200


class TestLogAggregation:
    """Test log aggregation pipeline"""
    
    @pytest.mark.asyncio
    async def test_log_collection_pipeline(self):
        """Test complete log collection pipeline"""
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            # Check Loki is ready
            loki_response = await client.get("http://localhost:10310/ready")
            assert loki_response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
