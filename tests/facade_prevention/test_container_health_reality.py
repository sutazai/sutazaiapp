#!/usr/bin/env python3
"""
Container Health Reality Tests - Facade Prevention Framework
===========================================================

This module implements comprehensive tests to prevent facade implementations in container health.
Tests verify that containers are actually healthy and functioning, not just claiming to be.

CRITICAL PURPOSE: Prevent orphaned containers and facade health checks that claim containers 
are healthy when they're actually broken or non-functional.
"""

import asyncio
import pytest
import docker
import json
import time
import subprocess
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import logging
import requests
import socket

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContainerHealthRealityTester:
    """
    Tests that verify containers are actually healthy rather than just claiming to be.
    
    FACADE PREVENTION: These tests catch containers that report healthy status 
    but are actually broken, orphaned, or non-functional.
    """
    
    def __init__(self):
        self.docker_client = None
        self.expected_containers = {
            "sutazai-postgres": {"port": 10000, "health_check": self._check_postgres_health},
            "sutazai-redis": {"port": 10001, "health_check": self._check_redis_health},
            "sutazai-neo4j": {"port": 10002, "health_check": self._check_neo4j_health},
            "sutazai-backend": {"port": 10010, "health_check": self._check_backend_health},
            "sutazai-frontend": {"port": 10011, "health_check": self._check_frontend_health},
            "sutazai-ollama": {"port": 10104, "health_check": self._check_ollama_health},
            "sutazai-chromadb": {"port": 10100, "health_check": self._check_chromadb_health},
            "sutazai-qdrant": {"port": 10101, "health_check": self._check_qdrant_health},
            "sutazai-prometheus": {"port": 10200, "health_check": self._check_prometheus_health},
            "sutazai-grafana": {"port": 10201, "health_check": self._check_grafana_health},
            "sutazai-kong": {"port": 10005, "health_check": self._check_kong_health},
            "sutazai-consul": {"port": 10006, "health_check": self._check_consul_health}
        }
    
    async def __aenter__(self):
        self.docker_client = docker.from_env()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.docker_client:
            self.docker_client.close()
    
    def get_all_containers(self) -> List[Dict]:
        """Get all containers, including stopped and orphaned ones."""
        try:
            containers = self.docker_client.containers.list(all=True)
            container_info = []
            
            for container in containers:
                info = {
                    "id": container.id,
                    "name": container.name,
                    "status": container.status,
                    "image": container.image.tags[0] if container.image.tags else "unknown",
                    "created": container.attrs.get("Created", ""),
                    "state": container.attrs.get("State", {}),
                    "config": container.attrs.get("Config", {}),
                    "network_settings": container.attrs.get("NetworkSettings", {})
                }
                container_info.append(info)
            
            return container_info
            
        except Exception as e:
            logger.error(f"Failed to get container list: {e}")
            return []
    
    async def test_orphaned_container_detection(self) -> Dict:
        """
        FACADE TEST: Detect orphaned containers that shouldn't exist.
        
        PREVENTS: Orphaned containers accumulating and causing resource waste or conflicts.
        """
        logger.info("🔍 Testing for orphaned containers...")
        
        all_containers = self.get_all_containers()
        
        # Identify containers that match sutazai pattern
        sutazai_containers = [c for c in all_containers if "sutazai" in c["name"].lower()]
        
        # Check for expected vs actual containers
        expected_names = set(self.expected_containers.keys())
        actual_names = set(c["name"] for c in sutazai_containers if c["status"] == "running")
        
        # Find orphaned containers (running but not expected)
        orphaned_containers = [
            c for c in sutazai_containers 
            if c["status"] == "running" and c["name"] not in expected_names
        ]
        
        # Find duplicate containers (multiple containers with similar names)
        name_counts = {}
        for container in sutazai_containers:
            base_name = container["name"].split("-")[0:2]  # e.g., sutazai-postgres
            base_name_str = "-".join(base_name)
            name_counts[base_name_str] = name_counts.get(base_name_str, 0) + 1
        
        duplicate_groups = {name: count for name, count in name_counts.items() if count > 1}
        
        # Find stopped containers that might be orphans
        stopped_containers = [c for c in sutazai_containers if c["status"] in ["exited", "stopped"]]
        
        return {
            "total_containers": len(all_containers),
            "sutazai_containers": len(sutazai_containers),
            "expected_containers": len(expected_names),
            "running_containers": len(actual_names),
            "orphaned_containers": len(orphaned_containers),
            "orphaned_container_details": orphaned_containers,
            "duplicate_groups": duplicate_groups,
            "stopped_containers": len(stopped_containers),
            "stopped_container_details": stopped_containers,
            "test_passed": len(orphaned_containers) == 0 and len(duplicate_groups) == 0
        }
    
    async def test_container_health_facade_detection(self) -> Dict:
        """
        FACADE TEST: Verify containers claiming to be healthy actually are.
        
        PREVENTS: Health checks passing but containers not actually functional.
        """
        logger.info("🏥 Testing container health reality...")
        
        health_results = {}
        total_containers = 0
        healthy_containers = 0
        facade_containers = 0
        
        for container_name, config in self.expected_containers.items():
            total_containers += 1
            
            try:
                container = self.docker_client.containers.get(container_name)
                
                # Get Docker's health status
                docker_health_status = container.attrs.get("State", {}).get("Health", {}).get("Status", "none")
                docker_status = container.status
                
                # REALITY CHECK: Actually test if the service works
                actual_health = await config["health_check"](config["port"])
                
                is_facade = (
                    docker_status == "running" and 
                    docker_health_status in ["healthy", "none"] and 
                    not actual_health["is_healthy"]
                )
                
                if actual_health["is_healthy"]:
                    healthy_containers += 1
                
                if is_facade:
                    facade_containers += 1
                
                health_results[container_name] = {
                    "docker_status": docker_status,
                    "docker_health_status": docker_health_status,
                    "actual_health": actual_health,
                    "is_facade": is_facade,
                    "test_passed": actual_health["is_healthy"]
                }
                
            except docker.errors.NotFound:
                health_results[container_name] = {
                    "docker_status": "not_found",
                    "docker_health_status": "not_found",
                    "actual_health": {"is_healthy": False, "error": "Container not found"},
                    "is_facade": False,
                    "test_passed": False
                }
            except Exception as e:
                health_results[container_name] = {
                    "docker_status": "error",
                    "docker_health_status": "error",
                    "actual_health": {"is_healthy": False, "error": str(e)},
                    "is_facade": False,
                    "test_passed": False
                }
        
        facade_ratio = facade_containers / total_containers if total_containers > 0 else 0
        healthy_ratio = healthy_containers / total_containers if total_containers > 0 else 0
        
        return {
            "total_containers": total_containers,
            "healthy_containers": healthy_containers,
            "facade_containers": facade_containers,
            "healthy_ratio": healthy_ratio,
            "facade_ratio": facade_ratio,
            "health_results": health_results,
            "test_passed": facade_ratio < 0.2 and healthy_ratio > 0.7  # Less than 20% facades, more than 70% healthy
        }
    
    async def test_port_binding_reality(self) -> Dict:
        """
        FACADE TEST: Verify containers actually bind to their claimed ports.
        
        PREVENTS: Containers claiming to expose ports but not actually binding to them.
        """
        logger.info("🔌 Testing port binding reality...")
        
        port_results = {}
        total_ports = 0
        working_ports = 0
        facade_ports = 0
        
        for container_name, config in self.expected_containers.items():
            port = config["port"]
            total_ports += 1
            
            try:
                container = self.docker_client.containers.get(container_name)
                
                # Check Docker's port configuration
                docker_ports = container.attrs.get("NetworkSettings", {}).get("Ports", {})
                docker_claims_port = any(str(port) in p for p in docker_ports.keys())
                
                # REALITY CHECK: Actually test if port is accessible
                port_accessible = await self._test_port_accessibility("localhost", port)
                
                is_facade = docker_claims_port and not port_accessible
                
                if port_accessible:
                    working_ports += 1
                
                if is_facade:
                    facade_ports += 1
                
                port_results[f"{container_name}:{port}"] = {
                    "docker_claims_port": docker_claims_port,
                    "port_accessible": port_accessible,
                    "is_facade": is_facade,
                    "test_passed": port_accessible
                }
                
            except docker.errors.NotFound:
                port_results[f"{container_name}:{port}"] = {
                    "docker_claims_port": False,
                    "port_accessible": False,
                    "is_facade": False,
                    "test_passed": False,
                    "error": "Container not found"
                }
            except Exception as e:
                port_results[f"{container_name}:{port}"] = {
                    "docker_claims_port": False,
                    "port_accessible": False,
                    "is_facade": False,
                    "test_passed": False,
                    "error": str(e)
                }
        
        return {
            "total_ports": total_ports,
            "working_ports": working_ports,
            "facade_ports": facade_ports,
            "port_results": port_results,
            "test_passed": facade_ports == 0 and working_ports > total_ports * 0.7
        }
    
    async def _test_port_accessibility(self, host: str, port: int, timeout: float = 5.0) -> bool:
        """Test if a port is actually accessible."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    # Health check methods for different services
    async def _check_postgres_health(self, port: int) -> Dict:
        """Check PostgreSQL actual health."""
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host="localhost",
                port=port,
                database="sutazai",
                user="sutazai",
                password="sutazai123",
                timeout=5.0
            )
            result = await conn.fetchval("SELECT 1")
            await conn.close()
            return {"is_healthy": result == 1, "details": "query_successful"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def _check_redis_health(self, port: int) -> Dict:
        """Check Redis actual health."""
        try:
            import redis.asyncio as redis
            client = redis.Redis(host="localhost", port=port, decode_responses=True, socket_timeout=5)
            pong = await client.ping()
            await client.aclose()
            return {"is_healthy": pong is True, "details": "ping_successful"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def _check_neo4j_health(self, port: int) -> Dict:
        """Check Neo4j actual health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/")
                return {"is_healthy": response.status_code == 200, "details": f"status_code_{response.status_code}"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def _check_backend_health(self, port: int) -> Dict:
        """Check FastAPI backend actual health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/health")
                if response.status_code == 200:
                    health_data = response.json()
                    return {"is_healthy": health_data.get("status") == "healthy", "details": health_data}
                else:
                    return {"is_healthy": False, "details": f"status_code_{response.status_code}"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def _check_frontend_health(self, port: int) -> Dict:
        """Check Streamlit frontend actual health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/")
                return {"is_healthy": response.status_code == 200, "details": f"status_code_{response.status_code}"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def _check_ollama_health(self, port: int) -> Dict:
        """Check Ollama actual health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"http://localhost:{port}/api/tags")
                if response.status_code == 200:
                    models = response.json()
                    return {"is_healthy": True, "details": f"models_available_{len(models.get('models', []))}"}
                else:
                    return {"is_healthy": False, "details": f"status_code_{response.status_code}"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def _check_chromadb_health(self, port: int) -> Dict:
        """Check ChromaDB actual health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/api/v1/heartbeat")
                return {"is_healthy": response.status_code == 200, "details": f"status_code_{response.status_code}"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def _check_qdrant_health(self, port: int) -> Dict:
        """Check Qdrant actual health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/")
                return {"is_healthy": response.status_code == 200, "details": f"status_code_{response.status_code}"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def _check_prometheus_health(self, port: int) -> Dict:
        """Check Prometheus actual health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/-/healthy")
                return {"is_healthy": response.status_code == 200, "details": f"status_code_{response.status_code}"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def _check_grafana_health(self, port: int) -> Dict:
        """Check Grafana actual health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/api/health")
                return {"is_healthy": response.status_code == 200, "details": f"status_code_{response.status_code}"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def _check_kong_health(self, port: int) -> Dict:
        """Check Kong API Gateway actual health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/")
                return {"is_healthy": response.status_code in [200, 404], "details": f"status_code_{response.status_code}"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def _check_consul_health(self, port: int) -> Dict:
        """Check Consul actual health."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"http://localhost:{port}/v1/status/leader")
                return {"is_healthy": response.status_code == 200, "details": f"status_code_{response.status_code}"}
        except Exception as e:
            return {"is_healthy": False, "error": str(e)}
    
    async def run_comprehensive_container_tests(self) -> Dict:
        """Run all container health reality tests and return comprehensive results."""
        logger.info("🚀 Starting comprehensive container health reality tests...")
        
        start_time = datetime.now()
        
        results = {
            "test_suite": "container_health_facade_prevention",
            "timestamp": start_time.isoformat(),
            "tests": {}
        }
        
        # Run all tests
        test_methods = [
            ("orphaned_container_detection", self.test_orphaned_container_detection),
            ("container_health_facade_detection", self.test_container_health_facade_detection),
            ("port_binding_reality", self.test_port_binding_reality)
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_name, test_method in test_methods:
            try:
                logger.info(f"Running {test_name} test...")
                test_result = await test_method()
                results["tests"][test_name] = test_result
                
                if test_result.get("test_passed", False):
                    passed_tests += 1
                    
            except Exception as e:
                logger.error(f"Test {test_name} failed with exception: {e}")
                results["tests"][test_name] = {
                    "status": "error",
                    "error": str(e),
                    "test_passed": False
                }
        
        # Calculate overall results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results.update({
            "summary": {
                "total_tests": total_tests,
                "passed_tests": passed_tests,
                "failed_tests": total_tests - passed_tests,
                "success_rate": passed_tests / total_tests,
                "duration_seconds": duration
            },
            "overall_status": "passed" if passed_tests == total_tests else "failed"
        })
        
        logger.info(f"Container health reality tests completed: {passed_tests}/{total_tests} passed")
        return results


# Pytest integration
@pytest.mark.asyncio
async def test_containers_are_not_facades():
    """
    Main facade prevention test for container health.
    
    This test MUST pass for deployment to prevent facade implementations.
    """
    async with ContainerHealthRealityTester() as tester:
        results = await tester.run_comprehensive_container_tests()
        
        # CRITICAL: Fail if any facade issues detected
        assert results["overall_status"] == "passed", f"Container health reality tests failed: {results}"
        
        # Check for specific facade issues
        health_test = results["tests"].get("container_health_facade_detection", {})
        facade_containers = health_test.get("facade_containers", 0)
        assert facade_containers == 0, f"Facade containers detected: {facade_containers}"
        
        # Log results for monitoring
        logger.info(f"✅ Container health reality verification passed: {results['summary']}")


@pytest.mark.asyncio
async def test_no_orphaned_containers():
    """Test that no orphaned containers exist."""
    async with ContainerHealthRealityTester() as tester:
        result = await tester.test_orphaned_container_detection()
        assert result["orphaned_containers"] == 0, f"Orphaned containers detected: {result['orphaned_container_details']}"


@pytest.mark.asyncio
async def test_container_ports_actually_work():
    """Test that container ports are actually accessible."""
    async with ContainerHealthRealityTester() as tester:
        result = await tester.test_port_binding_reality()
        assert result["facade_ports"] == 0, f"Facade ports detected: {result['port_results']}"
        assert result["working_ports"] > 0, "No container ports are working"


if __name__ == "__main__":
    async def main():
        async with ContainerHealthRealityTester() as tester:
            results = await tester.run_comprehensive_container_tests()
            print(json.dumps(results, indent=2))
            
            if results["overall_status"] != "passed":
                print(f"\n❌ CONTAINER HEALTH FACADE ISSUES DETECTED")
                exit(1)
            else:
                print(f"\n✅ All container health reality tests passed!")
                exit(0)
    
    asyncio.run(main())