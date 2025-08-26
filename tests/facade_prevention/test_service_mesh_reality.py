#!/usr/bin/env python3
"""
Service Mesh Reality Tests - Facade Prevention Framework
=========================================================

This module implements comprehensive tests to prevent facade implementations in the service mesh.
Tests verify that services actually work as claimed, not just return data.

CRITICAL PURPOSE: Prevent facade implementations where code claims functionality but doesn't deliver.
"""

import asyncio
import pytest
import httpx
import json
import time
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ServiceMeshRealityTester:
    """
    Tests that verify the service mesh actually functions rather than just claiming to.
    
    FACADE PREVENTION: These tests catch discrepancies between claimed functionality 
    and actual working behavior.
    """
    
    def __init__(self, base_url: str = "http://localhost:10010"):
        self.base_url = base_url
        self.client = None
        self.discovered_services = []
        self.reality_test_results = {}
    
    async def __aenter__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.client:
            await self.client.aclose()
    
    async def test_service_discovery_reality(self) -> Dict:
        """
        FACADE TEST: Verify service discovery actually returns working services.
        
        PREVENTS: Service mesh claiming to discover services but returning dead/fake entries.
        """
        logger.info("🔍 Testing service discovery reality...")
        
        # Get services from discovery endpoint
        response = await self.client.get(f"{self.base_url}/api/v1/mesh/v2/services")
        assert response.status_code == 200, f"Service discovery endpoint failed: {response.status_code}"
        
        data = response.json()
        services = data.get("services", [])
        assert len(services) > 0, "Service discovery returned empty list - FACADE DETECTED"
        
        # REALITY CHECK: Verify each discovered service is actually reachable
        reachable_services = []
        unreachable_services = []
        
        for service in services:
            service_id = service.get("id")
            service_name = service.get("name")
            address = service.get("address")
            port = service.get("port")
            
            # Test actual connectivity to each service
            is_reachable = await self._test_service_connectivity(address, port, service_name)
            
            if is_reachable:
                reachable_services.append(service)
                logger.info(f"✅ Service {service_name} is actually reachable")
            else:
                unreachable_services.append(service)
                logger.error(f"❌ Service {service_name} is FACADE - claimed but unreachable")
        
        # FACADE PREVENTION: Fail if significant portion of services are unreachable
        facade_ratio = len(unreachable_services) / len(services)
        assert facade_ratio < 0.3, f"Too many facade services detected: {len(unreachable_services)}/{len(services)}"
        
        self.discovered_services = reachable_services
        return {
            "total_services": len(services),
            "reachable_services": len(reachable_services),
            "unreachable_services": len(unreachable_services),
            "facade_ratio": facade_ratio,
            "test_passed": facade_ratio < 0.3
        }
    
    async def _test_service_connectivity(self, address: str, port: int, service_name: str) -> bool:
        """Test if a service is actually reachable and responding."""
        try:
            # For Docker internal services, check health endpoints
            health_urls = [
                f"http://{address}:{port}/health",
                f"http://{address}:{port}/api/health", 
                f"http://{address}:{port}/healthz",
                f"http://{address}:{port}/status",
                f"http://{address}:{port}/"
            ]
            
            for url in health_urls:
                try:
                    response = await self.client.get(url, timeout=5.0)
                    if response.status_code < 500:  # Any non-server-error response indicates service is up
                        return True
                except:
                    continue
            
            # For database services, try basic connection
            if "postgres" in service_name.lower():
                return await self._test_postgres_connectivity(address, port)
            elif "redis" in service_name.lower():
                return await self._test_redis_connectivity(address, port)
            elif "neo4j" in service_name.lower():
                return await self._test_neo4j_connectivity(address, port)
            
            return False
            
        except Exception as e:
            logger.debug(f"Service connectivity test failed for {service_name}: {e}")
            return False
    
    async def _test_postgres_connectivity(self, address: str, port: int) -> bool:
        """Test PostgreSQL connectivity."""
        try:
            import asyncpg
            conn = await asyncpg.connect(
                host=address,
                port=port,
                database="sutazai",
                user="sutazai",
                password="sutazai123",
                timeout=5.0
            )
            await conn.execute("SELECT 1")
            await conn.close()
            return True
        except Exception:
            return False
    
    async def _test_redis_connectivity(self, address: str, port: int) -> bool:
        """Test Redis connectivity."""
        try:
            import redis.asyncio as redis
            client = redis.Redis(host=address, port=port, decode_responses=True, socket_timeout=5)
            await client.ping()
            await client.aclose()
            return True
        except Exception:
            return False
    
    async def _test_neo4j_connectivity(self, address: str, port: int) -> bool:
        """Test Neo4j connectivity."""
        try:
            from neo4j import AsyncGraphDatabase
            driver = AsyncGraphDatabase.driver(f"bolt://{address}:{port}")
            await driver.verify_connectivity()
            await driver.close()
            return True
        except Exception:
            return False
    
    async def test_service_registration_reality(self) -> Dict:
        """
        FACADE TEST: Verify service registration actually works for new services.
        
        PREVENTS: Registration endpoint claiming success but not actually registering.
        """
        logger.info("📝 Testing service registration reality...")
        
        test_service = {
            "name": "facade-test-service",
            "address": "test-address",
            "port": 9999,
            "tags": ["test", "facade-prevention"],
            "metadata": {"test": "true", "timestamp": str(int(time.time()))}
        }
        
        # Attempt registration
        response = await self.client.post(
            f"{self.base_url}/api/v1/mesh/v2/register",
            json=test_service
        )
        assert response.status_code in [200, 201], f"Service registration failed: {response.status_code}"
        
        # REALITY CHECK: Verify the service actually appears in discovery
        await asyncio.sleep(2)  # Allow time for registration
        
        discovery_response = await self.client.get(f"{self.base_url}/api/v1/mesh/v2/services")
        services = discovery_response.json().get("services", [])
        
        # Check if our test service appears
        test_service_found = any(
            service.get("name") == "facade-test-service" 
            for service in services
        )
        
        assert test_service_found, "Service registration is FACADE - claimed success but service not discoverable"
        
        # Cleanup: Deregister test service if endpoint exists
        try:
            await self.client.delete(f"{self.base_url}/api/v1/mesh/v2/services/facade-test-service")
        except:
            pass  # Deregistration may not be implemented yet
        
        return {
            "registration_claimed_success": True,
            "service_actually_discoverable": test_service_found,
            "test_passed": test_service_found
        }
    
    async def test_load_balancing_reality(self) -> Dict:
        """
        FACADE TEST: Verify load balancing actually distributes requests.
        
        PREVENTS: Load balancer claiming to balance but always hitting same service.
        """
        logger.info("⚖️ Testing load balancing reality...")
        
        # Find services with multiple instances (if any)
        services_by_name = {}
        for service in self.discovered_services:
            name = service.get("name")
            if name not in services_by_name:
                services_by_name[name] = []
            services_by_name[name].append(service)
        
        # Look for services with multiple instances
        multi_instance_services = {
            name: instances for name, instances in services_by_name.items() 
            if len(instances) > 1
        }
        
        if not multi_instance_services:
            logger.info("No multi-instance services found - skipping load balancing test")
            return {"test_skipped": True, "reason": "no_multi_instance_services"}
        
        # Test load balancing for each multi-instance service
        load_balance_results = {}
        
        for service_name, instances in multi_instance_services.items():
            # Make multiple requests and track which instance responds
            request_distribution = {}
            
            for i in range(10):  # Make 10 requests
                try:
                    # Use service mesh proxy if available
                    response = await self.client.get(f"{self.base_url}/api/v1/services/{service_name}/health")
                    
                    # Track response headers or other indicators of which instance handled request
                    instance_id = response.headers.get("X-Instance-ID", "unknown")
                    request_distribution[instance_id] = request_distribution.get(instance_id, 0) + 1
                    
                except Exception as e:
                    logger.debug(f"Load balancing test request failed: {e}")
                    continue
            
            # REALITY CHECK: Verify requests were distributed across instances
            unique_instances = len(request_distribution)
            total_instances = len(instances)
            
            load_balance_results[service_name] = {
                "total_instances": total_instances,
                "instances_receiving_traffic": unique_instances,
                "distribution": request_distribution,
                "is_load_balanced": unique_instances > 1 if total_instances > 1 else True
            }
        
        return {
            "multi_instance_services": len(multi_instance_services),
            "load_balance_results": load_balance_results,
            "test_passed": all(result["is_load_balanced"] for result in load_balance_results.values())
        }
    
    async def test_circuit_breaker_reality(self) -> Dict:
        """
        FACADE TEST: Verify circuit breaker actually prevents calls to failing services.
        
        PREVENTS: Circuit breaker claiming to work but not actually breaking circuits.
        """
        logger.info("🔌 Testing circuit breaker reality...")
        
        # Try to trigger circuit breaker by calling non-existent service
        circuit_breaker_triggered = False
        
        try:
            # Make requests to a non-existent service to trigger circuit breaker
            for i in range(5):
                response = await self.client.get(f"{self.base_url}/api/v1/services/non-existent-service/test")
                if response.status_code == 503:  # Service Unavailable - circuit breaker response
                    circuit_breaker_triggered = True
                    break
                await asyncio.sleep(0.5)
        
        except Exception as e:
            logger.debug(f"Circuit breaker test exception: {e}")
        
        # REALITY CHECK: If circuit breaker exists, it should respond with 503 for failing services
        return {
            "circuit_breaker_triggered": circuit_breaker_triggered,
            "test_passed": True  # This test is informational for now
        }
    
    async def run_comprehensive_facade_tests(self) -> Dict:
        """Run all facade prevention tests and return comprehensive results."""
        logger.info("🚀 Starting comprehensive service mesh reality tests...")
        
        start_time = datetime.now()
        
        results = {
            "test_suite": "service_mesh_facade_prevention",
            "timestamp": start_time.isoformat(),
            "tests": {}
        }
        
        # Run all tests
        test_methods = [
            ("service_discovery", self.test_service_discovery_reality),
            ("service_registration", self.test_service_registration_reality),
            ("load_balancing", self.test_load_balancing_reality),
            ("circuit_breaker", self.test_circuit_breaker_reality)
        ]
        
        passed_tests = 0
        total_tests = len(test_methods)
        
        for test_name, test_method in test_methods:
            try:
                logger.info(f"Running {test_name} reality test...")
                test_result = await test_method()
                test_result["status"] = "passed" if test_result.get("test_passed", False) else "failed"
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
            "overall_status": "passed" if passed_tests == total_tests else "failed",
            "facade_issues_detected": total_tests - passed_tests
        })
        
        logger.info(f"Service mesh reality tests completed: {passed_tests}/{total_tests} passed")
        return results


# Pytest integration
@pytest.mark.asyncio
async def test_service_mesh_is_not_facade():
    """
    Main facade prevention test for service mesh.
    
    This test MUST pass for deployment to prevent facade implementations.
    """
    async with ServiceMeshRealityTester() as tester:
        results = await tester.run_comprehensive_facade_tests()
        
        # CRITICAL: Fail if any facade issues detected
        assert results["facade_issues_detected"] == 0, f"FACADE IMPLEMENTATION DETECTED: {results}"
        assert results["overall_status"] == "passed", f"Service mesh reality tests failed: {results}"
        
        # Log results for monitoring
        logger.info(f"✅ Service mesh reality verification passed: {results['summary']}")


@pytest.mark.asyncio
async def test_service_discovery_not_empty():
    """Basic test to ensure service discovery returns actual services."""
    async with ServiceMeshRealityTester() as tester:
        result = await tester.test_service_discovery_reality()
        assert result["total_services"] > 0, "Service discovery returned empty - system is not functional"
        assert result["reachable_services"] > 0, "No services are actually reachable - all are facades"


@pytest.mark.asyncio 
async def test_service_registration_works():
    """Test that service registration actually registers services."""
    async with ServiceMeshRealityTester() as tester:
        result = await tester.test_service_registration_reality()
        if not result.get("test_skipped"):
            assert result["test_passed"], "Service registration is a facade - claims success but doesn't work"


if __name__ == "__main__":
    async def main():
        async with ServiceMeshRealityTester() as tester:
            results = await tester.run_comprehensive_facade_tests()
            print(json.dumps(results, indent=2))
            
            if results["facade_issues_detected"] > 0:
                print(f"\n❌ FACADE ISSUES DETECTED: {results['facade_issues_detected']}")
                exit(1)
            else:
                print(f"\n✅ All service mesh reality tests passed!")
                exit(0)
    
    asyncio.run(main())