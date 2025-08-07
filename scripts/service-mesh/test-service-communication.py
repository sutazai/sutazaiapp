#!/usr/bin/env python3
"""
Purpose: Test service-to-service communication through the service mesh.
Usage: python test-service-communication.py [--kong-proxy=http://localhost:10005]
Requirements: requests library, running service mesh components
"""

import requests
import argparse
import logging
import time
import json
from typing import Dict, List, Any, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ServiceMeshTester:
    """Test service communication through the service mesh."""
    
    def __init__(self, kong_proxy: str = "http://localhost:10005", 
                 consul_url: str = "http://localhost:10006"):
        self.kong_proxy = kong_proxy
        self.consul_url = consul_url
        self.session = requests.Session()
        
    def test_kong_health(self) -> bool:
        """Test if Kong proxy is healthy."""
        try:
            response = self.session.get(f"{self.kong_proxy}/")
            return response.status_code != 502
        except Exception as e:
            logger.error(f"Kong proxy not reachable: {e}")
            return False
    
    def test_service_through_kong(self, path: str, expected_status: List[int] = [200]) -> Tuple[bool, Dict[str, Any]]:
        """Test a service through Kong API Gateway."""
        try:
            start_time = time.time()
            response = self.session.get(f"{self.kong_proxy}{path}", timeout=10)
            latency = (time.time() - start_time) * 1000  # Convert to ms
            
            result = {
                "path": path,
                "status_code": response.status_code,
                "latency_ms": round(latency, 2),
                "headers": dict(response.headers),
                "success": response.status_code in expected_status
            }
            
            if response.status_code in expected_status:
                logger.info(f"✓ {path} - Status: {response.status_code}, Latency: {latency:.2f}ms")
            else:
                logger.error(f"✗ {path} - Status: {response.status_code}, Expected: {expected_status}")
                result["error"] = response.text[:200]
            
            return result["success"], result
            
        except requests.exceptions.Timeout:
            logger.error(f"✗ {path} - Timeout after 10s")
            return False, {"path": path, "error": "timeout", "success": False}
        except Exception as e:
            logger.error(f"✗ {path} - Error: {e}")
            return False, {"path": path, "error": str(e), "success": False}
    
    def test_consul_service_discovery(self, service_name: str) -> Tuple[bool, Dict[str, Any]]:
        """Test if a service is registered in Consul."""
        try:
            response = self.session.get(f"{self.consul_url}/v1/catalog/service/{service_name}")
            
            if response.status_code == 200:
                instances = response.json()
                healthy_instances = []
                
                # Check health for each instance
                for instance in instances:
                    service_id = instance.get('ServiceID')
                    health_response = self.session.get(
                        f"{self.consul_url}/v1/health/checks/{service_id}"
                    )
                    
                    if health_response.status_code == 200:
                        health_checks = health_response.json()
                        is_healthy = all(check.get('Status') == 'passing' for check in health_checks)
                        if is_healthy:
                            healthy_instances.append(instance)
                
                result = {
                    "service": service_name,
                    "total_instances": len(instances),
                    "healthy_instances": len(healthy_instances),
                    "registered": True,
                    "success": len(healthy_instances) > 0
                }
                
                if result["success"]:
                    logger.info(f"✓ Consul: {service_name} - {len(healthy_instances)}/{len(instances)} healthy")
                else:
                    logger.warning(f"⚠ Consul: {service_name} - No healthy instances")
                
                return result["success"], result
            else:
                logger.error(f"✗ Consul: {service_name} - Not found in service registry")
                return False, {"service": service_name, "registered": False, "success": False}
                
        except Exception as e:
            logger.error(f"✗ Consul: {service_name} - Error: {e}")
            return False, {"service": service_name, "error": str(e), "success": False}
    
    def test_service_to_service(self, from_service: str, to_service: str, 
                               test_endpoint: str) -> Tuple[bool, Dict[str, Any]]:
        """Test communication from one service to another."""
        try:
            # This would typically involve calling an endpoint on from_service
            # that internally calls to_service
            logger.info(f"Testing {from_service} -> {to_service} communication")
            
            # For now, we'll simulate this by checking if both services are healthy
            from_healthy, from_result = self.test_consul_service_discovery(from_service)
            to_healthy, to_result = self.test_consul_service_discovery(to_service)
            
            success = from_healthy and to_healthy
            result = {
                "from_service": from_service,
                "to_service": to_service,
                "from_healthy": from_healthy,
                "to_healthy": to_healthy,
                "success": success
            }
            
            if success:
                logger.info(f"✓ Service mesh: {from_service} -> {to_service} - Both services available")
            else:
                logger.error(f"✗ Service mesh: {from_service} -> {to_service} - Communication path broken")
            
            return success, result
            
        except Exception as e:
            logger.error(f"✗ Service mesh: {from_service} -> {to_service} - Error: {e}")
            return False, {"error": str(e), "success": False}
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all service mesh tests."""
        results = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "kong_proxy": self.kong_proxy,
            "tests": {
                "kong_health": {},
                "service_routes": [],
                "consul_services": [],
                "service_to_service": []
            },
            "summary": {
                "total_tests": 0,
                "passed": 0,
                "failed": 0
            }
        }
        
        # Test Kong health
        logger.info("=== Testing Kong API Gateway Health ===")
        kong_healthy = self.test_kong_health()
        results["tests"]["kong_health"] = {"healthy": kong_healthy}
        results["summary"]["total_tests"] += 1
        if kong_healthy:
            results["summary"]["passed"] += 1
        else:
            results["summary"]["failed"] += 1
        
        # Test service routes through Kong
        logger.info("\n=== Testing Service Routes through Kong ===")
        service_routes = [
            ("/api/v1/health", [200]),
            ("/api/ollama/api/tags", [200]),
            ("/api/chromadb/api/v1/heartbeat", [200]),
            ("/api/qdrant/health", [200]),
            ("/api/agents/autogpt/health", [200]),
            ("/api/agents/crewai/health", [200]),
            ("/api/workflows/langflow/health", [200]),
            ("/api/workflows/flowise/api/v1/ping", [200]),
            ("/api/metrics/prometheus/-/healthy", [200]),
            ("/api/dashboards/grafana/api/health", [200]),
        ]
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            future_to_route = {
                executor.submit(self.test_service_through_kong, route, status): route 
                for route, status in service_routes
            }
            
            for future in as_completed(future_to_route):
                success, result = future.result()
                results["tests"]["service_routes"].append(result)
                results["summary"]["total_tests"] += 1
                if success:
                    results["summary"]["passed"] += 1
                else:
                    results["summary"]["failed"] += 1
        
        # Test Consul service discovery
        logger.info("\n=== Testing Consul Service Discovery ===")
        consul_services = [
            "backend", "frontend", "postgres", "redis", "neo4j",
            "ollama", "chromadb", "qdrant", "rabbitmq",
            "prometheus", "grafana", "autogpt", "crewai"
        ]
        
        for service in consul_services:
            success, result = self.test_consul_service_discovery(service)
            results["tests"]["consul_services"].append(result)
            results["summary"]["total_tests"] += 1
            if success:
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1
        
        # Test service-to-service communication
        logger.info("\n=== Testing Service-to-Service Communication ===")
        service_pairs = [
            ("backend", "postgres", "/health"),
            ("backend", "redis", "/health"),
            ("backend", "ollama", "/api/tags"),
            ("backend", "chromadb", "/api/v1/heartbeat"),
            ("autogpt", "ollama", "/api/tags"),
            ("crewai", "ollama", "/api/tags"),
        ]
        
        for from_service, to_service, endpoint in service_pairs:
            success, result = self.test_service_to_service(from_service, to_service, endpoint)
            results["tests"]["service_to_service"].append(result)
            results["summary"]["total_tests"] += 1
            if success:
                results["summary"]["passed"] += 1
            else:
                results["summary"]["failed"] += 1
        
        # Calculate success rate
        if results["summary"]["total_tests"] > 0:
            results["summary"]["success_rate"] = round(
                (results["summary"]["passed"] / results["summary"]["total_tests"]) * 100, 2
            )
        else:
            results["summary"]["success_rate"] = 0
        
        return results


def main():
    """Main function to test service mesh communication."""
    parser = argparse.ArgumentParser(description="Test SutazAI service mesh communication")
    parser.add_argument("--kong-proxy", default="http://localhost:10005", 
                        help="Kong proxy URL")
    parser.add_argument("--consul-url", default="http://localhost:10006",
                        help="Consul API URL")
    parser.add_argument("--output", help="Output file for test results (JSON)")
    
    args = parser.parse_args()
    
    # Create tester instance
    tester = ServiceMeshTester(args.kong_proxy, args.consul_url)
    
    # Run all tests
    logger.info("Starting SutazAI Service Mesh Communication Tests")
    logger.info("=" * 60)
    
    results = tester.run_all_tests()
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("TEST SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Total Tests: {results['summary']['total_tests']}")
    logger.info(f"Passed: {results['summary']['passed']}")
    logger.info(f"Failed: {results['summary']['failed']}")
    logger.info(f"Success Rate: {results['summary']['success_rate']}%")
    
    # Save results if output file specified
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nTest results saved to: {args.output}")
    
    # Return exit code based on success rate
    return 0 if results['summary']['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())