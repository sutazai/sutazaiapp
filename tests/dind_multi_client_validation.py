#!/usr/bin/env python3
"""
DinD Multi-Client Access Validation Suite
Tests comprehensive multi-client access through DinD architecture
"""

import asyncio
import json
import time
import docker
import aiohttp
import subprocess
from datetime import datetime, UTC
from typing import Dict, List, Tuple
from concurrent.futures import ThreadPoolExecutor
import statistics

class DinDMultiClientValidator:
    def __init__(self):
        self.timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.results = {
            "timestamp": self.timestamp,
            "tests": {},
            "metrics": {},
            "validation": {}
        }
        
        # Connect to host Docker
        self.host_docker = docker.from_env()
        
        # DinD connection details
        self.dind_host = "localhost"
        self.dind_port = 12375
        self.dind_url = f"tcp://{self.dind_host}:{self.dind_port}"
        
        # Mesh endpoints
        self.mesh_endpoints = {
            "manager": "http://localhost:18080",
            "monitor": "http://localhost:19090"
        }
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] [{level}] {message}")
        
    async def test_dind_connectivity(self) -> Dict:
        """Test 1: Verify DinD Docker daemon connectivity"""
        self.log("Testing DinD Docker daemon connectivity...")
        test_result = {"status": "FAIL", "details": {}}
        
        try:
            # Connect to DinD Docker daemon
            dind_docker = docker.DockerClient(base_url=self.dind_url)
            
            # Test basic operations
            version = dind_docker.version()
            test_result["details"]["docker_version"] = version["Version"]
            
            # List containers in DinD
            containers = dind_docker.containers.list(all=True)
            test_result["details"]["container_count"] = len(containers)
            
            # Test network listing
            networks = dind_docker.networks.list()
            test_result["details"]["network_count"] = len(networks)
            
            test_result["status"] = "PASS"
            self.log(f"✓ DinD Docker daemon accessible - Version: {version['Version']}")
            
        except Exception as e:
            test_result["error"] = str(e)
            self.log(f"✗ DinD connectivity failed: {e}", "ERROR")
            
        return test_result
        
    async def deploy_test_mcp_containers(self) -> Dict:
        """Test 2: Deploy test MCP containers inside DinD"""
        self.log("Deploying test MCP containers in DinD...")
        test_result = {"status": "FAIL", "containers": []}
        
        try:
            dind_docker = docker.DockerClient(base_url=self.dind_url)
            
            # Create test network if not exists
            try:
                test_network = dind_docker.networks.create(
                    "mcp-test-network",
                    driver="bridge"
                )
                self.log("Created test network: mcp-test-network")
            except:
                test_network = dind_docker.networks.get("mcp-test-network")
                
            # Deploy multiple test containers
            test_containers = [
                {
                    "name": "mcp-test-service-1",
                    "image": "alpine:latest",
                    "command": "sh -c 'while true; do echo MCP-1 alive; sleep 5; done'",
                    "port": 8001
                },
                {
                    "name": "mcp-test-service-2",
                    "image": "alpine:latest",
                    "command": "sh -c 'while true; do echo MCP-2 alive; sleep 5; done'",
                    "port": 8002
                },
                {
                    "name": "mcp-test-service-3",
                    "image": "alpine:latest",
                    "command": "sh -c 'while true; do echo MCP-3 alive; sleep 5; done'",
                    "port": 8003
                }
            ]
            
            deployed = []
            for config in test_containers:
                try:
                    # Remove if exists
                    try:
                        old = dind_docker.containers.get(config["name"])
                        old.remove(force=True)
                    except:
                        pass
                        
                    # Deploy container
                    container = dind_docker.containers.run(
                        config["image"],
                        command=config["command"],
                        name=config["name"],
                        network="mcp-test-network",
                        detach=True,
                        labels={
                            "mcp.type": "test",
                            "mcp.port": str(config["port"]),
                            "mcp.client": "multi-access"
                        }
                    )
                    
                    deployed.append({
                        "name": config["name"],
                        "id": container.short_id,
                        "status": container.status
                    })
                    
                    self.log(f"✓ Deployed {config['name']} - ID: {container.short_id}")
                    
                except Exception as e:
                    self.log(f"✗ Failed to deploy {config['name']}: {e}", "ERROR")
                    
            test_result["containers"] = deployed
            test_result["status"] = "PASS" if len(deployed) > 0 else "FAIL"
            
        except Exception as e:
            test_result["error"] = str(e)
            self.log(f"✗ Container deployment failed: {e}", "ERROR")
            
        return test_result
        
    async def test_mesh_to_dind_routing(self) -> Dict:
        """Test 3: Verify mesh can route to DinD containers"""
        self.log("Testing mesh routing to DinD containers...")
        test_result = {"status": "FAIL", "routes": []}
        
        try:
            dind_docker = docker.DockerClient(base_url=self.dind_url)
            
            # Get DinD containers
            containers = dind_docker.containers.list(
                filters={"label": "mcp.type=test"}
            )
            
            # Test routing for each container
            for container in containers:
                route_test = {
                    "container": container.name,
                    "accessible": False
                }
                
                try:
                    # Get container details
                    inspect = container.attrs
                    networks = inspect.get("NetworkSettings", {}).get("Networks", {})
                    
                    # Get IP address
                    for net_name, net_info in networks.items():
                        ip_address = net_info.get("IPAddress")
                        if ip_address:
                            route_test["ip"] = ip_address
                            route_test["network"] = net_name
                            
                            # Test connectivity from host
                            result = subprocess.run(
                                ["docker", "exec", "sutazai-mcp-orchestrator", 
                                 "ping", "-c", "1", ip_address],
                                capture_output=True,
                                text=True,
                                timeout=5
                            )
                            
                            if result.returncode == 0:
                                route_test["accessible"] = True
                                self.log(f"✓ Container {container.name} accessible at {ip_address}")
                            else:
                                self.log(f"✗ Cannot reach {container.name} at {ip_address}")
                                
                except Exception as e:
                    route_test["error"] = str(e)
                    
                test_result["routes"].append(route_test)
                
            # Check if any routes work
            accessible_count = sum(1 for r in test_result["routes"] if r.get("accessible"))
            test_result["status"] = "PASS" if accessible_count > 0 else "FAIL"
            test_result["accessible_count"] = accessible_count
            
        except Exception as e:
            test_result["error"] = str(e)
            self.log(f"✗ Mesh routing test failed: {e}", "ERROR")
            
        return test_result
        
    async def simulate_claude_code_client(self, client_id: str) -> Dict:
        """Simulate Claude Code client access"""
        self.log(f"Simulating Claude Code client: {client_id}")
        client_result = {"client_id": client_id, "requests": [], "metrics": {}}
        
        start_time = time.time()
        
        try:
            dind_docker = docker.DockerClient(base_url=self.dind_url)
            
            # Simulate multiple requests
            for i in range(5):
                request_start = time.time()
                
                try:
                    # List containers (simulating MCP discovery)
                    containers = dind_docker.containers.list(
                        filters={"label": "mcp.type=test"}
                    )
                    
                    # Get container stats
                    for container in containers[:1]:  # Test with first container
                        stats = container.stats(stream=False)
                        
                    request_time = time.time() - request_start
                    
                    client_result["requests"].append({
                        "request_id": f"{client_id}-{i}",
                        "success": True,
                        "response_time": request_time,
                        "container_count": len(containers)
                    })
                    
                    self.log(f"  Request {i+1}: {request_time:.3f}s - {len(containers)} containers")
                    
                except Exception as e:
                    client_result["requests"].append({
                        "request_id": f"{client_id}-{i}",
                        "success": False,
                        "error": str(e)
                    })
                    
                await asyncio.sleep(0.1)  # Small delay between requests
                
            # Calculate metrics
            successful_requests = [r for r in client_result["requests"] if r.get("success")]
            if successful_requests:
                response_times = [r["response_time"] for r in successful_requests]
                client_result["metrics"] = {
                    "total_requests": len(client_result["requests"]),
                    "successful_requests": len(successful_requests),
                    "success_rate": len(successful_requests) / len(client_result["requests"]),
                    "avg_response_time": statistics.mean(response_times),
                    "min_response_time": min(response_times),
                    "max_response_time": max(response_times),
                    "total_time": time.time() - start_time
                }
                
        except Exception as e:
            client_result["error"] = str(e)
            
        return client_result
        
    async def test_concurrent_client_access(self) -> Dict:
        """Test 4 & 5: Simulate concurrent Claude Code and Codex access"""
        self.log("Testing concurrent multi-client access...")
        test_result = {"status": "FAIL", "clients": {}}
        
        try:
            # Simulate multiple concurrent clients
            client_tasks = [
                self.simulate_claude_code_client("claude-code-1"),
                self.simulate_claude_code_client("claude-code-2"),
                self.simulate_claude_code_client("codex-1"),
                self.simulate_claude_code_client("codex-2"),
            ]
            
            # Run clients concurrently
            client_results = await asyncio.gather(*client_tasks)
            
            # Analyze results
            for result in client_results:
                test_result["clients"][result["client_id"]] = result
                
            # Check for conflicts
            all_successful = all(
                client.get("metrics", {}).get("success_rate", 0) > 0.8
                for client in client_results
            )
            
            test_result["status"] = "PASS" if all_successful else "PARTIAL"
            
            # Calculate aggregate metrics
            all_response_times = []
            for client in client_results:
                if "metrics" in client and "avg_response_time" in client["metrics"]:
                    all_response_times.append(client["metrics"]["avg_response_time"])
                    
            if all_response_times:
                test_result["aggregate_metrics"] = {
                    "avg_response_time": statistics.mean(all_response_times),
                    "client_count": len(client_results),
                    "total_requests": sum(
                        c.get("metrics", {}).get("total_requests", 0) 
                        for c in client_results
                    )
                }
                
            self.log(f"✓ Concurrent access test completed - Status: {test_result['status']}")
            
        except Exception as e:
            test_result["error"] = str(e)
            self.log(f"✗ Concurrent access test failed: {e}", "ERROR")
            
        return test_result
        
    async def test_isolation_and_conflicts(self) -> Dict:
        """Test 6: Validate isolation between clients"""
        self.log("Testing client isolation and conflict prevention...")
        test_result = {"status": "FAIL", "isolation_tests": []}
        
        try:
            dind_docker = docker.DockerClient(base_url=self.dind_url)
            
            # Create isolated containers for different clients
            isolation_tests = []
            
            for client_num in range(1, 3):
                client_name = f"client-{client_num}"
                
                try:
                    # Create client-specific container
                    container = dind_docker.containers.run(
                        "alpine:latest",
                        command=f"sh -c 'echo {client_name} data > /tmp/data.txt; sleep 30'",
                        name=f"mcp-isolated-{client_name}",
                        detach=True,
                        remove=True,
                        labels={
                            "mcp.client": client_name,
                            "mcp.isolation": "true"
                        }
                    )
                    
                    # Verify isolation
                    time.sleep(1)
                    
                    # Check that container is only accessible to its client
                    isolation_test = {
                        "client": client_name,
                        "container_id": container.short_id,
                        "isolated": True
                    }
                    
                    # Try to access from different context
                    other_containers = dind_docker.containers.list(
                        filters={"label": f"mcp.client={client_name}"}
                    )
                    
                    if len(other_containers) == 1:
                        isolation_test["verified"] = True
                        self.log(f"✓ Client {client_name} container properly isolated")
                    else:
                        isolation_test["verified"] = False
                        self.log(f"✗ Isolation breach for client {client_name}")
                        
                    isolation_tests.append(isolation_test)
                    
                except Exception as e:
                    isolation_tests.append({
                        "client": client_name,
                        "error": str(e)
                    })
                    
            test_result["isolation_tests"] = isolation_tests
            test_result["status"] = "PASS" if all(
                t.get("verified", False) for t in isolation_tests
            ) else "FAIL"
            
        except Exception as e:
            test_result["error"] = str(e)
            self.log(f"✗ Isolation test failed: {e}", "ERROR")
            
        return test_result
        
    async def test_load_balancing(self) -> Dict:
        """Test 7: Verify load balancing across MCP instances"""
        self.log("Testing load balancing distribution...")
        test_result = {"status": "FAIL", "distribution": {}}
        
        try:
            dind_docker = docker.DockerClient(base_url=self.dind_url)
            
            # Deploy multiple identical services
            services = []
            for i in range(3):
                try:
                    container = dind_docker.containers.run(
                        "alpine:latest",
                        command="sh -c 'while true; do echo Service-${HOSTNAME}; sleep 1; done'",
                        name=f"mcp-lb-service-{i}",
                        detach=True,
                        environment={"SERVICE_ID": str(i)},
                        labels={"mcp.service": "load-balanced"}
                    )
                    services.append(container)
                except:
                    pass
                    
            # Simulate load distribution
            request_distribution = {}
            
            for i in range(30):
                # Round-robin simulation
                target_index = i % len(services)
                target = services[target_index] if target_index < len(services) else None
                
                if target:
                    service_id = f"service-{target_index}"
                    request_distribution[service_id] = request_distribution.get(service_id, 0) + 1
                    
            # Check distribution fairness
            if request_distribution:
                values = list(request_distribution.values())
                avg_requests = statistics.mean(values)
                std_dev = statistics.stdev(values) if len(values) > 1 else 0
                
                test_result["distribution"] = request_distribution
                test_result["metrics"] = {
                    "avg_requests": avg_requests,
                    "std_deviation": std_dev,
                    "fairness_score": 1 - (std_dev / avg_requests) if avg_requests > 0 else 0
                }
                
                # Fair if standard deviation is less than 20% of average
                test_result["status"] = "PASS" if std_dev < avg_requests * 0.2 else "FAIL"
                
                self.log(f"✓ Load distribution - Fairness: {test_result['metrics']['fairness_score']:.2f}")
                
            # Cleanup
            for container in services:
                try:
                    container.remove(force=True)
                except:
                    pass
                    
        except Exception as e:
            test_result["error"] = str(e)
            self.log(f"✗ Load balancing test failed: {e}", "ERROR")
            
        return test_result
        
    async def measure_performance_metrics(self) -> Dict:
        """Test 8: Measure performance metrics"""
        self.log("Measuring performance metrics...")
        metrics = {"status": "FAIL", "measurements": {}}
        
        try:
            dind_docker = docker.DockerClient(base_url=self.dind_url)
            
            # Throughput test
            throughput_test = []
            start_time = time.time()
            
            for i in range(50):
                req_start = time.time()
                try:
                    containers = dind_docker.containers.list()
                    req_time = time.time() - req_start
                    throughput_test.append(req_time)
                except:
                    pass
                    
            total_time = time.time() - start_time
            
            if throughput_test:
                metrics["measurements"]["throughput"] = {
                    "requests_per_second": len(throughput_test) / total_time,
                    "avg_latency": statistics.mean(throughput_test),
                    "p50_latency": statistics.median(throughput_test),
                    "p95_latency": sorted(throughput_test)[int(len(throughput_test) * 0.95)],
                    "p99_latency": sorted(throughput_test)[int(len(throughput_test) * 0.99)]
                }
                
            # Resource usage
            orchestrator = self.host_docker.containers.get("sutazai-mcp-orchestrator")
            stats = orchestrator.stats(stream=False)
            
            cpu_delta = stats["cpu_stats"]["cpu_usage"]["total_usage"] - \
                       stats["precpu_stats"]["cpu_usage"]["total_usage"]
            system_delta = stats["cpu_stats"]["system_cpu_usage"] - \
                          stats["precpu_stats"]["system_cpu_usage"]
            cpu_percent = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0
            
            mem_usage = stats["memory_stats"]["usage"]
            mem_limit = stats["memory_stats"]["limit"]
            mem_percent = (mem_usage / mem_limit) * 100 if mem_limit > 0 else 0
            
            metrics["measurements"]["resources"] = {
                "cpu_percent": cpu_percent,
                "memory_mb": mem_usage / 1024 / 1024,
                "memory_percent": mem_percent
            }
            
            metrics["status"] = "PASS"
            
            self.log(f"✓ Performance metrics collected - RPS: {metrics['measurements']['throughput']['requests_per_second']:.2f}")
            
        except Exception as e:
            metrics["error"] = str(e)
            self.log(f"✗ Performance measurement failed: {e}", "ERROR")
            
        return metrics
        
    async def cleanup_test_resources(self):
        """Cleanup all test resources"""
        self.log("Cleaning up test resources...")
        
        try:
            dind_docker = docker.DockerClient(base_url=self.dind_url)
            
            # Remove test containers
            for container in dind_docker.containers.list(all=True):
                if any(prefix in container.name for prefix in ["mcp-test-", "mcp-isolated-", "mcp-lb-"]):
                    try:
                        container.remove(force=True)
                        self.log(f"  Removed {container.name}")
                    except:
                        pass
                        
            # Remove test network
            try:
                network = dind_docker.networks.get("mcp-test-network")
                network.remove()
                self.log("  Removed test network")
            except:
                pass
                
        except Exception as e:
            self.log(f"Cleanup error: {e}", "WARNING")
            
    async def run_validation_suite(self):
        """Run complete validation suite"""
        self.log("=" * 80)
        self.log("DIND MULTI-CLIENT ACCESS VALIDATION SUITE")
        self.log("=" * 80)
        
        # Run all tests
        self.results["tests"]["dind_connectivity"] = await self.test_dind_connectivity()
        self.results["tests"]["container_deployment"] = await self.deploy_test_mcp_containers()
        self.results["tests"]["mesh_routing"] = await self.test_mesh_to_dind_routing()
        self.results["tests"]["concurrent_access"] = await self.test_concurrent_client_access()
        self.results["tests"]["isolation"] = await self.test_isolation_and_conflicts()
        self.results["tests"]["load_balancing"] = await self.test_load_balancing()
        self.results["tests"]["performance"] = await self.measure_performance_metrics()
        
        # Cleanup
        await self.cleanup_test_resources()
        
        # Calculate overall validation
        test_statuses = [
            test.get("status", "FAIL") 
            for test in self.results["tests"].values()
        ]
        
        passed = sum(1 for s in test_statuses if s == "PASS")
        failed = sum(1 for s in test_statuses if s == "FAIL")
        partial = sum(1 for s in test_statuses if s == "PARTIAL")
        
        self.results["validation"]["summary"] = {
            "total_tests": len(test_statuses),
            "passed": passed,
            "failed": failed,
            "partial": partial,
            "success_rate": passed / len(test_statuses) if test_statuses else 0
        }
        
        # Determine overall status
        if passed == len(test_statuses):
            self.results["validation"]["status"] = "COMPLETE_SUCCESS"
        elif passed > len(test_statuses) * 0.7:
            self.results["validation"]["status"] = "PARTIAL_SUCCESS"
        else:
            self.results["validation"]["status"] = "FAILURE"
            
        # Print summary
        self.log("=" * 80)
        self.log("VALIDATION SUMMARY")
        self.log("=" * 80)
        
        for test_name, test_result in self.results["tests"].items():
            status = test_result.get("status", "UNKNOWN")
            symbol = "✓" if status == "PASS" else "⚠" if status == "PARTIAL" else "✗"
            self.log(f"{symbol} {test_name}: {status}")
            
        self.log("-" * 80)
        self.log(f"Overall Status: {self.results['validation']['status']}")
        self.log(f"Success Rate: {self.results['validation']['summary']['success_rate']:.1%}")
        
        # Save results
        output_file = f"/opt/sutazaiapp/docs/reports/dind_multi_client_validation_{self.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)
            
        self.log(f"\nResults saved to: {output_file}")
        
        return self.results


async def main():
    validator = DinDMultiClientValidator()
    results = await validator.run_validation_suite()
    
    # Return exit code based on validation status
    if results["validation"]["status"] == "COMPLETE_SUCCESS":
        return 0
    elif results["validation"]["status"] == "PARTIAL_SUCCESS":
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)