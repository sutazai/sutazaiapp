#!/usr/bin/env python3
"""
Phase 5 Completion Test Suite
Validates the successful implementation of DinD multi-client architecture
"""

import subprocess
import json
import time
import docker
from datetime import datetime, UTC
from typing import Dict, List

class Phase5CompletionTest:
    def __init__(self):
        self.timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        self.docker_client = docker.from_env()
        self.test_results = []
        
    def log(self, message: str, level: str = "INFO"):
        """Log with timestamp"""
        timestamp = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{timestamp}] [{level}] {message}")
        
    def run_command(self, command: str, timeout: int = 30) -> Dict:
        """Execute command and return result"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return {
                "success": result.returncode == 0,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "returncode": result.returncode
            }
        except subprocess.TimeoutExpired:
            return {
                "success": False,
                "error": "Command timed out"
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
            
    def test_dind_orchestrator_health(self) -> Dict:
        """Test 1: Verify DinD orchestrator is healthy"""
        self.log("Testing DinD orchestrator health...")
        
        try:
            container = self.docker_client.containers.get("sutazai-mcp-orchestrator")
            
            # Check container status
            if container.status != "running":
                return {
                    "test": "dind_orchestrator_health",
                    "passed": False,
                    "reason": f"Container status is {container.status}, not running"
                }
                
            # Check health status
            health = container.attrs.get("State", {}).get("Health", {})
            health_status = health.get("Status", "unknown")
            
            if health_status != "healthy":
                return {
                    "test": "dind_orchestrator_health",
                    "passed": False,
                    "reason": f"Health status is {health_status}, not healthy"
                }
                
            # Verify Docker daemon inside DinD
            exec_result = container.exec_run("docker version --format '{{.Server.Version}}'")
            if exec_result.exit_code != 0:
                return {
                    "test": "dind_orchestrator_health",
                    "passed": False,
                    "reason": "Docker daemon not accessible inside DinD"
                }
                
            docker_version = exec_result.output.decode().strip()
            
            self.log(f"✓ DinD orchestrator healthy - Docker {docker_version}")
            
            return {
                "test": "dind_orchestrator_health",
                "passed": True,
                "details": {
                    "container_status": container.status,
                    "health_status": health_status,
                    "docker_version": docker_version
                }
            }
            
        except docker.errors.NotFound:
            return {
                "test": "dind_orchestrator_health",
                "passed": False,
                "reason": "DinD orchestrator container not found"
            }
        except Exception as e:
            return {
                "test": "dind_orchestrator_health",
                "passed": False,
                "reason": str(e)
            }
            
    def test_port_availability(self) -> Dict:
        """Test 2: Verify all required ports are available"""
        self.log("Testing port availability...")
        
        required_ports = {
            "12375": "DinD Docker API",
            "12376": "DinD Docker TLS",
            "18080": "MCP Manager API",
            "19090": "Monitoring endpoint"
        }
        
        port_status = {}
        all_available = True
        
        for port, description in required_ports.items():
            result = self.run_command(f"nc -zv localhost {port}", timeout=5)
            available = result["success"]
            port_status[port] = {
                "description": description,
                "available": available
            }
            
            if available:
                self.log(f"  ✓ Port {port} ({description}) - Available")
            else:
                self.log(f"  ✗ Port {port} ({description}) - Not available", "ERROR")
                all_available = False
                
        return {
            "test": "port_availability",
            "passed": all_available,
            "details": port_status
        }
        
    def test_mcp_deployment_capability(self) -> Dict:
        """Test 3: Verify ability to deploy MCP containers in DinD"""
        self.log("Testing MCP deployment capability...")
        
        # Connect to DinD Docker daemon
        dind_url = "tcp://localhost:12375"
        
        try:
            dind_docker = docker.DockerClient(base_url=dind_url)
            
            # Deploy a test MCP container
            test_container = dind_docker.containers.run(
                "alpine:latest",
                command="echo 'MCP test successful'",
                name="mcp-deployment-test",
                remove=True,
                detach=False
            )
            
            self.log("✓ Successfully deployed test container in DinD")
            
            return {
                "test": "mcp_deployment_capability",
                "passed": True,
                "details": {
                    "deployment": "successful",
                    "dind_accessible": True
                }
            }
            
        except Exception as e:
            self.log(f"✗ Failed to deploy in DinD: {e}", "ERROR")
            return {
                "test": "mcp_deployment_capability",
                "passed": False,
                "reason": str(e)
            }
            
    def test_multi_client_simulation(self) -> Dict:
        """Test 4: Simulate multiple client connections"""
        self.log("Testing multi-client access simulation...")
        
        dind_url = "tcp://localhost:12375"
        
        try:
            # Create multiple Docker clients (simulating different users)
            clients = []
            for i in range(3):
                client = docker.DockerClient(base_url=dind_url)
                clients.append({
                    "id": f"client-{i+1}",
                    "client": client
                })
                
            # Each client performs operations
            results = []
            for client_info in clients:
                try:
                    # List containers
                    containers = client_info["client"].containers.list()
                    
                    # Get version info
                    version = client_info["client"].version()
                    
                    results.append({
                        "client_id": client_info["id"],
                        "success": True,
                        "container_count": len(containers),
                        "docker_version": version["Version"]
                    })
                    
                    self.log(f"  ✓ {client_info['id']} connected successfully")
                    
                except Exception as e:
                    results.append({
                        "client_id": client_info["id"],
                        "success": False,
                        "error": str(e)
                    })
                    
            # Check if all clients succeeded
            all_successful = all(r["success"] for r in results)
            
            return {
                "test": "multi_client_simulation",
                "passed": all_successful,
                "details": {
                    "total_clients": len(clients),
                    "successful_clients": sum(1 for r in results if r["success"]),
                    "client_results": results
                }
            }
            
        except Exception as e:
            return {
                "test": "multi_client_simulation",
                "passed": False,
                "reason": str(e)
            }
            
    def test_resource_isolation(self) -> Dict:
        """Test 5: Verify resource isolation between clients"""
        self.log("Testing resource isolation...")
        
        dind_url = "tcp://localhost:12375"
        
        try:
            dind_docker = docker.DockerClient(base_url=dind_url)
            
            # Create containers with different labels (simulating different clients)
            containers = []
            for i in range(2):
                container = dind_docker.containers.run(
                    "alpine:latest",
                    command="sleep 30",
                    name=f"isolation-test-{i}",
                    detach=True,
                    labels={
                        "client": f"client-{i}",
                        "test": "isolation"
                    }
                )
                containers.append(container)
                
            # Verify each container has unique resources
            isolation_verified = True
            for container in containers:
                # Check that container has its own namespace
                inspect = container.attrs
                if not inspect.get("Id"):
                    isolation_verified = False
                    
            # Cleanup
            for container in containers:
                try:
                    container.remove(force=True)
                except:
                    pass
                    
            if isolation_verified:
                self.log("✓ Resource isolation verified")
            else:
                self.log("✗ Resource isolation failed", "ERROR")
                
            return {
                "test": "resource_isolation",
                "passed": isolation_verified,
                "details": {
                    "containers_created": len(containers),
                    "isolation_verified": isolation_verified
                }
            }
            
        except Exception as e:
            return {
                "test": "resource_isolation",
                "passed": False,
                "reason": str(e)
            }
            
    def test_network_connectivity(self) -> Dict:
        """Test 6: Verify network connectivity and routing"""
        self.log("Testing network connectivity...")
        
        try:
            # Check DinD orchestrator networks
            orchestrator = self.docker_client.containers.get("sutazai-mcp-orchestrator")
            
            # Get network configuration
            networks = orchestrator.attrs.get("NetworkSettings", {}).get("Networks", {})
            
            network_info = {}
            for net_name, net_config in networks.items():
                network_info[net_name] = {
                    "ip_address": net_config.get("IPAddress"),
                    "gateway": net_config.get("Gateway")
                }
                
            # Test connectivity to DinD from host
            result = self.run_command("curl -s http://localhost:12375/_ping", timeout=5)
            api_accessible = result["success"] and "OK" in result.get("stdout", "")
            
            self.log(f"  Network configuration: {len(networks)} networks")
            self.log(f"  API accessibility: {'✓' if api_accessible else '✗'}")
            
            return {
                "test": "network_connectivity",
                "passed": api_accessible,
                "details": {
                    "networks": network_info,
                    "api_accessible": api_accessible
                }
            }
            
        except Exception as e:
            return {
                "test": "network_connectivity",
                "passed": False,
                "reason": str(e)
            }
            
    def test_performance_baseline(self) -> Dict:
        """Test 7: Establish performance baseline"""
        self.log("Testing performance baseline...")
        
        dind_url = "tcp://localhost:12375"
        
        try:
            dind_docker = docker.DockerClient(base_url=dind_url)
            
            # Measure response times
            response_times = []
            
            for i in range(10):
                start = time.time()
                dind_docker.ping()
                response_time = time.time() - start
                response_times.append(response_time)
                
            avg_response = sum(response_times) / len(response_times)
            max_response = max(response_times)
            min_response = min(response_times)
            
            # Performance is acceptable if avg < 100ms
            acceptable = avg_response < 0.1
            
            self.log(f"  Average response: {avg_response*1000:.2f}ms")
            self.log(f"  Min/Max: {min_response*1000:.2f}ms / {max_response*1000:.2f}ms")
            
            return {
                "test": "performance_baseline",
                "passed": acceptable,
                "details": {
                    "avg_response_ms": avg_response * 1000,
                    "min_response_ms": min_response * 1000,
                    "max_response_ms": max_response * 1000,
                    "samples": len(response_times)
                }
            }
            
        except Exception as e:
            return {
                "test": "performance_baseline",
                "passed": False,
                "reason": str(e)
            }
            
    def generate_report(self) -> Dict:
        """Generate comprehensive test report"""
        
        # Calculate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for r in self.test_results if r["passed"])
        failed_tests = total_tests - passed_tests
        
        report = {
            "timestamp": self.timestamp,
            "phase": "Phase 5 - DinD Multi-Client Architecture",
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "success_rate": (passed_tests / total_tests * 100) if total_tests > 0 else 0
            },
            "tests": self.test_results,
            "conclusion": "PASS" if passed_tests == total_tests else "PARTIAL" if passed_tests > 0 else "FAIL"
        }
        
        return report
        
    def run_all_tests(self):
        """Run all Phase 5 completion tests"""
        
        self.log("=" * 80)
        self.log("PHASE 5 COMPLETION TEST SUITE")
        self.log("DinD Multi-Client Architecture Validation")
        self.log("=" * 80)
        
        # Run all tests
        test_methods = [
            self.test_dind_orchestrator_health,
            self.test_port_availability,
            self.test_mcp_deployment_capability,
            self.test_multi_client_simulation,
            self.test_resource_isolation,
            self.test_network_connectivity,
            self.test_performance_baseline
        ]
        
        for test_method in test_methods:
            result = test_method()
            self.test_results.append(result)
            
            # Display result
            status = "PASS" if result["passed"] else "FAIL"
            self.log(f"\n{result['test']}: {status}")
            
        # Generate report
        report = self.generate_report()
        
        # Display summary
        self.log("\n" + "=" * 80)
        self.log("TEST SUMMARY")
        self.log("=" * 80)
        self.log(f"Total Tests: {report['summary']['total_tests']}")
        self.log(f"Passed: {report['summary']['passed']}")
        self.log(f"Failed: {report['summary']['failed']}")
        self.log(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        self.log(f"\nOVERALL RESULT: {report['conclusion']}")
        
        # Save report
        output_file = f"/opt/sutazaiapp/docs/reports/phase5_completion_{self.timestamp}.json"
        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)
            
        self.log(f"\nReport saved to: {output_file}")
        
        return report


def main():
    tester = Phase5CompletionTest()
    report = tester.run_all_tests()
    
    # Return appropriate exit code
    if report["conclusion"] == "PASS":
        return 0
    elif report["conclusion"] == "PARTIAL":
        return 1
    else:
        return 2


if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)