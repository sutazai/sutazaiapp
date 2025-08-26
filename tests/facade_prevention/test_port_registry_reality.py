#!/usr/bin/env python3
"""
Port Registry Reality Tests - Facade Prevention Framework
=========================================================

This module implements comprehensive tests to prevent facade implementations in port registry.
Tests verify that documented ports actually match what's being used in reality.

CRITICAL PURPOSE: Prevent discrepancies between documented port allocations and actual usage,
which can cause conflicts, deployment failures, and system instability.
"""

import asyncio
import pytest
import json
import socket
import subprocess
import re
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from pathlib import Path
import logging
import yaml

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PortRegistryRealityTester:
    """
    Tests that verify documented port allocations match actual usage.
    
    FACADE PREVENTION: These tests catch discrepancies between documented ports 
    and actual port bindings, preventing deployment conflicts and system failures.
    """
    
    def __init__(self):
        self.sutazai_base = Path("/opt/sutazaiapp")
        self.documented_ports = {}
        self.actual_ports = {}
        
        # Expected port ranges from documentation
        self.port_ranges = {
            "infrastructure": (10000, 10099),
            "vector_ai": (10100, 10199),
            "monitoring": (10200, 10299),
            "agents": (11000, 11999)
        }
    
    def load_documented_port_registry(self) -> Dict:
        """Load documented port allocations from various sources."""
        logger.info("📋 Loading documented port registry...")
        
        documented_ports = {}
        
        # Load from CLAUDE.md
        claude_ports = self._extract_ports_from_claude_md()
        documented_ports.update(claude_ports)
        
        # Load from docker-compose.yml
        docker_ports = self._extract_ports_from_docker_compose()
        documented_ports.update(docker_ports)
        
        # Load from PortRegistry.md if it exists
        port_registry_file = self.sutazai_base / "IMPORTANT" / "PortRegistry.md"
        if port_registry_file.exists():
            registry_ports = self._extract_ports_from_port_registry(port_registry_file)
            documented_ports.update(registry_ports)
        
        # Load from README.md
        readme_ports = self._extract_ports_from_readme()
        documented_ports.update(readme_ports)
        
        self.documented_ports = documented_ports
        logger.info(f"Loaded {len(documented_ports)} documented port allocations")
        return documented_ports
    
    def _extract_ports_from_claude_md(self) -> Dict:
        """Extract port allocations from CLAUDE.md."""
        claude_file = self.sutazai_base / "CLAUDE.md"
        if not claude_file.exists():
            return {}
        
        ports = {}
        try:
            content = claude_file.read_text()
            
            # Extract port patterns like "10000: PostgreSQL database"
            port_patterns = [
                r'(\d{4,5}):\s*([^\n]+)',  # "10000: PostgreSQL database"
                r'(\d{4,5})\s*-\s*(\d{4,5}):\s*([^\n]+)',  # "10002-10003: Neo4j"
                r'localhost:(\d{4,5})',  # "localhost:10010"
                r'http://[^:]+:(\d{4,5})',  # "http://localhost:10010"
            ]
            
            for pattern in port_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    if len(match) == 2:  # Single port
                        port, description = match
                        ports[int(port)] = {
                            "service": description.strip(),
                            "source": "CLAUDE.md",
                            "type": "single"
                        }
                    elif len(match) == 3:  # Port range
                        start_port, end_port, description = match
                        for p in range(int(start_port), int(end_port) + 1):
                            ports[p] = {
                                "service": f"{description.strip()} (port {p})",
                                "source": "CLAUDE.md",
                                "type": "range"
                            }
        
        except Exception as e:
            logger.error(f"Failed to extract ports from CLAUDE.md: {e}")
        
        return ports
    
    def _extract_ports_from_docker_compose(self) -> Dict:
        """Extract port allocations from docker-compose.yml."""
        compose_file = self.sutazai_base / "docker-compose.yml"
        if not compose_file.exists():
            return {}
        
        ports = {}
        try:
            with open(compose_file, 'r') as f:
                compose_data = yaml.safe_load(f)
            
            services = compose_data.get("services", {})
            for service_name, service_config in services.items():
                service_ports = service_config.get("ports", [])
                
                for port_mapping in service_ports:
                    if isinstance(port_mapping, str):
                        # Parse "10000:5432" format
                        if ":" in port_mapping:
                            external_port, internal_port = port_mapping.split(":")
                            external_port = int(external_port)
                            ports[external_port] = {
                                "service": f"{service_name} ({internal_port})",
                                "source": "docker-compose.yml",
                                "type": "mapping",
                                "internal_port": internal_port
                            }
                    elif isinstance(port_mapping, dict):
                        # Parse long form port mapping
                        target = port_mapping.get("target")
                        published = port_mapping.get("published")
                        if published:
                            ports[int(published)] = {
                                "service": f"{service_name} ({target})",
                                "source": "docker-compose.yml", 
                                "type": "mapping",
                                "internal_port": target
                            }
        
        except Exception as e:
            logger.error(f"Failed to extract ports from docker-compose.yml: {e}")
        
        return ports
    
    def _extract_ports_from_port_registry(self, registry_file: Path) -> Dict:
        """Extract ports from dedicated PortRegistry.md file."""
        ports = {}
        try:
            content = registry_file.read_text()
            
            # Look for port registry format
            port_patterns = [
                r'(\d{4,5})\s*\|\s*([^|]+)\|',  # Markdown table format
                r'(\d{4,5}):\s*([^\n]+)',
            ]
            
            for pattern in port_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for port, description in matches:
                    ports[int(port)] = {
                        "service": description.strip(),
                        "source": "PortRegistry.md",
                        "type": "registry"
                    }
        
        except Exception as e:
            logger.error(f"Failed to extract ports from PortRegistry.md: {e}")
        
        return ports
    
    def _extract_ports_from_readme(self) -> Dict:
        """Extract port allocations from README.md."""
        readme_file = self.sutazai_base / "README.md"
        if not readme_file.exists():
            return {}
        
        ports = {}
        try:
            content = readme_file.read_text()
            
            # Extract port patterns
            port_patterns = [
                r'(\d{4,5}):\s*([^\n]+)',
                r'localhost:(\d{4,5})',
                r'http://[^:]+:(\d{4,5})',
            ]
            
            for pattern in port_patterns:
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    if len(match) == 2:
                        port, description = match
                        ports[int(port)] = {
                            "service": description.strip(),
                            "source": "README.md",
                            "type": "documentation"
                        }
        
        except Exception as e:
            logger.error(f"Failed to extract ports from README.md: {e}")
        
        return ports
    
    async def get_actual_port_usage(self) -> Dict:
        """Get actual port usage from the system."""
        logger.info("🔍 Scanning actual port usage...")
        
        actual_ports = {}
        
        # Get listening ports using netstat/ss
        listening_ports = await self._get_listening_ports()
        actual_ports.update(listening_ports)
        
        # Get Docker container ports
        docker_ports = await self._get_docker_container_ports()
        actual_ports.update(docker_ports)
        
        # Test port accessibility
        accessible_ports = await self._test_port_accessibility()
        
        # Merge accessibility information
        for port, info in accessible_ports.items():
            if port in actual_ports:
                actual_ports[port]["accessible"] = info["accessible"]
                actual_ports[port]["response_info"] = info.get("response_info", {})
            else:
                actual_ports[port] = info
        
        self.actual_ports = actual_ports
        logger.info(f"Found {len(actual_ports)} ports in actual usage")
        return actual_ports
    
    async def _get_listening_ports(self) -> Dict:
        """Get ports that are currently listening on the system."""
        ports = {}
        
        try:
            # Use ss command to get listening ports
            result = subprocess.run(
                ["ss", "-tlnp"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')
                for line in lines:
                    # Parse ss output to extract port and process info
                    if "LISTEN" in line:
                        parts = line.split()
                        if len(parts) >= 4:
                            local_address = parts[3]
                            if ":" in local_address:
                                try:
                                    port = int(local_address.split(":")[-1])
                                    process_info = parts[-1] if len(parts) > 4 else "unknown"
                                    
                                    ports[port] = {
                                        "status": "listening",
                                        "source": "ss_command",
                                        "process": process_info,
                                        "address": local_address
                                    }
                                except ValueError:
                                    continue
        
        except Exception as e:
            logger.error(f"Failed to get listening ports: {e}")
        
        return ports
    
    async def _get_docker_container_ports(self) -> Dict:
        """Get ports from Docker containers."""
        ports = {}
        
        try:
            # Get Docker container port mappings
            result = subprocess.run(
                ["docker", "ps", "--format", "table {{.Names}}\\t{{.Ports}}"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                lines = result.stdout.split('\n')[1:]  # Skip header
                for line in lines:
                    if line.strip():
                        parts = line.split('\t')
                        if len(parts) >= 2:
                            container_name = parts[0]
                            port_mappings = parts[1]
                            
                            # Parse port mappings like "0.0.0.0:10010->8000/tcp"
                            port_pattern = r'0\.0\.0\.0:(\d+)->(\d+)/(\w+)'
                            matches = re.findall(port_pattern, port_mappings)
                            
                            for external_port, internal_port, protocol in matches:
                                external_port = int(external_port)
                                ports[external_port] = {
                                    "status": "docker_mapped",
                                    "source": "docker_ps",
                                    "container": container_name,
                                    "internal_port": internal_port,
                                    "protocol": protocol
                                }
        
        except Exception as e:
            logger.error(f"Failed to get Docker container ports: {e}")
        
        return ports
    
    async def _test_port_accessibility(self) -> Dict:
        """Test if ports are actually accessible."""
        ports = {}
        
        # Test common SutazAI ports
        test_ports = list(range(10000, 10320)) + list(range(11000, 11100))
        
        for port in test_ports:
            accessible = await self._is_port_accessible("localhost", port)
            if accessible:
                # Try to get more info about the service
                response_info = await self._get_service_info(port)
                ports[port] = {
                    "status": "accessible",
                    "source": "accessibility_test",
                    "accessible": True,
                    "response_info": response_info
                }
        
        return ports
    
    async def _is_port_accessible(self, host: str, port: int, timeout: float = 2.0) -> bool:
        """Test if a port is accessible."""
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = sock.connect_ex((host, port))
            sock.close()
            return result == 0
        except Exception:
            return False
    
    async def _get_service_info(self, port: int) -> Dict:
        """Try to get information about the service running on a port."""
        try:
            import httpx
            
            # Try common HTTP endpoints
            urls = [
                f"http://localhost:{port}/",
                f"http://localhost:{port}/health",
                f"http://localhost:{port}/api/health",
                f"http://localhost:{port}/status"
            ]
            
            async with httpx.AsyncClient(timeout=3.0) as client:
                for url in urls:
                    try:
                        response = await client.get(url)
                        return {
                            "http_accessible": True,
                            "status_code": response.status_code,
                            "headers": dict(response.headers),
                            "url": url
                        }
                    except:
                        continue
            
            return {"http_accessible": False}
        
        except Exception as e:
            return {"error": str(e)}
    
    async def test_port_registry_accuracy(self) -> Dict:
        """
        FACADE TEST: Verify documented ports match actual usage.
        
        PREVENTS: Documentation claiming ports are used for specific services 
        when they're actually used differently or not at all.
        """
        logger.info("🎯 Testing port registry accuracy...")
        
        documented = self.documented_ports
        actual = self.actual_ports
        
        # Find discrepancies
        documented_ports = set(documented.keys())
        actual_ports = set(actual.keys())
        
        # Ports documented but not actually used
        documented_but_unused = documented_ports - actual_ports
        
        # Ports in use but not documented
        undocumented_but_used = actual_ports - documented_ports
        
        # Ports with mismatched service descriptions
        mismatched_services = []
        for port in documented_ports.intersection(actual_ports):
            doc_service = documented[port]["service"].lower()
            actual_info = actual[port]
            
            # Check if actual service matches documented service
            if "container" in actual_info:
                container_name = actual_info["container"].lower()
                # Simple check if container name relates to documented service
                service_words = doc_service.split()
                container_words = container_name.split("-")
                
                # Check for overlap in service keywords
                overlap = any(word in container_name for word in service_words if len(word) > 3)
                if not overlap:
                    mismatched_services.append({
                        "port": port,
                        "documented": doc_service,
                        "actual_container": container_name
                    })
        
        # Check for port conflicts in expected ranges
        port_conflicts = self._check_port_range_conflicts(actual_ports)
        
        accuracy_score = 1.0
        if len(documented_ports) > 0:
            accuracy_score = (
                len(documented_ports.intersection(actual_ports)) / 
                len(documented_ports.union(actual_ports))
            )
        
        return {
            "documented_ports": len(documented_ports),
            "actual_ports": len(actual_ports),
            "documented_but_unused": len(documented_but_unused),
            "documented_but_unused_list": list(documented_but_unused),
            "undocumented_but_used": len(undocumented_but_used),
            "undocumented_but_used_list": list(undocumented_but_used),
            "mismatched_services": len(mismatched_services),
            "mismatched_services_list": mismatched_services,
            "port_conflicts": port_conflicts,
            "accuracy_score": accuracy_score,
            "test_passed": (
                len(documented_but_unused) < 5 and 
                len(undocumented_but_used) < 10 and 
                len(mismatched_services) < 3 and
                accuracy_score > 0.7
            )
        }
    
    def _check_port_range_conflicts(self, actual_ports: set) -> Dict:
        """Check if ports are in their expected ranges."""
        conflicts = {}
        
        for port in actual_ports:
            expected_category = None
            for category, (start, end) in self.port_ranges.items():
                if start <= port <= end:
                    expected_category = category
                    break
            
            if not expected_category:
                if port >= 10000:  # Only check SutazAI-related ports
                    conflicts[port] = {
                        "issue": "outside_expected_ranges",
                        "port": port,
                        "expected_ranges": self.port_ranges
                    }
        
        return conflicts
    
    async def test_port_availability_reality(self) -> Dict:
        """
        FACADE TEST: Verify documented ports are actually available/in-use.
        
        PREVENTS: Documentation claiming ports are available when they're not,
        or claiming services are running when they're not accessible.
        """
        logger.info("🔌 Testing port availability reality...")
        
        documented = self.documented_ports
        availability_results = {}
        
        available_count = 0
        unavailable_count = 0
        
        for port, info in documented.items():
            is_accessible = await self._is_port_accessible("localhost", port)
            service_info = await self._get_service_info(port) if is_accessible else {}
            
            availability_results[port] = {
                "documented_service": info["service"],
                "is_accessible": is_accessible,
                "service_info": service_info,
                "port_claimed_available": True,  # Documentation implies availability
                "actually_available": is_accessible
            }
            
            if is_accessible:
                available_count += 1
            else:
                unavailable_count += 1
        
        availability_ratio = available_count / len(documented) if documented else 0
        
        return {
            "total_documented_ports": len(documented),
            "available_ports": available_count,
            "unavailable_ports": unavailable_count,
            "availability_ratio": availability_ratio,
            "availability_results": availability_results,
            "test_passed": availability_ratio > 0.6  # At least 60% of documented ports should be accessible
        }
    
    async def run_comprehensive_port_tests(self) -> Dict:
        """Run all port registry reality tests and return comprehensive results."""
        logger.info("🚀 Starting comprehensive port registry reality tests...")
        
        start_time = datetime.now()
        
        # Load documented ports
        await asyncio.create_task(asyncio.to_thread(self.load_documented_port_registry))
        
        # Get actual port usage
        await self.get_actual_port_usage()
        
        results = {
            "test_suite": "port_registry_facade_prevention",
            "timestamp": start_time.isoformat(),
            "documented_ports": self.documented_ports,
            "actual_ports": self.actual_ports,
            "tests": {}
        }
        
        # Run all tests
        test_methods = [
            ("port_registry_accuracy", self.test_port_registry_accuracy),
            ("port_availability_reality", self.test_port_availability_reality)
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
        
        logger.info(f"Port registry reality tests completed: {passed_tests}/{total_tests} passed")
        return results


# Pytest integration
@pytest.mark.asyncio
async def test_port_registry_is_not_facade():
    """
    Main facade prevention test for port registry.
    
    This test MUST pass for deployment to prevent facade implementations.
    """
    tester = PortRegistryRealityTester()
    results = await tester.run_comprehensive_port_tests()
    
    # CRITICAL: Fail if port registry is inaccurate
    assert results["overall_status"] == "passed", f"Port registry reality tests failed: {results}"
    
    # Check for specific accuracy issues
    accuracy_test = results["tests"].get("port_registry_accuracy", {})
    accuracy_score = accuracy_test.get("accuracy_score", 0)
    assert accuracy_score > 0.7, f"Port registry accuracy too low: {accuracy_score}"
    
    # Log results for monitoring
    logger.info(f"✅ Port registry reality verification passed: {results['summary']}")


@pytest.mark.asyncio
async def test_documented_ports_are_accessible():
    """Test that documented ports are actually accessible."""
    tester = PortRegistryRealityTester()
    tester.load_documented_port_registry()
    await tester.get_actual_port_usage()
    
    result = await tester.test_port_availability_reality()
    availability_ratio = result.get("availability_ratio", 0)
    assert availability_ratio > 0.5, f"Too many documented ports are inaccessible: {availability_ratio}"


@pytest.mark.asyncio
async def test_no_major_port_conflicts():
    """Test that there are no major conflicts between documented and actual ports."""
    tester = PortRegistryRealityTester()
    tester.load_documented_port_registry()
    await tester.get_actual_port_usage()
    
    result = await tester.test_port_registry_accuracy()
    undocumented_ports = result.get("undocumented_but_used", 0)
    mismatched_services = result.get("mismatched_services", 0)
    
    assert undocumented_ports < 20, f"Too many undocumented ports in use: {undocumented_ports}"
    assert mismatched_services < 5, f"Too many service mismatches: {mismatched_services}"


if __name__ == "__main__":
    async def main():
        tester = PortRegistryRealityTester()
        results = await tester.run_comprehensive_port_tests()
        print(json.dumps(results, indent=2))
        
        if results["overall_status"] != "passed":
            print(f"\n❌ PORT REGISTRY FACADE ISSUES DETECTED")
            exit(1)
        else:
            print(f"\n✅ All port registry reality tests passed!")
            exit(0)
    
    asyncio.run(main())