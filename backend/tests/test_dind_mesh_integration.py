#!/usr/bin/env python3
"""
Test script for DinD-Mesh Integration
Validates end-to-end communication: mesh → DinD → MCP
Tests multi-client concurrent access without resource conflicts
"""
import asyncio
import httpx
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
BACKEND_URL = "http://localhost:10010"  # Backend runs on port 10010
MCP_API_BASE = f"{BACKEND_URL}/api/v1/mcp"
DIND_API_BASE = f"{MCP_API_BASE}/dind"

class DinDMeshIntegrationTest:
    """Test suite for DinD-Mesh integration"""
    
    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)
        self.test_results = []
        
    async def test_dind_status(self) -> bool:
        """Test 1: Check DinD orchestrator status"""
        test_name = "DinD Status Check"
        try:
            logger.info(f"Running: {test_name}")
            response = await self.client.get(f"{DIND_API_BASE}/status")
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ {test_name} PASSED")
                logger.info(f"  - DinD initialized: {data.get('initialized', False)}")
                logger.info(f"  - Total services: {data.get('total', 0)}")
                logger.info(f"  - Healthy services: {data.get('healthy', 0)}")
                self.test_results.append({"test": test_name, "status": "PASSED", "data": data})
                return True
            else:
                logger.error(f"❌ {test_name} FAILED: Status code {response.status_code}")
                self.test_results.append({"test": test_name, "status": "FAILED", "error": f"Status {response.status_code}"})
                return False
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results.append({"test": test_name, "status": "FAILED", "error": str(e)})
            return False
    
    async def test_service_discovery(self) -> bool:
        """Test 2: Verify MCP services are discoverable through mesh"""
        test_name = "Service Discovery"
        try:
            logger.info(f"Running: {test_name}")
            
            # Get list of MCP services
            response = await self.client.get(f"{MCP_API_BASE}/services")
            
            if response.status_code == 200:
                services = response.json()
                logger.info(f"✅ {test_name} PASSED")
                logger.info(f"  - Found {len(services)} MCP services: {services}")
                self.test_results.append({"test": test_name, "status": "PASSED", "services": services})
                return True
            else:
                logger.error(f"❌ {test_name} FAILED: Status code {response.status_code}")
                self.test_results.append({"test": test_name, "status": "FAILED", "error": f"Status {response.status_code}"})
                return False
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results.append({"test": test_name, "status": "FAILED", "error": str(e)})
            return False
    
    async def test_multi_client_access(self) -> bool:
        """Test 3: Validate multi-client concurrent access"""
        test_name = "Multi-Client Access"
        try:
            logger.info(f"Running: {test_name}")
            
            # Simulate two clients accessing the same service
            test_service = "files"  # Use files MCP as test target
            
            # Client 1: Claude Code
            client1_request = {
                "method": "list",
                "params": {"path": "/tmp"}
            }
            
            # Client 2: Codex
            client2_request = {
                "method": "list",
                "params": {"path": "/var"}
            }
            
            # Send concurrent requests
            tasks = [
                self.client.post(
                    f"{DIND_API_BASE}/{test_service}/request?client_id=claude-code",
                    json=client1_request
                ),
                self.client.post(
                    f"{DIND_API_BASE}/{test_service}/request?client_id=codex",
                    json=client2_request
                )
            ]
            
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            
            success = True
            for i, response in enumerate(responses):
                client_name = ["claude-code", "codex"][i]
                if isinstance(response, Exception):
                    logger.error(f"  ❌ Client {client_name} failed: {response}")
                    success = False
                elif response.status_code == 200:
                    logger.info(f"  ✅ Client {client_name} request succeeded")
                else:
                    logger.error(f"  ❌ Client {client_name} got status {response.status_code}")
                    success = False
            
            if success:
                logger.info(f"✅ {test_name} PASSED")
                self.test_results.append({"test": test_name, "status": "PASSED"})
            else:
                logger.error(f"❌ {test_name} FAILED")
                self.test_results.append({"test": test_name, "status": "FAILED"})
            
            return success
            
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results.append({"test": test_name, "status": "FAILED", "error": str(e)})
            return False
    
    async def test_port_mapping(self) -> bool:
        """Test 4: Verify port mapping (11100-11199)"""
        test_name = "Port Mapping"
        try:
            logger.info(f"Running: {test_name}")
            
            # Get DinD status to check port allocations
            response = await self.client.get(f"{DIND_API_BASE}/status")
            
            if response.status_code == 200:
                data = response.json()
                port_allocations = data.get("port_allocations", {})
                
                # Check if ports are in expected range
                valid_ports = all(
                    11100 <= int(port) <= 11199 
                    for port in port_allocations.keys()
                )
                
                if valid_ports:
                    logger.info(f"✅ {test_name} PASSED")
                    logger.info(f"  - Port allocations: {port_allocations}")
                    self.test_results.append({"test": test_name, "status": "PASSED", "ports": port_allocations})
                    return True
                else:
                    logger.error(f"❌ {test_name} FAILED: Ports outside expected range")
                    self.test_results.append({"test": test_name, "status": "FAILED", "error": "Invalid port range"})
                    return False
            else:
                logger.error(f"❌ {test_name} FAILED: Could not get port allocations")
                self.test_results.append({"test": test_name, "status": "FAILED"})
                return False
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results.append({"test": test_name, "status": "FAILED", "error": str(e)})
            return False
    
    async def test_health_monitoring(self) -> bool:
        """Test 5: Verify health monitoring for DinD MCPs"""
        test_name = "Health Monitoring"
        try:
            logger.info(f"Running: {test_name}")
            
            # Get health status of all MCP services
            response = await self.client.get(f"{MCP_API_BASE}/health")
            
            if response.status_code == 200:
                health = response.json()
                summary = health.get("summary", {})
                
                logger.info(f"✅ {test_name} PASSED")
                logger.info(f"  - Total MCPs: {summary.get('total', 0)}")
                logger.info(f"  - Healthy: {summary.get('healthy', 0)}")
                logger.info(f"  - Health percentage: {summary.get('percentage_healthy', 0):.1f}%")
                
                self.test_results.append({"test": test_name, "status": "PASSED", "health": summary})
                return True
            else:
                logger.error(f"❌ {test_name} FAILED: Status code {response.status_code}")
                self.test_results.append({"test": test_name, "status": "FAILED"})
                return False
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results.append({"test": test_name, "status": "FAILED", "error": str(e)})
            return False
    
    async def test_service_isolation(self) -> bool:
        """Test 6: Verify service isolation in DinD"""
        test_name = "Service Isolation"
        try:
            logger.info(f"Running: {test_name}")
            
            # Check that services are properly isolated
            response = await self.client.get(f"{DIND_API_BASE}/status")
            
            if response.status_code == 200:
                data = response.json()
                services = data.get("services", {})
                
                # Each service should have its own container ID and port
                isolated = all(
                    service.get("container_id") and service.get("mesh_port")
                    for service in services.values()
                )
                
                if isolated:
                    logger.info(f"✅ {test_name} PASSED")
                    logger.info(f"  - All {len(services)} services properly isolated")
                    self.test_results.append({"test": test_name, "status": "PASSED"})
                    return True
                else:
                    logger.error(f"❌ {test_name} FAILED: Services not properly isolated")
                    self.test_results.append({"test": test_name, "status": "FAILED"})
                    return False
            else:
                logger.error(f"❌ {test_name} FAILED: Could not check isolation")
                self.test_results.append({"test": test_name, "status": "FAILED"})
                return False
                
        except Exception as e:
            logger.error(f"❌ {test_name} FAILED: {e}")
            self.test_results.append({"test": test_name, "status": "FAILED", "error": str(e)})
            return False
    
    async def run_all_tests(self):
        """Run all integration tests"""
        logger.info("=" * 60)
        logger.info("Starting DinD-Mesh Integration Tests")
        logger.info("=" * 60)
        
        tests = [
            self.test_dind_status,
            self.test_service_discovery,
            self.test_multi_client_access,
            self.test_port_mapping,
            self.test_health_monitoring,
            self.test_service_isolation
        ]
        
        passed = 0
        failed = 0
        
        for test in tests:
            result = await test()
            if result:
                passed += 1
            else:
                failed += 1
            logger.info("-" * 40)
        
        # Print summary
        logger.info("=" * 60)
        logger.info("Test Summary")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {passed + failed}")
        logger.info(f"Passed: {passed}")
        logger.info(f"Failed: {failed}")
        logger.info(f"Success Rate: {(passed / (passed + failed) * 100):.1f}%")
        
        # Save results
        with open("/opt/sutazaiapp/backend/tests/dind_integration_results.json", "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "summary": {
                    "total": passed + failed,
                    "passed": passed,
                    "failed": failed,
                    "success_rate": passed / (passed + failed) * 100
                },
                "results": self.test_results
            }, f, indent=2)
        
        logger.info("Results saved to dind_integration_results.json")
        
        # Close client
        await self.client.aclose()
        
        return failed == 0

async def main():
    """Main test runner"""
    tester = DinDMeshIntegrationTest()
    success = await tester.run_all_tests()
    
    if success:
        logger.info("\n✅ All tests PASSED! DinD-Mesh integration working correctly.")
    else:
        logger.error("\n❌ Some tests FAILED. Check the logs for details.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)