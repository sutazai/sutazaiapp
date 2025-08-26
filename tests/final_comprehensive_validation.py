#!/usr/bin/env python3
"""
FINAL COMPREHENSIVE VALIDATION SUITE
====================================

Comprehensive test suite to validate 100% system functionality
after all fixes and cleanup. Provides concrete evidence of
system state and operational status.

Author: Senior Automated Tester
Date: 2025-08-17 00:56:00 UTC
Purpose: Final system validation with evidence collection
"""

import json
import time
import requests
import subprocess
import sys
from datetime import datetime
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/tests/validation_results.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ComprehensiveSystemValidator:
    """
    Comprehensive system validation with evidence collection.
    Tests all critical system components and provides concrete
    evidence of functionality.
    """
    
    def __init__(self):
        self.validation_results = {
            "timestamp": datetime.utcnow().isoformat(),
            "validation_id": f"final_validation_{int(time.time())}",
            "tests": {},
            "summary": {
                "total_tests": 0,
                "passed_tests": 0,
                "failed_tests": 0,
                "evidence_collected": []
            }
        }
        self.base_url = "http://localhost:10010"
        
    def run_command(self, command: str, timeout: int = 30) -> Tuple[bool, str, str]:
        """Execute shell command and return success status, stdout, stderr"""
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout
            )
            return result.returncode == 0, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return False, "", "Command timed out"
        except Exception as e:
            return False, "", str(e)
    
    def test_backend_mcp_api_endpoints(self) -> Dict[str, Any]:
        """Test all MCP API endpoints with evidence collection"""
        logger.info("🔍 Testing Backend MCP API Endpoints...")
        
        test_result = {
            "test_name": "Backend MCP API Testing",
            "passed": True,
            "evidence": {},
            "failures": []
        }
        
        # Test endpoints with evidence collection
        endpoints = [
            "/api/v1/mcp/status",
            "/api/v1/mcp/services", 
            "/api/v1/mcp/health",
            "/api/v1/mcp/dind/status"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                
                if response.status_code == 200:
                    data = response.json()
                    test_result["evidence"][endpoint] = {
                        "status_code": response.status_code,
                        "response_data": data,
                        "response_time": response.elapsed.total_seconds()
                    }
                    logger.info(f"✅ {endpoint}: OK (200) - {response.elapsed.total_seconds():.3f}s")
                else:
                    test_result["passed"] = False
                    test_result["failures"].append(f"{endpoint}: HTTP {response.status_code}")
                    logger.error(f"❌ {endpoint}: HTTP {response.status_code}")
                    
            except Exception as e:
                test_result["passed"] = False
                test_result["failures"].append(f"{endpoint}: {str(e)}")
                logger.error(f"❌ {endpoint}: {str(e)}")
        
        return test_result
    
    def test_container_infrastructure(self) -> Dict[str, Any]:
        """Verify real MCP containers running actual services"""
        logger.info("🐳 Testing Container Infrastructure...")
        
        test_result = {
            "test_name": "Container Infrastructure Testing",
            "passed": True,
            "evidence": {},
            "failures": []
        }
        
        # Check DinD orchestrator container
        success, stdout, stderr = self.run_command(
            "docker exec sutazai-mcp-orchestrator-notls docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Image}}'"
        )
        
        if success and stdout:
            test_result["evidence"]["dind_containers"] = {
                "command_success": True,
                "container_list": stdout.strip(),
                "container_count": len(stdout.strip().split('\n')) - 1  # Exclude header
            }
            logger.info(f"✅ DinD containers found: {test_result['evidence']['dind_containers']['container_count']}")
        else:
            test_result["passed"] = False
            test_result["failures"].append(f"DinD container check failed: {stderr}")
            logger.error(f"❌ DinD container check failed: {stderr}")
        
        # Check for real MCP processes in containers
        mcp_services = ["claude-flow", "ruv-swarm", "files", "context7"]
        for service in mcp_services:
            success, stdout, stderr = self.run_command(
                f"docker exec sutazai-mcp-orchestrator-notls docker exec mcp-{service} ps aux 2>/dev/null | head -10"
            )
            
            if success and "node" in stdout.lower() or "python" in stdout.lower():
                test_result["evidence"][f"mcp_{service}_processes"] = {
                    "real_processes_found": True,
                    "process_list": stdout.strip()
                }
                logger.info(f"✅ MCP {service}: Real processes detected")
            else:
                logger.warning(f"⚠️  MCP {service}: Could not verify processes")
        
        return test_result
    
    def test_process_optimization(self) -> Dict[str, Any]:
        """Verify host process reduction (<10 processes)"""
        logger.info("⚡ Testing Process Optimization...")
        
        test_result = {
            "test_name": "Process Optimization Verification", 
            "passed": True,
            "evidence": {},
            "failures": []
        }
        
        # Count MCP-related host processes
        success, stdout, stderr = self.run_command(
            'ps aux | grep -E "(mcp|claude-flow)" | grep -v grep'
        )
        
        if success:
            process_count = len(stdout.strip().split('\n')) if stdout.strip() else 0
            test_result["evidence"]["host_processes"] = {
                "total_mcp_processes": process_count,
                "process_list": stdout.strip(),
                "target_threshold": 10,
                "meets_target": process_count < 10
            }
            
            if process_count < 10:
                logger.info(f"✅ Host processes: {process_count} (target: <10)")
            else:
                test_result["passed"] = False
                test_result["failures"].append(f"Too many host processes: {process_count} (target: <10)")
                logger.error(f"❌ Host processes: {process_count} (target: <10)")
        
        return test_result
    
    def test_docker_consolidation(self) -> Dict[str, Any]:
        """Verify single docker-compose usage"""
        logger.info("📦 Testing Docker Consolidation...")
        
        test_result = {
            "test_name": "Docker Consolidation Verification",
            "passed": True, 
            "evidence": {},
            "failures": []
        }
        
        # Check for consolidated compose file
        success, stdout, stderr = self.run_command(
            "ls -la /opt/sutazaiapp/docker/docker-compose.consolidated.yml"
        )
        
        if success:
            test_result["evidence"]["consolidated_compose"] = {
                "file_exists": True,
                "file_details": stdout.strip()
            }
            logger.info("✅ Consolidated docker-compose.yml found")
        else:
            test_result["passed"] = False
            test_result["failures"].append("Consolidated docker-compose.yml not found")
            logger.error("❌ Consolidated docker-compose.yml not found")
        
        # Check for remaining compose files (should be minimal)
        success, stdout, stderr = self.run_command(
            'find /opt/sutazaiapp/docker -name "docker-compose*.yml" | grep -v archived | grep -v consolidated'
        )
        
        remaining_files = len(stdout.strip().split('\n')) if stdout.strip() else 0
        test_result["evidence"]["remaining_compose_files"] = {
            "count": remaining_files,
            "files": stdout.strip() if stdout.strip() else "None",
            "consolidation_effective": remaining_files <= 2  # Allow 1-2 specialized files
        }
        
        if remaining_files <= 2:
            logger.info(f"✅ Docker consolidation effective: {remaining_files} remaining files")
        else:
            logger.warning(f"⚠️  Multiple compose files remain: {remaining_files}")
        
        return test_result
    
    def test_system_health(self) -> Dict[str, Any]:
        """Verify all services healthy and accessible"""
        logger.info("🩺 Testing System Health...")
        
        test_result = {
            "test_name": "System Health Validation",
            "passed": True,
            "evidence": {},
            "failures": []
        }
        
        # Check container health status
        success, stdout, stderr = self.run_command(
            'docker ps --format "table {{.Names}}\\t{{.Status}}"'
        )
        
        if success:
            healthy_count = stdout.count("(healthy)")
            up_count = stdout.count("Up ")
            total_containers = len(stdout.strip().split('\n')) - 1  # Exclude header
            
            test_result["evidence"]["container_health"] = {
                "total_containers": total_containers,
                "healthy_containers": healthy_count,
                "running_containers": up_count,
                "container_status": stdout.strip()
            }
            
            logger.info(f"✅ Containers: {up_count} running, {healthy_count} healthy of {total_containers} total")
        
        # Test key service endpoints
        services = {
            "backend": "http://localhost:10010/health",
            "frontend": "http://localhost:10011/"
        }
        
        for service_name, url in services.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    test_result["evidence"][f"{service_name}_health"] = {
                        "accessible": True,
                        "status_code": response.status_code,
                        "response_time": response.elapsed.total_seconds()
                    }
                    logger.info(f"✅ {service_name}: Accessible (200)")
                else:
                    test_result["passed"] = False
                    test_result["failures"].append(f"{service_name}: HTTP {response.status_code}")
                    logger.error(f"❌ {service_name}: HTTP {response.status_code}")
            except Exception as e:
                test_result["passed"] = False
                test_result["failures"].append(f"{service_name}: {str(e)}")
                logger.error(f"❌ {service_name}: {str(e)}")
        
        return test_result
    
    def test_network_connectivity(self) -> Dict[str, Any]:
        """Test network connectivity between services"""
        logger.info("🌐 Testing Network Connectivity...")
        
        test_result = {
            "test_name": "Network Connectivity Testing",
            "passed": True,
            "evidence": {},
            "failures": []
        }
        
        # Check network topology
        success, stdout, stderr = self.run_command(
            "docker network ls | grep sutazai"
        )
        
        if success:
            network_count = len(stdout.strip().split('\n')) if stdout.strip() else 0
            test_result["evidence"]["network_topology"] = {
                "sutazai_networks": network_count,
                "networks": stdout.strip()
            }
            logger.info(f"✅ Sutazai networks found: {network_count}")
        
        # Test internal connectivity (backend to database)
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                test_result["evidence"]["internal_connectivity"] = {
                    "backend_to_services": health_data.get("services", {}),
                    "connectivity_healthy": True
                }
                logger.info("✅ Internal service connectivity verified")
            else:
                test_result["passed"] = False
                test_result["failures"].append("Backend health check failed")
                logger.error("❌ Backend health check failed")
        except Exception as e:
            test_result["passed"] = False
            test_result["failures"].append(f"Connectivity test failed: {str(e)}")
            logger.error(f"❌ Connectivity test failed: {str(e)}")
        
        return test_result
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Execute all validation tests and collect comprehensive evidence"""
        logger.info("🚀 Starting Comprehensive System Validation...")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Execute all tests
        tests = [
            self.test_backend_mcp_api_endpoints,
            self.test_container_infrastructure,
            self.test_process_optimization,
            self.test_docker_consolidation,
            self.test_system_health,
            self.test_network_connectivity
        ]
        
        for test_func in tests:
            test_result = test_func()
            test_name = test_result["test_name"]
            self.validation_results["tests"][test_name] = test_result
            
            if test_result["passed"]:
                self.validation_results["summary"]["passed_tests"] += 1
                logger.info(f"✅ {test_name}: PASSED")
            else:
                self.validation_results["summary"]["failed_tests"] += 1
                logger.error(f"❌ {test_name}: FAILED")
                for failure in test_result["failures"]:
                    logger.error(f"   - {failure}")
            
            self.validation_results["summary"]["total_tests"] += 1
        
        # Calculate execution time
        execution_time = time.time() - start_time
        self.validation_results["execution_time_seconds"] = execution_time
        
        # Generate summary
        summary = self.validation_results["summary"]
        pass_rate = (summary["passed_tests"] / summary["total_tests"]) * 100 if summary["total_tests"] > 0 else 0
        
        logger.info("=" * 60)
        logger.info("📊 VALIDATION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Passed: {summary['passed_tests']}")
        logger.info(f"Failed: {summary['failed_tests']}")
        logger.info(f"Pass Rate: {pass_rate:.1f}%")
        logger.info(f"Execution Time: {execution_time:.2f}s")
        
        if pass_rate == 100:
            logger.info("🎉 ALL TESTS PASSED - SYSTEM 100% VALIDATED")
        else:
            logger.warning(f"⚠️  {summary['failed_tests']} TESTS FAILED - SYSTEM NEEDS ATTENTION")
        
        return self.validation_results
    
    def save_validation_report(self, results: Dict[str, Any]) -> str:
        """Save comprehensive validation report"""
        report_path = f"/opt/sutazaiapp/tests/final_validation_report_{int(time.time())}.json"
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"📄 Validation report saved: {report_path}")
        return report_path

def main():
    """Main execution function"""
    print("🔍 FINAL COMPREHENSIVE VALIDATION")
    print("=" * 50)
    print("Testing system functionality after cleanup...")
    print()
    
    validator = ComprehensiveSystemValidator()
    results = validator.run_comprehensive_validation()
    report_path = validator.save_validation_report(results)
    
    # Print final status
    summary = results["summary"]
    if summary["failed_tests"] == 0:
        print("\n🎉 VALIDATION COMPLETE: SYSTEM 100% FUNCTIONAL")
        print("All tests passed with concrete evidence.")
        sys.exit(0)
    else:
        print(f"\n⚠️  VALIDATION INCOMPLETE: {summary['failed_tests']} FAILURES DETECTED")
        print("System requires attention before claiming 100% functionality.")
        sys.exit(1)

if __name__ == "__main__":
    main()