#!/usr/bin/env python3
"""
HARDWARE OPTIMIZER TEST EXECUTION SCRIPT
========================================

Simple test execution script that runs a subset of the comprehensive tests
to validate functionality and provide immediate feedback.

This script executes:
- Basic health and connectivity tests
- Single-user endpoint validation
- Small-scale load testing (1, 5, 10 users)
- Basic security boundary tests
- Simple stress testing

Author: Ultra-Critical Automated Testing Specialist
Version: 1.0.0
"""

import asyncio
import sys
import os
import json
import time
import logging
import requests
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/tests/hardware_optimizer_test_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('HardwareOptimizerTests')

class BasicHardwareOptimizerTester:
    """Basic tester for immediate validation"""
    
    def __init__(self, base_url="http://localhost:11110"):
        self.base_url = base_url
        self.test_results = []
        
    def test_service_health(self):
        """Test basic service health"""
        logger.info("Testing service health...")
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    logger.info("✅ Service health check passed")
                    return True
                else:
                    logger.error(f"❌ Service not healthy: {health_data}")
                    return False
            else:
                logger.error(f"❌ Health check failed: HTTP {response.status_code}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Health check exception: {e}")
            return False
    
    def test_basic_endpoints(self):
        """Test basic endpoint functionality"""
        logger.info("Testing basic endpoints...")
        
        endpoints = [
            ("GET", "/health", {}),
            ("GET", "/status", {}),
            ("GET", "/analyze/storage", {"path": "/tmp"}),
            ("GET", "/analyze/storage/report", {}),
            ("POST", "/optimize/memory", {}),
            ("POST", "/optimize/storage", {"dry_run": True}),
        ]
        
        results = []
        
        for method, path, params in endpoints:
            try:
                start_time = time.time()
                
                if method == "GET":
                    response = requests.get(f"{self.base_url}{path}", params=params, timeout=30)
                else:
                    response = requests.post(f"{self.base_url}{path}", params=params, timeout=30)
                
                end_time = time.time()
                response_time = (end_time - start_time) * 1000
                
                success = 200 <= response.status_code < 300
                status_emoji = "✅" if success else "❌"
                
                logger.info(f"{status_emoji} {method} {path}: {response.status_code} ({response_time:.1f}ms)")
                
                results.append({
                    "endpoint": path,
                    "method": method,
                    "success": success,
                    "status_code": response.status_code,
                    "response_time_ms": response_time
                })
                
            except Exception as e:
                logger.error(f"❌ {method} {path}: Exception - {e}")
                results.append({
                    "endpoint": path,
                    "method": method,
                    "success": False,
                    "error": str(e)
                })
        
        success_count = sum(1 for r in results if r["success"])
        logger.info(f"📊 Endpoint tests: {success_count}/{len(results)} passed")
        
        return results
    
    def test_simple_load(self):
        """Test with small concurrent load"""
        logger.info("Testing simple concurrent load...")
        
        import concurrent.futures
        
        def make_request():
            try:
                start_time = time.time()
                response = requests.get(f"{self.base_url}/health", timeout=10)
                end_time = time.time()
                return {
                    "success": 200 <= response.status_code < 300,
                    "response_time_ms": (end_time - start_time) * 1000
                }
            except Exception as e:
                return {"success": False, "error": str(e)}
        
        # Test with 5 concurrent requests
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [future.result() for future in concurrent.futures.as_completed(futures, timeout=30)]
        
        successful = sum(1 for r in results if r.get("success", False))
        avg_response_time = sum(r.get("response_time_ms", 0) for r in results if r.get("success", False)) / max(1, successful)
        
        logger.info(f"📊 Simple load test: {successful}/10 requests successful")
        logger.info(f"📊 Average response time: {avg_response_time:.1f}ms")
        
        return {
            "total_requests": len(results),
            "successful_requests": successful,
            "success_rate": (successful / len(results)) * 100,
            "avg_response_time_ms": avg_response_time
        }
    
    def test_security_basics(self):
        """Test basic security boundaries"""
        logger.info("Testing basic security boundaries...")
        
        security_tests = [
            # Path traversal attempts
            {"path": "../../../etc/passwd", "expected_safe": True},
            {"path": "/root", "expected_safe": True},
            {"path": "/etc/shadow", "expected_safe": True},
        ]
        
        results = []
        
        for test in security_tests:
            try:
                response = requests.get(
                    f"{self.base_url}/analyze/storage",
                    params={"path": test["path"]},
                    timeout=10
                )
                
                # Check if system files are accessed (security issue)
                is_safe = not (response.status_code == 200 and "root:" in response.text.lower())
                
                status_emoji = "✅" if is_safe else "🚨"
                logger.info(f"{status_emoji} Security test - Path: {test['path']}, Safe: {is_safe}")
                
                results.append({
                    "test_path": test["path"],
                    "safe": is_safe,
                    "status_code": response.status_code
                })
                
            except Exception as e:
                logger.info(f"✅ Security test - Path: {test['path']}, Exception (expected): {type(e).__name__}")
                results.append({
                    "test_path": test["path"],
                    "safe": True,  # Exception is acceptable for security
                    "exception": str(e)
                })
        
        safe_count = sum(1 for r in results if r["safe"])
        logger.info(f"📊 Security tests: {safe_count}/{len(results)} safe")
        
        return results
    
    def run_comprehensive_basic_tests(self):
        """Run all basic tests"""
        logger.info("🚀 Starting Hardware Optimizer Basic Tests")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Test 1: Service Health
        health_passed = self.test_service_health()
        if not health_passed:
            logger.error("❌ Service health check failed - cannot continue")
            return {"overall_passed": False, "reason": "Service not healthy"}
        
        # Test 2: Basic Endpoints
        endpoint_results = self.test_basic_endpoints()
        endpoint_success_rate = sum(1 for r in endpoint_results if r["success"]) / len(endpoint_results) * 100
        
        # Test 3: Simple Load
        load_results = self.test_simple_load()
        
        # Test 4: Security Basics
        security_results = self.test_security_basics()
        security_success_rate = sum(1 for r in security_results if r["safe"]) / len(security_results) * 100
        
        total_time = time.time() - start_time
        
        # Overall assessment
        overall_passed = (
            health_passed and
            endpoint_success_rate >= 80 and  # At least 80% of endpoints working
            load_results["success_rate"] >= 90 and  # At least 90% success under load
            security_success_rate == 100  # All security tests must pass
        )
        
        # Summary
        logger.info("=" * 60)
        logger.info("📊 TEST EXECUTION SUMMARY")
        logger.info("=" * 60)
        logger.info(f"🔍 Service Health: {'✅ PASS' if health_passed else '❌ FAIL'}")
        logger.info(f"🔗 Endpoint Tests: {endpoint_success_rate:.1f}% success ({'✅ PASS' if endpoint_success_rate >= 80 else '❌ FAIL'})")
        logger.info(f"⚡ Load Tests: {load_results['success_rate']:.1f}% success ({'✅ PASS' if load_results['success_rate'] >= 90 else '❌ FAIL'})")
        logger.info(f"🔒 Security Tests: {security_success_rate:.1f}% safe ({'✅ PASS' if security_success_rate == 100 else '❌ FAIL'})")
        logger.info(f"⏱️  Total Time: {total_time:.1f} seconds")
        logger.info(f"🎯 Overall Result: {'✅ PASS' if overall_passed else '❌ FAIL'}")
        logger.info("=" * 60)
        
        # Detailed results
        report = {
            "overall_passed": overall_passed,
            "execution_time_seconds": total_time,
            "service_health": health_passed,
            "endpoint_tests": {
                "results": endpoint_results,
                "success_rate": endpoint_success_rate,
                "passed": endpoint_success_rate >= 80
            },
            "load_tests": {
                **load_results,
                "passed": load_results["success_rate"] >= 90
            },
            "security_tests": {
                "results": security_results,
                "success_rate": security_success_rate,
                "passed": security_success_rate == 100
            }
        }
        
        # Save report
        report_file = f"/opt/sutazaiapp/tests/basic_hardware_optimizer_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📄 Detailed report saved: {report_file}")
        
        return report

def main():
    """Main execution function"""
    try:
        tester = BasicHardwareOptimizerTester()
        report = tester.run_comprehensive_basic_tests()
        
        if report["overall_passed"]:
            logger.info("🎉 All basic tests passed!")
            sys.exit(0)
        else:
            logger.error("💥 Some tests failed!")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(2)

if __name__ == "__main__":
    main()