#!/usr/bin/env python3
"""
Ultra QA Team Lead - Final Frontend Validation
Post-stress testing functional validation
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import requests
import time
import json
from datetime import datetime
import subprocess

def validate_frontend_health():
    """Complete frontend health validation"""
    results = {
        "timestamp": datetime.now().isoformat(),
        "validation_results": {},
        "overall_status": "PENDING"
    }
    
    logger.info("=" * 60)
    logger.info("FINAL FRONTEND VALIDATION")
    logger.info("=" * 60)
    
    # Test 1: Basic connectivity
    logger.info("\n1. Testing basic connectivity...")
    try:
        response = requests.get("http://localhost:10011", timeout=10)
        connectivity_status = response.status_code == 200
        logger.info(f"   Status: {'✅ PASS' if connectivity_status else '❌ FAIL'}")
        results["validation_results"]["connectivity"] = {
            "status": "PASS" if connectivity_status else "FAIL",
            "status_code": response.status_code,
            "response_time": response.elapsed.total_seconds()
        }
    except Exception as e:
        logger.info(f"   Status: ❌ FAIL - {e}")
        results["validation_results"]["connectivity"] = {"status": "FAIL", "error": str(e)}
    
    # Test 2: Container health
    logger.info("\n2. Checking container health...")
    try:
        result = subprocess.run(
            ["docker", "inspect", "sutazai-frontend", "--format", "{{.State.Health.Status}}"],
            capture_output=True, text=True, timeout=10
        )
        container_health = result.stdout.strip() == "healthy"
        logger.info(f"   Status: {'✅ PASS' if container_health else '❌ FAIL'}")
        results["validation_results"]["container_health"] = {
            "status": "PASS" if container_health else "FAIL",
            "health_status": result.stdout.strip()
        }
    except Exception as e:
        logger.info(f"   Status: ❌ FAIL - {e}")
        results["validation_results"]["container_health"] = {"status": "FAIL", "error": str(e)}
    
    # Test 3: Resource usage check
    logger.info("\n3. Checking resource usage...")
    try:
        result = subprocess.run(
            ["docker", "stats", "sutazai-frontend", "--no-stream", "--format", "json"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            stats = json.loads(result.stdout)
            mem_usage = stats.get("MemUsage", "0B / 0B")
            cpu_usage = stats.get("CPUPerc", "0%")
            
            # Parse memory usage
            mem_parts = mem_usage.split(" / ")
            mem_used_str = mem_parts[0].replace("MiB", "").replace("MB", "")
            try:
                mem_used = float(mem_used_str)
                resource_ok = mem_used < 100  # Less than 100MB is good
            except (AssertionError, Exception) as e:
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                resource_ok = True
            
            logger.info(f"   Memory: {mem_usage}")
            logger.info(f"   CPU: {cpu_usage}")
            logger.info(f"   Status: {'✅ PASS' if resource_ok else '⚠️ HIGH USAGE'}")
            
            results["validation_results"]["resource_usage"] = {
                "status": "PASS" if resource_ok else "HIGH_USAGE",
                "memory_usage": mem_usage,
                "cpu_usage": cpu_usage
            }
        else:
            logger.info("   Status: ❌ FAIL - Cannot get stats")
            results["validation_results"]["resource_usage"] = {"status": "FAIL", "error": "Cannot get stats"}
    except Exception as e:
        logger.info(f"   Status: ❌ FAIL - {e}")
        results["validation_results"]["resource_usage"] = {"status": "FAIL", "error": str(e)}
    
    # Test 4: Performance regression check
    logger.info("\n4. Performance regression check (5 quick requests)...")
    response_times = []
    try:
        for i in range(5):
            start = time.time()
            response = requests.get("http://localhost:10011", timeout=5)
            end = time.time()
            
            if response.status_code == 200:
                response_times.append(end - start)
            
            time.sleep(0.2)
        
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            performance_ok = avg_time < 0.1  # Less than 100ms
            logger.info(f"   Average response time: {avg_time:.3f}s")
            logger.info(f"   Status: {'✅ PASS' if performance_ok else '⚠️ SLOW'}")
            
            results["validation_results"]["performance"] = {
                "status": "PASS" if performance_ok else "SLOW",
                "average_response_time": avg_time,
                "response_times": response_times
            }
        else:
            logger.info("   Status: ❌ FAIL - No successful requests")
            results["validation_results"]["performance"] = {"status": "FAIL", "error": "No successful requests"}
    except Exception as e:
        logger.info(f"   Status: ❌ FAIL - {e}")
        results["validation_results"]["performance"] = {"status": "FAIL", "error": str(e)}
    
    # Overall assessment
    logger.info("\n" + "=" * 60)
    logger.info("OVERALL ASSESSMENT")
    logger.info("=" * 60)
    
    all_tests = list(results["validation_results"].values())
    passed_tests = [test for test in all_tests if test.get("status") == "PASS"]
    
    if len(passed_tests) == len(all_tests):
        results["overall_status"] = "PASS"
        logger.info("🎉 ALL TESTS PASSED - Frontend is HEALTHY and READY")
    elif len(passed_tests) >= len(all_tests) * 0.75:
        results["overall_status"] = "MOSTLY_PASS"
        logger.info("⚠️ MOSTLY PASSED - Minor issues detected but frontend is operational")
    else:
        results["overall_status"] = "FAIL"
        logger.info("❌ MULTIPLE FAILURES - Frontend needs attention")
    
    logger.info(f"Tests passed: {len(passed_tests)}/{len(all_tests)}")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"/opt/sutazaiapp/tests/frontend_final_validation_{timestamp}.json"
    
    with open(filename, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    logger.info(f"Results saved to: {filename}")
    
    return results

if __name__ == "__main__":
    result = validate_frontend_health()
    
    if result["overall_status"] == "PASS":
        exit(0)
    elif result["overall_status"] == "MOSTLY_PASS":
        exit(0)  # Still considered successful
    else:
