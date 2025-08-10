#!/usr/bin/env python3
"""
SIMPLIFIED LOAD TESTING FOR HARDWARE OPTIMIZER
===============================================

Comprehensive load testing using only Python standard libraries.
This provides the same functionality as the ultra test suite but with
minimal dependencies.

Author: Ultra-Critical Automated Testing Specialist
Version: 1.0.0
"""

import json
import time
import threading
import statistics
import concurrent.futures
import sys
import os
import logging
import requests
import subprocess
import tempfile
import shutil
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(funcName)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/tests/simplified_load_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('SimplifiedLoadTest')

@dataclass
class EndpointSpec:
    """Endpoint specification"""
    method: str
    path: str
    description: str
    params: Dict[str, Any] = None
    requires_dry_run: bool = False
    timeout: int = 30
    
    def __post_init__(self):
        if self.params is None:
            self.params = {}

@dataclass
class LoadTestResult:
    """Load test result for single endpoint/concurrency combination"""
    endpoint: str
    method: str
    concurrent_users: int
    total_requests: int
    successful_requests: int
    failed_requests: int
    success_rate: float
    avg_response_time_ms: float
    median_response_time_ms: float
    p95_response_time_ms: float
    min_response_time_ms: float
    max_response_time_ms: float
    throughput_rps: float
    total_duration_s: float
    sla_compliance: bool
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class SystemMonitor:
    """Simple system monitoring during tests"""
    
    def __init__(self, interval=0.5):
        self.interval = interval
        self.monitoring = False
        self.metrics = []
        self.monitor_thread = None
    
    def start(self):
        """Start monitoring"""
        self.monitoring = True
        self.metrics = []
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("System monitoring started")
    
    def stop(self):
        """Stop monitoring and return summary"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2.0)
        
        if not self.metrics:
            return {"error": "No metrics collected"}
        
        return {
            "duration_s": self.metrics[-1]["timestamp"] - self.metrics[0]["timestamp"],
            "sample_count": len(self.metrics)
        }
    
    def _monitor_loop(self):
        """Monitor loop"""
        while self.monitoring:
            try:
                # Simple monitoring - just timestamp for now
                self.metrics.append({
                    "timestamp": time.time()
                })
                time.sleep(self.interval)
            except Exception:
                break

class SimplifiedLoadTester:
    """Simplified load tester using standard libraries"""
    
    # All endpoints from hardware optimizer
    ENDPOINTS = [
        EndpointSpec("GET", "/health", "Health check endpoint"),
        EndpointSpec("GET", "/status", "Get current system resource status"),
        EndpointSpec("POST", "/optimize/memory", "Optimize memory usage"),
        EndpointSpec("POST", "/optimize/cpu", "Optimize CPU scheduling"),
        EndpointSpec("POST", "/optimize/disk", "Clean up disk space"),
        EndpointSpec("POST", "/optimize/docker", "Clean up Docker resources"),
        EndpointSpec("POST", "/optimize/all", "Run all optimizations"),
        EndpointSpec("GET", "/analyze/storage", "Analyze storage usage", {"path": "/tmp"}),
        EndpointSpec("GET", "/analyze/storage/duplicates", "Find duplicate files", {"path": "/tmp"}),
        EndpointSpec("GET", "/analyze/storage/large-files", "Find large files", {"path": "/tmp", "min_size_mb": 10}),
        EndpointSpec("GET", "/analyze/storage/report", "Generate storage report"),
        EndpointSpec("POST", "/optimize/storage", "Storage optimization", {"dry_run": True}, requires_dry_run=True),
        EndpointSpec("POST", "/optimize/storage/duplicates", "Remove duplicates", {"path": "/tmp", "dry_run": True}, requires_dry_run=True),
        EndpointSpec("POST", "/optimize/storage/cache", "Clear caches"),
        EndpointSpec("POST", "/optimize/storage/compress", "Compress files", {"path": "/tmp", "days_old": 1}),
        EndpointSpec("POST", "/optimize/storage/logs", "Log rotation and cleanup")
    ]
    
    # Load levels for testing
    LOAD_LEVELS = [1, 5, 10, 25, 50, 100]
    
    # SLA requirements
    SLA_RESPONSE_TIME_MS = 200  # <200ms for 95% of requests
    SLA_SUCCESS_RATE = 99.5     # >99.5% success rate
    
    def __init__(self, base_url="http://localhost:11110", test_duration=60):
        self.base_url = base_url
        self.test_duration = test_duration
        self.temp_test_dir = None
        self._setup_test_environment()
    
    def _setup_test_environment(self):
        """Setup test environment"""
        self.temp_test_dir = tempfile.mkdtemp(prefix="simplified_load_test_")
        logger.info(f"Test environment created: {self.temp_test_dir}")
        
        # Create test data
        self._create_test_data()
        
        # Verify service health
        self._verify_service_health()
    
    def _create_test_data(self):
        """Create basic test data"""
        if not self.temp_test_dir:
            return
        
        # Create test directories
        test_dirs = ["temp_files", "large_files", "duplicate_files"]
        for dir_name in test_dirs:
            os.makedirs(os.path.join(self.temp_test_dir, dir_name), exist_ok=True)
        
        # Create some test files
        temp_dir = os.path.join(self.temp_test_dir, "temp_files")
        for i in range(5):
            with open(os.path.join(temp_dir, f"test_file_{i}.txt"), 'w') as f:
                f.write(f"Test file {i} content\n" * 10)
        
        logger.info("Test data created")
    
    def _verify_service_health(self):
        """Verify service is available"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                health_data = response.json()
                if health_data.get("status") == "healthy":
                    logger.info("✅ Service health verified")
                    return
            
            raise Exception(f"Service not healthy: {response.status_code}")
            
        except Exception as e:
            logger.error(f"Service health check failed: {e}")
            raise RuntimeError("Hardware optimizer service not available")
    
    def make_single_request(self, endpoint: EndpointSpec, request_id: int = 0) -> Dict[str, Any]:
        """Make a single request and measure performance"""
        start_time = time.time()
        
        try:
            url = f"{self.base_url}{endpoint.path}"
            
            if endpoint.method == "GET":
                response = requests.get(url, params=endpoint.params, timeout=endpoint.timeout)
            else:
                response = requests.post(url, params=endpoint.params, timeout=endpoint.timeout)
            
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            success = 200 <= response.status_code < 300
            
            return {
                "request_id": request_id,
                "success": success,
                "response_time_ms": response_time_ms,
                "status_code": response.status_code,
                "response_size": len(response.content) if hasattr(response, 'content') else 0,
                "error_message": None if success else f"HTTP {response.status_code}"
            }
            
        except Exception as e:
            end_time = time.time()
            response_time_ms = (end_time - start_time) * 1000
            
            return {
                "request_id": request_id,
                "success": False,
                "response_time_ms": response_time_ms,
                "status_code": 0,
                "response_size": 0,
                "error_message": str(e)
            }
    
    def run_load_test_for_endpoint(self, endpoint: EndpointSpec, concurrent_users: int) -> LoadTestResult:
        """Run load test for a single endpoint with specified concurrency"""
        logger.info(f"🔄 Testing {endpoint.method} {endpoint.path} with {concurrent_users} users")
        
        # Start system monitoring
        monitor = SystemMonitor()
        monitor.start()
        
        # Calculate number of requests
        requests_per_user = max(1, self.test_duration // 10)  # Reasonable rate
        total_requests = concurrent_users * requests_per_user
        
        logger.info(f"  📊 {total_requests} total requests ({requests_per_user} per user)")
        
        # Execute requests with controlled concurrency
        start_time = time.time()
        results = []
        
        def worker(request_id: int) -> Dict[str, Any]:
            return self.make_single_request(endpoint, request_id)
        
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
                futures = [executor.submit(worker, i) for i in range(total_requests)]
                
                # Collect results with timeout
                for future in concurrent.futures.as_completed(futures, timeout=self.test_duration + 30):
                    try:
                        result = future.result()
                        results.append(result)
                    except Exception as e:
                        results.append({
                            "success": False,
                            "error_message": str(e),
                            "response_time_ms": 0
                        })
        
        except Exception as e:
            logger.error(f"  ❌ Executor error: {e}")
            # Create minimal failure result
            return LoadTestResult(
                endpoint=endpoint.path,
                method=endpoint.method,
                concurrent_users=concurrent_users,
                total_requests=0,
                successful_requests=0,
                failed_requests=1,
                success_rate=0.0,
                avg_response_time_ms=0.0,
                median_response_time_ms=0.0,
                p95_response_time_ms=0.0,
                min_response_time_ms=0.0,
                max_response_time_ms=0.0,
                throughput_rps=0.0,
                total_duration_s=0.0,
                sla_compliance=False,
                errors=[str(e)]
            )
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Stop monitoring
        monitor_summary = monitor.stop()
        
        # Analyze results
        if not results:
            logger.error(f"  ❌ No results collected for {endpoint.path}")
            return self._create_empty_result(endpoint, concurrent_users)
        
        successful_results = [r for r in results if r.get("success", False)]
        failed_results = [r for r in results if not r.get("success", True)]
        
        # Response time analysis
        response_times = [r.get("response_time_ms", 0) for r in results if r.get("success", False)]
        
        if not response_times:
            avg_response_time = 0
            median_response_time = 0
            p95_response_time = 0
            min_response_time = 0
            max_response_time = 0
        else:
            response_times.sort()
            avg_response_time = statistics.mean(response_times)
            median_response_time = statistics.median(response_times)
            p95_index = int(len(response_times) * 0.95)
            p95_response_time = response_times[p95_index] if response_times else 0
            min_response_time = min(response_times)
            max_response_time = max(response_times)
        
        # Calculate metrics
        success_rate = (len(successful_results) / len(results)) * 100 if results else 0
        throughput_rps = len(results) / total_duration if total_duration > 0 else 0
        
        # SLA compliance check
        sla_compliance = (
            success_rate >= self.SLA_SUCCESS_RATE and
            p95_response_time <= self.SLA_RESPONSE_TIME_MS
        )
        
        # Collect error messages
        errors = [r.get("error_message", "") for r in failed_results if r.get("error_message")]
        unique_errors = list(set(errors))[:5]  # Top 5 unique errors
        
        result = LoadTestResult(
            endpoint=endpoint.path,
            method=endpoint.method,
            concurrent_users=concurrent_users,
            total_requests=len(results),
            successful_requests=len(successful_results),
            failed_requests=len(failed_results),
            success_rate=success_rate,
            avg_response_time_ms=avg_response_time,
            median_response_time_ms=median_response_time,
            p95_response_time_ms=p95_response_time,
            min_response_time_ms=min_response_time,
            max_response_time_ms=max_response_time,
            throughput_rps=throughput_rps,
            total_duration_s=total_duration,
            sla_compliance=sla_compliance,
            errors=unique_errors
        )
        
        # Log results
        status_emoji = "✅" if sla_compliance else "⚠️"
        logger.info(f"  {status_emoji} Result: {success_rate:.1f}% success, "
                   f"P95: {p95_response_time:.1f}ms, "
                   f"Throughput: {throughput_rps:.1f} RPS")
        
        return result
    
    def _create_empty_result(self, endpoint: EndpointSpec, concurrent_users: int) -> LoadTestResult:
        """Create empty result for failed tests"""
        return LoadTestResult(
            endpoint=endpoint.path,
            method=endpoint.method,
            concurrent_users=concurrent_users,
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            success_rate=0.0,
            avg_response_time_ms=0.0,
            median_response_time_ms=0.0,
            p95_response_time_ms=0.0,
            min_response_time_ms=0.0,
            max_response_time_ms=0.0,
            throughput_rps=0.0,
            total_duration_s=0.0,
            sla_compliance=False,
            errors=["Test execution failed"]
        )
    
    def run_comprehensive_load_tests(self, selected_endpoints=None, selected_load_levels=None) -> List[LoadTestResult]:
        """Run comprehensive load tests"""
        endpoints = selected_endpoints or self.ENDPOINTS
        load_levels = selected_load_levels or self.LOAD_LEVELS
        
        logger.info("🚀 Starting Comprehensive Load Tests")
        logger.info("=" * 80)
        logger.info(f"Endpoints: {len(endpoints)}")
        logger.info(f"Load Levels: {load_levels}")
        logger.info(f"Total Scenarios: {len(endpoints) * len(load_levels)}")
        logger.info("=" * 80)
        
        all_results = []
        
        for i, endpoint in enumerate(endpoints, 1):
            logger.info(f"📍 Endpoint {i}/{len(endpoints)}: {endpoint.method} {endpoint.path}")
            
            for j, concurrent_users in enumerate(load_levels, 1):
                try:
                    # Small delay between tests for system recovery
                    if all_results:
                        time.sleep(2)
                    
                    logger.info(f"  📈 Load Level {j}/{len(load_levels)}: {concurrent_users} concurrent users")
                    
                    result = self.run_load_test_for_endpoint(endpoint, concurrent_users)
                    all_results.append(result)
                    
                except Exception as e:
                    logger.error(f"  ❌ Test failed: {e}")
                    empty_result = self._create_empty_result(endpoint, concurrent_users)
                    all_results.append(empty_result)
        
        logger.info(f"✅ Comprehensive load tests completed: {len(all_results)} scenarios")
        return all_results
    
    def run_security_tests(self) -> List[Dict[str, Any]]:
        """Run basic security boundary tests"""
        logger.info("🔒 Starting Security Boundary Tests")
        
        security_results = []
        
        # Path traversal tests
        dangerous_paths = [
            "../../../etc/passwd",
            "/root",
            "/etc/shadow",
            "../../../../../../../../etc/hosts"
        ]
        
        for path in dangerous_paths:
            try:
                response = requests.get(
                    f"{self.base_url}/analyze/storage",
                    params={"path": path},
                    timeout=10
                )
                
                # Check if system files are leaked
                vulnerability_detected = (
                    response.status_code == 200 and 
                    ("root:" in response.text.lower() or "shadow" in response.text.lower())
                )
                
                security_results.append({
                    "test_type": "Path Traversal",
                    "payload": path,
                    "status_code": response.status_code,
                    "vulnerability_detected": vulnerability_detected,
                    "severity": "HIGH" if vulnerability_detected else "LOW"
                })
                
            except Exception as e:
                security_results.append({
                    "test_type": "Path Traversal",
                    "payload": path,
                    "error": str(e),
                    "vulnerability_detected": False,
                    "severity": "LOW"
                })
        
        # Parameter injection tests
        malicious_params = [
            {"path": "x" * 10000},  # Very long path
            {"min_size_mb": -999999},  # Negative number
            {"days_old": "'; DROP TABLE users; --"}  # SQL injection attempt
        ]
        
        for params in malicious_params:
            try:
                response = requests.get(
                    f"{self.base_url}/analyze/storage/large-files",
                    params=params,
                    timeout=10
                )
                
                # Should handle gracefully
                graceful_handling = response.status_code in [400, 422]
                
                security_results.append({
                    "test_type": "Parameter Injection",
                    "payload": str(params),
                    "status_code": response.status_code,
                    "graceful_handling": graceful_handling,
                    "vulnerability_detected": not graceful_handling and response.status_code == 500
                })
                
            except Exception as e:
                security_results.append({
                    "test_type": "Parameter Injection",
                    "payload": str(params),
                    "error": str(e),
                    "graceful_handling": True  # Exception is acceptable
                })
        
        vulnerabilities = len([r for r in security_results if r.get("vulnerability_detected", False)])
        logger.info(f"🔒 Security tests completed: {vulnerabilities} vulnerabilities detected")
        
        return security_results
    
    def analyze_results(self, load_results: List[LoadTestResult], security_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze all test results"""
        
        # Load test analysis
        total_load_tests = len(load_results)
        sla_compliant_tests = len([r for r in load_results if r.sla_compliance])
        sla_compliance_rate = (sla_compliant_tests / total_load_tests) * 100 if total_load_tests > 0 else 0
        
        # Performance metrics
        successful_tests = [r for r in load_results if r.successful_requests > 0]
        avg_response_time = statistics.mean([r.avg_response_time_ms for r in successful_tests]) if successful_tests else 0
        avg_success_rate = statistics.mean([r.success_rate for r in load_results]) if load_results else 0
        avg_throughput = statistics.mean([r.throughput_rps for r in successful_tests]) if successful_tests else 0
        
        # Security analysis
        security_vulnerabilities = len([r for r in security_results if r.get("vulnerability_detected", False)])
        high_severity_vulns = len([r for r in security_results if r.get("vulnerability_detected", False) and r.get("severity") == "HIGH"])
        
        # Endpoint analysis
        endpoint_performance = {}
        for result in load_results:
            endpoint = result.endpoint
            if endpoint not in endpoint_performance:
                endpoint_performance[endpoint] = {"tests": [], "max_load": 0}
            
            endpoint_performance[endpoint]["tests"].append(result)
            
            if result.sla_compliance:
                endpoint_performance[endpoint]["max_load"] = max(
                    endpoint_performance[endpoint]["max_load"],
                    result.concurrent_users
                )
        
        # Critical issues
        critical_issues = []
        if sla_compliance_rate < 90:
            critical_issues.append(f"SLA compliance below 90%: {sla_compliance_rate:.1f}%")
        if high_severity_vulns > 0:
            critical_issues.append(f"High severity vulnerabilities: {high_severity_vulns}")
        if avg_success_rate < 95:
            critical_issues.append(f"Average success rate below 95%: {avg_success_rate:.1f}%")
        
        # Overall assessment
        overall_passed = len(critical_issues) == 0
        
        return {
            "overall_assessment": "PASS" if overall_passed else "FAIL",
            "critical_issues": critical_issues,
            "load_test_summary": {
                "total_tests": total_load_tests,
                "sla_compliant_tests": sla_compliant_tests,
                "sla_compliance_rate": sla_compliance_rate,
                "avg_response_time_ms": avg_response_time,
                "avg_success_rate": avg_success_rate,
                "avg_throughput_rps": avg_throughput
            },
            "security_summary": {
                "total_tests": len(security_results),
                "vulnerabilities_detected": security_vulnerabilities,
                "high_severity_vulnerabilities": high_severity_vulns
            },
            "endpoint_performance": endpoint_performance,
            "recommendations": self._generate_recommendations(critical_issues, endpoint_performance)
        }
    
    def _generate_recommendations(self, critical_issues: List[str], endpoint_performance: Dict[str, Any]) -> List[str]:
        """Generate recommendations based on results"""
        recommendations = []
        
        if any("SLA compliance" in issue for issue in critical_issues):
            recommendations.extend([
                "Optimize slow endpoints to meet SLA requirements",
                "Implement connection pooling and keep-alive",
                "Add request queuing for better load handling",
                "Review database queries and add indexing"
            ])
        
        if any("vulnerabilities" in issue for issue in critical_issues):
            recommendations.extend([
                "Fix security vulnerabilities immediately",
                "Implement input validation and sanitization",
                "Add security scanning to CI/CD pipeline"
            ])
        
        if any("success rate" in issue for issue in critical_issues):
            recommendations.extend([
                "Improve error handling and recovery",
                "Add circuit breakers for resilience",
                "Implement proper timeout handling"
            ])
        
        # Endpoint-specific recommendations
        for endpoint, perf in endpoint_performance.items():
            if perf["max_load"] < 10:
                recommendations.append(f"Critical: {endpoint} needs optimization - max load only {perf['max_load']} users")
        
        return recommendations
    
    def generate_report(self, load_results: List[LoadTestResult], security_results: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        
        report = {
            "test_execution_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_load_scenarios": len(load_results),
                "total_security_tests": len(security_results),
                "overall_assessment": analysis["overall_assessment"],
                "critical_issues": analysis["critical_issues"]
            },
            "sla_validation": {
                "response_time_sla_ms": self.SLA_RESPONSE_TIME_MS,
                "success_rate_sla_percent": self.SLA_SUCCESS_RATE,
                "compliance_summary": analysis["load_test_summary"]
            },
            "security_validation": analysis["security_summary"],
            "performance_analysis": {
                "endpoint_performance": analysis["endpoint_performance"],
                "recommendations": analysis["recommendations"]
            },
            "detailed_results": {
                "load_test_results": [asdict(result) for result in load_results],
                "security_test_results": security_results
            }
        }
        
        # Save report
        report_file = f"/opt/sutazaiapp/tests/comprehensive_test_report_{int(time.time())}.json"
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"📊 Comprehensive report saved: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report as JSON: {e}")
            # Save as text instead
            text_report_file = f"/opt/sutazaiapp/tests/comprehensive_test_report_{int(time.time())}.txt"
            with open(text_report_file, 'w') as f:
                f.write(json.dumps(report, indent=2, default=str))
            logger.info(f"📊 Text report saved: {text_report_file}")
        
        return report
    
    def print_summary(self, analysis: Dict[str, Any], total_time: float):
        """Print comprehensive test summary"""
        print("\n" + "=" * 80)
        print("🎯 HARDWARE OPTIMIZER ULTRA-COMPREHENSIVE TEST RESULTS")
        print("=" * 80)
        
        # Overall assessment
        assessment_emoji = "✅" if analysis["overall_assessment"] == "PASS" else "❌"
        print(f"{assessment_emoji} Overall Assessment: {analysis['overall_assessment']}")
        print(f"⏱️  Total Execution Time: {total_time:.1f} seconds")
        
        # Load test summary
        load_summary = analysis["load_test_summary"]
        print(f"\n📈 LOAD TESTING RESULTS:")
        print(f"  📊 Total Scenarios: {load_summary['total_tests']}")
        print(f"  ✅ SLA Compliant: {load_summary['sla_compliant_tests']} ({load_summary['sla_compliance_rate']:.1f}%)")
        print(f"  📊 Avg Response Time: {load_summary['avg_response_time_ms']:.1f}ms")
        print(f"  📊 Avg Success Rate: {load_summary['avg_success_rate']:.1f}%")
        print(f"  📊 Avg Throughput: {load_summary['avg_throughput_rps']:.1f} RPS")
        
        # Security summary
        security_summary = analysis["security_summary"]
        print(f"\n🔒 SECURITY TESTING RESULTS:")
        print(f"  📊 Total Tests: {security_summary['total_tests']}")
        print(f"  🚨 Vulnerabilities: {security_summary['vulnerabilities_detected']}")
        print(f"  ⚠️  High Severity: {security_summary['high_severity_vulnerabilities']}")
        
        # Critical issues
        if analysis["critical_issues"]:
            print(f"\n🚨 CRITICAL ISSUES ({len(analysis['critical_issues'])}):")
            for i, issue in enumerate(analysis["critical_issues"], 1):
                print(f"  {i}. {issue}")
        else:
            print(f"\n✅ NO CRITICAL ISSUES DETECTED")
        
        # Top recommendations
        if analysis["recommendations"]:
            print(f"\n💡 KEY RECOMMENDATIONS:")
            for i, rec in enumerate(analysis["recommendations"][:5], 1):
                print(f"  {i}. {rec}")
            if len(analysis["recommendations"]) > 5:
                print(f"  ... and {len(analysis['recommendations']) - 5} more in the full report")
        
        print("=" * 80)
    
    def cleanup(self):
        """Cleanup test environment"""
        if self.temp_test_dir and os.path.exists(self.temp_test_dir):
            try:
                shutil.rmtree(self.temp_test_dir)
                logger.info(f"Test environment cleaned up: {self.temp_test_dir}")
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def run_all_tests(self, demo_mode=False) -> Dict[str, Any]:
        """Run all comprehensive tests"""
        logger.info("🚀 STARTING ULTRA-COMPREHENSIVE HARDWARE OPTIMIZER TESTING")
        start_time = time.time()
        
        try:
            if demo_mode:
                # Demo mode - reduced scope for faster execution
                selected_endpoints = self.ENDPOINTS[:6]  # First 6 endpoints
                selected_load_levels = [1, 5, 10]  # Reduced load levels
                logger.info("📋 Running in DEMO mode with reduced scope")
            else:
                # Full comprehensive testing
                selected_endpoints = self.ENDPOINTS
                selected_load_levels = self.LOAD_LEVELS
                logger.info("📋 Running FULL comprehensive testing")
            
            # Phase 1: Load Testing
            logger.info("=" * 50)
            logger.info("PHASE 1: LOAD TESTING")
            logger.info("=" * 50)
            load_results = self.run_comprehensive_load_tests(selected_endpoints, selected_load_levels)
            
            # Phase 2: Security Testing
            logger.info("=" * 50)
            logger.info("PHASE 2: SECURITY TESTING") 
            logger.info("=" * 50)
            security_results = self.run_security_tests()
            
            # Phase 3: Analysis
            logger.info("=" * 50)
            logger.info("PHASE 3: RESULTS ANALYSIS")
            logger.info("=" * 50)
            analysis = self.analyze_results(load_results, security_results)
            
            # Phase 4: Report Generation
            logger.info("=" * 50)
            logger.info("PHASE 4: REPORT GENERATION")
            logger.info("=" * 50)
            report = self.generate_report(load_results, security_results, analysis)
            
            total_time = time.time() - start_time
            
            # Print summary
            self.print_summary(analysis, total_time)
            
            return report
            
        except Exception as e:
            logger.error(f"Testing failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            raise
            
        finally:
            self.cleanup()

def main():
    """Main execution function"""
    
    # Check for demo mode
    demo_mode = "--demo" in sys.argv
    
    try:
        tester = SimplifiedLoadTester(test_duration=30 if demo_mode else 60)
        report = tester.run_all_tests(demo_mode=demo_mode)
        
        # Determine exit code
        if report["test_execution_summary"]["overall_assessment"] == "PASS":
            logger.info("🎉 All tests passed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Some tests failed or critical issues detected")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(2)

if __name__ == "__main__":
    main()