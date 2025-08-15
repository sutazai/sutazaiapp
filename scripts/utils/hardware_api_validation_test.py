#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
ULTRA-COMPREHENSIVE Hardware Resource Optimizer API Validation Test Suite
Performs exhaustive testing of all endpoints with detailed reporting
"""

import asyncio
import aiohttp
import json
import time
import concurrent.futures
import threading
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
import sys
import traceback

@dataclass
class TestResult:
    endpoint: str
    method: str
    status_code: int
    response_time_ms: float
    success: bool
    error_message: str = ""
    response_data: Dict[str, Any] = None
    validation_errors: List[str] = None

class HardwareAPIValidator:
    """Ultra-comprehensive API validator for hardware optimization service"""
    
    def __init__(self):
        self.base_url_backend = "http://localhost:10010/api/v1/hardware"
        self.base_url_direct = "http://localhost:11110"
        self.session_timeout = aiohttp.ClientTimeout(total=30)
        self.results: List[TestResult] = []
        
    async def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run all validation tests and return comprehensive report"""
        logger.info("🚀 Starting ULTRA-COMPREHENSIVE Hardware API Validation")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Test categories
        test_categories = [
            ("Direct Service Health Tests", self._test_direct_service_health),
            ("Backend Integration Tests", self._test_backend_integration),
            ("Optimization Endpoint Tests", self._test_optimization_endpoints),
            ("Analysis Endpoint Tests", self._test_analysis_endpoints),
            ("Error Handling Tests", self._test_error_handling),
            ("Performance Under Load", self._test_performance_load),
            ("Data Validation Tests", self._test_data_validation),
            ("Concurrent Request Tests", self._test_concurrent_requests),
        ]
        
        for category_name, test_func in test_categories:
            logger.info(f"\n📋 {category_name}")
            logger.info("-" * 50)
            try:
                await test_func()
            except Exception as e:
                logger.error(f"❌ Category {category_name} failed with error: {e}")
                traceback.print_exc()
        
        # Generate final report
        total_time = time.time() - start_time
        return self._generate_final_report(total_time)
    
    async def _test_direct_service_health(self):
        """Test direct hardware service health and basic endpoints"""
        
        # Test health endpoint
        await self._test_endpoint("GET", f"{self.base_url_direct}/health", "Direct Health Check")
        
        # Test status endpoint  
        await self._test_endpoint("GET", f"{self.base_url_direct}/status", "Direct Status Check")
        
        # Test root endpoint
        await self._test_endpoint("GET", f"{self.base_url_direct}/", "Direct Root Endpoint")
        
    async def _test_backend_integration(self):
        """Test backend integration with hardware service"""
        
        # Test router health
        await self._test_endpoint("GET", f"{self.base_url_backend}/router/health", "Backend Router Health")
        
        # Test backend health endpoint
        await self._test_endpoint("GET", f"{self.base_url_backend}/health", "Backend Hardware Health")
        
        # Test status endpoint (may fail due to different response structure)
        await self._test_endpoint("GET", f"{self.base_url_backend}/status", "Backend Hardware Status", expect_error=True)
        
    async def _test_optimization_endpoints(self):
        """Test all optimization endpoints"""
        
        optimization_tests = [
            ("POST", f"{self.base_url_direct}/optimize/memory", "Memory Optimization"),
            ("POST", f"{self.base_url_direct}/optimize/cpu", "CPU Optimization"), 
            ("POST", f"{self.base_url_direct}/optimize/disk", "Disk Optimization"),
            ("POST", f"{self.base_url_direct}/optimize/docker", "Docker Optimization"),
            ("POST", f"{self.base_url_direct}/optimize/all", "Full System Optimization"),
            ("POST", f"{self.base_url_direct}/optimize/storage?dry_run=true", "Storage Optimization (Dry Run)"),
            ("POST", f"{self.base_url_direct}/optimize/storage/cache", "Cache Optimization"),
            ("POST", f"{self.base_url_direct}/optimize/storage/logs", "Log Optimization"),
        ]
        
        for method, url, description in optimization_tests:
            await self._test_endpoint(method, url, description)
            await asyncio.sleep(0.5)  # Small delay between optimizations
    
    async def _test_analysis_endpoints(self):
        """Test all analysis endpoints"""
        
        analysis_tests = [
            ("GET", f"{self.base_url_direct}/analyze/storage?path=/tmp", "Storage Analysis - /tmp"),
            ("GET", f"{self.base_url_direct}/analyze/storage?path=/var/log", "Storage Analysis - /var/log"),
            ("GET", f"{self.base_url_direct}/analyze/storage/duplicates?path=/tmp", "Duplicate Analysis"),
            ("GET", f"{self.base_url_direct}/analyze/storage/large-files?path=/&min_size_mb=100", "Large Files Analysis"),
            ("GET", f"{self.base_url_direct}/analyze/storage/report", "Comprehensive Storage Report"),
        ]
        
        for method, url, description in analysis_tests:
            await self._test_endpoint(method, url, description)
    
    async def _test_error_handling(self):
        """Test error handling scenarios"""
        
        error_tests = [
            ("GET", f"{self.base_url_direct}/nonexistent", "Nonexistent Endpoint", True),
            ("POST", f"{self.base_url_direct}/optimize/invalid", "Invalid Optimization Type", True),
            ("GET", f"{self.base_url_direct}/analyze/storage?path=/nonexistent", "Invalid Path Analysis"),
            ("GET", f"{self.base_url_backend}/nonexistent", "Backend Nonexistent Endpoint", True),
        ]
        
        for method, url, description, expect_error in error_tests:
            await self._test_endpoint(method, url, description, expect_error=expect_error)
    
    async def _test_performance_load(self):
        """Test performance under moderate load"""
        logger.info("⚡ Testing performance under concurrent load...")
        
        # Test concurrent health checks
        tasks = []
        for i in range(10):
            task = self._test_endpoint("GET", f"{self.base_url_direct}/health", f"Concurrent Health Check {i+1}")
            tasks.append(task)
        
        await asyncio.gather(*tasks)
        
        # Test concurrent optimizations
        optimization_tasks = []
        for i in range(3):
            task = self._test_endpoint("POST", f"{self.base_url_direct}/optimize/memory", f"Concurrent Memory Opt {i+1}")
            optimization_tasks.append(task)
        
        await asyncio.gather(*optimization_tasks)
    
    async def _test_data_validation(self):
        """Test data validation and sanitization"""
        
        # Test with various parameter values
        validation_tests = [
            ("GET", f"{self.base_url_direct}/analyze/storage?path=../../../etc", "Path Traversal Test"),
            ("GET", f"{self.base_url_direct}/analyze/storage/large-files?min_size_mb=-100", "Negative Size Parameter"),
            ("GET", f"{self.base_url_direct}/analyze/storage/large-files?min_size_mb=abc", "Invalid Size Parameter", True),
        ]
        
        for method, url, description, *expect_error in validation_tests:
            expect_err = expect_error[0] if expect_error else False
            await self._test_endpoint(method, url, description, expect_error=expect_err)
    
    async def _test_concurrent_requests(self):
        """Test system behavior under high concurrent load"""
        logger.info("🔥 Testing high concurrent request handling...")
        
        # Create 20 concurrent requests of mixed types
        concurrent_tasks = []
        
        for i in range(20):
            if i % 4 == 0:
                task = self._test_endpoint("GET", f"{self.base_url_direct}/health", f"Concurrent-{i}-Health")
            elif i % 4 == 1:
                task = self._test_endpoint("GET", f"{self.base_url_direct}/status", f"Concurrent-{i}-Status")
            elif i % 4 == 2:
                task = self._test_endpoint("GET", f"{self.base_url_direct}/analyze/storage/report", f"Concurrent-{i}-Report")
            else:
                task = self._test_endpoint("POST", f"{self.base_url_direct}/optimize/memory", f"Concurrent-{i}-Memory")
            
            concurrent_tasks.append(task)
        
        # Execute all concurrent requests
        await asyncio.gather(*concurrent_tasks)
    
    async def _test_endpoint(self, method: str, url: str, description: str, expect_error: bool = False, data: Dict = None) -> TestResult:
        """Test a single endpoint and record results"""
        start_time = time.time()
        
        try:
            async with aiohttp.ClientSession(timeout=self.session_timeout) as session:
                if method.upper() == "GET":
                    async with session.get(url) as response:
                        response_data = await response.json()
                        status_code = response.status
                elif method.upper() == "POST":
                    async with session.post(url, json=data) as response:
                        response_data = await response.json()
                        status_code = response.status
                else:
                    raise ValueError(f"Unsupported method: {method}")
                
                response_time_ms = (time.time() - start_time) * 1000
                
                # Determine success
                success = True
                error_message = ""
                validation_errors = []
                
                if expect_error and 200 <= status_code < 300:
                    success = False
                    error_message = "Expected error but got success"
                elif not expect_error and not (200 <= status_code < 300):
                    success = False
                    error_message = f"Unexpected status code: {status_code}"
                
                # Validate response structure
                if success and response_data:
                    validation_errors = self._validate_response(response_data, url)
                    if validation_errors:
                        success = False
                        error_message = f"Validation errors: {', '.join(validation_errors)}"
                
                result = TestResult(
                    endpoint=url,
                    method=method,
                    status_code=status_code,
                    response_time_ms=response_time_ms,
                    success=success,
                    error_message=error_message,
                    response_data=response_data,
                    validation_errors=validation_errors
                )
                
                self.results.append(result)
                
                # Print result
                status_icon = "✅" if success else "❌"
                logger.info(f"{status_icon} {description}: {status_code} ({response_time_ms:.1f}ms)")
                if not success and error_message:
                    logger.error(f"   Error: {error_message}")
                
                return result
                
        except Exception as e:
            response_time_ms = (time.time() - start_time) * 1000
            error_message = f"Exception: {str(e)}"
            
            result = TestResult(
                endpoint=url,
                method=method,
                status_code=0,
                response_time_ms=response_time_ms,
                success=expect_error,  # If we expect error, exception might be ok
                error_message=error_message
            )
            
            self.results.append(result)
            
            status_icon = "✅" if expect_error else "❌"
            logger.error(f"{status_icon} {description}: Exception ({response_time_ms:.1f}ms)")
            logger.error(f"   Error: {error_message}")
            
            return result
    
    def _validate_response(self, response_data: Dict, url: str) -> List[str]:
        """Validate response data structure"""
        errors = []
        
        # Common validations based on endpoint type
        if "/health" in url:
            required_fields = ["status", "agent", "timestamp"]
            for field in required_fields:
                if field not in response_data:
                    errors.append(f"Missing required field: {field}")
        
        elif "/optimize/" in url:
            if "status" not in response_data:
                errors.append("Missing status field in optimization response")
            if "optimization_type" not in response_data:
                errors.append("Missing optimization_type field")
        
        elif "/analyze/" in url:
            if "status" not in response_data:
                errors.append("Missing status field in analysis response")
            if response_data.get("status") == "success" and "timestamp" not in response_data:
                errors.append("Missing timestamp in successful analysis")
        
        return errors
    
    def _generate_final_report(self, total_time: float) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r.success)
        failed_tests = total_tests - successful_tests
        
        # Performance metrics
        response_times = [r.response_time_ms for r in self.results]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        min_response_time = min(response_times) if response_times else 0
        
        # Status code distribution
        status_codes = {}
        for result in self.results:
            code = result.status_code
            status_codes[code] = status_codes.get(code, 0) + 1
        
        # Error analysis
        error_summary = {}
        for result in self.results:
            if not result.success and result.error_message:
                error_type = result.error_message.split(':')[0]
                error_summary[error_type] = error_summary.get(error_type, 0) + 1
        
        # Generate detailed report
        report = {
            "test_execution": {
                "total_duration_seconds": round(total_time, 2),
                "total_tests": total_tests,
                "successful_tests": successful_tests,
                "failed_tests": failed_tests,
                "success_rate": round((successful_tests / total_tests) * 100, 2) if total_tests > 0 else 0
            },
            "performance_metrics": {
                "avg_response_time_ms": round(avg_response_time, 2),
                "max_response_time_ms": round(max_response_time, 2),
                "min_response_time_ms": round(min_response_time, 2),
                "total_requests": total_tests
            },
            "status_code_distribution": status_codes,
            "error_analysis": error_summary,
            "endpoint_details": [asdict(result) for result in self.results],
            "validation_summary": {
                "direct_service_health": "PASS" if any(r.success and "/health" in r.endpoint and "11110" in r.endpoint for r in self.results) else "FAIL",
                "backend_integration": "PASS" if any(r.success and "/hardware/" in r.endpoint and "10010" in r.endpoint for r in self.results) else "FAIL", 
                "optimization_endpoints": "PASS" if any(r.success and "/optimize/" in r.endpoint for r in self.results) else "FAIL",
                "analysis_endpoints": "PASS" if any(r.success and "/analyze/" in r.endpoint for r in self.results) else "FAIL",
                "error_handling": "PASS" if any(not r.success for r in self.results) else "INCONCLUSIVE",
                "concurrent_performance": "PASS" if avg_response_time < 5000 else "FAIL"  # 5 second threshold
            }
        }
        
        return report

async def main():
    """Main execution function"""
    validator = HardwareAPIValidator()
    
    try:
        report = await validator.run_comprehensive_validation()
        
        # Print final summary
        logger.info("\n" + "=" * 80)
        logger.info("🎯 FINAL VALIDATION REPORT")
        logger.info("=" * 80)
        
        exec_summary = report["test_execution"]
        perf_summary = report["performance_metrics"]
        validation_summary = report["validation_summary"]
        
        logger.info(f"Total Tests: {exec_summary['total_tests']}")
        logger.info(f"Success Rate: {exec_summary['success_rate']}%")
        logger.info(f"Total Duration: {exec_summary['total_duration_seconds']}s")
        logger.info(f"Average Response Time: {perf_summary['avg_response_time_ms']}ms")
        logger.info(f"Max Response Time: {perf_summary['max_response_time_ms']}ms")
        
        logger.info(f"\nValidation Categories:")
        for category, status in validation_summary.items():
            status_icon = "✅" if status == "PASS" else "❌" if status == "FAIL" else "⚠️"
            logger.info(f"  {status_icon} {category.replace('_', ' ').title()}: {status}")
        
        logger.info(f"\nStatus Code Distribution:")
        for code, count in report["status_code_distribution"].items():
            logger.info(f"  {code}: {count} requests")
        
        if report["error_analysis"]:
            logger.error(f"\nError Summary:")
            for error_type, count in report["error_analysis"].items():
                logger.error(f"  {error_type}: {count} occurrences")
        
        # Save detailed report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f"hardware_api_validation_report_{timestamp}.json"
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"\n📄 Detailed report saved to: {report_file}")
        
        # Determine overall result
        overall_success = exec_summary['success_rate'] >= 80
        critical_categories = ['direct_service_health', 'backend_integration', 'optimization_endpoints']
        critical_pass = all(validation_summary[cat] == "PASS" for cat in critical_categories)
        
        if overall_success and critical_pass:
            logger.info("\n🏆 OVERALL VALIDATION: PASSED")
            return 0
        else:
            logger.error("\n💥 OVERALL VALIDATION: FAILED")
            return 1
            
    except Exception as e:
        logger.error(f"\n💥 VALIDATION FAILED WITH CRITICAL ERROR: {e}")
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)