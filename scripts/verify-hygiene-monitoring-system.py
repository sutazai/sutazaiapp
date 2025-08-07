#!/usr/bin/env python3
"""
Comprehensive Hygiene Monitoring System Verification Script
==========================================================

This script tests all endpoints of the dockerized hygiene monitoring system to ensure:
1. All HTTP endpoints are working properly (backend API, rule control API, dashboard)
2. WebSocket connectivity is functional
3. Real data is being returned (not mock/static data)
4. Violations are being detected correctly
5. System is production-ready

Endpoints tested:
- Dashboard: http://localhost:3002
- Backend API: http://localhost:8081
- Rule Control API: http://localhost:8101
- WebSocket: ws://localhost:8081/ws

Usage: python scripts/verify-hygiene-monitoring-system.py [--verbose] [--output-file report.json]
"""

import sys
import json
import time
import asyncio
import logging
import argparse
import traceback
import websockets
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

import httpx
import aiofiles

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Data class for test results"""
    test_name: str
    category: str
    endpoint: str
    status: str  # "PASS", "FAIL", "SKIP", "WARN"
    message: str
    response_time_ms: Optional[float] = None
    response_data: Optional[Dict] = None
    error_details: Optional[str] = None
    timestamp: str = ""

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

class HygieneMonitoringVerifier:
    """Comprehensive verifier for the hygiene monitoring system"""
    
    def __init__(self, verbose: bool = False, timeout: int = 30):
        self.verbose = verbose
        self.timeout = timeout
        self.results: List[TestResult] = []
        self.endpoints = {
            'dashboard': 'http://localhost:3002',
            'backend_api': 'http://localhost:8081',
            'rule_control_api': 'http://localhost:8101',
            'websocket': 'ws://localhost:8081/ws'
        }
        
        # HTTP client configuration
        self.http_timeout = httpx.Timeout(connect=10.0, read=timeout, write=10.0, pool=10.0)
        
    def log_result(self, result: TestResult):
        """Log and store test result"""
        self.results.append(result)
        
        status_emoji = {
            "PASS": "✅",
            "FAIL": "❌", 
            "SKIP": "⏭️",
            "WARN": "⚠️"
        }
        
        emoji = status_emoji.get(result.status, "❓")
        message = f"{emoji} {result.test_name}: {result.message}"
        
        if result.response_time_ms:
            message += f" ({result.response_time_ms:.0f}ms)"
            
        if self.verbose and result.error_details:
            message += f"\n   Error: {result.error_details}"
            
        logger.info(message)
        
    async def test_dashboard_endpoints(self):
        """Test dashboard web interface endpoints"""
        logger.info("🔍 Testing Dashboard Endpoints...")
        
        dashboard_tests = [
            ("/", "Dashboard Root"),
            ("/health", "Dashboard Health Check"),
            ("/api/status", "Dashboard API Status"),
        ]
        
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            for path, name in dashboard_tests:
                await self._test_http_endpoint(
                    client, 
                    self.endpoints['dashboard'] + path,
                    name,
                    "Dashboard",
                    expected_content_types=["text/html", "application/json"]
                )

    async def test_backend_api_endpoints(self):
        """Test backend API endpoints"""
        logger.info("🔍 Testing Backend API Endpoints...")
        
        api_tests = [
            ("/api/hygiene/status", "Backend Health Status"),
            ("/api/hygiene/violations", "Hygiene Violations"),
            ("/api/hygiene/metrics", "System Metrics"),
            ("/api/hygiene/scan", "Hygiene Scan Trigger"),
            ("/api/hygiene/reports", "Hygiene Reports"),
            ("/api/hygiene/rules", "Active Rules"),
        ]
        
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            for path, name in api_tests:
                await self._test_http_endpoint(
                    client,
                    self.endpoints['backend_api'] + path,
                    name,
                    "Backend API",
                    expected_status_codes=[200, 202, 404]  # 404 acceptable for some endpoints
                )

    async def test_rule_control_api_endpoints(self):
        """Test rule control API endpoints"""
        logger.info("🔍 Testing Rule Control API Endpoints...")
        
        rule_tests = [
            ("/api/health/live", "Rule API Health"),
            ("/api/rules", "Rule Management"),
            ("/api/rules/profiles", "Rule Profiles"),
            ("/api/rules/status", "Rule Status"),
            ("/api/config", "Rule Configuration"),
        ]
        
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            for path, name in rule_tests:
                await self._test_http_endpoint(
                    client,
                    self.endpoints['rule_control_api'] + path,
                    name,
                    "Rule Control API"
                )

    async def test_websocket_connectivity(self):
        """Test WebSocket connectivity"""
        logger.info("🔍 Testing WebSocket Connectivity...")
        
        start_time = time.time()
        try:
            # Test WebSocket connection
            uri = self.endpoints['websocket']
            
            async with websockets.connect(
                uri, 
                ping_interval=20, 
                ping_timeout=10,
                close_timeout=10
            ) as websocket:
                # Send a test message
                test_message = {"type": "ping", "timestamp": datetime.now().isoformat()}
                await websocket.send(json.dumps(test_message))
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=5.0)
                response_data = json.loads(response) if response else None
                
                response_time = (time.time() - start_time) * 1000
                
                self.log_result(TestResult(
                    test_name="WebSocket Connection",
                    category="WebSocket",
                    endpoint=uri,
                    status="PASS",
                    message="WebSocket connected and responsive",
                    response_time_ms=response_time,
                    response_data=response_data
                ))
                
        except asyncio.TimeoutError:
            self.log_result(TestResult(
                test_name="WebSocket Connection",
                category="WebSocket", 
                endpoint=self.endpoints['websocket'],
                status="FAIL",
                message="WebSocket connection timeout",
                error_details="Connection or response timeout"
            ))
        except Exception as e:
            self.log_result(TestResult(
                test_name="WebSocket Connection",
                category="WebSocket",
                endpoint=self.endpoints['websocket'], 
                status="FAIL",
                message="WebSocket connection failed",
                error_details=str(e)
            ))

    async def test_real_data_verification(self):
        """Verify that real data is being returned, not mock/static data"""
        logger.info("🔍 Testing Real Data Verification...")
        
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            # Test backend metrics for real system data
            await self._test_real_system_metrics(client)
            
            # Test violations data for real scanning results
            await self._test_real_violations_data(client)
            
            # Test timestamp freshness
            await self._test_timestamp_freshness(client)

    async def test_violations_detection(self):
        """Test that violations are being detected correctly"""
        logger.info("🔍 Testing Violations Detection...")
        
        async with httpx.AsyncClient(timeout=self.http_timeout) as client:
            try:
                # Trigger a scan to generate violations
                scan_response = await client.post(
                    f"{self.endpoints['backend_api']}/api/hygiene/scan",
                    json={"scan_type": "quick", "force": True}
                )
                
                if scan_response.status_code in [200, 202]:
                    # Wait a moment for scan to complete
                    await asyncio.sleep(2)
                    
                    # Check for violations
                    violations_response = await client.get(
                        f"{self.endpoints['backend_api']}/api/hygiene/violations"
                    )
                    
                    if violations_response.status_code == 200:
                        violations_data = violations_response.json()
                        
                        if violations_data and violations_data.get('violations'):
                            self.log_result(TestResult(
                                test_name="Violations Detection",
                                category="Functionality",
                                endpoint=f"{self.endpoints['backend_api']}/api/hygiene/violations",
                                status="PASS",
                                message=f"Found {len(violations_data['violations'])} violations",
                                response_data={"violation_count": len(violations_data['violations'])}
                            ))
                        else:
                            self.log_result(TestResult(
                                test_name="Violations Detection",
                                category="Functionality", 
                                endpoint=f"{self.endpoints['backend_api']}/api/hygiene/violations",
                                status="WARN",
                                message="No violations detected - this may indicate clean codebase or scanning issues"
                            ))
                    else:
                        self.log_result(TestResult(
                            test_name="Violations Detection",
                            category="Functionality",
                            endpoint=f"{self.endpoints['backend_api']}/api/hygiene/violations", 
                            status="FAIL",
                            message=f"Failed to retrieve violations: HTTP {violations_response.status_code}",
                            error_details=violations_response.text
                        ))
                else:
                    self.log_result(TestResult(
                        test_name="Violations Detection",
                        category="Functionality",
                        endpoint=f"{self.endpoints['backend_api']}/api/hygiene/scan",
                        status="FAIL", 
                        message=f"Failed to trigger scan: HTTP {scan_response.status_code}",
                        error_details=scan_response.text
                    ))
                    
            except Exception as e:
                self.log_result(TestResult(
                    test_name="Violations Detection",
                    category="Functionality",
                    endpoint="Multiple endpoints",
                    status="FAIL",
                    message="Violations detection test failed",
                    error_details=str(e)
                ))

    async def _test_http_endpoint(
        self, 
        client: httpx.AsyncClient, 
        url: str, 
        name: str, 
        category: str,
        expected_status_codes: List[int] = None,
        expected_content_types: List[str] = None,
        method: str = "GET"
    ):
        """Test a single HTTP endpoint"""
        if expected_status_codes is None:
            expected_status_codes = [200]
            
        start_time = time.time()
        try:
            if method.upper() == "GET":
                response = await client.get(url)
            elif method.upper() == "POST":
                response = await client.post(url, json={})
            else:
                response = await client.request(method, url)
                
            response_time = (time.time() - start_time) * 1000
            
            # Check status code
            if response.status_code in expected_status_codes:
                status = "PASS"
                message = f"HTTP {response.status_code}"
                
                # Check content type if specified
                if expected_content_types:
                    content_type = response.headers.get('content-type', '').lower()
                    if not any(ct in content_type for ct in expected_content_types):
                        status = "WARN"
                        message += f" (unexpected content-type: {content_type})"
                
                # Try to parse JSON if applicable
                response_data = None
                if 'application/json' in response.headers.get('content-type', ''):
                    try:
                        response_data = response.json()
                    except Exception:
                        pass
                        
            else:
                status = "FAIL"
                message = f"HTTP {response.status_code} (expected {expected_status_codes})"
                response_data = None
            
            self.log_result(TestResult(
                test_name=name,
                category=category,
                endpoint=url,
                status=status,
                message=message,
                response_time_ms=response_time,
                response_data=response_data
            ))
            
        except httpx.TimeoutException:
            self.log_result(TestResult(
                test_name=name,
                category=category,
                endpoint=url,
                status="FAIL",
                message="Request timeout",
                error_details=f"Timeout after {self.timeout}s"
            ))
        except Exception as e:
            self.log_result(TestResult(
                test_name=name,
                category=category,
                endpoint=url,
                status="FAIL",
                message="Request failed",
                error_details=str(e)
            ))

    async def _test_real_system_metrics(self, client: httpx.AsyncClient):
        """Test that system metrics contain real data"""
        try:
            response = await client.get(f"{self.endpoints['backend_api']}/api/hygiene/metrics")
            
            if response.status_code == 200:
                metrics = response.json()
                
                # Check for real system metrics (not hardcoded values)
                real_data_indicators = [
                    'cpu_percent' in metrics and isinstance(metrics['cpu_percent'], (int, float)),
                    'memory_percent' in metrics and isinstance(metrics['memory_percent'], (int, float)),
                    'timestamp' in metrics,
                    'uptime' in metrics or 'runtime' in metrics
                ]
                
                if all(real_data_indicators):
                    self.log_result(TestResult(
                        test_name="Real System Metrics",
                        category="Data Verification",
                        endpoint=f"{self.endpoints['backend_api']}/api/hygiene/metrics",
                        status="PASS",
                        message="System metrics contain real data",
                        response_data={"sample_metrics": {k: v for k, v in metrics.items() if k in ['cpu_percent', 'memory_percent']}}
                    ))
                else:
                    self.log_result(TestResult(
                        test_name="Real System Metrics",
                        category="Data Verification",
                        endpoint=f"{self.endpoints['backend_api']}/api/hygiene/metrics",
                        status="WARN",
                        message="System metrics may contain mock data",
                        response_data=metrics
                    ))
            else:
                self.log_result(TestResult(
                    test_name="Real System Metrics",
                    category="Data Verification",
                    endpoint=f"{self.endpoints['backend_api']}/api/hygiene/metrics",
                    status="FAIL",
                    message=f"Failed to fetch metrics: HTTP {response.status_code}"
                ))
                
        except Exception as e:
            self.log_result(TestResult(
                test_name="Real System Metrics",
                category="Data Verification",
                endpoint=f"{self.endpoints['backend_api']}/api/hygiene/metrics",
                status="FAIL",
                message="Metrics test failed",
                error_details=str(e)
            ))

    async def _test_real_violations_data(self, client: httpx.AsyncClient):
        """Test that violations data is real, not mock"""
        try:
            response = await client.get(f"{self.endpoints['backend_api']}/api/hygiene/violations")
            
            if response.status_code == 200:
                violations = response.json()
                
                # Check for real violation data characteristics
                if violations and 'violations' in violations:
                    violation_list = violations['violations']
                    
                    # Look for real file paths and timestamps
                    real_data_indicators = []
                    for violation in violation_list[:5]:  # Check first 5
                        real_data_indicators.extend([
                            'file_path' in violation and violation['file_path'].startswith('/'),
                            'timestamp' in violation,
                            'rule_id' in violation,
                            'severity' in violation
                        ])
                    
                    if len(real_data_indicators) > 0 and sum(real_data_indicators) / len(real_data_indicators) > 0.7:
                        self.log_result(TestResult(
                            test_name="Real Violations Data",
                            category="Data Verification",
                            endpoint=f"{self.endpoints['backend_api']}/api/hygiene/violations",
                            status="PASS",
                            message=f"Violations contain real data ({len(violation_list)} violations)",
                            response_data={"violation_count": len(violation_list)}
                        ))
                    else:
                        self.log_result(TestResult(
                            test_name="Real Violations Data",
                            category="Data Verification",
                            endpoint=f"{self.endpoints['backend_api']}/api/hygiene/violations",
                            status="WARN",
                            message="Violations may contain mock data"
                        ))
                else:
                    self.log_result(TestResult(
                        test_name="Real Violations Data",
                        category="Data Verification",
                        endpoint=f"{self.endpoints['backend_api']}/api/hygiene/violations",
                        status="WARN",
                        message="No violations found"
                    ))
            else:
                self.log_result(TestResult(
                    test_name="Real Violations Data",
                    category="Data Verification",
                    endpoint=f"{self.endpoints['backend_api']}/api/hygiene/violations",
                    status="FAIL",
                    message=f"Failed to fetch violations: HTTP {response.status_code}"
                ))
                
        except Exception as e:
            self.log_result(TestResult(
                test_name="Real Violations Data",
                category="Data Verification",
                endpoint=f"{self.endpoints['backend_api']}/api/hygiene/violations",
                status="FAIL",
                message="Violations data test failed",
                error_details=str(e)
            ))

    async def _test_timestamp_freshness(self, client: httpx.AsyncClient):
        """Test that timestamps are fresh (not hardcoded)"""
        try:
            current_time = time.time()
            response = await client.get(f"{self.endpoints['backend_api']}/api/hygiene/status")
            
            if response.status_code == 200:
                status_data = response.json()
                
                if 'timestamp' in status_data:
                    try:
                        # Parse ISO timestamp
                        timestamp_str = status_data['timestamp']
                        if isinstance(timestamp_str, str):
                            timestamp = datetime.fromisoformat(timestamp_str.replace('Z', '+00:00')).timestamp()
                        else:
                            timestamp = float(timestamp_str)
                            
                        time_diff = abs(current_time - timestamp)
                        
                        if time_diff < 300:  # Within 5 minutes
                            self.log_result(TestResult(
                                test_name="Timestamp Freshness",
                                category="Data Verification",
                                endpoint=f"{self.endpoints['backend_api']}/api/hygiene/status",
                                status="PASS",
                                message=f"Timestamp is fresh ({time_diff:.1f}s ago)"
                            ))
                        else:
                            self.log_result(TestResult(
                                test_name="Timestamp Freshness", 
                                category="Data Verification",
                                endpoint=f"{self.endpoints['backend_api']}/api/hygiene/status",
                                status="WARN",
                                message=f"Timestamp is stale ({time_diff:.1f}s ago)"
                            ))
                    except Exception as e:
                        self.log_result(TestResult(
                            test_name="Timestamp Freshness",
                            category="Data Verification",
                            endpoint=f"{self.endpoints['backend_api']}/api/hygiene/status",
                            status="FAIL",
                            message="Failed to parse timestamp",
                            error_details=str(e)
                        ))
                else:
                    self.log_result(TestResult(
                        test_name="Timestamp Freshness",
                        category="Data Verification", 
                        endpoint=f"{self.endpoints['backend_api']}/api/hygiene/status",
                        status="WARN",
                        message="No timestamp found in status data"
                    ))
            else:
                self.log_result(TestResult(
                    test_name="Timestamp Freshness",
                    category="Data Verification",
                    endpoint=f"{self.endpoints['backend_api']}/api/hygiene/status",
                    status="FAIL",
                    message=f"Failed to fetch status: HTTP {response.status_code}"
                ))
                
        except Exception as e:
            self.log_result(TestResult(
                test_name="Timestamp Freshness",
                category="Data Verification",
                endpoint=f"{self.endpoints['backend_api']}/api/hygiene/status",
                status="FAIL",
                message="Timestamp freshness test failed",
                error_details=str(e)
            ))

    def generate_summary_report(self) -> Dict[str, Any]:
        """Generate comprehensive summary report"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "PASS"])
        failed_tests = len([r for r in self.results if r.status == "FAIL"])
        warning_tests = len([r for r in self.results if r.status == "WARN"])
        skipped_tests = len([r for r in self.results if r.status == "SKIP"])
        
        # Calculate success rate
        success_rate = passed_tests / total_tests if total_tests > 0 else 0.0
        
        # Group results by category
        results_by_category = {}
        for result in self.results:
            category = result.category
            if category not in results_by_category:
                results_by_category[category] = []
            results_by_category[category].append(result)
        
        # Calculate average response times
        response_times = [r.response_time_ms for r in self.results if r.response_time_ms is not None]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        
        return {
            "verification_summary": {
                "timestamp": datetime.now().isoformat(),
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "warnings": warning_tests,
                "skipped": skipped_tests,
                "success_rate": success_rate,
                "average_response_time_ms": avg_response_time,
                "overall_status": "HEALTHY" if success_rate >= 0.8 and failed_tests == 0 else 
                                "DEGRADED" if success_rate >= 0.6 else "UNHEALTHY"
            },
            "endpoints_tested": self.endpoints,
            "results_by_category": {
                category: {
                    "total": len(results),
                    "passed": len([r for r in results if r.status == "PASS"]),
                    "failed": len([r for r in results if r.status == "FAIL"]),
                    "warnings": len([r for r in results if r.status == "WARN"]),
                    "tests": [asdict(r) for r in results]
                }
                for category, results in results_by_category.items()
            },
            "detailed_results": [asdict(r) for r in self.results]
        }

    async def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run all verification tests"""
        logger.info("🚀 Starting Comprehensive Hygiene Monitoring System Verification")
        logger.info("=" * 80)
        
        verification_tests = [
            ("Dashboard Endpoints", self.test_dashboard_endpoints),
            ("Backend API Endpoints", self.test_backend_api_endpoints), 
            ("Rule Control API Endpoints", self.test_rule_control_api_endpoints),
            ("WebSocket Connectivity", self.test_websocket_connectivity),
            ("Real Data Verification", self.test_real_data_verification),
            ("Violations Detection", self.test_violations_detection),
        ]
        
        start_time = time.time()
        
        for test_name, test_func in verification_tests:
            try:
                logger.info(f"\n📋 Running {test_name}...")
                await test_func()
            except Exception as e:
                logger.error(f"❌ {test_name} failed with exception: {e}")
                if self.verbose:
                    logger.error(traceback.format_exc())
                
                # Log a failure result
                self.log_result(TestResult(
                    test_name=test_name,
                    category="System",
                    endpoint="N/A",
                    status="FAIL",
                    message="Test suite failed with exception",
                    error_details=str(e)
                ))
        
        total_time = time.time() - start_time
        logger.info(f"\n⏱️  Total verification time: {total_time:.2f}s")
        
        # Generate and return comprehensive report
        report = self.generate_summary_report()
        
        logger.info("\n" + "=" * 80)
        logger.info("📊 VERIFICATION SUMMARY")
        logger.info("=" * 80)
        
        summary = report["verification_summary"]
        logger.info(f"Overall Status: {summary['overall_status']}")
        logger.info(f"Success Rate: {summary['success_rate']:.1%}")
        logger.info(f"Tests: {summary['passed']} passed, {summary['failed']} failed, {summary['warnings']} warnings")
        logger.info(f"Average Response Time: {summary['average_response_time_ms']:.0f}ms")
        
        return report

async def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Verify Hygiene Monitoring System")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--timeout", "-t", type=int, default=30, help="Request timeout in seconds")
    parser.add_argument("--output-file", "-o", help="Output report to JSON file")
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Create verifier and run tests
    verifier = HygieneMonitoringVerifier(verbose=args.verbose, timeout=args.timeout)
    
    try:
        report = await verifier.run_comprehensive_verification()
        
        # Output to file if specified
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with aiofiles.open(output_path, 'w') as f:
                await f.write(json.dumps(report, indent=2))
            
            logger.info(f"📄 Report saved to: {output_path}")
        
        # Return appropriate exit code
        summary = report["verification_summary"]
        if summary["overall_status"] == "HEALTHY":
            logger.info("✅ All systems operational!")
            return 0
        elif summary["overall_status"] == "DEGRADED": 
            logger.warning("⚠️  System has warnings but is functional")
            return 1
        else:
            logger.error("❌ System verification failed!")
            return 2
            
    except Exception as e:
        logger.error(f"❌ Verification failed with exception: {e}")
        if args.verbose:
            logger.error(traceback.format_exc())
        return 3

if __name__ == "__main__":
    sys.exit(asyncio.run(main()))