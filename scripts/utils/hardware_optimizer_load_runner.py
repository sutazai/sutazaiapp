#!/usr/bin/env python3
"""
ULTRA-CRITICAL LOAD TESTING RUNNER FOR HARDWARE RESOURCE OPTIMIZER
==================================================================

Dedicated load testing runner that focuses specifically on concurrent load testing
with automated SLA validation and performance metrics collection.

This runner executes the full load testing matrix:
- 16 endpoints × 6 load levels = 96 total load test scenarios
- Real-time performance monitoring during tests
- Automated SLA compliance validation
- Memory leak detection
- Performance regression analysis

Author: Ultra-Critical Automated Testing Specialist  
Version: 1.0.0
"""

import asyncio
import sys
import os
import json
import time
import logging
from pathlib import Path

# Add the tests directory to Python path for imports
sys.path.insert(0, '/opt/sutazaiapp/tests')

from hardware_optimizer_ultra_test_suite import UltraHardwareOptimizerTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/tests/load_test_execution.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('LoadTestRunner')

class LoadTestRunner:
    """Dedicated runner for comprehensive load testing"""
    
    def __init__(self):
        self.tester = UltraHardwareOptimizerTester()
        self.results_dir = "/opt/sutazaiapp/tests/load_test_results"
        self._create_results_directory()
    
    def _create_results_directory(self):
        """Create results directory structure"""
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(f"{self.results_dir}/charts", exist_ok=True)
        os.makedirs(f"{self.results_dir}/raw_data", exist_ok=True)
        logger.info(f"Results directory created: {self.results_dir}")
    
    async def run_load_testing_matrix(self):
        """Execute the complete load testing matrix"""
        logger.info("🚀 Starting Ultra-Critical Load Testing Matrix")
        logger.info("=" * 80)
        logger.info(f"Endpoints to test: {len(self.tester.ENDPOINTS)}")
        logger.info(f"Load levels: {self.tester.LOAD_LEVELS}")
        logger.info(f"Total test scenarios: {len(self.tester.ENDPOINTS) * len(self.tester.LOAD_LEVELS)}")
        logger.info(f"SLA Requirements: <{self.tester.SLA_RESPONSE_TIME_MS}ms (95%), >{self.tester.SLA_SUCCESS_RATE}% success")
        logger.info("=" * 80)
        
        start_time = time.time()
        
        # Execute comprehensive load tests
        load_results = await self.tester.run_comprehensive_load_tests()
        
        # Generate performance analysis
        analysis = self._analyze_load_test_results(load_results)
        
        # Generate charts
        self.tester.generate_performance_charts()
        
        total_time = time.time() - start_time
        
        # Create final report
        report = {
            "execution_metadata": {
                "timestamp": time.time(),
                "duration_seconds": total_time,
                "total_scenarios": len(load_results),
                "endpoints_tested": len(self.tester.ENDPOINTS),
                "load_levels": self.tester.LOAD_LEVELS
            },
            "sla_compliance": analysis["sla_compliance"],
            "performance_summary": analysis["performance_summary"],
            "detailed_results": [result.__dict__ for result in load_results],
            "critical_findings": analysis["critical_findings"],
            "recommendations": analysis["recommendations"]
        }
        
        # Save comprehensive report
        report_file = f"{self.results_dir}/load_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Print summary
        self._print_load_test_summary(analysis, total_time)
        
        logger.info(f"📊 Complete load testing report saved: {report_file}")
        
        return report
    
    def _analyze_load_test_results(self, results):
        """Analyze load test results for comprehensive insights"""
        total_tests = len(results)
        sla_compliant = len([r for r in results if r.sla_compliance])
        sla_compliance_rate = (sla_compliant / total_tests) * 100 if total_tests > 0 else 0
        
        # Performance analysis by endpoint
        endpoint_analysis = {}
        for result in results:
            endpoint = result.endpoint
            if endpoint not in endpoint_analysis:
                endpoint_analysis[endpoint] = {
                    "tests": [],
                    "max_successful_load": 0,
                    "performance_degradation_point": None
                }
            
            endpoint_analysis[endpoint]["tests"].append(result)
            
            # Find maximum load that maintains SLA compliance
            if result.sla_compliance:
                endpoint_analysis[endpoint]["max_successful_load"] = max(
                    endpoint_analysis[endpoint]["max_successful_load"],
                    result.concurrent_users
                )
        
        # Identify performance degradation points
        for endpoint, data in endpoint_analysis.items():
            tests = sorted(data["tests"], key=lambda x: x.concurrent_users)
            
            for i, test in enumerate(tests):
                if not test.sla_compliance and i > 0:
                    # Performance degraded at this load level
                    data["performance_degradation_point"] = test.concurrent_users
                    break
        
        # Memory leak analysis
        memory_leaks = [r for r in results if r.memory_leak_detected]
        
        # Critical findings
        critical_findings = []
        
        if sla_compliance_rate < 90:
            critical_findings.append({
                "severity": "HIGH",
                "type": "SLA_COMPLIANCE", 
                "description": f"SLA compliance rate {sla_compliance_rate:.1f}% below acceptable threshold (90%)"
            })
        
        if memory_leaks:
            critical_findings.append({
                "severity": "HIGH",
                "type": "MEMORY_LEAK",
                "description": f"Memory leaks detected in {len(memory_leaks)} test scenarios"
            })
        
        # Performance bottlenecks
        low_performance_endpoints = []
        for endpoint, data in endpoint_analysis.items():
            if data["max_successful_load"] < 25:  # Can't handle 25 concurrent users
                low_performance_endpoints.append(endpoint)
        
        if low_performance_endpoints:
            critical_findings.append({
                "severity": "MEDIUM",
                "type": "PERFORMANCE_BOTTLENECK",
                "description": f"Endpoints with low performance: {low_performance_endpoints}"
            })
        
        # Generate recommendations
        recommendations = self._generate_load_test_recommendations(
            critical_findings, endpoint_analysis, results
        )
        
        return {
            "sla_compliance": {
                "total_tests": total_tests,
                "sla_compliant": sla_compliant,
                "compliance_rate": sla_compliance_rate,
                "passed": sla_compliance_rate >= 90
            },
            "performance_summary": {
                "avg_response_time_ms": sum(r.avg_response_time_ms for r in results) / len(results) if results else 0,
                "avg_success_rate": sum(r.success_rate for r in results) / len(results) if results else 0,
                "avg_throughput_rps": sum(r.throughput_rps for r in results) / len(results) if results else 0,
                "peak_memory_mb": max(r.memory_peak_mb for r in results) if results else 0,
                "memory_leaks_detected": len(memory_leaks)
            },
            "endpoint_analysis": endpoint_analysis,
            "critical_findings": critical_findings,
            "recommendations": recommendations
        }
    
    def _generate_load_test_recommendations(self, critical_findings, endpoint_analysis, results):
        """Generate specific recommendations based on load test analysis"""
        recommendations = []
        
        # SLA compliance recommendations
        if any(f["type"] == "SLA_COMPLIANCE" for f in critical_findings):
            recommendations.extend([
                "Implement connection pooling and keep-alive connections",
                "Add request queuing with priority handling",
                "Optimize database queries and add proper indexing",
                "Consider implementing caching for frequently accessed data",
                "Review and optimize file system operations"
            ])
        
        # Memory leak recommendations
        if any(f["type"] == "MEMORY_LEAK" for f in critical_findings):
            recommendations.extend([
                "Implement proper resource cleanup in try-finally blocks",
                "Review object lifecycle management and garbage collection",
                "Add memory profiling to identify leak sources",
                "Implement memory monitoring and alerting",
                "Consider using context managers for resource management"
            ])
        
        # Performance bottleneck recommendations
        if any(f["type"] == "PERFORMANCE_BOTTLENECK" for f in critical_findings):
            recommendations.extend([
                "Implement asynchronous processing for long-running operations",
                "Add horizontal scaling capabilities",
                "Optimize file system operations with batch processing",
                "Consider implementing request rate limiting",
                "Add circuit breakers for external dependencies"
            ])
        
        # Endpoint-specific recommendations
        for endpoint, data in endpoint_analysis.items():
            if data["max_successful_load"] < 10:
                recommendations.append(
                    f"Critical: {endpoint} needs immediate optimization - only handles {data['max_successful_load']} concurrent users"
                )
            elif data["performance_degradation_point"] and data["performance_degradation_point"] < 25:
                recommendations.append(
                    f"Warning: {endpoint} performance degrades at {data['performance_degradation_point']} concurrent users"
                )
        
        # General operational recommendations
        recommendations.extend([
            "Implement comprehensive APM (Application Performance Monitoring)",
            "Set up automated load testing in CI/CD pipeline",
            "Create performance budgets and SLA monitoring",
            "Establish baseline performance metrics for regression testing"
        ])
        
        return recommendations
    
    def _print_load_test_summary(self, analysis, total_time):
        """Print comprehensive load testing summary"""
        logger.info("\n" + "=" * 80)
        logger.error("🎯 ULTRA-CRITICAL LOAD TESTING SUMMARY")
        logger.info("=" * 80)
        
        # Overall results
        sla_data = analysis["sla_compliance"]
        logger.info(f"📊 SLA Compliance: {sla_data['sla_compliant']}/{sla_data['total_tests']} ({sla_data['compliance_rate']:.1f}%)")
        logger.info(f"⏱️  Total Execution Time: {total_time:.1f} seconds")
        logger.info(f"{'✅ PASS' if sla_data['passed'] else '❌ FAIL'} - Overall Load Testing Assessment")
        
        # Performance metrics
        perf = analysis["performance_summary"]
        logger.info(f"\n📈 Performance Metrics:")
        logger.info(f"  • Average Response Time: {perf['avg_response_time_ms']:.1f}ms")
        logger.info(f"  • Average Success Rate: {perf['avg_success_rate']:.1f}%")
        logger.info(f"  • Average Throughput: {perf['avg_throughput_rps']:.1f} RPS")
        logger.info(f"  • Peak Memory Usage: {perf['peak_memory_mb']:.1f}MB")
        
        # Critical findings
        if analysis["critical_findings"]:
            logger.error(f"\n🚨 Critical Findings ({len(analysis['critical_findings'])}):")
            for finding in analysis["critical_findings"]:
                severity_emoji = "🔴" if finding["severity"] == "HIGH" else "🟡"
                logger.info(f"  {severity_emoji} [{finding['severity']}] {finding['type']}: {finding['description']}")
        else:
            logger.error("\n✅ No Critical Issues Detected")
        
        # Top performing endpoints
        endpoint_data = analysis["endpoint_analysis"]
        top_performers = sorted(
            endpoint_data.items(),
            key=lambda x: x[1]["max_successful_load"],
            reverse=True
        )[:5]
        
        logger.info(f"\n🏆 Top Performing Endpoints:")
        for endpoint, data in top_performers:
            logger.info(f"  • {endpoint}: {data['max_successful_load']} max concurrent users")
        
        # Recommendations
        if analysis["recommendations"]:
            logger.info(f"\n💡 Key Recommendations ({len(analysis['recommendations'])}:")
            for i, rec in enumerate(analysis["recommendations"][:5], 1):  # Show top 5
                logger.info(f"  {i}. {rec}")
            if len(analysis["recommendations"]) > 5:
                logger.info(f"  ... and {len(analysis['recommendations']) - 5} more recommendations in the full report")
        
        logger.info("=" * 80)

async def main():
    """Main execution function"""
    try:
        runner = LoadTestRunner()
        report = await runner.run_load_testing_matrix()
        
        # Exit with appropriate code
        sla_passed = report["sla_compliance"]["passed"]
        critical_issues = len(report["critical_findings"])
        
        if sla_passed and critical_issues == 0:
            logger.info("✅ All load tests passed with no critical issues")
            sys.exit(0)
        elif sla_passed:
            logger.warning(f"⚠️  Load tests passed but {critical_issues} critical issues found")
            sys.exit(1)
        else:
            logger.error("❌ Load tests failed - SLA compliance below threshold")
            sys.exit(2)
            
    except Exception as e:
        logger.error(f"Load testing execution failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(3)

if __name__ == "__main__":
    asyncio.run(main())