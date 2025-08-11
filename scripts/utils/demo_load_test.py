#!/usr/bin/env python3
"""
DEMO LOAD TEST - HARDWARE OPTIMIZER ULTRA TESTING
==================================================

Demonstration of the ultra-comprehensive load testing capabilities.
This script runs a subset of the full testing suite to showcase functionality
without overwhelming the system.

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

# Add tests directory to path
sys.path.insert(0, '/opt/sutazaiapp/tests')

from hardware_optimizer_ultra_test_suite import UltraHardwareOptimizerTester

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/sutazaiapp/tests/demo_load_test.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('DemoLoadTest')

class DemoLoadTester:
    """Demonstration load tester with controlled scope"""
    
    def __init__(self):
        self.tester = UltraHardwareOptimizerTester()
        # Override load levels for demo (smaller scope)
        self.tester.LOAD_LEVELS = [1, 5, 10]  # Reduced from [1, 5, 10, 25, 50, 100]
        # Select key endpoints for demo
        self.demo_endpoints = [
            self.tester.ENDPOINTS[0],  # /health
            self.tester.ENDPOINTS[1],  # /status  
            self.tester.ENDPOINTS[7],  # /analyze/storage
            self.tester.ENDPOINTS[10], # /analyze/storage/report
            self.tester.ENDPOINTS[2],  # /optimize/memory
        ]
        
    async def run_demo_load_test(self):
        """Run demonstration load test"""
        logger.info("🚀 Starting Hardware Optimizer DEMO Load Test")
        logger.info("=" * 70)
        logger.info(f"Endpoints: {len(self.demo_endpoints)}")
        logger.info(f"Load Levels: {self.tester.LOAD_LEVELS}")
        logger.info(f"Total Scenarios: {len(self.demo_endpoints) * len(self.tester.LOAD_LEVELS)}")
        logger.info("=" * 70)
        
        start_time = time.time()
        all_results = []
        
        # Test each endpoint at each load level
        for endpoint in self.demo_endpoints:
            for concurrent_users in self.tester.LOAD_LEVELS:
                logger.info(f"Testing {endpoint.method} {endpoint.path} with {concurrent_users} users...")
                
                try:
                    # Small delay between tests
                    if all_results:
                        await asyncio.sleep(2)
                    
                    # Run load test for this endpoint/load combination
                    metrics = await self.tester._run_load_test_for_endpoint(endpoint, concurrent_users)
                    all_results.append(metrics)
                    
                    # Log immediate results
                    status_emoji = "✅" if metrics.sla_compliance else "⚠️"
                    logger.info(f"{status_emoji} Result: {metrics.success_rate:.1f}% success, "
                              f"P95: {metrics.p95_response_time_ms:.1f}ms, "
                              f"Throughput: {metrics.throughput_rps:.1f} RPS")
                    
                except Exception as e:
                    logger.error(f"❌ Test failed: {e}")
                    # Create failure metrics
                    empty_metrics = self.tester._create_empty_metrics(endpoint, concurrent_users)
                    all_results.append(empty_metrics)
        
        total_time = time.time() - start_time
        
        # Analyze results
        analysis = self._analyze_demo_results(all_results)
        
        # Print summary
        self._print_demo_summary(analysis, total_time)
        
        # Save results
        report = {
            "demo_metadata": {
                "timestamp": time.time(),
                "duration_seconds": total_time,
                "endpoints_tested": len(self.demo_endpoints),
                "load_levels": self.tester.LOAD_LEVELS,
                "total_scenarios": len(all_results)
            },
            "results_summary": analysis,
            "detailed_results": [result.__dict__ for result in all_results]
        }
        
        report_file = f"/opt/sutazaiapp/tests/demo_load_test_report_{int(time.time())}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"📊 Demo report saved: {report_file}")
        
        return report
    
    def _analyze_demo_results(self, results):
        """Analyze demo test results"""
        total_tests = len(results)
        sla_compliant = len([r for r in results if r.sla_compliance])
        sla_compliance_rate = (sla_compliant / total_tests) * 100 if total_tests > 0 else 0
        
        # Performance metrics
        avg_response_time = sum(r.avg_response_time_ms for r in results) / len(results) if results else 0
        avg_success_rate = sum(r.success_rate for r in results) / len(results) if results else 0
        avg_throughput = sum(r.throughput_rps for r in results) / len(results) if results else 0
        peak_memory = max(r.memory_peak_mb for r in results) if results else 0
        
        # Memory leaks
        memory_leaks = len([r for r in results if r.memory_leak_detected])
        
        # Endpoint performance analysis
        endpoint_performance = {}
        for result in results:
            endpoint = result.endpoint
            if endpoint not in endpoint_performance:
                endpoint_performance[endpoint] = []
            endpoint_performance[endpoint].append(result)
        
        # Find best and worst performing endpoints
        endpoint_scores = {}
        for endpoint, tests in endpoint_performance.items():
            # Calculate average performance score
            avg_sla_compliance = sum(1 for t in tests if t.sla_compliance) / len(tests)
            avg_response = sum(t.avg_response_time_ms for t in tests) / len(tests)
            
            # Combined score (higher is better)
            score = avg_sla_compliance * 100 - (avg_response / 10)  # Penalize high response times
            endpoint_scores[endpoint] = score
        
        best_endpoint = max(endpoint_scores.items(), key=lambda x: x[1]) if endpoint_scores else None
        worst_endpoint = min(endpoint_scores.items(), key=lambda x: x[1]) if endpoint_scores else None
        
        return {
            "sla_compliance": {
                "total_tests": total_tests,
                "sla_compliant": sla_compliant,
                "compliance_rate": sla_compliance_rate,
                "passed": sla_compliance_rate >= 90
            },
            "performance_metrics": {
                "avg_response_time_ms": avg_response_time,
                "avg_success_rate": avg_success_rate,
                "avg_throughput_rps": avg_throughput,
                "peak_memory_mb": peak_memory
            },
            "quality_indicators": {
                "memory_leaks_detected": memory_leaks,
                "zero_memory_leaks": memory_leaks == 0,
                "memory_under_limit": peak_memory <= self.tester.MAX_MEMORY_USAGE_MB
            },
            "endpoint_analysis": {
                "best_performing": best_endpoint,
                "worst_performing": worst_endpoint,
                "endpoint_scores": endpoint_scores
            }
        }
    
    def _print_demo_summary(self, analysis, total_time):
        """Print demo test summary"""
        print("\n" + "=" * 70)
        print("🎯 HARDWARE OPTIMIZER DEMO LOAD TEST RESULTS")
        print("=" * 70)
        
        # SLA Compliance
        sla = analysis["sla_compliance"]
        sla_emoji = "✅" if sla["passed"] else "❌"
        print(f"{sla_emoji} SLA Compliance: {sla['sla_compliant']}/{sla['total_tests']} ({sla['compliance_rate']:.1f}%)")
        
        # Performance
        perf = analysis["performance_metrics"]
        print(f"📈 Avg Response Time: {perf['avg_response_time_ms']:.1f}ms")
        print(f"📈 Avg Success Rate: {perf['avg_success_rate']:.1f}%")
        print(f"📈 Avg Throughput: {perf['avg_throughput_rps']:.1f} RPS")
        print(f"💾 Peak Memory: {perf['peak_memory_mb']:.1f}MB")
        
        # Quality Indicators
        quality = analysis["quality_indicators"]
        memory_emoji = "✅" if quality["zero_memory_leaks"] else "⚠️"
        limit_emoji = "✅" if quality["memory_under_limit"] else "❌"
        print(f"{memory_emoji} Memory Leaks: {quality['memory_leaks_detected']} detected")
        print(f"{limit_emoji} Memory Limit: {perf['peak_memory_mb']:.1f}MB (limit: {self.tester.MAX_MEMORY_USAGE_MB}MB)")
        
        # Best/Worst Endpoints
        endpoint_analysis = analysis["endpoint_analysis"]
        if endpoint_analysis["best_performing"]:
            best_endpoint, best_score = endpoint_analysis["best_performing"]
            print(f"🏆 Best Endpoint: {best_endpoint} (score: {best_score:.1f})")
        
        if endpoint_analysis["worst_performing"]:
            worst_endpoint, worst_score = endpoint_analysis["worst_performing"]
            print(f"⚠️  Needs Attention: {worst_endpoint} (score: {worst_score:.1f})")
        
        print(f"⏱️  Total Test Time: {total_time:.1f} seconds")
        
        # Overall Assessment
        overall_passed = (
            sla["passed"] and 
            quality["zero_memory_leaks"] and 
            quality["memory_under_limit"]
        )
        
        overall_emoji = "🎉" if overall_passed else "⚠️"
        overall_status = "EXCELLENT" if overall_passed else "NEEDS IMPROVEMENT"
        print(f"{overall_emoji} Overall Assessment: {overall_status}")
        
        print("=" * 70)

async def main():
    """Main demo execution"""
    try:
        demo_tester = DemoLoadTester()
        report = await demo_tester.run_demo_load_test()
        
        # Determine exit code based on results
        overall_passed = (
            report["results_summary"]["sla_compliance"]["passed"] and
            report["results_summary"]["quality_indicators"]["zero_memory_leaks"] and
            report["results_summary"]["quality_indicators"]["memory_under_limit"]
        )
        
        if overall_passed:
            logger.info("🎉 Demo load test completed successfully!")
            sys.exit(0)
        else:
            logger.warning("⚠️  Demo completed with some issues")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Demo load test failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(2)

if __name__ == "__main__":
    asyncio.run(main())