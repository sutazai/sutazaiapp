#!/usr/bin/env python3
"""
Test Execution Orchestrator for Hardware Resource Optimizer
===========================================================

Main orchestrator for running comprehensive testing suites:
- Coordinates all test types (E2E, performance, manual, stress)
- Manages test dependencies and execution order
- Provides unified reporting and analysis
- Handles test environment setup and cleanup
- Supports CI/CD integration

Author: QA Team Lead
Version: 1.0.0
"""

import os
import sys
import json
import time
import asyncio
import argparse
import subprocess
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from pathlib import Path
import threading
import requests

# Add current directory to path for imports
# Path handled by pytest configuration))

from comprehensive_e2e_test_framework import E2ETestFramework
from performance_stress_tests import PerformanceBenchmarkSuite
from manual_test_procedures import ManualTestProcedures
from automated_continuous_tests import ContinuousTestingOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_orchestrator.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TestOrchestrator')

class TestExecutionOrchestrator:
    """Orchestrates comprehensive testing execution"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self.base_url = self.config.get('agent_url', 'http://localhost:8116')
        self.results = {}
        self.start_time = None
        self.agent_available = False
        
        # Initialize test components
        self.e2e_framework = None
        self.performance_suite = None
        self.manual_tester = None
        self.continuous_orchestrator = None
        
        logger.info("Test Execution Orchestrator initialized")
    
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration for test orchestrator"""
        return {
            'agent_url': 'http://localhost:8116',
            'timeout': 30,
            'test_suites': {
                'e2e': True,
                'performance': True,
                'stress': True,
                'manual': False,  # Requires user interaction
                'continuous': False  # For CI/CD integration
            },
            'test_environment': {
                'setup_required': True,
                'cleanup_after': True,
                'preserve_logs': True
            },
            'reporting': {
                'detailed_json': True,
                'summary_report': True,
                'performance_charts': False,
                'email_report': False
            },
            'thresholds': {
                'min_success_rate': 95.0,
                'max_avg_response_time': 2.0,
                'max_p95_response_time': 5.0
            }
        }
    
    def check_agent_availability(self) -> bool:
        """Check if the agent is running and responsive"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    logger.info(f"Agent is healthy at {self.base_url}")
                    self.agent_available = True
                    return True
            
            logger.error(f"Agent unhealthy at {self.base_url}")
            return False
            
        except Exception as e:
            logger.error(f"Agent not available at {self.base_url}: {e}")
            return False
    
    def setup_test_environment(self) -> bool:
        """Setup the test environment"""
        logger.info("Setting up test environment...")
        
        try:
            # Create test directories
            test_dirs = ['logs', 'reports', 'temp']
            for directory in test_dirs:
                os.makedirs(directory, exist_ok=True)
            
            # Initialize test components
            self.e2e_framework = E2ETestFramework(self.base_url, self.config['timeout'])
            self.performance_suite = PerformanceBenchmarkSuite(self.base_url)
            self.manual_tester = ManualTestProcedures(self.base_url)
            self.continuous_orchestrator = ContinuousTestingOrchestrator()
            
            logger.info("Test environment setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Test environment setup failed: {e}")
            return False
    
    def cleanup_test_environment(self):
        """Cleanup test environment"""
        logger.info("Cleaning up test environment...")
        
        try:
            if self.e2e_framework:
                self.e2e_framework.cleanup_test_environment()
            
            # Cleanup temporary files if configured
            if self.config.get('test_environment', {}).get('cleanup_after', True):
                import shutil
                temp_dirs = ['temp']
                for directory in temp_dirs:
                    if os.path.exists(directory):
                        shutil.rmtree(directory)
            
            logger.info("Test environment cleanup complete")
            
        except Exception as e:
            logger.error(f"Test environment cleanup failed: {e}")
    
    def run_e2e_tests(self) -> Dict[str, Any]:
        """Run comprehensive E2E tests"""
        logger.info("Executing E2E test suite...")
        
        try:
            if not self.e2e_framework:
                self.e2e_framework = E2ETestFramework(self.base_url, self.config['timeout'])
            
            results = self.e2e_framework.run_comprehensive_test_suite()
            
            # Validate results against thresholds
            success_rate = results.get('test_summary', {}).get('success_rate', 0)
            min_success_rate = self.config['thresholds']['min_success_rate']
            
            results['threshold_validation'] = {
                'success_rate_pass': success_rate >= min_success_rate,
                'success_rate': success_rate,
                'threshold': min_success_rate
            }
            
            logger.info(f"E2E tests completed - Success rate: {success_rate:.1f}%")
            return results
            
        except Exception as e:
            logger.error(f"E2E tests failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'test_type': 'e2e'
            }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance and stress tests"""
        logger.info("Executing performance test suite...")
        
        try:
            if not self.performance_suite:
                self.performance_suite = PerformanceBenchmarkSuite(self.base_url)
            
            results = self.performance_suite.run_comprehensive_performance_suite()
            
            # Validate performance against thresholds
            summary = results.get('summary', {})
            benchmark_summary = summary.get('benchmark_summary', {})
            
            avg_response_time = benchmark_summary.get('avg_response_time', 0)
            max_avg_response_time = self.config['thresholds']['max_avg_response_time']
            
            results['threshold_validation'] = {
                'avg_response_time_pass': avg_response_time <= max_avg_response_time,
                'avg_response_time': avg_response_time,
                'threshold': max_avg_response_time
            }
            
            logger.info(f"Performance tests completed - Avg response time: {avg_response_time:.3f}s")
            return results
            
        except Exception as e:
            logger.error(f"Performance tests failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'test_type': 'performance'
            }
    
    def run_manual_tests(self) -> Dict[str, Any]:
        """Run manual test procedures"""
        logger.info("Executing manual test procedures...")
        
        try:
            if not self.manual_tester:
                self.manual_tester = ManualTestProcedures(self.base_url)
            
            logger.info("\n" + "="*80)
            logger.info("STARTING MANUAL TEST PROCEDURES")
            logger.info("="*80)
            logger.info("The system will now guide you through manual testing procedures.")
            logger.info("Please follow the instructions carefully.")
            logger.info("="*80)
            
            results = self.manual_tester.run_manual_test_suite()
            
            logger.info(f"Manual tests completed - Status: {results.get('status')}")
            return results
            
        except Exception as e:
            logger.error(f"Manual tests failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'test_type': 'manual'
            }
    
    def run_continuous_tests(self) -> Dict[str, Any]:
        """Run continuous testing validation"""
        logger.info("Executing continuous testing validation...")
        
        try:
            if not self.continuous_orchestrator:
                self.continuous_orchestrator = ContinuousTestingOrchestrator()
            
            # Run a subset of continuous tests for validation
            health_result = self.continuous_orchestrator.run_health_check()
            smoke_result = self.continuous_orchestrator.run_smoke_tests()
            
            results = {
                'status': 'completed',
                'test_type': 'continuous_validation',
                'health_check': health_result,
                'smoke_tests': smoke_result,
                'timestamp': datetime.now().isoformat()
            }
            
            logger.info("Continuous testing validation completed")
            return results
            
        except Exception as e:
            logger.error(f"Continuous tests failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'test_type': 'continuous'
            }
    
    def generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        logger.info("Generating comprehensive test report...")
        
        # Calculate overall statistics
        total_duration = time.time() - self.start_time if self.start_time else 0
        
        executed_suites = list(self.results.keys())
        successful_suites = []
        failed_suites = []
        
        overall_success_rate = 0
        total_tests = 0
        successful_tests = 0
        
        for suite_name, suite_results in self.results.items():
            if suite_results.get('status') in ['success', 'completed']:
                successful_suites.append(suite_name)
                
                # Extract test counts
                if 'test_summary' in suite_results:
                    summary = suite_results['test_summary']
                    total_tests += summary.get('total_tests', 0)
                    successful_tests += summary.get('successful_tests', 0)
                elif suite_name == 'manual':
                    total_tests += suite_results.get('total_tests', 0)
                    successful_tests += suite_results.get('passed_tests', 0)
            else:
                failed_suites.append(suite_name)
        
        if total_tests > 0:
            overall_success_rate = (successful_tests / total_tests) * 100
        
        # Performance summary
        performance_summary = {}
        if 'performance' in self.results:
            perf_results = self.results['performance']
            if 'summary' in perf_results:
                performance_summary = perf_results['summary']
        
        # Threshold validation summary
        threshold_failures = []
        for suite_name, suite_results in self.results.items():
            threshold_validation = suite_results.get('threshold_validation', {})
            for check, passed in threshold_validation.items():
                if check.endswith('_pass') and not passed:
                    threshold_failures.append({
                        'suite': suite_name,
                        'check': check,
                        'value': threshold_validation.get(check.replace('_pass', ''), 'unknown'),
                        'threshold': threshold_validation.get('threshold', 'unknown')
                    })
        
        report = {
            'test_execution_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_duration_seconds': total_duration,
                'agent_url': self.base_url,
                'executed_suites': executed_suites,
                'successful_suites': successful_suites,
                'failed_suites': failed_suites,
                'suite_success_rate': len(successful_suites) / len(executed_suites) * 100 if executed_suites else 0
            },
            'test_statistics': {
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': total_tests - successful_tests,
                'overall_success_rate': overall_success_rate
            },
            'threshold_validation': {
                'total_checks': len(threshold_failures) + sum(
                    len([k for k in sr.get('threshold_validation', {}).keys() if k.endswith('_pass')])
                    for sr in self.results.values()
                ),
                'failed_checks': len(threshold_failures),
                'failures': threshold_failures
            },
            'performance_summary': performance_summary,
            'detailed_results': self.results,
            'agent_status': {
                'available': self.agent_available,
                'last_checked': datetime.now().isoformat()
            }
        }
        
        return report
    
    def save_report(self, report: Dict[str, Any], filename: str = None) -> str:
        """Save comprehensive report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"comprehensive_test_report_{timestamp}.json"
        
        filepath = os.path.join('reports', filename)
        os.makedirs('reports', exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Comprehensive report saved to: {filepath}")
        return filepath
    
    def print_summary(self, report: Dict[str, Any]):
        """Print executive summary of test results"""
        logger.info("\n" + "="*80)
        logger.info("COMPREHENSIVE TEST EXECUTION SUMMARY")
        logger.info("="*80)
        
        summary = report['test_execution_summary']
        stats = report['test_statistics']
        thresholds = report['threshold_validation']
        
        logger.info(f"Agent URL: {summary['agent_url']}")
        logger.info(f"Execution Time: {summary['total_duration_seconds']:.1f} seconds")
        logger.info(f"Timestamp: {summary['timestamp']}")
        
        logger.info(f"\nTest Suites Executed: {len(summary['executed_suites'])}")
        for suite in summary['executed_suites']:
            status = "✅ PASSED" if suite in summary['successful_suites'] else "❌ FAILED"
            logger.info(f"  - {suite.upper()}: {status}")
        
        logger.info(f"\nOverall Statistics:")
        logger.info(f"  - Total Tests: {stats['total_tests']}")
        logger.info(f"  - Successful: {stats['successful_tests']}")
        logger.error(f"  - Failed: {stats['failed_tests']}")
        logger.info(f"  - Success Rate: {stats['overall_success_rate']:.1f}%")
        
        logger.info(f"\nThreshold Validation:")
        logger.info(f"  - Total Checks: {thresholds['total_checks']}")
        logger.error(f"  - Failed Checks: {thresholds['failed_checks']}")
        
        if thresholds['failures']:
            logger.info(f"  - Failures:")
            for failure in thresholds['failures']:
                logger.info(f"    * {failure['suite']} - {failure['check']}: {failure['value']} (threshold: {failure['threshold']})")
        
        # Overall assessment
        overall_pass = (
            len(summary['failed_suites']) == 0 and
            stats['overall_success_rate'] >= 95 and
            thresholds['failed_checks'] == 0
        )
        
        logger.info(f"\nOVERALL ASSESSMENT: {'✅ PASS' if overall_pass else '❌ FAIL'}")
        
        if not overall_pass:
            logger.info(f"\nIssues Found:")
            if summary['failed_suites']:
                logger.error(f"  - Failed test suites: {', '.join(summary['failed_suites'])}")
            if stats['overall_success_rate'] < 95:
                logger.info(f"  - Low success rate: {stats['overall_success_rate']:.1f}% (minimum: 95%)")
            if thresholds['failed_checks'] > 0:
                logger.error(f"  - {thresholds['failed_checks']} threshold validation failures")
        
        logger.info("="*80)
    
    def execute_comprehensive_testing(self, test_types: List[str] = None) -> Dict[str, Any]:
        """Execute comprehensive testing suite"""
        logger.info("Starting comprehensive testing execution...")
        self.start_time = time.time()
        
        # Check agent availability
        if not self.check_agent_availability():
            return {
                'status': 'failed',
                'error': 'Agent not available',
                'timestamp': datetime.now().isoformat()
            }
        
        # Setup test environment
        if not self.setup_test_environment():
            return {
                'status': 'failed',
                'error': 'Test environment setup failed',
                'timestamp': datetime.now().isoformat()
            }
        
        # Determine which tests to run
        if not test_types:
            test_types = [
                suite for suite, enabled 
                in self.config['test_suites'].items() 
                if enabled
            ]
        
        logger.info(f"Executing test suites: {', '.join(test_types)}")
        
        try:
            # Execute test suites in order
            if 'e2e' in test_types:
                self.results['e2e'] = self.run_e2e_tests()
            
            if 'performance' in test_types:
                self.results['performance'] = self.run_performance_tests()
            
            if 'stress' in test_types:
                # Stress tests are included in performance suite
                logger.info("Stress tests included in performance suite")
            
            if 'continuous' in test_types:
                self.results['continuous'] = self.run_continuous_tests()
            
            if 'manual' in test_types:
                self.results['manual'] = self.run_manual_tests()
            
            # Generate comprehensive report
            report = self.generate_comprehensive_report()
            
            # Save report
            report_file = self.save_report(report)
            report['report_file'] = report_file
            
            # Print summary
            self.print_summary(report)
            
            return report
            
        except Exception as e:
            logger.error(f"Comprehensive testing failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'partial_results': self.results,
                'timestamp': datetime.now().isoformat()
            }
        
        finally:
            # Cleanup
            self.cleanup_test_environment()

def main():
    """Main entry point for test orchestrator"""
    parser = argparse.ArgumentParser(description="Hardware Optimizer Comprehensive Testing")
    parser.add_argument("--url", default="http://localhost:8116", help="Agent URL")
    parser.add_argument("--config", help="Configuration file path")
    parser.add_argument("--tests", nargs="+", 
                       choices=["e2e", "performance", "stress", "manual", "continuous", "all"],
                       default=["all"], help="Test types to run")
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--no-cleanup", action="store_true", help="Skip environment cleanup")
    
    args = parser.parse_args()
    
    # Load configuration if provided
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    
    # Update configuration with command line arguments
    if not config:
        config = {}
    
    config['agent_url'] = args.url
    
    if args.no_cleanup:
        config.setdefault('test_environment', {})['cleanup_after'] = False
    
    # Determine test types
    test_types = args.tests
    if 'all' in test_types:
        test_types = ['e2e', 'performance', 'continuous']  # Exclude manual unless explicitly requested
    
    # Initialize and run orchestrator
    orchestrator = TestExecutionOrchestrator(config)
    results = orchestrator.execute_comprehensive_testing(test_types)
    
    # Save to specified output file if provided
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\nResults also saved to: {args.output}")
    
    # Exit with appropriate code
    if results.get('status') == 'failed':
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()