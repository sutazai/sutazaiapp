#!/usr/bin/env python3
"""
Master Test Runner for SutazAI System
Executes all test suites with proper organization per Rules 1-19
"""

import os
import sys
import subprocess
import time
import json
import argparse
from datetime import datetime
from typing import Dict, List, Any
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestSuiteRunner:
    """Comprehensive test suite runner with reporting"""
    
    def __init__(self, base_dir: str = None):
        self.base_dir = base_dir or os.path.dirname(__file__)
        self.results = {}
        self.start_time = None
        self.end_time = None
    
    def run_unit_tests(self) -> Dict[str, Any]:
        """Run unit tests"""
        print("\n=== Running Unit Tests ===")
        
        unit_args = [
            '-v',
            '--tb=short',
            '-m', 'unit',
            '--cov=backend',
            '--cov-report=html:tests/reports/coverage/unit',
            '--cov-report=json:tests/reports/coverage/unit.json',
            '--junit-xml=tests/reports/junit/unit.xml',
            'tests/unit/'
        ]
        
        result = pytest.main(unit_args)
        return {
            'name': 'unit',
            'exit_code': result,
            'passed': result == 0
        }
    
    def run_integration_tests(self) -> Dict[str, Any]:
        """Run integration tests"""
        print("\n=== Running Integration Tests ===")
        
        integration_args = [
            '-v',
            '--tb=short',
            '-m', 'integration',
            '--junit-xml=tests/reports/junit/integration.xml',
            'tests/integration/'
        ]
        
        result = pytest.main(integration_args)
        return {
            'name': 'integration',
            'exit_code': result,
            'passed': result == 0
        }
    
    def run_e2e_tests(self) -> Dict[str, Any]:
        """Run E2E tests"""
        print("\n=== Running E2E Tests ===")
        
        e2e_args = [
            '-v',
            '--tb=short',
            '-m', 'e2e',
            '--junit-xml=tests/reports/junit/e2e.xml',
            'tests/e2e/'
        ]
        
        result = pytest.main(e2e_args)
        return {
            'name': 'e2e',
            'exit_code': result,
            'passed': result == 0
        }
    
    def run_performance_tests(self) -> Dict[str, Any]:
        """Run performance tests"""
        print("\n=== Running Performance Tests ===")
        
        perf_args = [
            '-v',
            '--tb=short',
            '-m', 'performance and not slow',
            '--junit-xml=tests/reports/junit/performance.xml',
            'tests/performance/'
        ]
        
        result = pytest.main(perf_args)
        return {
            'name': 'performance',
            'exit_code': result,
            'passed': result == 0
        }
    
    def run_security_tests(self) -> Dict[str, Any]:
        """Run security tests"""
        print("\n=== Running Security Tests ===")
        
        security_args = [
            '-v',
            '--tb=short',
            '-m', 'security',
            '--junit-xml=tests/reports/junit/security.xml',
            'tests/security/'
        ]
        
        result = pytest.main(security_args)
        return {
            'name': 'security',
            'exit_code': result,
            'passed': result == 0
        }
    
    def run_slow_tests(self) -> Dict[str, Any]:
        """Run slow tests (performance + load tests)"""
        print("\n=== Running Slow Tests ===")
        
        slow_args = [
            '-v',
            '--tb=short',
            '-m', 'slow',
            '--junit-xml=tests/reports/junit/slow.xml',
            'tests/'
        ]
        
        result = pytest.main(slow_args)
        return {
            'name': 'slow',
            'exit_code': result,
            'passed': result == 0
        }
    
    def setup_test_environment(self) -> bool:
        """Setup test environment and check dependencies"""
        print("=== Setting up Test Environment ===")
        
        # Create reports directory
        reports_dir = os.path.join(self.base_dir, 'reports')
        os.makedirs(reports_dir, exist_ok=True)
        
        # Create subdirectories
        for subdir in ['junit', 'coverage', 'performance', 'security']:
            os.makedirs(os.path.join(reports_dir, subdir), exist_ok=True)
        
        # Check if system is running
        try:
            import httpx
            import asyncio
            
            async def check_system():
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get('http://localhost:10010/health')
                    return response.status_code == 200
            
            system_running = asyncio.run(check_system())
            
            if system_running:
                print("✓ SutazAI system is running")
                return True
            else:
                print("⚠ SutazAI system is not responding")
                return False
                
        except Exception as e:
            print(f"⚠ Could not check system status: {e}")
            return False
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive test report"""
        duration = (self.end_time - self.start_time).total_seconds()
        
        total_suites = len(self.results)
        passed_suites = sum(1 for r in self.results.values() if r['passed'])
        
        report = {
            'timestamp': datetime.utcnow().isoformat(),
            'duration_seconds': duration,
            'summary': {
                'total_suites': total_suites,
                'passed_suites': passed_suites,
                'failed_suites': total_suites - passed_suites,
                'success_rate': (passed_suites / total_suites) * 100 if total_suites > 0 else 0
            },
            'suites': self.results,
            'recommendations': self._get_recommendations()
        }
        
        # Save report
        report_file = os.path.join(
            self.base_dir, 'reports', 
            f'test_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        )
        
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Test report saved: {report_file}")
        return report
    
    def _get_recommendations(self) -> List[str]:
        """Get recommendations based on test results"""
        recommendations = []
        
        failed_suites = [name for name, result in self.results.items() if not result['passed']]
        
        if failed_suites:
            recommendations.append(f"Fix failing test suites: {', '.join(failed_suites)}")
        
        if 'unit' in self.results and not self.results['unit']['passed']:
            recommendations.append("Unit tests failing - check core functionality")
        
        if 'integration' in self.results and not self.results['integration']['passed']:
            recommendations.append("Integration tests failing - check API endpoints")
        
        if 'security' in self.results and not self.results['security']['passed']:
            recommendations.append("Security tests failing - review security measures")
        
        if 'performance' in self.results and not self.results['performance']['passed']:
            recommendations.append("Performance tests failing - optimize system performance")
        
        if not recommendations:
            recommendations.append("All tests passing - system is healthy!")
        
        return recommendations
    
    def print_summary(self, report: Dict[str, Any]):
        """Print test summary"""
        print("\n" + "=" * 60)
        print("📋 TEST EXECUTION SUMMARY")
        print("=" * 60)
        
        summary = report['summary']
        print(f"⏱  Duration: {report['duration_seconds']:.1f} seconds")
        print(f"📊 Success Rate: {summary['success_rate']:.1f}%")
        print(f"✅ Passed Suites: {summary['passed_suites']}/{summary['total_suites']}")
        
        if summary['failed_suites'] > 0:
            print(f"❌ Failed Suites: {summary['failed_suites']}")
        
        print("\n📋 Test Suite Results:")
        for name, result in self.results.items():
            status = "✅ PASS" if result['passed'] else "❌ FAIL"
            print(f"  {name.upper():12} {status}")
        
        print("\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
        
        print("\n" + "=" * 60)
    
    def run_all(self, suites: List[str] = None) -> Dict[str, Any]:
        """Run all test suites"""
        self.start_time = datetime.utcnow()
        
        # Default to all suites if none specified
        if suites is None:
            suites = ['unit', 'integration', 'security', 'performance', 'e2e']
        
        # Setup environment
        system_ready = self.setup_test_environment()
        
        # Run test suites
        suite_runners = {
            'unit': self.run_unit_tests,
            'integration': self.run_integration_tests,
            'e2e': self.run_e2e_tests,
            'performance': self.run_performance_tests,
            'security': self.run_security_tests,
            'slow': self.run_slow_tests
        }
        
        for suite_name in suites:
            if suite_name in suite_runners:
                try:
                    result = suite_runners[suite_name]()
                    self.results[suite_name] = result
                except Exception as e:
                    print(f"Error running {suite_name} tests: {e}")
                    self.results[suite_name] = {
                        'name': suite_name,
                        'exit_code': -1,
                        'passed': False,
                        'error': str(e)
                    }
        
        self.end_time = datetime.utcnow()
        
        # Generate report
        report = self.generate_report()
        self.print_summary(report)
        
        return report


def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description='SutazAI Test Suite Runner')
    parser.add_argument(
        '--suites', 
        nargs='+', 
        choices=['unit', 'integration', 'e2e', 'performance', 'security', 'slow', 'all'],
        default=['all'],
        help='Test suites to run'
    )
    parser.add_argument(
        '--fast', 
        action='store_true',
        help='Skip slow tests (performance load tests)'
    )
    parser.add_argument(
        '--ci', 
        action='store_true',
        help='CI mode - exit with non-zero code if any tests fail'
    )
    
    args = parser.parse_args()
    
    # Determine which suites to run
    if 'all' in args.suites:
        suites = ['unit', 'integration', 'security', 'performance', 'e2e']
        if not args.fast:
            suites.append('slow')
    else:
        suites = args.suites
    
    # Run tests
    runner = TestSuiteRunner()
    report = runner.run_all(suites)
    
    # Exit with appropriate code for CI
    if args.ci:
        failed_suites = report['summary']['failed_suites']
        if failed_suites > 0:
            print(f"\n❌ CI Mode: {failed_suites} test suite(s) failed")
            sys.exit(1)
        else:
            print("\n✅ CI Mode: All test suites passed")
            sys.exit(0)


if __name__ == '__main__':
    main()
