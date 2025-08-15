#!/usr/bin/env python3
"""
import logging

logger = logging.getLogger(__name__)
SutazAI Ultra Test Runner
Professional test execution with comprehensive reporting per Rules 1-19
"""

import os
import sys
import subprocess
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional
import tempfile

class UltraTestRunner:
    """Professional test runner for SutazAI system."""
    
    def __init__(self):
        self.base_dir = Path(__file__).parent.parent
        self.tests_dir = Path(__file__).parent
        self.reports_dir = self.tests_dir / 'reports'
        self.reports_dir.mkdir(exist_ok=True)
        
        # Test suites configuration
        self.test_suites = {
            'unit': {
                'path': 'tests/unit',
                'markers': 'unit',
                'description': 'Unit tests for individual components',
                'timeout': 300,
                'fast': True
            },
            'integration': {
                'path': 'tests/integration',
                'markers': 'integration',
                'description': 'Integration tests for API and services',
                'timeout': 600,
                'fast': True
            },
            'e2e': {
                'path': 'tests/e2e',
                'markers': 'e2e',
                'description': 'End-to-end user workflow tests',
                'timeout': 900,
                'fast': False
            },
            'security': {
                'path': 'tests/security',
                'markers': 'security',
                'description': 'Security vulnerability tests',
                'timeout': 600,
                'fast': True
            },
            'performance': {
                'path': 'tests/performance',
                'markers': 'performance and not slow',
                'description': 'Performance and load tests',
                'timeout': 1200,
                'fast': False
            },
            'regression': {
                'path': 'tests/regression',
                'markers': 'regression',
                'description': 'Regression and compatibility tests',
                'timeout': 600,
                'fast': True
            },
            'monitoring': {
                'path': 'tests/monitoring',
                'markers': 'monitoring',
                'description': 'System monitoring and validation tests',
                'timeout': 300,
                'fast': True
            }
        }
    
    def setup_environment(self):
        """Setup test environment and validate prerequisites."""
        # Set test environment variables
        os.environ['TESTING'] = 'true'
        os.environ['LOG_LEVEL'] = 'WARNING'
        os.environ['PYTEST_CURRENT_TEST'] = 'true'
        
        # Set default service URLs if not provided
        if 'TEST_BASE_URL' not in os.environ:
            os.environ['TEST_BASE_URL'] = 'http://localhost:10010'
        if 'FRONTEND_URL' not in os.environ:
            os.environ['FRONTEND_URL'] = 'http://localhost:10011'
        if 'OLLAMA_URL' not in os.environ:
            os.environ['OLLAMA_URL'] = 'http://localhost:10104'
        
        # Add project paths to PYTHONPATH
        pythonpath = os.environ.get('PYTHONPATH', '')
        project_paths = [
            str(self.base_dir / 'backend'),
            str(self.base_dir / 'agents'),
            str(self.base_dir),
        ]
        
        for path in project_paths:
            if path not in pythonpath:
                pythonpath = f"{path}:{pythonpath}" if pythonpath else path
        
        os.environ['PYTHONPATH'] = pythonpath
        
        logger.info("✅ Test environment configured")
    
    def check_system_health(self) -> bool:
        """Check if system services are running for integration tests."""
        try:
            import requests
            
            # Check backend health
            response = requests.get(f"{os.environ['TEST_BASE_URL']}/health", timeout=5)
            if response.status_code == 200:
                logger.info("✅ Backend service is healthy")
                return True
            else:
                logger.info("⚠️ Backend service not responding, skipping integration tests")
                return False
                
        except Exception as e:
            logger.info(f"⚠️ Cannot connect to backend service: {e}")
            logger.info("🔄 Running tests in unit-only mode")
            return False
    
    def run_test_suite(self, suite_name: str, fast_mode: bool = False, verbose: bool = False) -> Dict:
        """Run a specific test suite and return results."""
        suite = self.test_suites.get(suite_name)
        if not suite:
            raise ValueError(f"Unknown test suite: {suite_name}")
        
        if fast_mode and not suite['fast']:
            logger.info(f"⏭️ Skipping {suite_name} tests in fast mode")
            return {'skipped': True, 'reason': 'fast_mode'}
        
        logger.info(f"🧪 Running {suite_name} tests: {suite['description']}")
        
        # Build pytest command - start with basics
        cmd = [
            sys.executable, '-m', 'pytest',
            suite['path'],
            f'-m', suite['markers'],
            '--tb=short',
            '--durations=10',
        ]
        
        # Add optional plugins only if available
        try:
            import pytest_timeout
            cmd.append(f'--timeout={suite["timeout"]}')
        except ImportError:
            pass  # Skip timeout if plugin not available
        
        try:
            import pytest_cov
            cmd.extend([
                f'--junitxml={self.reports_dir}/junit_{suite_name}.xml',
                f'--cov-report=json:{self.reports_dir}/coverage_{suite_name}.json',
            ])
            # Add coverage for relevant suites
            if suite_name in ['unit', 'integration']:
                cmd.extend([
                    '--cov=backend',
                    '--cov=agents',
                    f'--cov-report=html:{self.reports_dir}/coverage_{suite_name}_html'
                ])
        except ImportError:
            # Fallback to basic junit if pytest-cov not available
            cmd.append(f'--junitxml={self.reports_dir}/junit_{suite_name}.xml')
        
        if verbose:
            cmd.extend(['-v', '--tb=long'])
        else:
            cmd.extend(['-q'])
        
        # Run the tests
        start_time = time.time()
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=suite['timeout'],
                cwd=self.base_dir
            )
            
            duration = time.time() - start_time
            
            return {
                'suite': suite_name,
                'success': result.returncode == 0,
                'duration': duration,
                'returncode': result.returncode,
                'stdout': result.stdout,
                'stderr': result.stderr,
                'command': ' '.join(cmd)
            }
            
        except subprocess.TimeoutExpired:
            return {
                'suite': suite_name,
                'success': False,
                'duration': suite['timeout'],
                'returncode': -1,
                'error': f"Test suite timed out after {suite['timeout']} seconds",
                'command': ' '.join(cmd)
            }
    
    def generate_report(self, results: List[Dict], output_file: Optional[str] = None):
        """Generate comprehensive test execution report."""
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        
        if not output_file:
            output_file = str(self.reports_dir / f'comprehensive_test_report_{timestamp}.json')
        
        # Calculate summary statistics
        total_suites = len(results)
        successful_suites = sum(1 for r in results if r.get('success', False))
        skipped_suites = sum(1 for r in results if r.get('skipped', False))
        failed_suites = total_suites - successful_suites - skipped_suites
        total_duration = sum(r.get('duration', 0) for r in results)
        
        report = {
            'timestamp': timestamp,
            'summary': {
                'total_suites': total_suites,
                'successful': successful_suites,
                'failed': failed_suites,
                'skipped': skipped_suites,
                'success_rate': (successful_suites / max(total_suites - skipped_suites, 1)) * 100,
                'total_duration': total_duration
            },
            'environment': {
                'python_version': sys.version,
                'working_directory': str(self.base_dir),
                'test_base_url': os.environ.get('TEST_BASE_URL'),
                'testing_mode': os.environ.get('TESTING', 'false')
            },
            'results': results
        }
        
        # Write JSON report
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Write text summary
        text_file = output_file.replace('.json', '.txt')
        with open(text_file, 'w') as f:
            f.write(f"SutazAI Ultra Test Report - {timestamp}\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Total Suites: {total_suites}\n")
            f.write(f"Successful: {successful_suites}\n")
            f.write(f"Failed: {failed_suites}\n")
            f.write(f"Skipped: {skipped_suites}\n")
            f.write(f"Success Rate: {report['summary']['success_rate']:.1f}%\n")
            f.write(f"Total Duration: {total_duration:.1f} seconds\n\n")
            
            for result in results:
                if result.get('skipped'):
                    f.write(f"⏭️ {result.get('suite', 'unknown')}: SKIPPED - {result.get('reason', 'unknown')}\n")
                elif result.get('success'):
                    f.write(f"✅ {result['suite']}: PASSED ({result['duration']:.1f}s)\n")
                else:
                    f.write(f"❌ {result['suite']}: FAILED ({result.get('duration', 0):.1f}s)\n")
                    if 'error' in result:
                        f.write(f"   Error: {result['error']}\n")
        
        logger.info(f"\n📊 Test report written to: {output_file}")
        logger.info(f"📝 Text summary written to: {text_file}")
        
        return report
    
    def run_all_tests(self, suites: Optional[List[str]] = None, fast_mode: bool = False, 
                     verbose: bool = False, ci_mode: bool = False):
        """Run all or specified test suites."""
        self.setup_environment()
        
        # Check system health for integration tests
        system_healthy = self.check_system_health()
        
        # Determine which suites to run
        if suites is None:
            suites_to_run = list(self.test_suites.keys())
        else:
            suites_to_run = suites
            
        # Skip integration tests if system is not healthy
        if not system_healthy:
            suites_to_run = [s for s in suites_to_run if s not in ['integration', 'e2e']]
            logger.info("⚠️ Skipping integration and e2e tests due to system health")
        
        # Run test suites
        results = []
        overall_success = True
        
        logger.info(f"\n🚀 Starting test execution for {len(suites_to_run)} suites")
        logger.info("=" * 50)
        
        for suite_name in suites_to_run:
            try:
                result = self.run_test_suite(suite_name, fast_mode, verbose)
                results.append(result)
                
                if result.get('success'):
                    logger.info(f"✅ {suite_name}: PASSED ({result['duration']:.1f}s)")
                elif result.get('skipped'):
                    logger.info(f"⏭️ {suite_name}: SKIPPED - {result.get('reason', 'unknown')}")
                else:
                    logger.error(f"❌ {suite_name}: FAILED ({result.get('duration', 0):.1f}s)")
                    if not result.get('skipped'):
                        overall_success = False
                    
                    if verbose and 'stderr' in result and result['stderr']:
                        logger.error(f"   Error output: {result['stderr'][:500]}...")
                        
            except Exception as e:
                logger.error(f"❌ {suite_name}: ERROR - {e}")
                results.append({
                    'suite': suite_name,
                    'success': False,
                    'error': str(e),
                    'duration': 0
                })
                overall_success = False
        
        # Generate comprehensive report
        report = self.generate_report(results)
        
        # Print summary
        logger.info("\n" + "=" * 50)
        logger.info(f"🎯 Test Execution Complete")
        logger.info(f"Success Rate: {report['summary']['success_rate']:.1f}%")
        logger.info(f"Total Duration: {report['summary']['total_duration']:.1f} seconds")
        
        if overall_success:
            logger.info("🎉 All tests passed!")
        else:
            logger.error("⚠️ Some tests failed - check reports for details")
        
        # Exit with appropriate code for CI
        if ci_mode:
            sys.exit(0 if overall_success else 1)
        
        return overall_success

def main():
    """Main entry point for test runner."""
    parser = argparse.ArgumentParser(description='SutazAI Ultra Test Runner')
    parser.add_argument('--fast', action='store_true', 
                       help='Run only fast tests (skip slow e2e and performance tests)')
    parser.add_argument('--suites', nargs='+', 
                       choices=['unit', 'integration', 'e2e', 'security', 'performance', 'regression', 'monitoring'],
                       help='Specific test suites to run')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output with detailed test information')
    parser.add_argument('--ci', action='store_true',
                       help='CI mode - exit with proper return codes')
    
    args = parser.parse_args()
    
    runner = UltraTestRunner()
    success = runner.run_all_tests(
        suites=args.suites,
        fast_mode=args.fast,
        verbose=args.verbose,
        ci_mode=args.ci
    )
    
    return success

if __name__ == '__main__':
    main()