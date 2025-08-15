#!/usr/bin/env python3
"""
Simple Endpoint Testing for Hardware Resource Optimizer
=======================================================

Simplified testing script to validate all endpoints without dependencies.
This script tests all 16 endpoints with various parameter combinations.

Author: QA Team Lead
Version: 1.0.0
"""

import json
import time
import requests
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('SimpleEndpointTest')

class SimpleEndpointTester:
    """Simple endpoint testing without complex dependencies"""
    
    def __init__(self, base_url="http://localhost:8116", timeout=30):
        self.base_url = base_url
        self.timeout = timeout
        self.session = requests.Session()
        self.session.timeout = timeout
        self.results = []
    
    def test_endpoint(self, method, endpoint, params=None, description=""):
        """Test a single endpoint"""
        logger.info(f"Testing {method} {endpoint}: {description}")
        
        start_time = time.time()
        result = {
            'method': method,
            'endpoint': endpoint,
            'params': params,
            'description': description,
            'timestamp': datetime.now().isoformat(),
            'success': False,
            'status_code': None,
            'response_data': None,
            'error': None,
            'duration': 0
        }
        
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = self.session.get(url, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, params=params)
            else:
                raise ValueError(f"Unsupported method: {method}")
            
            result['status_code'] = response.status_code
            result['duration'] = time.time() - start_time
            
            if response.status_code == 200:
                try:
                    result['response_data'] = response.json()
                    result['success'] = True
                    logger.info(f"  ✅ SUCCESS - {response.status_code} in {result['duration']:.3f}s")
                except json.JSONDecodeError:
                    result['response_data'] = response.text
                    result['success'] = True
                    logger.info(f"  ✅ SUCCESS - {response.status_code} (non-JSON) in {result['duration']:.3f}s")
            else:
                result['error'] = response.text
                logger.warning(f"  ⚠️ ERROR - {response.status_code}: {response.text[:100]}")
        
        except Exception as e:
            result['error'] = str(e)
            result['duration'] = time.time() - start_time
            logger.error(f"  ❌ EXCEPTION - {str(e)}")
        
        self.results.append(result)
        return result
    
    def run_all_endpoint_tests(self):
        """Test all endpoints with various parameters"""
        logger.info("Starting comprehensive endpoint testing...")
        
        # Health and Status Tests
        self.test_endpoint("GET", "/health", description="Basic health check")
        self.test_endpoint("GET", "/status", description="System status check")
        
        # Core Optimization Tests
        self.test_endpoint("POST", "/optimize/memory", description="Memory optimization")
        self.test_endpoint("POST", "/optimize/cpu", description="CPU optimization")
        self.test_endpoint("POST", "/optimize/disk", description="Disk optimization")
        self.test_endpoint("POST", "/optimize/docker", description="Docker cleanup")
        self.test_endpoint("POST", "/optimize/all", description="Comprehensive optimization")
        
        # Storage Analysis Tests
        self.test_endpoint("GET", "/analyze/storage", {"path": "/tmp"}, "Storage analysis - /tmp")
        self.test_endpoint("GET", "/analyze/storage", {"path": "/var"}, "Storage analysis - /var")
        self.test_endpoint("GET", "/analyze/storage", {"path": "/"}, "Storage analysis - root")
        
        self.test_endpoint("GET", "/analyze/storage/duplicates", {"path": "/tmp"}, "Duplicate analysis - /tmp")
        self.test_endpoint("GET", "/analyze/storage/duplicates", {"path": "/var"}, "Duplicate analysis - /var")
        
        self.test_endpoint("GET", "/analyze/storage/large-files", {"path": "/", "min_size_mb": 100}, "Large files - 100MB threshold")
        self.test_endpoint("GET", "/analyze/storage/large-files", {"path": "/", "min_size_mb": 50}, "Large files - 50MB threshold")
        self.test_endpoint("GET", "/analyze/storage/large-files", {"path": "/tmp", "min_size_mb": 10}, "Large files - /tmp 10MB")
        
        self.test_endpoint("GET", "/analyze/storage/report", description="Comprehensive storage report")
        
        # Storage Optimization Tests (Dry Run)
        self.test_endpoint("POST", "/optimize/storage", {"dry_run": True}, "Storage optimization - dry run")
        self.test_endpoint("POST", "/optimize/storage/duplicates", {"path": "/tmp", "dry_run": True}, "Duplicate removal - dry run")
        self.test_endpoint("POST", "/optimize/storage/cache", description="Cache cleanup")
        self.test_endpoint("POST", "/optimize/storage/compress", {"path": "/var/log", "days_old": 30}, "File compression")
        self.test_endpoint("POST", "/optimize/storage/logs", description="Log optimization")
        
        # Error Handling Tests
        self.test_endpoint("GET", "/analyze/storage", {"path": "/nonexistent"}, "Invalid path test")
        self.test_endpoint("GET", "/analyze/storage", {"path": "/etc"}, "Protected path test")
        self.test_endpoint("GET", "/analyze/storage/large-files", {"path": "/", "min_size_mb": -1}, "Negative size test")
        
        # Parameter Edge Cases
        self.test_endpoint("GET", "/analyze/storage", {"path": ""}, "Empty path test")
        self.test_endpoint("POST", "/optimize/storage/compress", {"path": "/tmp", "days_old": 0}, "Zero days compression")
        
        logger.info("All endpoint tests completed")
    
    def generate_report(self):
        """Generate comprehensive test report"""
        total_tests = len(self.results)
        successful_tests = sum(1 for r in self.results if r['success'])
        failed_tests = total_tests - successful_tests
        
        # Calculate statistics
        response_times = [r['duration'] for r in self.results if r['success']]
        avg_response_time = sum(response_times) / len(response_times) if response_times else 0
        max_response_time = max(response_times) if response_times else 0
        
        # Group results by endpoint
        endpoint_summary = {}
        for result in self.results:
            endpoint = result['endpoint']
            if endpoint not in endpoint_summary:
                endpoint_summary[endpoint] = {'total': 0, 'success': 0, 'methods': set()}
            
            endpoint_summary[endpoint]['total'] += 1
            endpoint_summary[endpoint]['methods'].add(result['method'])
            if result['success']:
                endpoint_summary[endpoint]['success'] += 1
        
        # Convert sets to lists for JSON serialization
        for endpoint_data in endpoint_summary.values():
            endpoint_data['methods'] = list(endpoint_data['methods'])
        
        report = {
            'test_summary': {
                'timestamp': datetime.now().isoformat(),
                'total_tests': total_tests,
                'successful_tests': successful_tests,
                'failed_tests': failed_tests,
                'success_rate_percent': (successful_tests / total_tests * 100) if total_tests > 0 else 0,
                'avg_response_time_seconds': avg_response_time,
                'max_response_time_seconds': max_response_time
            },
            'endpoint_coverage': {
                'unique_endpoints_tested': len(endpoint_summary),
                'endpoint_summary': endpoint_summary
            },
            'detailed_results': self.results,
            'agent_url': self.base_url
        }
        
        return report
    
    def print_summary(self, report):
        """Print test summary"""
        summary = report['test_summary']
        coverage = report['endpoint_coverage']
        
        logger.info("\n" + "="*80)
        logger.info("HARDWARE RESOURCE OPTIMIZER - ENDPOINT TEST SUMMARY")
        logger.info("="*80)
        
        logger.info(f"Agent URL: {self.base_url}")
        logger.info(f"Test Timestamp: {summary['timestamp']}")
        logger.info(f"Total Tests: {summary['total_tests']}")
        logger.info(f"Successful Tests: {summary['successful_tests']}")
        logger.error(f"Failed Tests: {summary['failed_tests']}")
        logger.info(f"Success Rate: {summary['success_rate_percent']:.1f}%")
        logger.info(f"Average Response Time: {summary['avg_response_time_seconds']:.3f}s")
        logger.info(f"Max Response Time: {summary['max_response_time_seconds']:.3f}s")
        logger.info(f"Unique Endpoints Tested: {coverage['unique_endpoints_tested']}")
        
        logger.info(f"\nEndpoint Coverage:")
        for endpoint, data in coverage['endpoint_summary'].items():
            success_rate = (data['success'] / data['total'] * 100) if data['total'] > 0 else 0
            methods = ', '.join(data['methods'])
            logger.info(f"  {endpoint} ({methods}): {data['success']}/{data['total']} ({success_rate:.1f}%)")
        
        logger.error(f"\nFailed Tests:")
        failed_results = [r for r in self.results if not r['success']]
        if failed_results:
            for result in failed_results:
                logger.error(f"  ❌ {result['method']} {result['endpoint']}: {result['error'][:50]}...")
        else:
            logger.error("  ✅ No failed tests")
        
        # Overall assessment
        if summary['success_rate_percent'] >= 90:
            logger.info(f"\n🎉 OVERALL ASSESSMENT: EXCELLENT ({summary['success_rate_percent']:.1f}% success rate)")
        elif summary['success_rate_percent'] >= 75:
            logger.info(f"\n👍 OVERALL ASSESSMENT: GOOD ({summary['success_rate_percent']:.1f}% success rate)")
        elif summary['success_rate_percent'] >= 50:
            logger.info(f"\n⚠️ OVERALL ASSESSMENT: NEEDS IMPROVEMENT ({summary['success_rate_percent']:.1f}% success rate)")
        else:
            logger.info(f"\n❌ OVERALL ASSESSMENT: POOR ({summary['success_rate_percent']:.1f}% success rate)")
        
        logger.info("="*80)
    
    def save_report(self, report, filename=None):
        """Save report to file"""
        if not filename:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"endpoint_test_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"\nDetailed report saved to: {filename}")
        return filename

def main():
    """Main execution"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Simple Hardware Optimizer Endpoint Testing")
    parser.add_argument("--url", default="http://localhost:8116", help="Agent URL")
    parser.add_argument("--output", help="Output report file")
    parser.add_argument("--timeout", type=int, default=30, help="Request timeout")
    
    args = parser.parse_args()
    
    # Initialize tester
    tester = SimpleEndpointTester(args.url, args.timeout)
    
    # Check if agent is available
    try:
        response = requests.get(f"{args.url}/health", timeout=10)
        if response.status_code != 200:
            logger.info(f"❌ Agent not available at {args.url}")
            return 1
        logger.info(f"✅ Agent is available at {args.url}")
    except Exception as e:
        logger.info(f"❌ Cannot connect to agent at {args.url}: {e}")
        return 1
    
    # Run tests
    start_time = time.time()
    tester.run_all_endpoint_tests()
    total_duration = time.time() - start_time
    
    # Generate and print report
    report = tester.generate_report()
    report['test_summary']['total_duration_seconds'] = total_duration
    
    tester.print_summary(report)
    
    # Save report
    output_file = args.output or f"endpoint_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    tester.save_report(report, output_file)
    
    # Return appropriate exit code
    success_rate = report['test_summary']['success_rate_percent']
    return 0 if success_rate >= 80 else 1

if __name__ == "__main__":
    exit(main())