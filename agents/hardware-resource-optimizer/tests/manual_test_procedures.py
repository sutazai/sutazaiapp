#!/usr/bin/env python3
"""
Manual Test Procedures for Hardware Resource Optimizer
======================================================

This module provides structured manual testing procedures for complex scenarios
that require human validation or cannot be fully automated:
- Integration testing with system components
- User workflow validation
- Edge case exploration
- Visual verification of results
- Complex scenario testing

Author: QA Team Lead
Version: 1.0.0
"""

import os
import sys
import json
import time
import requests
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('ManualTests')

class ManualTestProcedures:
    """Structured manual testing procedures"""
    
    def __init__(self, base_url: str = "http://localhost:8116"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.timeout = 30
        self.test_results = []
    
    def print_test_header(self, test_name: str, description: str):
        """Print formatted test header"""
        print("\n" + "="*80)
        print(f"MANUAL TEST: {test_name}")
        print("="*80)
        print(f"Description: {description}")
        print("-"*80)
    
    def print_step(self, step_num: int, instruction: str):
        """Print formatted step instruction"""
        print(f"\nSTEP {step_num}: {instruction}")
    
    def wait_for_user_confirmation(self, prompt: str = "Press Enter when ready to continue...") -> str:
        """Wait for user input/confirmation"""
        return input(f"\n{prompt} ")
    
    def get_user_validation(self, question: str) -> bool:
        """Get user validation (yes/no)"""
        while True:
            response = input(f"\n{question} (y/n): ").lower().strip()
            if response in ['y', 'yes']:
                return True
            elif response in ['n', 'no']:
                return False
            else:
                print("Please enter 'y' for yes or 'n' for no")
    
    def make_request(self, method: str, endpoint: str, params: Dict = None, show_response: bool = True) -> Dict[str, Any]:
        """Make API request and optionally show response"""
        try:
            url = f"{self.base_url}{endpoint}"
            
            if method.upper() == "GET":
                response = self.session.get(url, params=params)
            elif method.upper() == "POST":
                response = self.session.post(url, params=params)
            
            result = {
                'method': method,
                'endpoint': endpoint,
                'params': params,
                'status_code': response.status_code,
                'success': response.status_code == 200
            }
            
            if response.status_code == 200:
                try:
                    result['data'] = response.json()
                    if show_response:
                        print(f"\nResponse ({response.status_code}):")
                        print(json.dumps(result['data'], indent=2))
                except json.JSONDecodeError:
                    result['data'] = response.text
                    if show_response:
                        print(f"\nResponse ({response.status_code}): {response.text}")
            else:
                result['error'] = response.text
                if show_response:
                    print(f"\nError ({response.status_code}): {response.text}")
            
            return result
            
        except Exception as e:
            result = {
                'method': method,
                'endpoint': endpoint,
                'params': params,
                'success': False,
                'error': str(e)
            }
            if show_response:
                print(f"\nRequest failed: {e}")
            return result
    
    def test_basic_health_workflow(self):
        """Test MT-001: Basic Health Check Workflow"""
        self.print_test_header(
            "MT-001: Basic Health Check Workflow",
            "Verify the basic health check workflow functions correctly"
        )
        
        self.print_step(1, "Check initial agent health status")
        health_result = self.make_request("GET", "/health")
        
        if health_result['success']:
            health_data = health_result['data']
            print("\nValidation Points:")
            print(f"- Agent status: {health_data.get('status')}")
            print(f"- Agent ID: {health_data.get('agent')}")
            print(f"- Docker available: {health_data.get('docker_available')}")
            print(f"- System status present: {'system_status' in health_data}")
            
            all_good = self.get_user_validation("Does the health check response look correct?")
        else:
            all_good = False
            print("❌ Health check failed!")
        
        self.print_step(2, "Check system status endpoint")
        status_result = self.make_request("GET", "/status")
        
        if status_result['success']:
            status_data = status_result['data']
            print("\nValidation Points:")
            print(f"- CPU percent: {status_data.get('cpu_percent')}%")
            print(f"- Memory percent: {status_data.get('memory_percent')}%")
            print(f"- Disk percent: {status_data.get('disk_percent')}%")
            
            status_good = self.get_user_validation("Do the system metrics look reasonable?")
            all_good = all_good and status_good
        else:
            all_good = False
            print("❌ Status check failed!")
        
        test_result = {
            'test_id': 'MT-001',
            'test_name': 'Basic Health Check Workflow',
            'timestamp': datetime.now().isoformat(),
            'passed': all_good,
            'health_check': health_result,
            'status_check': status_result
        }
        
        self.test_results.append(test_result)
        
        if all_good:
            print("\n✅ MT-001 PASSED: Basic health workflow is working correctly")
        else:
            print("\n❌ MT-001 FAILED: Issues found in basic health workflow")
        
        return test_result
    
    def test_memory_optimization_workflow(self):
        """Test MT-002: Memory Optimization Workflow"""
        self.print_test_header(
            "MT-002: Memory Optimization Workflow",
            "Test complete memory optimization workflow with before/after validation"
        )
        
        self.print_step(1, "Get initial system memory status")
        initial_status = self.make_request("GET", "/status")
        
        if initial_status['success']:
            initial_memory = initial_status['data'].get('memory_percent', 0)
            print(f"\nInitial memory usage: {initial_memory}%")
        else:
            print("❌ Could not get initial memory status")
            return {'test_id': 'MT-002', 'passed': False, 'error': 'Initial status failed'}
        
        self.print_step(2, "Run memory optimization")
        print("About to run memory optimization...")
        ready = self.get_user_validation("Are you ready to proceed with memory optimization?")
        
        if not ready:
            print("Test aborted by user")
            return {'test_id': 'MT-002', 'passed': False, 'error': 'User aborted'}
        
        optimize_result = self.make_request("POST", "/optimize/memory")
        
        if optimize_result['success']:
            opt_data = optimize_result['data']
            print("\nOptimization Results:")
            print(f"- Status: {opt_data.get('status')}")
            print(f"- Actions taken: {len(opt_data.get('actions_taken', []))}")
            print(f"- Initial memory: {opt_data.get('initial_memory_percent')}%")
            print(f"- Final memory: {opt_data.get('final_memory_percent')}%")
            print(f"- Memory freed: {opt_data.get('memory_freed_mb', 0):.2f} MB")
            
            for action in opt_data.get('actions_taken', []):
                print(f"  - {action}")
        else:
            print("❌ Memory optimization failed!")
            return {'test_id': 'MT-002', 'passed': False, 'error': 'Optimization failed'}
        
        self.print_step(3, "Verify memory optimization results")
        time.sleep(2)  # Brief pause for system to settle
        final_status = self.make_request("GET", "/status")
        
        if final_status['success']:
            final_memory = final_status['data'].get('memory_percent', 0)
            print(f"\nFinal memory usage: {final_memory}%")
            memory_change = initial_memory - final_memory
            print(f"Memory change: {memory_change:.1f}%")
        else:
            print("❌ Could not get final memory status")
        
        # User validation
        optimization_effective = self.get_user_validation("Did the memory optimization appear to work effectively?")
        actions_reasonable = self.get_user_validation("Do the optimization actions look reasonable and safe?")
        
        test_result = {
            'test_id': 'MT-002',
            'test_name': 'Memory Optimization Workflow',
            'timestamp': datetime.now().isoformat(),
            'passed': optimization_effective and actions_reasonable,
            'initial_status': initial_status,
            'optimization_result': optimize_result,
            'final_status': final_status,
            'user_validation': {
                'optimization_effective': optimization_effective,
                'actions_reasonable': actions_reasonable
            }
        }
        
        self.test_results.append(test_result)
        
        if test_result['passed']:
            print("\n✅ MT-002 PASSED: Memory optimization workflow working correctly")
        else:
            print("\n❌ MT-002 FAILED: Issues found in memory optimization workflow")
        
        return test_result
    
    def test_storage_analysis_workflow(self):
        """Test MT-003: Storage Analysis Workflow"""
        self.print_test_header(
            "MT-003: Storage Analysis Workflow",
            "Test comprehensive storage analysis across different paths and scenarios"
        )
        
        test_paths = ["/tmp", "/var/log", "/opt"]
        analysis_results = []
        
        for i, path in enumerate(test_paths, 1):
            self.print_step(i, f"Analyze storage for path: {path}")
            
            analysis_result = self.make_request("GET", "/analyze/storage", {"path": path})
            analysis_results.append(analysis_result)
            
            if analysis_result['success'] and analysis_result['data'].get('status') == 'success':
                data = analysis_result['data']
                print(f"\nStorage Analysis for {path}:")
                print(f"- Total files: {data.get('total_files', 0):,}")
                print(f"- Total size: {data.get('total_size_mb', 0):.2f} MB")
                print(f"- Top extensions:")
                
                ext_breakdown = data.get('extension_breakdown', {})
                for ext, info in list(ext_breakdown.items())[:5]:
                    size_mb = info.get('total_size', 0) / (1024 * 1024)
                    print(f"  - {ext}: {info.get('count', 0)} files, {size_mb:.2f} MB")
                
                reasonable = self.get_user_validation(f"Do the results for {path} look reasonable?")
                if not reasonable:
                    print(f"⚠️ User flagged issues with {path} analysis")
            
            else:
                print(f"❌ Analysis failed for {path}")
        
        self.print_step(len(test_paths) + 1, "Test duplicate analysis")
        duplicate_result = self.make_request("GET", "/analyze/storage/duplicates", {"path": "/tmp"})
        
        if duplicate_result['success'] and duplicate_result['data'].get('status') == 'success':
            dup_data = duplicate_result['data']
            print(f"\nDuplicate Analysis:")
            print(f"- Duplicate groups: {dup_data.get('duplicate_groups', 0)}")
            print(f"- Total duplicates: {dup_data.get('total_duplicates', 0)}")
            print(f"- Space wasted: {dup_data.get('space_wasted_mb', 0):.2f} MB")
        
        self.print_step(len(test_paths) + 2, "Test large files analysis")
        large_files_result = self.make_request("GET", "/analyze/storage/large-files", 
                                             {"path": "/", "min_size_mb": 100})
        
        if large_files_result['success'] and large_files_result['data'].get('status') == 'success':
            large_data = large_files_result['data']
            print(f"\nLarge Files Analysis:")
            print(f"- Large files found: {large_data.get('large_files_count', 0)}")
            print(f"- Total size: {large_data.get('total_size_mb', 0):.2f} MB")
        
        self.print_step(len(test_paths) + 3, "Generate storage report")
        report_result = self.make_request("GET", "/analyze/storage/report")
        
        if report_result['success'] and report_result['data'].get('status') == 'success':
            report_data = report_result['data']
            disk_usage = report_data.get('disk_usage', {})
            print(f"\nStorage Report Summary:")
            print(f"- Total disk: {disk_usage.get('total_gb', 0):.1f} GB")
            print(f"- Used disk: {disk_usage.get('used_gb', 0):.1f} GB")
            print(f"- Free disk: {disk_usage.get('free_gb', 0):.1f} GB")
            print(f"- Usage: {disk_usage.get('usage_percent', 0):.1f}%")
        
        # Overall validation
        all_analyses_working = self.get_user_validation("Are all storage analysis functions working correctly?")
        results_accurate = self.get_user_validation("Do the analysis results appear accurate and useful?")
        
        test_result = {
            'test_id': 'MT-003',
            'test_name': 'Storage Analysis Workflow',
            'timestamp': datetime.now().isoformat(),
            'passed': all_analyses_working and results_accurate,
            'analysis_results': analysis_results,
            'duplicate_result': duplicate_result,
            'large_files_result': large_files_result,
            'report_result': report_result,
            'user_validation': {
                'all_analyses_working': all_analyses_working,
                'results_accurate': results_accurate
            }
        }
        
        self.test_results.append(test_result)
        
        if test_result['passed']:
            print("\n✅ MT-003 PASSED: Storage analysis workflow working correctly")
        else:
            print("\n❌ MT-003 FAILED: Issues found in storage analysis workflow")
        
        return test_result
    
    def test_optimization_safety_workflow(self):
        """Test MT-004: Optimization Safety Workflow"""
        self.print_test_header(
            "MT-004: Optimization Safety Workflow",
            "Test safety mechanisms in optimization operations (dry run, path protection, etc.)"
        )
        
        self.print_step(1, "Test dry run functionality")
        print("Testing storage optimization in dry run mode...")
        
        dry_run_result = self.make_request("POST", "/optimize/storage", {"dry_run": True})
        
        if dry_run_result['success']:
            dry_data = dry_run_result['data']
            print(f"\nDry Run Results:")
            print(f"- Status: {dry_data.get('status')}")
            print(f"- Dry run flag: {dry_data.get('dry_run')}")
            print(f"- Actions taken: {len(dry_data.get('actions_taken', []))}")
            print(f"- Estimated space freed: {dry_data.get('estimated_space_freed_mb', 0):.2f} MB")
            
            print("\nActions that would be taken:")
            for action in dry_data.get('actions_taken', [])[:10]:  # Show first 10
                print(f"  - {action}")
            
            dry_run_working = self.get_user_validation("Does the dry run appear to be working correctly (no actual changes)?")
        else:
            dry_run_working = False
            print("❌ Dry run failed!")
        
        self.print_step(2, "Test protected path handling")
        print("Testing access to protected paths...")
        
        protected_paths = ["/etc", "/boot", "/usr/bin"]
        protection_working = True
        
        for path in protected_paths:
            print(f"\nTesting protection for: {path}")
            protected_result = self.make_request("GET", "/analyze/storage", {"path": path}, show_response=False)
            
            if protected_result['success']:
                if protected_result['data'].get('status') == 'error':
                    print(f"✅ {path} properly protected")
                else:
                    print(f"⚠️ {path} may not be properly protected")
                    protection_working = False
            else:
                print(f"❌ Error testing {path}")
                protection_working = False
        
        path_protection_good = self.get_user_validation("Is path protection working correctly?")
        protection_working = protection_working and path_protection_good
        
        self.print_step(3, "Test duplicate removal safety (dry run)")
        duplicate_dry_result = self.make_request("POST", "/optimize/storage/duplicates", 
                                                {"path": "/tmp", "dry_run": True})
        
        if duplicate_dry_result['success']:
            dup_data = duplicate_dry_result['data']
            print(f"\nDuplicate Removal Dry Run:")
            print(f"- Status: {dup_data.get('status')}")
            print(f"- Dry run: {dup_data.get('dry_run')}")
            print(f"- Duplicates found: {dup_data.get('duplicates_removed', 0)}")
            print(f"- Space that would be freed: {dup_data.get('space_freed_mb', 0):.2f} MB")
            
            duplicate_safety_good = self.get_user_validation("Does duplicate removal safety look good?")
        else:
            duplicate_safety_good = False
            print("❌ Duplicate dry run failed!")
        
        # Overall safety assessment
        overall_safety = self.get_user_validation("Do all safety mechanisms appear to be working correctly?")
        
        test_result = {
            'test_id': 'MT-004',
            'test_name': 'Optimization Safety Workflow',
            'timestamp': datetime.now().isoformat(),
            'passed': dry_run_working and protection_working and duplicate_safety_good and overall_safety,
            'dry_run_result': dry_run_result,
            'protection_tests': protected_paths,
            'duplicate_dry_result': duplicate_dry_result,
            'user_validation': {
                'dry_run_working': dry_run_working,
                'path_protection_good': path_protection_good,
                'duplicate_safety_good': duplicate_safety_good,
                'overall_safety': overall_safety
            }
        }
        
        self.test_results.append(test_result)
        
        if test_result['passed']:
            print("\n✅ MT-004 PASSED: Optimization safety mechanisms working correctly")
        else:
            print("\n❌ MT-004 FAILED: Safety mechanism issues found")
        
        return test_result
    
    def test_comprehensive_optimization_workflow(self):
        """Test MT-005: Comprehensive Optimization Workflow"""
        self.print_test_header(
            "MT-005: Comprehensive Optimization Workflow",
            "Test the complete optimization workflow with all components"
        )
        
        self.print_step(1, "Get baseline system status")
        baseline = self.make_request("GET", "/status")
        
        if baseline['success']:
            baseline_data = baseline['data']
            print(f"\nBaseline System Status:")
            print(f"- CPU: {baseline_data.get('cpu_percent')}%")
            print(f"- Memory: {baseline_data.get('memory_percent')}%")
            print(f"- Disk: {baseline_data.get('disk_percent')}%")
        
        self.print_step(2, "Run comprehensive optimization")
        print("⚠️  WARNING: This will run actual optimization (not dry run)")
        print("This may make real changes to the system!")
        
        proceed = self.get_user_validation("Do you want to proceed with comprehensive optimization?")
        
        if not proceed:
            print("Test skipped by user choice")
            return {'test_id': 'MT-005', 'passed': False, 'error': 'User skipped'}
        
        comprehensive_result = self.make_request("POST", "/optimize/all")
        
        if comprehensive_result['success']:
            comp_data = comprehensive_result['data']
            print(f"\nComprehensive Optimization Results:")
            print(f"- Status: {comp_data.get('status')}")
            print(f"- Duration: {comp_data.get('duration_seconds', 0):.2f} seconds")
            print(f"- Total actions: {comp_data.get('total_actions', 0)}")
            
            # Show before/after comparison
            before = comp_data.get('before', {})
            after = comp_data.get('after', {})
            
            if before and after:
                print(f"\nBefore/After Comparison:")
                print(f"- CPU: {before.get('cpu_percent', 0):.1f}% → {after.get('cpu_percent', 0):.1f}%")
                print(f"- Memory: {before.get('memory_percent', 0):.1f}% → {after.get('memory_percent', 0):.1f}%")
                print(f"- Disk: {before.get('disk_percent', 0):.1f}% → {after.get('disk_percent', 0):.1f}%")
            
            # Show detailed results
            detailed = comp_data.get('detailed_results', {})
            for opt_type, result in detailed.items():
                if result.get('status') == 'success':
                    actions = result.get('actions_taken', [])
                    print(f"\n{opt_type.title()} Optimization:")
                    for action in actions[:3]:  # Show first 3 actions
                        print(f"  - {action}")
                    if len(actions) > 3:
                        print(f"  ... and {len(actions) - 3} more actions")
        
        else:
            print("❌ Comprehensive optimization failed!")
            return {'test_id': 'MT-005', 'passed': False, 'error': 'Optimization failed'}
        
        self.print_step(3, "Verify system stability after optimization")
        time.sleep(5)  # Wait for system to settle
        
        post_opt_status = self.make_request("GET", "/status")
        
        if post_opt_status['success']:
            print(f"\nPost-optimization system check: ✅ System responsive")
        else:
            print(f"\n❌ Post-optimization system check failed!")
        
        # User validation
        optimization_completed = self.get_user_validation("Did the comprehensive optimization complete successfully?")
        system_stable = self.get_user_validation("Does the system appear stable after optimization?")
        results_reasonable = self.get_user_validation("Do the optimization results look reasonable?")
        
        test_result = {
            'test_id': 'MT-005',
            'test_name': 'Comprehensive Optimization Workflow',
            'timestamp': datetime.now().isoformat(),
            'passed': optimization_completed and system_stable and results_reasonable,
            'baseline': baseline,
            'comprehensive_result': comprehensive_result,
            'post_opt_status': post_opt_status,
            'user_validation': {
                'optimization_completed': optimization_completed,
                'system_stable': system_stable,
                'results_reasonable': results_reasonable
            }
        }
        
        self.test_results.append(test_result)
        
        if test_result['passed']:
            print("\n✅ MT-005 PASSED: Comprehensive optimization workflow working correctly")
        else:
            print("\n❌ MT-005 FAILED: Issues found in comprehensive optimization")
        
        return test_result
    
    def run_manual_test_suite(self) -> Dict[str, Any]:
        """Run the complete manual test suite"""
        print("\n" + "="*80)
        print("HARDWARE RESOURCE OPTIMIZER - MANUAL TEST SUITE")
        print("="*80)
        print("This suite will guide you through comprehensive manual testing")
        print("of the Hardware Resource Optimizer agent.")
        print("\nPlease follow the instructions carefully and provide honest")
        print("assessments of what you observe.")
        print("="*80)
        
        ready = self.get_user_validation("Are you ready to begin the manual test suite?")
        if not ready:
            print("Manual test suite cancelled.")
            return {'status': 'cancelled'}
        
        start_time = time.time()
        
        # Run each manual test
        try:
            self.test_basic_health_workflow()
            self.test_memory_optimization_workflow()
            self.test_storage_analysis_workflow()
            self.test_optimization_safety_workflow()
            self.test_comprehensive_optimization_workflow()
            
        except KeyboardInterrupt:
            print("\n\nManual test suite interrupted by user.")
            return {'status': 'interrupted', 'partial_results': self.test_results}
        
        except Exception as e:
            print(f"\n\nManual test suite failed with error: {e}")
            return {'status': 'error', 'error': str(e), 'partial_results': self.test_results}
        
        # Generate summary
        total_tests = len(self.test_results)
        passed_tests = sum(1 for t in self.test_results if t.get('passed', False))
        
        print("\n" + "="*80)
        print("MANUAL TEST SUITE SUMMARY")
        print("="*80)
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success Rate: {(passed_tests / total_tests * 100):.1f}%" if total_tests > 0 else "0%")
        print(f"Duration: {(time.time() - start_time):.1f} seconds")
        
        # Show individual test results
        for test in self.test_results:
            status = "✅ PASSED" if test.get('passed', False) else "❌ FAILED"
            print(f"- {test['test_id']} {test['test_name']}: {status}")
        
        summary = {
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': total_tests - passed_tests,
            'success_rate': (passed_tests / total_tests * 100) if total_tests > 0 else 0,
            'duration_seconds': time.time() - start_time,
            'test_results': self.test_results
        }
        
        # Save results to file
        filename = f"manual_test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nDetailed results saved to: {filename}")
        print("="*80)
        
        return summary

def main():
    """Main entry point for manual testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Hardware Optimizer Manual Testing")
    parser.add_argument("--url", default="http://localhost:8116", help="Agent URL")
    parser.add_argument("--test", choices=["health", "memory", "storage", "safety", "comprehensive", "all"],
                       default="all", help="Specific test to run")
    
    args = parser.parse_args()
    
    tester = ManualTestProcedures(args.url)
    
    if args.test == "health":
        tester.test_basic_health_workflow()
    elif args.test == "memory":
        tester.test_memory_optimization_workflow()
    elif args.test == "storage":
        tester.test_storage_analysis_workflow()
    elif args.test == "safety":
        tester.test_optimization_safety_workflow()
    elif args.test == "comprehensive":
        tester.test_comprehensive_optimization_workflow()
    else:  # all
        tester.run_manual_test_suite()

if __name__ == "__main__":
    main()