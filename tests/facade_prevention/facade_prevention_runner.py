#!/usr/bin/env python3
"""
Facade Prevention Test Runner - CI/CD Integration
=================================================

This module provides a comprehensive test runner for facade prevention tests.
It integrates with CI/CD pipelines to prevent facade implementations from being deployed.

CRITICAL PURPOSE: Provide a single entry point for all facade prevention tests
that can be integrated into CI/CD pipelines to catch facade implementations before deployment.
"""

import asyncio
import sys
import json
import time
import argparse
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
import logging

# Import all facade prevention test modules
from test_service_mesh_reality import ServiceMeshRealityTester
from test_mcp_reality import MCPRealityTester
from test_container_health_reality import ContainerHealthRealityTester
from test_port_registry_reality import PortRegistryRealityTester
from test_api_functionality_reality import APIFunctionalityRealityTester
from test_end_to_end_workflows import EndToEndWorkflowRealityTester

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FacadePreventionRunner:
    """
    Master test runner for facade prevention.
    
    This runner executes all facade prevention tests and provides comprehensive
    results for CI/CD integration.
    """
    
    def __init__(self, base_url: str = "http://localhost:10010", frontend_url: str = "http://localhost:10011"):
        self.base_url = base_url
        self.frontend_url = frontend_url
        self.test_suites = {
            "service_mesh": {
                "class": ServiceMeshRealityTester,
                "description": "Service mesh reality and functionality tests",
                "critical": True,
                "timeout": 300  # 5 minutes
            },
            "mcp_servers": {
                "class": MCPRealityTester,
                "description": "MCP server reality and connectivity tests",
                "critical": True,
                "timeout": 600  # 10 minutes
            },
            "container_health": {
                "class": ContainerHealthRealityTester,
                "description": "Container health and orphan detection tests",
                "critical": True,
                "timeout": 300  # 5 minutes
            },
            "port_registry": {
                "class": PortRegistryRealityTester,
                "description": "Port registry accuracy and availability tests",
                "critical": False,
                "timeout": 180  # 3 minutes
            },
            "api_functionality": {
                "class": APIFunctionalityRealityTester,
                "description": "API functionality reality tests",
                "critical": True,
                "timeout": 300  # 5 minutes
            },
            "end_to_end_workflows": {
                "class": EndToEndWorkflowRealityTester,
                "description": "End-to-end workflow reality tests",
                "critical": True,
                "timeout": 600  # 10 minutes
            }
        }
    
    async def run_test_suite(self, suite_name: str, suite_config: Dict) -> Dict:
        """Run a single test suite."""
        logger.info(f"🧪 Running test suite: {suite_name}")
        
        suite_result = {
            "suite_name": suite_name,
            "description": suite_config["description"],
            "critical": suite_config["critical"],
            "start_time": datetime.now().isoformat(),
            "timeout": suite_config["timeout"]
        }
        
        try:
            # Initialize tester class
            tester_class = suite_config["class"]
            
            # Handle different initialization patterns
            if suite_name == "mcp_servers":
                tester = tester_class()
                test_result = await asyncio.wait_for(
                    tester.run_comprehensive_mcp_facade_tests(),
                    timeout=suite_config["timeout"]
                )
            elif suite_name == "port_registry":
                tester = tester_class()
                test_result = await asyncio.wait_for(
                    tester.run_comprehensive_port_tests(),
                    timeout=suite_config["timeout"]
                )
            elif suite_name in ["service_mesh", "container_health", "api_functionality", "end_to_end_workflows"]:
                if suite_name == "end_to_end_workflows":
                    async with tester_class(self.base_url, self.frontend_url) as tester:
                        test_result = await asyncio.wait_for(
                            tester.run_comprehensive_workflow_tests(),
                            timeout=suite_config["timeout"]
                        )
                else:
                    async with tester_class(self.base_url) as tester:
                        if suite_name == "service_mesh":
                            test_result = await asyncio.wait_for(
                                tester.run_comprehensive_facade_tests(),
                                timeout=suite_config["timeout"]
                            )
                        elif suite_name == "container_health":
                            test_result = await asyncio.wait_for(
                                tester.run_comprehensive_container_tests(),
                                timeout=suite_config["timeout"]
                            )
                        elif suite_name == "api_functionality":
                            test_result = await asyncio.wait_for(
                                tester.run_comprehensive_api_tests(),
                                timeout=suite_config["timeout"]
                            )
            else:
                raise ValueError(f"Unknown test suite: {suite_name}")
            
            # Process results
            suite_result.update({
                "end_time": datetime.now().isoformat(),
                "status": "completed",
                "test_result": test_result,
                "suite_passed": test_result.get("overall_status") == "passed",
                "facade_issues": test_result.get("facade_issues_detected", 0)
            })
            
        except asyncio.TimeoutError:
            suite_result.update({
                "end_time": datetime.now().isoformat(),
                "status": "timeout",
                "error": f"Test suite timed out after {suite_config['timeout']} seconds",
                "suite_passed": False,
                "facade_issues": 999  # High number to indicate failure
            })
            
        except Exception as e:
            logger.error(f"Test suite {suite_name} failed: {e}")
            suite_result.update({
                "end_time": datetime.now().isoformat(),
                "status": "error",
                "error": str(e),
                "suite_passed": False,
                "facade_issues": 999  # High number to indicate failure
            })
        
        return suite_result
    
    async def run_all_tests(self, suite_filter: Optional[List[str]] = None, fail_fast: bool = False) -> Dict:
        """Run all facade prevention tests."""
        logger.info("🚀 Starting comprehensive facade prevention test suite")
        
        start_time = datetime.now()
        
        # Filter test suites if specified
        if suite_filter:
            filtered_suites = {k: v for k, v in self.test_suites.items() if k in suite_filter}
        else:
            filtered_suites = self.test_suites
        
        results = {
            "test_run": "facade_prevention_comprehensive",
            "timestamp": start_time.isoformat(),
            "suite_filter": suite_filter,
            "fail_fast": fail_fast,
            "suites": {}
        }
        
        passed_suites = 0
        failed_suites = 0
        critical_failures = 0
        total_facade_issues = 0
        
        # Run test suites
        for suite_name, suite_config in filtered_suites.items():
            suite_result = await self.run_test_suite(suite_name, suite_config)
            results["suites"][suite_name] = suite_result
            
            if suite_result["suite_passed"]:
                passed_suites += 1
                logger.info(f"✅ {suite_name} passed")
            else:
                failed_suites += 1
                logger.error(f"❌ {suite_name} failed")
                
                if suite_config["critical"]:
                    critical_failures += 1
                
                if fail_fast:
                    logger.error(f"Stopping execution due to fail_fast and failure in {suite_name}")
                    break
            
            total_facade_issues += suite_result.get("facade_issues", 0)
        
        # Calculate overall results
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        total_suites = len(filtered_suites)
        
        # Determine overall status
        overall_status = "passed"
        if critical_failures > 0:
            overall_status = "critical_failure"
        elif total_facade_issues > 0:
            overall_status = "facade_detected"
        elif failed_suites > 0:
            overall_status = "failed"
        
        results.update({
            "end_time": end_time.isoformat(),
            "duration_seconds": duration,
            "summary": {
                "total_suites": total_suites,
                "passed_suites": passed_suites,
                "failed_suites": failed_suites,
                "critical_failures": critical_failures,
                "total_facade_issues": total_facade_issues,
                "success_rate": passed_suites / total_suites if total_suites > 0 else 0
            },
            "overall_status": overall_status,
            "deployment_safe": overall_status == "passed"
        })
        
        return results
    
    def generate_ci_report(self, results: Dict, output_file: Optional[str] = None) -> str:
        """Generate CI/CD friendly report."""
        report = {
            "facade_prevention_report": {
                "version": "1.0",
                "timestamp": results["timestamp"],
                "duration_seconds": results.get("duration_seconds", 0),
                "overall_status": results["overall_status"],
                "deployment_safe": results["deployment_safe"],
                "summary": results["summary"],
                "critical_issues": [],
                "facade_issues": [],
                "recommendations": []
            }
        }
        
        # Extract critical issues
        for suite_name, suite_result in results["suites"].items():
            if not suite_result["suite_passed"] and suite_result.get("critical", False):
                report["facade_prevention_report"]["critical_issues"].append({
                    "suite": suite_name,
                    "description": suite_result["description"],
                    "error": suite_result.get("error", "Test failed"),
                    "facade_issues": suite_result.get("facade_issues", 0)
                })
            
            if suite_result.get("facade_issues", 0) > 0:
                report["facade_prevention_report"]["facade_issues"].append({
                    "suite": suite_name,
                    "facade_count": suite_result["facade_issues"],
                    "details": suite_result.get("test_result", {})
                })
        
        # Generate recommendations
        if results["overall_status"] != "passed":
            if results["summary"]["critical_failures"] > 0:
                report["facade_prevention_report"]["recommendations"].append(
                    "CRITICAL: Fix critical test failures before deployment"
                )
            if results["summary"]["total_facade_issues"] > 0:
                report["facade_prevention_report"]["recommendations"].append(
                    "FACADE DETECTED: Review and fix facade implementations"
                )
            if results["summary"]["failed_suites"] > 0:
                report["facade_prevention_report"]["recommendations"].append(
                    "Fix failed test suites to ensure system reliability"
                )
        else:
            report["facade_prevention_report"]["recommendations"].append(
                "All facade prevention tests passed - deployment is safe"
            )
        
        # Save report if output file specified
        if output_file:
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"CI report saved to {output_file}")
        
        return json.dumps(report, indent=2)
    
    def print_summary(self, results: Dict):
        """Print a human-readable summary."""
        print("\n" + "="*80)
        print("🛡️  FACADE PREVENTION TEST RESULTS")
        print("="*80)
        
        summary = results["summary"]
        status = results["overall_status"]
        
        # Overall status
        status_emoji = {
            "passed": "✅",
            "failed": "❌", 
            "critical_failure": "🚨",
            "facade_detected": "⚠️"
        }
        
        print(f"\n{status_emoji.get(status, '❓')} Overall Status: {status.upper()}")
        print(f"📊 Success Rate: {summary['success_rate']:.1%}")
        print(f"⏱️  Duration: {results.get('duration_seconds', 0):.1f} seconds")
        
        # Suite breakdown
        print(f"\n📋 Test Suite Breakdown:")
        print(f"   Total Suites: {summary['total_suites']}")
        print(f"   Passed: {summary['passed_suites']}")
        print(f"   Failed: {summary['failed_suites']}")
        print(f"   Critical Failures: {summary['critical_failures']}")
        
        # Facade detection
        if summary['total_facade_issues'] > 0:
            print(f"\n⚠️  FACADE ISSUES DETECTED: {summary['total_facade_issues']}")
        else:
            print(f"\n✅ No facade implementations detected")
        
        # Individual suite results
        print(f"\n📝 Individual Suite Results:")
        for suite_name, suite_result in results["suites"].items():
            status_icon = "✅" if suite_result["suite_passed"] else "❌"
            critical_icon = "🚨" if suite_result.get("critical", False) and not suite_result["suite_passed"] else ""
            facade_count = suite_result.get("facade_issues", 0)
            facade_info = f" (🎭 {facade_count} facade issues)" if facade_count > 0 else ""
            
            print(f"   {status_icon} {critical_icon} {suite_name}: {suite_result['description']}{facade_info}")
            
            if not suite_result["suite_passed"]:
                error = suite_result.get("error", "Unknown error")
                print(f"      Error: {error}")
        
        # Deployment recommendation
        print(f"\n🚀 Deployment Recommendation:")
        if results["deployment_safe"]:
            print("   ✅ DEPLOYMENT IS SAFE - All facade prevention tests passed")
        else:
            print("   ❌ DEPLOYMENT IS NOT SAFE - Fix issues before deploying")
            if summary['critical_failures'] > 0:
                print("      🚨 Critical failures detected")
            if summary['total_facade_issues'] > 0:
                print("      🎭 Facade implementations detected")
        
        print("="*80)


async def main():
    """Main entry point for CI/CD integration."""
    parser = argparse.ArgumentParser(description="Facade Prevention Test Runner")
    parser.add_argument("--suites", nargs="+", help="Specific test suites to run")
    parser.add_argument("--fail-fast", action="store_true", help="Stop on first failure")
    parser.add_argument("--base-url", default="http://localhost:10010", help="Backend base URL")
    parser.add_argument("--frontend-url", default="http://localhost:10011", help="Frontend URL")
    parser.add_argument("--output", help="Output file for CI report")
    parser.add_argument("--quiet", action="store_true", help="Quiet mode - output")
    parser.add_argument("--json-only", action="store_true", help="Output only JSON results")
    
    args = parser.parse_args()
    
    if args.quiet:
        logging.getLogger().setLevel(logging.WARNING)
    
    # Initialize runner
    runner = FacadePreventionRunner(args.base_url, args.frontend_url)
    
    # Run tests
    try:
        results = await runner.run_all_tests(
            suite_filter=args.suites,
            fail_fast=args.fail_fast
        )
        
        # Generate CI report
        ci_report = runner.generate_ci_report(results, args.output)
        
        if args.json_only:
            print(json.dumps(results, indent=2))
        else:
            # Print summary
            runner.print_summary(results)
            
            if not args.quiet:
                print(f"\n📄 CI Report:")
                print(ci_report)
        
        # Exit with appropriate code
        if results["deployment_safe"]:
            sys.exit(0)
        else:
            if results["summary"]["critical_failures"] > 0:
                sys.exit(2)  # Critical failure
            elif results["summary"]["total_facade_issues"] > 0:
                sys.exit(3)  # Facade detected
            else:
                sys.exit(1)  # General failure
                
    except KeyboardInterrupt:
        logger.error("Test execution interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Test execution failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())