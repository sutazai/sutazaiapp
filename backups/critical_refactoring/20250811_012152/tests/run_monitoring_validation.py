#!/usr/bin/env python3
"""
SutazAI Monitoring System Validation Runner
==========================================

Comprehensive validation script that runs all monitoring system tests
and generates a complete validation report.
"""

import sys
import os
import subprocess
import json
from datetime import datetime
from pathlib import Path


def run_test_suite(test_file, description):
    """Run a test suite and return results"""
    print(f"\n🧪 Running {description}...")
    print("-" * 50)
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=120
        )
        
        success = result.returncode == 0
        
        return {
            "test_file": test_file,
            "description": description,
            "success": success,
            "return_code": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr
        }
        
    except subprocess.TimeoutExpired:
        return {
            "test_file": test_file,
            "description": description,
            "success": False,
            "return_code": -1,
            "error": "Test timed out after 120 seconds"
        }
    except Exception as e:
        return {
            "test_file": test_file,
            "description": description,
            "success": False,
            "return_code": -1,
            "error": str(e)
        }


def run_validation():
    """Run complete validation suite"""
    print("🔍 SutazAI Monitoring System - Complete Validation Suite")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    
    # Define test suites
    test_suites = [
        ("tests/test_agent_detection_validation.py", "Agent Detection and Validation Tests"),
        ("tests/test_live_monitoring_validation.py", "Live System Validation Tests")
    ]
    
    results = []
    
    # Run each test suite
    for test_file, description in test_suites:
        if os.path.exists(test_file):
            result = run_test_suite(test_file, description)
            results.append(result)
            
            # Print immediate results
            if result["success"]:
                print("✅ PASSED")
            else:
                print("❌ FAILED")
                if "error" in result:
                    print(f"Error: {result['error']}")
        else:
            print(f"⚠️  Test file not found: {test_file}")
            results.append({
                "test_file": test_file,
                "description": description,
                "success": False,
                "error": "Test file not found"
            })
    
    # Generate summary
    total_tests = len(results)
    passed_tests = len([r for r in results if r["success"]])
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    
    print("\n" + "=" * 60)
    print("📊 VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Test Suites Run: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {total_tests - passed_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Show individual results
    for result in results:
        status_icon = "✅" if result["success"] else "❌"
        print(f"\n{status_icon} {result['description']}")
        if not result["success"]:
            if "error" in result:
                print(f"  Error: {result['error']}")
            elif result.get("stderr"):
                print(f"  Error: {result['stderr'][:200]}...")
    
    # Overall assessment
    print(f"\n🎯 OVERALL ASSESSMENT")
    if success_rate == 100:
        print("🟢 PERFECT - All validation tests passed!")
        assessment = "EXCELLENT"
    elif success_rate >= 80:
        print("🟡 GOOD - Most tests passed with minor issues")
        assessment = "GOOD"
    elif success_rate >= 60:
        print("🟠 FAIR - Some significant issues need attention")
        assessment = "FAIR"
    else:
        print("🔴 POOR - Major issues require immediate attention")
        assessment = "POOR"
    
    # Save detailed results
    report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total_test_suites": total_tests,
            "passed_test_suites": passed_tests,
            "failed_test_suites": total_tests - passed_tests,
            "success_rate": success_rate,
            "assessment": assessment
        },
        "detailed_results": results
    }
    
    report_file = f"tests/validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📄 Detailed validation report saved to: {report_file}")
    
    # Check if monitoring files exist and are accessible
    print(f"\n🔍 MONITORING SYSTEM FILES CHECK")
    print("-" * 40)
    
    monitoring_files = [
        "/opt/sutazaiapp/scripts/monitoring/static_monitor.py",
        "/opt/sutazaiapp/scripts/agents/quick-status-check.py", 
        "/opt/sutazaiapp/agents/communication_config.json"
    ]
    
    for file_path in monitoring_files:
        if os.path.exists(file_path):
            size = os.path.getsize(file_path)
            print(f"✅ {file_path} ({size:,} bytes)")
        else:
            print(f"❌ {file_path} (missing)")
    
    # Quick system status
    print(f"\n🐳 QUICK SYSTEM STATUS")
    print("-" * 40)
    
    try:
        docker_result = subprocess.run(
            ["docker", "ps", "--filter", "name=sutazai-", "--format", "table {{.Names}}\t{{.Status}}"],
            capture_output=True, text=True, timeout=10
        )
        
        if docker_result.returncode == 0:
            lines = docker_result.stdout.strip().split('\n')
            container_count = len(lines) - 1  # Subtract header
            print(f"SutazAI Containers: {container_count}")
            
            # Count by status
            statuses = {"running": 0, "restarting": 0, "exited": 0, "other": 0}
            for line in lines[1:]:  # Skip header
                if "Up" in line:
                    statuses["running"] += 1
                elif "Restarting" in line:
                    statuses["restarting"] += 1
                elif "Exited" in line:
                    statuses["exited"] += 1
                else:
                    statuses["other"] += 1
            
            for status, count in statuses.items():
                if count > 0:
                    print(f"  {status.title()}: {count}")
        else:
            print("⚠️  Could not get Docker container status")
            
    except Exception as e:
        print(f"⚠️  Error checking Docker status: {e}")
    
    return success_rate >= 80


def main():
    """Main validation runner"""
    try:
        success = run_validation()
        
        print(f"\n{'='*60}")
        if success:
            print("🎉 MONITORING SYSTEM VALIDATION COMPLETED SUCCESSFULLY!")
        else:
            print("⚠️  MONITORING SYSTEM VALIDATION COMPLETED WITH ISSUES")
        print(f"{'='*60}")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Validation interrupted by user")
        return 2
    except Exception as e:
        print(f"\n\n❌ Validation failed with error: {e}")
        return 3


if __name__ == "__main__":
    sys.exit(main())