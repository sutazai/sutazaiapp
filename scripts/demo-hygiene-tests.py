#!/usr/bin/env python3
"""
Purpose: Demonstration script showing comprehensive hygiene testing system
Usage: python demo-hygiene-tests.py [--quick]
Requirements: All hygiene testing components
"""

import sys
import subprocess
import time
from pathlib import Path

def main():
    project_root = Path("/opt/sutazaiapp")
    
    print("🧼 HYGIENE ENFORCEMENT SYSTEM TESTING DEMONSTRATION")
    print("=" * 60)
    
    # Quick mode for faster demo
    quick_mode = "--quick" in sys.argv
    
    if quick_mode:
        print("🚀 Running in QUICK mode...")
    else:
        print("🔍 Running COMPREHENSIVE tests...")
        
    print()
    
    # 1. Test Master Test Runner
    print("1️⃣  Testing Master Test Runner")
    print("-" * 40)
    
    cmd = [sys.executable, str(project_root / "scripts/test-hygiene-system.py"), "--setup-only"]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode == 0:
        print("✅ Master test runner: WORKING")
    else:
        print("❌ Master test runner: FAILED")
        print(result.stderr[:200])
    
    print()
    
    # 2. Test Deployment Validator
    print("2️⃣  Testing Deployment Validator")
    print("-" * 40)
    
    cmd = [str(project_root / "scripts/validate-hygiene-deployment.sh"), "--environment=dev"]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
    
    # Count successes and warnings
    output_lines = result.stdout.split('\n')
    successes = len([line for line in output_lines if "SUCCESS" in line])
    warnings = len([line for line in output_lines if "WARN" in line])
    errors = len([line for line in output_lines if "ERROR" in line])
    
    print(f"✅ Deployment validation: {successes} passed, {warnings} warnings, {errors} errors")
    
    if result.returncode == 0:
        print("✅ Overall deployment validation: PASSED")
    else:
        print("⚠️  Overall deployment validation: ISSUES FOUND")
    
    print()
    
    # 3. Test Individual Components (if not quick mode)
    if not quick_mode:
        print("3️⃣  Testing Individual Components")
        print("-" * 40)
        
        components = ["orchestrator", "coordinator", "hooks", "monitoring"]
        
        for component in components:
            cmd = [sys.executable, str(project_root / "scripts/test-hygiene-system.py"), 
                   "--component", component]
            
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
                if result.returncode == 0:
                    print(f"✅ {component.capitalize()}: WORKING")
                else:
                    print(f"⚠️  {component.capitalize()}: ISSUES")
            except subprocess.TimeoutExpired:
                print(f"⏰ {component.capitalize()}: TIMEOUT")
        
        print()
    
    # 4. Test Performance (basic)
    print("4️⃣  Testing Performance")
    print("-" * 40)
    
    start_time = time.time()
    
    # Simple performance test - file scanning
    test_files = list(project_root.glob("**/*.py"))
    scan_time = time.time() - start_time
    
    print(f"📁 File scanning: {len(test_files)} Python files in {scan_time:.2f}s")
    
    if scan_time < 5.0:
        print("✅ Performance: GOOD")
    elif scan_time < 10.0:
        print("⚠️  Performance: ACCEPTABLE")
    else:
        print("❌ Performance: SLOW")
    
    print()
    
    # 5. Test Directory Structure
    print("5️⃣  Testing Directory Structure")
    print("-" * 40)
    
    required_paths = [
        "scripts/test-hygiene-system.py",
        "scripts/validate-hygiene-deployment.sh",
        "tests/hygiene/__init__.py",
        "tests/hygiene/test_orchestrator.py",
        "tests/hygiene/test_coordinator.py",
        "tests/hygiene/test_git_hooks.py",
        "tests/hygiene/test_monitoring.py",
        "tests/hygiene/test_fixtures.py",
        "tests/hygiene/test_failure_scenarios.py",
        "tests/hygiene/test_performance.py",
        "docs/hygiene-testing-guide.md"
    ]
    
    missing_count = 0
    for path in required_paths:
        full_path = project_root / path
        if full_path.exists():
            print(f"✅ {path}")
        else:
            print(f"❌ {path} - MISSING")
            missing_count += 1
    
    if missing_count == 0:
        print("✅ All required files present")
    else:
        print(f"⚠️  {missing_count} files missing")
    
    print()
    
    # 6. Summary
    print("6️⃣  SUMMARY")
    print("-" * 40)
    
    print("📋 Test Suite Components:")
    print("   • Master Test Runner - Comprehensive test orchestration")
    print("   • Unit Tests - Individual component validation")
    print("   • Integration Tests - Cross-component testing")
    print("   • Performance Tests - Resource usage and timing")
    print("   • Failure Scenario Tests - Error handling validation")
    print("   • Deployment Validator - Production readiness checks")
    print()
    
    print("🎯 Coverage Areas:")
    print("   • Agent Orchestrator")
    print("   • Enforcement Coordinator")
    print("   • Git Hooks")
    print("   • Monitoring System")
    print("   • Automated Maintenance")
    print("   • Test Fixtures and Mocks")
    print()
    
    print("🚀 Usage Examples:")
    print("   # Run all tests:")
    print("   python3 scripts/test-hygiene-system.py")
    print()
    print("   # Test specific component:")
    print("   python3 scripts/test-hygiene-system.py --component orchestrator")
    print()
    print("   # Validate deployment:")
    print("   scripts/validate-hygiene-deployment.sh --environment=prod")
    print()
    print("   # Run performance tests:")
    print("   python3 scripts/test-hygiene-system.py --component performance")
    print()
    
    print("✨ HYGIENE TESTING SYSTEM DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())