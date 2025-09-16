#!/usr/bin/env python3
"""
Test runner script for Sutazai AI Application
Runs all tests and generates coverage report
"""

import sys
import subprocess
import os
from pathlib import Path
import json
import time

def check_services():
    """Check if required services are running"""
    import requests
    
    services = {
        "Backend API": "http://localhost:10200/api/v1/health",
        "Frontend": "http://localhost:11000",
    }
    
    print("=" * 60)
    print("CHECKING SERVICES")
    print("=" * 60)
    
    all_healthy = True
    for name, url in services.items():
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                print(f"✅ {name}: RUNNING")
            else:
                print(f"⚠️  {name}: UNHEALTHY (status: {response.status_code})")
                all_healthy = False
        except Exception as e:
            print(f"❌ {name}: NOT RUNNING ({str(e)})")
            all_healthy = False
    
    print()
    return all_healthy

def install_test_dependencies():
    """Install test dependencies if needed"""
    print("=" * 60)
    print("INSTALLING TEST DEPENDENCIES")
    print("=" * 60)
    
    dependencies = [
        "pytest",
        "pytest-cov",
        "pytest-asyncio",
        "pytest-timeout",
        "pytest-html",
        "requests",
        "websocket-client",
        "aiohttp",
    ]
    
    for dep in dependencies:
        print(f"Installing {dep}...")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", dep])
    
    print("✅ Test dependencies installed\n")

def run_tests():
    """Run the test suite with coverage"""
    print("=" * 60)
    print("RUNNING TEST SUITE")
    print("=" * 60)
    
    test_dir = Path(__file__).parent / "tests"
    
    # Pytest arguments
    pytest_args = [
        sys.executable, "-m", "pytest",
        str(test_dir),
        "-v",  # Verbose output
        "--tb=short",  # Short traceback
        "--cov=.",  # Coverage for current directory
        "--cov-report=html",  # HTML coverage report
        "--cov-report=term-missing",  # Terminal coverage with missing lines
        "--cov-report=json",  # JSON coverage for parsing
        "--html=test_report.html",  # HTML test report
        "--self-contained-html",  # Include CSS/JS in HTML
        "--timeout=60",  # 60 second timeout per test
        "--maxfail=10",  # Stop after 10 failures
        "-W", "ignore::DeprecationWarning",  # Ignore deprecation warnings
    ]
    
    # Run tests
    result = subprocess.run(pytest_args, capture_output=False, text=True)
    
    return result.returncode

def generate_summary():
    """Generate test summary from results"""
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    # Try to load coverage data
    coverage_file = Path("coverage.json")
    if coverage_file.exists():
        with open(coverage_file) as f:
            coverage_data = json.load(f)
            
        total_coverage = coverage_data.get("totals", {}).get("percent_covered", 0)
        print(f"\n📊 OVERALL COVERAGE: {total_coverage:.1f}%")
        
        # Show file coverage
        files = coverage_data.get("files", {})
        if files:
            print("\n📁 FILE COVERAGE:")
            for file_path, file_data in sorted(files.items())[:10]:  # Top 10 files
                if "test" not in file_path:  # Skip test files
                    coverage = file_data["summary"]["percent_covered"]
                    file_name = Path(file_path).name
                    print(f"  - {file_name}: {coverage:.1f}%")
    
    # Check for HTML report
    html_report = Path("htmlcov/index.html")
    if html_report.exists():
        print(f"\n📋 HTML Coverage Report: {html_report.absolute()}")
    
    test_report = Path("test_report.html")
    if test_report.exists():
        print(f"📋 HTML Test Report: {test_report.absolute()}")
    
    print("\n" + "=" * 60)

def run_specific_test_category(category):
    """Run specific category of tests"""
    print(f"\n🔍 Running {category} tests...")
    
    markers = {
        "integration": "-m integration",
        "e2e": "-m e2e",
        "websocket": "-m websocket",
        "agent": "-m agent",
        "performance": "-m performance",
    }
    
    if category in markers:
        pytest_args = [
            sys.executable, "-m", "pytest",
            "tests",
            markers[category],
            "-v",
            "--tb=short",
        ]
        subprocess.run(pytest_args)

def main():
    """Main test runner"""
    print("🚀 SUTAZAI AI APPLICATION TEST RUNNER")
    print("=" * 60)
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print()
    
    # Check services
    if not check_services():
        print("⚠️  WARNING: Some services are not running!")
        print("Tests may fail. Continue anyway? (y/n): ", end="")
        response = input().strip().lower()
        if response != 'y':
            print("Exiting...")
            return 1
    
    # Install dependencies
    install_test_dependencies()
    
    # Run full test suite
    print("\n🏃 Running full test suite...\n")
    exit_code = run_tests()
    
    # Generate summary
    generate_summary()
    
    # Print final result
    if exit_code == 0:
        print("\n✅ ALL TESTS PASSED!")
    else:
        print(f"\n❌ TESTS FAILED (exit code: {exit_code})")
    
    # Offer to run specific categories
    print("\n📝 Run specific test category? (or press Enter to exit)")
    print("Options: integration, e2e, websocket, agent, performance")
    category = input("Category: ").strip().lower()
    
    if category:
        run_specific_test_category(category)
    
    return exit_code

if __name__ == "__main__":
    sys.exit(main())