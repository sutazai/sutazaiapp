#!/usr/bin/env python3
"""
Test script for the enhanced storage optimization endpoints
"""

import requests
import json
import time

BASE_URL = "http://localhost:8116"

def test_endpoint(method, endpoint, params=None, expected_status=200):
    """Test an endpoint and return the result"""
    try:
        if method.upper() == "GET":
            response = requests.get(f"{BASE_URL}{endpoint}", params=params)
        elif method.upper() == "POST":
            response = requests.post(f"{BASE_URL}{endpoint}", params=params)
        else:
            return {"error": f"Unsupported method: {method}"}
        
        if response.status_code == expected_status:
            return {"success": True, "data": response.json()}
        else:
            return {"error": f"Status {response.status_code}, expected {expected_status}"}
    except Exception as e:
        return {"error": str(e)}

def main():
    """Test all storage optimization endpoints"""
    print("Testing Enhanced Hardware Resource Optimizer Storage Features")
    print("=" * 60)
    
    # Test storage analysis endpoints
    analysis_tests = [
        ("GET", "/analyze/storage", {"path": "/tmp"}),
        ("GET", "/analyze/storage/duplicates", {"path": "/tmp"}),
        ("GET", "/analyze/storage/large-files", {"path": "/var/log", "min_size_mb": 1}),
        ("GET", "/analyze/storage/report", None),
    ]
    
    print("\n1. Testing Storage Analysis Endpoints:")
    for method, endpoint, params in analysis_tests:
        print(f"   Testing {method} {endpoint}...", end=" ")
        result = test_endpoint(method, endpoint, params)
        if result.get("success"):
            print("✓ PASS")
        else:
            print(f"✗ FAIL: {result.get('error')}")
    
    # Test storage optimization endpoints (dry-run mode)
    optimization_tests = [
        ("POST", "/optimize/storage", {"dry_run": True}),
        ("POST", "/optimize/storage/duplicates", {"path": "/tmp", "dry_run": True}),
        ("POST", "/optimize/storage/cache", None),
        ("POST", "/optimize/storage/compress", {"path": "/var/log", "days_old": 365}),
        ("POST", "/optimize/storage/logs", None),
    ]
    
    print("\n2. Testing Storage Optimization Endpoints:")
    for method, endpoint, params in optimization_tests:
        print(f"   Testing {method} {endpoint}...", end=" ")
        result = test_endpoint(method, endpoint, params)
        if result.get("success"):
            print("✓ PASS")
        else:
            print(f"✗ FAIL: {result.get('error')}")
    
    # Test enhanced "optimize all"
    print("\n3. Testing Enhanced Comprehensive Optimization:")
    print("   Testing POST /optimize/all...", end=" ")
    result = test_endpoint("POST", "/optimize/all")
    if result.get("success"):
        data = result["data"]
        if "storage" in data.get("detailed_results", {}):
            print("✓ PASS (includes storage optimization)")
        else:
            print("✗ FAIL: Storage optimization not included")
    else:
        print(f"✗ FAIL: {result.get('error')}")
    
    # Performance test
    print("\n4. Performance Test:")
    print("   Testing response times...", end=" ")
    start_time = time.time()
    result = test_endpoint("GET", "/analyze/storage/report")
    end_time = time.time()
    
    if result.get("success"):
        response_time = end_time - start_time
        if response_time < 5.0:  # Should respond within 5 seconds
            print(f"✓ PASS (Response time: {response_time:.2f}s)")
        else:
            print(f"⚠ SLOW (Response time: {response_time:.2f}s)")
    else:
        print(f"✗ FAIL: {result.get('error')}")
    
    print("\n5. Feature Verification:")
    # Test that safety features are working
    print("   Testing safety protections...", end=" ")
    result = test_endpoint("GET", "/analyze/storage", {"path": "/etc"})
    if result.get("success"):
        data = result["data"]
        if data.get("status") == "error" and "not accessible or safe" in data.get("error", ""):
            print("✓ PASS (Protected paths blocked)")
        else:
            print("⚠ WARNING: Protected path analysis allowed")
    else:
        print(f"✗ FAIL: {result.get('error')}")
    
    print("\n" + "=" * 60)
    print("Storage optimization enhancement testing complete!")

if __name__ == "__main__":
    main()