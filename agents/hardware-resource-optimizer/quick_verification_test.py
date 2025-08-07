#!/usr/bin/env python3
"""
Quick Verification Test for Hardware Resource Optimizer
Tests core functionality to ensure everything works properly.
"""

import os
import requests
import tempfile
import json
import time
from pathlib import Path

BASE_URL = "http://localhost:8116"

def test_endpoint(method, endpoint, params=None, description=""):
    """Test a single endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        start_time = time.time()
        if method == "GET":
            response = requests.get(url, params=params, timeout=30)
        elif method == "POST":
            response = requests.post(url, params=params, timeout=60)
        else:
            return False, "Unsupported method"
            
        response_time = time.time() - start_time
        
        if response.status_code == 200:
            return True, {
                "status": "success",
                "data": response.json(),
                "response_time": response_time,
                "description": description
            }
        else:
            return False, {
                "status": "error",
                "status_code": response.status_code,
                "text": response.text,
                "response_time": response_time,
                "description": description
            }
            
    except Exception as e:
        return False, {"error": str(e), "description": description}

def main():
    """Run quick verification tests"""
    print("🔍 Quick Verification Test for Hardware Resource Optimizer")
    print("=" * 60)
    
    # Test cases
    tests = [
        ("GET", "/health", None, "Health check"),
        ("GET", "/status", None, "Status check"),
        ("POST", "/optimize/memory", {"dry_run": "true"}, "Memory optimization (dry run)"),
        ("POST", "/optimize/cpu", None, "CPU optimization"),
        ("POST", "/optimize/disk", None, "Disk optimization"),
        ("POST", "/optimize/docker", None, "Docker optimization"),
        ("GET", "/analyze/storage", {"path": "/tmp", "limit": "5"}, "Storage analysis"),
        ("POST", "/optimize/storage/cache", {"path": "/tmp"}, "Cache cleanup"),
        ("POST", "/optimize/all", None, "Comprehensive optimization")
    ]
    
    results = []
    passed = 0
    failed = 0
    
    for method, endpoint, params, description in tests:
        print(f"\n🧪 Testing: {description}")
        print(f"   {method} {endpoint}")
        
        success, result = test_endpoint(method, endpoint, params, description)
        
        if success:
            print(f"   ✅ PASS ({result['response_time']:.3f}s)")
            passed += 1
        else:
            print(f"   ❌ FAIL - {result}")
            failed += 1
            
        results.append({
            "test": description,
            "method": method,
            "endpoint": endpoint,
            "params": params,
            "success": success,
            "result": result
        })
    
    # Create test files for duplicate testing
    print(f"\n🧪 Testing: Duplicate file detection and cleanup")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create duplicate files
        test_content = b"This is test content for duplicates"
        for i in range(3):
            with open(f"{temp_dir}/file_{i}.txt", "wb") as f:
                f.write(test_content)
            with open(f"{temp_dir}/duplicate_{i}.txt", "wb") as f:
                f.write(test_content)
                
        files_before = len(os.listdir(temp_dir))
        
        # Test duplicate detection
        success, result = test_endpoint("GET", "/analyze/storage/duplicates", 
                                       {"path": temp_dir}, "Duplicate detection")
        
        if success:
            print(f"   ✅ Duplicate detection PASS")
            duplicates_found = len(result['data'].get('duplicate_groups', []))
            print(f"      Found {duplicates_found} duplicate groups")
        else:
            print(f"   ❌ Duplicate detection FAIL")
            
        # Test duplicate cleanup
        success, result = test_endpoint("POST", "/optimize/storage/duplicates", 
                                       {"path": temp_dir}, "Duplicate cleanup")
        
        files_after = len(os.listdir(temp_dir))
        
        if success:
            print(f"   ✅ Duplicate cleanup PASS")
            print(f"      Files before: {files_before}, after: {files_after}")
            print(f"      Files removed: {files_before - files_after}")
        else:
            print(f"   ❌ Duplicate cleanup FAIL - {result}")
            
        results.append({
            "test": "Duplicate file operations",
            "files_before": files_before,
            "files_after": files_after,
            "files_removed": files_before - files_after,
            "success": success
        })
    
    # Summary
    total_tests = len(tests) + 1  # +1 for duplicate test
    print(f"\n" + "=" * 60)
    print(f"📊 QUICK VERIFICATION RESULTS")
    print(f"=" * 60)
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed + (1 if success else 0)}")
    print(f"Failed: {failed + (0 if success else 1)}")
    print(f"Pass Rate: {((passed + (1 if success else 0)) / total_tests) * 100:.1f}%")
    
    if passed == len(tests) and success:
        print(f"\n🎉 ALL TESTS PASSED! Hardware Resource Optimizer is working perfectly!")
        return True
    else:
        print(f"\n⚠️  Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)