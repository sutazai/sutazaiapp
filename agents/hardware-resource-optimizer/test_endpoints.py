#!/usr/bin/env python3
"""
Test script for Hardware Resource Optimizer endpoints
Tests all endpoints to ensure they work correctly
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import requests
import json
import time
import sys
from typing import Dict, Any

BASE_URL = "http://localhost:8116"

def test_endpoint(endpoint: str, method: str = "GET", expected_status: int = 200) -> Dict[str, Any]:
    """Test a specific endpoint"""
    url = f"{BASE_URL}{endpoint}"
    
    try:
        if method == "GET":
            response = requests.get(url, timeout=10)
        elif method == "POST":
            response = requests.post(url, timeout=30)
        else:
            return {"success": False, "error": f"Unsupported method: {method}"}
        
        success = response.status_code == expected_status
        
        try:
            data = response.json()
        except (ValueError, TypeError, KeyError, AttributeError) as e:
            # TODO: Review this exception handling
            logger.error(f"Unexpected exception: {e}", exc_info=True)
            data = response.text
        
        return {
            "success": success,
            "status_code": response.status_code,
            "data": data,
            "response_time": response.elapsed.total_seconds()
        }
        
    except Exception as e:
        return {"success": False, "error": str(e)}

def main():
    """Test all endpoints"""
    print("Testing Hardware Resource Optimizer endpoints...")
    print("=" * 60)
    
    # Test endpoints
    endpoints = [
        ("/health", "GET"),
        ("/optimize/memory", "POST"),
        ("/optimize/cpu", "POST"),
        ("/optimize/disk", "POST"),
        ("/optimize/docker", "POST"),
        ("/optimize/all", "POST")
    ]
    
    results = {}
    all_passed = True
    
    for endpoint, method in endpoints:
        print(f"\nTesting {method} {endpoint}...")
        result = test_endpoint(endpoint, method)
        results[endpoint] = result
        
        if result["success"]:
            print(f"✅ PASS - Status: {result['status_code']}, Time: {result.get('response_time', 0):.2f}s")
            
            # Print first few lines of response for verification
            if isinstance(result["data"], dict):
                status = result["data"].get("status", "unknown")
                print(f"   Response status: {status}")
                
                if "actions_taken" in result["data"]:
                    actions = result["data"]["actions_taken"]
                    print(f"   Actions taken: {len(actions) if isinstance(actions, list) else 'N/A'}")
            else:
                print(f"   Response: {str(result['data'])[:100]}...")
        else:
            print(f"❌ FAIL - {result.get('error', 'Unknown error')}")
            if "status_code" in result:
                print(f"   Status: {result['status_code']}")
            all_passed = False
    
    print("\n" + "=" * 60)
    print("SUMMARY:")
    
    for endpoint in results:
        status = "✅ PASS" if results[endpoint]["success"] else "❌ FAIL"
        print(f"  {endpoint}: {status}")
    
    if all_passed:
        print("\n🎉 All endpoints working correctly!")
        return 0
    else:
        print("\n💥 Some endpoints failed!")
        return 1

if __name__ == "__main__":
    sys.exit(main())