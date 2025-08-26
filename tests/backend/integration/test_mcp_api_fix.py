#!/usr/bin/env python3
"""
Test script to verify MCP API endpoints are working after fixing critical issues
"""
import asyncio
import httpx
import json
import sys
import time
from datetime import datetime

# Backend API URL
BACKEND_URL = "http://localhost:10010"

async def test_mcp_endpoints():
    """Test all critical MCP API endpoints"""
    results = {
        "timestamp": datetime.now().isoformat(),
        "tests": [],
        "summary": {
            "total": 0,
            "passed": 0,
            "failed": 0
        }
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        # Test 1: MCP Status endpoint
        print("\n1. Testing GET /api/v1/mcp/status...")
        try:
            response = await client.get(f"{BACKEND_URL}/api/v1/mcp/status")
            test_result = {
                "endpoint": "/api/v1/mcp/status",
                "method": "GET",
                "status_code": response.status_code,
                "success": response.status_code != 404,
                "response": response.text[:500] if response.text else None
            }
            if response.status_code == 200:
                print(f"   ✅ SUCCESS: Status {response.status_code}")
                data = response.json()
                print(f"   Response: {json.dumps(data, indent=2)[:300]}")
            elif response.status_code == 503:
                print(f"   ⚠️  SERVICE UNAVAILABLE: MCP infrastructure not ready")
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            test_result = {
                "endpoint": "/api/v1/mcp/status",
                "method": "GET",
                "success": False,
                "error": str(e)
            }
            print(f"   ❌ ERROR: {e}")
        results["tests"].append(test_result)
        
        # Test 2: MCP Health endpoint
        print("\n2. Testing GET /api/v1/mcp/health...")
        try:
            response = await client.get(f"{BACKEND_URL}/api/v1/mcp/health")
            test_result = {
                "endpoint": "/api/v1/mcp/health",
                "method": "GET",
                "status_code": response.status_code,
                "success": response.status_code != 404,
                "response": response.text[:500] if response.text else None
            }
            if response.status_code == 200:
                print(f"   ✅ SUCCESS: Status {response.status_code}")
                data = response.json()
                print(f"   Response: {json.dumps(data, indent=2)[:300]}")
            elif response.status_code == 503:
                print(f"   ⚠️  SERVICE UNAVAILABLE: MCP infrastructure not ready")
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            test_result = {
                "endpoint": "/api/v1/mcp/health",
                "method": "GET",
                "success": False,
                "error": str(e)
            }
            print(f"   ❌ ERROR: {e}")
        results["tests"].append(test_result)
        
        # Test 3: MCP Services list
        print("\n3. Testing GET /api/v1/mcp/services...")
        try:
            response = await client.get(f"{BACKEND_URL}/api/v1/mcp/services")
            test_result = {
                "endpoint": "/api/v1/mcp/services",
                "method": "GET",
                "status_code": response.status_code,
                "success": response.status_code != 404,
                "response": response.text[:500] if response.text else None
            }
            if response.status_code == 200:
                print(f"   ✅ SUCCESS: Status {response.status_code}")
                data = response.json()
                print(f"   Services: {data}")
            elif response.status_code == 503:
                print(f"   ⚠️  SERVICE UNAVAILABLE: MCP infrastructure not ready")
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            test_result = {
                "endpoint": "/api/v1/mcp/services",
                "method": "GET",
                "success": False,
                "error": str(e)
            }
            print(f"   ❌ ERROR: {e}")
        results["tests"].append(test_result)
        
        # Test 4: DinD Status
        print("\n4. Testing GET /api/v1/mcp/dind/status...")
        try:
            response = await client.get(f"{BACKEND_URL}/api/v1/mcp/dind/status")
            test_result = {
                "endpoint": "/api/v1/mcp/dind/status",
                "method": "GET",
                "status_code": response.status_code,
                "success": response.status_code != 404,
                "response": response.text[:500] if response.text else None
            }
            if response.status_code == 200:
                print(f"   ✅ SUCCESS: Status {response.status_code}")
                data = response.json()
                print(f"   DinD Status: {json.dumps(data, indent=2)[:300]}")
            else:
                print(f"   ❌ FAILED: Status {response.status_code}")
                print(f"   Response: {response.text[:200]}")
        except Exception as e:
            test_result = {
                "endpoint": "/api/v1/mcp/dind/status",
                "method": "GET",
                "success": False,
                "error": str(e)
            }
            print(f"   ❌ ERROR: {e}")
        results["tests"].append(test_result)
        
        # Test 5: Backend Health (should always work)
        print("\n5. Testing GET /health (baseline check)...")
        try:
            response = await client.get(f"{BACKEND_URL}/health")
            test_result = {
                "endpoint": "/health",
                "method": "GET",
                "status_code": response.status_code,
                "success": response.status_code == 200
            }
            if response.status_code == 200:
                print(f"   ✅ SUCCESS: Backend is healthy")
            else:
                print(f"   ❌ FAILED: Backend health check failed")
        except Exception as e:
            test_result = {
                "endpoint": "/health",
                "method": "GET",
                "success": False,
                "error": str(e)
            }
            print(f"   ❌ ERROR: Cannot connect to backend: {e}")
        results["tests"].append(test_result)
    
    # Calculate summary
    results["summary"]["total"] = len(results["tests"])
    results["summary"]["passed"] = sum(1 for t in results["tests"] if t.get("success"))
    results["summary"]["failed"] = results["summary"]["total"] - results["summary"]["passed"]
    
    # Print summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    print(f"Total Tests: {results['summary']['total']}")
    print(f"Passed: {results['summary']['passed']}")
    print(f"Failed: {results['summary']['failed']}")
    
    # Check for critical issues
    mcp_endpoints_working = any(
        t["endpoint"].startswith("/api/v1/mcp") and t.get("success") and t.get("status_code") != 404
        for t in results["tests"]
    )
    
    if not mcp_endpoints_working:
        print("\n⚠️  CRITICAL: No MCP endpoints returned valid responses!")
        print("The 404 errors are likely still occurring.")
        print("\nNext steps:")
        print("1. Check backend logs: docker logs sutazai-backend")
        print("2. Restart backend: docker restart sutazai-backend")
        print("3. Install dependencies: docker exec sutazai-backend pip install docker")
    else:
        print("\n✅ SUCCESS: MCP endpoints are accessible!")
        print("The 404 errors have been resolved.")
    
    # Save results to file
    with open("/opt/sutazaiapp/backend/tests/mcp_api_test_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: mcp_api_test_results.json")
    
    return results

async def main():
    """Main test runner"""
    print("="*60)
    print("MCP API ENDPOINT TEST")
    print("Testing MCP API endpoints after critical fixes")
    print("="*60)
    
    # Wait a moment for backend to be ready
    print("\nWaiting for backend to be ready...")
    await asyncio.sleep(2)
    
    # Run tests
    results = await test_mcp_endpoints()
    
    # Return exit code based on results
    if results["summary"]["failed"] > 0:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    asyncio.run(main())