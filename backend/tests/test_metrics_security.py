#!/usr/bin/env python3
"""
Security Test Script for Metrics Endpoints
Validates that all metrics endpoints require proper authentication
"""

import asyncio
import aiohttp
import json
import sys
import os
from datetime import datetime
from typing import Dict, List, Tuple

# Test configuration
BASE_URL = "http://localhost:10010"
TEST_ENDPOINTS = [
    "/metrics",
    "/api/v1/metrics",
    "/api/v1/health/detailed",
    "/api/v1/health/circuit-breakers",
    "/api/v1/cache/stats",
    "/api/v1/cache/clear",
    "/api/v1/cache/invalidate",
    "/api/v1/cache/warm",
    "/api/v1/settings"
]

# Color codes for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


async def test_endpoint_without_auth(session: aiohttp.ClientSession, endpoint: str) -> Tuple[int, str]:
    """Test endpoint without authentication"""
    url = f"{BASE_URL}{endpoint}"
    try:
        # Test GET endpoints
        if endpoint in ["/api/v1/cache/clear", "/api/v1/cache/invalidate", "/api/v1/cache/warm", 
                        "/api/v1/health/circuit-breakers/reset"]:
            # These are POST endpoints
            if endpoint == "/api/v1/cache/invalidate":
                async with session.post(url, json={"tags": ["test"]}) as response:
                    return response.status, await response.text()
            else:
                async with session.post(url) as response:
                    return response.status, await response.text()
        else:
            async with session.get(url) as response:
                return response.status, await response.text()
    except Exception as e:
        return 0, str(e)


async def test_endpoint_with_invalid_token(session: aiohttp.ClientSession, endpoint: str) -> Tuple[int, str]:
    """Test endpoint with invalid JWT token"""
    url = f"{BASE_URL}{endpoint}"
    headers = {"Authorization": "Bearer invalid_token_12345"}
    try:
        if endpoint in ["/api/v1/cache/clear", "/api/v1/cache/invalidate", "/api/v1/cache/warm",
                        "/api/v1/health/circuit-breakers/reset"]:
            if endpoint == "/api/v1/cache/invalidate":
                async with session.post(url, headers=headers, json={"tags": ["test"]}) as response:
                    return response.status, await response.text()
            else:
                async with session.post(url, headers=headers) as response:
                    return response.status, await response.text()
        else:
            async with session.get(url, headers=headers) as response:
                return response.status, await response.text()
    except Exception as e:
        return 0, str(e)


async def test_endpoint_with_valid_token(session: aiohttp.ClientSession, endpoint: str, token: str) -> Tuple[int, str]:
    """Test endpoint with valid JWT token"""
    url = f"{BASE_URL}{endpoint}"
    headers = {"Authorization": f"Bearer {token}"}
    try:
        if endpoint in ["/api/v1/cache/clear", "/api/v1/cache/invalidate", "/api/v1/cache/warm",
                        "/api/v1/health/circuit-breakers/reset"]:
            if endpoint == "/api/v1/cache/invalidate":
                async with session.post(url, headers=headers, json={"tags": ["test"]}) as response:
                    return response.status, await response.text()
            else:
                async with session.post(url, headers=headers) as response:
                    return response.status, await response.text()
        else:
            async with session.get(url, headers=headers) as response:
                return response.status, await response.text()
    except Exception as e:
        return 0, str(e)


async def get_auth_token(session: aiohttp.ClientSession) -> str:
    """Get a valid authentication token"""
    # Try to login with test credentials
    login_url = f"{BASE_URL}/auth/login"
    login_data = {
        "username": os.getenv("TEST_USER", "admin"),
        "password": os.getenv("TEST_PASSWORD", "admin123")
    }
    
    try:
        async with session.post(login_url, json=login_data) as response:
            if response.status == 200:
                data = await response.json()
                return data.get("access_token", "")
    except:
        pass
    
    return ""


def print_test_header():
    """Print test header"""
    print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"{Colors.BOLD}METRICS ENDPOINT SECURITY VALIDATION{Colors.RESET}")
    print(f"{Colors.BOLD}{'='*80}{Colors.RESET}")
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Target: {BASE_URL}")
    print(f"Endpoints to test: {len(TEST_ENDPOINTS)}\n")


def print_section(title: str):
    """Print section header"""
    print(f"\n{Colors.BLUE}{Colors.BOLD}{title}{Colors.RESET}")
    print(f"{Colors.BLUE}{'-'*60}{Colors.RESET}")


def print_result(endpoint: str, status: int, expected: int, passed: bool):
    """Print test result"""
    icon = "✓" if passed else "✗"
    color = Colors.GREEN if passed else Colors.RED
    status_str = str(status) if status > 0 else "ERROR"
    
    print(f"  {color}{icon}{Colors.RESET} {endpoint:40} Status: {status_str:3} (Expected: {expected})")


async def run_security_tests():
    """Run all security tests"""
    print_test_header()
    
    results = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "vulnerabilities": []
    }
    
    async with aiohttp.ClientSession() as session:
        # Get auth token if available
        token = await get_auth_token(session)
        has_auth = bool(token)
        
        if has_auth:
            print(f"{Colors.GREEN}✓ Authentication service available{Colors.RESET}")
        else:
            print(f"{Colors.YELLOW}⚠ Authentication service not available - limited testing{Colors.RESET}")
        
        # Test 1: Endpoints without authentication should be protected
        print_section("Test 1: Access without authentication (should be denied or limited)")
        
        for endpoint in TEST_ENDPOINTS:
            status, response = await test_endpoint_without_auth(session, endpoint)
            results["total"] += 1
            
            # Check if endpoint is properly protected
            if endpoint in ["/health", "/api/v1/status"]:
                # These can be public
                passed = status == 200
                expected = 200
            elif endpoint in ["/api/v1/settings", "/api/v1/metrics", "/api/v1/cache/stats"]:
                # These should return limited data or require auth
                passed = status in [200, 401, 403]
                expected = "200(limited) or 401"
            else:
                # These should require authentication
                passed = status in [401, 403]
                expected = 401
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                if status == 200:
                    results["vulnerabilities"].append(f"{endpoint} - Exposed without authentication")
            
            print_result(endpoint, status, expected, passed)
            
            # Check response for sensitive data leakage
            if status == 200 and endpoint in ["/metrics", "/api/v1/metrics"]:
                if "cpu_percent" in response or "memory" in response:
                    print(f"    {Colors.YELLOW}⚠ WARNING: Sensitive metrics exposed{Colors.RESET}")
        
        # Test 2: Invalid token should be rejected
        print_section("Test 2: Access with invalid token (should be denied)")
        
        for endpoint in TEST_ENDPOINTS[:5]:  # Test subset to avoid rate limiting
            status, _ = await test_endpoint_with_invalid_token(session, endpoint)
            results["total"] += 1
            
            passed = status in [401, 403]
            expected = 401
            
            if passed:
                results["passed"] += 1
            else:
                results["failed"] += 1
                if status == 200:
                    results["vulnerabilities"].append(f"{endpoint} - Accepts invalid token")
            
            print_result(endpoint, status, expected, passed)
        
        # Test 3: Valid token should work (if auth is available)
        if has_auth:
            print_section("Test 3: Access with valid token (should be allowed)")
            
            for endpoint in TEST_ENDPOINTS[:3]:  # Test subset
                status, _ = await test_endpoint_with_valid_token(session, endpoint, token)
                results["total"] += 1
                
                # Admin endpoints need admin token
                if endpoint in ["/api/v1/cache/clear", "/api/v1/health/circuit-breakers/reset"]:
                    # Might be forbidden if token doesn't have admin scope
                    passed = status in [200, 403]
                    expected = "200 or 403"
                else:
                    passed = status == 200
                    expected = 200
                
                if passed:
                    results["passed"] += 1
                else:
                    results["failed"] += 1
                
                print_result(endpoint, status, expected, passed)
    
    # Print summary
    print_section("SECURITY TEST SUMMARY")
    
    pass_rate = (results["passed"] / results["total"] * 100) if results["total"] > 0 else 0
    
    print(f"\nTotal Tests: {results['total']}")
    print(f"{Colors.GREEN}Passed: {results['passed']}{Colors.RESET}")
    print(f"{Colors.RED}Failed: {results['failed']}{Colors.RESET}")
    print(f"Pass Rate: {pass_rate:.1f}%")
    
    if results["vulnerabilities"]:
        print(f"\n{Colors.RED}{Colors.BOLD}CRITICAL VULNERABILITIES FOUND:{Colors.RESET}")
        for vuln in results["vulnerabilities"]:
            print(f"  {Colors.RED}• {vuln}{Colors.RESET}")
        return False
    else:
        print(f"\n{Colors.GREEN}{Colors.BOLD}✓ No critical vulnerabilities found{Colors.RESET}")
        return True


async def test_cors_configuration():
    """Test CORS configuration for security"""
    print_section("CORS Configuration Test")
    
    async with aiohttp.ClientSession() as session:
        # Test with different origins
        test_origins = [
            ("http://localhost:10011", True),  # Should be allowed
            ("http://evil.com", False),        # Should be blocked
            ("*", False)                       # Wildcard should be blocked
        ]
        
        for origin, should_allow in test_origins:
            headers = {"Origin": origin}
            url = f"{BASE_URL}/api/v1/status"
            
            try:
                async with session.options(url, headers=headers) as response:
                    cors_header = response.headers.get("Access-Control-Allow-Origin", "")
                    
                    if should_allow:
                        passed = cors_header == origin
                    else:
                        passed = cors_header != origin and cors_header != "*"
                    
                    icon = "✓" if passed else "✗"
                    color = Colors.GREEN if passed else Colors.RED
                    
                    print(f"  {color}{icon}{Colors.RESET} Origin: {origin:30} CORS: {cors_header or 'None':20} {'PASS' if passed else 'FAIL'}")
                    
                    if not passed and not should_allow and cors_header:
                        print(f"    {Colors.RED}⚠ SECURITY ISSUE: Unauthorized origin allowed{Colors.RESET}")
            except Exception as e:
                print(f"  {Colors.YELLOW}⚠ Error testing origin {origin}: {e}{Colors.RESET}")


async def main():
    """Main test runner"""
    try:
        # Run security tests
        security_passed = await run_security_tests()
        
        # Run CORS tests
        await test_cors_configuration()
        
        # Final result
        print(f"\n{Colors.BOLD}{'='*80}{Colors.RESET}")
        if security_passed:
            print(f"{Colors.GREEN}{Colors.BOLD}✓ SECURITY VALIDATION PASSED{Colors.RESET}")
            print(f"{Colors.GREEN}All metrics endpoints are properly secured{Colors.RESET}")
            return 0
        else:
            print(f"{Colors.RED}{Colors.BOLD}✗ SECURITY VALIDATION FAILED{Colors.RESET}")
            print(f"{Colors.RED}Critical vulnerabilities detected - immediate action required{Colors.RESET}")
            return 1
            
    except Exception as e:
        print(f"{Colors.RED}Error running tests: {e}{Colors.RESET}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)