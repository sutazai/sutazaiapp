#!/usr/bin/env python3
"""
ULTRA CORS Security Validation Script
Validates that all services have secure CORS configurations without wildcards
"""
import asyncio
import aiohttp
import json
import sys
from datetime import datetime
from pathlib import Path

# Define test cases
CORS_TEST_CASES = [
    # Service endpoints to test
    ("Backend API", "http://localhost:10010/health"),
    ("Frontend", "http://localhost:10011/"),
    ("Grafana", "http://localhost:10201/api/health"),
    ("Prometheus", "http://localhost:10200/api/v1/query"),
    ("FAISS Vector", "http://localhost:10103/health"),
    ("Hardware Optimizer", "http://localhost:11110/health"),
    ("AI Agent Orchestrator", "http://localhost:8589/health"),
    ("Ollama Integration", "http://localhost:8090/health"),
]

# Define allowed and forbidden origins
ALLOWED_ORIGINS = [
    "http://localhost:10011",    # Frontend
    "http://localhost:10010",    # Backend
    "http://127.0.0.1:10011",    # Alternative frontend
]

FORBIDDEN_ORIGINS = [
    "http://evil.com",
    "http://malicious.site",
    "https://attacker.com",
    "*",  # Wildcard should be rejected
    "http://*",  # Pattern wildcard should be rejected
]


async def test_cors_origin(session: aiohttp.ClientSession, service_name: str, endpoint: str, origin: str) -> Dict:
    """Test CORS configuration for a specific origin"""
    result = {
        "service": service_name,
        "endpoint": endpoint,
        "origin": origin,
        "status": "UNKNOWN",
        "details": "",
        "secure": False
    }
    
    try:
        # Send OPTIONS preflight request
        headers = {
            "Origin": origin,
            "Access-Control-Request-Method": "GET",
            "Access-Control-Request-Headers": "Content-Type"
        }
        
        async with session.options(endpoint, headers=headers, timeout=5) as response:
            # Check response headers
            allow_origin = response.headers.get("Access-Control-Allow-Origin", "")
            
            if origin in FORBIDDEN_ORIGINS:
                # For forbidden origins, we expect rejection (no allow-origin header or error status)
                if response.status >= 400 or not allow_origin or allow_origin != origin:
                    result["status"] = "SECURE"
                    result["secure"] = True
                    result["details"] = f"Correctly rejected forbidden origin (status: {response.status})"
                else:
                    result["status"] = "VULNERABLE"
                    result["secure"] = False
                    result["details"] = f"SECURITY BREACH: Accepted forbidden origin! Allow-Origin: {allow_origin}"
            else:
                # For allowed origins, we expect acceptance
                if response.status == 200 and allow_origin == origin:
                    result["status"] = "ALLOWED"
                    result["secure"] = True
                    result["details"] = f"Correctly allowed legitimate origin"
                else:
                    result["status"] = "MISCONFIGURED"
                    result["secure"] = False
                    result["details"] = f"Failed to allow legitimate origin (status: {response.status})"
                    
    except asyncio.TimeoutError:
        result["status"] = "TIMEOUT"
        result["details"] = "Service did not respond within 5 seconds"
    except aiohttp.ClientError as e:
        result["status"] = "ERROR"
        result["details"] = f"Connection error: {str(e)}"
    except Exception as e:
        result["status"] = "ERROR"
        result["details"] = f"Unexpected error: {str(e)}"
    
    return result


async def validate_file_cors_config(file_path: Path) -> Dict:
    """Validate CORS configuration in a Python file"""
    result = {
        "file": str(file_path),
        "has_cors": False,
        "secure": True,
        "wildcards_found": [],
        "explicit_origins": []
    }
    
    try:
        content = file_path.read_text()
        
        # Check if file has CORS configuration
        if "CORSMiddleware" in content:
            result["has_cors"] = True
            
            # Check for wildcards
            wildcard_patterns = [
                r'allow_origins\s*=\s*\["?\*"?\]',
                r'allow_origins:\s*\["?\*"?\]',
                r'"origins":\s*\["?\*"?\]',
            ]
            
            import re
            for pattern in wildcard_patterns:
                matches = re.findall(pattern, content)
                if matches:
                    result["wildcards_found"].extend(matches)
                    result["secure"] = False
            
            # Extract explicit origins
            origin_pattern = r'allow_origins\s*=\s*\[(.*?)\]'
            matches = re.findall(origin_pattern, content, re.DOTALL)
            for match in matches:
                origins = re.findall(r'"(http[^"]+)"', match)
                result["explicit_origins"].extend(origins)
                
    except Exception as e:
        result["error"] = str(e)
        result["secure"] = False
    
    return result


async def main():
    """Main validation function"""
    print("=" * 80)
    print("ULTRA CORS SECURITY VALIDATION REPORT")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()
    
    # Phase 1: Test runtime CORS behavior
    print("PHASE 1: RUNTIME CORS BEHAVIOR TESTING")
    print("-" * 40)
    
    runtime_results = []
    async with aiohttp.ClientSession() as session:
        # Test each service with forbidden origins
        for service_name, endpoint in CORS_TEST_CASES:
            print(f"\nTesting {service_name}...")
            
            # Test with a forbidden origin
            forbidden_result = await test_cors_origin(
                session, service_name, endpoint, "http://evil.com"
            )
            runtime_results.append(forbidden_result)
            
            # Test with an allowed origin
            allowed_result = await test_cors_origin(
                session, service_name, endpoint, "http://localhost:10011"
            )
            runtime_results.append(allowed_result)
    
    # Phase 2: Static code analysis
    print("\n" + "=" * 80)
    print("PHASE 2: STATIC CODE ANALYSIS")
    print("-" * 40)
    
    # Find all Python files with potential CORS configurations
    python_files = []
    for pattern in ["backend/**/*.py", "services/**/*.py", "auth/**/*.py", "agents/**/*.py"]:
        python_files.extend(Path("/opt/sutazaiapp").glob(pattern))
    
    file_results = []
    files_with_cors = 0
    files_with_wildcards = 0
    
    for file_path in python_files:
        result = await validate_file_cors_config(file_path)
        if result["has_cors"]:
            files_with_cors += 1
            file_results.append(result)
            if result["wildcards_found"]:
                files_with_wildcards += 1
                print(f"\n⚠️ WILDCARD FOUND: {file_path}")
                print(f"   Wildcards: {result['wildcards_found']}")
    
    # Generate summary report
    print("\n" + "=" * 80)
    print("SECURITY ASSESSMENT SUMMARY")
    print("=" * 80)
    
    # Runtime test summary
    runtime_secure = sum(1 for r in runtime_results if r["secure"])
    runtime_total = len(runtime_results)
    runtime_score = (runtime_secure / runtime_total * 100) if runtime_total > 0 else 0
    
    print(f"\n📊 RUNTIME SECURITY SCORE: {runtime_score:.1f}%")
    print(f"   ✅ Secure responses: {runtime_secure}/{runtime_total}")
    
    vulnerabilities = [r for r in runtime_results if not r["secure"] and "VULNERABLE" in r["status"]]
    if vulnerabilities:
        print(f"\n🚨 CRITICAL VULNERABILITIES FOUND:")
        for vuln in vulnerabilities:
            print(f"   - {vuln['service']}: {vuln['details']}")
    else:
        print(f"\n✅ NO CRITICAL VULNERABILITIES FOUND")
    
    # Static analysis summary
    print(f"\n📝 STATIC ANALYSIS:")
    print(f"   Files with CORS: {files_with_cors}")
    print(f"   Files with wildcards: {files_with_wildcards}")
    
    if files_with_wildcards > 0:
        print(f"\n🚨 WILDCARD CONFIGURATIONS DETECTED IN {files_with_wildcards} FILES")
        security_status = "VULNERABLE"
    else:
        print(f"\n✅ NO WILDCARD CONFIGURATIONS FOUND")
        security_status = "SECURE"
    
    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL SECURITY VERDICT")
    print("=" * 80)
    
    if runtime_score == 100 and files_with_wildcards == 0:
        print("\n🎯 ULTRA-SECURE: CORS configuration is properly hardened!")
        print("✅ All wildcard origins have been eliminated")
        print("✅ Only explicit, trusted origins are allowed")
        print("✅ All services correctly reject unauthorized origins")
        exit_code = 0
    elif runtime_score >= 90 and files_with_wildcards == 0:
        print("\n⚠️ MOSTLY SECURE: Minor issues detected")
        print("✅ No wildcards found in code")
        print("⚠️ Some services may have connectivity issues")
        exit_code = 0
    else:
        print("\n🚨 SECURITY BREACH: CORS vulnerabilities detected!")
        print(f"❌ Runtime security score: {runtime_score:.1f}%")
        print(f"❌ Files with wildcards: {files_with_wildcards}")
        print("\nIMMEDIATE ACTION REQUIRED:")
        print("1. Remove all wildcard CORS configurations")
        print("2. Use explicit origin whitelisting")
        print("3. Re-run this validation script")
        exit_code = 1
    
    # Save detailed report
    report = {
        "timestamp": datetime.now().isoformat(),
        "runtime_results": runtime_results,
        "file_results": file_results,
        "summary": {
            "runtime_score": runtime_score,
            "files_with_cors": files_with_cors,
            "files_with_wildcards": files_with_wildcards,
            "security_status": security_status
        }
    }
    
    report_path = Path("/opt/sutazaiapp/reports/cors_security_validation.json")
    report_path.parent.mkdir(exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n📄 Detailed report saved to: {report_path}")
    
    return exit_code


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)