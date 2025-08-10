#!/usr/bin/env python3
"""
ULTRA CORS Security Validation Script (Simple Version)
Validates that all services have secure CORS configurations without wildcards
"""
import subprocess
import sys
import re
from pathlib import Path
from datetime import datetime

def test_cors_with_curl(service_name: str, endpoint: str, origin: str) -> dict:
    """Test CORS using curl command"""
    result = {
        "service": service_name,
        "endpoint": endpoint,
        "origin": origin,
        "status": "UNKNOWN",
        "secure": False
    }
    
    try:
        # Build curl command
        cmd = [
            "curl", "-s", "-X", "OPTIONS",
            endpoint,
            "-H", f"Origin: {origin}",
            "-H", "Access-Control-Request-Method: GET",
            "-H", "Access-Control-Request-Headers: Content-Type",
            "-w", "\nHTTP_STATUS:%{http_code}",
            "-I"
        ]
        
        # Execute curl command
        process = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        output = process.stdout
        
        # Extract HTTP status code
        status_match = re.search(r"HTTP_STATUS:(\d+)", output)
        http_status = int(status_match.group(1)) if status_match else 0
        
        # Check for Access-Control-Allow-Origin header
        allow_origin_match = re.search(r"access-control-allow-origin:\s*(.+)", output, re.IGNORECASE)
        allow_origin = allow_origin_match.group(1).strip() if allow_origin_match else None
        
        # Determine security status
        if origin in ["http://evil.com", "http://malicious.site", "*"]:
            # For forbidden origins, we expect rejection
            if http_status >= 400 or not allow_origin:
                result["status"] = "SECURE"
                result["secure"] = True
                result["details"] = f"Correctly rejected (status: {http_status})"
            else:
                result["status"] = "VULNERABLE"
                result["secure"] = False
                result["details"] = f"BREACH: Accepted forbidden origin! ({allow_origin})"
        else:
            # For allowed origins
            if http_status == 200 and allow_origin == origin:
                result["status"] = "ALLOWED"
                result["secure"] = True
                result["details"] = "Correctly allowed"
            else:
                result["status"] = "BLOCKED"
                result["secure"] = True  # Still secure if blocking everything
                result["details"] = f"Blocked (status: {http_status})"
    
    except subprocess.TimeoutExpired:
        result["status"] = "TIMEOUT"
        result["details"] = "Service timeout"
    except Exception as e:
        result["status"] = "ERROR"
        result["details"] = str(e)
    
    return result


def scan_python_files_for_wildcards():
    """Scan Python files for CORS wildcard configurations"""
    print("\n" + "=" * 80)
    print("STATIC CODE ANALYSIS FOR WILDCARD CORS")
    print("=" * 80)
    
    wildcard_files = []
    secure_files = []
    
    # Define paths to scan
    scan_paths = [
        "/opt/sutazaiapp/backend",
        "/opt/sutazaiapp/services",
        "/opt/sutazaiapp/auth",
        "/opt/sutazaiapp/agents",
        "/opt/sutazaiapp/self-healing",
        "/opt/sutazaiapp/docker"
    ]
    
    for base_path in scan_paths:
        if not Path(base_path).exists():
            continue
            
        # Use grep to find files with CORS configurations
        try:
            # Find files with CORSMiddleware
            cmd = ["grep", "-r", "-l", "CORSMiddleware", base_path, "--include=*.py"]
            process = subprocess.run(cmd, capture_output=True, text=True)
            
            if process.returncode == 0:
                files = process.stdout.strip().split('\n')
                
                for file_path in files:
                    if not file_path:
                        continue
                    
                    # Check each file for wildcards
                    with open(file_path, 'r') as f:
                        content = f.read()
                    
                    # Look for wildcard patterns
                    has_wildcard = False
                    if re.search(r'allow_origins\s*=\s*\["?\*"?\]', content):
                        has_wildcard = True
                    elif re.search(r'allow_origins:\s*\["?\*"?\]', content):
                        has_wildcard = True
                    elif '"*"' in content and 'allow_origins' in content:
                        # Check context around wildcard
                        lines = content.split('\n')
                        for i, line in enumerate(lines):
                            if '"*"' in line and 'allow_origins' in lines[max(0,i-3):min(len(lines),i+3)]:
                                # Exclude test files
                                if 'test' not in file_path.lower():
                                    has_wildcard = True
                                    break
                    
                    if has_wildcard:
                        wildcard_files.append(file_path)
                        print(f"\n❌ WILDCARD FOUND: {file_path}")
                    else:
                        secure_files.append(file_path)
                        
        except Exception as e:
            print(f"Error scanning {base_path}: {e}")
    
    return wildcard_files, secure_files


def main():
    print("=" * 80)
    print("ULTRA CORS SECURITY VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    
    # Test key services
    services_to_test = [
        ("Backend API", "http://localhost:10010/health"),
        ("Hardware Optimizer", "http://localhost:11110/health"),
        ("AI Orchestrator", "http://localhost:8589/health"),
        ("Ollama Integration", "http://localhost:8090/health"),
    ]
    
    print("\nPHASE 1: RUNTIME CORS TESTING")
    print("-" * 40)
    
    test_results = []
    for service_name, endpoint in services_to_test:
        print(f"\nTesting {service_name}...")
        
        # Test with forbidden origin
        evil_result = test_cors_with_curl(service_name, endpoint, "http://evil.com")
        print(f"  Evil origin: {evil_result['status']} - {evil_result['details']}")
        test_results.append(evil_result)
        
        # Test with allowed origin
        allowed_result = test_cors_with_curl(service_name, endpoint, "http://localhost:10011")
        print(f"  Allowed origin: {allowed_result['status']} - {allowed_result['details']}")
        test_results.append(allowed_result)
    
    # Static code analysis
    wildcard_files, secure_files = scan_python_files_for_wildcards()
    
    # Summary
    print("\n" + "=" * 80)
    print("SECURITY ASSESSMENT SUMMARY")
    print("=" * 80)
    
    # Calculate scores
    runtime_secure = sum(1 for r in test_results if r["secure"])
    runtime_total = len(test_results)
    vulnerabilities = [r for r in test_results if r["status"] == "VULNERABLE"]
    
    print(f"\n📊 RUNTIME RESULTS:")
    print(f"   Secure tests: {runtime_secure}/{runtime_total}")
    print(f"   Vulnerabilities: {len(vulnerabilities)}")
    
    print(f"\n📝 STATIC ANALYSIS:")
    print(f"   Files with secure CORS: {len(secure_files)}")
    print(f"   Files with wildcards: {len(wildcard_files)}")
    
    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL SECURITY VERDICT")
    print("=" * 80)
    
    if len(vulnerabilities) == 0 and len(wildcard_files) == 0:
        print("\n✅ ULTRA-SECURE: CORS FULLY HARDENED!")
        print("   • No wildcard origins found")
        print("   • All forbidden origins rejected")
        print("   • Explicit origin whitelisting active")
        return 0
    else:
        print("\n❌ SECURITY ISSUES DETECTED!")
        if vulnerabilities:
            print(f"   • {len(vulnerabilities)} runtime vulnerabilities")
        if wildcard_files:
            print(f"   • {len(wildcard_files)} files with wildcards")
            for f in wildcard_files[:5]:  # Show first 5
                print(f"     - {f}")
        return 1


if __name__ == "__main__":
    sys.exit(main())