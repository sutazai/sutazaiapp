#!/usr/bin/env python3
"""
CORRECTED Security Test for Hardware Resource Optimizer
Tests path traversal protection with proper response validation
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import requests
import json
import sys
from datetime import datetime

def test_path_traversal_protection():
    """Test path traversal protection on vulnerable endpoints"""
    
    base_url = "http://localhost:11111"
    
    # Path traversal payloads
    payloads = [
        "../../../etc/passwd",
        "..\\..\\..\\windows\\system32\\config\\sam",
        "/etc/passwd",
        "../../etc/shadow",
        "../../../proc/version",
        "%2e%2e%2f%2e%2e%2f%2e%2e%2fetc%2fpasswd",
        "....//....//....//etc/passwd",
        "..%252f..%252f..%252fetc%252fpasswd"
    ]
    
    # Vulnerable endpoints
    endpoints = [
        "/analyze/storage",
        "/analyze/storage/duplicates", 
        "/optimize/storage/duplicates",
        "/optimize/storage/compress"
    ]
    
    results = {}
    
    logger.info("🔍 CORRECTED Security Test - Path Traversal Protection")
    logger.info("=" * 60)
    
    for endpoint in endpoints:
        results[endpoint] = {"blocked": 0, "allowed": 0, "total": len(payloads)}
        
        logger.info(f"\nTesting {endpoint}:")
        
        for payload in payloads:
            try:
                # Test GET endpoints with malicious path parameter
                if endpoint.startswith("/analyze"):
                    url = f"{base_url}{endpoint}?path={payload}"
                    response = requests.get(url, timeout=10)
                else:
                    # Test POST endpoints with malicious path parameter
                    url = f"{base_url}{endpoint}?path={payload}&dry_run=true"
                    response = requests.post(url, json={}, timeout=10)
                
                # Analyze response for security
                is_blocked = False
                reason = ""
                
                if response.status_code in [400, 403, 404]:
                    # HTTP-level blocking
                    is_blocked = True
                    reason = f"HTTP {response.status_code}"
                elif response.status_code == 200:
                    # Check application-level blocking
                    try:
                        data = response.json()
                        if 'status' in data and data['status'] == 'error':
                            error_msg = data.get('error', '').lower()
                            if any(keyword in error_msg for keyword in ['path', 'safe', 'accessible', 'traversal']):
                                is_blocked = True
                                reason = f"App-level: {data.get('error')}"
                            else:
                                reason = f"Other error: {data.get('error')}"
                        else:
                            # Success response = vulnerability
                            reason = "Success response (VULNERABILITY)"
                    except (AssertionError, Exception) as e:
                        logger.error(f"Unexpected exception: {e}", exc_info=True)
                        reason = "Invalid JSON response"
                else:
                    reason = f"HTTP {response.status_code}"
                
                if is_blocked:
                    results[endpoint]["blocked"] += 1
                    status = "✅ BLOCKED"
                else:
                    results[endpoint]["allowed"] += 1
                    status = "❌ ALLOWED"
                
                logger.info(f"  {payload:<40} {status:<12} {reason}")
                
            except Exception as e:
                logger.error(f"  {payload:<40} {'⚠️  ERROR':<12} {str(e)}")
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("🎯 SECURITY TEST SUMMARY")
    logger.info("=" * 60)
    
    overall_blocked = 0
    overall_total = 0
    
    for endpoint, result in results.items():
        total = result["total"]
        blocked = result["blocked"]
        allowed = result["allowed"]
        block_rate = (blocked / total * 100) if total > 0 else 0
        
        overall_blocked += blocked
        overall_total += total
        
        status_icon = "✅" if block_rate == 100 else "❌" if block_rate == 0 else "⚠️"
        
        logger.info(f"{status_icon} {endpoint:<35} {blocked:>2}/{total:<2} blocked ({block_rate:>5.1f}%)")
    
    overall_rate = (overall_blocked / overall_total * 100) if overall_total > 0 else 0
    
    logger.info("-" * 60)
    logger.info(f"📊 OVERALL SECURITY: {overall_blocked}/{overall_total} blocked ({overall_rate:.1f}%)")
    
    if overall_rate == 100:
        logger.info("🟢 SECURITY STATUS: EXCELLENT - All attacks blocked")
        return True
    elif overall_rate >= 80:
        logger.info("🟡 SECURITY STATUS: GOOD - Most attacks blocked") 
        return False
    else:
        logger.error("🔴 SECURITY STATUS: CRITICAL - Many attacks succeed")
        return False

def main():
    success = test_path_traversal_protection()
    return 0 if success else 1

if __name__ == "__main__":
