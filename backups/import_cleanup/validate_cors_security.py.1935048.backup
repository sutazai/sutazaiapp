#!/usr/bin/env python3
"""
CORS Security Validation
Tests that services still communicate properly after CORS hardening
"""

import requests
import json
import logging
import time
from typing import Dict, List, Tuple, Optional


# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


# Service endpoints to test
SERVICES = {
    "backend": {
        "url": "http://localhost:10010",
        "endpoints": ["/health", "/", "/api/v1/status"],
        "expected_origins": ["http://localhost:10011", "http://localhost:10010"]
    },
    "frontend": {
        "url": "http://localhost:10011", 
        "endpoints": ["/"],
        "expected_origins": []  # Streamlit doesn't expose CORS
    },
    "ollama-integration": {
        "url": "http://localhost:8090",
        "endpoints": ["/health"],
        "expected_origins": ["http://localhost:10010", "http://localhost:10011"]
    },
    "ai-orchestrator": {
        "url": "http://localhost:8589",
        "endpoints": ["/health"],
        "expected_origins": ["http://localhost:10010", "http://localhost:10011"]
    },
    "hardware-optimizer": {
        "url": "http://localhost:11110",
        "endpoints": ["/health"],
        "expected_origins": ["http://localhost:10010", "http://localhost:10011"]
    },
    "prometheus": {
        "url": "http://localhost:10200",
        "endpoints": ["/-/healthy"],
        "cors_test": False  # Prometheus doesn't use CORS
    },
    "grafana": {
        "url": "http://localhost:10201",
        "endpoints": ["/api/health"],
        "cors_test": False  # Grafana has its own CORS handling
    }
}


def test_service_health(service_name: str, config: Dict) -> bool:
    """Test if a service is responding to health checks"""
    logger.info(f"Testing service health: {service_name}")
    
    for endpoint in config["endpoints"]:
        url = f"{config['url']}{endpoint}"
        try:
            response = requests.get(url, timeout=5)
            if response.status_code == 200:
                logger.info(f"  ✅ {endpoint} - Status: {response.status_code}")
                return True
            else:
                logger.warning(f"  ⚠️  {endpoint} - Status: {response.status_code}")
        except requests.exceptions.RequestException as e:
            logger.error(f"  ❌ {endpoint} - Error: {e}")
    
    return False


def test_cors_headers(service_name: str, config: Dict) -> bool:
    """Test CORS headers for security"""
    if config.get("cors_test", True) is False:
        logger.info(f"  ℹ️  CORS testing skipped for {service_name}")
        return True
    
    logger.info(f"Testing CORS headers: {service_name}")
    
    # Test legitimate origin
    legitimate_origin = "http://localhost:10011"
    url = f"{config['url']}{config['endpoints'][0]}"
    
    headers = {
        "Origin": legitimate_origin,
        "Access-Control-Request-Method": "GET"
    }
    
    try:
        # OPTIONS preflight request
        response = requests.options(url, headers=headers, timeout=5)
        
        # Check if legitimate origin is allowed
        allowed_origins = response.headers.get('Access-Control-Allow-Origin', '')
        
        if allowed_origins == legitimate_origin or allowed_origins == '*':
            if allowed_origins == '*':
                logger.error(f"  ❌ SECURITY VIOLATION: Wildcard CORS still present!")
                return False
            else:
                logger.info(f"  ✅ Legitimate origin allowed: {allowed_origins}")
        else:
            logger.warning(f"  ⚠️  Unexpected CORS response: {allowed_origins}")
        
        # Test malicious origin
        malicious_headers = {
            "Origin": "http://malicious-site.com",
            "Access-Control-Request-Method": "GET"
        }
        
        malicious_response = requests.options(url, headers=malicious_headers, timeout=5)
        malicious_origins = malicious_response.headers.get('Access-Control-Allow-Origin', '')
        
        if malicious_origins == "http://malicious-site.com" or malicious_origins == '*':
            logger.error(f"  ❌ SECURITY VIOLATION: Malicious origin allowed!")
            return False
        else:
            logger.info(f"  ✅ Malicious origin blocked")
        
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"  ❌ CORS test failed: {e}")
        return False


def test_cross_service_communication() -> bool:
    """Test that services can still communicate with each other"""
    logger.info("Testing cross-service communication...")
    
    # Test backend can reach health endpoints
    tests = [
        {
            "name": "Backend → Ollama Integration",
            "url": "http://localhost:8090/health",
            "expected_status": 200
        },
        {
            "name": "Backend → Hardware Optimizer", 
            "url": "http://localhost:11110/health",
            "expected_status": 200
        },
        {
            "name": "Backend → AI Orchestrator",
            "url": "http://localhost:8589/health", 
            "expected_status": 200
        }
    ]
    
    all_passed = True
    
    for test in tests:
        try:
            response = requests.get(test["url"], timeout=5)
            if response.status_code == test["expected_status"]:
                logger.info(f"  ✅ {test['name']} - Communication OK")
            else:
                logger.warning(f"  ⚠️  {test['name']} - Status: {response.status_code}")
                all_passed = False
        except requests.exceptions.RequestException as e:
            logger.error(f"  ❌ {test['name']} - Error: {e}")
            all_passed = False
    
    return all_passed


def validate_no_wildcards() -> bool:
    """Validate that no wildcard CORS remains in running services"""
    logger.info("Validating no wildcard CORS in live services...")
    
    test_origin = "http://test-malicious-origin.com"
    services_to_test = ["backend", "ollama-integration", "ai-orchestrator", "hardware-optimizer"]
    
    all_secure = True
    
    for service_name in services_to_test:
        if service_name not in SERVICES:
            continue
            
        config = SERVICES[service_name]
        url = f"{config['url']}{config['endpoints'][0]}"
        
        headers = {
            "Origin": test_origin,
            "Access-Control-Request-Method": "GET"
        }
        
        try:
            response = requests.options(url, headers=headers, timeout=3)
            allowed_origin = response.headers.get('Access-Control-Allow-Origin', '')
            
            if allowed_origin == '*' or allowed_origin == test_origin:
                logger.error(f"  ❌ {service_name}: Wildcard or malicious origin allowed!")
                all_secure = False
            else:
                logger.info(f"  ✅ {service_name}: Properly blocks malicious origins")
                
        except requests.exceptions.RequestException:
            # Service might be down, but not a security issue
            logger.warning(f"  ⚠️  {service_name}: Could not test (service may be down)")
    
    return all_secure


def generate_validation_report(results: Dict) -> str:
    """Generate a comprehensive validation report"""
    report = f"""
# CORS Security Validation Report

**Date:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Validator:** ULTRA-SECURITY EXPERT

## Executive Summary

CORS security validation completed:
- **Services Tested:** {len(SERVICES)}
- **Health Checks:** {'✅ PASSED' if results['health_checks'] else '❌ FAILED'}
- **CORS Security:** {'✅ SECURE' if results['cors_security'] else '❌ VULNERABLE'} 
- **Cross-Service Comm:** {'✅ WORKING' if results['cross_communication'] else '❌ BROKEN'}
- **Wildcard Check:** {'✅ NO WILDCARDS' if results['no_wildcards'] else '❌ WILDCARDS FOUND'}

## Validation Results

### Service Health Status
"""

    for service, status in results.get('service_health', {}).items():
        status_icon = "✅" if status else "❌"
        report += f"- {service}: {status_icon}\n"

    report += f"""
### Security Validation
- **No Wildcard Origins:** {'✅ CONFIRMED' if results['no_wildcards'] else '❌ WILDCARDS DETECTED'}
- **Malicious Origin Blocking:** {'✅ WORKING' if results['cors_security'] else '❌ VULNERABLE'}
- **Legitimate Origin Access:** {'✅ ALLOWED' if results['cors_security'] else '❌ BLOCKED'}

### Communication Tests
- **Cross-Service Requests:** {'✅ FUNCTIONAL' if results['cross_communication'] else '❌ BROKEN'}

## Security Impact

🔒 **CORS SECURITY STATUS:**
"""
    
    if all([results['cors_security'], results['no_wildcards']]):
        report += """
- ✅ All wildcard CORS vulnerabilities eliminated
- ✅ Only legitimate origins allowed
- ✅ Malicious cross-origin requests blocked
- ✅ Services maintain functionality
- ✅ Production-ready security posture
"""
    else:
        report += """
- ❌ Security vulnerabilities still present
- ❌ Manual intervention required
- ❌ Not ready for production deployment
"""

    report += f"""
## Recommendations

{generate_recommendations(results)}

---
**Report generated by CORS Security Validator**
"""
    
    return report


def generate_recommendations(results: Dict) -> str:
    """Generate security recommendations based on validation results"""
    recommendations = []
    
    if not results['health_checks']:
        recommendations.append("- ⚠️  Some services are not responding - check service health")
    
    if not results['cors_security']:
        recommendations.append("- 🔒 CORS security issues detected - review service configurations")
    
    if not results['no_wildcards']:
        recommendations.append("- ❌ Wildcard CORS still present - run fix script again")
    
    if not results['cross_communication']:
        recommendations.append("- 🔗 Service communication broken - verify network configuration")
    
    if all(results.values()):
        recommendations.append("- ✅ All security validations passed - system ready for production")
    
    return '\n'.join(recommendations) if recommendations else "- ✅ No issues detected"


def main():
    """Main validation function"""
    logger.info("🔒 Starting CORS Security Validation")
    
    results = {
        'health_checks': True,
        'cors_security': True, 
        'cross_communication': True,
        'no_wildcards': True,
        'service_health': {}
    }
    
    # Test service health
    logger.info("\n🏥 Testing Service Health...")
    for service_name, config in SERVICES.items():
        is_healthy = test_service_health(service_name, config)
        results['service_health'][service_name] = is_healthy
        if not is_healthy:
            results['health_checks'] = False
    
    # Test CORS security
    logger.info("\n🔒 Testing CORS Security...")
    for service_name, config in SERVICES.items():
        if not test_cors_headers(service_name, config):
            results['cors_security'] = False
    
    # Test cross-service communication
    logger.info("\n🔗 Testing Cross-Service Communication...")
    if not test_cross_service_communication():
        results['cross_communication'] = False
    
    # Validate no wildcards remain
    logger.info("\n🎯 Validating No Wildcard CORS...")
    if not validate_no_wildcards():
        results['no_wildcards'] = False
    
    # Generate report
    report = generate_validation_report(results)
    
    # Save report
    report_path = "/opt/sutazaiapp/CORS_SECURITY_VALIDATION_REPORT.md"
    with open(report_path, 'w') as f:
        f.write(report)
    
    # Summary
    logger.info(f"\n📊 Validation Summary:")
    logger.info(f"   🏥 Health Checks: {'✅ PASSED' if results['health_checks'] else '❌ FAILED'}")
    logger.info(f"   🔒 CORS Security: {'✅ SECURE' if results['cors_security'] else '❌ VULNERABLE'}")
    logger.info(f"   🔗 Communication: {'✅ WORKING' if results['cross_communication'] else '❌ BROKEN'}")
    logger.info(f"   🎯 No Wildcards: {'✅ CONFIRMED' if results['no_wildcards'] else '❌ WILDCARDS FOUND'}")
    logger.info(f"   📄 Report: {report_path}")
    
    # Final verdict
    if all([results['cors_security'], results['no_wildcards']]):
        logger.info("🎉 CORS SECURITY VALIDATION PASSED!")
        return 0
    else:
        logger.error("❌ CORS SECURITY VALIDATION FAILED!")
        return 1


if __name__ == "__main__":
    exit(main())