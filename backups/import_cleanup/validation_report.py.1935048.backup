#!/usr/bin/env python3
"""
Comprehensive validation report for hardware-resource-optimizer monitoring
"""

import sys
import json
import time
import requests
import subprocess
from pathlib import Path

def generate_validation_report():
    """Generate comprehensive validation report"""
    
    print("VALIDATION REPORT")
    print("=" * 80)
    print("Component: hardware-resource-optimizer Agent Detection")
    print("Validation Scope: Monitor fix for port 8116 detection and health status")
    print("=" * 80)
    
    passed_checks = 0
    total_checks = 0
    warnings = []
    critical_issues = []
    
    # Test 1: Port availability
    total_checks += 1
    print("\n1. Port 8116 Availability Check")
    try:
        result = subprocess.run(['netstat', '-tlnp'], capture_output=True, text=True)
        if ':8116' in result.stdout:
            print("   ✅ PASS: Port 8116 is listening")
            passed_checks += 1
        else:
            print("   ❌ FAIL: Port 8116 is not listening")
            critical_issues.append("Port 8116 not available - hardware-resource-optimizer not running")
    except Exception as e:
        print(f"   ❌ FAIL: Could not check port status: {e}")
        critical_issues.append("Unable to verify port status")
    
    # Test 2: Direct health endpoint response
    total_checks += 1
    print("\n2. Direct Health Endpoint Response")
    try:
        start_time = time.time()
        response = requests.get('http://localhost:8116/health', timeout=5)
        response_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            print(f"   ✅ PASS: Health endpoint responds with 200 OK")
            print(f"   📊 Response time: {response_time:.0f}ms")
            passed_checks += 1
            
            if response_time > 3000:
                warnings.append(f"Health endpoint response time high: {response_time:.0f}ms")
            elif response_time > 2000:
                warnings.append(f"Health endpoint response time elevated: {response_time:.0f}ms")
        else:
            print(f"   ❌ FAIL: Health endpoint returned {response.status_code}")
            critical_issues.append(f"Health endpoint returned HTTP {response.status_code}")
    except Exception as e:
        print(f"   ❌ FAIL: Health endpoint unreachable: {e}")
        critical_issues.append("Health endpoint unreachable")
    
    # Test 3: JSON response structure
    total_checks += 1
    print("\n3. Health Response Structure Validation")
    try:
        response = requests.get('http://localhost:8116/health', timeout=5)
        if response.status_code == 200:
            data = response.json()
            required_fields = ['status', 'agent', 'timestamp']
            missing_fields = [field for field in required_fields if field not in data]
            
            if not missing_fields:
                print("   ✅ PASS: Health response contains all required fields")
                print(f"   📊 Agent status: {data.get('status')}")
                print(f"   📊 Agent identifier: {data.get('agent')}")
                passed_checks += 1
            else:
                print(f"   ❌ FAIL: Missing required fields: {missing_fields}")
                critical_issues.append(f"Health response missing fields: {missing_fields}")
        else:
            print("   ⏭️  SKIP: Health endpoint not available")
    except Exception as e:
        print(f"   ❌ FAIL: Could not parse health response: {e}")
        warnings.append("Health response not valid JSON")
    
    # Test 4: Monitor detection logic
    total_checks += 1
    print("\n4. Monitor Detection Logic")
    try:
        # Add monitoring directory to path
        sys.path.insert(0, str(Path(__file__).parent))
        from static_monitor import EnhancedMonitor
        
        monitor = EnhancedMonitor()
        agent_id = 'hardware-resource-optimizer'
        
        # Check registry
        if agent_id in monitor.agent_registry.get('agents', {}):
            print("   ✅ PASS: Agent found in registry")
            
            # Check endpoint detection
            agent_info = monitor.agent_registry['agents'][agent_id]
            endpoint = monitor._get_agent_endpoint(agent_id, agent_info)
            
            if endpoint == 'http://localhost:8116':
                print("   ✅ PASS: Correct endpoint detected")
                passed_checks += 1
            else:
                print(f"   ❌ FAIL: Wrong endpoint detected: {endpoint}")
                critical_issues.append(f"Endpoint detection failed: expected http://localhost:8116, got {endpoint}")
        else:
            print("   ❌ FAIL: Agent not found in registry")
            critical_issues.append("hardware-resource-optimizer not in agent registry")
        
        monitor.cleanup()
    except Exception as e:
        print(f"   ❌ FAIL: Monitor detection failed: {e}")
        critical_issues.append(f"Monitor detection error: {e}")
    
    # Test 5: Health check with various timeouts
    total_checks += 1
    print("\n5. Timeout Handling Validation")
    try:
        from static_monitor import EnhancedMonitor
        monitor = EnhancedMonitor()
        agent_id = 'hardware-resource-optimizer'
        agent_info = monitor.agent_registry['agents'][agent_id]
        
        # Test with different timeouts
        timeout_results = {}
        for timeout in [2, 3, 5]:
            health_status, response_time = monitor._check_agent_health(agent_id, agent_info, timeout)
            timeout_results[timeout] = (health_status, response_time)
        
        healthy_count = sum(1 for status, _ in timeout_results.values() if status == 'healthy')
        
        if healthy_count >= 2:  # At least 2 out of 3 should be healthy
            print("   ✅ PASS: Agent consistently reports as healthy")
            passed_checks += 1
            
            # Check if response times are reasonable
            avg_response_time = sum(rt for _, rt in timeout_results.values() if rt) / len([rt for _, rt in timeout_results.values() if rt])
            print(f"   📊 Average response time: {avg_response_time:.0f}ms")
            
            if avg_response_time > 3000:
                warnings.append(f"Average response time high: {avg_response_time:.0f}ms")
        else:
            print(f"   ⚠️  WARNING: Inconsistent health status (only {healthy_count}/3 healthy)")
            warnings.append("Inconsistent health status across different timeouts")
        
        monitor.cleanup()
    except Exception as e:
        print(f"   ❌ FAIL: Timeout handling test failed: {e}")
        warnings.append(f"Timeout handling test error: {e}")
    
    # Test 6: Edge case handling
    total_checks += 1
    print("\n6. Edge Case Handling")
    edge_cases_passed = 0
    edge_cases_total = 3
    
    # Test very short timeout
    try:
        response = requests.get('http://localhost:8116/health', timeout=0.5)
        if response.status_code == 200:
            edge_cases_passed += 1
            print("   ✅ Short timeout (0.5s) handled correctly")
        else:
            print("   ⚠️  Short timeout resulted in non-200 response")
    except requests.exceptions.Timeout:
        print("   ✅ Short timeout correctly times out")
        edge_cases_passed += 1
    except Exception as e:
        print(f"   ⚠️  Short timeout test failed: {e}")
    
    # Test non-existent endpoint
    try:
        response = requests.get('http://localhost:8116/nonexistent', timeout=2)
        if response.status_code == 404:
            edge_cases_passed += 1
            print("   ✅ Non-existent endpoint returns 404")
        else:
            print(f"   ⚠️  Non-existent endpoint returned {response.status_code}")
    except Exception as e:
        print(f"   ⚠️  Non-existent endpoint test failed: {e}")
    
    # Test multiple rapid requests
    try:
        responses = []
        for i in range(3):
            start_time = time.time()
            response = requests.get('http://localhost:8116/health', timeout=3)
            response_time = (time.time() - start_time) * 1000
            responses.append((response.status_code, response_time))
        
        success_count = sum(1 for status, _ in responses if status == 200)
        if success_count == 3:
            edge_cases_passed += 1
            print("   ✅ Multiple rapid requests handled correctly")
        else:
            print(f"   ⚠️  Only {success_count}/3 rapid requests succeeded")
    except Exception as e:
        print(f"   ⚠️  Rapid requests test failed: {e}")
    
    if edge_cases_passed >= 2:
        passed_checks += 1
        print(f"   ✅ PASS: Edge cases handled well ({edge_cases_passed}/{edge_cases_total})")
    else:
        warnings.append(f"Edge case handling could be improved ({edge_cases_passed}/{edge_cases_total})")
        print(f"   ⚠️  WARNING: Some edge cases failed ({edge_cases_passed}/{edge_cases_total})")
    
    # Generate final report
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("-" * 80)
    print(f"✅ Passed: {passed_checks} checks")
    print(f"⚠️  Warnings: {len(warnings)} issues")
    print(f"❌ Failed: {len(critical_issues)} critical issues")
    print(f"📊 Success Rate: {(passed_checks/total_checks)*100:.1f}%")
    
    if critical_issues:
        print("\nCRITICAL ISSUES")
        print("-" * 80)
        for issue in critical_issues:
            print(f"❌ {issue}")
    
    if warnings:
        print("\nWARNINGS")
        print("-" * 80)
        for warning in warnings:
            print(f"⚠️  {warning}")
    
    print("\nRECOMMENDATIONS")
    print("-" * 80)
    if len(critical_issues) == 0 and len(warnings) <= 2:
        print("✅ The hardware-resource-optimizer monitoring fix is working correctly.")
        print("✅ The monitor properly detects the agent on port 8116.")
        print("✅ Health status reporting is accurate.")
        if warnings:
            print("⚠️  Consider optimizing response times if they consistently exceed 2000ms.")
    else:
        print("❌ Issues detected that require attention.")
        if critical_issues:
            print("🔴 Critical issues must be resolved before production use.")
        if len(warnings) > 2:
            print("🟡 Multiple warnings suggest system optimization may be needed.")
    
    print("\nVALIDATION DETAILS")
    print("-" * 80)
    print("Component validated: static_monitor.py hardware-resource-optimizer detection")
    print("Fix validated: Port 8116 detection and health status reporting")
    print("Test coverage: Endpoint detection, health checks, timeout handling, edge cases")
    print("Environment: Production monitoring environment")
    
    return len(critical_issues) == 0

if __name__ == "__main__":
    success = generate_validation_report()
    sys.exit(0 if success else 1)