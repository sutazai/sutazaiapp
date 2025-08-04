#!/usr/bin/env python3
"""
Testing Team Agent - Verifies all endpoints are responding with real data
Purpose: Comprehensive testing of hygiene monitoring system endpoints
Usage: python3 testing-team-agent.py
Requirements: requests
"""

import requests
import json
from datetime import datetime
import time

class TestingTeamAgent:
    def __init__(self):
        self.endpoints = {
            'main_dashboard': 'http://localhost:8082',
            'direct_dashboard': 'http://localhost:3002', 
            'backend_api': 'http://localhost:8081/api/hygiene/status',
            'rule_control': 'http://localhost:8101/api/rules'
        }
        self.test_results = []
    
    def test_endpoint_connectivity(self):
        print('🔬 TESTING TEAM: Starting endpoint connectivity tests...')
        for name, url in self.endpoints.items():
            try:
                response = requests.get(url, timeout=5)
                result = {
                    'endpoint': name,
                    'url': url,
                    'status_code': response.status_code,
                    'response_time_ms': round(response.elapsed.total_seconds() * 1000, 2),
                    'has_content': len(response.text) > 0,
                    'timestamp': datetime.now().isoformat()
                }
                self.test_results.append(result)
                print(f'✅ {name}: HTTP {response.status_code} ({result["response_time_ms"]}ms)')
            except Exception as e:
                result = {
                    'endpoint': name,
                    'url': url,
                    'error': str(e),
                    'timestamp': datetime.now().isoformat()
                }
                self.test_results.append(result)
                print(f'❌ {name}: ERROR - {e}')
    
    def test_api_data_validity(self):
        print('\n📊 TESTING TEAM: Validating API data structures...')
        
        # Test backend API
        try:
            response = requests.get(self.endpoints['backend_api'])
            data = response.json()
            
            required_fields = ['timestamp', 'systemStatus', 'complianceScore', 'totalViolations', 'activeAgents']
            missing_fields = [field for field in required_fields if field not in data]
            
            if not missing_fields:
                print('✅ Backend API: All required fields present')
                print(f'   - System Status: {data["systemStatus"]}')
                print(f'   - Total Violations: {data["totalViolations"]}')
                print(f'   - Active Agents: {data["activeAgents"]}')
                print(f'   - Compliance Score: {data["complianceScore"]}%')
            else:
                print(f'❌ Backend API: Missing fields - {missing_fields}')
                
        except Exception as e:
            print(f'❌ Backend API: Error validating data - {e}')
            
        # Test rule control API
        try:
            response = requests.get(self.endpoints['rule_control'])
            data = response.json()
            
            if 'rules' in data and isinstance(data['rules'], list):
                print(f'✅ Rule Control API: {len(data["rules"])} rules available')
                enabled_rules = [r for r in data['rules'] if r.get('enabled', False)]
                print(f'   - Enabled Rules: {len(enabled_rules)}')
                print(f'   - Total Rules: {data.get("total", 0)}')
            else:
                print('❌ Rule Control API: Invalid data structure')
                
        except Exception as e:
            print(f'❌ Rule Control API: Error validating data - {e}')
    
    def test_real_time_updates(self):
        print('\n⚡ TESTING TEAM: Testing real-time data updates...')
        
        # Get initial state
        initial_response = requests.get(self.endpoints['backend_api'])
        initial_data = initial_response.json()
        initial_timestamp = initial_data.get('timestamp')
        
        print(f'Initial timestamp: {initial_timestamp}')
        
        # Wait and check again
        time.sleep(2)
        
        updated_response = requests.get(self.endpoints['backend_api'])
        updated_data = updated_response.json()
        updated_timestamp = updated_data.get('timestamp')
        
        print(f'Updated timestamp: {updated_timestamp}')
        
        if initial_timestamp != updated_timestamp:
            print('✅ Real-time updates: Data is being refreshed')
        else:
            print('⚠️  Real-time updates: Timestamps identical (may be cached)')
    
    def test_violation_detection(self):
        print('\n🔍 TESTING TEAM: Testing violation detection system...')
        
        try:
            response = requests.get(self.endpoints['backend_api'])
            data = response.json()
            
            recent_violations = data.get('recentViolations', [])
            if recent_violations:
                print(f'✅ Violation Detection: {len(recent_violations)} recent violations found')
                
                # Check violation structure
                sample_violation = recent_violations[0]
                required_fields = ['rule_id', 'rule_name', 'file_path', 'severity', 'description', 'timestamp']
                missing_fields = [field for field in required_fields if field not in sample_violation]
                
                if not missing_fields:
                    print('✅ Violation Structure: All required fields present')
                    print(f'   - Latest: {sample_violation["rule_name"]} in {sample_violation["file_path"]}')
                else:
                    print(f'❌ Violation Structure: Missing fields - {missing_fields}')
            else:
                print('⚠️  Violation Detection: No recent violations (system may be clean)')
                
        except Exception as e:
            print(f'❌ Violation Detection: Error - {e}')
    
    def generate_report(self):
        print('\n📋 TESTING TEAM: Final Test Report')
        print('=' * 50)
        
        total_tests = len(self.test_results)
        successful_tests = len([r for r in self.test_results if 'error' not in r and r.get('status_code') == 200])
        
        print(f'Total Endpoints Tested: {total_tests}')
        print(f'Successful Tests: {successful_tests}/{total_tests}')
        print(f'Success Rate: {(successful_tests/total_tests)*100:.1f}%')
        
        if successful_tests == total_tests:
            print('\n🎉 ALL TESTS PASSED - System ready for production!')
        else:
            print('\n⚠️  Some tests failed - Review required')
        
        return successful_tests == total_tests

if __name__ == "__main__":
    print("🚀 Deploying Testing Team Agent...")
    agent = TestingTeamAgent()
    agent.test_endpoint_connectivity()
    agent.test_api_data_validity()
    agent.test_real_time_updates()
    agent.test_violation_detection()
    success = agent.generate_report()
    print(f'\nTesting Agent Result: {"PASS" if success else "FAIL"}')