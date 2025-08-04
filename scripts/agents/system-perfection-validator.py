#!/usr/bin/env python3
"""
System Perfection Validator - Final validation of entire hygiene monitoring system
Purpose: Comprehensive end-to-end system validation ensuring 100% perfection
Usage: python3 system-perfection-validator.py
Requirements: requests
"""

import requests
import json
from datetime import datetime
import time

class SystemPerfectionValidator:
    def __init__(self):
        self.endpoints = {
            'main_dashboard': 'http://localhost:8082',
            'direct_dashboard': 'http://localhost:3002',
            'backend_api': 'http://localhost:8081/api/hygiene/status',
            'rule_api': 'http://localhost:8101/api/rules'
        }
        self.validation_results = {
            'total_tests': 0,
            'passed_tests': 0,
            'failed_tests': 0,
            'critical_issues': [],
            'warnings': [],
            'system_health': {}
        }
    
    def validate_all_services_online(self):
        print('🔍 PERFECTION VALIDATOR: Validating all services are online...')
        
        all_online = True
        for service_name, url in self.endpoints.items():
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    print(f'   ✅ {service_name}: ONLINE (HTTP {response.status_code})')
                    self.validation_results['passed_tests'] += 1
                else:
                    print(f'   ❌ {service_name}: ERROR (HTTP {response.status_code})')
                    self.validation_results['failed_tests'] += 1
                    self.validation_results['critical_issues'].append(f'{service_name} not responding correctly')
                    all_online = False
                self.validation_results['total_tests'] += 1
            except Exception as e:
                print(f'   ❌ {service_name}: OFFLINE - {e}')
                self.validation_results['failed_tests'] += 1
                self.validation_results['critical_issues'].append(f'{service_name} offline: {e}')
                all_online = False
                self.validation_results['total_tests'] += 1
        
        return all_online
    
    def validate_real_data_flow(self):
        print('\n📊 PERFECTION VALIDATOR: Validating real data flow (no static/fake data)...')
        
        real_data = True
        
        try:
            # Get multiple snapshots to verify data is changing
            snapshots = []
            for i in range(3):
                response = requests.get(self.endpoints['backend_api'])
                data = response.json()
                snapshots.append({
                    'timestamp': data.get('timestamp'),
                    'violations': data.get('totalViolations'),
                    'cpu_usage': data.get('systemMetrics', {}).get('cpu_usage'),
                    'snapshot_num': i + 1
                })
                if i < 2:
                    time.sleep(1)
            
            # Verify timestamps are updating (real-time)
            timestamps = [s['timestamp'] for s in snapshots]
            unique_timestamps = len(set(timestamps))
            
            if unique_timestamps > 1:
                print('   ✅ Real-time Data: Timestamps updating correctly')
                self.validation_results['passed_tests'] += 1
            else:
                print('   ⚠️  Real-time Data: Timestamps not changing (potential caching)')
                self.validation_results['warnings'].append('Timestamps not updating between requests')
                real_data = False
            
            # Verify violation data structure is complete
            latest_data = snapshots[-1]
            backend_response = requests.get(self.endpoints['backend_api'])
            full_data = backend_response.json()
            
            required_fields = ['timestamp', 'systemStatus', 'totalViolations', 'recentViolations', 'agentHealth', 'systemMetrics']
            missing_fields = [field for field in required_fields if field not in full_data or not full_data[field]]
            
            if not missing_fields:
                print('   ✅ Data Completeness: All required fields present with data')
                self.validation_results['passed_tests'] += 1
            else:
                print(f'   ❌ Data Completeness: Missing or empty fields: {missing_fields}')
                self.validation_results['failed_tests'] += 1
                self.validation_results['critical_issues'].append(f'Missing data fields: {missing_fields}')
                real_data = False
            
            # Verify violations are real (not static test data)
            recent_violations = full_data.get('recentViolations', [])
            if recent_violations:
                # Check if violations have realistic timestamps (within last hour)
                latest_violation = recent_violations[0]
                violation_time = datetime.fromisoformat(latest_violation['timestamp'].replace('Z', '+00:00'))
                current_time = datetime.now(violation_time.tzinfo)
                time_diff = (current_time - violation_time).total_seconds()
                
                if time_diff < 3600:  # Within last hour
                    print(f'   ✅ Violation Data: Recent violations detected ({len(recent_violations)} violations)')
                    self.validation_results['passed_tests'] += 1
                else:
                    print(f'   ⚠️  Violation Data: Latest violation is {time_diff/60:.1f} minutes old')
                    self.validation_results['warnings'].append('Violations may be stale')
            else:
                print('   ℹ️  Violation Data: No recent violations (system may be clean)')
                # This is not a failure, but worth noting
            
            self.validation_results['total_tests'] += 2  # For the tests above
            
        except Exception as e:
            print(f'   ❌ Real Data Flow: Error validating - {e}')
            self.validation_results['failed_tests'] += 1
            self.validation_results['critical_issues'].append(f'Data flow validation failed: {e}')
            real_data = False
            self.validation_results['total_tests'] += 1
        
        return real_data
    
    def validate_system_performance(self):
        print('\n⚡ PERFECTION VALIDATOR: Validating system performance...')
        
        performance_good = True
        
        # Test response times for all endpoints
        for service_name, url in self.endpoints.items():
            response_times = []
            try:
                for i in range(3):
                    start_time = time.time()
                    response = requests.get(url, timeout=10)
                    response_time = (time.time() - start_time) * 1000
                    response_times.append(response_time)
                
                avg_response_time = sum(response_times) / len(response_times)
                max_response_time = max(response_times)
                
                print(f'   📊 {service_name}: Avg {avg_response_time:.2f}ms, Max {max_response_time:.2f}ms')
                
                # Performance thresholds
                if avg_response_time < 1000:  # Less than 1 second average
                    print(f'     ✅ Performance: Excellent')
                    self.validation_results['passed_tests'] += 1
                elif avg_response_time < 3000:  # Less than 3 seconds average
                    print(f'     ✅ Performance: Acceptable')
                    self.validation_results['passed_tests'] += 1
                else:
                    print(f'     ❌ Performance: Too slow')
                    self.validation_results['failed_tests'] += 1
                    self.validation_results['critical_issues'].append(f'{service_name} response time too slow: {avg_response_time:.2f}ms')
                    performance_good = False
                
                self.validation_results['total_tests'] += 1
                
            except Exception as e:
                print(f'   ❌ {service_name}: Performance test failed - {e}')
                self.validation_results['failed_tests'] += 1
                self.validation_results['critical_issues'].append(f'{service_name} performance test failed')
                performance_good = False
                self.validation_results['total_tests'] += 1
        
        return performance_good
    
    def validate_system_health(self):
        print('\n🏥 PERFECTION VALIDATOR: Validating system health metrics...')
        
        health_good = True
        
        try:
            response = requests.get(self.endpoints['backend_api'])
            data = response.json()
            
            # Check system metrics
            system_metrics = data.get('systemMetrics', {})
            if system_metrics:
                cpu_usage = system_metrics.get('cpu_usage', 0)
                memory_usage = system_metrics.get('memory_percentage', 0)
                disk_usage = system_metrics.get('disk_percentage', 0)
                
                print(f'   💻 System Resources:')
                print(f'     - CPU Usage: {cpu_usage}%')
                print(f'     - Memory Usage: {memory_usage}%')
                print(f'     - Disk Usage: {disk_usage}%')
                
                # Health thresholds
                if cpu_usage < 80:
                    print('     ✅ CPU: Healthy')
                    self.validation_results['passed_tests'] += 1
                else:
                    print('     ⚠️  CPU: High usage')
                    self.validation_results['warnings'].append(f'High CPU usage: {cpu_usage}%')
                
                if memory_usage < 90:
                    print('     ✅ Memory: Healthy')
                    self.validation_results['passed_tests'] += 1
                else:
                    print('     ⚠️  Memory: High usage')
                    self.validation_results['warnings'].append(f'High memory usage: {memory_usage}%')
                
                if disk_usage < 95:
                    print('     ✅ Disk: Healthy')
                    self.validation_results['passed_tests'] += 1
                else:
                    print('     ❌ Disk: Critical usage')
                    self.validation_results['failed_tests'] += 1
                    self.validation_results['critical_issues'].append(f'Critical disk usage: {disk_usage}%')
                    health_good = False
                
                self.validation_results['total_tests'] += 3
                
                # Store health metrics
                self.validation_results['system_health'] = {
                    'cpu_usage': cpu_usage,
                    'memory_usage': memory_usage,
                    'disk_usage': disk_usage,
                    'timestamp': datetime.now().isoformat()
                }
            
            # Check agent health
            agent_health = data.get('agentHealth', [])
            active_agents = [agent for agent in agent_health if agent.get('status') == 'ACTIVE']
            
            print(f'   🤖 Agent Status: {len(active_agents)} active agents')
            if len(active_agents) >= 2:  # Expecting at least 2 agents
                print('     ✅ Agents: Sufficient active agents')
                self.validation_results['passed_tests'] += 1
            else:
                print('     ❌ Agents: Insufficient active agents')
                self.validation_results['failed_tests'] += 1
                self.validation_results['critical_issues'].append('Insufficient active agents')
                health_good = False
            
            self.validation_results['total_tests'] += 1
            
        except Exception as e:
            print(f'   ❌ System Health: Error validating - {e}')
            self.validation_results['failed_tests'] += 1
            self.validation_results['critical_issues'].append(f'System health validation failed: {e}')
            health_good = False
            self.validation_results['total_tests'] += 1
        
        return health_good
    
    def validate_end_to_end_functionality(self):
        print('\n🔄 PERFECTION VALIDATOR: Validating end-to-end functionality...')
        
        e2e_working = True
        
        # Test full workflow: Dashboard -> Backend -> Rules -> Response
        try:
            # 1. Check dashboards load
            dashboard_loads = []
            for dashboard_name, url in [('Main', self.endpoints['main_dashboard']), 
                                       ('Direct', self.endpoints['direct_dashboard'])]:
                response = requests.get(url)
                dashboard_loads.append(response.status_code == 200)
                print(f'   📱 {dashboard_name} Dashboard: {"✅ LOADS" if response.status_code == 200 else "❌ FAILS"}')
            
            # 2. Check backend provides data
            backend_response = requests.get(self.endpoints['backend_api'])
            backend_data = backend_response.json()
            backend_working = 'systemStatus' in backend_data and backend_data['systemStatus'] == 'MONITORING'
            print(f'   ⚙️  Backend API: {"✅ FUNCTIONAL" if backend_working else "❌ NOT WORKING"}')
            
            # 3. Check rules are accessible
            rules_response = requests.get(self.endpoints['rule_api'])
            rules_data = rules_response.json()
            rules_working = 'rules' in rules_data and len(rules_data['rules']) > 0
            print(f'   📋 Rules API: {"✅ FUNCTIONAL" if rules_working else "❌ NOT WORKING"}')
            
            # 4. Verify data consistency
            violations_count = backend_data.get('totalViolations', 0)
            recent_violations = len(backend_data.get('recentViolations', []))
            data_consistent = violations_count >= recent_violations  # Total should be >= recent
            print(f'   📊 Data Consistency: {"✅ VALID" if data_consistent else "❌ INVALID"}')
            
            # Overall E2E assessment
            all_working = all(dashboard_loads) and backend_working and rules_working and data_consistent
            
            if all_working:
                print('   🎯 End-to-End: FULLY FUNCTIONAL')
                self.validation_results['passed_tests'] += 1
            else:
                print('   ❌ End-to-End: PARTIAL FAILURE')
                self.validation_results['failed_tests'] += 1
                self.validation_results['critical_issues'].append('End-to-end workflow not fully functional')
                e2e_working = False
            
            self.validation_results['total_tests'] += 1
            
        except Exception as e:
            print(f'   ❌ End-to-End: Error validating - {e}')
            self.validation_results['failed_tests'] += 1
            self.validation_results['critical_issues'].append(f'E2E validation failed: {e}')
            e2e_working = False
            self.validation_results['total_tests'] += 1
        
        return e2e_working
    
    def generate_perfection_report(self):
        print('\n' + '='*60)
        print('🎯 SYSTEM PERFECTION VALIDATION REPORT')
        print('='*60)
        
        # Calculate success rates
        total_tests = self.validation_results['total_tests']
        passed_tests = self.validation_results['passed_tests']
        failed_tests = self.validation_results['failed_tests']
        success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f'\n📊 TEST RESULTS:')
        print(f'   Total Tests: {total_tests}')
        print(f'   Passed: {passed_tests}')
        print(f'   Failed: {failed_tests}')
        print(f'   Success Rate: {success_rate:.1f}%')
        
        # Critical issues
        if self.validation_results['critical_issues']:
            print(f'\n🚨 CRITICAL ISSUES ({len(self.validation_results["critical_issues"])}):')
            for issue in self.validation_results['critical_issues']:
                print(f'   ❌ {issue}')
        
        # Warnings
        if self.validation_results['warnings']:
            print(f'\n⚠️  WARNINGS ({len(self.validation_results["warnings"])}):')
            for warning in self.validation_results['warnings']:
                print(f'   ⚠️  {warning}')
        
        # System health summary
        if self.validation_results['system_health']:
            health = self.validation_results['system_health']
            print(f'\n💻 SYSTEM HEALTH:')
            print(f'   CPU: {health["cpu_usage"]}%')
            print(f'   Memory: {health["memory_usage"]}%')
            print(f'   Disk: {health["disk_usage"]}%')
        
        # Final assessment
        is_perfect = (failed_tests == 0 and success_rate >= 95)
        
        print('\n' + '='*60)
        if is_perfect:
            print('🏆 SYSTEM STATUS: 100% PERFECTION ACHIEVED!')
            print('✅ All systems operational with real data')
            print('✅ No critical issues detected')
            print('✅ Performance within acceptable limits')
            print('✅ End-to-end functionality verified')
        else:
            print('⚠️  SYSTEM STATUS: PERFECTION NOT ACHIEVED')
            print(f'   Success rate: {success_rate:.1f}% (target: 95%+)')
            print(f'   Critical issues: {len(self.validation_results["critical_issues"])}')
            print('   Review issues above and resolve before production')
        
        print('='*60)
        
        return is_perfect

if __name__ == "__main__":
    print("🚀 Deploying System Perfection Validator...")
    print("🎯 Validating 100% system perfection with NO static/fake data...")
    
    validator = SystemPerfectionValidator()
    
    # Run all validations
    services_online = validator.validate_all_services_online()
    real_data_flow = validator.validate_real_data_flow()
    performance_good = validator.validate_system_performance()
    health_good = validator.validate_system_health()
    e2e_working = validator.validate_end_to_end_functionality()
    
    # Generate final report
    is_perfect = validator.generate_perfection_report()
    
    print(f'\n🎯 FINAL RESULT: {"PERFECT" if is_perfect else "NEEDS ATTENTION"}')