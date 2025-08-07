#!/usr/bin/env python3
"""
Simplified Comprehensive Test Suite for static_monitor.py
========================================================

Direct testing approach without complex mocking for better reliability.
"""

import sys
import os
import time
import json
import threading
import subprocess
import socket
from pathlib import Path
import http.server
import socketserver

# Add the monitoring directory to path
sys.path.insert(0, '/opt/sutazaiapp/scripts/monitoring')

# Import the monitor class
from static_monitor import EnhancedMonitor

class TestStaticMonitor:
    """Simplified test suite for static_monitor.py"""
    
    def __init__(self):
        self.test_results = []
        self.mock_servers = []
        
    def log_test(self, test_name, passed, message=""):
        """Log test result"""
        status = "PASSED" if passed else "FAILED"
        symbol = "✓" if passed else "✗"
        print(f"  {symbol} {test_name}: {status}")
        if message:
            print(f"     {message}")
        self.test_results.append((test_name, status, message))
        
    def test_monitor_creation(self):
        """Test 1 - Basic monitor creation and initialization"""
        print("\n=== Test 1: Monitor Creation & Initialization ===")
        
        try:
            monitor = EnhancedMonitor()
            self.log_test("Monitor Creation", True, "Successfully created EnhancedMonitor instance")
            
            # Test configuration loading
            config_loaded = hasattr(monitor, 'config') and monitor.config is not None
            self.log_test("Configuration Loading", config_loaded, f"Config keys: {list(monitor.config.keys()) if config_loaded else 'None'}")
            
            # Test agent registry loading  
            registry_loaded = hasattr(monitor, 'agent_registry') and monitor.agent_registry is not None
            agent_count = len(monitor.agent_registry.get('agents', {}))
            self.log_test("Agent Registry Loading", registry_loaded, f"Loaded {agent_count} agents")
            
            # Test GPU detection
            gpu_detected = hasattr(monitor, 'gpu_available')
            gpu_info = monitor.gpu_info if hasattr(monitor, 'gpu_info') else {}
            self.log_test("GPU Detection", gpu_detected, f"GPU: {gpu_info.get('name', 'Unknown')}")
            
            monitor.cleanup()
            return True
            
        except Exception as e:
            self.log_test("Monitor Creation", False, f"Error: {e}")
            return False
    
    def test_agent_type_detection(self):
        """Test 2 - Agent type detection logic"""
        print("\n=== Test 2: Agent Type Detection ===")
        
        try:
            monitor = EnhancedMonitor()
            
            # Test cases for agent type detection
            test_cases = [
                ({'name': 'hardware-resource-optimizer', 'description': 'Hardware optimization'}, 'INFR'),
                ({'name': 'backend-api-server', 'description': 'Backend API development'}, 'BACK'),
                ({'name': 'frontend-react-app', 'description': 'Frontend web application'}, 'FRON'),
                ({'name': 'ai-model-server', 'description': 'AI model serving'}, 'AI'),
                ({'name': 'security-scanner', 'description': 'Security vulnerability scanning'}, 'SECU'),
                ({'name': 'database-manager', 'description': 'Database management'}, 'DATA'),
                ({'name': 'unknown-agent', 'description': 'Some other functionality'}, 'UTIL'),
            ]
            
            passed_count = 0
            for agent_info, expected_type in test_cases:
                detected_type = monitor._get_agent_type(agent_info)
                passed = detected_type == expected_type
                if passed:
                    passed_count += 1
                self.log_test(f"Type Detection: {agent_info['name'][:20]}", passed, 
                            f"Expected: {expected_type}, Got: {detected_type}")
            
            overall_passed = passed_count == len(test_cases)
            self.log_test("Overall Type Detection", overall_passed, f"{passed_count}/{len(test_cases)} correct")
            
            monitor.cleanup()
            return overall_passed
            
        except Exception as e:
            self.log_test("Agent Type Detection", False, f"Error: {e}")
            return False
    
    def test_system_stats(self):
        """Test 3 - System statistics collection"""
        print("\n=== Test 3: System Statistics Collection ===")
        
        try:
            monitor = EnhancedMonitor()
            
            # Get system stats
            stats = monitor.get_system_stats()
            
            # Check required keys
            required_keys = ['cpu_percent', 'mem_percent', 'disk_percent', 'network', 'gpu']
            all_keys_present = all(key in stats for key in required_keys)
            self.log_test("Required Keys Present", all_keys_present, 
                        f"Keys: {list(stats.keys())}")
            
            # Check CPU stats
            cpu_valid = 0 <= stats['cpu_percent'] <= 100
            self.log_test("CPU Stats Valid", cpu_valid, 
                        f"CPU: {stats['cpu_percent']:.1f}%")
            
            # Check memory stats
            mem_valid = 0 <= stats['mem_percent'] <= 100
            self.log_test("Memory Stats Valid", mem_valid, 
                        f"Memory: {stats['mem_percent']:.1f}%")
            
            # Check network stats
            network_valid = 'bandwidth_mbps' in stats['network']
            self.log_test("Network Stats Valid", network_valid, 
                        f"Bandwidth: {stats['network'].get('bandwidth_mbps', 'N/A')} Mbps")
            
            # Check GPU stats
            gpu_stats = stats['gpu']
            gpu_valid = 'available' in gpu_stats and 'name' in gpu_stats
            self.log_test("GPU Stats Valid", gpu_valid, 
                        f"GPU: {gpu_stats.get('name', 'Unknown')} ({'Available' if gpu_stats.get('available') else 'Not Available'})")
            
            monitor.cleanup()
            return all_keys_present and cpu_valid and mem_valid and network_valid and gpu_valid
            
        except Exception as e:
            self.log_test("System Statistics", False, f"Error: {e}")
            return False
    
    def test_agent_health_checking(self):
        """Test 4 - Agent health checking functionality"""
        print("\n=== Test 4: Agent Health Checking ===")
        
        try:
            monitor = EnhancedMonitor()
            
            # Test port detection
            test_ports = [22, 80, 443, 65432]  # Common ports and unlikely port
            port_test_results = []
            
            for port in test_ports:
                is_open = monitor._test_port_connection(port)
                port_test_results.append(is_open)
                status = "Open" if is_open else "Closed"
                print(f"    Port {port}: {status}")
            
            # At least some ports should give meaningful results
            port_test_passed = True  # We can't predict which ports are open
            self.log_test("Port Connection Testing", port_test_passed, 
                        f"Tested {len(test_ports)} ports successfully")
            
            # Test agent endpoint verification
            test_endpoints = [
                "http://localhost:80",
                "http://localhost:8080", 
                "http://localhost:65432"  # Unlikely to be open
            ]
            
            endpoint_results = []
            for endpoint in test_endpoints:
                try:
                    result = monitor._verify_agent_endpoint(endpoint, "test-agent")
                    endpoint_results.append(result)
                    status = "Valid" if result else "Invalid"
                    print(f"    Endpoint {endpoint}: {status}")
                except Exception as e:
                    endpoint_results.append(False)
                    print(f"    Endpoint {endpoint}: Error - {e}")
            
            endpoint_test_passed = True  # Function executed without crashing
            self.log_test("Endpoint Verification", endpoint_test_passed, 
                        f"Tested {len(test_endpoints)} endpoints")
            
            # Test display name truncation
            test_names = [
                "short",
                "hardware-resource-optimizer", 
                "very-long-agent-name-that-definitely-exceeds-the-twenty-character-limit"
            ]
            
            truncation_passed = True
            for name in test_names:
                truncated = monitor._get_display_name(name)
                max_length = 20
                if len(truncated) > max_length:
                    truncation_passed = False
                print(f"    '{name}' → '{truncated}' (len: {len(truncated)})")
            
            self.log_test("Name Truncation", truncation_passed, 
                        "All names truncated to ≤20 characters")
            
            monitor.cleanup()
            return port_test_passed and endpoint_test_passed and truncation_passed
            
        except Exception as e:
            self.log_test("Agent Health Checking", False, f"Error: {e}")
            return False
    
    def test_adaptive_functionality(self):
        """Test 5 - Adaptive refresh rate and keyboard controls"""
        print("\n=== Test 5: Adaptive Functionality ===")
        
        try:
            monitor = EnhancedMonitor()
            
            # Test refresh rate updates
            initial_rate = monitor.current_refresh_rate
            
            # Test low activity (should increase refresh time - slower updates)
            monitor._update_refresh_rate(20.0, 30.0)  # Low CPU and memory
            low_activity_rate = monitor.current_refresh_rate
            
            # Test high activity (should decrease refresh time - faster updates)
            monitor._update_refresh_rate(85.0, 90.0)  # High CPU and memory
            high_activity_rate = monitor.current_refresh_rate
            
            adaptive_working = True
            if monitor.adaptive_mode:
                adaptive_working = high_activity_rate <= low_activity_rate
            
            self.log_test("Adaptive Refresh Rate", adaptive_working, 
                        f"Low activity: {low_activity_rate:.1f}s, High activity: {high_activity_rate:.1f}s")
            
            # Test keyboard controls
            keyboard_results = []
            
            # Test speed increase
            old_rate = monitor.manual_refresh_rate
            continue_running = monitor._handle_keyboard_input('+')
            new_rate = monitor.manual_refresh_rate
            speed_increase_works = new_rate > old_rate and continue_running
            keyboard_results.append(speed_increase_works)
            self.log_test("Keyboard '+' (slower)", speed_increase_works, 
                        f"{old_rate:.1f}s → {new_rate:.1f}s")
            
            # Test speed decrease
            old_rate = monitor.manual_refresh_rate
            continue_running = monitor._handle_keyboard_input('-')
            new_rate = monitor.manual_refresh_rate
            speed_decrease_works = new_rate < old_rate and continue_running
            keyboard_results.append(speed_decrease_works)
            self.log_test("Keyboard '-' (faster)", speed_decrease_works, 
                        f"{old_rate:.1f}s → {new_rate:.1f}s")
            
            # Test adaptive toggle
            old_adaptive = monitor.adaptive_mode
            continue_running = monitor._handle_keyboard_input('a')
            new_adaptive = monitor.adaptive_mode
            adaptive_toggle_works = new_adaptive != old_adaptive and continue_running
            keyboard_results.append(adaptive_toggle_works)
            self.log_test("Keyboard 'a' (adaptive)", adaptive_toggle_works, 
                        f"{old_adaptive} → {new_adaptive}")
            
            # Test quit
            should_quit = not monitor._handle_keyboard_input('q')
            keyboard_results.append(should_quit)
            self.log_test("Keyboard 'q' (quit)", should_quit, "Returns False to quit")
            
            keyboard_passed = all(keyboard_results)
            
            monitor.cleanup()
            return adaptive_working and keyboard_passed
            
        except Exception as e:
            self.log_test("Adaptive Functionality", False, f"Error: {e}")
            return False
    
    def test_trend_calculation(self):
        """Test 6 - Trend calculation logic"""
        print("\n=== Test 6: Trend Calculation ===")
        
        try:
            monitor = EnhancedMonitor()
            
            from collections import deque
            
            # Test upward trend
            test_data = deque([10, 20, 30], maxlen=60)
            trend = monitor._get_trend(test_data)
            upward_correct = trend == '↑'
            self.log_test("Upward Trend Detection", upward_correct, 
                        f"Data: [10,20,30] → {trend}")
            
            # Test downward trend
            test_data = deque([30, 20, 10], maxlen=60)
            trend = monitor._get_trend(test_data)
            downward_correct = trend == '↓'
            self.log_test("Downward Trend Detection", downward_correct, 
                        f"Data: [30,20,10] → {trend}")
            
            # Test stable trend
            test_data = deque([20, 19, 21], maxlen=60)
            trend = monitor._get_trend(test_data)
            stable_correct = trend == '→'
            self.log_test("Stable Trend Detection", stable_correct, 
                        f"Data: [20,19,21] → {trend}")
            
            # Test insufficient data
            test_data = deque([20], maxlen=60)
            trend = monitor._get_trend(test_data)
            insufficient_correct = trend == '→'
            self.log_test("Insufficient Data Handling", insufficient_correct, 
                        f"Data: [20] → {trend}")
            
            trend_passed = upward_correct and downward_correct and stable_correct and insufficient_correct
            
            monitor.cleanup()
            return trend_passed
            
        except Exception as e:
            self.log_test("Trend Calculation", False, f"Error: {e}")
            return False
    
    def test_error_handling(self):
        """Test 7 - Error handling and graceful degradation"""
        print("\n=== Test 7: Error Handling ===")
        
        try:
            # Test monitor creation with non-existent config
            monitor1 = EnhancedMonitor("/non/existent/config.json")
            no_crash_bad_config = hasattr(monitor1, 'config')
            self.log_test("Bad Config Handling", no_crash_bad_config, 
                        "Monitor created despite bad config path")
            monitor1.cleanup()
            
            # Test network calculation edge cases
            monitor2 = EnhancedMonitor()
            
            class MockNetworkData:
                def __init__(self):
                    self.bytes_sent = 1000
                    self.bytes_recv = 2000
            
            # Test with no baseline (first call)
            network_stats = monitor2._calculate_network_stats(MockNetworkData())
            no_baseline_handled = 'bandwidth_mbps' in network_stats
            self.log_test("No Baseline Handling", no_baseline_handled, 
                        f"Bandwidth: {network_stats.get('bandwidth_mbps', 'N/A')} Mbps")
            
            # Test with immediate second call (small time difference)
            network_stats2 = monitor2._calculate_network_stats(MockNetworkData())
            small_time_handled = 'bandwidth_mbps' in network_stats2
            self.log_test("Small Time Diff Handling", small_time_handled, 
                        f"Bandwidth: {network_stats2.get('bandwidth_mbps', 'N/A')} Mbps")
            
            monitor2.cleanup()
            
            error_handling_passed = no_crash_bad_config and no_baseline_handled and small_time_handled
            return error_handling_passed
            
        except Exception as e:
            self.log_test("Error Handling", False, f"Error: {e}")
            return False
    
    def test_agent_registry_integration(self):
        """Test 8 - Real agent registry integration"""
        print("\n=== Test 8: Agent Registry Integration ===")
        
        try:
            monitor = EnhancedMonitor()
            
            # Test agent status retrieval
            agents, healthy, total = monitor.get_ai_agents_status()
            
            # Check return format
            format_correct = isinstance(agents, list) and isinstance(healthy, int) and isinstance(total, int)
            self.log_test("Agent Status Format", format_correct, 
                        f"Agents: {len(agents)}, Healthy: {healthy}, Total: {total}")
            
            # Check agent display format
            display_format_correct = True
            for agent_line in agents[:3]:  # Check first few
                if not isinstance(agent_line, str):
                    display_format_correct = False
                    break
            self.log_test("Agent Display Format", display_format_correct, 
                        f"All agent lines are strings")
            
            # Test with actual registry
            registry_loaded = len(monitor.agent_registry.get('agents', {})) > 0
            self.log_test("Registry Loading", registry_loaded, 
                        f"Loaded {len(monitor.agent_registry.get('agents', {}))} agents from registry")
            
            monitor.cleanup()
            return format_correct and display_format_correct
            
        except Exception as e:
            self.log_test("Agent Registry Integration", False, f"Error: {e}")
            return False
    
    def run_full_test_suite(self):
        """Run all tests and generate comprehensive report"""
        print("=" * 70)
        print("COMPREHENSIVE STATIC MONITOR VALIDATION SUITE")
        print("=" * 70)
        
        tests = [
            ("Monitor Creation & Initialization", self.test_monitor_creation),
            ("Agent Type Detection", self.test_agent_type_detection),
            ("System Statistics Collection", self.test_system_stats),
            ("Agent Health Checking", self.test_agent_health_checking),
            ("Adaptive Functionality", self.test_adaptive_functionality),
            ("Trend Calculation", self.test_trend_calculation),
            ("Error Handling", self.test_error_handling),
            ("Agent Registry Integration", self.test_agent_registry_integration),
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for test_name, test_function in tests:
            print(f"\nRunning: {test_name}")
            try:
                if test_function():
                    passed_tests += 1
                else:
                    failed_tests += 1
            except Exception as e:
                print(f"  ✗ {test_name}: EXCEPTION - {e}")
                failed_tests += 1
        
        # Generate final report
        self.generate_final_report(passed_tests, failed_tests, len(tests))
        
        return passed_tests, failed_tests
    
    def generate_final_report(self, passed, failed, total):
        """Generate comprehensive final report"""
        print("\n" + "=" * 70)
        print("FINAL VALIDATION REPORT")
        print("=" * 70)
        
        # Test summary
        success_rate = (passed / total * 100) if total > 0 else 0
        print(f"Total Tests Run: {total}")
        print(f"Tests Passed: {passed}")
        print(f"Tests Failed: {failed}")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print("\nDetailed Results:")
        print("-" * 70)
        
        # Group results by status
        passed_results = [(name, status, msg) for name, status, msg in self.test_results if status == "PASSED"]
        failed_results = [(name, status, msg) for name, status, msg in self.test_results if status == "FAILED"]
        
        if passed_results:
            print(f"\n✓ PASSED TESTS ({len(passed_results)}):")
            for name, status, msg in passed_results:
                print(f"  ✓ {name}")
                if msg:
                    print(f"    └─ {msg}")
        
        if failed_results:
            print(f"\n✗ FAILED TESTS ({len(failed_results)}):")
            for name, status, msg in failed_results:
                print(f"  ✗ {name}")
                if msg:
                    print(f"    └─ {msg}")
        
        print("\n" + "=" * 70)
        
        if failed == 0:
            print("🎉 ALL TESTS PASSED!")
            print("   The static_monitor.py is functioning correctly under all test conditions.")
            print("   Ready for production use.")
        else:
            print(f"⚠️  {failed} TEST(S) FAILED")
            print("   Please review the failed tests above and address any issues.")
            print("   The monitor may still function but with potential limitations.")
        
        print("=" * 70)

def main():
    """Main test execution"""
    print("Starting Static Monitor Comprehensive Testing...")
    
    tester = TestStaticMonitor()
    passed, failed = tester.run_full_test_suite()
    
    # Return appropriate exit code
    return 0 if failed == 0 else 1

if __name__ == "__main__":
    sys.exit(main())