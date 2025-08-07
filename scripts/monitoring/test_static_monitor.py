#!/usr/bin/env python3
"""
Comprehensive Test Suite for static_monitor.py
=============================================

This test suite validates all functionality of the static_monitor.py including:
- Agent type detection and categorization
- Port detection and health checking
- Adaptive refresh rate functionality
- Keyboard controls
- Metrics collection and trending
- Error handling and graceful degradation
"""

import sys
import os
import time
import json
import threading
import subprocess
import socket
from pathlib import Path
from typing import Dict, List, Tuple
import unittest
from unittest.mock import Mock, patch, MagicMock
import tempfile

# Add the monitoring directory to path
sys.path.insert(0, '/opt/sutazaiapp/scripts/monitoring')

# Import the monitor class
from static_monitor import EnhancedMonitor

class MockAgent:
    """Mock agent for testing port detection"""
    
    def __init__(self, port: int, agent_id: str, response_type: str = 'healthy'):
        self.port = port
        self.agent_id = agent_id
        self.response_type = response_type
        self.socket = None
        self.running = False
        
    def start(self):
        """Start mock HTTP server on the specified port"""
        try:
            import http.server
            import socketserver
            from threading import Thread
            
            class MockHandler(http.server.BaseHTTPRequestHandler):
                def __init__(self, response_type, *args, **kwargs):
                    self.response_type = response_type
                    super().__init__(*args, **kwargs)
                    
                def do_GET(self):
                    if self.path in ['/health', '/status', '/ping', '/api/health', '/heartbeat']:
                        if self.response_type == 'healthy':
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps({
                                'status': 'ok',
                                'agent_id': self.server.agent_id,
                                'timestamp': time.time()
                            }).encode())
                        elif self.response_type == 'warning':
                            self.send_response(200)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps({
                                'status': 'degraded',
                                'agent_id': self.server.agent_id,
                                'timestamp': time.time()
                            }).encode())
                        elif self.response_type == 'critical':
                            self.send_response(503)
                            self.send_header('Content-type', 'application/json')
                            self.end_headers()
                            self.wfile.write(json.dumps({
                                'status': 'critical',
                                'agent_id': self.server.agent_id,
                                'timestamp': time.time()
                            }).encode())
                        else:  # offline or error
                            self.send_response(500)
                            self.end_headers()
                    else:
                        self.send_response(404)
                        self.end_headers()
                        
                def log_message(self, format, *args):
                    pass  # Suppress log messages
            
            handler = lambda *args, **kwargs: MockHandler(self.response_type, *args, **kwargs)
            
            with socketserver.TCPServer(("", self.port), handler) as httpd:
                httpd.agent_id = self.agent_id
                self.running = True
                print(f"Mock agent {self.agent_id} started on port {self.port}")
                httpd.serve_forever()
                
        except Exception as e:
            print(f"Failed to start mock agent {self.agent_id} on port {self.port}: {e}")
            
    def stop(self):
        """Stop the mock agent"""
        self.running = False

class TestStaticMonitor(unittest.TestCase):
    """Comprehensive test suite for static_monitor.py"""
    
    def setUp(self):
        """Set up test environment"""
        # Create temporary config file
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        
        # Test configuration
        self.test_config = {
            'refresh_rate': 1.0,
            'adaptive_refresh': True,
            'thresholds': {
                'cpu_warning': 70,
                'cpu_critical': 85,
                'memory_warning': 75,
                'memory_critical': 90,
                'response_time_warning': 2000,
                'response_time_critical': 5000
            },
            'agent_monitoring': {
                'enabled': True,
                'timeout': 2,
                'max_agents_display': 6
            },
            'logging': {
                'enabled': False
            }
        }
        
        with open(self.config_path, 'w') as f:
            json.dump(self.test_config, f)
            
        # Create test agent registry
        self.agent_registry_path = os.path.join(self.temp_dir, 'agent_registry.json')
        self.test_agent_registry = {
            'agents': {
                'hardware-resource-optimizer': {
                    'name': 'hardware-resource-optimizer',
                    'description': 'Hardware optimization agent for infrastructure management',
                    'capabilities': ['optimization', 'monitoring'],
                    'config_path': 'configs/hardware-resource-optimizer_universal.json'
                },
                'senior-backend-developer': {
                    'name': 'senior-backend-developer',
                    'description': 'Backend development agent with API expertise',
                    'capabilities': ['code_generation', 'testing'],
                    'config_path': 'configs/senior-backend-developer_universal.json'
                },
                'senior-frontend-developer': {
                    'name': 'senior-frontend-developer',
                    'description': 'Frontend development agent with React/Vue expertise',
                    'capabilities': ['code_generation', 'testing'],
                    'config_path': 'configs/senior-frontend-developer_universal.json'
                },
                'ollama-integration-specialist': {
                    'name': 'ollama-integration-specialist',
                    'description': 'AI model integration specialist',
                    'capabilities': ['ai', 'integration'],
                    'config_path': 'configs/ollama-integration-specialist_universal.json'
                }
            }
        }
        
        with open(self.agent_registry_path, 'w') as f:
            json.dump(self.test_agent_registry, f)
        
        # Mock agents for testing
        self.mock_agents = []
        
    def tearDown(self):
        """Clean up test environment"""
        # Stop all mock agents
        for agent in self.mock_agents:
            agent.stop()
        
        # Clean up temporary files
        import shutil
        shutil.rmtree(self.temp_dir)
        
    def test_agent_type_detection(self):
        """Test 1 - Agent type detection and categorization"""
        print("\n=== Test 1: Agent Type Detection ===")
        
        # Create monitor instance
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open_registry(self.test_agent_registry)):
                monitor = EnhancedMonitor(self.config_path)
                
                # Test each agent type classification
                test_cases = [
                    ('hardware-resource-optimizer', 'INFR'),
                    ('senior-backend-developer', 'BACK'),
                    ('senior-frontend-developer', 'FRON'),
                    ('ollama-integration-specialist', 'AI')
                ]
                
                for agent_id, expected_type in test_cases:
                    agent_info = self.test_agent_registry['agents'][agent_id]
                    detected_type = monitor._get_agent_type(agent_info)
                    print(f"  Agent: {agent_id[:25]:<25} Expected: {expected_type:<4} Detected: {detected_type:<4} {'✓' if detected_type == expected_type else '✗'}")
                    self.assertEqual(detected_type, expected_type, 
                                   f"Agent type detection failed for {agent_id}")
        
        print("✓ Agent type detection test passed")
        
    def test_port_detection_and_health_checks(self):
        """Test 2 - Port detection and health checking for various agent states"""
        print("\n=== Test 2: Port Detection and Health Checks ===")
        
        # Start mock agents with different health states
        test_ports = [8116, 8000, 8001, 8002]
        health_states = ['healthy', 'warning', 'critical', 'offline']
        
        mock_threads = []
        
        try:
            # Start mock agents
            for i, (port, state) in enumerate(zip(test_ports, health_states)):
                if state != 'offline':  # Don't start offline agents
                    agent_id = f"test-agent-{i}"
                    mock_agent = MockAgent(port, agent_id, state)
                    self.mock_agents.append(mock_agent)
                    
                    # Start in separate thread
                    thread = threading.Thread(target=mock_agent.start, daemon=True)
                    thread.start()
                    mock_threads.append(thread)
                    time.sleep(0.1)  # Give time to start
            
            # Give servers time to fully start
            time.sleep(1)
            
            # Create monitor and test port detection
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open_registry(self.test_agent_registry)):
                    monitor = EnhancedMonitor(self.config_path)
                    
                    # Test port connection detection
                    for port in test_ports[:3]:  # Skip offline port
                        is_open = monitor._test_port_connection(port)
                        print(f"  Port {port}: {'Open' if is_open else 'Closed'} {'✓' if is_open else '✗'}")
                        self.assertTrue(is_open, f"Port {port} should be detected as open")
                    
                    # Test offline port
                    is_open = monitor._test_port_connection(test_ports[3])
                    print(f"  Port {test_ports[3]} (offline): {'Open' if is_open else 'Closed'} {'✓' if not is_open else '✗'}")
                    self.assertFalse(is_open, f"Port {test_ports[3]} should be detected as closed")
                    
                    # Test health check responses
                    for i, port in enumerate(test_ports[:3]):
                        endpoint = f"http://localhost:{port}"
                        agent_id = f"test-agent-{i}"
                        if monitor._verify_agent_endpoint(endpoint, agent_id):
                            health_status, response_time = monitor._check_agent_health(
                                agent_id, {'name': agent_id}, 3
                            )
                            expected_status = health_states[i]
                            print(f"  Agent {agent_id}: {health_status} (RT: {response_time:.0f}ms) {'✓' if health_status == expected_status else '✗'}")
                            self.assertEqual(health_status, expected_status, 
                                           f"Health status mismatch for {agent_id}")
                        
        finally:
            # Clean up mock agents
            for agent in self.mock_agents:
                agent.stop()
        
        print("✓ Port detection and health check test passed")
        
    def test_adaptive_refresh_functionality(self):
        """Test 3 - Adaptive refresh rate based on system load"""
        print("\n=== Test 3: Adaptive Refresh Rate Functionality ===")
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open_registry(self.test_agent_registry)):
                monitor = EnhancedMonitor(self.config_path)
                
                # Test adaptive mode
                monitor.adaptive_mode = True
                monitor.manual_refresh_rate = 2.0
                
                # Test low activity scenario
                monitor._update_refresh_rate(30.0, 40.0)  # Low CPU and memory
                low_activity_rate = monitor.current_refresh_rate
                print(f"  Low activity (CPU:30%, MEM:40%): {low_activity_rate:.1f}s")
                
                # Test high activity scenario
                monitor._update_refresh_rate(75.0, 80.0)  # High CPU and memory
                high_activity_rate = monitor.current_refresh_rate
                print(f"  High activity (CPU:75%, MEM:80%): {high_activity_rate:.1f}s")
                
                # Test very high activity scenario
                monitor._update_refresh_rate(90.0, 90.0)  # Very high CPU and memory
                very_high_activity_rate = monitor.current_refresh_rate
                print(f"  Very high activity (CPU:90%, MEM:90%): {very_high_activity_rate:.1f}s")
                
                # Verify adaptive behavior
                self.assertGreater(low_activity_rate, high_activity_rate, 
                                 "Low activity should have slower refresh rate")
                self.assertGreater(high_activity_rate, very_high_activity_rate, 
                                 "High activity should have faster refresh rate than very high")
                
                # Test manual mode
                monitor.adaptive_mode = False
                monitor.manual_refresh_rate = 1.5
                monitor._update_refresh_rate(90.0, 90.0)  # Should not change in manual mode
                manual_rate = monitor.current_refresh_rate
                print(f"  Manual mode (CPU:90%, MEM:90%): {manual_rate:.1f}s")
                self.assertEqual(manual_rate, 1.5, "Manual mode should maintain set rate")
        
        print("✓ Adaptive refresh rate test passed")
        
    def test_keyboard_controls(self):
        """Test 4 - Keyboard control functionality"""
        print("\n=== Test 4: Keyboard Controls ===")
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open_registry(self.test_agent_registry)):
                monitor = EnhancedMonitor(self.config_path)
                
                initial_rate = monitor.manual_refresh_rate
                
                # Test increase speed (slower refresh)
                result = monitor._handle_keyboard_input('+')
                increased_rate = monitor.manual_refresh_rate
                print(f"  '+' key: {initial_rate:.1f}s → {increased_rate:.1f}s {'✓' if increased_rate > initial_rate else '✗'}")
                self.assertGreater(increased_rate, initial_rate, "'+' should increase refresh rate")
                self.assertTrue(result, "'+' should return True to continue")
                
                # Test decrease speed (faster refresh)
                result = monitor._handle_keyboard_input('-')
                decreased_rate = monitor.manual_refresh_rate
                print(f"  '-' key: {increased_rate:.1f}s → {decreased_rate:.1f}s {'✓' if decreased_rate < increased_rate else '✗'}")
                self.assertLess(decreased_rate, increased_rate, "'-' should decrease refresh rate")
                self.assertTrue(result, "'-' should return True to continue")
                
                # Test adaptive mode toggle
                initial_adaptive = monitor.adaptive_mode
                result = monitor._handle_keyboard_input('a')
                toggled_adaptive = monitor.adaptive_mode
                print(f"  'a' key: Adaptive {initial_adaptive} → {toggled_adaptive} {'✓' if toggled_adaptive != initial_adaptive else '✗'}")
                self.assertNotEqual(toggled_adaptive, initial_adaptive, "'a' should toggle adaptive mode")
                self.assertTrue(result, "'a' should return True to continue")
                
                # Test reset
                monitor.manual_refresh_rate = 5.0  # Set to different value
                monitor.adaptive_mode = False
                result = monitor._handle_keyboard_input('r')
                reset_rate = monitor.manual_refresh_rate
                reset_adaptive = monitor.adaptive_mode
                print(f"  'r' key: Rate reset to {reset_rate:.1f}s, Adaptive: {reset_adaptive} {'✓' if reset_rate == 1.0 and reset_adaptive else '✗'}")
                self.assertEqual(reset_rate, 1.0, "'r' should reset to base rate")
                self.assertTrue(reset_adaptive, "'r' should reset adaptive mode")
                self.assertTrue(result, "'r' should return True to continue")
                
                # Test quit
                result = monitor._handle_keyboard_input('q')
                print(f"  'q' key: Returns {result} {'✓' if not result else '✗'}")
                self.assertFalse(result, "'q' should return False to quit")
        
        print("✓ Keyboard controls test passed")
        
    def test_metrics_and_trends(self):
        """Test 5 - Metrics collection and trend calculation"""
        print("\n=== Test 5: Metrics and Trends ===")
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open_registry(self.test_agent_registry)):
                monitor = EnhancedMonitor(self.config_path)
                
                # Test trend calculation with various patterns
                from collections import deque
                
                # Test upward trend
                monitor.history['cpu'] = deque([50, 60, 70], maxlen=60)
                upward_trend = monitor._get_trend(monitor.history['cpu'])
                print(f"  Upward trend (50→60→70): {upward_trend} {'✓' if upward_trend == '↑' else '✗'}")
                self.assertEqual(upward_trend, '↑', "Should detect upward trend")
                
                # Test downward trend
                monitor.history['memory'] = deque([70, 60, 50], maxlen=60)
                downward_trend = monitor._get_trend(monitor.history['memory'])
                print(f"  Downward trend (70→60→50): {downward_trend} {'✓' if downward_trend == '↓' else '✗'}")
                self.assertEqual(downward_trend, '↓', "Should detect downward trend")
                
                # Test stable trend
                monitor.history['network'] = deque([50, 49, 51], maxlen=60)
                stable_trend = monitor._get_trend(monitor.history['network'])
                print(f"  Stable trend (50→49→51): {stable_trend} {'✓' if stable_trend == '→' else '✗'}")
                self.assertEqual(stable_trend, '→', "Should detect stable trend")
                
                # Test insufficient data
                monitor.history['test'] = deque([50], maxlen=60)
                insufficient_trend = monitor._get_trend(monitor.history['test'])
                print(f"  Insufficient data: {insufficient_trend} {'✓' if insufficient_trend == '→' else '✗'}")
                self.assertEqual(insufficient_trend, '→', "Should default to stable with insufficient data")
        
        print("✓ Metrics and trends test passed")
        
    def test_gpu_detection(self):
        """Test 6 - GPU detection and WSL2 compatibility"""
        print("\n=== Test 6: GPU Detection ===")
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open_registry(self.test_agent_registry)):
                monitor = EnhancedMonitor(self.config_path)
                
                # Test WSL environment detection
                wsl_info = monitor._detect_wsl_environment()
                print(f"  WSL Detection: WSL={wsl_info['is_wsl']}, WSL2={wsl_info['is_wsl2']}")
                
                # Test GPU availability
                gpu_available = monitor.gpu_available
                gpu_stats = monitor.get_gpu_stats()
                print(f"  GPU Available: {gpu_available}")
                print(f"  GPU Name: {gpu_stats.get('name', 'N/A')}")
                print(f"  GPU Driver Type: {monitor.gpu_driver_type or 'None'}")
                
                # Verify GPU stats structure
                required_keys = ['available', 'usage', 'memory', 'temperature', 'name']
                for key in required_keys:
                    self.assertIn(key, gpu_stats, f"GPU stats should contain '{key}' key")
                
                # GPU stats should be valid numbers or zero
                if gpu_stats['available']:
                    self.assertIsInstance(gpu_stats['usage'], (int, float), "GPU usage should be numeric")
                    self.assertIsInstance(gpu_stats['memory'], (int, float), "GPU memory should be numeric")
                    self.assertIsInstance(gpu_stats['temperature'], (int, float), "GPU temperature should be numeric")
        
        print("✓ GPU detection test passed")
        
    def test_error_handling(self):
        """Test 7 - Error handling and graceful degradation"""
        print("\n=== Test 7: Error Handling ===")
        
        # Test with invalid config file
        invalid_config_path = os.path.join(self.temp_dir, 'invalid_config.json')
        with open(invalid_config_path, 'w') as f:
            f.write("invalid json content")
        
        try:
            with patch('pathlib.Path.exists', return_value=True):
                with patch('builtins.open', mock_open_registry(self.test_agent_registry)):
                    monitor = EnhancedMonitor(invalid_config_path)
                    # Should fall back to default config
                    self.assertIsNotNone(monitor.config, "Should use default config on invalid file")
                    print("  ✓ Invalid config handling: Graceful fallback to defaults")
        except Exception as e:
            self.fail(f"Should handle invalid config gracefully: {e}")
        
        # Test with missing agent registry
        with patch('pathlib.Path.exists', return_value=False):
            try:
                monitor = EnhancedMonitor(self.config_path)
                # Should handle missing registry gracefully
                self.assertEqual(monitor.agent_registry, {'agents': {}}, 
                               "Should use empty registry on missing file")
                print("  ✓ Missing agent registry: Graceful fallback to empty registry")
            except Exception as e:
                self.fail(f"Should handle missing registry gracefully: {e}")
        
        # Test network calculation with edge cases
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open_registry(self.test_agent_registry)):
                monitor = EnhancedMonitor(self.config_path)
                
                # Mock network data
                class MockNetwork:
                    def __init__(self):
                        self.bytes_sent = 1000
                        self.bytes_recv = 2000
                
                # Test with no baseline
                network_stats = monitor._calculate_network_stats(MockNetwork())
                self.assertEqual(network_stats['bandwidth_mbps'], 0.0, 
                               "Should handle no baseline gracefully")
                print("  ✓ Network calculation: Handles missing baseline")
                
                # Test with very small time difference
                monitor.network_baseline = {
                    'bytes_sent': 1000,
                    'bytes_recv': 2000,
                    'timestamp': time.time()
                }
                network_stats = monitor._calculate_network_stats(MockNetwork())
                self.assertIsInstance(network_stats['bandwidth_mbps'], (int, float), 
                                    "Should handle small time differences")
                print("  ✓ Network calculation: Handles small time differences")
        
        print("✓ Error handling test passed")
        
    def test_display_name_truncation(self):
        """Test 8 - Display name truncation and formatting"""
        print("\n=== Test 8: Display Name Truncation ===")
        
        with patch('pathlib.Path.exists', return_value=True):
            with patch('builtins.open', mock_open_registry(self.test_agent_registry)):
                monitor = EnhancedMonitor(self.config_path)
                
                test_cases = [
                    ('short-name', 'short-name'),
                    ('hardware-resource-optimizer', 'hw-resource-optim'),
                    ('very-long-agent-name-that-exceeds-limit', 'very-long-agent-name'),
                    ('ai-senior-full-stack-developer-advanced', 'ai-senior-full-stack'),
                ]
                
                for input_name, expected_output in test_cases:
                    result = monitor._get_display_name(input_name)
                    max_length = 20
                    print(f"  '{input_name}' → '{result}' (len: {len(result)}) {'✓' if len(result) <= max_length else '✗'}")
                    self.assertLessEqual(len(result), max_length, 
                                       f"Display name should not exceed {max_length} characters")
                    if input_name == expected_output:
                        self.assertEqual(result, expected_output, 
                                       f"Short names should remain unchanged")
        
        print("✓ Display name truncation test passed")
        
    def run_comprehensive_tests(self):
        """Run all tests and generate a comprehensive report"""
        print("=" * 60)
        print("COMPREHENSIVE STATIC MONITOR TEST SUITE")
        print("=" * 60)
        
        test_results = []
        
        tests = [
            ("Agent Type Detection", self.test_agent_type_detection),
            ("Port Detection & Health Checks", self.test_port_detection_and_health_checks),
            ("Adaptive Refresh Functionality", self.test_adaptive_refresh_functionality),
            ("Keyboard Controls", self.test_keyboard_controls),
            ("Metrics and Trends", self.test_metrics_and_trends),
            ("GPU Detection", self.test_gpu_detection),
            ("Error Handling", self.test_error_handling),
            ("Display Name Truncation", self.test_display_name_truncation),
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for test_name, test_method in tests:
            try:
                test_method()
                test_results.append((test_name, "PASSED", None))
                passed_tests += 1
            except Exception as e:
                test_results.append((test_name, "FAILED", str(e)))
                failed_tests += 1
                print(f"✗ {test_name} FAILED: {e}")
        
        # Generate final report
        print("\n" + "=" * 60)
        print("TEST RESULTS SUMMARY")
        print("=" * 60)
        
        for test_name, status, error in test_results:
            status_symbol = "✓" if status == "PASSED" else "✗"
            print(f"{status_symbol} {test_name:<35} {status}")
            if error:
                print(f"    Error: {error}")
        
        print("\n" + "-" * 60)
        print(f"Total Tests: {len(tests)}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {failed_tests}")
        print(f"Success Rate: {(passed_tests/len(tests)*100):.1f}%")
        
        if failed_tests == 0:
            print("\n🎉 ALL TESTS PASSED! Static monitor is functioning correctly.")
        else:
            print(f"\n⚠️  {failed_tests} test(s) failed. Please review the issues above.")
        
        return passed_tests, failed_tests

def mock_open_registry(registry_data):
    """Helper function to mock file opening for agent registry"""
    def mock_open_func(file_path, mode='r'):
        if 'agent_registry.json' in str(file_path):
            from io import StringIO
            return StringIO(json.dumps(registry_data))
        else:
            # For other files, use the actual open
            return open(file_path, mode)
    return mock_open_func

def run_integration_test():
    """Run a quick integration test with the actual monitor"""
    print("\n" + "=" * 60)
    print("INTEGRATION TEST - Testing with actual system")
    print("=" * 60)
    
    try:
        # Test creating monitor with actual system
        monitor = EnhancedMonitor()
        print("✓ Monitor created successfully")
        
        # Test getting system stats
        stats = monitor.get_system_stats()
        print(f"✓ System stats retrieved: CPU={stats['cpu_percent']:.1f}%, MEM={stats['mem_percent']:.1f}%")
        
        # Test agent status
        agents, healthy, total = monitor.get_ai_agents_status()
        print(f"✓ Agent status retrieved: {healthy}/{total} agents healthy")
        
        # Test configuration loading
        print(f"✓ Configuration loaded: Refresh rate={monitor.current_refresh_rate:.1f}s")
        
        # Test GPU detection
        gpu_stats = monitor.get_gpu_stats()
        gpu_status = "Available" if gpu_stats['available'] else "Not Available"
        print(f"✓ GPU detection: {gpu_status} ({gpu_stats['name']})")
        
        # Clean up
        monitor.cleanup()
        print("✓ Monitor cleanup completed")
        
        print("\n🎉 INTEGRATION TEST PASSED!")
        return True
        
    except Exception as e:
        print(f"✗ INTEGRATION TEST FAILED: {e}")
        return False

if __name__ == "__main__":
    # Run comprehensive test suite
    test_suite = TestStaticMonitor()
    test_suite.setUp()
    
    try:
        passed, failed = test_suite.run_comprehensive_tests()
        
        # Run integration test
        integration_passed = run_integration_test()
        
        # Final summary
        print("\n" + "=" * 60)
        print("FINAL VALIDATION SUMMARY")
        print("=" * 60)
        
        if failed == 0 and integration_passed:
            print("🌟 STATIC MONITOR VALIDATION: COMPLETE SUCCESS")
            print("   All functionality tested and working correctly.")
            sys.exit(0)
        else:
            print("⚠️  STATIC MONITOR VALIDATION: ISSUES FOUND")
            print("   Some tests failed. Please review and fix issues.")
            sys.exit(1)
            
    finally:
        test_suite.tearDown()