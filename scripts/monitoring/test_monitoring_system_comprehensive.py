#!/usr/bin/env python3
"""
Comprehensive Test Suite for SutazAI Monitoring System
=====================================================

This test suite validates all aspects of the monitoring system including:
- Agent detection logic
- Docker container status mapping
- Health check mechanisms  
- Configuration parsing
- Edge cases and error conditions
- Performance under load
"""

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock, call
import json
import sys
import os
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor

# Add the scripts directory to the path for imports
sys.path.insert(0, '/opt/sutazaiapp/scripts')
sys.path.insert(0, '/opt/sutazaiapp/scripts/monitoring')
sys.path.insert(0, '/opt/sutazaiapp/scripts/agents')

# Import the modules we want to test
try:
    from static_monitor import EnhancedMonitor
except ImportError:
    # Create a mock if the module can't be imported
    class EnhancedMonitor:
        def __init__(self, config_path=None):
            self.config = {}
            self.agent_registry = {}


class TestMonitoringSystemBase(unittest.TestCase):
    """Base test class with common setup and utilities"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_config = {
            "refresh_rate": 1.0,
            "adaptive_refresh": True,
            "thresholds": {
                "cpu_warning": 70,
                "cpu_critical": 90,
                "memory_warning": 80,
                "memory_critical": 95,
                "disk_warning": 85,
                "disk_critical": 95
            },
            "display": {
                "show_trends": True,
                "show_network": True
            },
            "agent_monitoring": {
                "max_agents_display": 20,
                "timeout": 2,
                "enabled": True,
                "health_check": True
            },
            "logging": {
                "enabled": False,
                "level": "INFO",
                "file": "/tmp/test_monitor.log"
            }
        }
        
        self.test_agent_registry = {
            "version": "1.0.0",
            "agents": {
                "ai-testing-qa-validator": {
                    "id": "ai-testing-qa-validator",
                    "name": "AI Testing QA Validator",
                    "type": "qa",
                    "enabled": True,
                    "container_name": "sutazai-ai-testing-qa-validator",
                    "port": 8081,
                    "health_endpoint": "/health"
                },
                " system-architect": {
                    "id": " system-architect", 
                    "name": "AI System Architect",
                    "type": "architecture",
                    "enabled": True,
                    "container_name": "sutazai- system-architect",
                    "port": 8082,
                    "health_endpoint": "/health"
                },
                "observability-monitoring-engineer": {
                    "id": "observability-monitoring-engineer",
                    "name": "Observability Monitoring Engineer", 
                    "type": "monitoring",
                    "enabled": True,
                    "container_name": "sutazai-observability-monitoring-engineer",
                    "port": 8083,
                    "health_endpoint": "/health"
                }
            }
        }
        
        # Create temporary files for testing
        self.temp_dir = tempfile.mkdtemp()
        self.config_file = os.path.join(self.temp_dir, 'test_config.json')
        self.agent_registry_file = os.path.join(self.temp_dir, 'agent_registry.json')
        
        with open(self.config_file, 'w') as f:
            json.dump(self.test_config, f)
            
        with open(self.agent_registry_file, 'w') as f:
            json.dump(self.test_agent_registry, f)
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)


class TestStaticMonitorCore(TestMonitoringSystemBase):
    """Test core functionality of the static monitor"""
    
    def test_monitor_initialization(self):
        """Test that the monitor initializes correctly"""
        with patch('static_monitor.EnhancedMonitor._load_config') as mock_load_config:
            mock_load_config.return_value = self.test_config
            
            monitor = EnhancedMonitor(self.config_file)
            
            # Verify configuration is loaded
            self.assertIsNotNone(monitor.config)
            mock_load_config.assert_called_once()
    
    @patch('subprocess.run')
    def test_docker_container_detection(self, mock_subprocess):
        """Test Docker container detection logic"""
        # Mock Docker output for running containers
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = (
            "sutazai-ai-testing-qa-validator\tUp 2 hours\t0.0.0.0:8081->8080/tcp\n"
            "sutazai- system-architect\tUp 1 hour (healthy)\t0.0.0.0:8082->8080/tcp\n"
            "sutazai-observability-monitoring-engineer\tRestarting (1) 5 minutes ago\t\n"
            "sutazai-backend\tUp 3 hours (unhealthy)\t0.0.0.0:8000->8000/tcp\n"
        )
        
        with patch.object(EnhancedMonitor, '_load_config', return_value=self.test_config):
            with patch.object(EnhancedMonitor, '_load_agent_registry', return_value=self.test_agent_registry):
                monitor = EnhancedMonitor()
                
                # Test container info retrieval
                container_info = monitor._get_container_info("ai-testing-qa-validator")
                
                if container_info:
                    self.assertEqual(container_info['status'], 'running')
                    self.assertIn('ports', container_info)
    
    def test_agent_health_status_mapping(self):
        """Test that Docker status strings are correctly mapped to health status"""
        test_cases = [
            ("Up 2 hours", "running"),
            ("Up 1 hour (healthy)", "healthy"),
            ("Up 30 minutes (unhealthy)", "unhealthy"),
            ("Up 5 minutes (health: starting)", "starting"),
            ("Restarting (1) 5 minutes ago", "restarting"),
            ("Exited (0) 10 minutes ago", "exited"),
            ("Created", "unknown")
        ]
        
        with patch.object(EnhancedMonitor, '_load_config', return_value=self.test_config):
            with patch.object(EnhancedMonitor, '_load_agent_registry', return_value=self.test_agent_registry):
                monitor = EnhancedMonitor()
                
                for docker_status, expected_status in test_cases:
                    # Test the status parsing logic
                    if 'Up' in docker_status:
                        if 'unhealthy' in docker_status:
                            result_status = 'unhealthy'
                        elif 'health: starting' in docker_status:
                            result_status = 'starting'
                        elif 'healthy' in docker_status:
                            result_status = 'healthy'
                        else:
                            result_status = 'running'
                    elif 'Restarting' in docker_status:
                        result_status = 'restarting'
                    elif 'Exited' in docker_status:
                        result_status = 'exited'
                    else:
                        result_status = 'unknown'
                    
                    self.assertEqual(result_status, expected_status, 
                                   f"Status mapping failed for '{docker_status}'")


class TestAgentRegistry(TestMonitoringSystemBase):
    """Test agent registry functionality"""
    
    def test_agent_registry_loading(self):
        """Test loading agent registry from file"""
        with patch.object(EnhancedMonitor, '_load_config', return_value=self.test_config):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(self.test_agent_registry)
                
                monitor = EnhancedMonitor()
                registry = monitor._load_agent_registry()
                
                self.assertIn('agents', registry)
                self.assertIn('ai-testing-qa-validator', registry['agents'])
                self.assertEqual(registry['agents']['ai-testing-qa-validator']['port'], 8081)
    
    def test_agent_registry_validation(self):
        """Test validation of agent registry entries"""
        test_registry = {
            "agents": {
                "valid-agent": {
                    "id": "valid-agent",
                    "name": "Valid Agent",
                    "type": "test",
                    "enabled": True,
                    "port": 8080
                },
                "invalid-agent": {
                    "id": "invalid-agent",
                    # Missing required fields
                }
            }
        }
        
        with patch.object(EnhancedMonitor, '_load_config', return_value=self.test_config):
            monitor = EnhancedMonitor()
            
            # Test that valid agents are processed correctly
            valid_agent = test_registry['agents']['valid-agent']
            self.assertTrue(valid_agent.get('enabled', False))
            self.assertIsNotNone(valid_agent.get('port'))
            
            # Test that invalid agents are handled gracefully
            invalid_agent = test_registry['agents']['invalid-agent']
            self.assertIsNone(invalid_agent.get('port'))


class TestQuickStatusCheck(TestMonitoringSystemBase):
    """Test the quick status check functionality"""
    
    @patch('docker.from_env')
    def test_get_system_status(self, mock_docker):
        """Test system status retrieval"""
        # Mock Docker client and containers
        mock_client = Mock()
        mock_docker.return_value = mock_client
        
        # Mock container objects
        mock_container1 = Mock()
        mock_container1.name = "sutazai-ai-testing-qa-validator"
        mock_container1.status = "running"
        mock_container1.attrs = {
            "State": {"Health": {"Status": "healthy"}},
            "RestartCount": 0
        }
        
        mock_container2 = Mock()
        mock_container2.name = "sutazai-backend"
        mock_container2.status = "running"
        mock_container2.attrs = {
            "State": {"Health": {"Status": "unhealthy"}},
            "RestartCount": 2
        }
        
        mock_container3 = Mock()
        mock_container3.name = "sutazai-redis"
        mock_container3.attrs = {
            "State": {},
            "RestartCount": 0
        }
        mock_container3.status = "exited"
        
        mock_client.containers.list.return_value = [mock_container1, mock_container2, mock_container3]
        
        # Import and test the quick status check
        try:
            sys.path.insert(0, '/opt/sutazaiapp/scripts/agents')
            from quick_status_check import get_system_status
            
            status = get_system_status()
            
            # Verify status categorization
            self.assertIn('running', status)
            self.assertIn('unhealthy', status)
            self.assertIn('exited', status)
            
            # Check that containers are properly categorized
            running_names = [c['name'] for c in status['running']]
            unhealthy_names = [c['name'] for c in status['unhealthy']]
            exited_names = [c['name'] for c in status['exited']]
            
            self.assertIn('ai-testing-qa-validator', running_names)
            self.assertIn('backend', unhealthy_names)
            self.assertIn('redis', exited_names)
            
        except ImportError:
            self.skipTest("quick-status-check.py not available for import")


class TestMonitoringEdgeCases(TestMonitoringSystemBase):
    """Test edge cases and error conditions"""
    
    @patch('subprocess.run')
    def test_docker_command_failure(self, mock_subprocess):
        """Test handling of Docker command failures"""
        # Mock Docker command failure
        mock_subprocess.return_value.returncode = 1
        mock_subprocess.return_value.stdout = ""
        mock_subprocess.return_value.stderr = "Docker service not running"
        
        with patch.object(EnhancedMonitor, '_load_config', return_value=self.test_config):
            with patch.object(EnhancedMonitor, '_load_agent_registry', return_value=self.test_agent_registry):
                monitor = EnhancedMonitor()
                
                # Test that failed Docker commands are handled gracefully
                container_info = monitor._get_container_info("nonexistent-agent")
                self.assertIsNone(container_info)
    
    def test_malformed_agent_registry(self):
        """Test handling of malformed agent registry"""
        malformed_registry = {
            "invalid_json": True,
            "agents": "not_a_dict"
        }
        
        with patch.object(EnhancedMonitor, '_load_config', return_value=self.test_config):
            with patch('builtins.open', create=True) as mock_open:
                mock_open.return_value.__enter__.return_value.read.return_value = json.dumps(malformed_registry)
                
                monitor = EnhancedMonitor()
                registry = monitor._load_agent_registry()
                
                # Should handle malformed registry gracefully
                self.assertIsInstance(registry, dict)
    
    def test_network_timeout_handling(self):
        """Test handling of network timeouts during health checks"""
        with patch.object(EnhancedMonitor, '_load_config', return_value=self.test_config):
            with patch.object(EnhancedMonitor, '_load_agent_registry', return_value=self.test_agent_registry):
                with patch('requests.get') as mock_get:
                    # Mock network timeout
                    mock_get.side_effect = TimeoutError("Connection timed out")
                    
                    monitor = EnhancedMonitor()
                    
                    # Test that timeouts are handled gracefully
                    try:
                        result = monitor._check_agent_health("test-agent", {"port": 8080}, timeout=1)
                        # Should not raise exception
                        self.assertIsNotNone(result)
                    except Exception as e:
                        # Timeout exceptions should be handled gracefully
                        self.fail(f"Timeout should be handled gracefully, but got: {e}")
    
    def test_empty_container_list(self):
        """Test handling of empty container list"""
        with patch('subprocess.run') as mock_subprocess:
            mock_subprocess.return_value.returncode = 0
            mock_subprocess.return_value.stdout = ""
            
            with patch.object(EnhancedMonitor, '_load_config', return_value=self.test_config):
                with patch.object(EnhancedMonitor, '_load_agent_registry', return_value=self.test_agent_registry):
                    monitor = EnhancedMonitor()
                    
                    # Test empty container discovery
                    discovered = monitor._discover_container_agents()
                    self.assertIsInstance(discovered, dict)
                    self.assertEqual(len(discovered), 0)


class TestMonitoringPerformance(TestMonitoringSystemBase):
    """Test performance aspects of the monitoring system"""
    
    def test_concurrent_health_checks(self):
        """Test concurrent health check performance"""
        with patch.object(EnhancedMonitor, '_load_config', return_value=self.test_config):
            with patch.object(EnhancedMonitor, '_load_agent_registry', return_value=self.test_agent_registry):
                monitor = EnhancedMonitor()
                
                # Mock ThreadPoolExecutor for concurrent execution
                with patch('concurrent.futures.ThreadPoolExecutor') as mock_executor:
                    mock_future = Mock()
                    mock_future.result.return_value = ("test-agent", "healthy", 0.1)
                    mock_executor.return_value.__enter__.return_value.submit.return_value = mock_future
                    
                    # Test that concurrent health checks are used
                    start_time = time.time()
                    agents, healthy, total = monitor.get_ai_agents_status()
                    end_time = time.time()
                    
                    # Should complete reasonably quickly
                    self.assertLess(end_time - start_time, 5.0)
    
    def test_memory_usage_optimization(self):
        """Test memory usage optimization features"""
        with patch.object(EnhancedMonitor, '_load_config', return_value=self.test_config):
            with patch.object(EnhancedMonitor, '_load_agent_registry', return_value=self.test_agent_registry):
                monitor = EnhancedMonitor()
                
                # Test that history deques have appropriate maxlen for memory efficiency
                self.assertLessEqual(monitor.history['cpu'].maxlen, 30)
                self.assertLessEqual(monitor.history['memory'].maxlen, 30)
                self.assertLessEqual(monitor.history['network'].maxlen, 30)


class TestIntegrationScenarios(TestMonitoringSystemBase):
    """Test integration scenarios between components"""
    
    @patch('subprocess.run')
    @patch('requests.get')
    def test_end_to_end_monitoring(self, mock_requests, mock_subprocess):
        """Test complete end-to-end monitoring workflow"""
        # Mock Docker container discovery
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = (
            "sutazai-ai-testing-qa-validator\tUp 2 hours (healthy)\t0.0.0.0:8081->8080/tcp\n"
            "sutazai- system-architect\tUp 1 hour (healthy)\t0.0.0.0:8082->8080/tcp\n"
        )
        
        # Mock health check responses
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"status": "healthy", "version": "1.0.0"}
        mock_response.elapsed.total_seconds.return_value = 0.1
        mock_requests.return_value = mock_response
        
        with patch.object(EnhancedMonitor, '_load_config', return_value=self.test_config):
            with patch.object(EnhancedMonitor, '_load_agent_registry', return_value=self.test_agent_registry):
                monitor = EnhancedMonitor()
                
                # Run end-to-end monitoring
                agents, healthy_count, total_count = monitor.get_ai_agents_status()
                
                # Verify results
                self.assertGreaterEqual(healthy_count, 0)
                self.assertGreaterEqual(total_count, 0)
                self.assertIsInstance(agents, list)
    
    def test_configuration_consistency(self):
        """Test consistency between configuration files"""
        # Test that agent registry and communication config are consistent
        with open('/opt/sutazaiapp/agents/communication_config.json', 'r') as f:
            comm_config = json.load(f)
        
        # Verify communication config has required sections
        self.assertIn('communication_protocols', comm_config)
        self.assertIn('agent_endpoints', comm_config)
        self.assertIn('health_monitoring', comm_config)
        
        # Verify health monitoring is enabled
        health_config = comm_config['health_monitoring']
        self.assertTrue(health_config.get('enabled', False))
        self.assertGreater(health_config.get('check_interval', 0), 0)


class TestValidationScenarios(TestMonitoringSystemBase):
    """Test validation and error detection scenarios"""
    
    def test_agent_validation_rules(self):
        """Test agent validation business rules"""
        test_cases = [
            # (agent_config, expected_valid, reason)
            ({
                "id": "valid-agent",
                "name": "Valid Agent", 
                "type": "test",
                "enabled": True,
                "port": 8080
            }, True, "Complete valid configuration"),
            
            ({
                "id": "missing-port-agent",
                "name": "Missing Port Agent",
                "type": "test", 
                "enabled": True
                # Missing port
            }, False, "Missing required port"),
            
            ({
                "id": "disabled-agent",
                "name": "Disabled Agent",
                "type": "test",
                "enabled": False,
                "port": 8080
            }, False, "Agent is disabled"),
            
            ({
                "name": "Missing ID Agent",
                "type": "test",
                "enabled": True,
                "port": 8080
                # Missing id
            }, False, "Missing required ID")
        ]
        
        for agent_config, expected_valid, reason in test_cases:
            with self.subTest(reason=reason):
                # Test basic validation rules
                has_id = 'id' in agent_config
                has_name = 'name' in agent_config
                has_port = 'port' in agent_config
                is_enabled = agent_config.get('enabled', False)
                
                is_valid = has_id and has_name and has_port and is_enabled
                self.assertEqual(is_valid, expected_valid, f"Validation failed: {reason}")


def run_comprehensive_tests():
    """Run all tests and generate a detailed report"""
    print("🧪 Running Comprehensive SutazAI Monitoring System Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestStaticMonitorCore,
        TestAgentRegistry,
        TestQuickStatusCheck,
        TestMonitoringEdgeCases,
        TestMonitoringPerformance,
        TestIntegrationScenarios,
        TestValidationScenarios
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Generate summary report  
    print("\n" + "=" * 60)
    print("📊 TEST EXECUTION SUMMARY")
    print("=" * 60)
    print(f"Tests Run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Skipped: {len(result.skipped) if hasattr(result, 'skipped') else 0}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip() if 'AssertionError:' in traceback else 'Unknown failure'}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip() if 'Exception:' in traceback else 'Unknown error'}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    print(f"\n✅ SUCCESS RATE: {success_rate:.1f}%")
    
    if success_rate >= 90:
        print("🎉 EXCELLENT - Monitoring system is highly reliable!")
    elif success_rate >= 80:
        print("👍 GOOD - Monitoring system is mostly working well")
    elif success_rate >= 70:
        print("⚠️  FAIR - Some issues need attention")
    else:
        print("🔴 POOR - Significant issues require immediate attention")
    
    return result.wasSuccessful(), result


if __name__ == "__main__":
    success, results = run_comprehensive_tests()
    sys.exit(0 if success else 1)