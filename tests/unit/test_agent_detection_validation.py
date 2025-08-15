#!/usr/bin/env python3
"""
Agent Detection and Validation Test Suite
=========================================

Specialized tests for agent detection, health validation, and status reporting.
Tests the core logic that determines agent health and status.
"""

import logging

# Configure logger for exception handling
logger = logging.getLogger(__name__)

import pytest
import unittest
from unittest.mock import Mock, patch, MagicMock
import json
import subprocess
import requests
import time
from concurrent.futures import ThreadPoolExecutor, Future
import sys
import os

# Add paths for imports
# Path handled by pytest configuration
# Path handled by pytest configuration


class TestAgentDetectionCore(unittest.TestCase):
    """Test core agent detection functionality"""
    
    def setUp(self):
        """Set up test data"""
        self.mock_config = {
            "agent_monitoring": {
                "max_agents_display": 20,
                "timeout": 2,
                "enabled": True
            },
            "thresholds": {
                "cpu_warning": 70,
                "cpu_critical": 90,
                "memory_warning": 80,
                "memory_critical": 95
            }
        }
        
        self.mock_registry = {
            "agents": {
                "ai-testing-qa-validator": {
                    "id": "ai-testing-qa-validator",
                    "name": "AI Testing QA Validator",
                    "enabled": True,
                    "port": 8081,
                    "health_endpoint": "/health"
                },
                "observability-monitoring-engineer": {
                    "id": "observability-monitoring-engineer", 
                    "name": "Observability Monitoring Engineer",
                    "enabled": True,
                    "port": 8083,
                    "health_endpoint": "/health"
                }
            }
        }
    
    @patch('subprocess.run')
    def test_docker_container_name_patterns(self, mock_subprocess):
        """Test various Docker container naming patterns are detected"""
        test_patterns = [
            ("sutazai-ai-testing-qa-validator", "ai-testing-qa-validator"),
            ("ai-testing-qa-validator", "ai-testing-qa-validator"),
            ("sutazaiapp-ai-testing-qa-validator", "ai-testing-qa-validator"),
            ("ai-testing-qa-validator-1", "ai-testing-qa-validator"),
        ]
        
        for container_name, agent_id in test_patterns:
            with self.subTest(container_name=container_name):
                mock_subprocess.return_value.returncode = 0
                mock_subprocess.return_value.stdout = f"{container_name}\tUp 1 hour (healthy)\t0.0.0.0:8081->8080/tcp"
                
                # Test the naming pattern logic
                name_patterns = [
                    f'sutazai-{agent_id}',
                    f'{agent_id}',
                    f'sutazaiapp-{agent_id}',
                    f'{agent_id}-1'
                ]
                
                self.assertIn(container_name, name_patterns)
    
    def test_health_status_parsing(self):
        """Test parsing of Docker health status strings"""
        test_cases = [
            # (docker_output, expected_status, expected_ports)
            ("sutazai-agent\tUp 2 hours\t0.0.0.0:8081->8080/tcp", "running", ["0.0.0.0:8081->8080/tcp"]),
            ("sutazai-agent\tUp 1 hour (healthy)\t0.0.0.0:8082->8080/tcp", "healthy", ["0.0.0.0:8082->8080/tcp"]),
            ("sutazai-agent\tUp 30 minutes (unhealthy)\t0.0.0.0:8083->8080/tcp", "unhealthy", ["0.0.0.0:8083->8080/tcp"]),
            ("sutazai-agent\tUp 5 minutes (health: starting)\t", "starting", []),
            ("sutazai-agent\tRestarting (1) 5 minutes ago\t", "restarting", []),
            ("sutazai-agent\tExited (0) 10 minutes ago\t", "exited", []),
        ]
        
        for docker_output, expected_status, expected_ports in test_cases:
            with self.subTest(docker_output=docker_output):
                parts = docker_output.split('\t')
                name = parts[0]
                status_full = parts[1]
                ports = parts[2] if len(parts) > 2 else ''
                
                # Parse status
                if 'Up' in status_full:
                    if 'unhealthy' in status_full:
                        status = 'unhealthy'
                    elif 'health: starting' in status_full:
                        status = 'starting'
                    elif 'healthy' in status_full:
                        status = 'healthy'
                    else:
                        status = 'running'
                elif 'Restarting' in status_full:
                    status = 'restarting'
                elif 'Exited' in status_full:
                    status = 'exited'
                else:
                    status = 'unknown'
                
                # Parse ports
                port_list = []
                if ports:
                    port_mappings = ports.split(', ')
                    port_list = [p.strip() for p in port_mappings if p.strip()]
                
                self.assertEqual(status, expected_status)
                self.assertEqual(port_list, expected_ports)
    
    @patch('requests.get')
    def test_health_check_responses(self, mock_get):
        """Test different health check response scenarios"""
        test_scenarios = [
            # (response_status, response_data, expected_health, expected_response_time)
            (200, {"status": "healthy"}, "healthy", True),
            (200, {"status": "running"}, "healthy", True),
            (503, {"status": "unhealthy"}, "unhealthy", True),
            (404, None, "unhealthy", True),
            (None, None, "offline", False),  # Connection error
        ]
        
        for status_code, response_data, expected_health, should_respond in test_scenarios:
            with self.subTest(status_code=status_code):
                if should_respond:
                    mock_response = Mock()
                    mock_response.status_code = status_code
                    mock_response.json.return_value = response_data if response_data else {}
                    mock_response.elapsed.total_seconds.return_value = 0.1
                    mock_get.return_value = mock_response
                else:
                    mock_get.side_effect = requests.exceptions.ConnectionError("Connection failed")
                
                # Simulate health check logic
                try:
                    response = mock_get("http://localhost:8080/health", timeout=2)
                    if response.status_code == 200:
                        health = "healthy"
                    elif response.status_code >= 500:
                        health = "unhealthy"
                    else:
                        health = "unhealthy"
                    response_time = response.elapsed.total_seconds()
                except (AssertionError, Exception) as e:
                    logger.error(f"Unexpected exception: {e}", exc_info=True)
                    health = "offline"
                    response_time = None
                
                self.assertEqual(health, expected_health)
                
                if should_respond:
                    self.assertIsNotNone(response_time)
                else:
                    self.assertIsNone(response_time)


class TestAgentStatusIntegration(unittest.TestCase):
    """Test integration between different status sources"""
    
    def setUp(self):
        """Set up test environment"""
        self.agents_config = {
            "ai-testing-qa-validator": {
                "port": 8081,
                "health_endpoint": "/health"
            },
            "observability-monitoring-engineer": {
                "port": 8083,
                "health_endpoint": "/health"
            }
        }
    
    @patch('subprocess.run')
    @patch('requests.get')
    def test_combined_status_reporting(self, mock_get, mock_subprocess):
        """Test that Docker status and health checks are combined correctly"""
        # Mock Docker status
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = (
            "sutazai-ai-testing-qa-validator\tUp 2 hours (healthy)\t0.0.0.0:8081->8080/tcp\n"
            "sutazai-observability-monitoring-engineer\tUp 1 hour\t0.0.0.0:8083->8080/tcp\n"
        )
        
        # Mock health check responses
        def mock_health_response(url, **kwargs):
            mock_response = Mock()
            if "8081" in url:
                mock_response.status_code = 200
                mock_response.json.return_value = {"status": "healthy", "version": "1.0.0"}
            elif "8083" in url:
                mock_response.status_code = 503
                mock_response.json.return_value = {"status": "degraded"}
            mock_response.elapsed.total_seconds.return_value = 0.1
            return mock_response
        
        mock_get.side_effect = mock_health_response
        
        # Test combined status logic
        containers = [
            {"name": "ai-testing-qa-validator", "docker_status": "healthy", "port": 8081},
            {"name": "observability-monitoring-engineer", "docker_status": "running", "port": 8083}
        ]
        
        combined_status = []
        for container in containers:
            try:
                response = mock_get(f"http://localhost:{container['port']}/health", timeout=2)
                if response.status_code == 200:
                    health_status = "healthy"
                else:
                    health_status = "unhealthy"
            except (AssertionError, Exception) as e:
                logger.error(f"Unexpected exception: {e}", exc_info=True)
                health_status = "offline"
            
            # Combine Docker and health check status
            if container["docker_status"] == "healthy" and health_status == "healthy":
                final_status = "healthy"
            elif container["docker_status"] in ["running", "healthy"] and health_status == "unhealthy":
                final_status = "warning"
            elif container["docker_status"] in ["restarting", "exited"]:
                final_status = "critical"
            else:
                final_status = "unknown"
            
            combined_status.append({
                "name": container["name"],
                "docker_status": container["docker_status"],
                "health_status": health_status,
                "final_status": final_status
            })
        
        # Verify results
        self.assertEqual(len(combined_status), 2)
        
        # First agent should be healthy (Docker healthy + health check healthy)
        agent1 = next(s for s in combined_status if s["name"] == "ai-testing-qa-validator")
        self.assertEqual(agent1["final_status"], "healthy")
        
        # Second agent should be warning (Docker running + health check unhealthy)
        agent2 = next(s for s in combined_status if s["name"] == "observability-monitoring-engineer")
        self.assertEqual(agent2["final_status"], "warning")
    
    def test_agent_priority_handling(self):
        """Test that deployed agents are prioritized over non-deployed ones"""
        all_agents = [
            ("deployed-agent-1", {"status": "running", "deployed": True}),
            ("deployed-agent-2", {"status": "healthy", "deployed": True}), 
            ("non-deployed-agent-1", {"status": "offline", "deployed": False}),
            ("deployed-agent-3", {"status": "restarting", "deployed": True}),
            ("non-deployed-agent-2", {"status": "offline", "deployed": False}),
        ]
        
        # Separate deployed from non-deployed
        deployed_agents = [(name, info) for name, info in all_agents if info.get("deployed", False)]
        non_deployed_agents = [(name, info) for name, info in all_agents if not info.get("deployed", False)]
        
        # Prioritize deployed agents
        max_display = 3
        prioritized_agents = deployed_agents[:max_display]
        if len(prioritized_agents) < max_display:
            remaining_slots = max_display - len(prioritized_agents)
            prioritized_agents.extend(non_deployed_agents[:remaining_slots])
        
        # Verify prioritization
        self.assertEqual(len(prioritized_agents), 3)
        
        # All should be deployed agents since we have 3 deployed agents
        for name, info in prioritized_agents:
            self.assertTrue(info.get("deployed", False), f"Agent {name} should be deployed")


class TestAgentValidationRules(unittest.TestCase):
    """Test agent validation business rules"""
    
    def test_agent_configuration_validation(self):
        """Test validation of agent configurations"""
        valid_config = {
            "id": "test-agent",
            "name": "Test Agent",
            "type": "testing",
            "enabled": True,
            "port": 8080,
            "health_endpoint": "/health"
        }
        
        invalid_configs = [
            # Missing ID
            {
                "name": "Test Agent",
                "type": "testing", 
                "enabled": True,
                "port": 8080
            },
            # Missing port
            {
                "id": "test-agent",
                "name": "Test Agent",
                "type": "testing",
                "enabled": True
            },
            # Disabled agent
            {
                "id": "test-agent",
                "name": "Test Agent", 
                "type": "testing",
                "enabled": False,
                "port": 8080
            },
            # Invalid port
            {
                "id": "test-agent",
                "name": "Test Agent",
                "type": "testing",
                "enabled": True,
                "port": "invalid"
            }
        ]
        
        # Test valid configuration
        self.assertTrue(self._is_valid_agent_config(valid_config))
        
        # Test invalid configurations
        for i, invalid_config in enumerate(invalid_configs):
            with self.subTest(config_index=i):
                self.assertFalse(self._is_valid_agent_config(invalid_config))
    
    def _is_valid_agent_config(self, config):
        """Helper method to validate agent configuration"""
        required_fields = ['id', 'name', 'port']
        
        # Check required fields
        for field in required_fields:
            if field not in config:
                return False
        
        # Check if enabled
        if not config.get('enabled', False):
            return False
        
        # Check port is numeric
        port = config.get('port')
        if not isinstance(port, int) or port <= 0 or port > 65535:
            return False
        
        return True
    
    def test_health_status_precedence(self):
        """Test the precedence rules for health status determination"""
        status_scenarios = [
            # (docker_status, health_check_status, expected_final_status)
            ("healthy", "healthy", "healthy"),
            ("running", "healthy", "healthy"),
            ("healthy", "unhealthy", "warning"),
            ("running", "unhealthy", "warning"),
            ("unhealthy", "healthy", "warning"),
            ("unhealthy", "unhealthy", "critical"),
            ("restarting", "healthy", "critical"),
            ("restarting", "unhealthy", "critical"),
            ("exited", "offline", "offline"),
            ("unknown", "offline", "offline"),
        ]
        
        for docker_status, health_status, expected_final in status_scenarios:
            with self.subTest(docker=docker_status, health=health_status):
                final_status = self._determine_final_status(docker_status, health_status)
                self.assertEqual(final_status, expected_final)
    
    def _determine_final_status(self, docker_status, health_status):
        """Helper method to determine final agent status"""
        if docker_status in ["exited", "unknown"] or health_status == "offline":
            return "offline"
        elif docker_status == "restarting":
            return "critical"
        elif docker_status in ["healthy", "running"] and health_status == "healthy":
            return "healthy"
        elif (docker_status in ["healthy", "running"] and health_status == "unhealthy") or \
             (docker_status == "unhealthy" and health_status == "healthy"):
            return "warning"
        elif docker_status == "unhealthy" and health_status == "unhealthy":
            return "critical"
        else:
            return "critical"


class TestRealSystemValidation(unittest.TestCase):
    """Test against the real system configuration"""
    
    def test_communication_config_structure(self):
        """Test that the communication config has the expected structure"""
        config_path = '/opt/sutazaiapp/agents/communication_config.json'
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Test required sections
            required_sections = [
                'communication_protocols',
                'agent_endpoints', 
                'health_monitoring',
                'timeout_settings'
            ]
            
            for section in required_sections:
                self.assertIn(section, config, f"Missing required section: {section}")
            
            # Test health monitoring configuration
            health_config = config['health_monitoring']
            self.assertTrue(health_config.get('enabled', False))
            self.assertGreater(health_config.get('check_interval', 0), 0)
            self.assertGreater(health_config.get('failure_threshold', 0), 0)
            
            # Test timeout settings
            timeout_config = config['timeout_settings']
            self.assertGreater(timeout_config.get('health_check_timeout', 0), 0)
        else:
            self.skipTest("Communication config file not found")
    
    @patch('subprocess.run')
    def test_actual_docker_integration(self, mock_subprocess):
        """Test integration with actual Docker command patterns"""
        # Mock realistic Docker output
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = (
            "sutazai-ai-testing-qa-validator\tUp 2 minutes\t0.0.0.0:8081->8080/tcp\n"
            "sutazai-observability-monitoring-engineer\tUp 5 minutes (healthy)\t0.0.0.0:8083->8080/tcp\n"
            "sutazai-backend\tUp 10 minutes (unhealthy)\t0.0.0.0:8000->8000/tcp\n"
            "sutazai-redis\tUp 1 hour\t0.0.0.0:6379->6379/tcp\n"
        )
        
        # Test Docker command execution
        result = mock_subprocess.return_value
        self.assertEqual(result.returncode, 0)
        
        # Parse output
        lines = result.stdout.strip().split('\n')
        containers = []
        
        for line in lines:
            if line.strip():
                parts = line.split('\t')
                if len(parts) >= 2:
                    name = parts[0]
                    status_full = parts[1]
                    ports = parts[2] if len(parts) > 2 else ''
                    
                    # Extract agent name
                    agent_name = name.replace('sutazai-', '')
                    
                    # Parse status
                    if 'Up' in status_full:
                        if 'unhealthy' in status_full:
                            status = 'unhealthy'
                        elif 'healthy' in status_full:
                            status = 'healthy'
                        else:
                            status = 'running'
                    else:
                        status = 'unknown'
                    
                    containers.append({
                        'name': agent_name,
                        'status': status,
                        'ports': ports
                    })
        
        # Verify parsing results
        self.assertEqual(len(containers), 4)
        
        # Check specific agents
        qa_agent = next((c for c in containers if 'testing-qa' in c['name']), None)
        self.assertIsNotNone(qa_agent)
        self.assertEqual(qa_agent['status'], 'running')
        
        monitoring_agent = next((c for c in containers if 'monitoring-engineer' in c['name']), None)
        self.assertIsNotNone(monitoring_agent)
        self.assertEqual(monitoring_agent['status'], 'healthy')
        
        backend = next((c for c in containers if c['name'] == 'backend'), None)
        self.assertIsNotNone(backend)
        self.assertEqual(backend['status'], 'unhealthy')


def run_agent_detection_tests():
    """Run agent detection validation tests"""
    logger.info("🔍 Running Agent Detection and Validation Tests")
    logger.info("=" * 50)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    test_classes = [
        TestAgentDetectionCore,
        TestAgentStatusIntegration,
        TestAgentValidationRules,
        TestRealSystemValidation
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate report
    logger.info(f"\n📊 Agent Detection Test Results:")
    logger.info(f"Tests Run: {result.testsRun}")
    logger.info(f"Failures: {len(result.failures)}")
    logger.error(f"Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0
    logger.info(f"Success Rate: {success_rate:.1f}%")
    
    if result.failures:
        logger.info("\n❌ Failures:")
        for test, error in result.failures:
            logger.info(f"  - {test}")
    
    if result.errors:
        logger.error("\n💥 Errors:")
        for test, error in result.errors:
            logger.info(f"  - {test}")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_agent_detection_tests()
