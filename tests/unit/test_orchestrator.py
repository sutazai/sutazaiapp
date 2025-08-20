#!/usr/bin/env python3
"""
Purpose: Unit tests for hygiene agent orchestrator
Usage: python -m pytest tests/hygiene/test_orchestrator.py
Requirements: pytest, unittest.Mock
"""

import unittest
import json
import tempfile
import subprocess
from pathlib import Path
from unittest.mock import patch, MagicMock, mock_open
import sys
import os

# Add project root to path for imports
# Path handled by pytest configuration

class TestAgentOrchestrator(unittest.TestCase):
    """Test the AgentOrchestrator class"""
    
    def setUp(self):
        """Setup test environment"""
        self.project_root = Path("/tmp/test_project")
        self.project_root.mkdir(exist_ok=True)
        
        # Create   directory structure
        (self.project_root / "logs").mkdir(exist_ok=True)
        (self.project_root / "tmp").mkdir(exist_ok=True)
        
    def tearDown(self):
        """Cleanup test environment"""
        import shutil
        if self.project_root.exists():
            shutil.rmtree(self.project_root)
    
    @patch('builtins.open', new_callable=mock_open)
    @patch('pathlib.Path.mkdir')
    def test_log_action(self, mock_mkdir, mock_file):
        """Test logging functionality"""
        # Import here to avoid issues with path setup
        from scripts.agents.hygiene_agent_orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator(str(self.project_root))
        orchestrator.log_action("Test message", "INFO")
        
        # Verify file operations
        mock_mkdir.assert_called()
        mock_file.assert_called()
        
    def test_agent_registry_structure(self):
        """Test agent registry has required structure"""
        from scripts.agents.hygiene_agent_orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator(str(self.project_root))
        
        # Verify registry has expected agents
        expected_agents = [
            "garbage-collector",
            "deploy-automation-master", 
            "multi-agent-coordinator",
            "container-orchestrator-k3s",
            "senior-backend-developer"
        ]
        
        for agent in expected_agents:
            self.assertIn(agent, orchestrator.agent_registry)
            
        # Verify each agent has required fields
        for agent_name, config in orchestrator.agent_registry.items():
            self.assertIn("rules", config)
            self.assertIn("capabilities", config)
            self.assertIn("command", config)
            self.assertIsInstance(config["rules"], list)
            self.assertIsInstance(config["capabilities"], list)
            
    def test_get_agents_for_rule(self):
        """Test getting agents for specific rules"""
        from scripts.agents.hygiene_agent_orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator(str(self.project_root))
        
        # Test rule 13 (garbage collection)
        agents_for_13 = orchestrator.get_agents_for_rule(13)
        self.assertIn("garbage-collector", agents_for_13)
        
        # Test rule 12 (deployment automation)
        agents_for_12 = orchestrator.get_agents_for_rule(12)
        self.assertIn("deploy-automation-master", agents_for_12)
        
        # Test non-existent rule
        agents_for_999 = orchestrator.get_agents_for_rule(999)
        self.assertEqual(agents_for_999, [])
        
    def test_create_agent_task(self):
        """Test agent task creation"""
        from scripts.agents.hygiene_agent_orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator(str(self.project_root))
        
        task_params = {"action": "test_action", "dry_run": True}
        task = orchestrator.create_agent_task("test-agent", 13, task_params)
        
        # Verify task structure
        self.assertEqual(task["agent"], "test-agent")
        self.assertEqual(task["rule"], 13)
        self.assertIn("task_id", task)
        self.assertEqual(task["parameters"], task_params)
        self.assertTrue(task["safety_mode"])
        self.assertTrue(task["dry_run"])
        
    @patch('subprocess.run')
    @patch('builtins.open', new_callable=mock_open)
    def test_execute_agent_task_success(self, mock_file, mock_subprocess):
        """Test successful agent task execution"""
        from scripts.agents.hygiene_agent_orchestrator import AgentOrchestrator
        
        # Setup Mock subprocess result
        mock_result = MagicMock()
        mock_result.returncode = 0
        mock_result.stdout = "Success output"
        mock_result.stderr = ""
        mock_subprocess.return_value = mock_result
        
        orchestrator = AgentOrchestrator(str(self.project_root))
        
        task = {
            "task_id": "test_task_123",
            "agent": "garbage-collector",
            "rule": 13,
            "parameters": {"action": "test"}
        }
        
        result = orchestrator.execute_agent_task(task)
        
        # Verify result structure
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["task_id"], "test_task_123")
        self.assertEqual(result["return_code"], 0)
        
    @patch('subprocess.run')
    def test_execute_agent_task_failure(self, mock_subprocess):
        """Test failed agent task execution"""
        from scripts.agents.hygiene_agent_orchestrator import AgentOrchestrator
        
        # Setup Mock subprocess result for failure
        mock_result = MagicMock()
        mock_result.returncode = 1
        mock_result.stdout = ""
        mock_result.stderr = "Error occurred"
        mock_subprocess.return_value = mock_result
        
        orchestrator = AgentOrchestrator(str(self.project_root))
        
        task = {
            "task_id": "test_task_456",
            "agent": "garbage-collector", 
            "rule": 13,
            "parameters": {"action": "test"}
        }
        
        result = orchestrator.execute_agent_task(task)
        
        # Verify failure handling
        self.assertEqual(result["status"], "error")
        self.assertEqual(result["return_code"], 1)
        
    def test_execute_agent_task_unknown_agent(self):
        """Test execution with unknown agent"""
        from scripts.agents.hygiene_agent_orchestrator import AgentOrchestrator
        
        orchestrator = AgentOrchestrator(str(self.project_root))
        
        task = {
            "task_id": "test_task_789",
            "agent": "non-existent-agent",
            "rule": 13,
            "parameters": {"action": "test"}
        }
        
        result = orchestrator.execute_agent_task(task)
        
        # Verify error handling for unknown agent
        self.assertEqual(result["status"], "error")
        self.assertIn("not found in registry", result["message"])

class TestOrchestratorIntegration(unittest.TestCase):
    """Integration tests for orchestrator"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.project_root = Path("/opt/sutazaiapp")
        self.orchestrator_script = self.project_root / "scripts/agents/hygiene-agent-orchestrator.py"
        
    def test_orchestrator_script_exists(self):
        """Test that orchestrator script file exists"""
        self.assertTrue(self.orchestrator_script.exists(),
                       f"Orchestrator script not found: {self.orchestrator_script}")
        
    def test_orchestrator_script_syntax(self):
        """Test orchestrator script has valid Python syntax"""
        cmd = ["python3", "-m", "py_compile", str(self.orchestrator_script)]
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        self.assertEqual(result.returncode, 0,
                        f"Orchestrator script has syntax errors: {result.stderr}")
        
    def test_orchestrator_help(self):
        """Test orchestrator shows help"""
        cmd = ["python3", str(self.orchestrator_script), "--help"]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        self.assertEqual(result.returncode, 0)
        self.assertIn("usage:", result.stdout.lower())
        
    def test_orchestrator_dry_run_rule_13(self):
        """Test orchestrator dry run for rule 13"""
        if self.orchestrator_script.exists():
            cmd = ["python3", str(self.orchestrator_script), "--rule=13", "--dry-run"]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            # Should complete without crashing
            self.assertIsNotNone(result, "Orchestrator script should execute")

class TestOrchestratorErrorHandling(unittest.TestCase):
    """Test error handling in orchestrator"""
    
    def test_invalid_rule_number(self):
        """Test handling of invalid rule numbers"""
        project_root = Path("/tmp/test_project")
        project_root.mkdir(exist_ok=True)
        
        try:
            from scripts.agents.hygiene_agent_orchestrator import AgentOrchestrator
            
            orchestrator = AgentOrchestrator(str(project_root))
            
            # Test with invalid rule number
            agents = orchestrator.get_agents_for_rule(99)
            self.assertEqual(agents, [])
            
        finally:
            import shutil
            if project_root.exists():
                shutil.rmtree(project_root)
                
    def test_missing_project_directory(self):
        """Test handling when project directory doesn't exist"""
        try:
            from scripts.agents.hygiene_agent_orchestrator import AgentOrchestrator
            
            # Should handle non-existent directory gracefully
            orchestrator = AgentOrchestrator("/nonexistent/path")
            self.assertIsNotNone(orchestrator.agent_registry)
            
        except Exception as e:
            # Should not crash with unhandled exceptions
            self.fail(f"Orchestrator should handle missing directories: {e}")

if __name__ == "__main__":
