"""
Test Base Agent

This module contains tests for the base agent class.
"""

import unittest

import sys

sys.path.append("/opt/sutazaiapp")

from ai_agents.base_agent import AgentError
from ai_agents.base_agent import BaseAgentImplementation


class TestBaseAgent(unittest.TestCase):
    """Tests for BaseAgent."""

    def setUp(self):
        """Set up test environment."""
        self.config = {
            "type": "test_agent",
            "model": {
                "name": "test-model",
                "version": "1.0",
                "max_tokens": 1000,
                "temperature": 0.7,
            },
            "capabilities": ["test_capability"],
            "settings": {"timeout": 30, "retry_attempts": 3},
        }

    def test_validate_config(self):
        """Test configuration validation."""
        # Test valid config
        agent = BaseAgentImplementation(self.config)
        agent._validate_config(self.config)

        # Test invalid config (missing required field)
        invalid_config = {"type": "test_agent", "model": {"name": "test-model"}}
        with self.assertRaises(AgentError):
            agent._validate_config(invalid_config)

    def test_initialize(self):
        """Test agent initialization."""
        agent = BaseAgentImplementation(self.config)
        agent.initialize()
        self.assertTrue(agent.is_initialized)

    def test_execute(self):
        """Test agent execution."""
        agent = BaseAgentImplementation(self.config)
        agent.initialize()
        result = agent.execute()
        self.assertIsNotNone(result)
        self.assertTrue(agent.is_initialized)

    def test_execute_uninitialized(self):
        """Test executing uninitialized agent."""
        agent = BaseAgentImplementation(self.config)
        with self.assertRaises(AgentError):
            agent.execute()

    def test_cleanup(self):
        """Test agent cleanup."""
        agent = BaseAgentImplementation(self.config)
        agent.initialize()
        agent.cleanup()
        self.assertFalse(agent.is_initialized)

    def test_heartbeat(self):
        """Test agent heartbeat."""
        agent = BaseAgentImplementation(self.config)
        agent.initialize()

        # Get initial heartbeat
        initial_heartbeat = agent.get_heartbeat()
        self.assertIsNotNone(initial_heartbeat)

        # Update heartbeat
        agent.update_heartbeat()

        # Get updated heartbeat
        updated_heartbeat = agent.get_heartbeat()
        self.assertGreater(updated_heartbeat, initial_heartbeat)

    def test_capabilities(self):
        """Test agent capabilities."""
        agent = BaseAgentImplementation(self.config)
        capabilities = agent.get_capabilities()
        self.assertEqual(capabilities, self.config["capabilities"])

    def test_execution_history(self):
        """Test execution history tracking."""
        agent = BaseAgentImplementation(self.config)
        agent.initialize()

        # Execute multiple times
        for _ in range(3):
            agent.execute()

        # Check history
        history = agent.get_history()
        self.assertEqual(len(history), 3)

        # Clear history
        agent.clear_history()
        self.assertEqual(len(agent.get_history()), 0)

    def test_error_handling(self):
        """Test error handling during execution."""
        agent = BaseAgentImplementation(self.config)
        agent.initialize()

        # Simulate error during execution
        agent._simulate_error = True

        with self.assertRaises(AgentError):
            agent.execute()

        # Check error was logged
        history = agent.get_history()
        self.assertEqual(len(history), 1)
        self.assertFalse(history[0]["success"])
        self.assertIsNotNone(history[0]["error"])


class TestBaseAgentImplementation(unittest.TestCase):
    """Tests for BaseAgentImplementation."""

    def setUp(self):
        """Set up test environment."""
        self.config = {
            "type": "test_agent",
            "model": {
                "name": "test-model",
                "version": "1.0",
                "max_tokens": 1000,
                "temperature": 0.7,
            },
            "capabilities": ["test_capability"],
            "settings": {"timeout": 30, "retry_attempts": 3},
        }

    def test_implementation(self):
        """Test concrete implementation of abstract methods."""
        agent = BaseAgentImplementation(self.config)

        # Test initialization
        agent.initialize()
        self.assertTrue(agent.is_initialized)

        # Test execution
        result = agent.execute()
        self.assertIsNotNone(result)
        self.assertEqual(
            result, {"status": "success", "message": "Test execution completed"}
        )

        # Test cleanup
        agent.cleanup()
        self.assertFalse(agent.is_initialized)

    def test_execution_logging(self):
        """Test execution logging."""
        agent = BaseAgentImplementation(self.config)
        agent.initialize()

        # Execute and check logging
        result = agent.execute()
        history = agent.get_history()

        self.assertEqual(len(history), 1)
        self.assertTrue(history[0]["success"])
        self.assertEqual(history[0]["result"], result)
        self.assertIsNotNone(history[0]["timestamp"])
        self.assertIsNotNone(history[0]["duration"])


if __name__ == "__main__":
    unittest.main()
