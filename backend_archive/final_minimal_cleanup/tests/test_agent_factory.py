#!/usr/bin/env python3
"""
Test Agent Factory

This module contains tests for the agent factory.
"""

import os
import json
import unittest
from pathlib import Path

import sys

sys.path.append("/opt/sutazaiapp")

from ai_agents.agent_factory import AgentFactory
from ai_agents.base_agent import BaseAgent, AgentError


# Create a mock DocumentProcessorAgent class for testing
class DocumentProcessorAgent(BaseAgent):
    """Mock document processor agent for testing."""

    def __init__(self, config):
        super().__init__(config)
        self.type = config["type"]

    def initialize(self):
        """Initialize the agent."""
        pass

    def execute(self, task=None):
        """Execute a task."""
        return {"status": "success"}

    def cleanup(self):
        """Clean up agent resources."""
        pass

    def _initialize(self):
        """Internal initialization implementation."""
        pass

    def _execute(self, task):
        """Internal execution implementation."""
        return {"status": "success"}

    def _cleanup(self):
        """Internal cleanup implementation."""
        pass


class TestAgentFactory(unittest.TestCase):
    """Tests for AgentFactory."""

    def setUp(self):
        """Set up test environment."""
        # Create test config directory if it doesn't exist
        self.config_dir = Path("/opt/sutazaiapp/config")
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Create a test config file
        self.test_config_path = self.config_dir / "test_agents.json"
        self._create_test_config()

        # Initialize factory with test config path
        self.factory = AgentFactory(config_path=str(self.test_config_path))

        # Register the mock document processor agent class
        self.factory.agent_classes["document_processor"] = DocumentProcessorAgent

    def tearDown(self):
        """Clean up test environment."""
        if os.path.exists(self.test_config_path):
            os.remove(self.test_config_path)

    def _create_test_config(self):
        """Create a test configuration file."""
        test_config = {
            "document_processor": {
                "type": "document_processor",
                "model": {
                    "name": "gpt-4",
                    "version": "1.0",
                    "max_tokens": 2000,
                    "temperature": 0.7,
                },
                "capabilities": ["text_extraction", "format_conversion"],
                "settings": {"batch_size": 10, "timeout": 30, "retry_attempts": 3},
            }
        }

        with open(self.test_config_path, "w") as f:
            json.dump(test_config, f)

    def test_load_agents(self):
        """Test loading agents from configuration."""
        agents = self.factory.get_available_agents()
        self.assertIn("document_processor", agents)

    def test_create_agent(self):
        """Test creating an agent instance."""
        agent = self.factory.create_agent("document_processor")
        self.assertIsNotNone(agent)
        self.assertEqual(agent.type, "document_processor")

    def test_create_nonexistent_agent(self):
        """Test creating a nonexistent agent."""
        with self.assertRaises(AgentError):
            self.factory.create_agent("nonexistent_agent")

    def test_validate_config(self):
        """Test configuration validation."""
        # Test valid config
        valid_config = {
            "type": "test_agent",
            "model": {"name": "test-model", "version": "1.0"},
            "capabilities": ["test_capability"],
        }
        self.factory._validate_config(valid_config)

        # Test invalid config (missing required field)
        invalid_config = {"type": "test_agent", "model": {"name": "test-model"}}
        with self.assertRaises(AgentError):
            self.factory._validate_config(invalid_config)

    def test_register_agent(self):
        """Test registering a new agent."""
        new_agent_config = {
            "type": "new_agent",
            "model": {"name": "test-model", "version": "1.0"},
            "capabilities": ["test_capability"],
        }

        self.factory.register_agent("new_agent", new_agent_config)
        agents = self.factory.get_available_agents()
        self.assertIn("new_agent", agents)

    def test_register_duplicate_agent(self):
        """Test registering a duplicate agent."""
        new_agent_config = {
            "type": "document_processor",
            "model": {"name": "test-model", "version": "1.0"},
            "capabilities": ["test_capability"],
        }

        # Make sure the agent already exists
        self.assertIn("document_processor", self.factory.get_available_agents())

        with self.assertRaises(AgentError):
            self.factory.register_agent("document_processor", new_agent_config)


if __name__ == "__main__":
    unittest.main()
