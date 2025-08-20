#!/usr/bin/env python3
"""
Regression tests for Ollama integration
Ensures backward compatibility and existing functionality continues to work
"""

import pytest
import asyncio
import sys
import os
from unittest.mock import AsyncMock, Mock, patch, MagicMock
import tempfile
import json

# Add the agents directory to the path
# Path handled by pytest configuration, '..', 'agents'))

from agents.core.base_agent import BaseAgentV2, BaseAgent, AgentStatus
from core.ollama_integration import OllamaIntegration, OllamaConfig


class TestBackwardCompatibility:
    """Test that existing agents continue to work with new system"""
    
    def test_base_agent_alias_compatibility(self):
        """Test that BaseAgent alias still works"""
        # BaseAgent should be an alias for BaseAgent
        assert BaseAgent is BaseAgent
        
        # Should be able to create instance using old name
        agent = BaseAgent()
        assert isinstance(agent, BaseAgent)
        assert agent.agent_version == "2.0.0"
    
    def test_initialization_backward_compatibility(self):
        """Test that agents can be initialized with old patterns"""
        # Test with   parameters (old style)
        agent = BaseAgent()
        assert agent.agent_name is not None
        assert agent.agent_type is not None
        assert agent.status == AgentStatus.INITIALIZING
        
        # Test with config path only (common old pattern)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({"capabilities": ["test"]}, f)
            config_path = f.name
        
        try:
            agent = BaseAgent(config_path=config_path)
            assert agent.config["capabilities"] == ["test"]
        finally:
            os.unlink(config_path)
    
    def test_environment_variable_compatibility(self):
        """Test that existing environment variables are respected"""
        test_env = {
            'AGENT_NAME': 'legacy-test-agent',
            'AGENT_TYPE': 'legacy-type',
            'BACKEND_URL': 'http://legacy-backend:8000',
            'OLLAMA_URL': 'http://legacy-ollama:10104'
        }
        
        with patch.dict(os.environ, test_env):
            agent = BaseAgent()
            
            assert agent.agent_name == 'legacy-test-agent'
            assert agent.agent_type == 'legacy-type'
            assert agent.backend_url == 'http://legacy-backend:8000'
            assert agent.ollama_url == 'http://legacy-ollama:10104'
    
    def test_config_loading_backward_compatibility(self):
        """Test that existing config file formats are supported"""
        # Test with   config (common in existing agents)
         _config = {
            "capabilities": ["text-processing"]
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump( _config, f)
            config_path = f.name
        
        try:
            agent = BaseAgent(config_path=config_path)
            
            # Should load   config and fill in defaults
            assert agent.config["capabilities"] == ["text-processing"]
            assert agent.config["max_retries"] == 3  # Default value
            assert agent.config["timeout"] == 300  # Default value
        finally:
            os.unlink(config_path)
        
        # Test with full legacy config
        legacy_config = {
            "capabilities": ["nlp", "analysis"],
            "max_retries": 2,
            "timeout": 180,
            "batch_size": 15,
            "legacy_setting": "keep_this"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(legacy_config, f)
            config_path = f.name
        
        try:
            agent = BaseAgent(config_path=config_path)
            
            # Should preserve all existing settings
            for key, value in legacy_config.items():
                assert agent.config[key] == value
        finally:
            os.unlink(config_path)


class TestExistingAgentMethods:
    """Test that existing agent methods continue to work"""
    
    @pytest.fixture
    def legacy_agent(self):
        """Create agent in legacy style"""
        return BaseAgent()
    
    def test_synchronous_ollama_query_compatibility(self, legacy_agent):
        """Test that sync Ollama query method still exists and works"""
        # The sync method should exist for backward compatibility
        assert hasattr(legacy_agent, 'query_ollama_sync')
        
        # Test that it handles the "no running loop" case
        # (In a real scenario, this would work if called from sync context)
        result = legacy_agent.query_ollama_sync("test prompt")
        # Should return None because async components aren't set up
        assert result is None
    
    @pytest.mark.asyncio
    async def test_basic_task_processing_compatibility(self, legacy_agent):
        """Test that basic task processing works as before"""
        # Test with   task structure (common in existing agents)
         _task = {
            "id": "legacy-task-001",
            "type": "test"
        }
        
        result = await legacy_agent.process_task( _task)
        
        # Should handle   task structure
        assert result.task_id == "legacy-task-001"
        assert result.status == "completed"
        assert isinstance(result.result, dict)
        assert result.processing_time >= 0
    
    @pytest.mark.asyncio
    async def test_async_method_compatibility(self, legacy_agent):
        """Test that async methods work as expected"""
        await legacy_agent._setup_async_components()
        
        # Test async methods that existing agents might use
        methods_to_test = [
            'health_check',
            '_update_metrics',
        ]
        
        for method_name in methods_to_test:
            assert hasattr(legacy_agent, method_name)
            method = getattr(legacy_agent, method_name)
            assert callable(method)
        
        # Test health check specifically
        health_status = await legacy_agent.health_check()
        assert isinstance(health_status, dict)
        assert 'agent_name' in health_status
        assert 'healthy' in health_status
        
        await legacy_agent._cleanup_async_components()


class TestOllamaConfigRegression:
    """Test Ollama configuration backward compatibility"""
    
    def test_model_assignment_stability(self):
        """Test that existing agent model assignments haven't changed"""
        # Test critical agent model assignments that shouldn't change
        stable_assignments = {
            " system-architect": "tinyllama",
            "ai-product-manager": "tinyllama2.5-coder:7b",
            "garbage-collector": "tinyllama",
            "testing-qa-validator": "tinyllama2.5-coder:7b",
            "ai-senior-backend-developer": "tinyllama2.5-coder:7b"
        }
        
        for agent_name, expected_model in stable_assignments.items():
            actual_model = OllamaConfig.get_model_for_agent(agent_name)
            assert actual_model == expected_model, \
                f"Model assignment changed for {agent_name}: expected {expected_model}, got {actual_model}"
    
    def test_model_config_parameter_stability(self):
        """Test that model configuration parameters are stable"""
        # Test that default model configs haven't broken existing behavior
        test_agents = [" system-architect", "ai-product-manager", "garbage-collector"]
        
        for agent_name in test_agents:
            config = OllamaConfig.get_model_config(agent_name)
            
            # Verify all required parameters exist
            required_params = ["model", "temperature", "max_tokens", "top_p"]
            for param in required_params:
                assert param in config, f"Missing required parameter {param} for {agent_name}"
            
            # Verify parameter ranges are reasonable
            assert 0 <= config["temperature"] <= 1, f"Invalid temperature for {agent_name}"
            assert config["max_tokens"] > 0, f"Invalid max_tokens for {agent_name}"
            assert 0 <= config["top_p"] <= 1, f"Invalid top_p for {agent_name}"
    
    def test_unknown_agent_fallback_stability(self):
        """Test that unknown agents still get reasonable defaults"""
        unknown_agents = ["new-agent", "test-agent", "non-existent-agent"]
        
        for agent_name in unknown_agents:
            model = OllamaConfig.get_model_for_agent(agent_name)
            config = OllamaConfig.get_model_config(agent_name)
            
            # Should fall back to default model
            assert model == OllamaConfig.DEFAULT_MODEL
            assert config["model"] == OllamaConfig.DEFAULT_MODEL
            
            # Should have complete config
            assert "temperature" in config
            assert "max_tokens" in config


class TestExistingAgentTypes:
    """Test that existing agent types continue to work"""
    
    @pytest.mark.asyncio
    async def test_opus_model_agents(self):
        """Test agents that use Opus model"""
        opus_agents = [" system-architect", "complex-problem-solver", "ethical-governor"]
        
        for agent_name in opus_agents:
            with patch.dict(os.environ, {'AGENT_NAME': agent_name, 'AGENT_TYPE': 'test'}):
                agent = BaseAgent()
                
                # Should be assigned Opus model
                assert agent.default_model == OllamaConfig.OPUS_MODEL
                
                # Should have appropriate config
                config = OllamaConfig.get_model_config(agent_name)
                assert config["model"] == OllamaConfig.OPUS_MODEL
                assert config["max_tokens"] == 4096  # Opus gets more tokens
                assert config["temperature"] == 0.8  # More creative
    
    @pytest.mark.asyncio
    async def test_sonnet_model_agents(self):
        """Test agents that use Sonnet model"""
        sonnet_agents = ["ai-product-manager", "testing-qa-validator", "senior-backend-developer"]
        
        for agent_name in sonnet_agents:
            with patch.dict(os.environ, {'AGENT_NAME': agent_name, 'AGENT_TYPE': 'test'}):
                agent = BaseAgent()
                
                # Should be assigned Sonnet model
                assert agent.default_model == OllamaConfig.SONNET_MODEL
                
                # Should have balanced config
                config = OllamaConfig.get_model_config(agent_name)
                assert config["model"] == OllamaConfig.SONNET_MODEL
                assert config["max_tokens"] == 2048  # Balanced tokens
                assert config["temperature"] == 0.7  # Balanced creativity
    
    @pytest.mark.asyncio
    async def test_default_model_agents(self):
        """Test agents that use default model"""
        default_agents = ["garbage-collector", "cpu-only-hardware-optimizer", "resource-visualiser"]
        
        for agent_name in default_agents:
            with patch.dict(os.environ, {'AGENT_NAME': agent_name, 'AGENT_TYPE': 'test'}):
                agent = BaseAgent()
                
                # Should be assigned default model
                assert agent.default_model == OllamaConfig.DEFAULT_MODEL
                
                # Should have conservative config
                config = OllamaConfig.get_model_config(agent_name)
                assert config["model"] == OllamaConfig.DEFAULT_MODEL
                assert config["max_tokens"] == 1024  # Conservative tokens
                assert config["temperature"] == 0.5  # Less creative


class TestLegacyBehaviorPreservation:
    """Test that legacy behaviors are preserved"""
    
    @pytest.mark.asyncio
    async def test_agent_lifecycle_compatibility(self):
        """Test that agent lifecycle works as in legacy system"""
        agent = BaseAgent()
        
        # Test initial state
        assert agent.status == AgentStatus.INITIALIZING
        
        # Test setup
        await agent._setup_async_components()
        
        # Test that components are initialized
        assert agent.http_client is not None
        assert agent.ollama_pool is not None
        assert agent.circuit_breaker is not None
        
        # Test cleanup
        await agent._cleanup_async_components()
        
        # Components should still exist but be cleaned up
        assert agent.http_client is not None  # Reference preserved for safety
    
    @pytest.mark.asyncio
    async def test_error_handling_compatibility(self):
        """Test that error handling works as expected in legacy scenarios"""
        agent = BaseAgent()
        
        # Test error handling in task processing
        problematic_task = {
            "id": "error-test",
            "type": "test",
            "data": None  # This might cause issues in some implementations
        }
        
        result = await agent.process_task(problematic_task)
        
        # Should handle gracefully and not crash
        assert result is not None
        assert result.task_id == "error-test"
        # May succeed or fail, but should be handled gracefully
        assert result.status in ["completed", "failed"]
    
    def test_metrics_compatibility(self):
        """Test that metrics structure is compatible"""
        agent = BaseAgent()
        
        # Test that all expected metrics fields exist
        expected_fields = [
            'tasks_processed', 'tasks_failed', 'tasks_queued',
            'total_processing_time', 'avg_processing_time',
            'startup_time', 'ollama_requests', 'ollama_failures'
        ]
        
        for field in expected_fields:
            assert hasattr(agent.metrics, field), f"Missing metrics field: {field}"
            
            # Test that fields have reasonable default values
            value = getattr(agent.metrics, field)
            if field == 'startup_time':
                assert value is not None
            else:
                assert isinstance(value, (int, float))


class TestIntegrationRegression:
    """Test that integration points haven't broken"""
    
    @pytest.mark.asyncio
    async def test_ollama_integration_api_compatibility(self):
        """Test that Ollama integration API hasn't broken"""
        integration = OllamaIntegration()
        
        # Test that all expected methods exist
        expected_methods = [
            'ensure_model_available', 'pull_model', 'generate', 
            'chat', 'embeddings'
        ]
        
        for method_name in expected_methods:
            assert hasattr(integration, method_name)
            assert callable(getattr(integration, method_name))
        
        # Test basic method signatures (should accept expected parameters)
        # These will fail due to network issues, but should not fail due to API changes
        
        try:
            await integration.ensure_model_available("tinyllama")
        except Exception as e:
            # Should be network/service error, not API error
            assert "TypeError" not in str(type(e))
        
        try:
            await integration.generate("test prompt")
        except Exception as e:
            # Should be network/service error, not API error
            assert "TypeError" not in str(type(e))
        
        try:
            await integration.chat([{"role": "user", "content": "test"}])
        except Exception as e:
            # Should be network/service error, not API error
            assert "TypeError" not in str(type(e))
    
    @pytest.mark.asyncio
    async def test_agent_ollama_integration_compatibility(self):
        """Test that agent-Ollama integration hasn't broken"""
        agent = BaseAgent()
        await agent._setup_async_components()
        
        # Test that Ollama query methods exist and work
        assert hasattr(agent, 'query_ollama')
        assert hasattr(agent, 'query_ollama_chat')
        assert hasattr(agent, 'query_ollama_sync')
        
        # Test method signatures
        try:
            await agent.query_ollama("test prompt")
        except Exception as e:
            # Should be service error, not API error
            assert "TypeError" not in str(type(e))
        
        try:
            await agent.query_ollama_chat([{"role": "user", "content": "test"}])
        except Exception as e:
            # Should be service error, not API error
            assert "TypeError" not in str(type(e))
        
        await agent._cleanup_async_components()


class TestPerformanceRegression:
    """Test that performance hasn't regressed"""
    
    @pytest.mark.asyncio
    async def test_agent_initialization_performance(self):
        """Test that agent initialization time is reasonable"""
        import time
        
        start_time = time.time()
        agent = BaseAgent()
        init_time = time.time() - start_time
        
        # Initialization should be fast (under 1 second)
        assert init_time < 1.0, f"Agent initialization took {init_time:.3f}s, too slow"
    
    @pytest.mark.asyncio
    async def test_task_processing_performance_baseline(self):
        """Test that basic task processing performance is reasonable"""
        import time
        
        agent = BaseAgent()
        
        task = {
            "id": "perf-test",
            "type": "baseline",
            "data": {"simple": "task"}
        }
        
        start_time = time.time()
        result = await agent.process_task(task)
        processing_time = time.time() - start_time
        
        # Basic task should process quickly
        assert processing_time < 0.5, f"Basic task processing took {processing_time:.3f}s, too slow"
        assert result.status == "completed"
    
    @pytest.mark.asyncio
    async def test_memory_usage_regression(self):
        """Test that memory usage hasn't significantly increased"""
        import psutil
        import gc
        
        # Get baseline memory
        gc.collect()
        process = psutil.Process()
        baseline_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create and use agent
        agent = BaseAgent()
        await agent._setup_async_components()
        
        # Process some tasks
        for i in range(10):
            task = {"id": f"mem-test-{i}", "type": "test", "data": {}}
            await agent.process_task(task)
        
        await agent._cleanup_async_components()
        
        # Check final memory
        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - baseline_memory
        
        # Memory increase should be reasonable (less than 50MB for this test)
        assert memory_increase < 50, f"Memory increased by {memory_increase:.1f}MB, possible memory leak"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])