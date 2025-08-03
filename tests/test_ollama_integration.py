#!/usr/bin/env python3
"""
Comprehensive unit tests for Ollama integration
Tests all aspects of OllamaIntegration and OllamaConfig classes
"""

import pytest
import asyncio
import httpx
import json
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from datetime import datetime
import sys
import os

# Add the agents directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'agents'))

from core.ollama_integration import OllamaIntegration, OllamaConfig


class TestOllamaIntegration:
    """Test suite for OllamaIntegration class"""
    
    @pytest.fixture
    def ollama_integration(self):
        """Create OllamaIntegration instance for testing"""
        return OllamaIntegration(
            base_url="http://test-ollama:11434",
            default_model="tinyllama",
            timeout=30
        )
    
    @pytest.fixture
    def mock_httpx_client(self):
        """Mock httpx.AsyncClient"""
        client = AsyncMock()
        return client
    
    @pytest.mark.asyncio
    async def test_initialization(self, ollama_integration):
        """Test OllamaIntegration initialization"""
        assert ollama_integration.base_url == "http://test-ollama:11434"
        assert ollama_integration.default_model == "tinyllama"
        assert ollama_integration.timeout == 30
        assert ollama_integration.client is not None
    
    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager functionality"""
        async with OllamaIntegration() as integration:
            assert integration is not None
            assert integration.client is not None
    
    @pytest.mark.asyncio
    async def test_ensure_model_available_success(self, ollama_integration):
        """Test successful model availability check"""
        # Mock successful model list response
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "models": [
                {"name": "tinyllama:latest"},
                {"name": "qwen2.5-coder:7b"}
            ]
        }
        
        with patch.object(ollama_integration.client, 'get', return_value=mock_response):
            result = await ollama_integration.ensure_model_available("tinyllama")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_ensure_model_available_needs_pull(self, ollama_integration):
        """Test model availability when model needs to be pulled"""
        # Mock model list response without target model
        mock_list_response = Mock()
        mock_list_response.status_code = 200
        mock_list_response.json.return_value = {
            "models": [{"name": "other-model:latest"}]
        }
        
        # Mock successful pull response
        mock_pull_response = Mock()
        mock_pull_response.status_code = 200
        
        with patch.object(ollama_integration.client, 'get', return_value=mock_list_response), \
             patch.object(ollama_integration.client, 'post', return_value=mock_pull_response):
            result = await ollama_integration.ensure_model_available("tinyllama")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_ensure_model_available_failure(self, ollama_integration):
        """Test model availability check failure"""
        mock_response = Mock()
        mock_response.status_code = 500
        
        with patch.object(ollama_integration.client, 'get', return_value=mock_response):
            result = await ollama_integration.ensure_model_available("tinyllama")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_pull_model_success(self, ollama_integration):
        """Test successful model pull"""
        mock_response = Mock()
        mock_response.status_code = 200
        
        with patch.object(ollama_integration.client, 'post', return_value=mock_response):
            result = await ollama_integration.pull_model("tinyllama")
            assert result is True
    
    @pytest.mark.asyncio
    async def test_pull_model_failure(self, ollama_integration):
        """Test failed model pull"""
        mock_response = Mock()
        mock_response.status_code = 404
        
        with patch.object(ollama_integration.client, 'post', return_value=mock_response):
            result = await ollama_integration.pull_model("nonexistent-model")
            assert result is False
    
    @pytest.mark.asyncio
    async def test_generate_success(self, ollama_integration):
        """Test successful text generation"""
        # Mock model availability check
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            # Mock successful generation response
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "Generated text response"
            }
            
            with patch.object(ollama_integration.client, 'post', return_value=mock_response):
                result = await ollama_integration.generate(
                    prompt="Test prompt",
                    model="tinyllama",
                    system="You are a helpful assistant"
                )
                
                assert result == "Generated text response"
    
    @pytest.mark.asyncio
    async def test_generate_model_unavailable(self, ollama_integration):
        """Test generation when model is unavailable"""
        with patch.object(ollama_integration, 'ensure_model_available', return_value=False):
            result = await ollama_integration.generate("Test prompt")
            assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_api_error(self, ollama_integration):
        """Test generation with API error"""
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 500
            
            with patch.object(ollama_integration.client, 'post', return_value=mock_response):
                result = await ollama_integration.generate("Test prompt")
                assert result is None
    
    @pytest.mark.asyncio
    async def test_generate_with_parameters(self, ollama_integration):
        """Test generation with custom parameters"""
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "Custom response"}
            
            with patch.object(ollama_integration.client, 'post', return_value=mock_response) as mock_post:
                await ollama_integration.generate(
                    prompt="Test prompt",
                    temperature=0.8,
                    max_tokens=1024
                )
                
                # Verify parameters were passed correctly
                call_args = mock_post.call_args
                payload = call_args[1]['json']
                assert payload['options']['temperature'] == 0.8
                assert payload['options']['num_predict'] == 1024
    
    @pytest.mark.asyncio
    async def test_chat_success(self, ollama_integration):
        """Test successful chat completion"""
        messages = [
            {"role": "user", "content": "Hello"}
        ]
        
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "message": {"content": "Hello! How can I help you?"}
            }
            
            with patch.object(ollama_integration.client, 'post', return_value=mock_response):
                result = await ollama_integration.chat(messages)
                assert result == "Hello! How can I help you?"
    
    @pytest.mark.asyncio
    async def test_chat_model_unavailable(self, ollama_integration):
        """Test chat when model is unavailable"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(ollama_integration, 'ensure_model_available', return_value=False):
            result = await ollama_integration.chat(messages)
            assert result is None
    
    @pytest.mark.asyncio
    async def test_chat_api_error(self, ollama_integration):
        """Test chat with API error"""
        messages = [{"role": "user", "content": "Hello"}]
        
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 400
            
            with patch.object(ollama_integration.client, 'post', return_value=mock_response):
                result = await ollama_integration.chat(messages)
                assert result is None
    
    @pytest.mark.asyncio
    async def test_embeddings_success(self, ollama_integration):
        """Test successful embeddings generation"""
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "embedding": [0.1, 0.2, 0.3, 0.4, 0.5]
            }
            
            with patch.object(ollama_integration.client, 'post', return_value=mock_response):
                result = await ollama_integration.embeddings("Test text")
                assert result == [0.1, 0.2, 0.3, 0.4, 0.5]
    
    @pytest.mark.asyncio
    async def test_embeddings_api_error(self, ollama_integration):
        """Test embeddings with API error"""
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 500
            
            with patch.object(ollama_integration.client, 'post', return_value=mock_response):
                result = await ollama_integration.embeddings("Test text")
                assert result is None
    
    @pytest.mark.asyncio
    async def test_network_timeout(self, ollama_integration):
        """Test handling of network timeouts"""
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            with patch.object(ollama_integration.client, 'post', side_effect=httpx.TimeoutException("Timeout")):
                result = await ollama_integration.generate("Test prompt")
                assert result is None
    
    @pytest.mark.asyncio
    async def test_connection_error(self, ollama_integration):
        """Test handling of connection errors"""
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            with patch.object(ollama_integration.client, 'post', side_effect=httpx.ConnectError("Connection failed")):
                result = await ollama_integration.generate("Test prompt")
                assert result is None


class TestOllamaConfig:
    """Test suite for OllamaConfig class"""
    
    def test_get_model_for_agent_opus(self):
        """Test getting Opus model for complex agents"""
        agent_name = "ai-system-architect"
        model = OllamaConfig.get_model_for_agent(agent_name)
        assert model == OllamaConfig.OPUS_MODEL
    
    def test_get_model_for_agent_sonnet(self):
        """Test getting Sonnet model for balanced agents"""
        agent_name = "ai-product-manager"
        model = OllamaConfig.get_model_for_agent(agent_name)
        assert model == OllamaConfig.SONNET_MODEL
    
    def test_get_model_for_agent_default(self):
        """Test getting default model for simple agents"""
        agent_name = "garbage-collector"
        model = OllamaConfig.get_model_for_agent(agent_name)
        assert model == OllamaConfig.DEFAULT_MODEL
    
    def test_get_model_for_unknown_agent(self):
        """Test getting model for unknown agent"""
        agent_name = "unknown-agent"
        model = OllamaConfig.get_model_for_agent(agent_name)
        assert model == OllamaConfig.DEFAULT_MODEL
    
    def test_get_model_config_opus(self):
        """Test getting config for Opus model agent"""
        agent_name = "ai-system-architect"
        config = OllamaConfig.get_model_config(agent_name)
        
        assert config["model"] == OllamaConfig.OPUS_MODEL
        assert config["temperature"] == 0.8
        assert config["max_tokens"] == 4096
        assert config["top_p"] == 0.95
    
    def test_get_model_config_sonnet(self):
        """Test getting config for Sonnet model agent"""
        agent_name = "ai-product-manager"
        config = OllamaConfig.get_model_config(agent_name)
        
        assert config["model"] == OllamaConfig.SONNET_MODEL
        assert config["temperature"] == 0.7
        assert config["max_tokens"] == 2048
        assert config["top_p"] == 0.9
    
    def test_get_model_config_default(self):
        """Test getting config for default model agent"""
        agent_name = "garbage-collector"
        config = OllamaConfig.get_model_config(agent_name)
        
        assert config["model"] == OllamaConfig.DEFAULT_MODEL
        assert config["temperature"] == 0.5
        assert config["max_tokens"] == 1024
        assert config["top_p"] == 0.8
    
    def test_model_constants(self):
        """Test that model constants are defined correctly"""
        assert OllamaConfig.OPUS_MODEL == "deepseek-r1:8b"
        assert OllamaConfig.SONNET_MODEL == "qwen2.5-coder:7b"
        assert OllamaConfig.DEFAULT_MODEL == "tinyllama"
    
    def test_agent_models_coverage(self):
        """Test that agent models mapping has reasonable coverage"""
        # Check that we have models assigned for different types of agents
        opus_agents = [k for k, v in OllamaConfig.AGENT_MODELS.items() if v == OllamaConfig.OPUS_MODEL]
        sonnet_agents = [k for k, v in OllamaConfig.AGENT_MODELS.items() if v == OllamaConfig.SONNET_MODEL]
        default_agents = [k for k, v in OllamaConfig.AGENT_MODELS.items() if v == OllamaConfig.DEFAULT_MODEL]
        
        assert len(opus_agents) > 0, "Should have some agents using Opus model"
        assert len(sonnet_agents) > 0, "Should have some agents using Sonnet model"
        assert len(default_agents) > 0, "Should have some agents using default model"
        
        # Verify some specific assignments
        assert "ai-system-architect" in opus_agents
        assert "ai-product-manager" in sonnet_agents
        assert "garbage-collector" in default_agents
    
    def test_config_parameters_complete(self):
        """Test that all config parameters are present"""
        config = OllamaConfig.get_model_config("test-agent")
        
        required_params = [
            "model", "temperature", "max_tokens", 
            "top_p", "frequency_penalty", "presence_penalty"
        ]
        
        for param in required_params:
            assert param in config, f"Missing parameter: {param}"
    
    def test_config_parameter_ranges(self):
        """Test that config parameters are within reasonable ranges"""
        for agent_type in ["opus", "sonnet", "default"]:
            if agent_type == "opus":
                config = OllamaConfig.get_model_config("ai-system-architect")
            elif agent_type == "sonnet":
                config = OllamaConfig.get_model_config("ai-product-manager")
            else:
                config = OllamaConfig.get_model_config("garbage-collector")
            
            # Temperature should be between 0 and 1
            assert 0 <= config["temperature"] <= 1
            
            # Top_p should be between 0 and 1
            assert 0 <= config["top_p"] <= 1
            
            # Max tokens should be positive and reasonable
            assert 0 < config["max_tokens"] <= 8192
            
            # Penalties should be reasonable
            assert -2 <= config["frequency_penalty"] <= 2
            assert -2 <= config["presence_penalty"] <= 2


class TestOllamaIntegrationErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.fixture
    def ollama_integration(self):
        return OllamaIntegration()
    
    @pytest.mark.asyncio
    async def test_malformed_json_response(self, ollama_integration):
        """Test handling of malformed JSON responses"""
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.side_effect = json.JSONDecodeError("Invalid JSON", "", 0)
            
            with patch.object(ollama_integration.client, 'post', return_value=mock_response):
                result = await ollama_integration.generate("Test prompt")
                assert result is None
    
    @pytest.mark.asyncio
    async def test_empty_response(self, ollama_integration):
        """Test handling of empty responses"""
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {}
            
            with patch.object(ollama_integration.client, 'post', return_value=mock_response):
                result = await ollama_integration.generate("Test prompt")
                assert result == ""  # Should return empty string, not None
    
    @pytest.mark.asyncio
    async def test_large_prompt_handling(self, ollama_integration):
        """Test handling of very large prompts"""
        large_prompt = "test " * 10000  # Very large prompt
        
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "Response to large prompt"}
            
            with patch.object(ollama_integration.client, 'post', return_value=mock_response):
                result = await ollama_integration.generate(large_prompt)
                assert result == "Response to large prompt"
    
    @pytest.mark.asyncio
    async def test_special_characters_in_prompt(self, ollama_integration):
        """Test handling of special characters in prompts"""
        special_prompt = "Test with special chars: áéíóú ñ 中文 🚀 \n\t\r"
        
        with patch.object(ollama_integration, 'ensure_model_available', return_value=True):
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "Response with special chars"}
            
            with patch.object(ollama_integration.client, 'post', return_value=mock_response):
                result = await ollama_integration.generate(special_prompt)
                assert result == "Response with special chars"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])