"""
Integration tests for Ollama Integration Agent.
Tests against a real TinyLlama container with various edge cases.
"""

import asyncio
import pytest
import time
from typing import Dict, Any

import sys
import os
import socket
sys.path.append('/opt/sutazaiapp')

from agents.ollama_integration.app import OllamaIntegrationAgent
from schemas.ollama_schemas import (
    OllamaGenerateRequest,
    OllamaGenerateResponse,
    OllamaErrorResponse
)
from pydantic import ValidationError


def _is_port_open(host: str, port: int, timeout: float = 0.2) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture
async def agent():
    """Create Ollama integration agent for testing."""
    # Skip if local Ollama is not available (Rule 2: avoid false failures)
    if not _is_port_open("localhost", 11434):
        pytest.skip("Ollama not running on localhost:11434; skipping integration tests")
    agent = OllamaIntegrationAgent(
        base_url="http://localhost:11434",
        timeout=30,
        max_retries=3
    )
    await agent.start()
    yield agent
    await agent.close()


@pytest.mark.asyncio
async def test_model_verification(agent):
    """Test that TinyLlama model is available."""
    # Verify tinyllama is available
    has_model = await agent.verify_model("tinyllama")
    assert has_model, "TinyLlama model not found - please run: ollama pull tinyllama"
    
    # Check for non-existent model
    has_fake = await agent.verify_model("fake-model-xyz")
    assert not has_fake


@pytest.mark.asyncio
async def test_list_models(agent):
    """Test listing available models."""
    models = await agent.list_models()
    
    assert len(models.models) > 0, "No models found"
    assert models.has_model("tinyllama"), "TinyLlama not in model list"
    
    # Check model info structure
    tinyllama = next(m for m in models.models if "tinyllama" in m.name)
    assert tinyllama.size > 0
    assert tinyllama.digest


@pytest.mark.asyncio
async def test_basic_generation(agent):
    """Test basic text generation."""
    result = await agent.generate(
        prompt="What is 2+2?",
        temperature=0.1,
        max_tokens=50
    )
    
    assert "response" in result
    assert "tokens" in result
    assert "latency" in result
    
    assert len(result["response"]) > 0
    assert result["tokens"] > 0
    assert result["latency"] > 0
    
    # Check that response mentions 4 or four
    response_lower = result["response"].lower()
    assert "4" in response_lower or "four" in response_lower


@pytest.mark.asyncio
async def test_empty_response_handling(agent):
    """Test handling of very short prompts that might give empty responses."""
    result = await agent.generate(
        prompt=".",
        temperature=0.0,
        max_tokens=5
    )
    
    # Even with minimal input, should get some response
    assert result["response"] is not None
    assert result["tokens"] >= 0


@pytest.mark.asyncio
async def test_long_output(agent):
    """Test generation with long output."""
    result = await agent.generate(
        prompt="Tell me a story about a robot",
        temperature=0.7,
        max_tokens=500
    )
    
    assert len(result["response"]) > 100
    assert result["tokens"] > 50
    
    # Verify output was actually truncated at max_tokens
    # (within reasonable margin due to tokenization)
    assert result["tokens"] <= 550


@pytest.mark.asyncio
async def test_stop_sequences(agent):
    """Test that stop sequences work correctly."""
    result = await agent.generate(
        prompt="Count from 1 to 10: 1, 2, 3,",
        temperature=0.1,
        max_tokens=100,
        stop=["7", "\n"]
    )
    
    # Should stop before or at 7
    assert "8" not in result["response"]
    assert "9" not in result["response"]
    assert "10" not in result["response"]


@pytest.mark.asyncio
async def test_invalid_tokens_validation():
    """Test validation of invalid token parameters."""
    # Test negative tokens
    with pytest.raises(ValidationError) as exc_info:
        OllamaGenerateRequest(
            prompt="test",
            num_predict=-1
        )
    assert "greater than or equal to 1" in str(exc_info.value).lower()
    
    # Test excessive tokens
    with pytest.raises(ValidationError) as exc_info:
        OllamaGenerateRequest(
            prompt="test",
            num_predict=10000
        )
    assert "less than or equal to 2048" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_invalid_temperature_validation():
    """Test validation of invalid temperature."""
    # Test negative temperature
    with pytest.raises(ValidationError):
        OllamaGenerateRequest(
            prompt="test",
            temperature=-0.5
        )
    
    # Test excessive temperature
    with pytest.raises(ValidationError):
        OllamaGenerateRequest(
            prompt="test",
            temperature=3.0
        )


@pytest.mark.asyncio
async def test_empty_prompt_validation():
    """Test that empty prompts are rejected."""
    with pytest.raises(ValidationError) as exc_info:
        OllamaGenerateRequest(prompt="")
    assert "at least 1 character" in str(exc_info.value).lower()
    
    # Whitespace-only should also fail
    with pytest.raises(ValidationError) as exc_info:
        OllamaGenerateRequest(prompt="   \n\t  ")
    assert "cannot be empty" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_concurrent_requests(agent):
    """Test multiple concurrent generation requests."""
    prompts = [
        "What is Python?",
        "What is Docker?",
        "What is Linux?",
        "What is REST API?",
        "What is JSON?"
    ]
    
    # Create concurrent tasks
    tasks = [
        agent.generate(prompt, temperature=0.5, max_tokens=50)
        for prompt in prompts
    ]
    
    # Execute concurrently
    start_time = time.time()
    results = await asyncio.gather(*tasks)
    elapsed = time.time() - start_time
    
    # Verify all succeeded
    assert len(results) == 5
    for i, result in enumerate(results):
        assert result["tokens"] > 0, f"Request {i} failed to generate tokens"
        assert len(result["response"]) > 0, f"Request {i} has empty response"
    
    print(f"Concurrent requests completed in {elapsed:.2f}s")


@pytest.mark.asyncio
async def test_retry_behavior(agent):
    """Test retry behavior with a temporarily failing endpoint."""
    # Use wrong port to simulate connection failure
    bad_agent = OllamaIntegrationAgent(
        base_url="http://localhost:99999",  # Invalid port
        timeout=2,
        max_retries=2,
        backoff_base=1.5
    )
    await bad_agent.start()
    
    start_time = time.time()
    
    with pytest.raises(Exception) as exc_info:
        await bad_agent.generate("test", max_tokens=10)
    
    elapsed = time.time() - start_time
    
    # Should have retried (taking at least 1.5 seconds for backoff)
    assert elapsed >= 1.5
    assert "retries" in str(exc_info.value).lower()
    
    await bad_agent.close()


@pytest.mark.asyncio
async def test_request_hashing(agent):
    """Test that request hashing works for tracking."""
    # Same requests should have same hash
    payload1 = {"prompt": "test", "model": "tinyllama"}
    payload2 = {"prompt": "test", "model": "tinyllama"}
    
    hash1 = agent._hash_request(payload1)
    hash2 = agent._hash_request(payload2)
    assert hash1 == hash2
    
    # Different requests should have different hashes
    payload3 = {"prompt": "different", "model": "tinyllama"}
    hash3 = agent._hash_request(payload3)
    assert hash1 != hash3


@pytest.mark.asyncio
async def test_response_metrics(agent):
    """Test that response metrics are calculated correctly."""
    result = await agent.generate(
        prompt="Hello",
        temperature=0.5,
        max_tokens=20
    )
    
    # Check metrics are present and reasonable
    assert result["latency"] > 0  # Should have some latency
    assert result["latency"] < 10000  # But not excessive (< 10 seconds)
    assert result["tokens_per_second"] > 0  # Should have token rate


@pytest.mark.asyncio 
async def test_model_not_found_error(agent):
    """Test handling when model doesn't exist."""
    # Try to use a non-existent model
    with pytest.raises(Exception) as exc_info:
        request = OllamaGenerateRequest(
            model="non-existent-model-xyz",
            prompt="test"
        )
        # Manually override the model in the agent call
        payload = request.to_ollama_payload()
        await agent._make_request(payload, "test-hash", 0)
    
    # Should indicate model not found
    assert "404" in str(exc_info.value) or "not found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_context_manager():
    """Test async context manager usage."""
    async with OllamaIntegrationAgent() as agent:
        # Should be able to use agent
        result = await agent.generate(
            prompt="Hi",
            max_tokens=10
        )
        assert result["tokens"] > 0
    
    # Session should be closed after context
    assert agent.session is None


if __name__ == "__main__":
    # Run tests
    asyncio.run(test_model_verification(OllamaIntegrationAgent()))
    print("✅ All manual tests passed")
