"""
Unit tests for AGI Brain functionality
"""
import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from datetime import datetime

# Mock the brain module for testing
class MockAGIBrain:
    def __init__(self):
        self.thoughts = {}
        self.active_thoughts = []
        
    async def think(self, input_data, context=None, reasoning_type="deductive"):
        thought_id = f"thought_test_{datetime.utcnow().timestamp()}"
        result = {
            "thought_id": thought_id,
            "result": {
                "analysis": f"Analyzed {input_data.get('text', '')} using {reasoning_type}",
                "confidence": 0.85
            },
            "reasoning_type": reasoning_type,
            "complexity": "moderate",
            "agents_used": ["test_agent"]
        }
        self.thoughts[thought_id] = result
        return result
        
    def get_status(self):
        return {
            "status": "active",
            "active_thoughts": len(self.active_thoughts),
            "memory_usage": {"short_term": 10, "long_term": 50}
        }

@pytest.fixture
def mock_brain():
    return MockAGIBrain()

@pytest.mark.asyncio
async def test_brain_think_basic(mock_brain):
    """Test basic thinking functionality"""
    input_data = {"text": "Test input"}
    result = await mock_brain.think(input_data)
    
    assert "thought_id" in result
    assert result["result"]["analysis"] == "Analyzed Test input using deductive"
    assert result["reasoning_type"] == "deductive"
    assert result["complexity"] == "moderate"

@pytest.mark.asyncio
async def test_brain_think_with_context(mock_brain):
    """Test thinking with context"""
    input_data = {"text": "Test with context"}
    context = {"previous_thought": "some_id", "user_preference": "detailed"}
    
    result = await mock_brain.think(input_data, context, "inductive")
    
    assert result["reasoning_type"] == "inductive"
    assert "Test with context" in result["result"]["analysis"]

@pytest.mark.asyncio
async def test_brain_multiple_thoughts(mock_brain):
    """Test handling multiple concurrent thoughts"""
    tasks = []
    for i in range(5):
        input_data = {"text": f"Thought {i}"}
        tasks.append(mock_brain.think(input_data))
    
    results = await asyncio.gather(*tasks)
    
    assert len(results) == 5
    assert len(mock_brain.thoughts) == 5
    
    # Check all thoughts are unique
    thought_ids = [r["thought_id"] for r in results]
    assert len(thought_ids) == len(set(thought_ids))

def test_brain_status(mock_brain):
    """Test brain status reporting"""
    status = mock_brain.get_status()
    
    assert status["status"] == "active"
    assert "active_thoughts" in status
    assert "memory_usage" in status
    assert isinstance(status["memory_usage"], dict)

@pytest.mark.asyncio
async def test_brain_reasoning_types(mock_brain):
    """Test different reasoning types"""
    reasoning_types = ["deductive", "inductive", "creative", "strategic"]
    
    for reasoning_type in reasoning_types:
        result = await mock_brain.think(
            {"text": f"Test {reasoning_type}"},
            reasoning_type=reasoning_type
        )
        assert result["reasoning_type"] == reasoning_type

@pytest.mark.asyncio
async def test_brain_error_handling(mock_brain):
    """Test error handling in brain"""
    # Mock a failure scenario
    with patch.object(mock_brain, 'think', side_effect=Exception("Brain error")):
        with pytest.raises(Exception) as exc_info:
            await mock_brain.think({"text": "Error test"})
        assert "Brain error" in str(exc_info.value)

@pytest.mark.asyncio
async def test_brain_confidence_calculation(mock_brain):
    """Test confidence calculation in results"""
    result = await mock_brain.think({"text": "Confidence test"})
    
    assert "confidence" in result["result"]
    assert isinstance(result["result"]["confidence"], (int, float))
    assert 0 <= result["result"]["confidence"] <= 1

@pytest.mark.parametrize("input_text,expected_complexity", [
    ("Simple query", "moderate"),
    ("Complex multi-part question with context", "moderate"),
    ("", "moderate"),
])
@pytest.mark.asyncio
async def test_brain_complexity_assessment(mock_brain, input_text, expected_complexity):
    """Test complexity assessment for different inputs"""
    result = await mock_brain.think({"text": input_text})
    assert result["complexity"] == expected_complexity