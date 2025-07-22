"""
Unit tests for AGI Brain functionality - Real Implementation
"""
import pytest
import asyncio
import sys
import os
from datetime import datetime
from typing import Dict, Any

# Add the backend directory to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from app.core.agi_brain import AGIBrain, ReasoningType, TaskComplexity

@pytest.fixture
async def brain():
    """Create real AGI Brain instance"""
    brain_instance = AGIBrain()
    yield brain_instance
    # Cleanup
    brain_instance.active_thoughts.clear()
    brain_instance.memory_short_term.clear()
    brain_instance.memory_long_term.clear()

@pytest.mark.asyncio
async def test_brain_initialization(brain):
    """Test AGI Brain initialization"""
    assert brain.knowledge_graph == {}
    assert brain.active_thoughts == []
    assert brain.memory_short_term == []
    assert brain.memory_long_term == {}
    assert brain.learning_rate == 0.01
    assert brain.creativity_factor == 0.3
    assert brain.confidence_threshold == 0.7

@pytest.mark.asyncio
async def test_brain_think_basic(brain):
    """Test basic thinking functionality with real brain"""
    input_data = {"text": "Analyze the benefits of renewable energy"}
    
    result = await brain.think(
        input_data=input_data,
        reasoning_type=ReasoningType.DEDUCTIVE
    )
    
    assert "thought_id" in result
    assert result["thought_id"].startswith("thought_")
    assert "result" in result
    assert result["reasoning_type"] == "deductive"
    assert result["complexity"] in ["simple", "moderate", "complex", "expert", "research"]
    assert isinstance(result["agents_used"], list)
    assert result["confidence"] >= 0 and result["confidence"] <= 1

@pytest.mark.asyncio
async def test_brain_think_with_context(brain):
    """Test thinking with context"""
    input_data = {"text": "What are the implications of this data?"}
    context = {
        "previous_analysis": "Data shows 30% increase in efficiency",
        "domain": "energy",
        "user_preference": "detailed_analysis"
    }
    
    result = await brain.think(
        input_data=input_data,
        context=context,
        reasoning_type=ReasoningType.ANALYTICAL
    )
    
    assert result["thought_id"] is not None
    assert len(brain.active_thoughts) > 0
    
    # Check that context was processed
    thought = brain.active_thoughts[0]
    assert thought["context"] == context

@pytest.mark.asyncio
async def test_brain_understanding_process(brain):
    """Test the understanding phase of thinking"""
    input_data = {
        "text": "Create a Python function to calculate fibonacci numbers"
    }
    
    understanding = await brain._understand_input(input_data, None)
    
    assert "intent" in understanding
    assert understanding["intent"] == "create"  # Should detect 'create' intent
    assert "entities" in understanding
    assert "domain" in understanding
    assert understanding["domain"] == "code"  # Should detect code domain
    assert "requirements" in understanding

@pytest.mark.asyncio
async def test_brain_complexity_assessment(brain):
    """Test complexity assessment"""
    # Simple task
    simple_understanding = {
        "entities": ["hello"],
        "requirements": [],
        "domain": "general"
    }
    complexity = await brain._assess_complexity(simple_understanding)
    assert complexity == TaskComplexity.SIMPLE
    
    # Complex task
    complex_understanding = {
        "entities": ["system", "architecture", "microservices", "kubernetes", "monitoring"],
        "requirements": ["scalability", "security", "performance", "reliability"],
        "domain": "code"
    }
    complexity = await brain._assess_complexity(complex_understanding)
    assert complexity in [TaskComplexity.COMPLEX, TaskComplexity.EXPERT]

@pytest.mark.asyncio
async def test_brain_agent_selection(brain):
    """Test agent selection based on task"""
    # Code task
    code_understanding = {"domain": "code"}
    agents = await brain._select_agents(code_understanding, TaskComplexity.MODERATE)
    assert any(agent in ["gpt_engineer", "aider", "tabbyml"] for agent in agents)
    assert len(agents) <= 2  # Moderate complexity limits agents
    
    # Security task
    security_understanding = {"domain": "security"}
    agents = await brain._select_agents(security_understanding, TaskComplexity.COMPLEX)
    assert any(agent in ["semgrep", "pentestgpt"] for agent in agents)
    assert len(agents) <= 3  # Complex allows more agents

@pytest.mark.asyncio
async def test_brain_reasoning_chain_creation(brain):
    """Test reasoning chain creation for different types"""
    understanding = {"domain": "general", "requirements": ["analysis"]}
    agents = ["langchain", "autogpt"]
    
    # Test deductive reasoning
    deductive_chain = await brain._create_reasoning_chain(
        understanding, agents, ReasoningType.DEDUCTIVE
    )
    assert len(deductive_chain) == 3
    assert deductive_chain[0]["step"] == "establish_premises"
    assert deductive_chain[1]["step"] == "apply_rules"
    assert deductive_chain[2]["step"] == "derive_conclusion"
    
    # Test creative reasoning
    creative_chain = await brain._create_reasoning_chain(
        understanding, agents, ReasoningType.CREATIVE
    )
    assert len(creative_chain) == 4
    assert "brainstorm_ideas" in [step["step"] for step in creative_chain]
    assert "evaluate_novelty" in [step["step"] for step in creative_chain]

@pytest.mark.asyncio
async def test_brain_memory_management(brain):
    """Test memory storage and retrieval"""
    # Create some thoughts
    for i in range(5):
        await brain.think(
            {"text": f"Test thought {i}"},
            reasoning_type=ReasoningType.DEDUCTIVE
        )
    
    # Check short-term memory
    assert len(brain.memory_short_term) == 5
    
    # Test memory summarization
    last_memory = brain.memory_short_term[-1]
    assert "thought_id" in last_memory
    assert "timestamp" in last_memory
    assert "summary" in last_memory

@pytest.mark.asyncio
async def test_brain_learning_process(brain):
    """Test the learning mechanism"""
    initial_learning_rate = brain.learning_rate
    
    # Successful high-confidence thought
    thought = {
        "id": "test_thought_1",
        "input": {"text": "test"},
        "reasoning_type": "deductive"
    }
    
    result = {
        "quality_metrics": {"confidence": 0.95},
        "agents_used": ["test_agent"]
    }
    
    await brain._learn_from_experience(thought, result)
    
    # Learning rate should increase slightly for success
    assert brain.learning_rate > initial_learning_rate

@pytest.mark.asyncio
async def test_brain_concurrent_thoughts(brain):
    """Test handling multiple concurrent thoughts"""
    tasks = []
    num_thoughts = 5
    
    for i in range(num_thoughts):
        task = brain.think(
            {"text": f"Concurrent thought {i}"},
            reasoning_type=ReasoningType.DEDUCTIVE
        )
        tasks.append(task)
    
    results = await asyncio.gather(*tasks)
    
    # All thoughts should complete successfully
    assert len(results) == num_thoughts
    
    # All thought IDs should be unique
    thought_ids = [r["thought_id"] for r in results]
    assert len(thought_ids) == len(set(thought_ids))
    
    # Active thoughts should be tracked
    assert len(brain.active_thoughts) == num_thoughts

@pytest.mark.asyncio
async def test_brain_error_handling(brain):
    """Test error handling in thinking process"""
    # Test with invalid input
    result = await brain.think(
        {"invalid_key": "no text field"},
        reasoning_type=ReasoningType.DEDUCTIVE
    )
    
    # Should still return a result (graceful handling)
    assert "thought_id" in result
    # But may have lower confidence or error indication
    assert result.get("status") == "failed" or result["confidence"] < brain.confidence_threshold

@pytest.mark.asyncio
async def test_brain_domain_detection(brain):
    """Test domain detection from input"""
    test_cases = [
        ("Write a function to sort an array", "code"),
        ("Check for SQL injection vulnerabilities", "security"),
        ("Create API documentation", "documentation"),
        ("Analyze sales data trends", "analysis"),
        ("Design a new logo", "creative"),
        ("Scrape data from website", "web"),
        ("What is the weather today?", "general")
    ]
    
    for text, expected_domain in test_cases:
        domain = brain._identify_domain({"text": text})
        assert domain == expected_domain

@pytest.mark.asyncio
async def test_brain_knowledge_graph_update(brain):
    """Test knowledge graph updates during learning"""
    # Perform several thoughts
    reasoning_types = [ReasoningType.DEDUCTIVE, ReasoningType.INDUCTIVE, ReasoningType.CREATIVE]
    
    for reasoning_type in reasoning_types:
        await brain.think(
            {"text": f"Test {reasoning_type.value} reasoning"},
            reasoning_type=reasoning_type
        )
    
    # Knowledge graph should be updated
    assert len(brain.knowledge_graph) > 0
    
    # Check structure
    for key, learnings in brain.knowledge_graph.items():
        assert isinstance(learnings, list)
        if learnings:
            assert "thought_id" in learnings[0]
            assert "reasoning_type" in learnings[0]

@pytest.mark.asyncio
async def test_brain_confidence_calculation(brain):
    """Test confidence calculation logic"""
    # Test various result scenarios
    test_results = [
        {"refined": True, "intermediate_insights": ["insight1", "insight2"]},  # High confidence
        {"refined": False, "intermediate_insights": []},  # Base confidence
        {"refined": True, "intermediate_insights": []},  # Medium confidence
    ]
    
    confidences = []
    for result in test_results:
        confidence = brain._calculate_confidence(result)
        confidences.append(confidence)
        assert 0 <= confidence <= 1
    
    # Refined with insights should have highest confidence
    assert confidences[0] > confidences[1]
    assert confidences[0] > confidences[2]

@pytest.mark.asyncio
async def test_brain_result_validation(brain):
    """Test result validation and refinement"""
    mock_result = {
        "chain_results": [{"output": "test"}],
        "final_output": "Test output"
    }
    
    understanding = {
        "requirements": ["accuracy", "completeness"]
    }
    
    refined_result = await brain._validate_and_refine(mock_result, understanding)
    
    assert "quality_metrics" in refined_result
    assert "consistency" in refined_result["quality_metrics"]
    assert "completeness" in refined_result["quality_metrics"]
    assert "confidence" in refined_result["quality_metrics"]

@pytest.mark.parametrize("reasoning_type", [
    ReasoningType.DEDUCTIVE,
    ReasoningType.INDUCTIVE,
    ReasoningType.ABDUCTIVE,
    ReasoningType.ANALOGICAL,
    ReasoningType.CAUSAL,
    ReasoningType.CREATIVE,
    ReasoningType.STRATEGIC
])
@pytest.mark.asyncio
async def test_brain_all_reasoning_types(brain, reasoning_type):
    """Test all reasoning types work correctly"""
    result = await brain.think(
        {"text": f"Test {reasoning_type.value} reasoning"},
        reasoning_type=reasoning_type
    )
    
    assert result["reasoning_type"] == reasoning_type.value
    assert result["thought_id"] is not None
    assert "result" in result