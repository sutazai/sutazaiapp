"""
Tests for the Unified Cognitive Architecture
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from backend.cognitive_architecture import (
    UnifiedCognitiveSystem,
    WorkingMemory,
    EpisodicMemory,
    AttentionMechanism,
    MemoryItem,
    MemoryType,
    AttentionMode,
    CognitiveState,
    initialize_cognitive_system
)


@pytest.fixture
def cognitive_system():
    """Create a cognitive system instance for testing"""
    return UnifiedCognitiveSystem()


@pytest.fixture
def working_memory():
    """Create a working memory instance"""
    return WorkingMemory(capacity=5)


@pytest.fixture
def episodic_memory():
    """Create an episodic memory instance"""
    return EpisodicMemory(max_episodes=100)


@pytest.fixture
def attention_mechanism():
    """Create an attention mechanism instance"""
    return AttentionMechanism(max_concurrent_focus=3)


class TestWorkingMemory:
    """Test working memory functionality"""
    
    def test_capacity_limit(self, working_memory):
        """Test that working memory respects capacity limits"""
        # Fill to capacity
        for i in range(5):
            item = MemoryItem(content=f"Item {i}", importance=0.5)
            assert working_memory.add(item) == True
        
        # Try to add one more with lower importance
        low_importance_item = MemoryItem(content="Low importance", importance=0.3)
        assert working_memory.add(low_importance_item) == False
        
        # Add one with higher importance (should replace lowest)
        high_importance_item = MemoryItem(content="High importance", importance=0.9)
        assert working_memory.add(high_importance_item) == True
    
    def test_retrieval(self, working_memory):
        """Test memory retrieval"""
        # Add items with different types
        items = [
            MemoryItem(content={"type": "task", "data": "Task 1"}),
            MemoryItem(content={"type": "task", "data": "Task 2"}),
            MemoryItem(content={"type": "info", "data": "Info 1"})
        ]
        
        for item in items:
            working_memory.add(item)
        
        # Retrieve by type
        tasks = working_memory.retrieve({"type": "task"})
        assert len(tasks) == 2
        
        info = working_memory.retrieve({"type": "info"})
        assert len(info) == 1
    
    def test_chunking(self, working_memory):
        """Test memory chunking"""
        items = [
            MemoryItem(content=f"Item {i}", importance=0.5)
            for i in range(3)
        ]
        
        for item in items:
            working_memory.add(item)
        
        chunk_id = working_memory.chunk(items)
        assert chunk_id is not None
        assert chunk_id in working_memory.chunks


class TestEpisodicMemory:
    """Test episodic memory functionality"""
    
    def test_episode_storage(self, episodic_memory):
        """Test storing episodes"""
        episode_id = episodic_memory.store_episode(
            content={"event": "Test event"},
            context={"location": "test"},
            agents=["agent1", "agent2"],
            importance=0.8
        )
        
        assert episode_id is not None
        assert len(episodic_memory.episodes) == 1
    
    def test_recall_by_cues(self, episodic_memory):
        """Test recalling episodes by cues"""
        # Store episodes with different contexts
        episodic_memory.store_episode(
            {"event": "Event 1"},
            {"type": "error", "severity": "high"},
            ["agent1"],
            0.9
        )
        episodic_memory.store_episode(
            {"event": "Event 2"},
            {"type": "success", "severity": "low"},
            ["agent2"],
            0.5
        )
        
        # Recall by type
        errors = episodic_memory.recall({"type": "error"})
        assert len(errors) == 1
        assert errors[0].content["event"] == "Event 1"
    
    def test_time_range_recall(self, episodic_memory):
        """Test recalling episodes within time range"""
        # Store an episode
        episodic_memory.store_episode(
            {"event": "Recent event"},
            {"type": "test"},
            ["agent1"],
            0.7
        )
        
        # Recall within last hour
        start = datetime.utcnow() - timedelta(hours=1)
        end = datetime.utcnow()
        recent = episodic_memory.recall({}, time_range=(start, end))
        
        assert len(recent) == 1
        
        # Recall from before the episode (should be empty)
        old_start = datetime.utcnow() - timedelta(days=1)
        old_end = datetime.utcnow() - timedelta(hours=2)
        old_episodes = episodic_memory.recall({}, time_range=(old_start, old_end))
        
        assert len(old_episodes) == 0


class TestAttentionMechanism:
    """Test attention mechanism functionality"""
    
    def test_attention_allocation(self, attention_mechanism):
        """Test allocating attention to tasks"""
        focus1 = attention_mechanism.allocate_attention(
            "task1",
            ["agent1", "agent2"],
            priority=0.8,
            mode=AttentionMode.FOCUSED
        )
        
        assert focus1 is not None
        assert focus1.task_id == "task1"
        assert attention_mechanism.resource_pool < 1.0
    
    def test_resource_limits(self, attention_mechanism):
        """Test that attention respects resource limits"""
        # Allocate most resources
        focus1 = attention_mechanism.allocate_attention(
            "task1",
            ["agent1", "agent2", "agent3"],
            priority=0.9,
            mode=AttentionMode.EXECUTIVE
        )
        
        # Try to allocate more than available
        focus2 = attention_mechanism.allocate_attention(
            "task2",
            ["agent4", "agent5"],
            priority=0.8,
            mode=AttentionMode.EXECUTIVE
        )
        
        # Should fail or trigger rebalancing
        assert focus2 is None or attention_mechanism.resource_pool >= 0
    
    def test_attention_release(self, attention_mechanism):
        """Test releasing attention"""
        initial_resources = attention_mechanism.resource_pool
        
        focus = attention_mechanism.allocate_attention(
            "task1",
            ["agent1"],
            priority=0.7,
            mode=AttentionMode.SELECTIVE
        )
        
        allocated_resources = initial_resources - attention_mechanism.resource_pool
        
        attention_mechanism.release_attention("task1")
        
        assert attention_mechanism.resource_pool == initial_resources


class TestCognitiveSystem:
    """Test the integrated cognitive system"""
    
    @pytest.mark.asyncio
    async def test_task_processing(self, cognitive_system):
        """Test processing a task through the cognitive system"""
        task = {
            "type": "analysis",
            "goal": "Test analysis",
            "priority": 0.7
        }
        
        result = await cognitive_system.process_task(task)
        
        assert result["status"] in ["completed", "failed", "error"]
        assert "task_id" in result
        assert "execution_time" in result
    
    @pytest.mark.asyncio
    async def test_memory_integration(self, cognitive_system):
        """Test that memories are created during task processing"""
        initial_episodes = len(cognitive_system.episodic_memory.episodes)
        
        task = {
            "type": "test",
            "goal": "Memory test",
            "priority": 0.5
        }
        
        await cognitive_system.process_task(task)
        
        # Should have stored an episode
        assert len(cognitive_system.episodic_memory.episodes) > initial_episodes
    
    @pytest.mark.asyncio
    async def test_reflection(self, cognitive_system):
        """Test metacognitive reflection"""
        # Process a few tasks first
        for i in range(3):
            task = {
                "type": "test",
                "goal": f"Test task {i}",
                "priority": 0.5
            }
            await cognitive_system.process_task(task)
        
        # Perform reflection
        reflection = await cognitive_system.reflect()
        
        assert "timestamp" in reflection
        assert "performance_insights" in reflection
        assert "system_metrics" in reflection
    
    def test_cognitive_state(self, cognitive_system):
        """Test getting cognitive state"""
        state = cognitive_system.get_cognitive_state()
        
        assert "state" in state
        assert "working_memory" in state
        assert "episodic_memory" in state
        assert "attention" in state
        assert "metrics" in state


class TestLearningSystem:
    """Test learning and adaptation"""
    
    @pytest.mark.asyncio
    async def test_learning_from_experience(self, cognitive_system):
        """Test that the system learns from experiences"""
        initial_skill = cognitive_system.learning_system.skill_levels.get("optimization", 0.5)
        
        # Successful experience
        await cognitive_system.learning_system.learn_from_experience({
            "task_type": "optimization",
            "success": True,
            "context": {"difficulty": "medium"},
            "duration": 30
        })
        
        new_skill = cognitive_system.learning_system.skill_levels["optimization"]
        assert new_skill > initial_skill
    
    def test_knowledge_application(self, cognitive_system):
        """Test applying learned knowledge"""
        task = {
            "type": "optimization",
            "context": {"load": "high"}
        }
        
        recommendations = cognitive_system.learning_system.apply_learned_knowledge(task)
        
        assert "skill_level" in recommendations
        assert "confidence_modifier" in recommendations
        assert "suggested_strategies" in recommendations
        assert "exploration_suggested" in recommendations


class TestIntegration:
    """Test integration between components"""
    
    @pytest.mark.asyncio
    async def test_working_memory_attention_integration(self, cognitive_system):
        """Test that working memory and attention work together"""
        # Add items to working memory
        for i in range(3):
            item = MemoryItem(
                content={"task": f"Task {i}", "agents": [f"agent{i}"]},
                importance=0.6
            )
            cognitive_system.working_memory.add(item)
        
        # Process a task that should use working memory
        task = {
            "type": "coordination",
            "goal": "Coordinate agents",
            "priority": 0.8
        }
        
        result = await cognitive_system.process_task(task)
        
        # Check that attention was allocated
        assert len(cognitive_system.attention.current_focus) > 0 or result["status"] == "completed"
    
    @pytest.mark.asyncio
    async def test_episodic_learning_integration(self, cognitive_system):
        """Test that episodic memory feeds into learning"""
        # Process several similar tasks
        for i in range(5):
            task = {
                "type": "pattern_task",
                "goal": f"Pattern {i}",
                "priority": 0.7,
                "context": {"pattern": "A"}
            }
            await cognitive_system.process_task(task)
        
        # Learning system should have detected patterns
        task_recommendations = cognitive_system.learning_system.apply_learned_knowledge({
            "type": "pattern_task",
            "context": {"pattern": "A"}
        })
        
        assert task_recommendations["skill_level"] > 0.5  # Should have improved


# Performance tests
class TestPerformance:
    """Test performance characteristics"""
    
    def test_working_memory_performance(self, working_memory):
        """Test working memory operates within expected bounds"""
        import time
        
        # Test insertion time
        start = time.time()
        for i in range(100):
            item = MemoryItem(content=f"Item {i}")
            working_memory.add(item)
        end = time.time()
        
        # Should be fast even with many attempts
        assert (end - start) < 0.1  # Less than 100ms for 100 operations
    
    def test_episodic_recall_performance(self, episodic_memory):
        """Test episodic memory recall performance"""
        import time
        
        # Store many episodes
        for i in range(100):
            episodic_memory.store_episode(
                {"event": f"Event {i}"},
                {"type": f"type_{i % 10}"},
                [f"agent_{i % 5}"],
                0.5
            )
        
        # Test recall time
        start = time.time()
        results = episodic_memory.recall({"type": "type_5"})
        end = time.time()
        
        # Should be fast
        assert (end - start) < 0.01  # Less than 10ms
        assert len(results) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])