"""
Cognitive Architecture Demonstration
===================================

This script demonstrates the capabilities of the unified cognitive architecture
including working memory, episodic memory, attention mechanisms, and learning.
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, Any, List

from .unified_cognitive_system import (
    UnifiedCognitiveSystem,
    MemoryItem,
    MemoryType,
    AttentionMode,
    ReasoningType,
    initialize_cognitive_system
)
from .cognitive_integration import initialize_cognitive_integration


async def demonstrate_working_memory(cognitive_system: UnifiedCognitiveSystem):
    """Demonstrate working memory capabilities"""
    print("\n=== Working Memory Demonstration ===")
    
    # Add items to working memory
    items = [
        {"type": "task", "content": "Analyze system logs", "priority": 0.8},
        {"type": "data", "content": "Server CPU at 85%", "priority": 0.9},
        {"type": "goal", "content": "Optimize performance", "priority": 0.7},
        {"type": "constraint", "content": "Minimize downtime", "priority": 0.85},
        {"type": "insight", "content": "Peak usage at 3pm", "priority": 0.6}
    ]
    
    for item in items:
        memory_item = MemoryItem(
            content=item,
            memory_type=MemoryType.WORKING,
            importance=item["priority"]
        )
        success = cognitive_system.working_memory.add(memory_item)
        print(f"Added to working memory: {item['content']} - Success: {success}")
    
    # Demonstrate retrieval
    query = {"type": "task"}
    retrieved = cognitive_system.working_memory.retrieve(query)
    print(f"\nRetrieved items matching {query}: {len(retrieved)} items")
    for item in retrieved:
        print(f"  - {item.content['content']} (importance: {item.importance})")
    
    # Demonstrate chunking
    chunk_items = retrieved[:3] if len(retrieved) >= 3 else retrieved
    if chunk_items:
        chunk_id = cognitive_system.working_memory.chunk(chunk_items)
        print(f"\nCreated chunk {chunk_id} from {len(chunk_items)} items")
    
    # Show attention weights
    print("\nCurrent attention weights:")
    for item_id, weight in cognitive_system.working_memory.attention_weights.items():
        print(f"  - Item {item_id[:8]}: {weight:.3f}")


async def demonstrate_episodic_memory(cognitive_system: UnifiedCognitiveSystem):
    """Demonstrate episodic memory capabilities"""
    print("\n=== Episodic Memory Demonstration ===")
    
    # Store episodes
    episodes = [
        {
            "content": {"event": "System optimization", "result": "20% improvement"},
            "context": {"time": "morning", "load": "high"},
            "agents": ["performance-optimizer", "system-monitor"],
            "importance": 0.9
        },
        {
            "content": {"event": "Security scan", "result": "No threats found"},
            "context": {"time": "night", "type": "scheduled"},
            "agents": ["security-scanner"],
            "importance": 0.7
        },
        {
            "content": {"event": "User complaint", "result": "Resolved"},
            "context": {"time": "afternoon", "severity": "low"},
            "agents": ["customer-support", "system-debugger"],
            "importance": 0.6
        }
    ]
    
    episode_ids = []
    for ep in episodes:
        ep_id = cognitive_system.episodic_memory.store_episode(
            ep["content"],
            ep["context"],
            ep["agents"],
            ep["importance"]
        )
        episode_ids.append(ep_id)
        print(f"Stored episode: {ep['content']['event']} (ID: {ep_id[:8]})")
    
    # Recall by cues
    print("\nRecalling episodes with time='morning':")
    recalled = cognitive_system.episodic_memory.recall({"time": "morning"})
    for memory in recalled:
        print(f"  - {memory.content}")
    
    # Recall with time range
    start_time = datetime.utcnow() - timedelta(hours=1)
    end_time = datetime.utcnow()
    print(f"\nRecalling episodes from last hour:")
    recent_episodes = cognitive_system.episodic_memory.recall(
        {},
        time_range=(start_time, end_time)
    )
    print(f"  Found {len(recent_episodes)} recent episodes")
    
    # Consolidate memories
    cognitive_system.episodic_memory.consolidate()
    print("\nMemory consolidation complete")


async def demonstrate_attention_mechanism(cognitive_system: UnifiedCognitiveSystem):
    """Demonstrate attention allocation and management"""
    print("\n=== Attention Mechanism Demonstration ===")
    
    # Allocate attention to different tasks
    tasks = [
        {
            "id": "task-001",
            "agents": ["analyzer", "optimizer"],
            "priority": 0.9,
            "mode": AttentionMode.FOCUSED
        },
        {
            "id": "task-002",
            "agents": ["monitor"],
            "priority": 0.6,
            "mode": AttentionMode.SUSTAINED
        },
        {
            "id": "task-003",
            "agents": ["security", "logger"],
            "priority": 0.7,
            "mode": AttentionMode.SELECTIVE
        }
    ]
    
    allocated_tasks = []
    for task in tasks:
        focus = cognitive_system.attention.allocate_attention(
            task["id"],
            task["agents"],
            task["priority"],
            task["mode"]
        )
        
        if focus:
            allocated_tasks.append(task["id"])
            print(f"Allocated attention to {task['id']}: {focus.allocated_resources:.2f} resources")
        else:
            print(f"Failed to allocate attention to {task['id']} - insufficient resources")
    
    # Show attention distribution
    distribution = cognitive_system.attention.get_attention_distribution()
    print("\nCurrent attention distribution:")
    for agent, weight in distribution.items():
        print(f"  - {agent}: {weight:.3f}")
    
    # Release attention from first task
    if allocated_tasks:
        cognitive_system.attention.release_attention(allocated_tasks[0])
        print(f"\nReleased attention from {allocated_tasks[0]}")
        print(f"Available resources: {cognitive_system.attention.resource_pool:.2f}")


async def demonstrate_reasoning(cognitive_system: UnifiedCognitiveSystem):
    """Demonstrate reasoning capabilities"""
    print("\n=== Reasoning Demonstration ===")
    
    # Complex analysis task
    task = {
        "type": "complex_analysis",
        "goal": "Identify system bottlenecks and propose optimizations",
        "priority": 0.9,
        "reasoning_type": ReasoningType.DEDUCTIVE.value,
        "context": {
            "current_performance": "degraded",
            "affected_services": ["api", "database"],
            "time_of_day": "peak_hours"
        }
    }
    
    print(f"Processing task: {task['goal']}")
    result = await cognitive_system.process_task(task)
    
    print(f"\nTask completed with status: {result['status']}")
    print(f"Confidence: {result.get('confidence', 0):.2f}")
    print(f"Agents used: {result.get('agents_used', [])}")
    print(f"Execution time: {result.get('execution_time', 0):.2f}s")
    
    if "reasoning_chain_id" in result:
        print(f"Reasoning chain: {result['reasoning_chain_id'][:8]}")
    
    # Creative problem solving
    creative_task = {
        "type": "creative_solution",
        "goal": "Design innovative caching strategy",
        "priority": 0.8,
        "reasoning_type": ReasoningType.CREATIVE.value
    }
    
    print(f"\n\nProcessing creative task: {creative_task['goal']}")
    creative_result = await cognitive_system.process_task(creative_task)
    
    print(f"Creative task status: {creative_result['status']}")
    if "learning_applied" in creative_result:
        print(f"Learning recommendations applied: {creative_result['learning_applied']}")


async def demonstrate_metacognition(cognitive_system: UnifiedCognitiveSystem):
    """Demonstrate metacognitive monitoring and reflection"""
    print("\n=== Metacognitive Monitoring Demonstration ===")
    
    # Simulate task performance monitoring
    tasks_results = [
        {"task": {"type": "analysis"}, "predicted": 0.8, "success": True},
        {"task": {"type": "analysis"}, "predicted": 0.9, "success": True},
        {"task": {"type": "optimization"}, "predicted": 0.7, "success": False},
        {"task": {"type": "analysis"}, "predicted": 0.85, "success": True},
        {"task": {"type": "optimization"}, "predicted": 0.6, "success": True},
    ]
    
    for tr in tasks_results:
        cognitive_system.metacognitive_monitor.monitor_performance(
            tr["task"],
            {"duration": 10, "resources_used": {"cpu": 0.5}},
            tr["predicted"],
            tr["success"]
        )
    
    # Perform reflection
    print("Performing metacognitive reflection...")
    reflection = await cognitive_system.reflect()
    
    print(f"\nReflection timestamp: {reflection['timestamp']}")
    
    if "performance_insights" in reflection:
        insights = reflection["performance_insights"]
        print(f"Overall success rate: {insights.get('overall_success_rate', 0):.2%}")
        
        if "confidence_calibration" in insights:
            print("\nConfidence calibration scores:")
            for task_type, score in insights["confidence_calibration"].items():
                print(f"  - {task_type}: {score:.3f}")
        
        if "improvement_areas" in insights:
            print("\nIdentified improvement areas:")
            for area in insights["improvement_areas"]:
                print(f"  - {area}")
        
        if "recommended_adjustments" in insights:
            print("\nRecommended adjustments:")
            for adj in insights["recommended_adjustments"][:3]:
                print(f"  - {adj['type']}: {adj.get('recommendation', 'N/A')}")


async def demonstrate_learning(cognitive_system: UnifiedCognitiveSystem):
    """Demonstrate learning and adaptation"""
    print("\n=== Learning System Demonstration ===")
    
    # Simulate learning from experiences
    experiences = [
        {
            "task_type": "optimization",
            "success": True,
            "context": {"load": "high", "time": "peak"},
            "approach": {"strategy": "caching", "agents": ["cache-manager"]},
            "duration": 45
        },
        {
            "task_type": "optimization",
            "success": False,
            "context": {"load": "high", "time": "peak"},
            "approach": {"strategy": "scaling", "agents": ["auto-scaler"]},
            "duration": 120
        },
        {
            "task_type": "optimization",
            "success": True,
            "context": {"load": "medium", "time": "normal"},
            "approach": {"strategy": "tuning", "agents": ["performance-tuner"]},
            "duration": 30
        }
    ]
    
    print("Learning from experiences...")
    for exp in experiences:
        await cognitive_system.learning_system.learn_from_experience(exp)
        print(f"  - Learned from {exp['task_type']} task (success: {exp['success']})")
    
    # Apply learned knowledge
    new_task = {
        "type": "optimization",
        "context": {"load": "high", "time": "peak"}
    }
    
    recommendations = cognitive_system.learning_system.apply_learned_knowledge(new_task)
    
    print(f"\nLearning recommendations for new task:")
    print(f"  - Skill level: {recommendations['skill_level']:.2f}")
    print(f"  - Confidence modifier: {recommendations['confidence_modifier']:.2f}")
    print(f"  - Exploration suggested: {recommendations['exploration_suggested']}")
    
    if recommendations["suggested_strategies"]:
        print("  - Suggested strategies:")
        for strategy in recommendations["suggested_strategies"]:
            print(f"    * {strategy.get('approach', {})}")
    
    # Show skill levels
    print("\nCurrent skill levels:")
    for task_type, skill in cognitive_system.learning_system.skill_levels.items():
        print(f"  - {task_type}: {skill:.3f}")
    
    # Consolidate learning
    cognitive_system.learning_system.consolidate_learning()
    print("\nLearning consolidation complete")


async def demonstrate_full_cognitive_cycle(cognitive_system: UnifiedCognitiveSystem):
    """Demonstrate a complete cognitive processing cycle"""
    print("\n=== Full Cognitive Cycle Demonstration ===")
    
    # Complex multi-step task
    complex_task = {
        "id": "demo-task-001",
        "type": "system_optimization",
        "goal": "Comprehensive system analysis and optimization",
        "priority": 0.95,
        "reasoning_type": ReasoningType.STRATEGIC.value,
        "max_agents": 7,
        "context": {
            "trigger": "performance_degradation",
            "metrics": {
                "response_time": 2500,  # ms
                "error_rate": 0.05,
                "cpu_usage": 0.85
            },
            "constraints": {
                "downtime": "minimal",
                "budget": "moderate"
            }
        }
    }
    
    print(f"Starting complex task: {complex_task['goal']}")
    print(f"Initial metrics: {json.dumps(complex_task['context']['metrics'], indent=2)}")
    
    # Process the task
    start_time = datetime.utcnow()
    result = await cognitive_system.process_task(complex_task)
    end_time = datetime.utcnow()
    
    print(f"\n{'='*50}")
    print(f"Task completed in {(end_time - start_time).total_seconds():.2f} seconds")
    print(f"Status: {result['status']}")
    print(f"Confidence: {result.get('confidence', 0):.2%}")
    
    if result.get('agents_used'):
        print(f"\nAgents involved ({len(result['agents_used'])}):")
        for agent in result['agents_used']:
            print(f"  - {agent}")
    
    if result.get('memory_references'):
        print(f"\nMemory references used: {len(result['memory_references'])}")
    
    # Show cognitive state after processing
    state = cognitive_system.get_cognitive_state()
    print(f"\nCognitive state after processing:")
    print(f"  - Working memory utilization: {state['working_memory']['items']}/{state['working_memory']['capacity']}")
    print(f"  - Episodic memories: {state['episodic_memory']['episodes']}")
    print(f"  - Active reasoning chains: {state['active_reasoning']}")
    print(f"  - Attention allocation: {len(state['attention']['current_focus'])} active focuses")
    
    # Final reflection
    print(f"\n{'='*50}")
    print("Performing final reflection...")
    final_reflection = await cognitive_system.reflect()
    
    metrics = final_reflection.get('system_metrics', {})
    print(f"\nSystem metrics:")
    print(f"  - Total tasks processed: {metrics.get('total_tasks_processed', 0)}")
    print(f"  - Success rate: {metrics.get('successful_tasks', 0)}/{metrics.get('total_tasks_processed', 1)}")
    print(f"  - Average response time: {metrics.get('average_response_time', 0):.2f}s")
    print(f"  - Memory utilization: {metrics.get('memory_utilization', 0):.1%}")


async def main():
    """Main demonstration function"""
    print("SutazAI Unified Cognitive Architecture Demonstration")
    print("=" * 60)
    
    # Initialize the cognitive system
    print("\nInitializing cognitive architecture...")
    
    # For demo, we'll create a standalone instance
    cognitive_system = UnifiedCognitiveSystem()
    
    print("Cognitive system initialized successfully!")
    
    # Run demonstrations
    await demonstrate_working_memory(cognitive_system)
    await asyncio.sleep(0.5)
    
    await demonstrate_episodic_memory(cognitive_system)
    await asyncio.sleep(0.5)
    
    await demonstrate_attention_mechanism(cognitive_system)
    await asyncio.sleep(0.5)
    
    await demonstrate_reasoning(cognitive_system)
    await asyncio.sleep(0.5)
    
    await demonstrate_learning(cognitive_system)
    await asyncio.sleep(0.5)
    
    await demonstrate_metacognition(cognitive_system)
    await asyncio.sleep(0.5)
    
    await demonstrate_full_cognitive_cycle(cognitive_system)
    
    print("\n" + "=" * 60)
    print("Demonstration complete!")
    
    # Show final summary
    final_state = cognitive_system.get_cognitive_state()
    print(f"\nFinal system state:")
    print(f"  - Cognitive state: {final_state['state']}")
    print(f"  - Total memories: {final_state['episodic_memory']['episodes']} episodes")
    print(f"  - Learning progress: {len(final_state['learning']['skill_levels'])} skills tracked")
    print(f"  - System health: Operational")


if __name__ == "__main__":
    asyncio.run(main())