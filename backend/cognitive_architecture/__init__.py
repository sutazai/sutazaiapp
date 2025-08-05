"""
Cognitive Architecture Module for SutazAI
========================================

This module provides human-like cognitive capabilities for the multi-agent system.
"""

from .unified_cognitive_system import (
    UnifiedCognitiveSystem,
    WorkingMemory,
    EpisodicMemory,
    AttentionMechanism,
    ExecutiveControl,
    MetacognitiveMonitor,
    LearningSystem,
    MemoryType,
    AttentionMode,
    CognitiveState,
    MemoryItem,
    AttentionFocus,
    ReasoningChain,
    get_cognitive_system,
    initialize_cognitive_system
)

from .cognitive_integration import (
    CognitiveIntegrationManager,
    integrate_with_knowledge_graph,
    integrate_with_agents,
    initialize_cognitive_integration
)

__all__ = [
    # Main system
    "UnifiedCognitiveSystem",
    "get_cognitive_system",
    "initialize_cognitive_system",
    
    # Components
    "WorkingMemory",
    "EpisodicMemory", 
    "AttentionMechanism",
    "ExecutiveControl",
    "MetacognitiveMonitor",
    "LearningSystem",
    
    # Data structures
    "MemoryItem",
    "AttentionFocus",
    "ReasoningChain",
    
    # Enums
    "MemoryType",
    "AttentionMode",
    "CognitiveState",
    
    # Integration
    "CognitiveIntegrationManager",
    "integrate_with_knowledge_graph",
    "integrate_with_agents",
    "initialize_cognitive_integration"
]