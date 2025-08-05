"""
Cognitive Integration Manager
============================

This module handles the integration between the cognitive architecture
and existing SutazAI systems including the knowledge graph, agent registry,
and Ollama-based agent infrastructure.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Set
import json

from .unified_cognitive_system import (
    UnifiedCognitiveSystem,
    MemoryItem,
    MemoryType,
    initialize_cognitive_system,
    get_cognitive_system
)
from ..knowledge_graph.manager import get_knowledge_graph_manager
from ..knowledge_graph.query_engine import QueryEngine
from ..ai_agents.core.agent_registry import get_agent_registry

logger = logging.getLogger(__name__)


class CognitiveIntegrationManager:
    """
    Manages integration between cognitive architecture and existing systems
    """
    
    def __init__(self):
        self.cognitive_system: Optional[UnifiedCognitiveSystem] = None
        self.knowledge_graph = None
        self.agent_registry = None
        self.integration_status = {
            "cognitive_system": False,
            "knowledge_graph": False,
            "agent_registry": False,
            "ollama": False
        }
        self.sync_tasks: Set[asyncio.Task] = set()
        
    async def initialize(self) -> bool:
        """Initialize all integrations"""
        try:
            logger.info("Initializing Cognitive Integration Manager")
            
            # Get existing system components
            self.knowledge_graph = get_knowledge_graph_manager()
            self.agent_registry = get_agent_registry()
            
            # Initialize cognitive system
            self.cognitive_system = initialize_cognitive_system(
                agent_registry=self.agent_registry,
                knowledge_graph_manager=self.knowledge_graph
            )
            
            self.integration_status["cognitive_system"] = True
            
            # Initialize specific integrations
            if self.knowledge_graph:
                success = await self._initialize_knowledge_graph_integration()
                self.integration_status["knowledge_graph"] = success
            
            if self.agent_registry:
                success = await self._initialize_agent_integration()
                self.integration_status["agent_registry"] = success
            
            # Start synchronization tasks
            self._start_sync_tasks()
            
            logger.info(f"Cognitive Integration initialized. Status: {self.integration_status}")
            return all(self.integration_status.values())
            
        except Exception as e:
            logger.error(f"Failed to initialize Cognitive Integration: {e}")
            return False
    
    async def _initialize_knowledge_graph_integration(self) -> bool:
        """Set up knowledge graph integration"""
        try:
            if not self.knowledge_graph or not self.cognitive_system:
                return False
            
            query_engine = self.knowledge_graph.get_query_engine()
            if not query_engine:
                return False
            
            # Create cognitive system nodes in knowledge graph
            await self._create_cognitive_nodes(query_engine)
            
            # Set up bidirectional sync
            await self._setup_knowledge_sync()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph integration: {e}")
            return False
    
    async def _initialize_agent_integration(self) -> bool:
        """Set up agent registry integration"""
        try:
            if not self.agent_registry or not self.cognitive_system:
                return False
            
            # Register cognitive system as a meta-agent
            await self._register_cognitive_agent()
            
            # Set up agent monitoring
            await self._setup_agent_monitoring()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize agent integration: {e}")
            return False
    
    async def _create_cognitive_nodes(self, query_engine: QueryEngine):
        """Create cognitive system representation in knowledge graph"""
        # Create main cognitive system node
        cognitive_node = {
            "type": "CognitiveSystem",
            "name": "UnifiedCognitiveArchitecture",
            "components": [
                "WorkingMemory",
                "EpisodicMemory",
                "AttentionMechanism",
                "ExecutiveControl",
                "MetacognitiveMonitor",
                "LearningSystem"
            ],
            "created_at": datetime.utcnow().isoformat()
        }
        
        await query_engine.create_node(cognitive_node)
        
        # Create component nodes
        components = [
            {
                "type": "CognitiveComponent",
                "name": "WorkingMemory",
                "capacity": 7,
                "function": "Short-term information processing"
            },
            {
                "type": "CognitiveComponent", 
                "name": "EpisodicMemory",
                "capacity": 10000,
                "function": "Experience and event storage"
            },
            {
                "type": "CognitiveComponent",
                "name": "AttentionMechanism",
                "capacity": 3,
                "function": "Focus and resource allocation"
            },
            {
                "type": "CognitiveComponent",
                "name": "ExecutiveControl",
                "function": "High-level coordination and planning"
            },
            {
                "type": "CognitiveComponent",
                "name": "MetacognitiveMonitor",
                "function": "Self-awareness and performance monitoring"
            },
            {
                "type": "CognitiveComponent",
                "name": "LearningSystem",
                "function": "Adaptation and improvement"
            }
        ]
        
        for component in components:
            await query_engine.create_node(component)
            
            # Create relationship
            await query_engine.create_relationship(
                "UnifiedCognitiveArchitecture",
                component["name"],
                "HAS_COMPONENT",
                {"integration_date": datetime.utcnow().isoformat()}
            )
    
    async def _setup_knowledge_sync(self):
        """Set up synchronization between cognitive memory and knowledge graph"""
        # This will be called periodically to sync memories to graph
        pass
    
    async def _register_cognitive_agent(self):
        """Register the cognitive system as a meta-agent"""
        if not self.agent_registry:
            return
        
        cognitive_agent_config = {
            "agent_id": "cognitive_system",
            "agent_type": "meta_cognitive",
            "name": "Unified Cognitive System",
            "description": "Central cognitive architecture coordinating all agents",
            "capabilities": [
                "working_memory_management",
                "episodic_memory_storage",
                "attention_allocation",
                "executive_control",
                "metacognitive_monitoring",
                "adaptive_learning",
                "multi_agent_coordination"
            ],
            "model_config": {
                "type": "cognitive_architecture",
                "version": "1.0",
                "components": 6
            },
            "max_concurrent_tasks": 10
        }
        
        # Register with agent registry
        # await self.agent_registry.register_agent(cognitive_agent_config)
    
    async def _setup_agent_monitoring(self):
        """Set up monitoring of agent activities"""
        # Monitor agent performance and feed into metacognitive system
        pass
    
    def _start_sync_tasks(self):
        """Start background synchronization tasks"""
        tasks = [
            self._sync_memories_to_graph(),
            self._sync_learning_insights(),
            self._monitor_agent_performance()
        ]
        
        for coro in tasks:
            task = asyncio.create_task(coro)
            self.sync_tasks.add(task)
            task.add_done_callback(self.sync_tasks.discard)
    
    async def _sync_memories_to_graph(self):
        """Periodically sync episodic memories to knowledge graph"""
        while True:
            try:
                if self.cognitive_system and self.knowledge_graph:
                    # Get recent episodic memories
                    recent_memories = self.cognitive_system.episodic_memory.episodes[-10:]
                    
                    query_engine = self.knowledge_graph.get_query_engine()
                    if query_engine:
                        for memory in recent_memories:
                            # Create memory node in graph
                            memory_node = {
                                "type": "EpisodicMemory",
                                "memory_id": memory.id,
                                "content": json.dumps(memory.content) if isinstance(memory.content, dict) else str(memory.content),
                                "importance": memory.importance,
                                "timestamp": memory.timestamp.isoformat(),
                                "source_agents": memory.source_agents
                            }
                            
                            await query_engine.create_node(memory_node)
                
                await asyncio.sleep(300)  # Sync every 5 minutes
                
            except Exception as e:
                logger.error(f"Error syncing memories to graph: {e}")
                await asyncio.sleep(300)
    
    async def _sync_learning_insights(self):
        """Sync learning insights to knowledge graph"""
        while True:
            try:
                if self.cognitive_system and self.knowledge_graph:
                    # Get learning insights
                    learning_insights = self.cognitive_system.metacognitive_monitor.learning_insights
                    
                    query_engine = self.knowledge_graph.get_query_engine()
                    if query_engine and learning_insights:
                        latest_insight = learning_insights[-1]
                        
                        # Create insight node
                        insight_node = {
                            "type": "LearningInsight",
                            "timestamp": latest_insight["timestamp"].isoformat(),
                            "success_rate": latest_insight["insights"].get("overall_success_rate", 0),
                            "improvement_areas": json.dumps(latest_insight["insights"].get("improvement_areas", [])),
                            "recommendations": json.dumps(latest_insight["insights"].get("recommended_adjustments", []))
                        }
                        
                        await query_engine.create_node(insight_node)
                
                await asyncio.sleep(600)  # Sync every 10 minutes
                
            except Exception as e:
                logger.error(f"Error syncing learning insights: {e}")
                await asyncio.sleep(600)
    
    async def _monitor_agent_performance(self):
        """Monitor agent performance for metacognitive analysis"""
        while True:
            try:
                if self.cognitive_system and self.agent_registry:
                    # Get agent performance metrics
                    # This would integrate with actual agent monitoring
                    pass
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error monitoring agent performance: {e}")
                await asyncio.sleep(60)
    
    async def process_with_cognitive_system(self, task: Dict[str, Any], 
                                          context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Process a task using the cognitive system"""
        if not self.cognitive_system:
            return {"error": "Cognitive system not initialized"}
        
        # Enhance context with system state
        enhanced_context = context or {}
        enhanced_context["integration_status"] = self.integration_status
        enhanced_context["timestamp"] = datetime.utcnow().isoformat()
        
        # Process through cognitive system
        result = await self.cognitive_system.process_task(task, enhanced_context)
        
        # Store result in knowledge graph if available
        if self.knowledge_graph and result.get("status") == "completed":
            await self._store_task_result(task, result)
        
        return result
    
    async def _store_task_result(self, task: Dict[str, Any], result: Dict[str, Any]):
        """Store task result in knowledge graph"""
        try:
            query_engine = self.knowledge_graph.get_query_engine()
            if query_engine:
                # Create task result node
                result_node = {
                    "type": "TaskResult",
                    "task_id": result.get("task_id"),
                    "task_type": task.get("type", "unknown"),
                    "status": result.get("status"),
                    "confidence": result.get("confidence", 0),
                    "execution_time": result.get("execution_time", 0),
                    "agents_used": json.dumps(result.get("agents_used", [])),
                    "timestamp": datetime.utcnow().isoformat()
                }
                
                await query_engine.create_node(result_node)
                
        except Exception as e:
            logger.error(f"Failed to store task result: {e}")
    
    def get_cognitive_state(self) -> Dict[str, Any]:
        """Get current state of the cognitive system"""
        if not self.cognitive_system:
            return {"error": "Cognitive system not initialized"}
        
        state = self.cognitive_system.get_cognitive_state()
        state["integration_status"] = self.integration_status
        
        return state
    
    async def perform_reflection(self) -> Dict[str, Any]:
        """Trigger cognitive reflection"""
        if not self.cognitive_system:
            return {"error": "Cognitive system not initialized"}
        
        reflection = await self.cognitive_system.reflect()
        
        # Store reflection in knowledge graph
        if self.knowledge_graph:
            query_engine = self.knowledge_graph.get_query_engine()
            if query_engine:
                reflection_node = {
                    "type": "CognitiveReflection",
                    "timestamp": reflection["timestamp"],
                    "insights": json.dumps(reflection["performance_insights"]),
                    "metrics": json.dumps(reflection["system_metrics"])
                }
                
                await query_engine.create_node(reflection_node)
        
        return reflection
    
    async def shutdown(self):
        """Shutdown the integration manager"""
        logger.info("Shutting down Cognitive Integration Manager")
        
        # Cancel sync tasks
        for task in self.sync_tasks:
            task.cancel()
        
        if self.sync_tasks:
            await asyncio.gather(*self.sync_tasks, return_exceptions=True)
        
        logger.info("Cognitive Integration Manager shutdown complete")


# Global instance
_integration_manager: Optional[CognitiveIntegrationManager] = None


async def initialize_cognitive_integration() -> bool:
    """Initialize the cognitive integration system"""
    global _integration_manager
    
    try:
        _integration_manager = CognitiveIntegrationManager()
        success = await _integration_manager.initialize()
        
        if success:
            logger.info("Cognitive integration initialized successfully")
        else:
            logger.warning("Cognitive integration initialized with some components unavailable")
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to initialize cognitive integration: {e}")
        return False


def get_integration_manager() -> Optional[CognitiveIntegrationManager]:
    """Get the global integration manager instance"""
    return _integration_manager


async def integrate_with_knowledge_graph(knowledge_graph_manager):
    """Integrate cognitive system with knowledge graph"""
    if _integration_manager:
        _integration_manager.knowledge_graph = knowledge_graph_manager
        return await _integration_manager._initialize_knowledge_graph_integration()
    return False


async def integrate_with_agents(agent_registry):
    """Integrate cognitive system with agent registry"""
    if _integration_manager:
        _integration_manager.agent_registry = agent_registry
        return await _integration_manager._initialize_agent_integration()
    return False