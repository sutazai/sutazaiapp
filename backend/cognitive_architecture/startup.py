"""
Cognitive Architecture Startup Integration
=========================================

This module handles the startup and integration of the cognitive architecture
with the main SutazAI application.
"""

import asyncio
import logging
from typing import Optional

from fastapi import FastAPI
from .api import router as cognitive_router
from .cognitive_integration import initialize_cognitive_integration, get_integration_manager
from ..knowledge_graph.manager import get_knowledge_graph_manager
from ..ai_agents.core.agent_registry import get_agent_registry

logger = logging.getLogger(__name__)


async def startup_cognitive_system(app: FastAPI) -> bool:
    """
    Initialize and start the cognitive architecture system
    
    Args:
        app: FastAPI application instance
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        logger.info("Starting Cognitive Architecture initialization...")
        
        # Include the cognitive API router
        app.include_router(cognitive_router)
        logger.info("Cognitive API endpoints registered")
        
        # Initialize the cognitive system
        success = await initialize_cognitive_integration()
        
        if success:
            logger.info("Cognitive Architecture initialized successfully")
            
            # Store reference in app state
            app.state.cognitive_system = get_integration_manager()
            
            # Log initial state
            if app.state.cognitive_system:
                state = app.state.cognitive_system.get_cognitive_state()
                logger.info(f"Cognitive system state: {state}")
                
                # Trigger initial reflection
                logger.info("Performing initial cognitive reflection...")
                reflection = await app.state.cognitive_system.perform_reflection()
                logger.info(f"Initial reflection complete: {reflection.get('performance_insights', {})}")
        else:
            logger.warning("Cognitive Architecture initialized with partial functionality")
            
        return success
        
    except Exception as e:
        logger.error(f"Failed to start Cognitive Architecture: {e}")
        return False


async def shutdown_cognitive_system(app: FastAPI):
    """
    Shutdown the cognitive architecture system
    
    Args:
        app: FastAPI application instance
    """
    try:
        logger.info("Shutting down Cognitive Architecture...")
        
        if hasattr(app.state, 'cognitive_system') and app.state.cognitive_system:
            await app.state.cognitive_system.shutdown()
            logger.info("Cognitive Architecture shutdown complete")
            
    except Exception as e:
        logger.error(f"Error during Cognitive Architecture shutdown: {e}")


def integrate_with_main_app(app: FastAPI):
    """
    Integrate cognitive architecture with the main application
    
    Args:
        app: FastAPI application instance
    """
    # Add startup event
    @app.on_event("startup")
    async def cognitive_startup():
        await startup_cognitive_system(app)
    
    # Add shutdown event
    @app.on_event("shutdown")
    async def cognitive_shutdown():
        await shutdown_cognitive_system(app)
    
    logger.info("Cognitive Architecture integration configured")


# Utility functions for other modules to interact with cognitive system

async def process_with_cognition(task: dict, context: Optional[dict] = None) -> dict:
    """
    Process a task through the cognitive system
    
    Args:
        task: Task dictionary with type, goal, etc.
        context: Optional context information
        
    Returns:
        dict: Processing result
    """
    manager = get_integration_manager()
    if not manager:
        return {"error": "Cognitive system not available"}
    
    return await manager.process_with_cognitive_system(task, context)


def get_cognitive_insights() -> dict:
    """
    Get current cognitive system insights
    
    Returns:
        dict: Current state and insights
    """
    manager = get_integration_manager()
    if not manager:
        return {"error": "Cognitive system not available"}
    
    return manager.get_cognitive_state()


async def store_cognitive_memory(content: dict, importance: float = 0.5, 
                               context: Optional[dict] = None) -> Optional[str]:
    """
    Store information in cognitive memory
    
    Args:
        content: Content to store
        importance: Importance level (0-1)
        context: Optional context
        
    Returns:
        str: Memory ID if successful
    """
    manager = get_integration_manager()
    if not manager or not manager.cognitive_system:
        return None
    
    return manager.cognitive_system.episodic_memory.store_episode(
        content,
        context or {},
        [],
        importance
    )


async def allocate_cognitive_attention(task_id: str, agents: list[str], 
                                     priority: float = 0.5) -> bool:
    """
    Allocate cognitive attention to a task
    
    Args:
        task_id: Task identifier
        agents: List of agent IDs
        priority: Priority level (0-1)
        
    Returns:
        bool: True if attention allocated successfully
    """
    manager = get_integration_manager()
    if not manager or not manager.cognitive_system:
        return False
    
    from .unified_cognitive_system import AttentionMode
    focus = manager.cognitive_system.attention.allocate_attention(
        task_id,
        agents,
        priority,
        AttentionMode.FOCUSED
    )
    
    return focus is not None


# Configuration for cognitive system parameters
COGNITIVE_CONFIG = {
    "working_memory_capacity": 7,
    "episodic_memory_max_episodes": 10000,
    "max_concurrent_attention": 3,
    "learning_rate": 0.01,
    "exploration_rate": 0.1,
    "confidence_threshold": 0.7,
    "metacognitive_reflection_interval": 3600,  # 1 hour
    "memory_sync_interval": 300,  # 5 minutes
    "learning_consolidation_interval": 1800  # 30 minutes
}


def configure_cognitive_system(config: dict):
    """
    Update cognitive system configuration
    
    Args:
        config: Configuration dictionary
    """
    global COGNITIVE_CONFIG
    COGNITIVE_CONFIG.update(config)
    logger.info(f"Cognitive system configuration updated: {COGNITIVE_CONFIG}")