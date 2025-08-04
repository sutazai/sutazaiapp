#!/usr/bin/env python3
"""
SutazAI Enhanced Agent System Core Module
Initialization and exports for the enhanced agent system

This module provides the core components for the enhanced SutazAI agent system:
- BaseAgentV2: Enhanced async base agent class
- OllamaConnectionPool: Efficient connection pooling
- CircuitBreaker: Fault tolerance pattern
- RequestQueue: Priority-based request management
- Migration utilities for backward compatibility
"""

# Version information
__version__ = "2.0.0"
__author__ = "SutazAI System"
__description__ = "Enhanced async agent system with Ollama integration"

# Core imports
from .base_agent_v2 import BaseAgentV2, AgentStatus, AgentMetrics, TaskResult
from .ollama_pool import OllamaConnectionPool, create_ollama_pool
from .circuit_breaker import CircuitBreaker, CircuitState, create_ollama_circuit_breaker, create_api_circuit_breaker
from .request_queue import RequestQueue, RequestPriority, create_ollama_queue, create_api_queue
from .migration_helper import LegacyAgentWrapper, create_agent_factory, AgentMigrationValidator
from .ollama_integration import OllamaIntegration, OllamaConfig

# Convenience imports
from .base_agent_v2 import BaseAgent  # Backward compatibility alias

# Export all public classes and functions
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__description__",
    
    # Core classes
    "BaseAgentV2",
    "BaseAgent",  # Backward compatibility
    "OllamaConnectionPool",
    "CircuitBreaker", 
    "RequestQueue",
    "OllamaIntegration",
    "OllamaConfig",
    
    # Enums and data classes
    "AgentStatus",
    "AgentMetrics", 
    "TaskResult",
    "CircuitState",
    "RequestPriority",
    
    # Migration utilities
    "LegacyAgentWrapper",
    "AgentMigrationValidator",
    
    # Factory functions
    "create_agent_factory",
    "create_ollama_pool",
    "create_ollama_circuit_breaker",
    "create_api_circuit_breaker", 
    "create_ollama_queue",
    "create_api_queue",
]

# Module-level configuration
import logging
import os

# Setup logging if not already configured
if not logging.getLogger().handlers:
    logging.basicConfig(
        level=os.getenv('LOG_LEVEL', 'INFO'),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

logger = logging.getLogger(__name__)

# Validate required dependencies
try:
    import httpx
    import asyncio
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    raise

# Module initialization
logger.info(f"SutazAI Enhanced Agent System v{__version__} initialized")

# Validate Ollama connectivity on import (optional)
def validate_ollama_connection():
    """Validate Ollama connectivity (non-blocking)"""
    import asyncio
    
    async def check():
        try:
            async with OllamaConnectionPool(max_connections=1) as pool:
                await pool.health_check()
                logger.info("Ollama connectivity validated")
                return True
        except Exception as e:
            logger.warning(f"Ollama not available: {e}")
            return False
    
    try:
        # Only validate if we're not in an existing event loop
        try:
            loop = asyncio.get_running_loop()
            logger.debug("Skipping Ollama validation (already in event loop)")
        except RuntimeError:
            # No running loop, we can validate
            result = asyncio.run(check())
            return result
    except Exception as e:
        logger.debug(f"Ollama validation skipped: {e}")
        return False

# Run validation if environment variable is set
if os.getenv('VALIDATE_OLLAMA_ON_IMPORT', '').lower() in ('true', '1', 'yes'):
    validate_ollama_connection()

# Provide utility functions at module level
def get_agent_config(agent_name: str):
    """Get configuration for a specific agent"""
    return OllamaConfig.get_model_config(agent_name)

def list_available_models():
    """List available Ollama models"""
    return list(set(OllamaConfig.AGENT_MODELS.values()))

def get_agents_by_model(model: str):
    """Get list of agents using a specific model"""
    return [
        agent for agent, agent_model in OllamaConfig.AGENT_MODELS.items()
        if agent_model == model
    ]

# Module-level statistics
def get_system_stats():
    """Get system-wide statistics"""
    return {
        "version": __version__,
        "total_agent_types": len(OllamaConfig.AGENT_MODELS),
        "opus_agents": len(get_agents_by_model(OllamaConfig.OPUS_MODEL)),
        "sonnet_agents": len(get_agents_by_model(OllamaConfig.SONNET_MODEL)), 
        "default_agents": len(get_agents_by_model(OllamaConfig.DEFAULT_MODEL)),
        "models_used": len(list_available_models())
    }

# Print system stats if requested
if os.getenv('SHOW_SYSTEM_STATS', '').lower() in ('true', '1', 'yes'):
    stats = get_system_stats()
    logger.info(f"System Stats: {stats}")