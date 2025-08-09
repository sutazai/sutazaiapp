#!/usr/bin/env python3
"""
SutazAI Enhanced Agent System Core Module
Initialization and exports for the enhanced agent system

This module provides the core components for the enhanced SutazAI agent system:
- BaseAgent: Enhanced async base agent class
- OllamaConnectionPool: Efficient connection pooling
- CircuitBreaker: Fault tolerance pattern
- RequestQueue: Priority-based request management
- Migration utilities for backward compatibility
"""

# Version information
__version__ = "2.0.0"
__author__ = "SutazAI System"
__description__ = "Enhanced async agent system with Ollama integration"

# Core imports - using the consolidated base agent
from .base_agent import BaseAgent, AgentStatus, AgentMetrics, TaskResult, AgentCapability, AgentMessage, AgentConfig

# Enhanced components (optional)
try:
    from .ollama_pool import OllamaConnectionPool, create_ollama_pool
    from .circuit_breaker import CircuitBreaker, CircuitState, create_ollama_circuit_breaker, create_api_circuit_breaker
    from .request_queue import RequestQueue, RequestPriority, create_ollama_queue, create_api_queue
    from .migration_helper import LegacyAgentWrapper, create_agent_factory, AgentMigrationValidator
    from .ollama_integration import OllamaIntegration, OllamaConfig
    ENHANCED_COMPONENTS_AVAILABLE = True
except ImportError:
    # Enhanced components not available, provide fallbacks
    OllamaConnectionPool = None
    CircuitBreaker = None
    RequestQueue = None
    OllamaIntegration = None
    OllamaConfig = None
    ENHANCED_COMPONENTS_AVAILABLE = False

# Backward compatibility aliases
BaseAgentV2 = BaseAgent

# Export all public classes and functions
__all__ = [
    # Version info
    "__version__",
    "__author__", 
    "__description__",
    
    # Core classes
    "BaseAgent",
    "BaseAgentV2",  # Backward compatibility alias
    
    # Core enums and data classes
    "AgentStatus",
    "AgentMetrics", 
    "TaskResult",
    "AgentCapability",
    "AgentMessage",
    "AgentConfig",
]

# Add enhanced components to exports if available
if ENHANCED_COMPONENTS_AVAILABLE:
    __all__.extend([
        "OllamaConnectionPool",
        "CircuitBreaker", 
        "RequestQueue",
        "OllamaIntegration",
        "OllamaConfig",
        "CircuitState",
        "RequestPriority",
        "LegacyAgentWrapper",
        "AgentMigrationValidator",
        "create_agent_factory",
        "create_ollama_pool",
        "create_ollama_circuit_breaker",
        "create_api_circuit_breaker", 
        "create_ollama_queue",
        "create_api_queue",
    ])

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

# Validate required dependencies for basic functionality
try:
    import asyncio
    logger.debug("Basic async support available")
except ImportError as e:
    logger.error(f"Missing required dependency: {e}")
    raise

# Check for optional dependencies
try:
    import httpx
    logger.debug("HTTP client support available")
except ImportError:
    logger.warning("httpx not available - some HTTP functionality may be limited")

# Module initialization
logger.info(f"SutazAI Enhanced Agent System v{__version__} initialized")

# Validate Ollama connectivity on import (optional)
def validate_ollama_connection():
    """Validate Ollama connectivity (non-blocking)"""
    if not ENHANCED_COMPONENTS_AVAILABLE or not OllamaConnectionPool:
        logger.debug("Enhanced components not available, skipping Ollama validation")
        return False
        
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
    if ENHANCED_COMPONENTS_AVAILABLE and OllamaConfig:
        return OllamaConfig.get_model_config(agent_name)
    else:
        # Basic fallback configuration
        return {"model": "tinyllama", "temperature": 0.7}

def list_available_models():
    """List available Ollama models"""
    if ENHANCED_COMPONENTS_AVAILABLE and OllamaConfig:
        return list(set(OllamaConfig.AGENT_MODELS.values()))
    else:
        return ["tinyllama"]

def get_agents_by_model(model: str):
    """Get list of agents using a specific model"""
    if ENHANCED_COMPONENTS_AVAILABLE and OllamaConfig:
        return [
            agent for agent, agent_model in OllamaConfig.AGENT_MODELS.items()
            if agent_model == model
        ]
    else:
        return []

# Module-level statistics
def get_system_stats():
    """Get system-wide statistics"""
    if ENHANCED_COMPONENTS_AVAILABLE and OllamaConfig:
        return {
            "version": __version__,
            "enhanced_mode": True,
            "total_agent_types": len(OllamaConfig.AGENT_MODELS),
            "opus_agents": len(get_agents_by_model(OllamaConfig.OPUS_MODEL)),
            "sonnet_agents": len(get_agents_by_model(OllamaConfig.SONNET_MODEL)), 
            "default_agents": len(get_agents_by_model(OllamaConfig.DEFAULT_MODEL)),
            "models_used": len(list_available_models())
        }
    else:
        return {
            "version": __version__,
            "enhanced_mode": False,
            "basic_mode": True,
            "default_model": "tinyllama"
        }

# Print system stats if requested
if os.getenv('SHOW_SYSTEM_STATS', '').lower() in ('true', '1', 'yes'):
    stats = get_system_stats()
    logger.info(f"System Stats: {stats}")