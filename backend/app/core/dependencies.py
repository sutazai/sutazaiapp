"""
Dependency injection for FastAPI
"""
from typing import AsyncGenerator
from app.services.consolidated_ollama_service import (
    ConsolidatedOllamaService,
    get_ollama_service,
    get_ollama_embedding_service,
    get_model_manager,
    get_advanced_model_manager
)
from app.services.agent_orchestrator import AgentOrchestrator

# Singleton instances
_agent_orchestrator: AgentOrchestrator = None

# Ollama service dependency functions (all point to consolidated service)
async def get_consolidated_ollama_service() -> ConsolidatedOllamaService:
    """Get the consolidated Ollama service instance"""
    return await get_ollama_service()

# Compatibility functions for existing endpoints
async def get_model_manager() -> ConsolidatedOllamaService:
    """Get consolidated service (compatibility for model management)"""
    return await get_ollama_service()

async def get_advanced_model_manager() -> ConsolidatedOllamaService:
    """Get consolidated service (compatibility for advanced features)"""
    return await get_ollama_service()

async def get_ollama_embedding_service() -> ConsolidatedOllamaService:
    """Get consolidated service (compatibility for embeddings)"""
    return await get_ollama_service()

def get_agent_orchestrator() -> AgentOrchestrator:
    """Get agent orchestrator instance"""
    global _agent_orchestrator
    if _agent_orchestrator is None:
        _agent_orchestrator = AgentOrchestrator()
    return _agent_orchestrator