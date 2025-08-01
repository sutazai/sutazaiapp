"""
Dependency injection for FastAPI
"""
from typing import AsyncGenerator
from app.services.model_manager import ModelManager
from app.services.agent_orchestrator import AgentOrchestrator

# Singleton instances
_model_manager: ModelManager = None
_agent_orchestrator: AgentOrchestrator = None

async def get_model_manager() -> ModelManager:
    """Get model manager instance"""
    global _model_manager
    if _model_manager is None:
        _model_manager = ModelManager()
        await _model_manager.initialize()
    return _model_manager

def get_agent_orchestrator() -> AgentOrchestrator:
    """Get agent orchestrator instance"""
    global _agent_orchestrator
    if _agent_orchestrator is None:
        _agent_orchestrator = AgentOrchestrator()
    return _agent_orchestrator