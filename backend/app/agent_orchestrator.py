"""
Compatibility shim for Agent Orchestrator.
Source of truth lives at app.services.agent_orchestrator.
This module re-exports the canonical implementation to preserve imports.
"""
from app.services.agent_orchestrator import AgentOrchestrator  # noqa: F401