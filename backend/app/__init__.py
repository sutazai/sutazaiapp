"""
SutazAI AGI/ASI System - Backend Application Package
"""

from .agi_brain import AGIBrain
from .agent_orchestrator import AgentOrchestrator
from .knowledge_manager import KnowledgeManager
from .self_improvement import SelfImprovementSystem
from .reasoning_engine import ReasoningEngine

__all__ = [
    "AGIBrain",
    "AgentOrchestrator", 
    "KnowledgeManager",
    "SelfImprovementSystem",
    "ReasoningEngine"
]

__version__ = "3.0.0" 