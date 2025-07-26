"""
Self-Improvement System Wrapper
Temporary wrapper to fix import issues
"""

# Import from the actual location
from .services.self_improvement import SelfImprovementService as SelfImprovementSystem

__all__ = ['SelfImprovementSystem']