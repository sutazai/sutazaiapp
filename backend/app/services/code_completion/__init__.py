"""
Code completion service module
"""
from .interfaces import CodeCompletionClient
from .factory import code_completion_factory

__all__ = ["CodeCompletionClient", "code_completion_factory"]