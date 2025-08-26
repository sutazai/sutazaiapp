"""
Training service module
"""
from .interfaces import Trainer, TrainingConfig, TrainingResult
from .factory import trainer_factory

__all__ = ["Trainer", "TrainingConfig", "TrainingResult", "trainer_factory"]