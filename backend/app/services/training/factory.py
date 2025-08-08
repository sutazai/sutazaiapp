"""
Factory for creating trainers based on configuration
"""
import logging
from typing import Optional
from backend.app.core.config import Settings
from .interfaces import Trainer
from .default_trainer import DefaultTrainer
from .fsdp_trainer import FsdpTrainer

logger = logging.getLogger(__name__)

_trainer: Optional[Trainer] = None

def trainer_factory(settings: Settings) -> Trainer:
    """
    Factory function to create appropriate trainer
    
    Args:
        settings: Application settings
        
    Returns:
        Trainer implementation based on configuration
    """
    global _trainer
    
    # Return cached trainer if available
    if _trainer is not None:
        return _trainer
    
    # Create appropriate trainer based on feature flag
    if settings.ENABLE_FSDP:
        logger.info("FSDP training enabled")
        _trainer = FsdpTrainer()
    else:
        logger.info("Using default trainer (FSDP disabled)")
        _trainer = DefaultTrainer()
    
    return _trainer

def reset_trainer():
    """
    Reset the cached trainer (useful for testing)
    """
    global _trainer
    _trainer = None