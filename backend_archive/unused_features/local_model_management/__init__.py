#!/usr/bin/env python3
"""
SutazAI V7 Local Model Management System
100% offline model management with advanced capabilities
"""

from .model_registry import ModelRegistry, ModelMetadata, ModelVersion
from .model_loader import ModelLoader, ModelConfig, LoadedModel
from .model_optimizer import ModelOptimizer, OptimizationConfig
from .quantization_engine import QuantizationEngine, QuantizationConfig
from .inference_engine import InferenceEngine, InferenceConfig
from .storage_manager import StorageManager, StorageConfig
from .cache_manager import CacheManager, CacheConfig
from .validation_engine import ValidationEngine, ValidationConfig
from .local_model_manager import LocalModelManager, LocalModelConfig

__version__ = "1.0.0"
__all__ = [
    "ModelRegistry",
    "ModelMetadata", 
    "ModelVersion",
    "ModelLoader",
    "ModelConfig",
    "LoadedModel",
    "ModelOptimizer",
    "OptimizationConfig",
    "QuantizationEngine",
    "QuantizationConfig",
    "InferenceEngine",
    "InferenceConfig",
    "StorageManager",
    "StorageConfig",
    "CacheManager",
    "CacheConfig",
    "ValidationEngine",
    "ValidationConfig",
    "LocalModelManager",
    "LocalModelConfig"
]

# Export factory functions for convenience
def create_local_model_manager(config_path: str = None) -> LocalModelManager:
    """Create a new local model manager instance"""
    from .local_model_manager import create_local_model_manager
    return create_local_model_manager(config_path)

def create_model_registry(storage_path: str = None) -> ModelRegistry:
    """Create a new model registry instance"""
    from .model_registry import create_model_registry
    return create_model_registry(storage_path)

def create_inference_engine(model_path: str = None) -> InferenceEngine:
    """Create a new inference engine instance"""
    from .inference_engine import create_inference_engine
    return create_inference_engine(model_path)