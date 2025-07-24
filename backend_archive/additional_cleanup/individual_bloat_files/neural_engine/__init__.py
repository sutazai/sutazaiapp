#!/usr/bin/env python3
"""
SutazAI Neural Processing Engine
Advanced neural processing with biological modeling and neuromorphic computing
"""

from .neural_processor import NeuralProcessor, NeuralConfig
from .biological_modeling import BiologicalNeuralNetwork, BiologicalConfig
from .neuromorphic_engine import NeuromorphicEngine, NeuromorphicConfig
from .adaptive_learning import AdaptiveLearningSystem, AdaptiveConfig
from .neural_optimizer import NeuralOptimizer, OptimizationConfig
from .synaptic_plasticity import SynapticPlasticityManager, PlasticityConfig
from .neural_memory import NeuralMemorySystem, MemoryConfig

__version__ = "1.0.0"
__all__ = [
    "NeuralProcessor",
    "NeuralConfig",
    "BiologicalNeuralNetwork",
    "BiologicalConfig",
    "NeuromorphicEngine",
    "NeuromorphicConfig",
    "AdaptiveLearningSystem",
    "AdaptiveConfig",
    "NeuralOptimizer",
    "OptimizationConfig",
    "SynapticPlasticityManager",
    "PlasticityConfig",
    "NeuralMemorySystem",
    "MemoryConfig"
]

def create_neural_processor(config: dict = None) -> NeuralProcessor:
    """Factory function to create neural processor"""
    return NeuralProcessor(config=config)

def create_biological_network(config: dict = None) -> BiologicalNeuralNetwork:
    """Factory function to create biological neural network"""
    return BiologicalNeuralNetwork(config=config)

def create_neuromorphic_engine(config: dict = None) -> NeuromorphicEngine:
    """Factory function to create neuromorphic engine"""
    return NeuromorphicEngine(config=config)