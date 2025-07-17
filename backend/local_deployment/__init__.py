#!/usr/bin/env python3
"""
Local Deployment Module

This module provides complete offline model deployment and management capabilities
for 100% autonomous AI operation without external dependencies.
"""

from .offline_model_manager import (
    OfflineModelManager,
    ModelConfig,
    ModelFramework,
    ModelState,
    QuantizationType,
    ModelMetrics,
    ResourceMonitor,
    ModelOptimizationEngine
)

from .local_server import (
    LocalModelServer,
    TextGenerationRequest,
    TextGenerationResponse,
    ModelLoadRequest
)

__all__ = [
    'OfflineModelManager',
    'ModelConfig',
    'ModelFramework',
    'ModelState',
    'QuantizationType',
    'ModelMetrics',
    'ResourceMonitor',
    'ModelOptimizationEngine',
    'LocalModelServer',
    'TextGenerationRequest',
    'TextGenerationResponse',
    'ModelLoadRequest'
]