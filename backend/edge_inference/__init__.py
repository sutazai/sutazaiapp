"""
Edge Inference Optimization System for SutazAI

A comprehensive edge inference optimization system designed for high-performance,
resource-constrained edge computing environments.

Key Features:
- Intelligent routing and load balancing
- Advanced model caching and sharing
- Dynamic quantization and compression
- Inference batching and result caching
- Memory-efficient model loading
- Performance monitoring and telemetry
- Failover mechanisms and health checking
- Automated deployment tools

Architecture Overview:
- EdgeInferenceProxy: Main entry point for inference requests
- ModelCache: Intelligent model caching and sharing
- BatchProcessor: Advanced batching for throughput optimization
- QuantizationManager: Model compression for edge deployment
- MemoryManager: Dynamic memory management
- IntelligentRouter: Multi-objective routing optimization
- TelemetrySystem: Comprehensive monitoring and alerting
- FailoverManager: High availability and fault tolerance
- DeploymentManager: Automated deployment and orchestration
"""

from .proxy import EdgeInferenceProxy, RoutingStrategy, EdgeNode, InferenceRequest, InferenceResult
from .model_cache import EdgeModelCache, ModelFormat, CacheEvictionPolicy, CachedModel, SharedModelRegistry
from .batch_processor import SmartBatchProcessor, BatchStrategy, BatchRequest, BatchResult, RequestPriority
from .quantization import EdgeQuantizationManager, QuantizationType, QuantizationStrategy, QuantizationConfig
from .memory_manager import ModelMemoryManager, LoadStrategy, MemoryPool, ModelLoadInfo
from .intelligent_router import IntelligentRouter, RoutingObjective, RoutingNode, RoutingRequest, RoutingDecision
from .telemetry import EdgeTelemetrySystem, MetricType, AlertSeverity, Metric, Alert, PerformanceSummary
from .failover import FailoverManager, HealthChecker, FailoverStrategy, HealthStatus, NodeHealth
from .deployment import EdgeDeploymentManager, DeploymentConfig, EdgePlatform, DeploymentStatus

# Global instances
_proxy = None
_model_cache = None
_batch_processor = None
_quantization_manager = None
_memory_manager = None
_router = None
_telemetry = None
_failover_manager = None
_deployment_manager = None

def get_inference_proxy(**kwargs) -> EdgeInferenceProxy:
    """Get or create global inference proxy instance"""
    global _proxy
    if _proxy is None:
        _proxy = EdgeInferenceProxy(**kwargs)
    return _proxy

def get_model_cache(**kwargs) -> EdgeModelCache:
    """Get or create global model cache instance"""
    global _model_cache
    if _model_cache is None:
        _model_cache = EdgeModelCache(**kwargs)
    return _model_cache

def get_batch_processor(**kwargs) -> SmartBatchProcessor:
    """Get or create global batch processor instance"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = SmartBatchProcessor(**kwargs)
    return _batch_processor

def get_quantization_manager(**kwargs) -> EdgeQuantizationManager:
    """Get or create global quantization manager instance"""
    global _quantization_manager
    if _quantization_manager is None:
        _quantization_manager = EdgeQuantizationManager(**kwargs)
    return _quantization_manager

def get_memory_manager(**kwargs) -> ModelMemoryManager:
    """Get or create global memory manager instance"""
    global _memory_manager
    if _memory_manager is None:
        _memory_manager = ModelMemoryManager(**kwargs)
    return _memory_manager

def get_intelligent_router(**kwargs) -> IntelligentRouter:
    """Get or create global intelligent router instance"""
    global _router
    if _router is None:
        _router = IntelligentRouter(**kwargs)
    return _router

def get_telemetry_system(**kwargs) -> EdgeTelemetrySystem:
    """Get or create global telemetry system instance"""
    global _telemetry
    if _telemetry is None:
        _telemetry = EdgeTelemetrySystem(**kwargs)
    return _telemetry

def get_failover_manager(**kwargs) -> FailoverManager:
    """Get or create global failover manager instance"""
    global _failover_manager
    if _failover_manager is None:
        _failover_manager = FailoverManager(**kwargs)
    return _failover_manager

def get_deployment_manager() -> EdgeDeploymentManager:
    """Get or create global deployment manager instance"""
    global _deployment_manager
    if _deployment_manager is None:
        _deployment_manager = EdgeDeploymentManager()
    return _deployment_manager

async def initialize_edge_inference_system(config: dict = None) -> dict:
    """
    Initialize the complete edge inference system
    
    Args:
        config: Configuration dictionary with component settings
        
    Returns:
        Dictionary with initialized components and their status
    """
    config = config or {}
    
    # Initialize components
    proxy = get_inference_proxy(**config.get('proxy', {}))
    model_cache = get_model_cache(**config.get('model_cache', {}))
    batch_processor = get_batch_processor(**config.get('batch_processor', {}))
    quantization_manager = get_quantization_manager(**config.get('quantization', {}))
    memory_manager = get_memory_manager(**config.get('memory_manager', {}))
    router = get_intelligent_router(**config.get('router', {}))
    telemetry = get_telemetry_system(**config.get('telemetry', {}))
    failover_manager = get_failover_manager(**config.get('failover', {}))
    deployment_manager = get_deployment_manager()
    
    # Start services
    status = {}
    
    try:
        await memory_manager.start()
        status['memory_manager'] = 'started'
    except Exception as e:
        status['memory_manager'] = f'failed: {e}'
    
    try:
        await model_cache.start()
        status['model_cache'] = 'started'
    except Exception as e:
        status['model_cache'] = f'failed: {e}'
    
    try:
        await batch_processor.start()
        status['batch_processor'] = 'started'
    except Exception as e:
        status['batch_processor'] = f'failed: {e}'
    
    try:
        await router.start()
        status['router'] = 'started'
    except Exception as e:
        status['router'] = f'failed: {e}'
    
    try:
        await telemetry.start()
        status['telemetry'] = 'started'
    except Exception as e:
        status['telemetry'] = f'failed: {e}'
    
    try:
        await failover_manager.start()
        status['failover_manager'] = 'started'
    except Exception as e:
        status['failover_manager'] = f'failed: {e}'
    
    try:
        await proxy.start()
        status['proxy'] = 'started'
    except Exception as e:
        status['proxy'] = f'failed: {e}'
    
    status['deployment_manager'] = 'initialized'
    status['quantization_manager'] = 'initialized'
    
    return status

async def shutdown_edge_inference_system() -> dict:
    """
    Shutdown the complete edge inference system
    
    Returns:
        Dictionary with shutdown status for each component
    """
    global _proxy, _model_cache, _batch_processor, _memory_manager
    global _router, _telemetry, _failover_manager
    
    status = {}
    
    # Shutdown in reverse order
    if _proxy:
        try:
            await _proxy.stop()
            status['proxy'] = 'stopped'
        except Exception as e:
            status['proxy'] = f'error: {e}'
    
    if _failover_manager:
        try:
            await _failover_manager.stop()
            status['failover_manager'] = 'stopped'
        except Exception as e:
            status['failover_manager'] = f'error: {e}'
    
    if _telemetry:
        try:
            await _telemetry.stop()
            status['telemetry'] = 'stopped'
        except Exception as e:
            status['telemetry'] = f'error: {e}'
    
    if _router:
        try:
            await _router.stop()
            status['router'] = 'stopped'
        except Exception as e:
            status['router'] = f'error: {e}'
    
    if _batch_processor:
        try:
            await _batch_processor.stop()
            status['batch_processor'] = 'stopped'
        except Exception as e:
            status['batch_processor'] = f'error: {e}'
    
    if _model_cache:
        try:
            await _model_cache.stop()
            status['model_cache'] = 'stopped'
        except Exception as e:
            status['model_cache'] = f'error: {e}'
    
    if _memory_manager:
        try:
            await _memory_manager.stop()
            status['memory_manager'] = 'stopped'
        except Exception as e:
            status['memory_manager'] = f'error: {e}'
    
    return status

def get_system_status() -> dict:
    """
    Get status of all system components
    
    Returns:
        Dictionary with status information for each component
    """
    status = {
        'proxy': 'not_initialized',
        'model_cache': 'not_initialized',
        'batch_processor': 'not_initialized',
        'quantization_manager': 'not_initialized',
        'memory_manager': 'not_initialized',
        'router': 'not_initialized',
        'telemetry': 'not_initialized',
        'failover_manager': 'not_initialized',
        'deployment_manager': 'not_initialized'
    }
    
    if _proxy:
        status['proxy'] = 'initialized'
    if _model_cache:
        status['model_cache'] = 'initialized'
    if _batch_processor:
        status['batch_processor'] = 'initialized'
    if _quantization_manager:
        status['quantization_manager'] = 'initialized'
    if _memory_manager:
        status['memory_manager'] = 'initialized'
    if _router:
        status['router'] = 'initialized'
    if _telemetry:
        status['telemetry'] = 'initialized'
    if _failover_manager:
        status['failover_manager'] = 'initialized'
    if _deployment_manager:
        status['deployment_manager'] = 'initialized'
    
    return status

__version__ = "1.0.0"
__author__ = "SutazAI Edge Inference Team"
__description__ = "Advanced edge inference optimization system"

__all__ = [
    # Main classes
    'EdgeInferenceProxy',
    'EdgeModelCache', 
    'SmartBatchProcessor',
    'EdgeQuantizationManager',
    'ModelMemoryManager',
    'IntelligentRouter',
    'EdgeTelemetrySystem',
    'FailoverManager',
    'EdgeDeploymentManager',
    
    # Enums
    'RoutingStrategy',
    'ModelFormat',
    'CacheEvictionPolicy',
    'BatchStrategy',
    'RequestPriority',
    'QuantizationType',
    'QuantizationStrategy',
    'LoadStrategy',
    'MemoryPool',
    'RoutingObjective',
    'MetricType',
    'AlertSeverity',
    'HealthStatus',
    'FailoverStrategy',
    'EdgePlatform',
    'DeploymentStatus',
    
    # Data classes
    'EdgeNode',
    'InferenceRequest',
    'InferenceResult',
    'CachedModel',
    'BatchRequest',
    'BatchResult',
    'QuantizationConfig',
    'ModelLoadInfo',
    'RoutingNode',
    'RoutingRequest',
    'RoutingDecision',
    'Metric',
    'Alert',
    'PerformanceSummary',
    'NodeHealth',
    'DeploymentConfig',
    
    # Utility classes
    'SharedModelRegistry',
    
    # Global functions
    'get_inference_proxy',
    'get_model_cache',
    'get_batch_processor',
    'get_quantization_manager',
    'get_memory_manager',
    'get_intelligent_router',
    'get_telemetry_system',
    'get_failover_manager',
    'get_deployment_manager',
    'initialize_edge_inference_system',
    'shutdown_edge_inference_system',
    'get_system_status'
]