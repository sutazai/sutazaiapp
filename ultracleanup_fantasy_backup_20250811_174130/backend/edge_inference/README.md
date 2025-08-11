# SutazAI Edge Inference Optimization System

A comprehensive edge inference optimization system designed for high-performance, resource-constrained edge computing environments.

## Overview

This system provides advanced optimization techniques for deploying AI inference at the edge, including intelligent routing, model caching, dynamic quantization, batching, memory management, and comprehensive monitoring.

## Key Features

### 🚀 **Edge Inference Proxy**
- Intelligent request routing and load balancing
- Multiple routing strategies (round-robin, least-loaded, resource-aware, geographic)
- Support for 174+ concurrent connections
- Built for 12-core CPU optimization

### 🧠 **Model Caching and Sharing**
- Intelligent model caching with LRU/LFU eviction policies
- Cross-process model sharing and registry
- Memory mapping for efficient model loading
- Compression-aware caching

### ⚡ **Inference Batching**
- Smart batching with adaptive sizing
- Priority-aware request queuing
- Context-aware batching for similar requests
- Result caching with TTL and LRU eviction

### 🔧 **Model Quantization**
- Support for INT8, INT4, FP16, and mixed precision
- Automated quantization strategy selection
- Edge-optimized model compression
- Accuracy-preserving quantization

### 💾 **Dynamic Memory Management**
- Intelligent model loading and unloading
- Memory pool management
- Predictive model preloading
- Resource-aware scheduling

### 🎯 **Intelligent Routing**
- Multi-objective optimization
- Performance prediction and learning
- Circuit breaker patterns
- Geographic and affinity-based routing

### 📊 **Performance Monitoring**
- Comprehensive telemetry and metrics
- Real-time performance dashboards
- Alerting and anomaly detection
- Historical analytics

### 🛡️ **Failover and Health Checking**  
- Automated failover mechanisms
- Comprehensive health monitoring
- Circuit breaker implementation
- Recovery automation

### 🚀 **Edge Deployment Tools**
- Kubernetes, Docker, and bare-metal deployment
- Automated infrastructure provisioning
- Configuration management
- Deployment validation

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Client Apps   │    │   Load Balancer  │    │  Edge Inference │
│                 │───▶│                  │───▶│     Proxy       │
│   69 AI Agents  │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                       ┌─────────────────────────────────┼─────────────────┐
                       │                                 ▼                 │
                       │         ┌─────────────────────────────────────────┤
                       │         │        Intelligent Router               │
                       │         │   • Multi-objective optimization        │
                       │         │   • Performance prediction             │
                       │         │   • Circuit breaker patterns          │
                       │         └─────────────────────────────────────────┤
                       │                                 │                 │
          ┌────────────▼────────────┐       ┌────────────▼────────────┐    │
          │     Model Cache         │       │    Batch Processor      │    │
          │   • LRU/LFU eviction   │       │  • Smart batching       │    │
          │   • Memory mapping     │       │  • Priority queues      │    │
          │   • Compression        │       │  • Result caching       │    │
          └─────────────────────────┘       └─────────────────────────┘    │
                       │                                 │                 │
          ┌────────────▼────────────┐       ┌────────────▼────────────┐    │
          │   Memory Manager        │       │   Quantization Mgr      │    │
          │  • Dynamic loading      │       │  • INT8/INT4/FP16      │    │
          │  • Predictive preload   │       │  • Adaptive strategies  │    │
          │  • Resource pools       │       │  • Edge optimization    │    │
          └─────────────────────────┘       └─────────────────────────┘    │
                       │                                 │                 │
          ┌────────────▼────────────┐       ┌────────────▼────────────┐    │
          │   Telemetry System      │       │   Failover Manager      │    │
          │  • Real-time metrics    │       │  • Health monitoring    │    │
          │  • Alerting system      │       │  • Auto recovery        │    │
          │  • Performance analytics│       │  • Circuit breakers     │    │
          └─────────────────────────┘       └─────────────────────────┘    │
                                                         │                 │
                                          ┌─────────────▼─────────────────┤
                                          │    Deployment Manager          │
                                          │  • K8s/Docker/Bare-metal      │
                                          │  • Infrastructure automation   │
                                          │  • Configuration management    │
                                          └────────────────────────────────┘
```

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Initialize the system
cd /opt/sutazaiapp/backend
python -c "
import asyncio
from edge_inference import initialize_edge_inference_system
asyncio.run(initialize_edge_inference_system())
"
```

### Basic Usage

```python
import asyncio
from edge_inference import (
    initialize_edge_inference_system,
    get_inference_proxy,
    InferenceRequest
)

async def main():
    # Initialize system
    status = await initialize_edge_inference_system()
    print(f"System initialized: {status}")
    
    # Get proxy
    proxy = get_inference_proxy()
    
    # Create inference request
    request = InferenceRequest(
        request_id="test_001",
        model_name="tinyllama",
        prompt="Hello, how are you?",
        parameters={"temperature": 0.7},
        priority=1,
        timeout=30.0
    )
    
    # Process request
    result = await proxy.process_request(request)
    print(f"Response: {result.response}")
    print(f"Processing time: {result.processing_time}ms")

asyncio.run(main())
```

### REST API

```bash
# Start the API server
python -m edge_inference.api

# Health check
curl http://localhost:8000/health

# Process inference request
curl -X POST http://localhost:8000/inference \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_001",
    "model_name": "tinyllama", 
    "prompt": "Explain quantum computing",
    "parameters": {"temperature": 0.7},
    "priority": 1
  }'

# Get system metrics
curl http://localhost:8000/metrics
```

## Configuration

### System Configuration

```python
config = {
    'proxy': {
        'routing_strategy': 'resource_aware',
        'enable_batching': True,
        'enable_caching': True,
        'max_batch_size': 8,
        'batch_timeout_ms': 100.0
    },
    'model_cache': {
        'max_cache_size_gb': 4.0,
        'eviction_policy': 'hybrid',
        'enable_quantization': True,
        'enable_memory_mapping': True
    },
    'memory_manager': {
        'max_model_memory_gb': 8.0,
        'load_strategy': 'predictive',
        'enable_memory_pools': True
    },
    'router': {
        'routing_objective': 'multi_objective',
        'enable_prediction': True,
        'enable_learning': True
    },
    'telemetry': {
        'enable_database': True,
        'enable_system_monitoring': True,
        'metrics_retention_hours': 24
    },
    'failover': {
        'strategy': 'adaptive',
        'enable_circuit_breaker': True
    }
}

status = await initialize_edge_inference_system(config)
```

### Node Registration

```python
from edge_inference import get_inference_proxy, EdgeNode

proxy = get_inference_proxy()

# Register edge node
node = EdgeNode(
    node_id="edge_node_1",
    endpoint="http://192.168.1.100:8000",
    capabilities={"gpu": False, "cpu_cores": 4},
    cpu_cores=4,
    memory_gb=8.0,
    models_loaded={"tinyllama", "tinyllama"},
    location="datacenter_1",
    max_concurrent=20
)

proxy.register_node(node)
```

## Performance Optimization

### For SutazAI's 69 Agent System

```python
# Optimized configuration for 69 agents
config = {
    'proxy': {
        'routing_strategy': 'resource_aware',
        'max_batch_size': 16,  # Larger batches for high throughput
        'batch_timeout_ms': 50.0,  # Faster batching
        'enable_caching': True
    },
    'model_cache': {
        'max_cache_size_gb': 6.0,  # More cache for 69 agents
        'eviction_policy': 'hybrid',
        'enable_quantization': True
    },
    'memory_manager': {
        'max_model_memory_gb': 10.0,  # Larger memory pool
        'load_strategy': 'adaptive'
    },
    'router': {
        'routing_objective': 'maximize_throughput'  # Optimize for throughput
    }
}
```

### CPU-Only Optimization (12 cores)

```python
# CPU-optimized settings
config = {
    'proxy': {
        'routing_strategy': 'least_loaded',
        'enable_batching': True,
        'max_batch_size': 12,  # Match CPU cores
    },
    'model_cache': {
        'enable_quantization': True,
        'default_quantization': 'int8'  # Reduce memory usage
    },
    'memory_manager': {
        'enable_memory_pools': True,
        'cpu_affinity': True  # Pin to specific cores
    }
}
```

## Deployment Examples

### Kubernetes Deployment

```python
from edge_inference import get_deployment_manager, DeploymentConfig, EdgePlatform

deployment_manager = get_deployment_manager()

config = DeploymentConfig(
    name="sutazai-edge-inference",
    platform=EdgePlatform.KUBERNETES,
    replicas=3,
    resources={
        "image": "sutazai/edge-inference:latest",
        "cpu_request": "1",
        "cpu_limit": "2", 
        "memory_request": "2Gi",
        "memory_limit": "4Gi"
    },
    environment={
        "OLLAMA_HOST": "http://ollama-service:10104",
        "MODEL_CACHE_SIZE": "2GB",
        "ENABLE_BATCHING": "true"
    },
    health_check={
        "http_path": "/health",
        "port": 8000,
        "initial_delay": 30
    },
    networking={
        "external_access": True,
        "hostname": "inference.sutazai.local"
    }
)

job_id = await deployment_manager.deploy(config)
```

### Docker Deployment

```python
config = DeploymentConfig(
    name="sutazai-edge-single",
    platform=EdgePlatform.DOCKER_SWARM,
    resources={
        "image": "sutazai/edge-inference:latest",
        "memory_limit": "4Gi",
        "cpu_limit": "2"
    },
    environment={
        "OLLAMA_HOST": "http://localhost:10104",
        "LIGHTWEIGHT_MODE": "true"
    },
    networking={
        "host_port": 8000
    }
)
```

## Monitoring and Alerting

### Custom Metrics

```python
from edge_inference import get_telemetry_system, Metric, MetricType

telemetry = get_telemetry_system()

# Record custom metric
metric = Metric(
    name="custom_inference_latency",
    value=150.5,
    metric_type=MetricType.HISTOGRAM,
    timestamp=datetime.now(),
    labels={"model": "tinyllama", "node": "edge_1"},
    unit="ms"
)

telemetry.record_custom_metric(metric)
```

### Performance Summary

```python
# Get performance summary
summary = telemetry.get_performance_summary(time_window_minutes=60)

print(f"Total requests: {summary.total_requests}")
print(f"Success rate: {(summary.successful_requests / summary.total_requests) * 100:.1f}%")
print(f"Average latency: {summary.avg_latency_ms:.1f}ms")
print(f"P95 latency: {summary.p95_latency_ms:.1f}ms")
print(f"Throughput: {summary.throughput_rps:.1f} RPS")
```

## Advanced Features

### Model Quantization

```python
from edge_inference import get_quantization_manager, QuantizationConfig, QuantizationType

quant_manager = get_quantization_manager()

# Prepare model for edge deployment
quantized_path = await quant_manager.prepare_model_for_edge(
    model_path="/models/llama-7b.gguf",
    target_device="cpu",
    memory_limit_mb=2048,
    latency_target_ms=500
)

print(f"Quantized model ready: {quantized_path}")
```

### Intelligent Routing

```python
from edge_inference import get_intelligent_router, RoutingRequest

router = get_intelligent_router()

# Create routing request
request = RoutingRequest(
    request_id="route_001",
    model_name="tinyllama",
    priority_level=1,
    estimated_complexity=0.7,
    latency_requirement=200.0,
    client_location="datacenter_1"
)

# Get routing decision
decision = await router.route_request(request)
print(f"Selected node: {decision.selected_node.node_id}")
print(f"Confidence: {decision.confidence_score:.2f}")
print(f"Expected latency: {decision.expected_latency_ms:.1f}ms")
```

### Failover Management

```python
from edge_inference import get_failover_manager

failover = get_failover_manager()

# Get node status
status = failover.get_node_status("edge_node_1")
print(f"Node health: {status['health_status']}")
print(f"Consecutive failures: {status['consecutive_failures']}")

# Force failover if needed
failover.force_failover("edge_node_1", "Manual maintenance")
```

## Integration with SutazAI

### Agent Integration

```python
# Integration with existing SutazAI agent system
from app.services.model_manager import ModelManager
from edge_inference import get_model_cache, get_inference_proxy

async def integrate_with_agents():
    # Get existing model manager
    model_manager = ModelManager()
    
    # Initialize edge inference
    proxy = get_inference_proxy()
    cache = get_model_cache()
    
    # Preload models used by agents
    agent_models = ["tinyllama", "tinyllama", "tinyllama-coder"]
    for model in agent_models:
        await cache.preload_models([model], quantization_level="int8")
    
    # Register local Ollama as edge node
    ollama_node = EdgeNode(
        node_id="local_ollama",
        endpoint="http://localhost:10104",
        capabilities={"cpu_cores": 12, "memory_gb": 32},
        models_loaded=set(agent_models),
        max_concurrent=174  # Support 174+ connections
    )
    proxy.register_node(ollama_node)
```

### Energy Optimization Integration

```python
# Note: Energy optimization module removed (fantasy module - Rule 1 violation)
# from energy.power_optimizer import get_global_optimizer
from edge_inference import get_memory_manager, get_telemetry_system

# Standard resource management without fantasy modules
memory_manager = get_memory_manager()
telemetry = get_telemetry_system()

# Add energy-aware model management
async def energy_aware_inference():
    # Check system load
    system_metrics = telemetry.get_system_metrics()
    cpu_usage = system_metrics.get("cpu_usage", 0)
    
    if cpu_usage > 80:
        # High CPU usage - optimize for energy
        power_optimizer.start_optimization()
        
        # Reduce model cache size temporarily
        current_models = memory_manager.get_loaded_models()
        for model in current_models:
            if model.reference_count == 0:
                await memory_manager.unload_model(model.model_id)
```

## Troubleshooting

### Common Issues

**High Memory Usage**
```python
# Check model cache usage
cache = get_model_cache()
stats = cache.get_stats()
print(f"Cache size: {stats.total_size_mb}MB")
print(f"Hit ratio: {stats.hit_ratio:.2f}")

# Clear cache if needed
await cache.clear_cache()
```

**High Latency**
```python
# Check routing stats
router = get_intelligent_router()
stats = router.get_routing_stats()
print(f"Average decision time: {stats['avg_decision_time_ms']}ms")

# Switch to faster routing strategy
router.routing_strategy = RoutingObjective.MINIMIZE_LATENCY
```

**Node Failures**
```python
# Check node health
failover = get_failover_manager()
node_health = failover.health_checker.get_all_node_health()

for node_id, health in node_health.items():
    if health.status != HealthStatus.HEALTHY:
        print(f"Node {node_id}: {health.status} - {health.failure_reasons}")
```

## Performance Benchmarks

### Expected Performance (SutazAI Configuration)
- **Throughput**: 100-500 requests/second
- **Latency**: P95 < 500ms, P99 < 1000ms  
- **Memory Usage**: 4-8GB for model cache
- **CPU Utilization**: 60-80% under load
- **Cache Hit Rate**: 70-90%
- **Failover Time**: < 30 seconds

### Optimization Tips
1. **Enable batching** for high throughput scenarios
2. **Use quantization** for memory-constrained environments
3. **Implement caching** for frequently used models
4. **Monitor resource usage** and adjust cache sizes
5. **Use predictive loading** for known usage patterns

## API Reference

See `/opt/sutazaiapp/backend/edge_inference/api.py` for the complete REST API implementation.

Key endpoints:
- `POST /inference` - Process single inference request
- `POST /inference/batch` - Process batch requests
- `POST /nodes/register` - Register edge node
- `GET /metrics` - Get system metrics
- `POST /deploy` - Deploy edge system
- `GET /status` - Get system status

## Support

For issues, questions, or contributions, please refer to the SutazAI documentation or contact the development team.

## License

This edge inference optimization system is part of the SutazAI project and follows the same licensing terms.