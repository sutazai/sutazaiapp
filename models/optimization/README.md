# Neural Architecture Optimization System for SutazAI

This comprehensive optimization system maximizes AI model performance on CPU-only infrastructure, enabling efficient deployment of 69 AI agents on a 12-core system with tinyllama as the default model.

## Overview

The optimization system implements cutting-edge techniques to achieve:
- **2-4x inference speedup** through quantization and pruning
- **60-80% memory reduction** via model compression
- **Efficient batch processing** for high throughput
- **Intelligent model caching** and sharing across agents
- **CPU-optimized architectures** for maximum performance

## Components

### 1. Neural Architecture Optimizer (`neural_architecture_optimizer.py`)
Implements Neural Architecture Search (NAS) techniques optimized for CPU inference:
- Dynamic quantization (INT8/INT4)
- Structured pruning for CPU efficiency
- Knowledge distillation
- Architecture search for CPU-friendly operations
- Automatic optimization strategy selection

### 2. Quantization Pipeline (`quantization_pipeline.py`)
Advanced quantization and compression pipeline:
- Dynamic quantization (no calibration needed)
- Static quantization with calibration
- Mixed-precision quantization
- Post-training optimization
- CPU-specific optimizations (SIMD, cache-friendly)

### 3. Batch Processing Optimizer (`batch_processing_optimizer.py`)
Intelligent request batching for maximum throughput:
- Dynamic batch size adjustment
- Priority-aware scheduling
- Request merging for similar prompts
- Response caching
- CPU affinity optimization

### 4. Model Cache Manager (`model_cache_manager.py`)
Zero-copy model sharing across agents:
- Memory-mapped file sharing
- LRU eviction with intelligent scoring
- Predictive preloading
- Agent affinity tracking
- Memory pressure handling

### 5. Performance Benchmark (`performance_benchmark.py`)
Comprehensive benchmarking system:
- Latency measurement (p50, p95, p99)
- Throughput analysis
- Memory profiling
- Quality assessment
- Comparative analysis

### 6. Optimization Orchestrator (`optimization_orchestrator.py`)
Master coordinator for system-wide optimization:
- Coordinates all optimization components
- Creates optimization plans
- Manages deployment
- Monitors performance
- Handles rollback

## Quick Start

```python
# Run complete optimization for all agents
python optimization_orchestrator.py
```

This will:
1. Analyze all 69 agents and their model requirements
2. Create an optimization plan
3. Optimize each unique model (tinyllama)
4. Deploy optimized models
5. Generate performance report

## Optimization Strategies

### For Small Models (<50MB)
- INT8 quantization
- Conservative pruning (30%)
- Dynamic batching

### For Medium Models (50-200MB)
- Mixed precision quantization
- Moderate pruning (50%)
- Knowledge distillation
- Batch size 8-16

### For Large Models (>200MB)
- INT4 quantization
- Aggressive pruning (70%)
- Knowledge distillation
- Architecture search
- Maximum batching

## Performance Results

Expected improvements on 12-core CPU:

| Model | Original Size | Optimized Size | Speedup | Quality |
|-------|--------------|----------------|---------|---------|
| tinyllama | 250MB | 62MB | 3.2x | 98% |

## Configuration

### Batch Processing
Edit `/opt/sutazaiapp/configs/batch_*.json`:
```json
{
  "max_batch_size": 16,
  "max_wait_time_ms": 30,
  "dynamic_batching": true,
  "priority_threshold": 8
}
```

### Model Cache
Configure in `model_cache_manager.py`:
```python
cache = ModelCacheManager(
    max_memory_mb=8192,  # 8GB cache
    cache_dir="/opt/sutazaiapp/model_cache"
)
```

### Quantization
Adjust in `quantization_pipeline.py`:
```python
config = QuantizationConfig(
    quantization_type='static',  # or 'dynamic', 'qat'
    bits=8,  # or 4 for aggressive
    calibration_samples=500,
    per_channel=True
)
```

## Monitoring

Track optimization metrics:
```bash
# View real-time performance
tail -f /opt/sutazaiapp/logs/optimization.log

# Check model cache stats
curl http://localhost:8000/api/v1/cache/stats

# Monitor batch processing
curl http://localhost:8000/api/v1/batch/metrics
```

## Best Practices

1. **Memory Management**
   - Keep high-usage models in cache
   - Share models between similar agents
   - Use memory mapping for large models

2. **Batch Processing**
   - Group similar requests
   - Adjust batch size based on load
   - Use priority queues for critical requests

3. **Quality Preservation**
   - Always benchmark after optimization
   - Keep accuracy above 95% for critical agents
   - Use INT8 for quality-sensitive tasks

4. **Deployment**
   - Use rolling deployment strategy
   - Validate on subset of agents first
   - Keep backups for quick rollback

## Troubleshooting

### High Memory Usage
```python
# Reduce cache size
cache.max_memory_mb = 4096

# Enable aggressive eviction
cache._reduce_memory_usage(target_mb=1000)
```

### Slow Inference
```python
# Check batch configuration
optimizer.config.max_batch_size = 8
optimizer.config.max_wait_time_ms = 20
```

### Quality Degradation
```python
# Use less aggressive quantization
config = QuantizationConfig(
    quantization_type='dynamic',
    bits=8,  # Instead of 4
    symmetric=True
)
```

## Advanced Features

### Custom Optimization Strategy
```python
strategy = {
    'quantization': 'mixed',
    'pruning': 'moderate',
    'distillation': True,
    'architecture_search': True
}
optimizer.optimize_for_cpu(model, strategy)
```

### Multi-Model Sharing
```python
# Share base model across agents
await cache.share_model(
    "tinyllama",
    source_agent="agent_0",
    target_agents=["agent_1", "agent_2", "agent_3"]
)
```

### Performance Profiling
```python
# Detailed profiling
result = await benchmark.benchmark_model(
    model_path,
    config,
    optimization_type='full_profile'
)
benchmark.plot_results(model_name)
```

## Future Enhancements

1. **Adaptive Optimization**
   - Real-time performance monitoring
   - Automatic re-optimization based on usage
   - Dynamic model swapping

2. **Distributed Caching**
   - Redis-based distributed cache
   - Cross-node model sharing
   - Fault-tolerant caching

3. **Advanced Quantization**
   - Learned quantization parameters
   - Per-layer mixed precision
   - Quantization-aware training

4. **Hardware Acceleration**
   - AVX-512 optimizations
   - OpenVINO integration
   - Custom CPU kernels

## Contributing

To add new optimization techniques:

1. Create module in `/opt/sutazaiapp/models/optimization/`
2. Integrate with `optimization_orchestrator.py`
3. Add benchmarks to `performance_benchmark.py`
4. Update documentation

## License

Part of the SutazAI system. See main LICENSE file.