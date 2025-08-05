# SutazAI Federated Learning System

A comprehensive privacy-preserving federated learning implementation for distributed AI training across SutazAI's 69+ agent network. Optimized for CPU-only environments with 12-core constraint.

## 🌟 Features

### Core Capabilities
- **Multi-Algorithm Support**: FedAvg, FedProx, FedOpt with asynchronous training
- **Privacy-Preserving**: Differential privacy, secure aggregation, homomorphic encryption
- **CPU-Optimized**: Efficient computation for 12-core constraint environments
- **Model Versioning**: Automated checkpointing, rollback, and performance tracking
- **Real-time Monitoring**: Comprehensive metrics, alerts, and performance analytics
- **Web Dashboard**: Interactive interface for training management and monitoring

### Privacy & Security
- **Differential Privacy**: Gaussian and Laplace mechanisms with budget management
- **Secure Aggregation**: Multi-party computation with Byzantine fault tolerance
- **Model Compression**: Quantization and sparsification for bandwidth efficiency
- **Privacy Budget Tracking**: Automated ε-δ privacy accounting

### Monitoring & Analytics
- **Performance Tracking**: Real-time metrics collection and analysis
- **Anomaly Detection**: Statistical outlier detection with configurable thresholds
- **Client Performance**: Reliability scoring and contribution analysis
- **System Health**: Resource utilization and component health monitoring

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SutazAI Federated Learning                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ Coordinator  │  │   Monitor    │  │ Version Mgr  │          │
│  │              │  │              │  │              │          │
│  │ • Training   │  │ • Metrics    │  │ • Checkpoints│          │
│  │ • Scheduling │  │ • Alerts     │  │ • Rollback   │          │
│  │ • Aggregation│  │ • Analytics  │  │ • History    │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│           │                  │                  │               │
│  ┌────────┼──────────────────┼──────────────────┼──────────┐    │
│  │        │                  │                  │          │    │
│  │  ┌─────▼────┐  ┌─────────▼────┐  ┌──────────▼───┐      │    │
│  │  │Privacy   │  │ Aggregator   │  │  Dashboard   │      │    │
│  │  │Manager   │  │              │  │              │      │    │
│  │  │          │  │ • FedAvg     │  │ • Web UI     │      │    │
│  │  │• Diff Priv│  │ • FedProx    │  │ • Real-time  │      │    │
│  │  │• Sec Agg │  │ • FedOpt     │  │ • Control    │      │    │
│  │  └──────────┘  └──────────────┘  └──────────────┘      │    │
│  │                                                        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                  │                              │
│  ┌───────────────────────────────▼──────────────────────────┐  │
│  │                    Redis Message Bus                     │  │
│  └───────────────────────────────┬──────────────────────────┘  │
│                                  │                              │
│  ┌───────────────────────────────▼──────────────────────────┐  │
│  │              Federated Learning Clients                  │  │
│  │                                                          │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  │
│  │  │ Agent 1  │ │ Agent 2  │ │   ...    │ │Agent 69+ │    │  │
│  │  │          │ │          │ │          │ │          │    │  │
│  │  │• Local   │ │• Local   │ │• Local   │ │• Local   │    │  │
│  │  │  Training│ │  Training│ │  Training│ │  Training│    │  │
│  │  │• Privacy │ │• Privacy │ │• Privacy │ │• Privacy │    │  │
│  │  │• Updates │ │• Updates │ │• Updates │ │• Updates │    │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- Redis Server
- 12+ CPU cores
- 8GB+ RAM
- SutazAI agent infrastructure

### Installation

1. **Clone or navigate to the SutazAI backend**:
   ```bash
   cd /opt/sutazaiapp/backend
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   pip install fastapi uvicorn websockets chart.js
   ```

3. **Start Redis server**:
   ```bash
   redis-server
   ```

4. **Deploy federated learning system**:
   ```bash
   python scripts/deploy-federated-learning.py
   ```

5. **Access the dashboard**:
   Open http://localhost:8080 in your browser

### Configuration

Edit `/opt/sutazaiapp/backend/federated_learning/config.json`:

```json
{
  "redis_url": "redis://localhost:6379",
  "cpu_cores": 12,
  "max_concurrent_trainings": 3,
  "default_privacy_level": "medium",
  "dashboard_port": 8080,
  "min_clients_per_round": 3,
  "max_clients_per_round": 20
}
```

## 📊 Usage Examples

### Starting a Training Session

```python
from federated_learning.coordinator import FederatedCoordinator, TrainingConfiguration
from federated_learning.aggregator import AggregationAlgorithm
from federated_learning.privacy import PrivacyLevel, PrivacyBudget

# Initialize coordinator
coordinator = FederatedCoordinator()
await coordinator.initialize()

# Configure training
config = TrainingConfiguration(
    name="mnist_classification",
    algorithm=AggregationAlgorithm.FEDAVG,
    model_type="neural_network",
    target_accuracy=0.95,
    max_rounds=100,
    min_clients_per_round=5,
    max_clients_per_round=15,
    local_epochs=5,
    privacy_budget=PrivacyBudget(
        total_epsilon=1.0,
        total_delta=1e-5
    )
)

# Start training
training_id = await coordinator.start_training(config)
print(f"Training started: {training_id}")
```

### Monitoring Training Progress

```python
from federated_learning.monitoring import FederatedMonitor, MetricType, TrainingMetric

# Initialize monitor
monitor = FederatedMonitor()
await monitor.initialize()

# Record metrics
metric = TrainingMetric(
    training_id=training_id,
    round_number=1,
    metric_type=MetricType.ACCURACY,
    value=0.87,
    timestamp=datetime.utcnow()
)

await monitor.record_training_metric(metric)

# Get system health
health = monitor.get_system_health()
print(f"System health: {health['health_status']}")
```

### Client-Side Training

```python
from federated_learning.client import FederatedClient
import numpy as np

# Initialize FL client
client = FederatedClient(agent_id="agent_001")
await client.initialize()

# Add training data
x_train = np.random.randn(1000, 784)
y_train = np.random.randint(0, 10, 1000)
client.add_dataset("local_data", x_train, y_train)

# Client will automatically participate in federated training
```

## 🔧 Advanced Configuration

### Privacy Settings

```python
from federated_learning.privacy import PrivacyManager, DifferentialPrivacyConfig

# Configure differential privacy
privacy_config = DifferentialPrivacyConfig(
    mechanism=PrivacyMechanism.GAUSSIAN_DP,
    epsilon=0.1,  # Strong privacy
    delta=1e-6,
    clipping_norm=1.0,
    noise_multiplier=1.1
)

# Apply to training data
privacy_manager = PrivacyManager()
private_data = await privacy_manager.apply_privacy(x_train, y_train, privacy_config)
```

### Model Versioning

```python
from federated_learning.versioning import ModelVersionManager, RollbackConfig

# Setup automatic rollback
rollback_config = RollbackConfig(
    enabled=True,
    performance_threshold=0.05,  # 5% degradation triggers rollback
    consecutive_degradations=3,
    rollback_to_best=True
)

version_manager = ModelVersionManager()
await version_manager.initialize()
version_manager.setup_rollback_config(training_id, rollback_config)
```

### Custom Aggregation

```python
from federated_learning.aggregator import FederatedAggregator, AggregationConfig

# Configure Byzantine-robust aggregation
agg_config = AggregationConfig(
    algorithm=AggregationAlgorithm.KRUM,
    byzantine_tolerance=True,
    compression=CompressionType.QUANTIZATION,
    compression_ratio=0.1
)

aggregator = FederatedAggregator(cpu_cores=12)
result = await aggregator.aggregate(
    algorithm=AggregationAlgorithm.KRUM,
    client_updates=client_updates,
    round_number=round_num,
    config=agg_config
)
```

## 📈 Performance Optimization

### CPU Optimization
- **Parallelized Aggregation**: Utilizes all 12 CPU cores for model aggregation
- **Efficient Serialization**: Compressed model updates reduce memory usage
- **Asynchronous Processing**: Non-blocking coordination and communication
- **Resource-Aware Scheduling**: Adaptive client selection based on resource availability

### Memory Management
- **Model Compression**: Quantization and sparsification reduce memory footprint
- **Streaming Updates**: Process model updates without full materialization
- **Garbage Collection**: Automated cleanup of expired training sessions
- **Memory Pooling**: Reuse buffers for frequent operations

### Network Optimization
- **Update Compression**: Reduce communication overhead by 60-90%
- **Differential Updates**: Send only parameter changes, not full models
- **Adaptive Batching**: Group updates to minimize round-trip time
- **Connection Pooling**: Reuse connections for repeated communications

## 🔒 Security & Privacy

### Differential Privacy
- **Gaussian Mechanism**: ε-δ differential privacy with configurable parameters
- **Laplace Mechanism**: Pure ε-differential privacy for simpler scenarios
- **Privacy Budget Tracking**: Automated accounting prevents budget exhaustion
- **Adaptive Noise**: Dynamic noise scaling based on sensitivity analysis

### Secure Aggregation
- **Multi-Party Computation**: Cryptographically secure parameter aggregation
- **Byzantine Fault Tolerance**: Robust against up to 25% malicious clients
- **Homomorphic Encryption**: Optional encryption for sensitive model updates
- **Key Management**: Automated key generation and distribution

### Model Protection
- **Gradient Clipping**: Prevent gradient explosion and improve privacy
- **Model Compression**: Reduce information leakage through sparsification
- **Selective Sharing**: Control which model components are shared
- **Audit Logging**: Complete audit trail of all model access and modifications

## 📊 Monitoring & Analytics

### Real-Time Metrics
- **Training Progress**: Accuracy, loss, convergence status per round
- **Client Performance**: Reliability, contribution scores, resource usage
- **System Health**: CPU, memory, network utilization across components
- **Privacy Consumption**: ε-δ budget tracking with alerts

### Alerting System
- **Performance Degradation**: Automatic alerts for accuracy drops
- **Client Failures**: Notification when clients become unreliable
- **Resource Exhaustion**: Warnings for high CPU/memory usage
- **Privacy Violations**: Alerts when privacy budget is nearly exhausted

### Analytics Dashboard
- **Interactive Charts**: Real-time visualization of training metrics
- **Client Leaderboard**: Performance ranking and contribution analysis
- **System Overview**: Health status and resource utilization
- **Historical Analysis**: Trend analysis and performance comparisons

## 🛠️ API Reference

### Coordinator API

```python
# Start training
training_id = await coordinator.start_training(config)

# Get training status
status = coordinator.get_training_status(training_id)

# Stop training
await coordinator.stop_training(training_id)

# Get system statistics
stats = coordinator.get_coordinator_stats()
```

### Monitor API

```python
# Record metrics
await monitor.record_training_metric(metric)

# Get training progress
progress = monitor.get_training_progress(training_id)

# Get system health
health = monitor.get_system_health()

# Get alerts
alerts = monitor.get_alerts(training_id, severity="critical")
```

### Version Manager API

```python
# Create model version
version_id = await version_manager.create_model_version(training_id, aggregation_result)

# Get model version
model_data = await version_manager.get_model_version(version_id, training_id)

# Rollback to version
rollback_version = await version_manager.rollback_to_version(training_id, target_version)

# Compare versions
diff = await version_manager.compare_versions(training_id, version1, version2)
```

## 🌐 Web Dashboard

### Features
- **Training Management**: Start, stop, and configure training sessions
- **Real-Time Monitoring**: Live updates of training progress and metrics
- **Client Overview**: Performance and reliability statistics
- **System Health**: Resource usage and component status
- **Alert Management**: View and acknowledge system alerts
- **Model History**: Version tracking and rollback capabilities

### REST API Endpoints

```
GET  /api/health                     - System health check
POST /api/training/start             - Start new training
GET  /api/training/{id}/status       - Get training status
POST /api/training/{id}/stop         - Stop training
GET  /api/training/{id}/metrics      - Get training metrics
GET  /api/system/health              - Get system health
GET  /api/alerts                     - Get system alerts
```

### WebSocket Events

```javascript
// Connect to real-time updates
const ws = new WebSocket('ws://localhost:8080/ws/client_id');

// Subscribe to training updates
ws.send(JSON.stringify({
    action: 'subscribe',
    training_id: 'training_123'
}));

// Handle real-time events
ws.onmessage = function(event) {
    const message = JSON.parse(event.data);
    switch(message.type) {
        case 'metric_update':
            updateChart(message.metric);
            break;
        case 'training_completed':
            showNotification('Training completed!');
            break;
    }
};
```

## 🧪 Testing

### Unit Tests

```bash
# Run all tests
python -m pytest federated_learning/tests/

# Run specific test module
python -m pytest federated_learning/tests/test_coordinator.py

# Run with coverage
python -m pytest --cov=federated_learning
```

### Integration Tests

```bash
# End-to-end system test
python scripts/test-federated-system.py

# Performance benchmarks
python scripts/benchmark-aggregation.py

# Privacy validation
python scripts/validate-privacy.py
```

### Load Testing

```bash
# Simulate high client load
python scripts/simulate-clients.py --clients 50 --rounds 20

# Stress test aggregation
python scripts/stress-test-aggregation.py --concurrent 10
```

## 🐛 Troubleshooting

### Common Issues

**Redis Connection Failed**
```bash
# Check Redis status
redis-cli ping

# Start Redis server
redis-server /etc/redis/redis.conf
```

**Insufficient FL Clients**
```
Error: Not enough FL-capable agents
Solution: Ensure at least 3 agents have learning capability
```

**Memory Exhaustion**
```bash
# Monitor memory usage
htop

# Reduce concurrent trainings
# Edit config.json: "max_concurrent_trainings": 1
```

**Dashboard Not Accessible**
```bash
# Check if port is in use
netstat -tulpn | grep 8080

# Use different port
python scripts/deploy-federated-learning.py --config custom_config.json
```

### Debugging

Enable debug logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

Check system logs:
```bash
tail -f /opt/sutazaiapp/backend/logs/federated_deployment.log
```

### Performance Issues

**Slow Aggregation**
- Reduce model size or use compression
- Decrease number of clients per round
- Enable GPU acceleration if available

**High Memory Usage**
- Enable model compression
- Reduce checkpoint frequency
- Implement gradient accumulation

**Network Bottlenecks**
- Enable update compression
- Use differential updates
- Implement client scheduling

## 📝 Contributing

### Development Setup

```bash
# Clone repository
git clone https://github.com/sutazai/backend

# Install development dependencies
pip install -r requirements-dev.txt

# Run pre-commit hooks
pre-commit install

# Run tests before committing
python -m pytest
```

### Code Style

- Follow PEP 8 for Python code
- Use type hints for all functions
- Document all public APIs
- Write unit tests for new features

### Pull Request Process

1. Fork the repository
2. Create feature branch
3. Implement changes with tests
4. Update documentation
5. Submit pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of the SutazAI distributed agent framework
- Inspired by federated learning research from Google, CMU, and MIT
- Uses privacy-preserving techniques from differential privacy literature
- Implements secure aggregation protocols from cryptographic research

## 📞 Support

- **Documentation**: [docs.sutazai.com/federated-learning](https://docs.sutazai.com/federated-learning)
- **Issues**: [GitHub Issues](https://github.com/sutazai/backend/issues)
- **Community**: [Discord Server](https://discord.gg/sutazai)
- **Email**: federated-learning@sutazai.com

---

**🤖 SutazAI Federated Learning System v1.0**
*Privacy-preserving distributed AI training across 69+ intelligent agents*