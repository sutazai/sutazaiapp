# SutazAI MLflow System

**Comprehensive ML Experiment Tracking and Pipeline Automation for SutazAI's 69+ AI Agents**

## 🚀 Overview

The SutazAI MLflow System is a production-ready, high-performance experiment tracking and ML pipeline automation platform designed specifically for SutazAI's distributed AI agent architecture. It provides comprehensive tracking, analysis, and automation capabilities for all 69+ AI agents in the system.

### Key Features

- **🤖 Automated Agent Tracking**: Seamless experiment tracking for all 69+ AI agents
- **⚙️ ML Pipeline Automation**: End-to-end pipeline automation with hyperparameter tuning
- **📊 Advanced Analytics**: Comprehensive experiment comparison and performance analysis
- **🖥️ Interactive Dashboards**: Real-time monitoring and visualization
- **🔄 High-Volume Support**: Optimized for concurrent experiments and high throughput
- **📈 Model Registry**: Complete model lifecycle management
- **🔗 System Integration**: Deep integration with SutazAI infrastructure

## 📋 Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Architecture](#architecture)
- [Core Components](#core-components)
- [Configuration](#configuration)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Dashboard Guide](#dashboard-guide)
- [Integration Guide](#integration-guide)
- [Performance Tuning](#performance-tuning)
- [Troubleshooting](#troubleshooting)

## 🛠️ Installation

### Prerequisites

- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- 16GB+ RAM (recommended)
- 8+ CPU cores (recommended)

### System Dependencies

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install -y postgresql postgresql-contrib redis-server python3-dev

# CentOS/RHEL
sudo yum install -y postgresql-server postgresql-contrib redis python3-devel
```

### Python Installation

```bash
# Navigate to the MLflow system directory
cd /opt/sutazaiapp/backend/mlflow_system

# Install Python dependencies
pip install -r requirements.txt

# Or using virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\\Scripts\\activate
pip install -r requirements.txt
```

### Database Setup

```bash
# Create MLflow database
sudo -u postgres createdb mlflow_db
sudo -u postgres createuser mlflow
sudo -u postgres psql -c "ALTER USER mlflow WITH ENCRYPTED PASSWORD 'mlflow_secure_pwd';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow;"
```

### Redis Configuration

```bash
# Start Redis
sudo systemctl start redis
sudo systemctl enable redis

# Test Redis connection
redis-cli ping
```

## 🚀 Quick Start

### 1. Basic System Startup

```bash
# Start the complete MLflow system with dashboard
python start_system.py start --dashboard

# Start without dashboard
python start_system.py start --no-dashboard

# Start with custom configuration
python start_system.py start --config custom_config.yaml
```

### 2. Using Python API

```python
import asyncio
from mlflow_system import initialize_system, start_agent_tracking

async def main():
    # Initialize the system
    success = await initialize_system()
    if not success:
        print("Failed to initialize MLflow system")
        return
    
    # Start tracking for an agent
    tracker = await start_agent_tracking(
        agent_id="agent_001",
        agent_type="neural_network",
        agent_config={
            "framework": "pytorch",
            "model_type": "transformer"
        }
    )
    
    if tracker:
        # Start a run
        run_id = await tracker.start_run("training_session_1")
        
        # Log parameters
        await tracker.log_params({
            "learning_rate": 0.001,
            "batch_size": 32,
            "epochs": 100
        })
        
        # Log metrics
        for epoch in range(10):
            await tracker.log_metrics({
                "train_loss": 1.0 - (epoch * 0.1),
                "accuracy": 0.5 + (epoch * 0.05)
            }, step=epoch)
        
        # End the run
        await tracker.end_run("FINISHED")

# Run the example
asyncio.run(main())
```

### 3. CLI Usage

```bash
# Check system status
python start_system.py status

# Start tracking for specific agent
python start_system.py track-agent agent_001 --agent-type neural_network

# Compare experiments
python start_system.py compare exp_1 exp_2 exp_3 --metrics accuracy,loss

# View configuration
python start_system.py config
```

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SutazAI MLflow System                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │  Dashboard   │  │   API Layer  │  │ Integration  │          │
│  │              │  │              │  │              │          │
│  │ • Streamlit  │  │ • REST API   │  │ • SutazAI    │          │
│  │ • Plotly     │  │ • WebSocket  │  │ • PostgreSQL │          │
│  │ • Real-time  │  │ • AsyncIO    │  │ • Redis      │          │
│  └──────────────┘  └──────────────┘  └──────────────┘          │
│           │                  │                  │               │
│  ┌────────┼──────────────────┼──────────────────┼──────────┐    │
│  │        │                  │                  │          │    │
│  │  ┌─────▼────┐  ┌─────────▼────┐  ┌──────────▼───┐      │    │
│  │  │Agent     │  │ Pipeline     │  │  Analysis    │      │    │
│  │  │Tracker   │  │ Automation   │  │  Tools       │      │    │
│  │  │          │  │              │  │              │      │    │
│  │  │• 69+     │  │ • Hyperopt   │  │ • Comparison │      │    │
│  │  │  Agents  │  │ • AutoML     │  │ • Statistics │      │    │
│  │  │• Auto    │  │ • Deployment │  │ • Drift Det. │      │    │
│  │  │  Logging │  │ • Monitoring │  │ • Recomm.    │      │    │
│  │  └──────────┘  └──────────────┘  └──────────────┘      │    │
│  │                                                        │    │
│  └────────────────────────────────────────────────────────┘    │
│                                  │                              │
│  ┌───────────────────────────────▼──────────────────────────┐  │
│  │                 MLflow Core Services                     │  │
│  │                                                          │  │
│  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐    │  │
│  │  │Tracking  │ │Artifact  │ │Model     │ │Metrics   │    │  │
│  │  │Server    │ │Store     │ │Registry  │ │Backend   │    │  │
│  │  │          │ │          │ │          │ │          │    │  │
│  │  │• HTTP    │ │• Local   │ │• Versions│ │• Postgres│    │  │
│  │  │• REST    │ │• S3      │ │• Stages  │ │• High    │    │  │
│  │  │• Async   │ │• Compress│ │• Deploy  │ │  Volume  │    │  │
│  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘    │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

## 🧩 Core Components

### 1. Tracking Server (`tracking_server.py`)
- High-performance MLflow tracking server
- Optimized for 69+ concurrent agents
- PostgreSQL backend with connection pooling
- Automatic cleanup and monitoring

### 2. Agent Tracker (`agent_tracker.py`)
- Automated experiment tracking for each agent
- Framework-agnostic (PyTorch, TensorFlow, Scikit-learn)
- Real-time metrics and parameter logging
- System resource monitoring

### 3. Pipeline Automation (`pipeline_automation.py`)
- End-to-end ML pipeline automation
- Hyperparameter optimization (Optuna, Random Search, Grid Search)
- Automated model deployment
- A/B testing frameworks

### 4. Analysis Tools (`analysis_tools.py`)
- Statistical experiment comparison
- Model performance analysis
- Drift detection
- Automated recommendations

### 5. Dashboard (`dashboard.py`)
- Interactive Streamlit dashboard
- Real-time experiment monitoring
- Visualization and charts
- Pipeline management interface

### 6. Integration Layer (`integration.py`)
- SutazAI system integration
- Database synchronization
- API integration
- Messaging and notifications

## ⚙️ Configuration

### Main Configuration File

The system uses `/opt/sutazaiapp/backend/mlflow_system/mlflow_config.yaml`:

```yaml
# Tracking Server Configuration
tracking_uri: "postgresql://mlflow:mlflow_secure_pwd@localhost:5432/mlflow_db"
tracking_server_host: "0.0.0.0"
tracking_server_port: 5000

# Backend Store Configuration
backend_store_uri: "postgresql://mlflow:mlflow_secure_pwd@localhost:5432/mlflow_db"

# Artifact Store Configuration
artifact_root: "/opt/sutazaiapp/backend/mlflow_artifacts"
s3_artifact_root: null

# Performance Configuration
max_concurrent_experiments: 50
batch_logging_size: 100
batch_logging_timeout: 30

# High-Volume Settings
enable_async_logging: true
enable_compression: true
artifact_compression_level: 6

# Agent Integration
agent_tracking_enabled: true
auto_log_models: true
auto_log_params: true
auto_log_metrics: true

# Database Settings
db_pool_size: 20
db_max_overflow: 30
db_pool_timeout: 30
db_pool_recycle: 3600

# Cleanup Settings
artifact_retention_days: 90
experiment_retention_days: 365
enable_auto_cleanup: true

# Monitoring
enable_prometheus_metrics: true
metrics_port: 8080
```

### Environment Variables

```bash
export MLFLOW_TRACKING_URI="postgresql://mlflow:mlflow_secure_pwd@localhost:5432/mlflow_db"
export MLFLOW_ARTIFACT_ROOT="/opt/sutazaiapp/backend/mlflow_artifacts"
export MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING="true"
```

## 📖 Usage Examples

### Experiment Tracking

```python
import asyncio
from mlflow_system import start_agent_tracking

async def train_model():
    # Initialize agent tracking
    tracker = await start_agent_tracking(
        agent_id="sentiment_analyzer",
        agent_type="transformer",
        agent_config={
            "framework": "pytorch",
            "model_architecture": "bert-base"
        }
    )
    
    # Start experiment run
    run_id = await tracker.start_run("sentiment_training_v1")
    
    # Log hyperparameters
    await tracker.log_params({
        "learning_rate": 2e-5,
        "batch_size": 16,
        "max_epochs": 3,
        "dropout": 0.1,
        "warmup_steps": 500
    })
    
    # Training loop
    for epoch in range(3):
        # ... training code ...
        
        # Log training metrics
        await tracker.log_metrics({
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "learning_rate": current_lr
        }, step=epoch)
        
        # Log model checkpoint
        if epoch == 2:  # Final epoch
            await tracker.log_model(
                model=model,
                artifact_path="final_model",
                signature=model_signature,
                input_example=sample_input
            )
    
    # End the run
    await tracker.end_run("FINISHED")

asyncio.run(train_model())
```

### Pipeline Automation

```python
from mlflow_system.pipeline_automation import PipelineConfig, TuningAlgorithm

# Create pipeline configuration
pipeline_config = PipelineConfig(
    name="image_classification_pipeline",
    description="Automated image classification with hyperparameter tuning",
    agent_id="image_classifier_agent",
    agent_type="cnn",
    training_script="train_image_classifier.py",
    data_path="/data/images/",
    
    # Hyperparameter tuning
    enable_hyperparameter_tuning=True,
    tuning_algorithm=TuningAlgorithm.OPTUNA_TPE,
    tuning_trials=50,
    hyperparameter_space={
        "learning_rate": {"type": "float", "low": 1e-5, "high": 1e-2},
        "batch_size": {"type": "categorical", "choices": [16, 32, 64]},
        "dropout": {"type": "float", "low": 0.1, "high": 0.5}
    },
    
    # Deployment
    auto_deploy=True,
    deployment_threshold=0.90,
    
    # Monitoring
    enable_model_monitoring=True,
    monitoring_metrics=["accuracy", "precision", "recall", "f1_score"]
)

# Create and execute pipeline
async def run_pipeline():
    from mlflow_system import create_ml_pipeline, run_pipeline
    
    # Create pipeline
    success = await create_ml_pipeline(pipeline_config.to_dict())
    if success:
        # Execute pipeline
        await run_pipeline("image_classification_pipeline")

asyncio.run(run_pipeline())
```

### Experiment Comparison

```python
from mlflow_system import compare_experiments_simple

async def compare_models():
    # Compare multiple experiments
    result = await compare_experiments_simple(
        experiment_ids=["exp_1", "exp_2", "exp_3"],
        metrics=["accuracy", "precision", "recall", "f1_score"]
    )
    
    if result:
        print(f"Total runs analyzed: {result['total_runs']}")
        print(f"Best performing experiment: {result['best_experiment']}")
        
        # Print recommendations
        for recommendation in result['recommendations']:
            print(f"💡 {recommendation}")
        
        # Statistical significance
        for metric, tests in result['statistical_tests'].items():
            if tests.get('anova', {}).get('significant', False):
                print(f"📊 Significant differences found in {metric}")

asyncio.run(compare_models())
```

## 🖥️ Dashboard Guide

### Accessing the Dashboard

1. **Start the system with dashboard:**
   ```bash
   python start_system.py start --dashboard
   ```

2. **Open in browser:**
   ```
   http://localhost:8501
   ```

### Dashboard Features

#### 1. Overview Page
- System metrics and health status
- Recent experiment activity
- Agent tracking summary
- Quick statistics

#### 2. Experiments Management
- List all experiments
- Filter and search experiments
- View experiment details
- Compare experiment runs

#### 3. Agent Tracking
- Monitor all 69+ agents
- View tracking status
- Agent performance metrics
- Resource utilization

#### 4. Pipeline Management
- View available pipelines
- Start/stop pipeline execution
- Monitor pipeline progress
- Configure pipeline parameters

#### 5. Analysis & Comparison
- Compare multiple experiments
- Statistical analysis
- Performance trends
- Automated recommendations

#### 6. System Health
- Server status monitoring
- Database health
- Resource usage charts
- Error logs and alerts

## 🔗 Integration Guide

### SutazAI System Integration

The MLflow system integrates seamlessly with the existing SutazAI infrastructure:

#### Database Integration
```python
# Automatic sync with SutazAI database
from mlflow_system.integration import DatabaseIntegration

db_integration = DatabaseIntegration()
await db_integration.initialize()

# Get agent metadata
metadata = await db_integration.get_agent_metadata("agent_001")

# Store experiment results
await db_integration.store_experiment_results({
    "experiment_id": "exp_123",
    "agent_id": "agent_001",
    "metrics": {"accuracy": 0.95},
    "parameters": {"lr": 0.001}
})
```

#### API Integration
```python
# Notify SutazAI API of events
from mlflow_system.integration import APIIntegration

api_integration = APIIntegration()
await api_integration.initialize(mlflow_config)

# Notify experiment started
await api_integration.notify_experiment_started("exp_123", "agent_001")

# Notify model deployed
await api_integration.notify_model_deployed("model_v1", "1.0", "agent_001")
```

#### Messaging Integration
```python
# Redis messaging for real-time events
from mlflow_system.integration import MessagingIntegration

messaging = MessagingIntegration()
await messaging.initialize(redis_client)

# Publish experiment event
await messaging.publish_experiment_event(
    "experiment_completed",
    "exp_123",
    {"accuracy": 0.95, "duration": 3600}
)
```

## 🚀 Performance Tuning

### Database Optimization

1. **Connection Pooling:**
   ```yaml
   db_pool_size: 20
   db_max_overflow: 30
   db_pool_timeout: 30
   db_pool_recycle: 3600
   ```

2. **Indexes for Performance:**
   The system automatically creates optimized indexes for:
   - Experiment queries by name and status
   - Run queries by experiment and timestamp
   - Metric and parameter lookups
   - Model registry operations

3. **Batch Operations:**
   ```yaml
   batch_logging_size: 100
   batch_logging_timeout: 30
   enable_async_logging: true
   ```

### High-Volume Configuration

For systems with 100+ concurrent experiments:

```yaml
max_concurrent_experiments: 100
db_pool_size: 50
db_max_overflow: 100
artifact_compression_level: 9
enable_compression: true
```

### Memory Optimization

```yaml
# Artifact cleanup
artifact_retention_days: 30
experiment_retention_days: 180
enable_auto_cleanup: true

# Memory-efficient logging
batch_logging_size: 50
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Database Connection Errors
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Test connection
psql -h localhost -U mlflow -d mlflow_db

# Fix: Restart PostgreSQL
sudo systemctl restart postgresql
```

#### 2. Redis Connection Issues
```bash
# Check Redis status
sudo systemctl status redis

# Test connection
redis-cli ping

# Fix: Restart Redis
sudo systemctl restart redis
```

#### 3. Port Conflicts
```bash
# Check if ports are in use
netstat -tulpn | grep :5000  # MLflow server
netstat -tulpn | grep :8501  # Dashboard
netstat -tulpn | grep :8080  # Metrics

# Fix: Change ports in configuration
```

#### 4. High Memory Usage
```yaml
# Reduce memory usage
batch_logging_size: 25
max_concurrent_experiments: 25
enable_compression: true
artifact_compression_level: 9
```

#### 5. Slow Performance
```bash
# Check database performance
SELECT * FROM pg_stat_activity WHERE state = 'active';

# Check disk space
df -h /opt/sutazaiapp/backend/mlflow_artifacts

# Optimize database
VACUUM ANALYZE;
REINDEX DATABASE mlflow_db;
```

### Debug Mode

Enable debug logging:
```bash
python start_system.py start --debug
```

Or set environment variable:
```bash
export MLFLOW_DEBUG=true
```

### Log Files

Check system logs:
```bash
# System log
tail -f /opt/sutazaiapp/backend/logs/mlflow_system.log

# MLflow server log
tail -f /opt/sutazaiapp/backend/logs/mlflow_server.log

# Dashboard log
tail -f /opt/sutazaiapp/backend/logs/dashboard.log
```

## 📚 API Reference

### System Management

```python
from mlflow_system import *

# Initialize system
success = await initialize_system(config=None)

# Get system status
status = get_status()

# Shutdown system
await shutdown_system()
```

### Agent Tracking

```python
# Start agent tracking
tracker = await start_agent_tracking(
    agent_id="agent_001",
    agent_type="neural_network",
    agent_config={"framework": "pytorch"}
)

# Start run
run_id = await tracker.start_run("experiment_name")

# Log parameters
await tracker.log_param("param_name", "param_value")
await tracker.log_params({"param1": "value1", "param2": "value2"})

# Log metrics
await tracker.log_metric("metric_name", 0.95, step=1)
await tracker.log_metrics({"accuracy": 0.95, "loss": 0.05}, step=1)

# Log artifacts
await tracker.log_artifact("model.pkl", "models/")

# Log model
await tracker.log_model(model, "model_path")

# End run
await tracker.end_run("FINISHED")
```

### Pipeline Management

```python
# Create pipeline
config = {
    "name": "pipeline_name",
    "description": "Pipeline description",
    "agent_id": "agent_001",
    "training_script": "train.py"
}
success = await create_ml_pipeline(config)

# Execute pipeline
success = await run_pipeline("pipeline_name")
```

### Analysis Tools

```python
# Compare experiments
result = await compare_experiments_simple(
    experiment_ids=["exp1", "exp2"],
    metrics=["accuracy", "loss"]
)

# Analyze model performance
analysis = await analyze_model("model_name", days=30)
```

## 🤝 Contributing

1. **Code Style:** Follow PEP 8 and use Black for formatting
2. **Testing:** Add tests for new features using pytest
3. **Documentation:** Update documentation for any changes
4. **Type Hints:** Use type hints for all functions

## 📄 License

This project is part of the SutazAI system and follows the same licensing terms.

## 🆘 Support

- **Documentation:** Complete system documentation
- **Issues:** Report issues through the SutazAI issue tracker
- **Community:** Join the SutazAI community discussions

---

**🧪 SutazAI MLflow System v1.0**
*Comprehensive ML experiment tracking and automation for distributed AI systems*