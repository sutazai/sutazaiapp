# SutazAI Monitoring Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Architecture](#architecture)
3. [Components](#components)
4. [Installation](#installation)
5. [Configuration](#configuration)
6. [Usage](#usage)
7. [Metrics](#metrics)
8. [Alerting](#alerting)
9. [Troubleshooting](#troubleshooting)
10. [Security](#security)
11. [Best Practices](#best-practices)
12. [References](#references)

## Introduction

The SutazAI monitoring and logging system provides comprehensive observability for the entire platform. It combines structured logging, metrics collection, distributed tracing, and error tracking to ensure that all aspects of the system are properly monitored and analyzed.

### Key Features

- **Structured Logging**: JSON-formatted logs for easy parsing and analysis
- **Metrics Collection**: Real-time metrics from all system components
- **Distributed Tracing**: End-to-end request tracking
- **Error Tracking**: Centralized error collection and analysis
- **Dashboards**: Pre-built dashboards for various system aspects
- **Alerting**: Configurable alerts for critical events
- **Anomaly Detection**: Automated detection of unusual system behavior

## Architecture

The monitoring architecture consists of several layers:

1. **Collection Layer**: Collects logs, metrics, and traces from all system components
2. **Processing Layer**: Processes, enriches, and transforms monitoring data
3. **Storage Layer**: Stores monitoring data for short and long-term analysis
4. **Visualization Layer**: Provides interfaces for visualizing and analyzing monitoring data
5. **Alerting Layer**: Monitors data for anomalies and sends notifications

![Monitoring Architecture](../docs/images/monitoring_architecture.png)

## Components

### Logging Stack (ELK)

The logging stack is based on the ELK (Elasticsearch, Logstash, Kibana) stack, augmented with Filebeat and Metricbeat.

- **Elasticsearch**: Distributed search and analytics engine for storing logs
- **Logstash**: Data processing pipeline for log ingestion and transformation
- **Kibana**: Visualization platform for exploring and analyzing logs
- **Filebeat**: Lightweight shipper for forwarding logs
- **Metricbeat**: Lightweight shipper for metric collection

### Metrics Stack (Prometheus)

The metrics stack is based on Prometheus and Grafana.

- **Prometheus**: Time-series database for storing metrics
- **Grafana**: Visualization platform for metrics and dashboards
- **Node Exporter**: Exporter for system metrics
- **cAdvisor**: Container resource usage and performance metrics

### Error Tracking (Sentry)

Sentry provides real-time error tracking and monitoring.

- **Sentry**: Real-time error tracking and monitoring platform
- **Sentry SDK**: Client libraries for various languages and frameworks

### FastAPI Integration

FastAPI integration includes:

- **Middleware**: Request/response tracking middleware
- **Logging Integration**: Structured logging for all API endpoints
- **Metrics Endpoint**: Prometheus metrics endpoint for API metrics
- **Health Checks**: Application and dependency health checks

## Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Sufficient disk space (at least 10GB recommended)
- Network access to required ports (3000, 5601, 9090, 9000, etc.)

### Setup Process

1. Clone the SutazAI repository:
   ```bash
   git clone https://github.com/your-organization/sutazai.git
   cd sutazai
   ```

2. Run the setup script:
   ```bash
   cd monitoring
   ./setup_monitoring.sh
   ```

3. Configure the environment:
   ```bash
   # Update the .env file with your settings
   nano .env
   ```

4. Start the monitoring stack:
   ```bash
   ./start_monitoring.sh
   ```

### Verification

After installation, verify that all components are running correctly:

1. Check Docker Compose status:
   ```bash
   docker-compose ps
   ```

2. Access the following interfaces:
   - Grafana: http://localhost:3000 (admin/sutazai)
   - Kibana: http://localhost:5601 (elastic/sutazaisecure)
   - Prometheus: http://localhost:9090
   - Sentry: http://localhost:9000

## Configuration

### Logging Configuration

The logging system is configured in `/opt/sutazaiapp/utils/logging_setup.py`. Key configuration options:

- **LOG_LEVEL**: Controls the verbosity of logs (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- **LOG_DIR**: Directory for log files (default: `/opt/sutazaiapp/logs`)
- **REDACT_FIELDS**: Sensitive fields to redact from logs

### Prometheus Configuration

Prometheus is configured in `/opt/sutazaiapp/monitoring/prometheus.yml`. Key configuration options:

- **scrape_interval**: How frequently to scrape targets (default: 15s)
- **evaluation_interval**: How frequently to evaluate rules (default: 15s)
- **scrape_configs**: List of jobs and targets to scrape

### Grafana Configuration

Grafana dashboards are stored in `/opt/sutazaiapp/monitoring/grafana/dashboards/`. Default dashboards include:

- **System Overview**: CPU, memory, disk, and network metrics
- **API Performance**: Request counts, latencies, and error rates
- **Model Performance**: Inference times, memory usage, and errors

### ELK Stack Configuration

The ELK stack is configured with the following files:

- **Elasticsearch**: `/opt/sutazaiapp/monitoring/elk/elasticsearch/elasticsearch.yml`
- **Logstash**: `/opt/sutazaiapp/monitoring/elk/logstash/logstash.yml` and `/opt/sutazaiapp/monitoring/elk/logstash/pipeline/`
- **Kibana**: `/opt/sutazaiapp/monitoring/elk/kibana/kibana.yml`
- **Filebeat**: `/opt/sutazaiapp/monitoring/elk/filebeat/filebeat.yml`

### Sentry Configuration

Sentry is configured in `/opt/sutazaiapp/monitoring/sentry/.env`. Key configuration options:

- **SENTRY_SECRET_KEY**: Secret key for Sentry
- **SENTRY_DSN**: DSN for connecting to Sentry

## Usage

### Logging

SutazAI uses a structured logging approach with different loggers for different components:

```python
from utils.logging_setup import get_app_logger, get_model_logger, get_api_logger

# Application logs
app_logger = get_app_logger()
app_logger.info("Application started")

# Model logs
model_logger = get_model_logger("gpt4")
model_logger.info("Model loaded successfully")

# API logs
api_logger = get_api_logger()
api_logger.info("API request received", extra={"endpoint": "/api/v1/generate", "method": "POST"})
```

### Monitoring Middleware

The monitoring middleware is automatically applied to all FastAPI routes:

```python
from fastapi import FastAPI
from utils.monitoring import setup_monitoring

app = FastAPI()
setup_monitoring(app)
```

### Custom Metrics

You can add custom metrics to your application:

```python
from prometheus_client import Counter, Histogram, Gauge

# Define metrics
MODEL_INFERENCE_COUNT = Counter('model_inference_total', 'Total number of model inferences', ['model_name'])
MODEL_INFERENCE_LATENCY = Histogram('model_inference_seconds', 'Model inference latency in seconds', ['model_name'])
MODEL_MEMORY_USAGE = Gauge('model_memory_bytes', 'Model memory usage in bytes', ['model_name'])

# Use metrics
MODEL_INFERENCE_COUNT.labels(model_name="gpt4").inc()
with MODEL_INFERENCE_LATENCY.labels(model_name="gpt4").time():
    # Run inference
    result = model.generate(prompt)
MODEL_MEMORY_USAGE.labels(model_name="gpt4").set(memory_usage)
```

## Metrics

### System Metrics

- **CPU Usage**: `node_cpu_seconds_total`
- **Memory Usage**: `node_memory_MemTotal_bytes`, `node_memory_MemFree_bytes`
- **Disk Usage**: `node_filesystem_avail_bytes`, `node_filesystem_size_bytes`
- **Network Usage**: `node_network_receive_bytes_total`, `node_network_transmit_bytes_total`

### Container Metrics

- **Container CPU**: `container_cpu_usage_seconds_total`
- **Container Memory**: `container_memory_usage_bytes`
- **Container Network**: `container_network_receive_bytes_total`, `container_network_transmit_bytes_total`

### Application Metrics

- **HTTP Requests**: `sutazai_http_requests_total`
- **HTTP Latency**: `sutazai_http_request_duration_seconds`
- **Request Size**: `sutazai_http_request_size_bytes`
- **Response Size**: `sutazai_http_response_size_bytes`

### Model Metrics

- **Model Inference Count**: `sutazai_model_inference_total`
- **Model Inference Latency**: `sutazai_model_inference_duration_seconds`
- **Model Memory Usage**: `sutazai_model_memory_bytes`
- **Model Errors**: `sutazai_model_errors_total`

## Alerting

### Prometheus Alerting

AlertManager is configured to send alerts for various conditions:

```yaml
groups:
- name: example
  rules:
  - alert: HighCPUUsage
    expr: 100 - (avg by(instance) (irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High CPU usage detected
      description: CPU usage is above 80% for the last 5 minutes
```

### Grafana Alerting

Grafana alerts can be configured directly in the dashboard UI:

1. Open the dashboard
2. Edit the panel
3. Go to "Alert" tab
4. Configure the alert conditions

### Notification Channels

Alerts can be sent through various channels:

- Email
- Slack
- PagerDuty
- Webhook
- Custom integrations

## Troubleshooting

### Common Issues

#### Prometheus Issues

- **No data in Prometheus**: Check scrape targets at http://localhost:9090/targets
- **Targets down**: Check if the exporters are running and accessible
- **Configuration issues**: Validate prometheus.yml with `promtool check config prometheus.yml`

#### ELK Stack Issues

- **Elasticsearch not starting**: Check Docker logs with `docker-compose logs elasticsearch`
- **No logs in Kibana**: Check if indices exist with `curl -XGET 'localhost:9200/_cat/indices?v'`
- **Logstash pipeline issues**: Check configuration with `docker-compose exec logstash logstash --config.test_and_exit`

#### Grafana Issues

- **No data in Grafana**: Check data source configuration
- **Dashboard rendering issues**: Clear browser cache or try a different browser
- **Plugin issues**: Check plugin compatibility with your Grafana version

### Logs Location

- **Application Logs**: `/opt/sutazaiapp/logs/*.log`
- **Docker Container Logs**: `docker-compose logs <service_name>`
- **Elasticsearch Logs**: `/opt/sutazaiapp/monitoring/elk/elasticsearch/logs/`

## Security

### Access Control

- **Authentication**: All monitoring interfaces require authentication
- **Authorization**: Users are granted specific permissions based on roles
- **Network Security**: Services are only exposed on necessary ports

### Data Protection

- **Log Redaction**: Sensitive information is automatically redacted from logs
- **Encryption**: Data in transit is encrypted with TLS
- **Data Retention**: Logs and metrics are retained based on configurable policies

### Compliance Considerations

The monitoring system is designed to help meet compliance requirements:

- **SOC2**: Comprehensive logging for security and availability
- **GDPR**: Personal data is redacted from logs
- **HIPAA**: Health information is protected through data redaction

## Best Practices

### Log Management

1. **Use Structured Logging**: Always use structured (JSON) logs
2. **Include Context**: Include relevant context in logs (user ID, request ID, etc.)
3. **Log Levels**: Use appropriate log levels based on the message importance
4. **Avoid Sensitive Data**: Never log sensitive information (passwords, tokens, PII)

### Metrics Collection

1. **Focus on Key Metrics**: Monitor what matters, not everything
2. **Label Properly**: Use consistent labels for metrics
3. **Consider Cardinality**: Avoid high cardinality labels that explode the number of time series
4. **Rate vs Gauge**: Use rates for counters and gauges for current values

### Alerting Strategy

1. **Alert on Symptoms**: Alert on user-facing symptoms, not causes
2. **Reduce Noise**: Minimize false positives and alert fatigue
3. **Actionable Alerts**: Ensure all alerts are actionable
4. **Clear Documentation**: Document alert response procedures

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Elasticsearch Documentation](https://www.elastic.co/guide/index.html)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Sentry Documentation](https://docs.sentry.io/)

# SutazAI automation system/advanced automation Monitoring System

## Overview

The SutazAI monitoring system provides comprehensive observability and monitoring capabilities specifically designed for automation system/advanced automation systems. The system monitors various aspects of the automation system architecture, including processing components, ethical constraints, self-modification capabilities, hardware optimization, and security.

## Architecture

```
SutazAI Monitoring System
├── Processing Architecture Monitoring
│   ├── Spike Activity Tracking
│   ├── Attention Mechanism Monitoring
│   └── Connection Plasticity Tracking
├── Ethical Constraints Verification
│   ├── Decision Boundary Enforcement
│   ├── Ethical Properties Verification
│   └── Value Alignment Tracking
├── Self-Modification Monitoring
│   ├── Change Detection
│   ├── Dual-Execution Validation
│   └── Audit Trail Maintenance
├── Hardware Optimization Monitoring
│   ├── Resource Usage Tracking
│   ├── Model Optimization Metrics
│   └── Quantization Efficiency
├── Security Architecture Monitoring
│   ├── System Integrity Verification
│   ├── Anomaly Detection
│   └── Air-Gap Validation
└── FastAPI Integration
    ├── Middleware Integration
    ├── Metrics Collection
    └── Visualization and Alerts
```

## Components

The monitoring system consists of the following components:

1. **Prometheus**: Time-series database for metrics storage and querying
2. **AlertManager**: Handles alerts from Prometheus and sends notifications
3. **Grafana**: Visualization and dashboarding for metrics and logs
4. **ELK Stack**: Elasticsearch, Logstash, and Kibana for log aggregation
5. **Node Exporter**: Exports system metrics such as CPU, memory, and disk usage
6. **cAdvisor**: Container metrics exporter
7. **Custom Monitoring Modules**: Python modules for specialized automation system/advanced automation monitoring

## Installation

### Prerequisites

- Docker and Docker Compose
- Python 3.7+
- pip

### Setup

1. Clone the repository and navigate to the monitoring directory:

```bash
cd /opt/sutazaiapp/monitoring
```

2. Run the configuration script:

```bash
./configure_monitoring.sh
```

3. Start the monitoring system:

```bash
./run_all.sh
```

## Accessing the Monitoring System

- **Grafana**: http://localhost:3000 (default credentials: admin/sutazai)
- **Prometheus**: http://localhost:9090
- **AlertManager**: http://localhost:9093
- **Kibana**: http://localhost:5601
- **Monitoring API**: http://localhost:8100

## Processing Architecture Monitoring

The Processing Architecture Monitoring module tracks metrics related to the biologically-inspired processing architecture:

- **Spike Activity**: Monitor the rate and pattern of processing spikes across different network layers
- **Attention Mechanisms**: Track attention head activations and patterns
- **Connection Plasticity**: Measure changes in connection weights and connections

Example usage:

```python
from utils.processing_monitoring import ProcessingMonitor

# Initialize the processing monitor
processing_monitor = ProcessingMonitor(
    system_id="sutazai_main",
    log_dir="/opt/sutazaiapp/logs/processing"
)

# Record spike activity
processing_monitor.record_spike_activity(
    layer_id="transformer_layer_5",
    spike_rate=0.42,
    activation_pattern=[0.1, 0.3, 0.2, 0.8, 0.4]
)

# Record attention mechanism
processing_monitor.record_attention(
    head_id="attn_head_3",
    attention_scores=[[0.1, 0.8, 0.1], [0.2, 0.1, 0.7], [0.5, 0.3, 0.2]],
    attention_entropy=1.2
)

# Record connection changes
processing_monitor.record_synaptic_change(
    layer_id="transformer_layer_2",
    change_rate=0.03,
    connection_delta={"added": 5, "removed": 3, "modified": 12}
)
```

## Ethical Constraints Verification

The Ethical Constraints Verification module monitors and enforces ethical boundaries:

- **Decision Boundaries**: Enforce hard constraints on system behavior
- **Ethical Properties**: Verify formal ethical properties
- **Value Alignment**: Track alignment with human values

Example usage:

```python
from utils.ethics_verification import EthicsVerifier

# Initialize the ethics verifier
ethics_verifier = EthicsVerifier(
    system_id="sutazai_main",
    log_dir="/opt/sutazaiapp/logs/ethics",
    alert_on_violation=True
)

# Check content against ethical boundaries
result = ethics_verifier.check_content_boundaries(
    content="User provided text",
    content_type="text",
    check_types=["toxicity", "bias", "harm"]
)

# Verify a formal ethical property
property_status = ethics_verifier.verify_ethical_property(
    property_name="no_discrimination",
    context={"user_data": user_data, "model_output": model_output}
)

# Record value alignment metrics
ethics_verifier.record_value_alignment(
    alignment_scores={
        "honesty": 0.95,
        "fairness": 0.87,
        "helpfulness": 0.92
    }
)
```

## Self-Modification Monitoring

The Self-Modification Monitoring module tracks system changes and verifications:

- **Change Detection**: Monitor modifications to the system
- **Dual-Execution Validation**: Verify changes via secure dual-execution
- **Audit Trail**: Maintain complete logs of all modifications

Example usage:

```python
from utils.self_mod_monitoring import SelfModificationMonitor

# Initialize the self-modification monitor
self_mod_monitor = SelfModificationMonitor(
    system_id="sutazai_main",
    log_dir="/opt/sutazaiapp/logs/self_mod",
    enable_dual_execution=True
)

# Record a self-modification event
self_mod_monitor.record_modification(
    component="parameter_update",
    modification_type="weight_adjustment",
    details={
        "layer": "attention_layer_3",
        "parameter_count": 1024,
        "reason": "performance_optimization"
    }
)

# Verify a proposed modification
verification_result = self_mod_monitor.verify_modification(
    component="code_update",
    original_code=original_code,
    modified_code=modified_code,
    verification_criteria=["safety", "performance", "integrity"]
)

# Get the audit trail for a specific period
audit_trail = self_mod_monitor.get_audit_trail(
    start_time="2023-06-01T00:00:00Z",
    end_time="2023-06-30T23:59:59Z",
    component="all"
)
```

## Hardware Optimization Monitoring

The Hardware Optimization Monitoring module tracks resource usage and efficiency:

- **Resource Usage**: Monitor CPU, GPU, memory, and network utilization
- **Model Optimization**: Track model efficiency metrics
- **Quantization Efficiency**: Measure the effectiveness of model quantization

Example usage:

```python
from utils.hardware_monitor import HardwareMonitor

# Initialize the hardware monitor
hardware_monitor = HardwareMonitor(
    system_id="sutazai_main",
    log_dir="/opt/sutazaiapp/logs/hardware",
    collect_interval=30.0
)

# Start collecting metrics
hardware_monitor.start_collection()

# Record model optimization metrics
hardware_monitor.record_model_optimization(
    model_id="text_generation_v1",
    original_size_mb=850,
    optimized_size_mb=320,
    optimization_technique="quantization",
    performance_impact=-0.02
)

# Get current hardware profile
profile = hardware_monitor.get_hardware_profile()
print(f"CPU: {profile['cpu_usage']}%, GPU: {profile['gpu_usage']}%")

# Stop collection when done
hardware_monitor.stop_collection()
```

## Security Architecture Monitoring

The Security Architecture Monitoring module ensures system integrity and security:

- **System Integrity**: Verify the integrity of the system components
- **Anomaly Detection**: Detect unusual behavior or unauthorized access
- **Air-Gap Validation**: Ensure air-gapped environments remain secure

Example usage:

```python
from utils.security_monitoring import SecurityMonitor

# Initialize the security monitor
security_monitor = SecurityMonitor(
    system_id="sutazai_main",
    log_dir="/opt/sutazaiapp/logs/security",
    integrity_check_interval=300.0
)

# Log a security event
security_monitor.log_security_event(
    event_type="authentication",
    severity="info",
    details={
        "user": "api_user",
        "ip": "192.168.1.105",
        "success": True
    }
)

# Check system integrity
integrity_status = security_monitor.check_system_integrity(
    components=["models", "code", "configurations"]
)

# Detect anomalies in system behavior
anomalies = security_monitor.detect_anomalies(
    metrics_history=metrics_data,
    detection_window="1h"
)
```

## FastAPI Integration

The monitoring system integrates seamlessly with FastAPI applications:

```python
from fastapi import FastAPI
from utils.monitoring_integration import setup_monitoring

app = FastAPI(title="SutazAI API")

# Initialize and integrate monitoring
monitoring = setup_monitoring(
    app=app,
    system_id="sutazai_api",
    base_dir="/opt/sutazaiapp",
    enable_processing_monitoring=True,
    enable_ethics_monitoring=True,
    enable_self_mod_monitoring=True,
    enable_hardware_monitoring=True,
    enable_security_monitoring=True
)

# Your API routes go here
@app.get("/")
def read_root():
    return {"status": "ok"}
```

## Alerting

Alerts are configured in Prometheus and managed by AlertManager. Some critical alerts include:

- **Ethical Boundary Violations**: Immediate alerts for any ethical violations
- **Self-Modification Events**: Alerts for unauthorized self-modifications
- **Security Breaches**: Critical alerts for security violations
- **Processing Architecture Anomalies**: Warnings for unusual processing behavior
- **Resource Constraints**: Alerts for hardware limitations

## Dashboards

The system includes pre-configured Grafana dashboards:

- **automation system Overview Dashboard**: High-level overview of all metrics
- **Processing Architecture Dashboard**: Detailed view of processing metrics
- **Ethics and Safety Dashboard**: Ethics verification metrics
- **Self-Modification Dashboard**: System modification tracking
- **Hardware Optimization Dashboard**: Resource usage and efficiency
- **Security Dashboard**: System integrity and security metrics

## Troubleshooting

Common issues and solutions:

- **Prometheus not collecting metrics**: Ensure the target endpoints are accessible and correctly configured in `prometheus.yml`
- **AlertManager not sending notifications**: Check SMTP or Slack API configuration
- **Missing logs in Elasticsearch**: Verify Filebeat is running and configured to collect from the correct paths
- **Grafana dashboards not showing data**: Ensure Prometheus is correctly configured as a data source

## Security Considerations

- **Authentication**: All components have authentication enabled
- **Encryption**: TLS encryption for communication between components
- **Access Control**: Restricted access to monitoring interfaces
- **Audit Logging**: All access to monitoring components is logged

## Best Practices

- **Regular Backups**: Backup Prometheus and Elasticsearch data
- **Alert Tuning**: Regularly review and adjust alert thresholds
- **Log Rotation**: Configure log rotation to manage disk space
- **Dashboard Organization**: Organize dashboards by team or functionality
- **Documentation**: Keep documentation updated with new metrics and alerts
