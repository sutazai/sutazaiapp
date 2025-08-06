# Monitoring & Observability Stack Documentation

## Overview

The SutazAI system implements a comprehensive monitoring and observability stack based on Prometheus, Grafana, Loki, and associated tools. All components listed here are verified as running for 14+ hours in production.

## Stack Components Status

### Core Monitoring
- **Prometheus**: ✅ RUNNING 14+ hours (Port 10200)
- **Grafana**: ✅ RUNNING 15+ hours (Port 10201)
- **Loki**: ✅ RUNNING 15+ hours (Port 10202)
- **AlertManager**: ✅ RUNNING 15+ hours (Port 10203)

### Metric Exporters
- **Node Exporter**: ✅ RUNNING 15+ hours (Port 10220)
- **cAdvisor**: ✅ HEALTHY (Port 10221)
- **Blackbox Exporter**: ✅ RUNNING 15+ hours (Port 10229)
- **AI Metrics Exporter**: ❌ UNHEALTHY (Port 11068) - Needs fixing

### Log Collection
- **Promtail**: ✅ RUNNING 15+ hours (Log shipper to Loki)

## Prometheus Configuration

### Service Details
- **Port**: 10200
- **Docker Image**: `prom/prometheus:latest`
- **Container**: `sutazai-prometheus`
- **Config Path**: `/monitoring/prometheus/prometheus.yml`
- **Data Retention**: 15 days
- **Scrape Interval**: 15s

### Configuration File
```yaml
# prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s
  external_labels:
    cluster: 'sutazai-prod'
    environment: 'production'

# Alerting configuration
alerting:
  alertmanagers:
    - static_configs:
        - targets:
            - alertmanager:9093

# Rule files
rule_files:
  - "/etc/prometheus/alerts/*.yml"

# Scrape configurations
scrape_configs:
  # Prometheus self-monitoring
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # Node Exporter - System metrics
  - job_name: 'node'
    static_configs:
      - targets: ['node-exporter:9100']
    relabel_configs:
      - source_labels: [__address__]
        target_label: instance
        replacement: 'sutazai-host'

  # cAdvisor - Container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']

  # Backend API metrics
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:10010']
    metrics_path: '/metrics'

  # Agent metrics
  - job_name: 'agents'
    file_sd_configs:
      - files:
          - '/etc/prometheus/agents/*.json'
    relabel_configs:
      - source_labels: [__address__]
        target_label: __address__
        regex: '([^:]+):.*'
        replacement: '${1}:8080'

  # Database exporters
  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  # Blackbox probe for endpoint monitoring
  - job_name: 'blackbox'
    metrics_path: /probe
    params:
      module: [http_2xx]
    static_configs:
      - targets:
          - http://backend:10010/health
          - http://frontend:10011/health
          - http://ollama:11434/
    relabel_configs:
      - source_labels: [__address__]
        target_label: __param_target
      - source_labels: [__param_target]
        target_label: instance
      - target_label: __address__
        replacement: blackbox-exporter:9115
```

### Key Metrics Collected

#### System Metrics (Node Exporter)
```
# CPU Usage
rate(node_cpu_seconds_total[5m])

# Memory Usage
node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes

# Disk I/O
rate(node_disk_read_bytes_total[5m])
rate(node_disk_written_bytes_total[5m])

# Network Traffic
rate(node_network_receive_bytes_total[5m])
rate(node_network_transmit_bytes_total[5m])

# File System Usage
node_filesystem_avail_bytes / node_filesystem_size_bytes
```

#### Container Metrics (cAdvisor)
```
# Container CPU Usage
rate(container_cpu_usage_seconds_total[5m])

# Container Memory Usage
container_memory_usage_bytes

# Container Network I/O
rate(container_network_receive_bytes_total[5m])
rate(container_network_transmit_bytes_total[5m])

# Container Restart Count
container_restart_count
```

## Grafana Configuration

### Service Details
- **Port**: 10201
- **Docker Image**: `grafana/grafana:latest`
- **Container**: `sutazai-grafana`
- **Default User**: admin/admin (change on first login)
- **Config Path**: `/monitoring/grafana/`

### Pre-configured Dashboards

#### 1. System Overview Dashboard
```json
{
  "dashboard": {
    "title": "SutazAI System Overview",
    "panels": [
      {
        "title": "Service Health Status",
        "type": "stat",
        "targets": [
          {
            "expr": "up",
            "legendFormat": "{{job}}"
          }
        ]
      },
      {
        "title": "CPU Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "100 - (avg(irate(node_cpu_seconds_total{mode=\"idle\"}[5m])) * 100)"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100"
          }
        ]
      },
      {
        "title": "Container Count",
        "type": "stat",
        "targets": [
          {
            "expr": "count(container_last_seen)"
          }
        ]
      }
    ]
  }
}
```

#### 2. Agent Performance Dashboard
```json
{
  "dashboard": {
    "title": "AI Agent Performance",
    "panels": [
      {
        "title": "Agent Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(agent_requests_total[5m])",
            "legendFormat": "{{agent_name}}"
          }
        ]
      },
      {
        "title": "Agent Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, agent_processing_seconds_bucket)",
            "legendFormat": "p95 - {{agent_name}}"
          }
        ]
      },
      {
        "title": "Agent Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(agent_requests_total{status=\"error\"}[5m])"
          }
        ]
      }
    ]
  }
}
```

#### 3. Database Monitoring Dashboard
```json
{
  "dashboard": {
    "title": "Database Monitoring",
    "panels": [
      {
        "title": "PostgreSQL Connections",
        "type": "graph",
        "targets": [
          {
            "expr": "pg_stat_database_numbackends"
          }
        ]
      },
      {
        "title": "Redis Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "redis_memory_used_bytes"
          }
        ]
      },
      {
        "title": "Query Performance",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(pg_stat_database_xact_commit[5m])"
          }
        ]
      }
    ]
  }
}
```

### Data Sources Configuration
```yaml
# datasources.yml
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    
  - name: Loki
    type: loki
    access: proxy
    url: http://loki:3100
    
  - name: PostgreSQL
    type: postgres
    access: proxy
    url: postgres:5432
    database: sutazai
    user: sutazai
    secureJsonData:
      password: sutazai123
```

## Loki Log Aggregation

### Service Details
- **Port**: 10202 (HTTP) / 9095 (gRPC)
- **Docker Image**: `grafana/loki:latest`
- **Container**: `sutazai-loki`
- **Storage**: Local filesystem
- **Retention**: 7 days

### Configuration
```yaml
# loki-config.yml
auth_enabled: false

server:
  http_listen_port: 3100
  grpc_listen_port: 9095

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1
  chunk_idle_period: 5m
  chunk_retain_period: 30s

schema_config:
  configs:
    - from: 2023-01-01
      store: boltdb-shipper
      object_store: filesystem
      schema: v11
      index:
        prefix: index_
        period: 24h

storage_config:
  boltdb_shipper:
    active_index_directory: /loki/boltdb-shipper-active
    cache_location: /loki/boltdb-shipper-cache
    shared_store: filesystem
  filesystem:
    directory: /loki/chunks

limits_config:
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
  ingestion_rate_mb: 10
  ingestion_burst_size_mb: 20

chunk_store_config:
  max_look_back_period: 168h

table_manager:
  retention_deletes_enabled: true
  retention_period: 168h
```

### Log Queries in Grafana
```logql
# All container logs
{container_name=~".+"}

# Backend errors
{container_name="sutazai-backend"} |= "ERROR"

# Agent logs with processing time > 1s
{job="agents"} | json | processing_time > 1000

# Authentication failures
{container_name=~"sutazai-.*"} |= "authentication failed"

# Database connection errors
{container_name=~"sutazai-(postgres|redis|neo4j)"} |= "connection"
```

## Promtail Log Shipper

### Configuration
```yaml
# promtail-config.yml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # Docker container logs
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containers
          __path__: /var/lib/docker/containers/*/*-json.log
    pipeline_stages:
      - json:
          expressions:
            stream: stream
            time: time
            log: log
            container_name: container_name
            source: source
            tag: attrs.tag
      - regex:
          expression: '(?P<container_name>(?:[^|]*))'
          source: tag
      - timestamp:
          format: RFC3339Nano
          source: time
      - labels:
          stream:
          container_name:
      - output:
          source: log

  # System logs
  - job_name: syslog
    syslog:
      listen_address: 0.0.0.0:514
      labels:
        job: syslog
    relabel_configs:
      - source_labels: ['__syslog_message_hostname']
        target_label: 'host'
```

## AlertManager Configuration

### Service Details
- **Port**: 10203
- **Docker Image**: `prom/alertmanager:latest`
- **Container**: `sutazai-alertmanager`

### Alert Rules
```yaml
# alerts/system.yml
groups:
  - name: system
    rules:
      - alert: HighCPUUsage
        expr: (100 - (avg(irate(node_cpu_seconds_total{mode="idle"}[5m])) * 100)) > 80
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage detected"
          description: "CPU usage is above 80% (current value: {{ $value }}%)"

      - alert: HighMemoryUsage
        expr: (1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100 > 90
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage detected"
          description: "Memory usage is above 90% (current value: {{ $value }}%)"

      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"}) * 100 < 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Disk space is below 10% (current value: {{ $value }}%)"

      - alert: ServiceDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service is down"
          description: "{{ $labels.job }} is down"

  - name: agents
    rules:
      - alert: AgentHighErrorRate
        expr: rate(agent_requests_total{status="error"}[5m]) > 0.1
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High agent error rate"
          description: "Agent {{ $labels.agent_name }} error rate is above 10%"

      - alert: AgentSlowResponse
        expr: histogram_quantile(0.95, agent_processing_seconds_bucket) > 5
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Slow agent response"
          description: "Agent {{ $labels.agent_name }} p95 response time > 5s"

  - name: database
    rules:
      - alert: PostgreSQLDown
        expr: pg_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL is down"

      - alert: RedisDown
        expr: redis_up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Redis is down"

      - alert: HighDatabaseConnections
        expr: pg_stat_database_numbackends > 150
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High database connection count"
```

### AlertManager Configuration
```yaml
# alertmanager.yml
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'default'
  
  routes:
    - match:
        severity: critical
      receiver: 'critical'
      continue: true
    
    - match:
        severity: warning
      receiver: 'warning'

receivers:
  - name: 'default'
    webhook_configs:
      - url: 'http://backend:10010/api/v1/alerts'
        send_resolved: true

  - name: 'critical'
    webhook_configs:
      - url: 'http://backend:10010/api/v1/alerts/critical'
        send_resolved: true

  - name: 'warning'
    webhook_configs:
      - url: 'http://backend:10010/api/v1/alerts/warning'
        send_resolved: true

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'cluster', 'service']
```

## Custom Metrics Implementation

### Backend Metrics
```python
# backend/app/metrics.py
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response

# Define metrics
http_requests_total = Counter(
    'http_requests_total',
    'Total HTTP requests',
    ['method', 'endpoint', 'status']
)

http_request_duration = Histogram(
    'http_request_duration_seconds',
    'HTTP request duration',
    ['method', 'endpoint']
)

active_connections = Gauge(
    'active_connections',
    'Number of active connections'
)

# Middleware for automatic metrics
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    method = request.method
    endpoint = request.url.path
    
    active_connections.inc()
    
    with http_request_duration.labels(
        method=method,
        endpoint=endpoint
    ).time():
        response = await call_next(request)
    
    http_requests_total.labels(
        method=method,
        endpoint=endpoint,
        status=response.status_code
    ).inc()
    
    active_connections.dec()
    
    return response

# Metrics endpoint
@app.get("/metrics")
async def get_metrics():
    return Response(
        content=generate_latest(),
        media_type="text/plain"
    )
```

### Agent Metrics
```python
# agents/base_agent_metrics.py
from prometheus_client import Counter, Histogram, Gauge, Info

# Agent info
agent_info = Info(
    'agent_info',
    'Agent information'
)

# Processing metrics
agent_tasks_total = Counter(
    'agent_tasks_total',
    'Total tasks processed',
    ['agent_name', 'task_type', 'status']
)

agent_task_duration = Histogram(
    'agent_task_duration_seconds',
    'Task processing duration',
    ['agent_name', 'task_type'],
    buckets=[0.1, 0.5, 1.0, 2.5, 5.0, 10.0]
)

agent_active_tasks = Gauge(
    'agent_active_tasks',
    'Currently processing tasks',
    ['agent_name']
)

# Model metrics
model_inference_duration = Histogram(
    'model_inference_duration_seconds',
    'Model inference time',
    ['model_name']
)

model_token_usage = Counter(
    'model_token_usage_total',
    'Total tokens used',
    ['model_name', 'type']  # type: prompt/completion
)
```

## Monitoring Best Practices

### 1. Alert Fatigue Prevention
```yaml
# Only alert on actionable items
- Avoid alerting on normal fluctuations
- Set appropriate thresholds based on baselines
- Use alert grouping and inhibition rules
- Implement proper alert routing
```

### 2. Dashboard Design
```yaml
# Effective dashboard principles
- Start with high-level overview
- Allow drill-down into details
- Use consistent color schemes
- Include relevant time ranges
- Add helpful annotations
```

### 3. Log Structured Data
```python
# Structured logging example
import structlog

logger = structlog.get_logger()

logger.info(
    "task_processed",
    agent_name="researcher",
    task_id="123",
    duration_ms=1250,
    status="success"
)
```

### 4. Metric Naming Conventions
```
# Follow Prometheus naming conventions
<metric_name>_<unit>_<aggregation>

Examples:
- http_requests_total (counter)
- http_request_duration_seconds (histogram)
- memory_usage_bytes (gauge)
- agent_info (info)
```

## Troubleshooting

### Common Issues

#### 1. Prometheus Not Scraping Targets
```bash
# Check target status
curl http://localhost:10200/api/v1/targets

# Verify network connectivity
docker exec sutazai-prometheus curl http://backend:10010/metrics

# Check Prometheus logs
docker logs sutazai-prometheus
```

#### 2. Grafana Dashboard Not Loading
```bash
# Check data source configuration
curl -u admin:admin http://localhost:10201/api/datasources

# Test query directly in Prometheus
curl http://localhost:10200/api/v1/query?query=up

# Verify Grafana logs
docker logs sutazai-grafana
```

#### 3. Loki Not Receiving Logs
```bash
# Check Promtail status
docker logs sutazai-promtail

# Verify Loki is accessible
curl http://localhost:10202/ready

# Test manual log push
curl -X POST http://localhost:10202/loki/api/v1/push \
  -H "Content-Type: application/json" \
  -d '{"streams": [{"stream": {"job": "test"}, "values": [["'$(date +%s)'000000000", "test log"]]}]}'
```

#### 4. Alerts Not Firing
```bash
# Check alert rules
curl http://localhost:10200/api/v1/rules

# Verify AlertManager status
curl http://localhost:10203/api/v1/status

# Test alert manually
curl -X POST http://localhost:10203/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{"labels": {"alertname": "test", "severity": "warning"}}]'
```

## Performance Optimization

### 1. Prometheus Optimization
```yaml
# Reduce cardinality
- Limit label values
- Use recording rules for expensive queries
- Adjust scrape intervals appropriately
- Implement proper retention policies
```

### 2. Grafana Optimization
```yaml
# Dashboard performance
- Use query caching
- Limit time ranges
- Optimize panel queries
- Use variables for reusability
```

### 3. Loki Optimization
```yaml
# Log ingestion optimization
- Batch log entries
- Compress logs before sending
- Use appropriate chunk sizes
- Implement log sampling for high-volume sources
```

## Backup and Recovery

### Prometheus Data Backup
```bash
# Snapshot creation
curl -X POST http://localhost:10200/api/v1/admin/tsdb/snapshot

# Backup location
docker cp sutazai-prometheus:/prometheus/snapshots ./prometheus-backup
```

### Grafana Backup
```bash
# Export dashboards
curl -u admin:admin http://localhost:10201/api/dashboards/home | jq '.dashboards' > dashboards.json

# Export data sources
curl -u admin:admin http://localhost:10201/api/datasources > datasources.json

# Backup SQLite database
docker cp sutazai-grafana:/var/lib/grafana/grafana.db ./grafana.db
```

### Loki Backup
```bash
# Stop Loki for consistent backup
docker stop sutazai-loki

# Backup chunks and index
docker cp sutazai-loki:/loki ./loki-backup

# Restart Loki
docker start sutazai-loki
```

## Integration with CI/CD

### Prometheus Metrics in CI/CD
```yaml
# .gitlab-ci.yml example
test:
  script:
    - make test
    - curl -X POST http://prometheus-pushgateway:9091/metrics/job/ci-pipeline/instance/$CI_PIPELINE_ID \
        --data-binary @- << EOF
        ci_pipeline_duration_seconds{project="$CI_PROJECT_NAME",branch="$CI_COMMIT_REF_NAME"} $CI_PIPELINE_DURATION
        ci_pipeline_status{project="$CI_PROJECT_NAME",branch="$CI_COMMIT_REF_NAME",status="success"} 1
        EOF
```

### Deployment Annotations
```python
# Add deployment markers to Grafana
import requests
from datetime import datetime

def annotate_deployment(version, environment):
    annotation = {
        "dashboardId": 1,
        "time": int(datetime.now().timestamp() * 1000),
        "tags": ["deployment", environment],
        "text": f"Deployed version {version}"
    }
    
    response = requests.post(
        "http://localhost:10201/api/annotations",
        json=annotation,
        auth=("admin", "admin")
    )
    return response.json()
```

## Monitoring Stack Management

### Health Check Script
```bash
#!/bin/bash
# check-monitoring-health.sh

echo "Checking Monitoring Stack Health..."

services=(
    "prometheus:10200/-/healthy"
    "grafana:10201/api/health"
    "loki:10202/ready"
    "alertmanager:10203/api/v1/status"
    "node-exporter:10220/metrics"
    "cadvisor:10221/metrics"
)

for service in "${services[@]}"; do
    IFS=':' read -r name port_path <<< "$service"
    if curl -f "http://localhost:${port_path}" > /dev/null 2>&1; then
        echo "✅ $name is healthy"
    else
        echo "❌ $name is unhealthy"
    fi
done
```

### Stack Restart Procedure
```bash
# Graceful restart of monitoring stack
docker-compose stop prometheus grafana loki alertmanager promtail
docker-compose up -d prometheus grafana loki alertmanager promtail

# Wait for services to be healthy
sleep 30
./check-monitoring-health.sh
```