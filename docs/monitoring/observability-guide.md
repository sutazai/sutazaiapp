# Comprehensive Monitoring & Observability Guide

**Version:** 1.0  
**Last Updated:** August 8, 2025  
**System:** SutazAI Distributed AI Platform

This guide provides comprehensive documentation for the monitoring and observability infrastructure of the SutazAI system, based on the current production deployment with Prometheus, Grafana, Loki, and AlertManager.

## Table of Contents

1. [Monitoring Architecture Overview](#1-monitoring-architecture-overview)
2. [Metrics Collection](#2-metrics-collection)
3. [Logging Infrastructure](#3-logging-infrastructure)
4. [Visualization with Grafana](#4-visualization-with-grafana)
5. [Alerting Strategy](#5-alerting-strategy)
6. [Service-Specific Monitoring](#6-service-specific-monitoring)
7. [Performance Monitoring](#7-performance-monitoring)
8. [Health Checks](#8-health-checks)
9. [Distributed Tracing (Future)](#9-distributed-tracing-future)
10. [Observability Best Practices](#10-observability-best-practices)
11. [Troubleshooting with Observability](#11-troubleshooting-with-observability)
12. [Operational Dashboards](#12-operational-dashboards)

## 1. Monitoring Architecture Overview

### Current Monitoring Stack

The SutazAI system implements a comprehensive monitoring stack with the following components:

| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| Prometheus | 10200 | Metrics collection and storage | ✅ Operational |
| Grafana | 10201 | Visualization dashboards | ✅ Operational |
| Loki | 10202 | Log aggregation and storage | ✅ Operational |
| AlertManager | 10203 | Alert routing and management | ✅ Operational |
| Node Exporter | 10220 | System metrics collection | ✅ Operational |
| cAdvisor | 10221 | Container metrics collection | ✅ Operational |
| Blackbox Exporter | 10229 | Synthetic monitoring | ✅ Operational |
| Promtail | N/A | Log forwarding to Loki | ✅ Operational |

### Data Flow Architecture

```mermaid
graph TD
    A[Applications] --> B[Metrics Endpoints]
    A --> C[Log Files]
    B --> D[Prometheus]
    C --> E[Promtail]
    E --> F[Loki]
    D --> G[Grafana]
    F --> G
    D --> H[AlertManager]
    H --> I[Notification Channels]
    
    J[Node Exporter] --> D
    K[cAdvisor] --> D
    L[Blackbox Exporter] --> D
```

### Retention Policies

- **Prometheus**: 7 days (configurable via `--storage.tsdb.retention.time=7d`)
- **Loki**: 168 hours (7 days) for log retention
- **Grafana**: Persistent dashboards and configurations
- **AlertManager**: 120 hours alert history

### Service Discovery Configuration

The monitoring stack uses Docker service discovery for automatic target detection:

```yaml
# Prometheus service discovery
- job_name: 'docker-containers'
  docker_sd_configs:
    - host: unix:///var/run/docker.sock
      refresh_interval: 15s
  relabel_configs:
    - source_labels: ['__meta_docker_container_name']
      target_label: 'container'
```

### Monitoring Coverage Gaps

Current gaps that need attention:

1. **Agent Services**: 7 stub agents (ports 8002, 8551, 8587, 8588, 8589, 11015, 11063) only provide health endpoints
2. **Database Query Metrics**: Limited PostgreSQL query performance monitoring
3. **Vector Database Metrics**: Qdrant, ChromaDB, and FAISS lack comprehensive metrics
4. **Distributed Tracing**: Not yet implemented (Jaeger planned)
5. **Business Metrics**: Application-specific KPIs not tracked

## 2. Metrics Collection

### Prometheus Configuration

The main Prometheus configuration (`/monitoring/prometheus/prometheus.yml`) includes:

```yaml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"
  - "sutazai_critical_alerts.yml"
  - "ollama-agent-alerts.yml"

scrape_configs:
  # Core Infrastructure
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  # System Metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['sutazai-node-exporter:9100']

  # Container Metrics  
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['sutazai-cadvisor:8080']

  # Application Services
  - job_name: 'backend'
    static_configs:
      - targets: ['sutazai-backend:8000']
    metrics_path: '/metrics'

  # Agent Services (Health Endpoints Only)
  - job_name: 'ai-agent-orchestrator'
    static_configs:
      - targets: ['sutazai-ai-agent-orchestrator:8589']
    metrics_path: '/health'
    scrape_interval: 30s

  # Database Metrics
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['sutazai-postgres-exporter:9187']
```

### Service-Specific Metrics

#### Backend API (FastAPI) Metrics

```python
# Key metrics to monitor:
- http_requests_total{method, endpoint, status}
- http_request_duration_seconds{method, endpoint}
- http_requests_in_progress{method, endpoint}
- ollama_connection_status{model}
- database_connections_active
- redis_operations_total{operation}
```

Example Prometheus query for API response time:
```promql
histogram_quantile(0.95, 
  sum(rate(http_request_duration_seconds_bucket[5m])) 
  by (le, endpoint)
)
```

#### Agent Service Metrics (Limited)

Currently, agents only expose:
```python
# Available metrics (stub services):
- agent_health_status{agent_name}
- agent_last_heartbeat{agent_name}

# Missing metrics (to be implemented):
- agent_task_processing_duration
- agent_task_queue_size
- agent_error_rate
- agent_resource_utilization
```

#### Database Metrics

**PostgreSQL (via postgres-exporter):**
```promql
# Connection metrics
pg_stat_database_numbackends{datname="sutazai"}

# Query performance
pg_stat_statements_mean_time{user, datname}

# Database size
pg_database_size_bytes{datname="sutazai"}
```

**Redis:**
```promql
# Memory usage
redis_memory_used_bytes

# Commands per second
rate(redis_commands_total[1m])

# Connected clients
redis_connected_clients
```

**Neo4j:**
```promql
# Node/relationship counts
neo4j_database_nodes_count
neo4j_database_relationships_count

# Query performance
neo4j_database_query_execution_time
```

### Custom Metrics Implementation

For implementing custom metrics in Python services:

```python
from prometheus_client import Counter, Histogram, Gauge, generate_latest

# Define metrics
REQUEST_COUNT = Counter('app_requests_total', 'Total requests', ['method', 'endpoint'])
REQUEST_LATENCY = Histogram('app_request_duration_seconds', 'Request latency')
ACTIVE_CONNECTIONS = Gauge('app_active_connections', 'Active connections')

# Usage example
@REQUEST_LATENCY.time()
@REQUEST_COUNT.labels(method='POST', endpoint='/api/v1/task').inc()
def process_task():
    # Application logic
    pass
```

### Metric Naming Conventions

Follow Prometheus best practices:

- Use snake_case: `sutazai_agent_tasks_total`
- Include units: `_seconds`, `_bytes`, `_total`
- Be descriptive: `ollama_model_inference_duration_seconds`
- Use consistent prefixes: `sutazai_*`, `agent_*`, `ollama_*`

### Cardinality Management

Current high-cardinality metrics to monitor:

```promql
# Check cardinality
topk(10, count by (__name__)({__name__=~".+"}))

# Monitor series growth
prometheus_tsdb_symbol_table_size_bytes
prometheus_tsdb_head_series
```

**Guidelines:**
- Limit label combinations < 1M series
- Use recording rules for complex queries
- Avoid user IDs or UUIDs as labels

## 3. Logging Infrastructure

### Loki Configuration

Loki configuration (`/monitoring/loki/config.yml`):

```yaml
server:
  http_listen_port: 3100
  grpc_listen_port: 9096

ingester:
  lifecycler:
    address: 127.0.0.1
    ring:
      kvstore:
        store: inmemory
      replication_factor: 1

schema_config:
  configs:
    - from: 2020-10-24
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
  retention_period: 168h  # 7 days
  enforce_metric_name: false
  reject_old_samples: true
  reject_old_samples_max_age: 168h
```

### Log Aggregation Patterns

#### Promtail Configuration

Promtail collects logs from Docker containers:

```yaml
server:
  http_listen_port: 9080
  grpc_listen_port: 0

positions:
  filename: /tmp/positions.yaml

clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  - job_name: containers
    docker_sd_configs:
      - host: unix:///var/run/docker.sock
        refresh_interval: 5s
    relabel_configs:
      - source_labels: ['__meta_docker_container_name']
        target_label: 'container'
      - source_labels: ['__meta_docker_container_log_stream']
        target_label: 'logstream'
      - source_labels: ['__meta_docker_container_label_com_docker_compose_service']
        target_label: 'service'
```

### Structured Logging Formats

Recommended log format for all services:

```json
{
  "timestamp": "2025-08-08T10:15:30Z",
  "level": "INFO",
  "service": "sutazai-backend",
  "component": "ollama_integration",
  "message": "Model inference completed",
  "duration_ms": 1250,
  "model": "tinyllama",
  "tokens_generated": 45,
  "request_id": "req_123456",
  "user_id": "user_789"
}
```

Python logging setup:
```python
import logging
import json
from datetime import datetime

class StructuredLogger:
    def __init__(self, service_name):
        self.service = service_name
        self.logger = logging.getLogger(service_name)
        
    def info(self, message, **kwargs):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": "INFO",
            "service": self.service,
            "message": message,
            **kwargs
        }
        self.logger.info(json.dumps(log_data))
```

### Log Analysis Queries

Common LogQL queries for troubleshooting:

```logql
# Error rate by service
sum(rate({service=~".+"} |= "ERROR" [5m])) by (service)

# Slow requests (>1s)
{service="sutazai-backend"} | json | duration_ms > 1000

# Authentication failures
{service="sutazai-backend"} |= "authentication" |= "failed"

# Agent health issues
{container=~"sutazai-.*-agent"} |= "unhealthy"

# Database connection errors
{service=~"sutazai.*"} |= "database" |= "connection" |= "error"
```

### Log Retention Policies

- **Application logs**: 7 days (168h)
- **System logs**: 14 days
- **Security logs**: 30 days
- **Audit logs**: 90 days (when implemented)

## 4. Visualization with Grafana

### Dashboard Organization

The Grafana dashboard structure follows role-based organization:

```
├── Executive/
│   └── executive-overview.json
├── Operations/
│   └── operations-overview.json
├── Developer/
│   └── developer-overview.json
├── Infrastructure/
│   └── infrastructure-overview.json
├── Security/
│   └── security-overview.json
├── Business/
│   └── business-metrics.json
└── AI Models/
    └── ai-performance-dashboard.json
```

### Key Dashboards to Create

#### 1. System Overview Dashboard

**Panel Structure:**
- Service Health Status (Single Stat)
- Request Rate (Graph)
- Response Time (Graph)
- Error Rate (Graph)
- Resource Utilization (Graph)

**Key Queries:**
```promql
# Service health
up{job=~"sutazai.*"}

# Request rate
sum(rate(http_requests_total[5m])) by (service)

# 95th percentile response time
histogram_quantile(0.95, 
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
)

# Error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) / 
sum(rate(http_requests_total[5m])) * 100
```

#### 2. Service Health Dashboard

```json
{
  "dashboard": {
    "title": "SutazAI Service Health",
    "panels": [
      {
        "title": "Service Uptime",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=~\"sutazai.*\"}",
            "legendFormat": "{{instance}}"
          }
        ]
      },
      {
        "title": "Response Times",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, job))",
            "legendFormat": "50th percentile"
          },
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, job))",
            "legendFormat": "95th percentile"
          }
        ]
      }
    ]
  }
}
```

#### 3. Agent Performance Dashboard

**Focus Areas:**
- Agent health status (7 stub agents)
- Task processing metrics (when implemented)
- Resource utilization per agent
- Inter-agent communication latency (when implemented)

```promql
# Agent health status
up{job=~".*agent.*"}

# Agent response time (health checks)
probe_duration_seconds{job=~".*agent.*"}

# Agent memory usage
container_memory_usage_bytes{name=~"sutazai-.*-agent"}
```

#### 4. Database Metrics Dashboard

**PostgreSQL Panel:**
```promql
# Active connections
pg_stat_database_numbackends{datname="sutazai"}

# Query duration
pg_stat_statements_mean_time

# Database size growth
increase(pg_database_size_bytes{datname="sutazai"}[1h])
```

**Redis Panel:**
```promql
# Memory usage
redis_memory_used_bytes

# Hit ratio
redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) * 100

# Commands per second
rate(redis_commands_total[1m])
```

#### 5. API Performance Dashboard

```promql
# Request volume by endpoint
sum(rate(http_requests_total[5m])) by (endpoint)

# Error rate by endpoint
sum(rate(http_requests_total{status=~"5.."}[5m])) by (endpoint) /
sum(rate(http_requests_total[5m])) by (endpoint) * 100

# Slowest endpoints
topk(10, 
  histogram_quantile(0.95, 
    sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint)
  )
)
```

### Query Optimization

**Recording Rules** for frequently used queries (`/monitoring/prometheus/rules/`):

```yaml
groups:
  - name: sutazai_aggregations
    interval: 30s
    rules:
      - record: sutazai:request_rate_5m
        expr: sum(rate(http_requests_total[5m])) by (service, endpoint)
        
      - record: sutazai:error_rate_5m
        expr: |
          sum(rate(http_requests_total{status=~"5.."}[5m])) by (service) /
          sum(rate(http_requests_total[5m])) by (service) * 100
          
      - record: sutazai:response_time_95p
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
          )
```

### Variable Usage

Dashboard variables for dynamic filtering:

```json
{
  "templating": {
    "list": [
      {
        "name": "service",
        "type": "query",
        "query": "label_values(up{job=~\"sutazai.*\"}, job)",
        "multi": true,
        "includeAll": true
      },
      {
        "name": "time_range", 
        "type": "interval",
        "query": "5m,15m,1h,6h,1d",
        "current": "5m"
      }
    ]
  }
}
```

## 5. Alerting Strategy

### AlertManager Configuration

Main configuration (`/monitoring/alertmanager/alertmanager.yml`):

```yaml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@sutazai.com'

route:
  group_by: ['alertname', 'cluster', 'service']
  group_wait: 30s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical-alerts'
      group_wait: 10s
      repeat_interval: 5m
    - match:
        severity: warning
      receiver: 'warning-alerts'
      repeat_interval: 1h

receivers:
  - name: 'default'
    email_configs:
      - to: 'ops-team@sutazai.com'
        subject: '[SUTAZAI] {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          Severity: {{ .Labels.severity }}
          Service: {{ .Labels.service }}
          {{ end }}

  - name: 'critical-alerts'
    email_configs:
      - to: 'oncall@sutazai.com'
        subject: '[CRITICAL] {{ .GroupLabels.alertname }}'
    webhook_configs:
      - url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'

  - name: 'warning-alerts'
    email_configs:
      - to: 'monitoring@sutazai.com'
        subject: '[WARNING] {{ .GroupLabels.alertname }}'

inhibit_rules:
  - source_match:
      severity: 'critical'
    target_match:
      severity: 'warning'
    equal: ['alertname', 'service']
```

### Alert Rules by Severity

#### Critical Alerts (`/monitoring/prometheus/sutazai_critical_alerts.yml`)

```yaml
groups:
  - name: sutazai.critical
    rules:
      - alert: ServiceDown
        expr: up{job=~"sutazai.*"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "Service {{ $labels.job }} has been down for more than 1 minute"

      - alert: HighErrorRate
        expr: |
          (
            sum(rate(http_requests_total{status=~"5.."}[5m])) by (service) /
            sum(rate(http_requests_total[5m])) by (service) * 100
          ) > 10
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "High error rate on {{ $labels.service }}"
          description: "Error rate is {{ $value }}% for service {{ $labels.service }}"

      - alert: DatabaseConnectionFailure
        expr: pg_up != 1
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL connection failure"
          description: "Unable to connect to PostgreSQL database"

      - alert: OllamaModelUnavailable
        expr: up{job="ollama"} == 0
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "Ollama service unavailable"
          description: "Ollama LLM service has been unavailable for more than 3 minutes"
```

#### Warning Alerts

```yaml
  - name: sutazai.warnings
    rules:
      - alert: HighResponseTime
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service)
          ) > 2
        for: 10m
        labels:
          severity: warning
        annotations:
          summary: "High response time for {{ $labels.service }}"
          description: "95th percentile response time is {{ $value }}s"

      - alert: HighMemoryUsage
        expr: |
          (
            container_memory_usage_bytes{name=~"sutazai.*"} /
            container_spec_memory_limit_bytes{name=~"sutazai.*"} * 100
          ) > 80
        for: 15m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage for {{ $labels.name }}"
          description: "Memory usage is {{ $value }}% for container {{ $labels.name }}"

      - alert: AgentHealthCheckFailing
        expr: probe_success{job=~".*agent.*"} == 0
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Agent health check failing for {{ $labels.instance }}"
          description: "Agent {{ $labels.instance }} health check has been failing for 5 minutes"
```

### Notification Channels

**Slack Integration:**
```yaml
slack_configs:
  - api_url: 'https://hooks.slack.com/services/YOUR/SLACK/WEBHOOK'
    channel: '#sutazai-alerts'
    title: '{{ .GroupLabels.alertname }}'
    text: |
      {{ range .Alerts }}
      *Alert:* {{ .Annotations.summary }}
      *Description:* {{ .Annotations.description }}
      *Severity:* {{ .Labels.severity }}
      *Service:* {{ .Labels.service }}
      {{ end }}
```

**PagerDuty Integration:**
```yaml
pagerduty_configs:
  - routing_key: 'YOUR_PAGERDUTY_INTEGRATION_KEY'
    description: '{{ .GroupLabels.alertname }}: {{ .GroupLabels.service }}'
```

### Alert Fatigue Prevention

1. **Alert Grouping**: Group related alerts by service and timeframe
2. **Alert Suppression**: Use inhibit rules to suppress lower-severity alerts
3. **Rate Limiting**: Set appropriate repeat intervals (5m for critical, 1h for warnings)
4. **Alert Tuning**: Regular review and adjustment of thresholds

### Escalation Policies

```yaml
# Escalation matrix
Critical Alerts:
  - 0-5 min: Slack notification
  - 5-15 min: Email to on-call engineer
  - 15-30 min: PagerDuty escalation
  - 30+ min: Manager notification

Warning Alerts:  
  - 0-15 min: Slack notification
  - 15+ min: Email to team lead

Info Alerts:
  - Slack notification only
```

## 6. Service-Specific Monitoring

### Backend API (FastAPI) Monitoring

**Key Metrics to Track:**
```python
# In backend service (/backend/app/main.py)
from prometheus_client import Counter, Histogram, Gauge

# Request metrics
http_requests_total = Counter('http_requests_total', 
                             'Total HTTP requests', 
                             ['method', 'endpoint', 'status'])

http_request_duration = Histogram('http_request_duration_seconds',
                                 'HTTP request duration',
                                 ['method', 'endpoint'])

# Ollama integration metrics  
ollama_requests_total = Counter('ollama_requests_total',
                               'Total Ollama requests',
                               ['model', 'status'])

ollama_response_time = Histogram('ollama_response_time_seconds',
                                'Ollama response time',
                                ['model'])

# Database metrics
db_connections_active = Gauge('db_connections_active', 'Active DB connections')
```

**Health Check Endpoint:**
```python
@app.get("/health")
async def health_check():
    health_status = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "database": await check_database_health(),
            "redis": await check_redis_health(),
            "ollama": await check_ollama_health(),
        }
    }
    return health_status
```

### Agent Service Monitoring (Current Limitations)

**Current State (7 Stub Agents):**

| Agent | Port | Current Monitoring |
|-------|------|--------------------|
| AI Agent Orchestrator | 8589 | Health endpoint only |
| Multi-Agent Coordinator | 8587 | Health endpoint only |
| Resource Arbitration | 8588 | Health endpoint only |
| Task Assignment | 8551 | Health endpoint only |
| Hardware Optimizer | 8002 | Health endpoint only |
| Ollama Integration | 11015 | Health endpoint only |
| AI Metrics Exporter | 11063 | UNHEALTHY |

**Monitoring Gaps:**
```python
# Missing metrics (to be implemented when agents become functional):
agent_tasks_processed_total = Counter('agent_tasks_processed_total',
                                     'Tasks processed by agent',
                                     ['agent_name', 'task_type', 'status'])

agent_queue_size = Gauge('agent_queue_size', 'Agent task queue size', ['agent_name'])

agent_processing_duration = Histogram('agent_processing_duration_seconds',
                                     'Agent task processing time',
                                     ['agent_name', 'task_type'])

agent_resource_usage = Gauge('agent_resource_usage_percent',
                            'Agent resource utilization',
                            ['agent_name', 'resource_type'])
```

### Database Monitoring (PostgreSQL, Redis, Neo4j)

#### PostgreSQL Monitoring

**Via postgres-exporter (Port 10207):**
```promql
# Connection monitoring
pg_stat_database_numbackends{datname="sutazai"}

# Query performance  
pg_stat_statements_mean_time{user, datname}

# Database size
pg_database_size_bytes{datname="sutazai"}

# Lock monitoring
pg_locks_count{mode, locktype}

# Replication lag (if applicable)
pg_stat_replication_lag{application_name}
```

**Critical Alerts:**
```yaml
- alert: PostgreSQLDown
  expr: pg_up != 1
  for: 1m
  
- alert: PostgreSQLTooManyConnections  
  expr: pg_stat_database_numbackends > 80
  for: 5m

- alert: PostgreSQLSlowQueries
  expr: pg_stat_statements_mean_time > 1000  # 1 second
  for: 10m
```

#### Redis Monitoring

```promql
# Memory usage
redis_memory_used_bytes / redis_memory_max_bytes * 100

# Command rate
rate(redis_commands_total[1m])

# Hit ratio
redis_keyspace_hits_total / (redis_keyspace_hits_total + redis_keyspace_misses_total) * 100

# Connected clients
redis_connected_clients
```

#### Neo4j Monitoring

```promql
# Database metrics
neo4j_database_nodes_count
neo4j_database_relationships_count

# Query performance
neo4j_database_query_execution_time

# Memory usage
neo4j_memory_heap_used / neo4j_memory_heap_max * 100
```

### Vector Database Monitoring

**Current Integration Status:**
- **Qdrant** (Ports 10101/10102): ✅ Running, not integrated
- **FAISS** (Port 10103): ✅ Running, not integrated  
- **ChromaDB** (Port 10100): ⚠️ Connection issues

**Monitoring Implementation Needed:**
```python
# Vector database metrics (to be implemented)
vector_db_queries_total = Counter('vector_db_queries_total',
                                 'Vector database queries',
                                 ['db_type', 'operation'])

vector_db_query_duration = Histogram('vector_db_query_duration_seconds',
                                    'Vector query duration',
                                    ['db_type', 'operation'])

vector_db_collection_size = Gauge('vector_db_collection_size',
                                 'Vector collection size',
                                 ['db_type', 'collection'])
```

### Infrastructure Services Monitoring

#### Kong Gateway (Port 10005)

**Currently Not Configured** - Routes need to be defined.

Monitoring when active:
```promql
# Request rate through Kong
kong_http_requests_total{service, route}

# Response times
kong_request_latency{service, route}

# Upstream health
kong_upstream_target_health{upstream, target}
```

#### Consul (Port 10006)

```promql
# Service health
consul_health_service_status{service_name}

# Node health
consul_health_node_status{node}

# Catalog size
consul_catalog_services_total
```

#### RabbitMQ (Ports 10007/10008)

```promql
# Queue metrics
rabbitmq_queue_messages{queue}

# Consumer metrics  
rabbitmq_queue_consumers{queue}

# Message rate
rate(rabbitmq_queue_messages_published_total[1m])
```

## 7. Performance Monitoring

### Response Time Tracking

**SLI Definition:**
- **Target**: 95% of API requests < 1 second
- **Budget**: 99.9% uptime (43.2 minutes downtime/month)

**Key Queries:**
```promql
# API response time percentiles
histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.90, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))  
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))

# SLI compliance
(
  histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le)) < 1
) * 100
```

### Throughput Metrics

```promql
# Requests per second
sum(rate(http_requests_total[1m]))

# Requests per minute by endpoint
sum(rate(http_requests_total[1m]) * 60) by (endpoint)

# Peak throughput (max over 24h)
max_over_time(sum(rate(http_requests_total[5m]))[24h:])
```

### Error Rate Monitoring

```promql
# Overall error rate
sum(rate(http_requests_total{status=~"5.."}[5m])) / 
sum(rate(http_requests_total[5m])) * 100

# Error rate by endpoint
sum(rate(http_requests_total{status=~"5.."}[5m])) by (endpoint) /
sum(rate(http_requests_total[5m])) by (endpoint) * 100

# Error budget burn rate
(
  1 - (
    1 - sum(rate(http_requests_total{status=~"5.."}[30d])) / 
        sum(rate(http_requests_total[30d]))
  ) / 0.999  # 99.9% SLO
) * 100
```

### Saturation Metrics

#### CPU Saturation
```promql
# System load average
node_load1 / count(node_cpu_seconds_total{mode="idle"}) by (instance)

# CPU utilization by container
rate(container_cpu_usage_seconds_total[5m]) * 100

# CPU throttling
rate(container_cpu_cfs_throttled_seconds_total[5m])
```

#### Memory Saturation
```promql
# Memory usage percentage
(1 - (node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)) * 100

# Container memory usage
container_memory_usage_bytes / container_spec_memory_limit_bytes * 100

# Memory pressure (OOM kills)
increase(node_vmstat_oom_kill[1h])
```

#### Disk Saturation
```promql
# Disk usage percentage
(1 - (node_filesystem_avail_bytes / node_filesystem_size_bytes)) * 100

# Disk I/O utilization
rate(node_disk_io_time_seconds_total[5m])

# Disk I/O wait time
rate(node_cpu_seconds_total{mode="iowait"}[5m])
```

### Resource Utilization

**Recording Rules for Performance Metrics:**
```yaml
groups:
  - name: performance_slis
    interval: 30s
    rules:
      - record: sutazai:sli_availability
        expr: |
          (sum(rate(http_requests_total{status!~"5.."}[5m])) /
           sum(rate(http_requests_total[5m]))) * 100
           
      - record: sutazai:sli_latency  
        expr: |
          histogram_quantile(0.95,
            sum(rate(http_request_duration_seconds_bucket[5m])) by (le)
          )
          
      - record: sutazai:sli_error_rate
        expr: |
          (sum(rate(http_requests_total{status=~"5.."}[5m])) /
           sum(rate(http_requests_total[5m]))) * 100
```

## 8. Health Checks

### Service Health Endpoints

**Current Implementation Status:**

| Service | Endpoint | Status | Response Format |
|---------|----------|--------|-----------------|
| Backend API | `GET /health` | ✅ Working | JSON with service dependencies |
| Agent Services | `GET /health` | ⚠️ Stub only | `{"status": "healthy"}` |
| Frontend | `GET /health` | ⚠️ Not implemented | N/A |

### Dependency Health Checks

**Backend Health Check Example:**
```python
@app.get("/health", response_model=HealthResponse)
async def health_check():
    health_data = {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "17.0.0",
        "services": {}
    }
    
    # Check database
    try:
        await check_database_connection()
        health_data["services"]["database"] = {"status": "healthy", "response_time_ms": 15}
    except Exception as e:
        health_data["services"]["database"] = {"status": "unhealthy", "error": str(e)}
        health_data["status"] = "degraded"
    
    # Check Redis
    try:
        await check_redis_connection()
        health_data["services"]["redis"] = {"status": "healthy", "response_time_ms": 5}
    except Exception as e:
        health_data["services"]["redis"] = {"status": "unhealthy", "error": str(e)}
        health_data["status"] = "degraded"
    
    # Check Ollama
    try:
        result = await check_ollama_connection()
        health_data["services"]["ollama"] = {
            "status": "healthy" if result else "unhealthy",
            "models": ["tinyllama"],  # Current reality
            "response_time_ms": 100
        }
    except Exception as e:
        health_data["services"]["ollama"] = {"status": "unhealthy", "error": str(e)}
        health_data["status"] = "degraded"
    
    return health_data
```

### Synthetic Monitoring

**Blackbox Exporter Configuration** (`/monitoring/blackbox/config.yml`):
```yaml
modules:
  http_2xx:
    prober: http
    http:
      valid_status_codes: [200, 201, 204]
      method: GET
      
  http_post_2xx:
    prober: http
    http:
      method: POST
      headers:
        Content-Type: application/json
      body: '{"test": true}'
      
  tcp_connect:
    prober: tcp
    tcp:
      query_response:
        - expect: "^+OK"
```

**Prometheus Scrape Config:**
```yaml
- job_name: 'blackbox'
  metrics_path: /probe
  params:
    module: [http_2xx]
  static_configs:
    - targets:
      - http://sutazai-backend:8000/health
      - http://sutazai-frontend:8501/health  
      - http://sutazai-ai-agent-orchestrator:8589/health
  relabel_configs:
    - source_labels: [__address__]
      target_label: __param_target
    - source_labels: [__param_target]
      target_label: instance
    - target_label: __address__
      replacement: sutazai-blackbox-exporter:9115
```

### End-to-End Testing

**Synthetic Transaction Monitoring:**
```python
# E2E health check script
import asyncio
import aiohttp
import time

async def e2e_health_check():
    """End-to-end health check simulating user workflow"""
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        # Step 1: Check API health
        async with session.get('http://sutazai-backend:8000/health') as resp:
            if resp.status != 200:
                return {"status": "failed", "step": "api_health"}
        
        # Step 2: Test Ollama integration
        payload = {"model": "tinyllama", "prompt": "Hello"}
        async with session.post('http://sutazai-backend:8000/api/v1/generate', 
                              json=payload) as resp:
            if resp.status != 200:
                return {"status": "failed", "step": "llm_generation"}
        
        # Step 3: Test database connectivity
        async with session.get('http://sutazai-backend:8000/api/v1/tasks') as resp:
            if resp.status not in [200, 204]:  # 204 if no tasks
                return {"status": "failed", "step": "database_query"}
    
    duration = time.time() - start_time
    return {"status": "passed", "duration_seconds": duration}
```

### SLA Tracking

**Service Level Agreements:**

| Service | Availability SLA | Response Time SLA | Error Rate SLA |
|---------|------------------|-------------------|----------------|
| Backend API | 99.9% | 95th percentile < 1s | < 0.1% |
| Agent Services | 99.5% | Health check < 5s | < 1% |
| Database | 99.95% | Query < 100ms | < 0.01% |
| LLM Service | 99% | Generation < 10s | < 2% |

**SLA Monitoring Queries:**
```promql
# Availability SLA
(
  sum(rate(http_requests_total{status!~"5.."}[30d])) /
  sum(rate(http_requests_total[30d]))
) * 100

# Response time SLA  
(
  histogram_quantile(0.95, 
    sum(rate(http_request_duration_seconds_bucket[30d])) by (le)
  ) < 1
) * 100

# Error rate SLA
(
  1 - (
    sum(rate(http_requests_total{status=~"5.."}[30d])) /
    sum(rate(http_requests_total[30d]))
  )
) * 100
```

## 9. Distributed Tracing (Future)

### Jaeger Integration Plan

**Current Status**: Not implemented (planned future enhancement)

**Implementation Architecture:**
```mermaid
graph LR
    A[Application] --> B[Jaeger Client]
    B --> C[Jaeger Agent]
    C --> D[Jaeger Collector]
    D --> E[Cassandra/Elasticsearch]
    F[Jaeger UI] --> E
```

### Trace Context Propagation

**OpenTelemetry Integration (Planned):**
```python
from opentelemetry import trace
from opentelemetry.exporter.jaeger import JaegerExporter
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Tracer setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

jaeger_exporter = JaegerExporter(
    agent_host_name="jaeger-agent",
    agent_port=6831,
)

span_processor = BatchSpanProcessor(jaeger_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

# Usage in FastAPI
@app.middleware("http")
async def tracing_middleware(request: Request, call_next):
    with tracer.start_as_current_span(
        name=f"{request.method} {request.url.path}",
        attributes={
            "http.method": request.method,
            "http.url": str(request.url),
        }
    ) as span:
        response = await call_next(request)
        span.set_attribute("http.status_code", response.status_code)
        return response
```

### Span Collection

**Key Spans to Implement:**
- HTTP request/response spans
- Database query spans
- Ollama inference spans
- Inter-agent communication spans
- Cache operation spans

### Performance Analysis

**Trace Analysis Queries (Jaeger UI):**
- Slowest services by operation
- Error rate by service
- Service dependency mapping
- Request flow visualization

## 10. Observability Best Practices

### RED Method (Rate, Errors, Duration)

**For Services:**
```promql
# Rate - Requests per second
sum(rate(http_requests_total[5m])) by (service)

# Errors - Error percentage
sum(rate(http_requests_total{status=~"5.."}[5m])) by (service) /
sum(rate(http_requests_total[5m])) by (service) * 100

# Duration - Response time distribution
histogram_quantile(0.50, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))
histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))
histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, service))
```

### USE Method (Utilization, Saturation, Errors)

**For Resources:**
```promql
# Utilization - Percentage of time resource was busy
rate(node_cpu_seconds_total{mode!="idle"}[5m]) * 100

# Saturation - Degree of overload
node_load1 / count(node_cpu_seconds_total{mode="idle"}) by (instance)

# Errors - Error events
increase(node_netstat_TcpExt_ListenDrops[5m])
```

### Four Golden Signals

1. **Latency**: Time to process requests
2. **Traffic**: Demand on your system
3. **Errors**: Rate of failed requests  
4. **Saturation**: How full your service is

**Implementation:**
```yaml
# Recording rules for Golden Signals
- record: golden_signals:latency
  expr: histogram_quantile(0.99, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, job))

- record: golden_signals:traffic  
  expr: sum(rate(http_requests_total[5m])) by (job)

- record: golden_signals:errors
  expr: sum(rate(http_requests_total{status=~"5.."}[5m])) by (job)

- record: golden_signals:saturation
  expr: |
    avg(
      (rate(container_cpu_usage_seconds_total[5m]) / 
       container_spec_cpu_quota) * 100
    ) by (container)
```

### SLI/SLO/SLA Definitions

**Service Level Indicators (SLIs):**
- Availability: Percentage of successful requests
- Latency: Time to complete requests
- Throughput: Requests handled per time unit
- Quality: Correctness of responses

**Service Level Objectives (SLOs):**
- API Availability: 99.9% of requests succeed
- API Latency: 95% of requests complete in < 1 second
- LLM Generation: 99% of generations complete in < 10 seconds

**Error Budget Management:**
```promql
# Error budget remaining (monthly)
(
  1 - (
    sum(rate(http_requests_total{status=~"5.."}[30d])) / 
    sum(rate(http_requests_total[30d]))
  )
) / 0.001  # For 99.9% SLO

# Error budget burn rate
(
  sum(rate(http_requests_total{status=~"5.."}[1h])) / 
  sum(rate(http_requests_total[1h]))
) / 0.001  # Hourly burn rate
```

### Monitoring as Code

**Prometheus Configuration Management:**
```bash
# Version control all monitoring configs
git add monitoring/prometheus/
git add monitoring/grafana/dashboards/
git add monitoring/alertmanager/

# Automated deployment
kubectl apply -f monitoring/k8s/  # When using Kubernetes
docker-compose up -d  # Current Docker setup
```

**Dashboard Versioning:**
```json
{
  "dashboard": {
    "title": "SutazAI System Overview v2.1",
    "version": 21,
    "schemaVersion": 30,
    "tags": ["sutazai", "overview", "production"]
  }
}
```

## 11. Troubleshooting with Observability

### Common Investigation Workflows

#### 1. Service Degradation Investigation

**Step-by-Step Process:**
```bash
# Step 1: Check service health
curl http://localhost:10010/health

# Step 2: Check Prometheus targets
curl http://localhost:10200/api/v1/targets

# Step 3: Query error rate
curl -G http://localhost:10200/api/v1/query \
  --data-urlencode 'query=sum(rate(http_requests_total{status=~"5.."}[5m]))'

# Step 4: Check response times
curl -G http://localhost:10200/api/v1/query \
  --data-urlencode 'query=histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le))'
```

**Grafana Investigation Dashboard:**
- Open System Overview dashboard
- Check service health panel
- Review error rate trends
- Analyze response time distribution
- Examine resource utilization

#### 2. Database Performance Issues

**PostgreSQL Investigation:**
```sql
-- Check active connections
SELECT count(*) FROM pg_stat_activity WHERE state = 'active';

-- Find slow queries  
SELECT query, mean_time, calls 
FROM pg_stat_statements 
ORDER BY mean_time DESC LIMIT 10;

-- Check lock contention
SELECT blocked_locks.pid AS blocked_pid,
       blocking_locks.pid AS blocking_pid,
       blocked_activity.query AS blocked_query
FROM pg_locks blocked_locks
JOIN pg_stat_activity blocked_activity ON blocked_locks.pid = blocked_activity.pid
JOIN pg_locks blocking_locks ON blocking_locks.locktype = blocked_locks.locktype
WHERE NOT blocked_locks.granted;
```

**Prometheus Queries:**
```promql
# Database connection spikes
increase(pg_stat_database_numbackends[1h])

# Query duration trends
pg_stat_statements_mean_time > 1000

# Lock wait events
increase(pg_stat_database_conflicts_total[5m])
```

#### 3. Agent Communication Issues

**Current State** (Stub agents):
```bash
# Check agent health status
for port in 8002 8551 8587 8588 8589 11015 11063; do
  echo "Checking agent on port $port:"
  curl -s http://localhost:$port/health || echo "Failed"
done
```

**When Agents Become Functional:**
```promql
# Agent response time  
probe_duration_seconds{job=~".*agent.*"}

# Agent task queue size
agent_queue_size > 100

# Agent error rate
sum(rate(agent_tasks_processed_total{status="error"}[5m])) by (agent_name)
```

### Correlating Metrics and Logs

**Multi-datasource Investigation:**

1. **Identify Issue in Metrics:**
```promql
# High error rate detected
sum(rate(http_requests_total{status=~"5.."}[5m])) > 10
```

2. **Correlate with Logs:**
```logql
# Find error logs in the same timeframe
{service="sutazai-backend"} |= "ERROR" | json | status_code >= 500
```

3. **Cross-reference with Traces** (when implemented):
```
# Jaeger UI: Search for traces with errors in the same time window
# Filter by service and error status
```

**Example Correlation Workflow:**
```bash
# 1. Metrics show error spike at 14:30
curl -G http://localhost:10200/api/v1/query_range \
  --data-urlencode 'query=sum(rate(http_requests_total{status=~"5.."}[1m]))' \
  --data-urlencode 'start=2025-08-08T14:25:00Z' \
  --data-urlencode 'end=2025-08-08T14:35:00Z'

# 2. Check logs for the same period
curl -G http://localhost:10202/loki/api/v1/query_range \
  --data-urlencode 'query={service="sutazai-backend"} |= "ERROR"' \
  --data-urlencode 'start=2025-08-08T14:25:00Z' \
  --data-urlencode 'end=2025-08-08T14:35:00Z'

# 3. Investigate container metrics
curl -G http://localhost:10200/api/v1/query \
  --data-urlencode 'query=container_memory_usage_bytes{name="sutazai-backend"}[10m:1m]'
```

### Root Cause Analysis

**RCA Template:**

1. **Incident Timeline:**
   - When did the issue start?
   - What was the impact?
   - When was it resolved?

2. **Symptoms Observed:**
   - Metrics showing degradation
   - Error messages in logs
   - User-reported issues

3. **Investigation Steps:**
   - Queries run and results
   - Logs examined
   - External factors considered

4. **Root Cause:**
   - Technical cause identified
   - Why monitoring didn't catch it earlier
   - Prevention measures

**Example RCA:**
```markdown
# Incident: High Response Time on 2025-08-08 14:30

## Timeline
- 14:30: Alert fired for high response time (>2s)
- 14:32: Investigation started
- 14:45: Root cause identified (database connection pool exhausted)
- 14:50: Issue resolved by restarting backend service

## Symptoms
- 95th percentile response time: 5.2s (SLA: <1s)
- Database connection errors in logs
- No increase in traffic volume

## Root Cause
Backend service had connection pool leak, exhausting all 20 connections.

## Prevention
- Add connection pool monitoring
- Implement connection timeout alerts
- Add connection pool size to health check
```

### Performance Debugging

**CPU Performance Investigation:**
```promql
# CPU usage by container
rate(container_cpu_usage_seconds_total[5m]) * 100

# CPU throttling
rate(container_cpu_cfs_throttled_seconds_total[5m])

# System load
node_load1 / count(node_cpu_seconds_total{mode="idle"}) by (instance)
```

**Memory Performance Investigation:**
```promql
# Memory usage percentage
container_memory_usage_bytes / container_spec_memory_limit_bytes * 100

# Memory allocation rate
rate(go_memstats_alloc_bytes_total[5m])

# Garbage collection frequency  
rate(go_memstats_gc_completed_total[5m])
```

**Disk I/O Investigation:**
```promql
# Disk utilization
rate(node_disk_io_time_seconds_total[5m])

# Read/write IOPS
rate(node_disk_reads_completed_total[5m])
rate(node_disk_writes_completed_total[5m])

# Disk latency
rate(node_disk_read_time_seconds_total[5m]) / rate(node_disk_reads_completed_total[5m])
```

### Capacity Planning

**Growth Trend Analysis:**
```promql
# Request volume growth (weekly)
predict_linear(sum(rate(http_requests_total[1d]))[7d:1d], 86400 * 30)  # 30 days ahead

# Memory usage growth
predict_linear(avg(container_memory_usage_bytes)[7d:1h], 86400 * 30)

# Database size growth
predict_linear(pg_database_size_bytes[30d:1d], 86400 * 90)  # 90 days ahead
```

**Resource Forecasting:**
```promql
# CPU capacity planning
quantile(0.95,
  predict_linear(
    avg(rate(container_cpu_usage_seconds_total[5m]))[30d:1h], 
    86400 * 90
  )
)

# Memory capacity planning  
quantile(0.95,
  predict_linear(
    avg(container_memory_usage_bytes)[30d:1h],
    86400 * 90
  )
)
```

## 12. Operational Dashboards

### Executive Dashboard

**Key Metrics for Leadership:**

```json
{
  "dashboard": {
    "title": "SutazAI Executive Overview",
    "panels": [
      {
        "title": "System Health Score",
        "type": "stat",
        "description": "Overall system health (0-100)",
        "targets": [
          {
            "expr": "(\n  (sutazai:sli_availability / 100) * 0.4 +\n  (min(sutazai:sli_latency < 1) * 100) * 0.3 +\n  ((100 - sutazai:sli_error_rate) / 100) * 0.3\n) * 100",
            "legendFormat": "Health Score"
          }
        ]
      },
      {
        "title": "Active Users (24h)",
        "type": "stat", 
        "targets": [
          {
            "expr": "count(count(increase(http_requests_total{endpoint!~\".*/health\"}[24h])) by (user_id))",
            "legendFormat": "Active Users"
          }
        ]
      },
      {
        "title": "AI Requests Generated (24h)",
        "type": "stat",
        "targets": [
          {
            "expr": "sum(increase(ollama_requests_total[24h]))",
            "legendFormat": "AI Requests"
          }
        ]
      },
      {
        "title": "System Availability Trend",
        "type": "graph",
        "targets": [
          {
            "expr": "sutazai:sli_availability",
            "legendFormat": "Availability %"
          }
        ],
        "yAxes": [{"min": 99, "max": 100}]
      }
    ]
  }
}
```

### Developer Dashboard

**Development-Focused Metrics:**

```json
{
  "dashboard": {
    "title": "SutazAI Developer Overview", 
    "panels": [
      {
        "title": "API Response Times by Endpoint",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, sum(rate(http_request_duration_seconds_bucket[5m])) by (le, endpoint))",
            "legendFormat": "{{ endpoint }} - 95th percentile"
          }
        ]
      },
      {
        "title": "Error Rate by Endpoint", 
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"5..\"}[5m])) by (endpoint) / sum(rate(http_requests_total[5m])) by (endpoint) * 100",
            "legendFormat": "{{ endpoint }} error rate"
          }
        ]
      },
      {
        "title": "Database Query Performance",
        "type": "table",
        "targets": [
          {
            "expr": "topk(10, pg_stat_statements_mean_time)",
            "legendFormat": "Query: {{ query }}"
          }
        ]
      },
      {
        "title": "Agent Health Status",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=~\".*agent.*\"}",
            "legendFormat": "{{ job }}"
          }
        ]
      }
    ]
  }
}
```

### Operations Dashboard

**Infrastructure Operations Focus:**

```json
{
  "dashboard": {
    "title": "SutazAI Operations Overview",
    "panels": [
      {
        "title": "Container Status",
        "type": "table",
        "targets": [
          {
            "expr": "up{job=~\"sutazai.*\"}",
            "legendFormat": "Service: {{ job }}, Status: {{ up }}"
          }
        ]
      },
      {
        "title": "Resource Utilization",
        "type": "graph", 
        "targets": [
          {
            "expr": "avg(rate(container_cpu_usage_seconds_total[5m])) by (name) * 100",
            "legendFormat": "{{ name }} CPU %"
          },
          {
            "expr": "avg(container_memory_usage_bytes / container_spec_memory_limit_bytes * 100) by (name)",
            "legendFormat": "{{ name }} Memory %"
          }
        ]
      },
      {
        "title": "Alert Summary",
        "type": "table",
        "targets": [
          {
            "expr": "ALERTS{alertstate=\"firing\"}",
            "legendFormat": "Alert: {{ alertname }}, Service: {{ service }}"
          }
        ]
      },
      {
        "title": "Log Error Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate({service=~\"sutazai.*\"} |= \"ERROR\" [5m])) by (service)",
            "legendFormat": "{{ service }} error logs/sec"
          }
        ]
      }
    ]
  }
}
```

### Security Dashboard

**Security Monitoring Focus:**

```json
{
  "dashboard": {
    "title": "SutazAI Security Overview",
    "panels": [
      {
        "title": "Authentication Failures",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=\"401\"}[5m]))",
            "legendFormat": "Auth failures/sec"
          }
        ]
      },
      {
        "title": "Suspicious IP Activity",
        "type": "table", 
        "targets": [
          {
            "expr": "topk(10, sum(rate(http_requests_total[5m])) by (remote_addr))",
            "legendFormat": "IP: {{ remote_addr }}, RPS: {{ value }}"
          }
        ]
      },
      {
        "title": "Failed Requests by Status Code",
        "type": "pie",
        "targets": [
          {
            "expr": "sum(rate(http_requests_total{status=~\"4..|5..\"}[5m])) by (status)",
            "legendFormat": "{{ status }}"
          }
        ]
      },
      {
        "title": "Container Security Events",
        "type": "logs",
        "targets": [
          {
            "expr": "{job=\"docker\"} |= \"security\" or \"unauthorized\" or \"breach\"",
            "legendFormat": "Security Events"
          }
        ]
      }
    ]
  }
}
```

### Cost Monitoring Dashboard

**Resource Cost Tracking:**

```json
{
  "dashboard": {
    "title": "SutazAI Cost Monitoring",
    "panels": [
      {
        "title": "CPU Hours by Service",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(container_cpu_usage_seconds_total[5m])) by (name) * 3600",
            "legendFormat": "{{ name }} CPU hours/hour"
          }
        ]
      },
      {
        "title": "Memory Usage by Service",
        "type": "graph",
        "targets": [
          {
            "expr": "avg(container_memory_usage_bytes) by (name) / 1024^3",
            "legendFormat": "{{ name }} GB"
          }
        ]
      },
      {
        "title": "Storage Usage Trend",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(node_filesystem_size_bytes - node_filesystem_avail_bytes) / 1024^3",
            "legendFormat": "Used Storage (GB)"
          }
        ]
      },
      {
        "title": "Network Data Transfer",
        "type": "graph",
        "targets": [
          {
            "expr": "sum(rate(container_network_transmit_bytes_total[5m])) / 1024^2",
            "legendFormat": "Outbound MB/s"
          },
          {
            "expr": "sum(rate(container_network_receive_bytes_total[5m])) / 1024^2", 
            "legendFormat": "Inbound MB/s"
          }
        ]
      }
    ]
  }
}
```

## Quick Start Guide

### 1. Access Monitoring Services

```bash
# Open monitoring interfaces
open http://localhost:10200  # Prometheus
open http://localhost:10201  # Grafana (admin/admin)
open http://localhost:10202  # Loki (if web UI enabled)
open http://localhost:10203  # AlertManager
```

### 2. Import Dashboards

```bash
# Navigate to existing dashboard configs
ls /opt/sutazaiapp/monitoring/grafana/dashboards/

# Import via Grafana UI or API
curl -X POST http://admin:admin@localhost:10201/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/dashboards/sutazai-system-overview.json
```

### 3. Verify Alerting

```bash
# Check AlertManager targets
curl http://localhost:10203/api/v1/status

# Test alert firing
curl -X POST http://localhost:10203/api/v1/alerts \
  -H "Content-Type: application/json" \
  -d '[{"labels":{"alertname":"TestAlert","service":"test"}}]'
```

### 4. Query Examples

**Prometheus:**
```promql
# Check all services are up
up{job=~"sutazai.*"}

# Current request rate
sum(rate(http_requests_total[5m]))

# Error percentage
sum(rate(http_requests_total{status=~"5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100
```

**Loki:**
```logql
# Recent errors
{service=~"sutazai.*"} |= "ERROR" 

# Backend service logs
{service="sutazai-backend"} | json | level="INFO"
```

## Conclusion

This observability guide provides a comprehensive foundation for monitoring the SutazAI distributed AI platform. The current monitoring stack (Prometheus, Grafana, Loki, AlertManager) is operational and ready for production use.

**Next Steps:**
1. Implement missing agent metrics when services become functional
2. Add vector database monitoring integration
3. Deploy distributed tracing with Jaeger
4. Configure business-specific KPIs and alerts
5. Implement automated runbook procedures

**Key Files Referenced:**
- `/opt/sutazaiapp/monitoring/prometheus/prometheus.yml`
- `/opt/sutazaiapp/monitoring/grafana/dashboards/`
- `/opt/sutazaiapp/monitoring/alertmanager/alertmanager.yml`
- `/opt/sutazaiapp/docker-compose.yml`

For issues or enhancements, refer to the monitoring team or create issues in the project repository.

---
*Last Updated: August 8, 2025*  
*Version: 1.0*  
*Generated with Claude Code*