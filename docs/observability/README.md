# SutazAI Observability & Monitoring

This document describes the comprehensive observability and monitoring solution implemented for the SutazAI system, providing detailed metrics collection, visualization, and alerting capabilities.

## Overview

The SutazAI observability stack includes:
- **Prometheus** for metrics collection and storage
- **Grafana** for visualization and dashboards  
- **Loki** for log aggregation
- **AlertManager** for alert routing and notification
- **Custom JARVIS metrics** for application-specific monitoring

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Applications  │───▶│   Prometheus    │───▶│    Grafana      │
│   (Backend,     │    │   (Metrics      │    │  (Dashboards    │
│    Agents)      │    │   Collection)   │    │  & Alerts)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│      Loki       │    │  AlertManager   │    │   Exporters     │
│   (Log Agg.)    │    │  (Alerting)     │    │  (Node, cAdvisor)│
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## JARVIS Metrics Implementation

### Core Metrics (As Per Requirements)

#### 1. Request Metrics
- **`jarvis_requests_total`** - Counter tracking total requests by service/endpoint
  - Labels: `service`, `endpoint`, `method`, `status_code`
  - Purpose: Track request volume and success rates

#### 2. Latency Metrics  
- **`jarvis_latency_seconds_bucket`** - Histogram measuring request latency
  - Labels: `service`, `endpoint`, `method`
  - Buckets: 5ms, 10ms, 25ms, 50ms, 75ms, 100ms, 250ms, 500ms, 750ms, 1s, 2.5s, 5s, 7.5s, 10s
  - Purpose: Track response time distribution and percentiles

#### 3. Error Metrics
- **`jarvis_errors_total`** - Counter tracking errors by type
  - Labels: `service`, `endpoint`, `error_type`, `error_code`  
  - Purpose: Monitor error rates and categorization

### Additional System Metrics

#### System Resources
- `system_cpu_usage_percent` - CPU utilization
- `system_memory_usage_bytes` - Memory usage in bytes
- `system_memory_total_bytes` - Total system memory
- `system_disk_usage_bytes` - Disk usage in bytes
- `system_disk_total_bytes` - Total disk capacity

#### Service Health
- `service_health_status` - Service health (1=healthy, 0=unhealthy)
- `app_uptime_seconds` - Application uptime

#### Agent Metrics
- `sutazai_agent_tasks_total` - Total agent tasks executed
- `sutazai_agent_tasks_successful_total` - Successful agent tasks
- `sutazai_agent_task_duration_seconds` - Agent task execution time

#### Model Inference
- `sutazai_model_inference_requests_total` - Model inference requests
- `sutazai_model_inference_duration_seconds` - Model inference latency

## Instrumentation

### Automatic Instrumentation

The system uses middleware for automatic HTTP request instrumentation:

```python
from app.core.metrics import initialize_metrics

# Initialize metrics for FastAPI app
initialize_metrics(app, service_name="backend")
```

This automatically tracks:
- All HTTP requests
- Response times
- Status codes
- Error rates

### Manual Instrumentation

For specific functions, use decorators:

```python
from app.core.metrics import track_model_inference, track_agent_task

@track_model_inference("tinyllama", "text_generation")
async def query_ollama(model: str, prompt: str):
    # Model inference code
    pass

@track_agent_task("chat_agent", "conversation")
async def chat_with_ai(request: ChatRequest):
    # Agent task code
    pass
```

## Configuration

### Prometheus Configuration

Prometheus is configured to scrape metrics from multiple targets:

```yaml
scrape_configs:
  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['sutazai-backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
    scrape_timeout: 10s
```

### Metrics Endpoints

- **Backend**: `http://localhost:10010/metrics`
- **Prometheus**: `http://localhost:10200`
- **Grafana**: `http://localhost:10201`

## Dashboards

### Available Dashboards

1. **JARVIS System Overview** (`jarvis-system-overview.json`)
   - Request rate per service
   - Total requests
   - Response time percentiles (p50, p90, p99)
   - Error rates
   - System health status

2. **JARVIS Request Performance** (`jarvis-request-performance.json`)
   - Request rate by endpoint
   - Latency distribution heatmap
   - Response time by endpoint
   - Request success rate
   - Average response time

3. **JARVIS Error Tracking** (`jarvis-error-tracking.json`)
   - Error rate overview
   - Error percentage
   - Client vs server errors
   - Error distribution by status code
   - Top error endpoints

### Dashboard Import

To import dashboards into Grafana:

1. Open Grafana at `http://localhost:10201`
2. Login (admin/admin)
3. Go to Dashboard → Import
4. Upload JSON files from `/docs/observability/grafana/`

## Key Metrics Queries

### Request Rate
```promql
sum(rate(jarvis_requests_total[5m])) by (service)
```

### Latency Percentiles
```promql
histogram_quantile(0.95, sum(rate(jarvis_latency_seconds_bucket[5m])) by (le, service))
```

### Error Rate
```promql
sum(rate(jarvis_errors_total[5m])) / sum(rate(jarvis_requests_total[5m])) * 100
```

### Success Rate
```promql
(sum(rate(jarvis_requests_total{status_code!~"4..|5.."}[5m])) / sum(rate(jarvis_requests_total[5m]))) * 100
```

## Alerting

### Alert Rules

Key alerts to implement:

1. **High Error Rate**
   ```promql
   (sum(rate(jarvis_errors_total[5m])) / sum(rate(jarvis_requests_total[5m]))) * 100 > 5
   ```

2. **High Latency**
   ```promql
   histogram_quantile(0.95, sum(rate(jarvis_latency_seconds_bucket[5m])) by (le)) > 2
   ```

3. **Service Down**
   ```promql
   up{job="sutazai-backend"} == 0
   ```

4. **High Memory Usage**
   ```promql
   (system_memory_usage_bytes / system_memory_total_bytes) * 100 > 85
   ```

## Best Practices

### Metric Naming
- Use consistent prefixes (`jarvis_`, `sutazai_`)
- Include units in metric names (`_seconds`, `_bytes`, `_total`)
- Use descriptive labels

### Cardinality Management
- Avoid high-cardinality labels (user IDs, timestamps)
- Use bounded label values
- Aggregate metrics when possible

### Performance
- Use appropriate histogram buckets for latency
- Sample high-volume metrics if necessary
- Implement efficient scrape intervals

## Troubleshooting

### Common Issues

1. **Metrics Not Appearing**
   - Check Prometheus targets: `http://localhost:10200/targets`
   - Verify service is exposing `/metrics` endpoint
   - Check firewall/network connectivity

2. **Dashboard Not Loading**
   - Verify Prometheus data source configuration
   - Check metric names in queries
   - Ensure time ranges are appropriate

3. **High Memory Usage**
   - Reduce retention period in Prometheus
   - Implement metric sampling
   - Check for high-cardinality metrics

### Log Locations
- Prometheus: `docker-compose logs prometheus`
- Grafana: `docker-compose logs grafana`
- Backend: `docker-compose logs backend`

## Development

### Adding New Metrics

1. Define metric in `app/core/metrics.py`:
   ```python
   my_custom_metric = Counter(
       'my_custom_metric_total',
       'Description of metric',
       ['label1', 'label2']
   )
   ```

2. Instrument code:
   ```python
   my_custom_metric.labels(label1="value1", label2="value2").inc()
   ```

3. Update dashboard queries to include new metric

### Testing Metrics

```bash
# Test metrics endpoint
curl http://localhost:10010/metrics | grep jarvis

# Test specific metric
curl -s http://localhost:10200/api/v1/query?query=jarvis_requests_total
```

## Maintenance

### Regular Tasks
- Monitor disk usage for Prometheus data
- Update dashboard queries for new metrics
- Review and tune alert thresholds
- Archive old metrics data
- Update documentation for new metrics

### Backup
- Export Grafana dashboards regularly
- Backup Prometheus data directory
- Version control alert rules

This observability setup provides comprehensive monitoring capabilities for the SutazAI system, enabling effective performance monitoring, error tracking, and system health management.