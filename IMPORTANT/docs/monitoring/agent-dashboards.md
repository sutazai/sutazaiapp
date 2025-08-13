# Agent-Level Observability with Prometheus & Grafana

## Overview

This document describes the implementation of comprehensive observability for all SutazAI agent services using Prometheus metrics collection and Grafana visualization.

## Architecture

```
┌─────────────────┐     ┌──────────────┐     ┌─────────────┐
│   Agent         │────▶│  Prometheus  │────▶│   Grafana   │
│  /metrics       │     │   Scraper    │     │  Dashboard  │
└─────────────────┘     └──────────────┘     └─────────────┘
        │                       │                     │
        ▼                       ▼                     ▼
   Metrics Export         Time Series DB         Visualization
```

## Implementation Status

### ✅ Completed Components

1. **Metrics Module** (`/agents/core/metrics.py`)
   - Standardized metrics collection for all agents
   - Prometheus client integration
   - Decorator-based request tracking
   - Automatic metric generation

2. **Agent Integration**
   - Ollama Integration Agent (port 8090) - ✅ Metrics enabled
   - Task Assignment Coordinator (port 8551) - ✅ Metrics prepared
   - Resource Arbitration Agent (port 8588) - ✅ Metrics prepared
   - AI Agent Orchestrator (port 8589) - ✅ Metrics prepared
   - Hardware Resource Optimizer (port 8002) - ✅ Metrics prepared

3. **Prometheus Configuration**
   - Scrape targets configured for all agents
   - 30-second scrape interval
   - Service discovery via static configs

4. **Grafana Dashboard**
   - Agent Performance Dashboard created
   - Real-time metrics visualization
   - Alert annotations

5. **Alert Rules**
   - High error rate (>5%)
   - High latency (>300ms)
   - Agent down detection
   - Agent unhealthy status
   - High queue latency (>1s)

6. **Synthetic Load Testing**
   - Python-based load generator
   - Configurable rate and error injection
   - Statistical analysis and reporting

7. **CI/CD Integration**
   - GitHub Actions workflow for alert simulation
   - Automated testing of alert conditions
   - Daily scheduled runs

## Metrics Collected

### Standard Metrics (All Agents)

| Metric | Type | Description | Labels |
|--------|------|-------------|--------|
| `agent_request_total` | Counter | Total requests processed | agent, method, status |
| `agent_error_total` | Counter | Total errors encountered | agent, error_type |
| `agent_queue_latency_seconds` | Histogram | Queue wait time | agent |
| `agent_db_query_duration_seconds` | Histogram | Database query time | agent, query_type |
| `agent_processing_duration_seconds` | Histogram | Request processing time | agent, method |
| `agent_active_requests` | Gauge | Currently processing requests | agent |
| `agent_health_status` | Gauge | Health status (1=healthy, 0=unhealthy) | agent |
| `agent_last_request_timestamp` | Gauge | Unix timestamp of last request | agent |

## Quick Start Guide

### 1. Adding Metrics to a New Agent

```python
# In your agent's app.py
from agents.core.metrics import AgentMetrics, setup_metrics_endpoint

# During startup
metrics = AgentMetrics("your_agent_name")
setup_metrics_endpoint(app, metrics)

# Track requests
@metrics.track_request("method_name")
async def your_endpoint():
    # Your code here
    pass

# Manual metric recording
metrics.record_queue_latency(0.5)  # 500ms queue time
metrics.record_db_query("select", 0.02)  # 20ms query
metrics.increment_error("validation_error")
```

### 2. Running Load Tests

```bash
# Basic load test
python scripts/synthetic-load-test.py --duration 60 --rate 10

# Test with error injection
python scripts/synthetic-load-test.py --duration 60 --rate 10 --error-rate 0.1

# Test specific agent
python scripts/synthetic-load-test.py --agent ollama-integration --rate 20
```

### 3. Accessing Dashboards

- **Grafana**: http://localhost:10201 (admin/admin)
  - Navigate to Dashboards → Agent Performance Dashboard
  
- **Prometheus**: http://localhost:10200
  - Query examples:
    ```promql
    # Request rate
    rate(agent_request_total[5m])
    
    # Error percentage
    rate(agent_error_total[5m]) / rate(agent_request_total[5m])
    
    # P95 latency
    histogram_quantile(0.95, rate(agent_processing_duration_seconds_bucket[5m]))
    ```

## Alert Configuration

### Alert Thresholds

| Alert | Condition | Duration | Severity |
|-------|-----------|----------|----------|
| AgentHighErrorRate | Error rate > 5% | 2 minutes | Warning |
| AgentHighLatency | P95 latency > 300ms | 2 minutes | Warning |
| AgentDown | Agent unreachable | 1 minute | Critical |
| AgentUnhealthy | Health status = 0 | 2 minutes | Warning |
| AgentHighQueueLatency | P95 queue > 1s | 2 minutes | Warning |

### Testing Alerts

```bash
# Trigger high error rate alert
python scripts/synthetic-load-test.py \
  --duration 180 \
  --rate 10 \
  --error-rate 0.15 \
  --agent ollama-integration

# Check alert status
curl -s http://localhost:10200/api/v1/alerts | jq '.data.alerts'
```

## Troubleshooting

### Common Issues

1. **Metrics endpoint returns 404**
   - Ensure metrics module is copied to agent directory
   - Check that `setup_metrics_endpoint()` is called during startup
   - Verify prometheus-client is installed

2. **No data in Grafana**
   - Check Prometheus targets: http://localhost:10200/targets
   - Verify agent is running and healthy
   - Check network connectivity between containers

3. **Alerts not firing**
   - Verify alert rules are loaded: http://localhost:10200/rules
   - Check alert thresholds match actual metric values
   - Ensure sufficient data duration (most alerts require 2 minutes)

### Debug Commands

```bash
# Check if metrics endpoint is working
curl -s http://localhost:8090/metrics | head -20

# View Prometheus targets
curl -s http://localhost:10200/api/v1/targets | jq '.data.activeTargets'

# Check active alerts
curl -s http://localhost:10200/api/v1/alerts | jq '.data.alerts'

# View agent logs
docker logs sutazai-ollama-integration --tail 50
```

## Performance Impact

The metrics collection has   performance impact:
- CPU overhead: <1% per agent
- Memory overhead: ~5MB per agent
- Network overhead: ~2KB per scrape (every 30s)

## Best Practices

1. **Use descriptive metric names and labels**
   - Good: `agent_request_total{method="generate"}`
   - Bad: `counter1{type="req"}`

2. **Keep cardinality low**
   - Avoid high-cardinality labels (e.g., user_id, request_id)
   - Use buckets appropriate for your latency ranges

3. **Track business metrics**
   - Not just technical metrics (latency, errors)
   - Also business metrics (tasks processed, models loaded)

4. **Set realistic alert thresholds**
   - Based on actual baseline performance
   - Account for normal variations
   - Avoid alert fatigue

5. **Use metrics for capacity planning**
   - Track trends over time
   - Identify bottlenecks before they become critical
   - Plan scaling based on actual usage patterns

## Future Enhancements

- [ ] Distributed tracing with Jaeger
- [ ] Custom business metrics per agent
- [ ] Automated anomaly detection
- [ ] SLO/SLI tracking and reporting
- [ ] Integration with PagerDuty/Slack for alerts
- [ ] Historical trend analysis
- [ ] Predictive alerting based on ML models

## References

- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [Prometheus Client Python](https://github.com/prometheus/client_python)
- [SRE Book - Monitoring Distributed Systems](https://sre.google/sre-book/monitoring-distributed-systems/)