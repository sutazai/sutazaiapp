# SutazAI Monitoring Dashboards

## 📊 Overview

This directory contains comprehensive monitoring dashboards for the SutazAI system, designed to provide maximum visibility and actionable insights for infrastructure and DevOps teams.

## 📁 Dashboard Files

### Core System Dashboards

1. **sutazai-ollama-comprehensive.json**
   - **Purpose**: Comprehensive Ollama AI model performance monitoring
   - **Metrics**: Generation times, token rates, model usage, memory consumption, concurrent requests
   - **Key Features**:
     - Response time percentiles (p99, p95, p90, median)
     - Token generation rate tracking
     - Model-specific memory usage
     - Success rate and error monitoring
     - Real-time performance alerts

2. **sutazai-circuit-breaker.json**
   - **Purpose**: Circuit breaker status and reliability monitoring
   - **Metrics**: Trip events, recovery times, failure rates, service health
   - **Key Features**:
     - Real-time circuit breaker state visualization
     - Service failure rate tracking
     - Recovery time monitoring
     - Health summary for all protected services

3. **sutazai-api-performance.json**
   - **Purpose**: API endpoint performance and health monitoring
   - **Metrics**: Response times, error rates, throughput, request sizes
   - **Key Features**:
     - Response time distribution analysis
     - HTTP status code breakdown
     - Endpoint-specific performance metrics
     - Request/response size analytics

4. **sutazai-system-health.json**
   - **Purpose**: System-wide health overview and resource monitoring
   - **Metrics**: Service status, resource utilization, database connections
   - **Key Features**:
     - All services health status at a glance
     - CPU and memory usage by container
     - Database connection monitoring
     - System uptime tracking

## 🚀 Quick Setup

### 1. Automatic Import (Grafana Provisioning)

The dashboards are configured to be automatically imported via Grafana provisioning:

```bash
# Grafana will automatically load dashboards from:
/opt/sutazaiapp/monitoring/dashboards/

# Provisioning configuration:
/opt/sutazaiapp/monitoring/grafana/provisioning/dashboards/sutazai-dashboards.yml
```

### 2. Manual Import

If you need to manually import dashboards:

1. Open Grafana: http://localhost:10201
2. Login with admin credentials
3. Navigate to **Dashboards** → **Import**
4. Upload the JSON files from this directory
5. Configure datasources (Prometheus, Loki, Alertmanager)

### 3. Access Dashboards

Once imported, dashboards are organized in folders:
- **SutazAI System**: Core system monitoring dashboards
- **Performance Monitoring**: Performance-focused dashboards
- **Infrastructure**: Infrastructure and resource dashboards
- **Business Metrics**: Business and application metrics

## 📈 Dashboard Features

### Visualization Types
- **Timeseries**: Real-time metric trends
- **Stat Panels**: Key performance indicators
- **Gauge**: Resource utilization levels
- **Heatmaps**: Response time distribution
- **Tables**: Detailed metric breakdowns
- **Logs**: Contextual log analysis

### Alert Integration
- Integrated Grafana alerting rules
- Threshold-based notifications
- Circuit breaker status alerts
- Performance degradation warnings

### Template Variables
- Service filtering by name/type
- Time range selection
- Instance-based filtering
- Model-specific monitoring

## 🎯 Key Metrics

### Ollama Performance
```promql
# Response time percentiles
histogram_quantile(0.99, rate(ollama_request_duration_seconds_bucket[5m]))

# Token generation rate
rate(ollama_tokens_generated_total[5m])

# Model memory usage
ollama_model_memory_bytes/1024/1024
```

### Circuit Breaker Health
```promql
# Circuit breaker state (0=CLOSED, 1=OPEN, 2=HALF_OPEN)
circuit_breaker_state

# Failure rate by service
rate(circuit_breaker_failures_total[5m])

# Success rate calculation
(circuit_breaker_success_count / (circuit_breaker_success_count + circuit_breaker_failure_count)) * 100
```

### API Performance
```promql
# Response time percentiles
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Request rate by endpoint
sum by (endpoint) (rate(http_requests_total[5m]))

# Error rate calculation
sum(rate(http_requests_total{status_code=~"4..|5.."}[5m])) / sum(rate(http_requests_total[5m])) * 100
```

### System Health
```promql
# Service availability
up

# CPU usage by container
rate(container_cpu_usage_seconds_total[5m]) * 100

# Memory usage percentage
(container_memory_usage_bytes / container_spec_memory_limit_bytes) * 100
```

## 🔧 Customization

### Adding New Metrics

1. **Define Metric**: Add metric collection in your service
2. **Expose Endpoint**: Ensure Prometheus can scrape the metric
3. **Update Dashboard**: Add panel with appropriate query
4. **Configure Alerts**: Set thresholds and notification channels

### Dashboard Modifications

```json
{
  "targets": [
    {
      "expr": "your_custom_metric_query",
      "legendFormat": "{{instance}}",
      "refId": "A"
    }
  ]
}
```

### Alert Rules

```yaml
# Add to prometheus/alert_rules.yml
- alert: HighResponseTime
  expr: histogram_quantile(0.99, rate(http_request_duration_seconds_bucket[5m])) > 5
  for: 2m
  labels:
    severity: warning
  annotations:
    summary: "High API response time detected"
```

## 📱 Mobile Responsiveness

All dashboards are designed to be mobile-responsive:
- Adaptive grid layouts
- Touch-friendly controls
- Optimized for tablet viewing
- Responsive time picker

## 🔐 Security Considerations

- **Authentication**: Ensure Grafana authentication is enabled
- **Authorization**: Use appropriate user roles and permissions
- **Data Source Security**: Secure Prometheus and other data sources
- **Network Security**: Use HTTPS in production environments

## 🛠️ Troubleshooting

### Common Issues

1. **No Data Showing**
   ```bash
   # Check Prometheus targets
   curl http://localhost:10200/api/v1/targets
   
   # Verify metric existence
   curl http://localhost:10200/api/v1/query?query=up
   ```

2. **Dashboard Import Errors**
   - Verify JSON syntax
   - Check datasource configuration
   - Ensure required plugins are installed

3. **Performance Issues**
   - Adjust time ranges for heavy queries
   - Optimize PromQL expressions
   - Enable query caching

### Log Analysis

```bash
# Check Grafana logs
docker logs sutazai-grafana

# Verify dashboard provisioning
docker exec sutazai-grafana ls -la /var/lib/grafana/dashboards/
```

## 📊 Performance Baselines

### Expected Metrics
- **API Response Time**: < 500ms (p95)
- **Ollama Generation**: < 10s per request
- **Error Rate**: < 1%
- **Service Availability**: > 99.9%

### Resource Thresholds
- **CPU Usage**: < 80%
- **Memory Usage**: < 90%
- **Disk Usage**: < 85%
- **Network Latency**: < 100ms

## 🔄 Maintenance

### Regular Tasks
- Review and update alert thresholds
- Archive old dashboard versions
- Optimize slow-performing queries
- Update documentation

### Backup
```bash
# Backup dashboards
cp /opt/sutazaiapp/monitoring/dashboards/*.json /backup/dashboards/

# Export from Grafana API
curl -u admin:password http://localhost:10201/api/search > dashboard-list.json
```

## 📞 Support

For dashboard-related issues:

1. Check [Grafana Documentation](https://grafana.com/docs/)
2. Review Prometheus [Query Documentation](https://prometheus.io/docs/prometheus/latest/querying/)
3. Consult SutazAI system logs in `/logs/`

## 🎉 Success Metrics

These dashboards provide:
- **Complete Visibility**: 360° view of system health
- **Proactive Monitoring**: Issues detected before user impact
- **Performance Insights**: Data-driven optimization opportunities
- **Operational Excellence**: Reduced MTTR and improved reliability

---

*Last Updated: August 11, 2025*  
*Version: 1.0*  
*Maintainer: Infrastructure DevOps Team*