# Perfect JARVIS Post-Go-Live Monitoring Guide

## Overview

This document describes the comprehensive post-go-live monitoring system implemented for the Perfect JARVIS deployment, following Prompt 7.4.1 specifications.

## Monitoring Components

### 1. Real-Time System Monitoring

The monitoring system tracks:
- **System Health**: CPU, memory, disk usage
- **Container Status**: Running, stopped, unhealthy containers
- **Service Availability**: All 28 services with health checks
- **Performance Metrics**: Request rates, error rates, latency percentiles
- **Database Performance**: Connection pools, query performance

### 2. Automated Monitoring Script

**Location**: `/opt/sutazaiapp/scripts/monitoring/post-golive-monitor.sh`

#### Features
- Continuous monitoring with configurable intervals
- Automatic alert detection and notification
- Daily report generation
- Performance metrics collection from Prometheus
- Service health validation

#### Usage

```bash
# Run monitoring checks once
./scripts/monitoring/post-golive-monitor.sh --once

# Generate daily report
./scripts/monitoring/post-golive-monitor.sh --report

# Start continuous monitoring (checks every 5 minutes)
./scripts/monitoring/post-golive-monitor.sh --continuous

# Custom interval (checks every 60 seconds)
./scripts/monitoring/post-golive-monitor.sh --continuous 60
```

### 3. Metrics Collection

#### Key Metrics Tracked

| Metric | Description | Alert Threshold |
|--------|-------------|-----------------|
| Request Rate | Requests per second | N/A |
| Error Rate | Errors per second | > 0.05/sec |
| P95 Latency | 95th percentile response time | > 2 seconds |
| CPU Usage | System CPU utilization | > 90% |
| Memory Usage | System memory utilization | > 85% |
| Disk Usage | Disk space utilization | > 85% |
| Container Health | Unhealthy container count | > 0 |

### 4. Alert System

#### Alert Severity Levels

- **🔴 CRITICAL**: Immediate action required
  - Disk usage > 85%
  - Service down
  - Database unreachable

- **⚠️ WARNING**: Investigation needed
  - High error rate (> 5%)
  - High latency (P95 > 2s)
  - Memory usage > 85%

- **ℹ️ INFO**: Informational
  - Configuration changes
  - Scheduled maintenance
  - Performance degradation

### 5. Grafana Dashboards

Access dashboards at: http://localhost:10201

#### Available Dashboards

1. **JARVIS System Overview**
   - Real-time request rates
   - Error percentages
   - Service health status
   - System resource usage

2. **Request Performance**
   - Latency distribution
   - Endpoint-specific metrics
   - Response time trends
   - Throughput analysis

3. **Error Tracking**
   - Error categorization
   - Error rate trends
   - Failed endpoint analysis
   - Root cause indicators

### 6. Daily Reports

Reports are generated automatically at midnight and saved to:
`/opt/sutazaiapp/reports/post-golive/`

#### Report Contents
- Executive summary
- System health metrics
- Performance statistics
- Alert summary
- Recommendations
- Resource utilization trends

### 7. Log Aggregation

Logs are collected and stored in:
- **System Logs**: `/opt/sutazaiapp/logs/monitoring/`
- **Application Logs**: Docker container logs
- **Metrics Data**: `/opt/sutazaiapp/metrics/`
- **Reports**: `/opt/sutazaiapp/reports/post-golive/`

#### Log Analysis with Loki

```bash
# Search for errors in backend logs
curl -G "http://localhost:10202/loki/api/v1/query_range" \
  --data-urlencode 'query={container="sutazai-backend"} |= "ERROR"'

# Get recent warnings
curl -G "http://localhost:10202/loki/api/v1/query" \
  --data-urlencode 'query={job="jarvis"} |= "WARNING"' \
  --data-urlencode 'limit=100'
```

## Monitoring Procedures

### Daily Monitoring Tasks

1. **Morning Health Check (9:00 AM)**
   ```bash
   ./scripts/monitoring/post-golive-monitor.sh --once
   ```

2. **Review Overnight Alerts**
   - Check Grafana alert panel
   - Review alert reports in `/reports/post-golive/`

3. **Performance Review**
   - Check P95 latency trends
   - Review error rates
   - Analyze resource usage patterns

### Weekly Tasks

1. **Capacity Planning Review**
   - Analyze growth trends
   - Project resource needs
   - Plan scaling activities

2. **Performance Optimization**
   - Identify slow queries
   - Review inefficient endpoints
   - Optimize resource allocation

3. **Security Audit**
   - Review access logs
   - Check for suspicious activity
   - Validate security configurations

### Incident Response

When alerts are triggered:

1. **Acknowledge Alert**
   - Note alert time and severity
   - Begin incident log

2. **Initial Assessment**
   ```bash
   # Quick system check
   docker ps --format "table {{.Names}}\t{{.Status}}"
   
   # Check specific service
   docker logs <service-name> --tail=50
   ```

3. **Mitigation**
   - Follow runbook procedures
   - Implement temporary fixes
   - Document actions taken

4. **Resolution**
   - Apply permanent fix
   - Verify system stability
   - Update documentation

5. **Post-Incident Review**
   - Document root cause
   - Update runbooks
   - Implement preventive measures

## Performance Baselines

### Normal Operating Parameters

| Metric | Baseline | Acceptable Range |
|--------|----------|------------------|
| Request Rate | 10-50 req/s | 0-100 req/s |
| Error Rate | < 1% | 0-5% |
| P50 Latency | 50ms | 0-100ms |
| P95 Latency | 500ms | 0-2000ms |
| CPU Usage | 30-40% | 0-70% |
| Memory Usage | 40-50% | 0-80% |
| Disk I/O | 10-20 MB/s | 0-50 MB/s |

### Service-Specific SLAs

| Service | Availability | Response Time |
|---------|--------------|---------------|
| Backend API | 99.9% | < 200ms |
| Frontend | 99.9% | < 500ms |
| Ollama | 99.5% | < 3000ms |
| Database | 99.95% | < 50ms |

## Automation

### Continuous Monitoring Setup

1. **Create systemd service**:
```bash
sudo tee /etc/systemd/system/jarvis-monitor.service > /dev/null <<EOF
[Unit]
Description=JARVIS Post-Go-Live Monitoring
After=docker.service

[Service]
Type=simple
User=$USER
WorkingDirectory=/opt/sutazaiapp
ExecStart=/opt/sutazaiapp/scripts/monitoring/post-golive-monitor.sh --continuous 300
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable jarvis-monitor
sudo systemctl start jarvis-monitor
```

2. **Check service status**:
```bash
sudo systemctl status jarvis-monitor
```

### Alert Notifications

Configure Slack notifications:

```bash
# Set webhook URL
export SLACK_WEBHOOK_URL="https://hooks.slack.com/services/YOUR/WEBHOOK/URL"

# Test notification
curl -X POST "$SLACK_WEBHOOK_URL" \
  -H 'Content-Type: application/json' \
  -d '{"text": "JARVIS monitoring test notification"}'
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Identify memory-heavy containers
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}"

# Restart specific service
docker-compose restart <service-name>
```

#### High Latency
```bash
# Check slow queries
docker exec sutazai-postgres psql -U sutazai -c \
  "SELECT query, mean_exec_time FROM pg_stat_statements ORDER BY mean_exec_time DESC LIMIT 10;"

# Check connection pool
curl http://localhost:10010/metrics | grep connection
```

#### Service Unavailable
```bash
# Check service logs
docker logs <service-name> --tail=100

# Restart service
docker-compose up -d <service-name>

# Verify health
curl http://localhost:<port>/health
```

## Reporting

### Generate Custom Reports

```python
# generate_custom_report.py
import json
import requests
from datetime import datetime, timedelta

# Query Prometheus for metrics
def get_metrics(query, start, end):
    url = "http://localhost:10200/api/v1/query_range"
    params = {
        "query": query,
        "start": start.isoformat(),
        "end": end.isoformat(),
        "step": "1h"
    }
    response = requests.get(url, params=params)
    return response.json()

# Generate report
end = datetime.now()
start = end - timedelta(days=7)

metrics = {
    "request_rate": get_metrics("rate(jarvis_requests_total[1h])", start, end),
    "error_rate": get_metrics("rate(jarvis_errors_total[1h])", start, end),
    "p95_latency": get_metrics("histogram_quantile(0.95, rate(jarvis_latency_seconds_bucket[1h]))", start, end)
}

# Save report
with open(f"weekly_report_{end.strftime('%Y%m%d')}.json", "w") as f:
    json.dump(metrics, f, indent=2)
```

### Export Metrics

```bash
# Export Prometheus data
curl "http://localhost:10200/api/v1/export" > prometheus_export.json

# Export Grafana dashboards
for dashboard in $(curl -s http://localhost:10201/api/search | jq -r '.[].uid'); do
  curl "http://localhost:10201/api/dashboards/uid/$dashboard" > "dashboard_$dashboard.json"
done
```

## Best Practices

### Monitoring Guidelines

1. **Set Realistic Thresholds**
   - Base on historical data
   - Consider business impact
   - Allow for normal variations

2. **Reduce Alert Fatigue**
   - Consolidate related alerts
   - Set appropriate severity levels
   - Use alert suppression windows

3. **Maintain Documentation**
   - Update runbooks regularly
   - Document all incidents
   - Share learnings with team

4. **Regular Reviews**
   - Weekly performance reviews
   - Monthly capacity planning
   - Quarterly architecture review

### Data Retention

| Data Type | Retention Period | Storage Location |
|-----------|-----------------|------------------|
| Metrics | 30 days | Prometheus |
| Logs | 7 days | Loki |
| Reports | 90 days | Local filesystem |
| Alerts | 30 days | Database |

## Support Contacts

- **On-Call Engineer**: Check PagerDuty schedule
- **System Admin**: admin@example.com
- **Escalation**: management@example.com
- **Vendor Support**: See vendor contact list

---

*This monitoring system ensures the Perfect JARVIS deployment maintains optimal performance and reliability in production.*