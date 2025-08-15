# MCP Automation Monitoring System

## Overview

The MCP Automation Monitoring System provides comprehensive observability for the Model Context Protocol (MCP) automation infrastructure. It includes real-time monitoring, health checking, alerting, performance tracking, log aggregation, and SLA compliance monitoring.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    MCP Infrastructure                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │   MCP    │  │   MCP    │  │   MCP    │  │   MCP    │  │
│  │ Servers  │  │Automation│  │Databases │  │   APIs   │  │
│  └─────┬────┘  └─────┬────┘  └─────┬────┘  └─────┬────┘  │
└────────┼─────────────┼─────────────┼─────────────┼────────┘
         │             │             │             │
    ┌────▼─────────────▼─────────────▼─────────────▼────┐
    │           Monitoring Components                     │
    │  ┌──────────────┐  ┌──────────────┐              │
    │  │   Metrics    │  │    Health    │              │
    │  │  Collector   │  │   Monitor    │              │
    │  └──────┬───────┘  └──────┬───────┘              │
    │         │                  │                       │
    │  ┌──────▼───────┐  ┌──────▼───────┐              │
    │  │    Alert     │  │     Log      │              │
    │  │   Manager    │  │  Aggregator  │              │
    │  └──────┬───────┘  └──────┬───────┘              │
    │         │                  │                       │
    │  ┌──────▼───────┐  ┌──────▼───────┐              │
    │  │     SLA      │  │  Dashboard   │              │
    │  │   Monitor    │  │   Config     │              │
    │  └──────┬───────┘  └──────┬───────┘              │
    └─────────┼──────────────────┼──────────────────────┘
              │                  │
    ┌─────────▼──────────────────▼──────────────────────┐
    │          Monitoring HTTP Server                    │
    │                  (Port 10204)                      │
    └────────────────────┬───────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼─────┐  ┌──────▼──────┐  ┌────▼─────┐
    │Prometheus│  │   Grafana   │  │   Loki   │
    │  10200   │  │    10201    │  │  10202   │
    └──────────┘  └─────────────┘  └──────────┘
```

## Components

### 1. Metrics Collector (`metrics_collector.py`)
- Collects Prometheus metrics from MCP servers and automation workflows
- Tracks performance, availability, and resource usage
- Supports push gateway integration
- Provides custom metrics with labels and aggregations

### 2. Health Monitor (`health_monitor.py`)
- Performs comprehensive health checks on all system components
- Monitors MCP servers, APIs, databases, and containers
- Provides multi-level health status reporting
- Tracks dependencies and detects cascade failures

### 3. Alert Manager (`alert_manager.py`)
- Intelligent alert correlation and grouping
- Suppression rules and maintenance windows
- Multi-channel notifications (webhook, Slack, email)
- Alert lifecycle management

### 4. Dashboard Configuration (`dashboard_config.py`)
- Pre-configured Grafana dashboards
- Automated dashboard deployment
- Visual monitoring for all metrics
- Customizable panel configurations

### 5. Log Aggregator (`log_aggregator.py`)
- Structured log parsing and pattern recognition
- Loki integration for centralized logging
- Real-time log search and analysis
- Error pattern detection and trending

### 6. SLA Monitor (`sla_monitor.py`)
- Service Level Indicator (SLI) tracking
- Service Level Objective (SLO) compliance
- Error budget management
- Automated compliance reporting

### 7. Monitoring Server (`monitoring_server.py`)
- FastAPI-based HTTP server
- RESTful API for all monitoring operations
- WebSocket support for real-time updates
- HTML dashboard interface

## Installation

### Requirements
- Python 3.11+
- Docker and Docker Compose
- Access to Prometheus, Grafana, and Loki

### Quick Start

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Configure environment variables:**
```bash
export PROMETHEUS_PUSHGATEWAY=http://localhost:10200/metrics
export GRAFANA_URL=http://localhost:10201
export LOKI_URL=http://localhost:10202
export MONITORING_PORT=10204
```

3. **Deploy with Docker Compose:**
```bash
docker-compose -f docker-compose.monitoring.yml up -d
```

4. **Access the monitoring dashboard:**
```
http://localhost:10204
```

## API Endpoints

### Health Endpoints
- `GET /health` - Basic health check
- `GET /health/detailed` - Comprehensive health status

### Metrics Endpoints
- `GET /metrics` - Prometheus metrics
- `GET /metrics/summary` - Human-readable summary

### Alert Endpoints
- `GET /alerts` - List active alerts
- `POST /alerts` - Create manual alert
- `POST /alerts/{id}/acknowledge` - Acknowledge alert
- `POST /alerts/{id}/resolve` - Resolve alert

### SLA Endpoints
- `GET /sla/status` - Current SLA status
- `GET /sla/report` - Generate SLA report
- `POST /sla/measurement` - Record SLA measurement

### Log Endpoints
- `POST /logs` - Ingest log entry
- `GET /logs/search` - Search logs
- `GET /logs/analysis` - Error analysis

### Dashboard Endpoints
- `POST /dashboards/deploy/{key}` - Deploy dashboard

### WebSocket
- `WS /ws` - Real-time monitoring updates

## Configuration

### Prometheus Configuration
Edit `config/prometheus.yml` to configure:
- Scrape intervals
- Target endpoints
- Alert rules
- Remote storage

### Alert Rules
Edit `config/alert_rules.yml` to define:
- Alert conditions
- Severity levels
- Notification routing
- Escalation policies

### SLA Configuration
Define SLOs in configuration:
```yaml
slos:
  - name: api_availability
    target: 0.999  # 99.9%
    time_window:
      days: 30
    consequences: "Page on-call engineer"
```

## Dashboards

### Pre-configured Dashboards
1. **MCP Server Overview** - Server status and performance
2. **Automation Performance** - Workflow execution metrics
3. **System Health** - Resource utilization and trends
4. **Alerts Dashboard** - Active alerts and history
5. **SLA Compliance** - Service level tracking

### Custom Dashboards
Create custom dashboards using the Dashboard API:
```python
from dashboard_config import Dashboard, Panel

dashboard = Dashboard(
    uid="custom-dashboard",
    title="Custom Monitoring",
    panels=[...]
)
```

## Alerting

### Alert Severity Levels
- `INFO` - Informational alerts
- `WARNING` - Potential issues
- `ERROR` - Service degradation
- `CRITICAL` - Service failure
- `EMERGENCY` - System-wide impact

### Notification Channels
- **Webhook** - HTTP POST to configured endpoints
- **Slack** - Slack webhook integration
- **Email** - SMTP email notifications
- **Prometheus** - AlertManager integration
- **Log** - Local logging

## SLA Monitoring

### Default SLOs
- **MCP Availability**: 99.9% uptime
- **API Latency**: 95% requests < 200ms
- **Automation Success**: 98% success rate
- **Error Rate**: < 1% errors
- **Data Freshness**: < 60s lag
- **Resource Utilization**: < 80% usage

### Compliance Reporting
Generate reports with:
```bash
curl http://localhost:10204/sla/report
```

## Troubleshooting

### Common Issues

**Monitoring server won't start:**
- Check port 10204 is available
- Verify Docker network connectivity
- Check environment variables

**No metrics appearing:**
- Verify MCP servers are running
- Check Prometheus scrape configuration
- Ensure network connectivity

**Alerts not firing:**
- Check alert rule syntax
- Verify AlertManager configuration
- Review suppression rules

**Dashboard not loading:**
- Verify Grafana is running
- Check API key permissions
- Review dashboard JSON syntax

### Debug Mode
Enable debug logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Health Check
Verify system health:
```bash
curl http://localhost:10204/health/detailed
```

## Performance Tuning

### Optimization Tips
- Adjust collection intervals based on load
- Configure appropriate retention policies
- Use sampling for high-volume metrics
- Enable compression for log storage
- Implement metric aggregation

### Resource Requirements
- **CPU**: 2 cores minimum
- **Memory**: 4GB recommended
- **Storage**: 50GB for 30-day retention
- **Network**: 100Mbps recommended

## Security

### Best Practices
- Use TLS for all connections
- Implement authentication for APIs
- Restrict network access
- Rotate credentials regularly
- Audit log access

### API Authentication
Configure authentication in environment:
```bash
export API_KEY=your-secure-key
export GRAFANA_API_KEY=your-grafana-key
```

## Development

### Running Tests
```bash
pytest tests/ -v --cov=monitoring
```

### Code Style
```bash
black *.py
mypy *.py
pylint *.py
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit pull request

## License

Copyright © 2025 SutazAI Team. All rights reserved.

## Support

For issues and questions:
- Create an issue in the repository
- Contact: devops.team@sutazaiapp.com
- Documentation: [Internal Wiki]

---

*Built with ❤️ for reliable MCP automation monitoring*