# SutazAI Monitoring System

This directory contains the configuration and setup files for the SutazAI monitoring and logging system.

## Overview

The monitoring system provides comprehensive observability for the SutazAI platform, including:

- Metrics collection and visualization via Prometheus and Grafana
- Log aggregation and analysis via ELK stack (Elasticsearch, Logstash, Kibana)
- Error tracking and alerting via Sentry
- System metrics collection via Node Exporter and cAdvisor

## Setup Instructions

### Prerequisites

- Docker and Docker Compose
- Python 3.8+
- Sufficient disk space for logs and metrics (at least 10GB recommended)
- Network access to ports 3000, 5601, 9090, 9000

### Installation

1. Run the setup script to install dependencies and create required directories:

```bash
./setup_monitoring.sh
```

2. Update the configuration in `.env` file if needed.

3. Start the monitoring system:

```bash
./start_monitoring.sh
```

### Access Points

After starting the monitoring system, you can access the following interfaces:

- Grafana: http://localhost:3000 (default credentials: admin/sutazai)
- Kibana: http://localhost:5601 (default credentials: elastic/sutazaisecure)
- Prometheus: http://localhost:9090
- Sentry: http://localhost:9000

## Directory Structure

- `prometheus/`: Prometheus configuration and data
- `grafana/`: Grafana configuration, dashboards, and data
- `elk/`: ELK stack configuration and data
  - `logstash/`: Logstash configuration and pipeline
  - `filebeat/`: Filebeat configuration
  - `metricbeat/`: Metricbeat configuration
- `sentry/`: Sentry configuration and data

## Configuration

### Prometheus

Prometheus is configured to scrape metrics from:

- SutazAI backend API
- Node Exporter (system metrics)
- cAdvisor (container metrics)

Edit `prometheus/prometheus.yml` to adjust scraping intervals or add new targets.

### Grafana

Grafana comes with pre-built dashboards for:

- System Overview: CPU, memory, disk, and network usage
- API Performance: Request rates, latencies, and error rates
- Model Performance: Model inference time and error rates

Dashboards are stored in `grafana/dashboards/`.

### ELK Stack

The ELK stack is configured to collect and process logs from:

- Application logs in JSON format
- System logs via Filebeat
- Docker container logs
- Metrics via Metricbeat

Logstash pipelines are defined in `elk/logstash/pipeline/`.

### Sentry

Sentry is configured for error tracking. To enable it:

1. Rename `sentry/.env.example` to `sentry/.env`
2. Update the configuration values
3. Update the `SENTRY_DSN` value in the main `.env` file

## Maintenance

### Log Rotation

Logs are automatically rotated to prevent disk space issues. Adjust rotation settings in:

- `filebeat/filebeat.yml` for application logs
- Elasticsearch ILM policies for indexed logs

### Backup

To back up monitoring data:

1. For Prometheus: `docker-compose exec prometheus promtool tsdb snapshot /prometheus -o /prometheus/snapshots`
2. For Elasticsearch: Configure snapshots in Kibana or use the Elasticsearch snapshot API
3. For Grafana: Export dashboards and settings from the UI

## Troubleshooting

### Common Issues

1. **Services fail to start**:
   - Check Docker logs: `docker-compose logs <service_name>`
   - Verify port availability: `netstat -tulpn | grep <port>`
   - Check disk space: `df -h`

2. **No metrics in Grafana**:
   - Check Prometheus targets: http://localhost:9090/targets
   - Verify Prometheus data source in Grafana
   - Check network connectivity between services

3. **Missing logs in Kibana**:
   - Check Filebeat status: `docker-compose logs filebeat`
   - Verify Elasticsearch indices: `curl -XGET 'localhost:9200/_cat/indices?v'`
   - Check Logstash pipeline: `docker-compose logs logstash`

## Additional Resources

For more detailed information, see:

- [Full Documentation](/docs/monitoring.md)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)
- [ELK Stack Documentation](https://www.elastic.co/guide/index.html)
- [Sentry Documentation](https://docs.sentry.io/) 