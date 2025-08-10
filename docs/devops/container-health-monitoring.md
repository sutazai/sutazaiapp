# Container Health Monitoring Alert Setup

## Overview
Monitoring alert configuration for critical agent containers after resolving RabbitMQ blocking issues.

## Fixed Containers Status
- **task-assignment-coordinator** (Port 8551): HEALTHY
- **resource-arbitration-agent** (Port 8588): HEALTHY

## Health Check Endpoints
```bash
# Task Assignment Coordinator
curl http://127.0.0.1:8551/health

# Resource Arbitration Agent
curl http://127.0.0.1:8588/health
```

## RabbitMQ Integration
Both containers successfully establish message queues:
- `agent.task-assignment-coordinator`
- `agent.resource-arbitration-agent`

## Monitoring Commands
```bash
# Check container status
docker ps --filter name=task-assignment --filter name=resource-arbitration

# Continuous health monitoring
watch -n 5 'docker ps --filter name=task-assignment --filter name=resource-arbitration --format "table {{.Names}}\t{{.Status}}"'

# Queue monitoring
docker exec sutazai-rabbitmq rabbitmqctl list_queues
```

## Alert Thresholds
- Container restart count > 3 in 10 minutes
- Health endpoint response time > 5 seconds
- Health endpoint returning non-200 status

## Operational Notes
- Fix verified through container restart test
- Both containers maintain health through restart cycle
- Message processing initialization successful
- No blocking RabbitMQ connections detected

Last Updated: 2025-08-09T20:29:00Z