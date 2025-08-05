# Autonomous System Controller

## Overview

The Autonomous System Controller provides centralized management and monitoring for the SutazAI automation system. It implements real-time metrics collection, automated error recovery, and resource optimization.

## Implementation Details

### Core Components

1. **Monitoring Service**
```typescript
interface MonitoringService {
  collectMetrics(): Promise<SystemMetrics>;
  detectAnomalies(metrics: SystemMetrics): AnomalyReport;
  triggerAlerts(anomalies: AnomalyReport): void;
}
```

2. **Resource Manager**
```typescript
interface ResourceManager {
  allocateResources(request: ResourceRequest): Promise<Resources>;
  optimizeUsage(current: Resources): OptimizationPlan;
  enforceQuotas(usage: Resources): void;
}
```

### Control Loops

| Loop | Interval | Purpose |
|------|----------|---------|
| Monitoring | 5s | System metrics collection |
| Decision | 10s | Resource allocation |
| Optimization | 5min | Usage optimization |
| Safety | 5s | Constraint validation |

### API Endpoints 

Base URL: `http://controller:8080/api/v1`

| Endpoint | Method | Description |
|----------|---------|-------------|
| `/health` | GET | Controller health status |
| `/metrics` | GET | Current system metrics |
| `/decisions` | GET | Recent allocation decisions |

## Configuration

```yaml
# filepath: /opt/sutazaiapp/config/controller.yaml
monitoring:
  interval: 5  # seconds
  metrics:
    - cpu_usage
    - memory_usage
    - error_rate

decisions:
  confidence_threshold: 0.7
  max_allocation: 90%
  
safety:
  cpu_emergency: 95%
  memory_emergency: 90%
```

## Deployment

```bash
# Deploy using Docker Compose
docker compose -f docker/controller.yml up -d

# Verify deployment
curl http://localhost:8080/api/v1/health
```

## Monitoring

The controller exposes Prometheus metrics at `/metrics`:

```
# HELP controller_decisions_total Total number of allocation decisions
# TYPE controller_decisions_total counter
controller_decisions_total{type="scale_up"} 12
controller_decisions_total{type="scale_down"} 8
```

## Maintenance

- Run health checks daily: `./scripts/check-controller.sh`
- Review logs weekly: `docker logs controller`
- Update configuration monthly

## Owner

- Primary: @devops-lead
- Backup: @platform-team

See [Architecture Overview](../architecture.md) for system context.