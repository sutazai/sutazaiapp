# SutazAI External Services Integration Guide

## Overview

This guide provides comprehensive instructions for integrating external containers and services into the SutazAI ecosystem. The integration framework allows SutazAI to expand its capabilities by connecting to existing services while maintaining system stability and avoiding port conflicts.

## Architecture

### Integration Layers

```
┌─────────────────────────────────────────────────────────────┐
│                    External Services                         │
│  (PostgreSQL, MySQL, Redis, Kafka, Elasticsearch, etc.)     │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 SutazAI Adapter Layer                        │
│  (Protocol Translation, Monitoring, Health Checks)          │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│               SutazAI Integration Network                    │
│  (Service Mesh, API Gateway, Service Discovery)             │
└─────────────────────┬───────────────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────────────┐
│                 SutazAI Core Services                        │
│  (AI Agents, Hardware Optimizer, Monitoring)                │
└─────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. Discover External Services

```bash
# Run service discovery
./scripts/integrate-external-services.sh --discover

# This will:
# - Scan for running Docker containers
# - Detect services on common ports
# - Generate integration configuration
```

### 2. Deploy Integration Infrastructure

```bash
# Deploy all integration components
./scripts/integrate-external-services.sh --deploy

# This will:
# - Create configuration files
# - Build adapter images
# - Start integration services
# - Display service URLs
```

### 3. Monitor Integration

```bash
# Setup monitoring dashboards
./scripts/integrate-external-services.sh --monitor

# Access dashboards at:
# - Grafana: http://localhost:10050 (admin/admin)
# - Prometheus: http://localhost:10010
```

## Port Allocation Strategy

SutazAI uses a structured port allocation system to avoid conflicts:

| Port Range    | Purpose                    | Examples                           |
|---------------|----------------------------|------------------------------------|
| 10000-10099   | Core Integration Services  | API Gateway (10001), Discovery (10000) |
| 10100-10199   | Database Adapters          | PostgreSQL (10100), MySQL (10101)  |
| 10200-10299   | Application Services       | Custom app adapters                |
| 10300-10399   | Message Queue Adapters     | RabbitMQ (10120), Kafka (10122)   |
| 10400-10499   | Monitoring Services        | Prometheus (10010), Grafana (10050)|
| 10500-10599   | Utility Services           | Reserved for future use            |

## Service Integration Examples

### Example 1: PostgreSQL Integration

```yaml
# 1. External PostgreSQL is running on port 5432
# 2. SutazAI creates an adapter at port 10100
# 3. The adapter provides:
#    - Connection pooling
#    - Query monitoring
#    - Performance metrics
#    - Health checks

# Access PostgreSQL through SutazAI:
curl http://localhost:10100/health
curl http://localhost:10100/databases
curl http://localhost:10100/stats
```

### Example 2: Redis Integration

```yaml
# 1. External Redis is running on port 6379
# 2. SutazAI creates an adapter at port 10110
# 3. The adapter provides:
#    - Command monitoring
#    - Memory analysis
#    - Pub/Sub support
#    - Key pattern analysis

# Access Redis through SutazAI:
curl http://localhost:10110/health
curl http://localhost:10110/info
curl http://localhost:10110/keys
```

## Manual Service Integration

### Step 1: Create Service Adapter

```dockerfile
# Create custom adapter based on template
FROM sutazaiapp/service-adapter:latest

# Configure for your service
ENV TARGET_HOST=my-service
ENV TARGET_PORT=8080
ENV ADAPTER_PORT=10200
```

### Step 2: Register Service

```python
# Register in service registry
python services/external-service-registry.py --add my-service
```

### Step 3: Configure Monitoring

```yaml
# Add to Prometheus targets
- job_name: 'my-service'
  static_configs:
    - targets: ['my-service-adapter:9090']
```

## API Gateway Routes

All external services are accessible through the unified API Gateway:

```bash
# Base URL: http://localhost:10001

# Database Services
GET /postgres/health
GET /mysql/status
GET /mongodb/collections

# Cache Services
GET /redis/info
GET /memcached/stats

# Message Queues
GET /rabbitmq/queues
GET /kafka/topics

# Search Services
GET /elasticsearch/indices
```

## Monitoring and Observability

### Metrics Collection

All adapters expose Prometheus metrics at `/metrics`:

- Request count and latency
- Error rates
- Service-specific metrics
- Resource utilization

### Log Aggregation

Logs are collected by Loki and accessible through Grafana:

1. Navigate to http://localhost:10050
2. Select "Explore" from the left menu
3. Choose "Loki" as the data source
4. Query logs using LogQL

### Health Checks

Every adapter provides health endpoints:

```bash
# Check adapter health
curl http://localhost:10100/health

# Response:
{
  "status": "healthy",
  "target": {
    "host": "postgres",
    "port": 5432,
    "reachable": true
  },
  "adapter": {
    "version": "1.0.0",
    "uptime": 3600
  }
}
```

## Security Considerations

### Network Isolation

- External services remain on their original networks
- Adapters act as secure proxies
- All communication is logged and monitored

### Authentication

- Service-to-service authentication via mTLS
- API Gateway handles external authentication
- Secrets stored in Consul KV store

### Access Control

- Role-based access control (RBAC)
- Service-specific permissions
- Audit logging for all operations

## Troubleshooting

### Common Issues

1. **Service Discovery Fails**
   ```bash
   # Check Docker permissions
   sudo usermod -aG docker $USER
   
   # Verify Docker socket access
   ls -la /var/run/docker.sock
   ```

2. **Adapter Connection Errors**
   ```bash
   # Check target service accessibility
   docker exec sutazai-postgres-adapter ping postgres
   
   # Verify network connectivity
   docker network inspect sutazai-integration
   ```

3. **Port Conflicts**
   ```bash
   # Find process using port
   sudo lsof -i :10100
   
   # Check Docker port bindings
   docker ps --format "table {{.Names}}\t{{.Ports}}"
   ```

### Debug Mode

Enable debug logging for adapters:

```yaml
# Set in docker-compose.external-integration.yml
environment:
  - LOG_LEVEL=DEBUG
  - TRACE_ENABLED=true
```

## Advanced Configuration

### Custom Health Checks

```python
# Implement custom health logic
class CustomHealthCheck:
    async def check(self):
        # Custom validation logic
        return {
            'custom_metric': value,
            'threshold_met': True
        }
```

### Performance Tuning

```yaml
# Adapter configuration
performance:
  connection_pool_size: 50
  request_timeout: 30
  circuit_breaker:
    failure_threshold: 5
    recovery_timeout: 60
```

## Maintenance

### Regular Tasks

1. **Weekly**: Review adapter logs for errors
2. **Monthly**: Update adapter images
3. **Quarterly**: Audit service registry

### Backup and Recovery

```bash
# Backup service registry
consul snapshot save backup.snap

# Restore service registry
consul snapshot restore backup.snap
```

## Contributing

To add support for a new service type:

1. Create adapter template in `/docker/adapters/`
2. Add service definition to `SERVICE_TEMPLATES`
3. Update documentation
4. Submit pull request

## Support

For issues or questions:

1. Check logs: `docker logs sutazai-[service-name]`
2. Review health endpoints
3. Consult integration dashboard
4. File issue with debug information