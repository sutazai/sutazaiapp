# SutazAI Distributed Architecture Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the distributed SutazAI architecture, which transforms the system from a single-instance deployment to a highly available, horizontally scalable distributed system.

## Architecture Summary

### Key Components
- **Load Balancing**: HAProxy with automatic failover
- **Service Discovery**: Consul cluster (3 nodes)
- **Message Queue**: RabbitMQ cluster with mirrored queues
- **Cache Layer**: Redis cluster (6 nodes - 3 masters, 3 slaves)
- **Database**: PostgreSQL with streaming replication
- **AI Services**: Horizontally scaled Ollama and AI agents
- **Task Processing**: Distributed Celery workers
- **Monitoring**: Prometheus, Grafana, Jaeger for observability
- **Auto-scaling**: Custom autoscaler based on metrics

## Prerequisites

### System Requirements
- Docker Engine 20.10+ with Swarm mode
- Docker Compose 2.0+
- Minimum 32GB RAM (for development)
- 100GB available disk space
- Network connectivity between nodes (if multi-node)

### Required Tools
```bash
# Install required tools
sudo apt-get update
sudo apt-get install -y curl jq bc python3-pip
pip3 install docker pyyaml prometheus-api-client consul-py celery redis
```

## Deployment Steps

### 1. Initialize Docker Swarm

```bash
# On the manager node
docker swarm init --advertise-addr <MANAGER_IP>

# Save the join token for worker nodes
docker swarm join-token worker
```

### 2. Label Nodes for Service Placement

```bash
# Label nodes for specific services
docker node update --label-add consul=server1 <NODE_ID>
docker node update --label-add redis=master1 <NODE_ID>
docker node update --label-add postgres=primary <NODE_ID>
docker node update --label-add gpu=true <GPU_NODE_ID>  # For Ollama
```

### 3. Create Required Networks and Volumes

```bash
# Create overlay networks
docker network create --driver overlay --attachable ai-mesh
docker network create --driver overlay --attachable data-tier
docker network create --driver overlay --attachable cache-tier
docker network create --driver overlay --attachable monitoring

# Create volumes
docker volume create consul-data-1
docker volume create consul-data-2
docker volume create consul-data-3
docker volume create shared-models
```

### 4. Deploy Configuration Files

```bash
# Ensure all configuration files are in place
cd /opt/sutazaiapp
ls -la config/{consul,haproxy,redis,rabbitmq,prometheus,autoscaler}/
```

### 5. Deploy the Distributed Stack

```bash
# Deploy using Docker Stack
docker stack deploy -c docker-compose.distributed.yml sutazai

# Or use the migration script for zero-downtime migration
./scripts/zero-downtime-migration.sh
```

### 6. Verify Deployment

```bash
# Check service status
docker service ls

# Check Consul cluster
curl http://localhost:8500/v1/status/leader

# Check Redis cluster
docker exec $(docker ps -q -f name=redis-master-1) redis-cli cluster info

# Check RabbitMQ cluster
curl -u admin:admin http://localhost:15672/api/cluster-name
```

## Service Endpoints

### External Endpoints
- **Application**: http://localhost:80 (HAProxy)
- **Consul UI**: http://localhost:8500
- **RabbitMQ Management**: http://localhost:15672
- **Prometheus**: http://localhost:9090
- **Grafana**: http://localhost:3000
- **Jaeger UI**: http://localhost:16686
- **Flower (Celery)**: http://localhost:5555

### Internal Service Discovery
All services register with Consul and can be discovered via:
- DNS: `<service-name>.service.consul`
- HTTP API: `http://consul:8500/v1/catalog/service/<service-name>`

## Scaling Services

### Manual Scaling
```bash
# Scale AI agents
docker service scale sutazai_ai-agent=20

# Scale Celery workers
docker service scale sutazai_celery-worker=10

# Scale Ollama instances
docker service scale sutazai_ollama=5
```

### Auto-scaling
The autoscaler automatically adjusts service replicas based on metrics:

```bash
# Start autoscaler
docker service create \
  --name autoscaler \
  --mount type=bind,source=/var/run/docker.sock,target=/var/run/docker.sock \
  --mount type=bind,source=/opt/sutazaiapp/config/autoscaler,target=/config \
  sutazai/autoscaler:latest
```

### Scaling Rules
Edit `/opt/sutazaiapp/config/autoscaler/scaling-rules.yaml` to adjust:
- CPU/Memory thresholds
- Queue length triggers
- Min/max replicas
- Cooldown periods

## Monitoring & Observability

### Prometheus Queries
Key metrics to monitor:

```promql
# AI Agent CPU usage
avg(rate(container_cpu_usage_seconds_total{service="ai-agent"}[5m])) * 100

# Task queue length
sum(celery_queue_length{queue_name="inference"})

# Service response time (p95)
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Redis cluster status
redis_cluster_state{job="redis"}
```

### Grafana Dashboards
Import provided dashboards:
1. System Overview Dashboard
2. AI Services Dashboard
3. Queue Metrics Dashboard
4. Distributed Tracing Dashboard

### Distributed Tracing
View request traces across services:
1. Open Jaeger UI: http://localhost:16686
2. Select service: `ai-agent`
3. Find traces to analyze request flow

## High Availability Features

### Service Redundancy
- **Consul**: 3-node cluster with automatic leader election
- **RabbitMQ**: 3-node cluster with queue mirroring
- **Redis**: 3 masters with 3 slaves, automatic failover via Sentinel
- **PostgreSQL**: Primary with 2 read replicas
- **AI Services**: Multiple instances behind load balancers

### Failure Scenarios
The system handles various failures automatically:

1. **Node Failure**: Services redistribute to healthy nodes
2. **Service Crash**: Docker Swarm restarts failed services
3. **Network Partition**: Consul and RabbitMQ handle split-brain scenarios
4. **Database Failure**: Automatic failover to replica

## Data Management

### Backup Strategy
```bash
# Backup all critical data
./scripts/backup-distributed-data.sh

# Backup includes:
# - PostgreSQL databases
# - Redis snapshots
# - RabbitMQ definitions
# - Consul KV store
# - Vector database collections
```

### Data Partitioning
- **PostgreSQL**: Range partitioning by tenant_id
- **Redis**: Consistent hashing across masters
- **Vector DBs**: Sharded by collection name

## Security Considerations

### Network Security
- All internal communication on encrypted overlay networks
- mTLS between services (when enabled)
- API Gateway handles external authentication

### Access Control
```bash
# Set passwords via environment variables
export POSTGRES_PASSWORD=<secure_password>
export RABBITMQ_PASSWORD=<secure_password>
export REDIS_PASSWORD=<secure_password>
export GRAFANA_PASSWORD=<secure_password>
```

### Secrets Management
```bash
# Create Docker secrets
echo "mypassword" | docker secret create postgres_password -
echo "mytoken" | docker secret create consul_encrypt_key -

# Reference in services
# password_file: /run/secrets/postgres_password
```

## Troubleshooting

### Common Issues

1. **Services not starting**
   ```bash
   # Check service logs
   docker service logs sutazai_<service-name>
   
   # Check placement constraints
   docker service ps sutazai_<service-name>
   ```

2. **Consul cluster not forming**
   ```bash
   # Check Consul logs
   docker service logs sutazai_consul-server-1
   
   # Verify network connectivity
   docker exec <consul-container> consul members
   ```

3. **Redis cluster issues**
   ```bash
   # Check cluster status
   docker exec <redis-container> redis-cli cluster nodes
   
   # Fix cluster
   docker exec <redis-container> redis-cli --cluster fix localhost:6379
   ```

4. **High memory usage**
   ```bash
   # Check container stats
   docker stats --no-stream
   
   # Adjust memory limits in docker-compose.distributed.yml
   ```

### Health Checks
```bash
# Run comprehensive health check
./scripts/distributed-health-check.sh

# Check specific service
curl http://localhost:8500/v1/health/service/<service-name>
```

## Performance Tuning

### Container Resources
Adjust in `docker-compose.distributed.yml`:
```yaml
deploy:
  resources:
    limits:
      cpus: '4'
      memory: 8G
    reservations:
      cpus: '2'
      memory: 4G
```

### Database Optimization
```sql
-- PostgreSQL connection pooling
ALTER SYSTEM SET max_connections = 1000;
ALTER SYSTEM SET shared_buffers = '4GB';
ALTER SYSTEM SET effective_cache_size = '12GB';
```

### Redis Optimization
```conf
# In redis-cluster.conf
maxmemory 4gb
maxmemory-policy allkeys-lru
io-threads 4
io-threads-do-reads yes
```

## Maintenance

### Rolling Updates
```bash
# Update service image
docker service update \
  --image sutazai/ai-agent:v2 \
  --update-parallelism 2 \
  --update-delay 30s \
  sutazai_ai-agent
```

### Cleanup
```bash
# Remove unused images
docker system prune -a

# Clean old logs
find /opt/sutazaiapp/logs -name "*.log" -mtime +30 -delete

# Vacuum databases
docker exec <postgres-container> vacuumdb -U sutazai -d sutazai -z
```

## Disaster Recovery

### Full System Restore
```bash
# Stop all services
docker stack rm sutazai

# Restore from backup
./scripts/restore-from-backup.sh <backup-timestamp>

# Redeploy services
docker stack deploy -c docker-compose.distributed.yml sutazai
```

### Partial Recovery
```bash
# Recover specific service
docker service scale sutazai_<service>=0
docker service scale sutazai_<service>=<desired-replicas>

# Recover from Redis failure
docker exec <redis-master> redis-cli BGREWRITEAOF
```

## Migration from Single Instance

Use the provided zero-downtime migration script:
```bash
cd /opt/sutazaiapp
./scripts/zero-downtime-migration.sh
```

This script will:
1. Backup current state
2. Deploy distributed infrastructure
3. Migrate data
4. Gradually shift traffic
5. Decommission old services

## Support and Documentation

### Additional Resources
- Architecture diagram: `/opt/sutazaiapp/distributed_system_analysis_report.md`
- API documentation: http://localhost:8000/docs
- Monitoring guide: `/opt/sutazaiapp/docs/monitoring-guide.md`
- Scaling guide: `/opt/sutazaiapp/docs/scaling-guide.md`

### Getting Help
1. Check service logs
2. Review health check outputs
3. Consult Prometheus/Grafana metrics
4. Check distributed traces in Jaeger

For production deployments, ensure you have:
- Multi-node Docker Swarm cluster
- Persistent storage solutions
- Network load balancer
- Backup and disaster recovery plan
- Security hardening applied