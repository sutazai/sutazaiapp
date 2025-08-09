# SutazAI Deployment & Operations Playbook

**Version:** 1.0  
**Last Updated:** August 8, 2025  
**Based on:** CLAUDE.md System Truth Document  

## Executive Summary

This playbook provides comprehensive deployment and operational procedures for the SutazAI system based on the **actual system state** as of August 2025. The system currently runs 28 containers out of 59 defined services, with 7 agent stub services and a TinyLlama model (not gpt-oss as documentation claims).

**Key System Realities:**
- 28 containers actually running (verified by testing)
- 7 agent services are Flask stubs with health endpoints only
- TinyLlama 637MB model loaded (NOT gpt-oss)
- No actual AI logic - agents return hardcoded JSON responses
- PostgreSQL has empty tables (no schema deployed)
- ChromaDB has connection issues
- Service mesh (Kong/Consul/RabbitMQ) running but not configured

---

## 1. Deployment Procedures

### 1.1 Pre-Deployment Checklist

**Infrastructure Requirements:**
- [ ] Docker Engine 20.10+ installed
- [ ] Docker Compose 2.0+ installed
- [ ] Minimum 8GB RAM available
- [ ] 50GB disk space available
- [ ] Network ports 10000-10229 available

**System Prerequisites:**
```bash
# Verify Docker installation
docker --version
docker-compose --version

# Check available resources
docker system df
free -h
df -h

# Create required network (if not exists)
docker network create sutazai-network 2>/dev/null || true
```

**Configuration Verification:**
- [ ] Docker daemon running
- [ ] No port conflicts on target ports
- [ ] Firewall rules configured for internal communication
- [ ] System time synchronized

### 1.2 Docker Compose Deployment Steps

**Step 1: Environment Preparation**
```bash
cd /opt/sutazaiapp

# Ensure network exists
docker network create sutazai-network 2>/dev/null || true

# Verify docker-compose.yml exists and is valid
docker-compose config --quiet
```

**Step 2: Service Startup Sequence**

Services must start in specific order due to dependencies:

```bash
# Phase 1: Core Infrastructure (30 seconds)
docker-compose up -d postgres redis neo4j
sleep 30

# Phase 2: Service Mesh (20 seconds)
docker-compose up -d kong consul rabbitmq
sleep 20

# Phase 3: Monitoring Stack (15 seconds)
docker-compose up -d prometheus grafana loki alertmanager node-exporter cadvisor blackbox-exporter
sleep 15

# Phase 4: Vector Databases (20 seconds)
docker-compose up -d qdrant faiss chromadb
sleep 20

# Phase 5: AI Services (30 seconds)
docker-compose up -d ollama
sleep 30

# Phase 6: Application Layer (15 seconds)
docker-compose up -d backend frontend
sleep 15

# Phase 7: Agent Stubs (10 seconds)
docker-compose up -d ai-agent-orchestrator multi-agent-coordinator resource-arbitration-agent task-assignment-coordinator hardware-resource-optimizer ollama-integration-specialist ai-metrics-exporter
sleep 10
```

**Step 3: Complete System Deployment**
```bash
# Alternative: Start all services at once (less reliable)
docker-compose up -d

# Wait for initialization
sleep 60
```

### 1.3 Service Startup Dependencies

**Critical Dependencies:**
- Backend → PostgreSQL, Redis, Ollama
- Agents → Backend API (for registration)
- Monitoring → All services (for metrics collection)
- Frontend → Backend API

**Known Startup Issues:**
- ChromaDB: Takes 60+ seconds to initialize
- Ollama: Model loading takes 30-45 seconds
- Backend: May show "degraded" if Ollama not ready

### 1.4 Health Check Verification

**Core Services Health Check:**
```bash
# Backend API (should return degraded initially)
curl -s http://127.0.0.1:10010/health | jq

# Database connectivity
docker exec sutazai-postgres pg_isready -U sutazai

# Redis connectivity
docker exec sutazai-redis redis-cli ping

# Neo4j browser access
curl -s http://127.0.0.1:10002/browser/

# Ollama model status
curl -s http://127.0.0.1:10104/api/tags | jq
```

**Agent Services Health Check:**
```bash
# Agent stubs (all return {"status": "healthy"})
curl -s http://127.0.0.1:8589/health  # AI Agent Orchestrator
curl -s http://127.0.0.1:8587/health  # Multi-Agent Coordinator
curl -s http://127.0.0.1:8588/health  # Resource Arbitration
curl -s http://127.0.0.1:8551/health  # Task Assignment
curl -s http://127.0.0.1:8002/health  # Hardware Optimizer
curl -s http://127.0.0.1:11015/health # Ollama Integration
curl -s http://127.0.0.1:11063/health # AI Metrics (may fail - UNHEALTHY)
```

**Monitoring Stack Health:**
```bash
# Prometheus targets
curl -s http://127.0.0.1:10200/api/v1/targets | jq '.data.activeTargets | length'

# Grafana health
curl -s http://127.0.0.1:10201/api/health | jq

# Loki ready
curl -s http://127.0.0.1:10202/ready
```

### 1.5 Rollback Procedures

**Immediate Rollback (Critical Issues):**
```bash
# Stop all services
docker-compose down

# Remove problematic containers
docker container prune -f

# Restart with last known good configuration
docker-compose up -d
```

**Service-Specific Rollback:**
```bash
# Rollback specific service
docker-compose stop [service-name]
docker-compose rm -f [service-name]
docker-compose up -d [service-name]
```

**Complete System Rollback:**
```bash
# Stop and remove all containers
docker-compose down -v

# Remove all images (if needed)
docker image prune -a -f

# Redeploy from clean state
docker-compose up -d
```

---

## 2. Daily Operations

### 2.1 System Health Monitoring

**Daily Health Check Script:**
```bash
#!/bin/bash
# daily_health_check.sh

echo "=== SutazAI Daily Health Report $(date) ==="

# Container status
echo "Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai

# Service health endpoints
echo -e "\nCore Service Health:"
curl -s -m 5 http://127.0.0.1:10010/health | jq -r '.status // "ERROR"' | head -1
curl -s -m 5 http://127.0.0.1:10104/api/tags | jq -r '.models[0].name // "ERROR"' 2>/dev/null

# Database connectivity
echo -e "\nDatabase Status:"
docker exec sutazai-postgres pg_isready -U sutazai | cut -d' ' -f3-
docker exec sutazai-redis redis-cli ping 2>/dev/null || echo "Redis: ERROR"

# Resource utilization
echo -e "\nResource Usage:"
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}" | head -10

# Disk space
echo -e "\nDisk Usage:"
df -h | grep -E "(/$|/opt)"

echo "=== Health Check Complete ==="
```

### 2.2 Service Status Monitoring

**Container Status Dashboard:**
```bash
# View running containers with resource usage
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}\t{{.BlockIO}}"

# Check restart counts (high restarts indicate issues)
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.RunningFor}}"

# View recent container events
docker events --since 1h --filter type=container
```

### 2.3 Log Monitoring and Analysis

**Critical Log Locations:**
```bash
# Backend API logs (key for system issues)
docker-compose logs -f backend

# Ollama service logs (model loading issues)
docker-compose logs -f ollama

# Database logs (connection issues)
docker-compose logs -f postgres redis neo4j

# Agent stub logs (minimal output expected)
docker-compose logs -f ai-agent-orchestrator
```

**Log Analysis Commands:**
```bash
# Check for errors in last hour
docker-compose logs --since 1h | grep -i error

# Monitor connection issues
docker-compose logs -f | grep -i "connection\|timeout\|refused"

# Track restart events
docker-compose logs | grep -i "started\|stopped\|restart"
```

### 2.4 Performance Baseline Monitoring

**System Metrics Collection:**
```bash
# CPU and memory usage trends
docker stats --no-stream | awk 'NR>1 {cpu+=$3; mem+=$7} END {print "Avg CPU:", cpu/NR "%", "Total Mem:", mem "MB"}'

# Network I/O monitoring
docker stats --no-stream --format "table {{.Name}}\t{{.NetIO}}" | sort -k2 -hr

# Container resource limits
docker inspect $(docker ps -q) | jq -r '.[] | "\(.Name): CPU=\(.HostConfig.CpuShares) MEM=\(.HostConfig.Memory)"'
```

**Performance Thresholds:**
- CPU > 80%: Investigation required
- Memory > 85%: Scale or optimize
- Restart count > 5/day: Debug required
- Response time > 5s: Performance issue

---

## 3. Service Management

### 3.1 Starting/Stopping Services

**Individual Service Management:**
```bash
# Start specific service
docker-compose up -d [service-name]

# Stop specific service
docker-compose stop [service-name]

# Restart with fresh container
docker-compose restart [service-name]

# Remove and recreate container
docker-compose rm -sf [service-name]
docker-compose up -d [service-name]
```

**Service Groups:**
```bash
# Core infrastructure
CORE_SERVICES="postgres redis neo4j"
docker-compose up -d $CORE_SERVICES

# Monitoring stack
MONITORING_SERVICES="prometheus grafana loki alertmanager"
docker-compose up -d $MONITORING_SERVICES

# Agent stubs
AGENT_SERVICES="ai-agent-orchestrator multi-agent-coordinator resource-arbitration-agent task-assignment-coordinator hardware-resource-optimizer"
docker-compose up -d $AGENT_SERVICES
```

### 3.2 Graceful Shutdown Sequences

**Planned Maintenance Shutdown:**
```bash
#!/bin/bash
# graceful_shutdown.sh

echo "Starting graceful shutdown..."

# Phase 1: Stop frontend (prevent new requests)
docker-compose stop frontend
sleep 5

# Phase 2: Stop agent stubs
docker-compose stop ai-agent-orchestrator multi-agent-coordinator resource-arbitration-agent task-assignment-coordinator hardware-resource-optimizer ollama-integration-specialist ai-metrics-exporter
sleep 10

# Phase 3: Stop backend (complete ongoing requests)
docker-compose stop backend
sleep 15

# Phase 4: Stop AI services
docker-compose stop ollama
sleep 10

# Phase 5: Stop databases (ensure data consistency)
docker-compose stop postgres redis neo4j
sleep 10

# Phase 6: Stop monitoring and infrastructure
docker-compose stop prometheus grafana loki alertmanager kong consul rabbitmq
sleep 5

# Phase 7: Stop remaining services
docker-compose down

echo "Graceful shutdown complete"
```

### 3.3 Service Dependency Management

**Dependency Chain:**
1. **Core Infrastructure**: postgres, redis, neo4j
2. **Service Mesh**: kong, consul, rabbitmq
3. **AI Services**: ollama
4. **Application**: backend, frontend
5. **Agent Stubs**: All agent services
6. **Monitoring**: prometheus, grafana, loki

**Dependency Verification:**
```bash
# Check database connections before starting backend
docker exec sutazai-postgres pg_isready -U sutazai
docker exec sutazai-redis redis-cli ping

# Verify Ollama model loaded before backend
curl -s http://127.0.0.1:10104/api/tags | jq -r '.models[].name'

# Check backend health before starting agents
curl -s http://127.0.0.1:10010/health | jq -r '.status'
```

### 3.4 Container Lifecycle Management

**Container Health Monitoring:**
```bash
# Monitor container health status
docker ps --filter "health=unhealthy"

# View container inspection
docker inspect [container-name] | jq '.State.Health'

# Container resource usage over time
docker exec [container-name] cat /proc/meminfo | grep MemAvailable
```

**Container Cleanup:**
```bash
# Remove stopped containers
docker container prune -f

# Remove unused images
docker image prune -a -f

# Remove unused volumes (CAUTION: Data loss)
docker volume prune -f

# Remove unused networks
docker network prune -f
```

---

## 4. Incident Response

### 4.1 Incident Classification

**P0 - Critical System Down**
- Entire system inaccessible
- Multiple core services failing
- Data corruption detected
- Security breach confirmed

**P1 - Major Functionality Impaired**
- Backend API returning 50x errors
- Database connection failures
- Ollama service unavailable
- Frontend completely broken

**P2 - Minor Functionality Issues**
- Individual agent stub failures
- Performance degradation
- Monitoring gaps
- Configuration drift

**P3 - Maintenance and Improvements**
- Documentation updates
- Performance optimization
- Security hardening
- Feature requests

### 4.2 Escalation Procedures

**P0 Escalation Path:**
1. **Immediate (0-5 minutes)**: Execute emergency procedures
2. **5-15 minutes**: Assess impact and begin recovery
3. **15-30 minutes**: Escalate to senior operations
4. **30+ minutes**: Engage external support if needed

**Communication Channels:**
- Slack: #sutazai-incidents
- Email: ops-team@company.com
- Phone: Emergency contact list
- Status Page: Update system status

### 4.3 Common Incident Scenarios

**Scenario 1: Backend API Unresponsive**
```bash
# Diagnosis
curl -I http://127.0.0.1:10010/health
docker-compose logs backend | tail -50

# Quick Resolution
docker-compose restart backend

# Root Cause Investigation
docker exec sutazai-backend cat /proc/meminfo
docker stats sutazai-backend --no-stream
```

**Scenario 2: Database Connection Pool Exhausted**
```bash
# Diagnosis
docker exec sutazai-postgres psql -U sutazai -c "SELECT count(*) FROM pg_stat_activity;"
docker-compose logs postgres | grep -i "connection\|pool"

# Resolution
docker-compose restart backend postgres
```

**Scenario 3: Ollama Model Loading Failure**
```bash
# Diagnosis
curl http://127.0.0.1:10104/api/tags
docker-compose logs ollama | tail -30

# Resolution - Reload TinyLlama model
docker exec sutazai-ollama ollama pull tinyllama
docker-compose restart ollama backend
```

**Scenario 4: ChromaDB Connection Issues**
```bash
# Diagnosis
docker-compose logs chromadb | tail -20
curl -I http://127.0.0.1:10100/api/v1/heartbeat

# Resolution
docker-compose stop chromadb
docker volume rm sutazai-chromadb-data
docker-compose up -d chromadb
```

### 4.4 Response Runbooks

**Emergency System Restart:**
```bash
#!/bin/bash
# emergency_restart.sh

echo "EMERGENCY RESTART INITIATED $(date)"

# Stop all services
docker-compose down

# Wait for cleanup
sleep 10

# Remove any stuck containers
docker container prune -f

# Restart core services first
docker-compose up -d postgres redis neo4j
sleep 30

# Start application layer
docker-compose up -d backend ollama
sleep 30

# Start remaining services
docker-compose up -d

echo "EMERGENCY RESTART COMPLETE"
```

**Service Degradation Response:**
```bash
#!/bin/bash
# Check service health and restart unhealthy containers

for service in backend frontend ollama postgres redis; do
    echo "Checking $service..."
    if ! docker ps | grep -q "sutazai-$service.*healthy\|Up"; then
        echo "Restarting $service..."
        docker-compose restart $service
        sleep 10
    fi
done
```

---

## 5. Maintenance Procedures

### 5.1 Scheduled Maintenance Windows

**Weekly Maintenance (Sundays 02:00-04:00 UTC):**
- Log rotation and cleanup
- Performance metrics review
- Security updates check
- Backup verification

**Monthly Maintenance (First Sunday 01:00-05:00 UTC):**
- Docker image updates
- Database maintenance
- Configuration updates
- Capacity planning review

**Quarterly Maintenance (Planned 4-hour windows):**
- Major version updates
- Infrastructure changes
- Security audits
- Disaster recovery testing

### 5.2 Database Maintenance

**PostgreSQL Maintenance:**
```bash
# Connect to database
docker exec -it sutazai-postgres psql -U sutazai -d sutazai

-- Database statistics
SELECT schemaname, tablename, n_tup_ins, n_tup_upd, n_tup_del 
FROM pg_stat_user_tables;

-- Index usage
SELECT schemaname, tablename, indexname, idx_scan, idx_tup_read 
FROM pg_stat_user_indexes;

-- Database size
SELECT pg_size_pretty(pg_database_size('sutazai'));

-- Vacuum and analyze (if tables exist)
VACUUM ANALYZE;
```

**Redis Maintenance:**
```bash
# Connect to Redis
docker exec -it sutazai-redis redis-cli

# Check memory usage
INFO memory

# Check key statistics
INFO keyspace

# Clean expired keys (if any)
FLUSHALL  # CAUTION: This removes all data
```

**Neo4j Maintenance:**
```bash
# Connect to Neo4j
docker exec -it sutazai-neo4j cypher-shell -u neo4j -p sutazai_neo4j_2024

// Database statistics
CALL db.stats.retrieve('GRAPH COUNTS');

// Clear database (if needed)
MATCH (n) DETACH DELETE n;  // CAUTION: Removes all data
```

### 5.3 Log Rotation and Cleanup

**Docker Log Cleanup:**
```bash
#!/bin/bash
# log_cleanup.sh

# Truncate Docker logs older than 7 days
find /var/lib/docker/containers/ -name "*.log" -mtime +7 -exec truncate -s 0 {} \;

# Clean Docker system
docker system prune -f --filter "until=72h"

# Rotate application logs
docker-compose exec backend sh -c "find /app/logs -name '*.log' -mtime +7 -delete" 2>/dev/null || true
```

**Log Retention Policy:**
- Docker container logs: 7 days
- Application logs: 30 days
- Monitoring metrics: 90 days
- Audit logs: 1 year

### 5.4 Docker Image Updates

**Security Update Process:**
```bash
#!/bin/bash
# update_images.sh

echo "Checking for image updates..."

# Pull latest base images
docker pull postgres:15
docker pull redis:7-alpine
docker pull neo4j:5.11
docker pull ollama/ollama:latest
docker pull python:3.11-slim

# Rebuild custom images
docker-compose build --no-cache

# Rolling update (one service at a time)
for service in postgres redis neo4j ollama backend frontend; do
    echo "Updating $service..."
    docker-compose up -d --no-deps $service
    sleep 30
    
    # Verify health
    if docker ps | grep -q "sutazai-$service.*healthy\|Up"; then
        echo "$service updated successfully"
    else
        echo "ERROR: $service failed to start"
        docker-compose logs $service
        exit 1
    fi
done

echo "All services updated successfully"
```

---

## 6. Backup and Recovery

### 6.1 Backup Schedules

**Daily Backups (04:00 UTC):**
- PostgreSQL database dump
- Redis RDB snapshot
- Neo4j database backup
- Configuration files

**Weekly Backups (Sunday 05:00 UTC):**
- Complete system state
- Docker volumes
- Application logs
- Monitoring data

**Monthly Backups (First Sunday 06:00 UTC):**
- Full system image
- Long-term archival
- Compliance backups

### 6.2 Database Backup Procedures

**PostgreSQL Backup:**
```bash
#!/bin/bash
# backup_postgres.sh

BACKUP_DIR="/opt/sutazaiapp/backups/postgres"
DATE=$(date +"%Y%m%d_%H%M%S")

mkdir -p $BACKUP_DIR

# Create database dump
docker exec sutazai-postgres pg_dump -U sutazai -d sutazai | gzip > "$BACKUP_DIR/postgres_backup_$DATE.sql.gz"

# Verify backup
if [ -f "$BACKUP_DIR/postgres_backup_$DATE.sql.gz" ]; then
    echo "PostgreSQL backup completed: postgres_backup_$DATE.sql.gz"
    
    # Cleanup old backups (keep 30 days)
    find $BACKUP_DIR -name "postgres_backup_*.sql.gz" -mtime +30 -delete
else
    echo "ERROR: PostgreSQL backup failed"
    exit 1
fi
```

**Redis Backup:**
```bash
#!/bin/bash
# backup_redis.sh

BACKUP_DIR="/opt/sutazaiapp/backups/redis"
DATE=$(date +"%Y%m%d_%H%M%S")

mkdir -p $BACKUP_DIR

# Trigger Redis save
docker exec sutazai-redis redis-cli BGSAVE

# Wait for save to complete
sleep 10

# Copy RDB file
docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/redis_backup_$DATE.rdb"

if [ -f "$BACKUP_DIR/redis_backup_$DATE.rdb" ]; then
    echo "Redis backup completed: redis_backup_$DATE.rdb"
    find $BACKUP_DIR -name "redis_backup_*.rdb" -mtime +30 -delete
else
    echo "ERROR: Redis backup failed"
    exit 1
fi
```

**Neo4j Backup:**
```bash
#!/bin/bash
# backup_neo4j.sh

BACKUP_DIR="/opt/sutazaiapp/backups/neo4j"
DATE=$(date +"%Y%m%d_%H%M%S")

mkdir -p $BACKUP_DIR

# Create Neo4j dump
docker exec sutazai-neo4j neo4j-admin database dump --database=neo4j --to-path=/tmp
docker cp sutazai-neo4j:/tmp/neo4j.dump "$BACKUP_DIR/neo4j_backup_$DATE.dump"

if [ -f "$BACKUP_DIR/neo4j_backup_$DATE.dump" ]; then
    echo "Neo4j backup completed: neo4j_backup_$DATE.dump"
    find $BACKUP_DIR -name "neo4j_backup_*.dump" -mtime +30 -delete
else
    echo "ERROR: Neo4j backup failed"
    exit 1
fi
```

### 6.3 Configuration Backup

**System Configuration Backup:**
```bash
#!/bin/bash
# backup_config.sh

BACKUP_DIR="/opt/sutazaiapp/backups/config"
DATE=$(date +"%Y%m%d_%H%M%S")

mkdir -p $BACKUP_DIR

# Create configuration archive
tar -czf "$BACKUP_DIR/config_backup_$DATE.tar.gz" \
    docker-compose.yml \
    config/ \
    .env \
    secrets_secure/ \
    monitoring/prometheus-rules.yml \
    nginx/nginx.conf

if [ -f "$BACKUP_DIR/config_backup_$DATE.tar.gz" ]; then
    echo "Configuration backup completed: config_backup_$DATE.tar.gz"
    find $BACKUP_DIR -name "config_backup_*.tar.gz" -mtime +30 -delete
else
    echo "ERROR: Configuration backup failed"
    exit 1
fi
```

### 6.4 Data Restoration Steps

**PostgreSQL Restoration:**
```bash
#!/bin/bash
# restore_postgres.sh [backup_file]

BACKUP_FILE="$1"

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    exit 1
fi

echo "Restoring PostgreSQL from $BACKUP_FILE..."

# Stop backend to prevent connections
docker-compose stop backend

# Drop and recreate database
docker exec sutazai-postgres psql -U sutazai -c "DROP DATABASE IF EXISTS sutazai;"
docker exec sutazai-postgres psql -U sutazai -c "CREATE DATABASE sutazai;"

# Restore from backup
gunzip -c "$BACKUP_FILE" | docker exec -i sutazai-postgres psql -U sutazai -d sutazai

# Restart backend
docker-compose start backend

echo "PostgreSQL restoration complete"
```

**Complete System Restoration:**
```bash
#!/bin/bash
# restore_system.sh [backup_date]

BACKUP_DATE="$1"
BACKUP_DIR="/opt/sutazaiapp/backups"

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <YYYYMMDD_HHMMSS>"
    exit 1
fi

echo "Restoring system from backups dated $BACKUP_DATE..."

# Stop all services
docker-compose down

# Restore databases
./restore_postgres.sh "$BACKUP_DIR/postgres/postgres_backup_$BACKUP_DATE.sql.gz"
cp "$BACKUP_DIR/redis/redis_backup_$BACKUP_DATE.rdb" /var/lib/docker/volumes/sutazai-redis-data/_data/dump.rdb

# Restore configuration
tar -xzf "$BACKUP_DIR/config/config_backup_$BACKUP_DATE.tar.gz"

# Start services
docker-compose up -d

echo "System restoration complete"
```

---

## 7. Performance Management

### 7.1 Performance Monitoring

**Real-time Performance Metrics:**
```bash
# Container resource usage
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# System load and memory
top -bn1 | head -5
free -h
iostat -x 1 3

# Database performance
docker exec sutazai-postgres psql -U sutazai -c "SELECT count(*) as connections FROM pg_stat_activity;"
docker exec sutazai-redis redis-cli info stats | grep -E "instantaneous_ops_per_sec|used_memory_human"
```

**Performance Baseline Collection:**
```bash
#!/bin/bash
# collect_baseline.sh

METRICS_DIR="/opt/sutazaiapp/monitoring/baselines"
DATE=$(date +"%Y%m%d_%H%M%S")

mkdir -p $METRICS_DIR

# System metrics
{
    echo "=== System Performance Baseline $DATE ==="
    echo "CPU Usage:"
    top -bn1 | grep "Cpu(s)" | sed "s/.*, *\([0-9.]*\)%* id.*/\1/" | awk '{print 100 - $1"%"}'
    
    echo "Memory Usage:"
    free -h | awk 'NR==2{printf "%.2f%%\n", $3*100/$2 }'
    
    echo "Disk Usage:"
    df -h | awk '$NF=="/"{printf "%s\n", $5}'
    
    echo "Container Stats:"
    docker stats --no-stream --format "{{.Name}}: {{.CPUPerc}} CPU, {{.MemUsage}}"
    
    echo "Response Times:"
    time curl -s http://127.0.0.1:10010/health > /dev/null
    time curl -s http://127.0.0.1:10104/api/tags > /dev/null
    
} > "$METRICS_DIR/baseline_$DATE.txt"

echo "Performance baseline saved: baseline_$DATE.txt"
```

### 7.2 Bottleneck Identification

**Common Bottleneck Scenarios:**

**CPU Bottlenecks:**
```bash
# Identify CPU-intensive containers
docker stats --no-stream | sort -k3 -hr

# Process-level CPU usage
docker exec [container] top -bn1 | head -10

# CPU throttling detection
docker exec [container] cat /sys/fs/cgroup/cpu/cpu.stat
```

**Memory Bottlenecks:**
```bash
# Memory usage by container
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"

# OOM killer events
dmesg | grep -i "killed process"

# Memory leaks detection
for container in $(docker ps --format "{{.Names}}"); do
    echo "$container memory trend:"
    docker exec $container cat /proc/meminfo | grep MemAvailable
done
```

**Disk I/O Bottlenecks:**
```bash
# Disk I/O by container
docker stats --no-stream --format "table {{.Name}}\t{{.BlockIO}}"

# System I/O wait
iostat -x 1 5

# Container filesystem usage
docker system df
```

**Network Bottlenecks:**
```bash
# Network I/O by container
docker stats --no-stream --format "table {{.Name}}\t{{.NetIO}}"

# Port connectivity issues
netstat -tlnp | grep -E ":(10010|10104|8589)"

# DNS resolution performance
docker exec sutazai-backend nslookup sutazai-postgres
```

### 7.3 Resource Optimization

**Container Resource Limits:**
```yaml
# docker-compose.yml optimization
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 1G
          cpus: '1.0'
        reservations:
          memory: 512M
          cpus: '0.5'
  
  ollama:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '2.0'
        reservations:
          memory: 1G
          cpus: '1.0'
```

**Database Connection Pool Optimization:**
```python
# Backend database settings
DATABASE_CONFIG = {
    'pool_size': 10,
    'max_overflow': 20,
    'pool_timeout': 30,
    'pool_recycle': 3600
}
```

**Ollama Performance Tuning:**
```bash
# Optimize Ollama for CPU-only deployment
docker exec sutazai-ollama ollama run tinyllama --num-threads 4 --num-gpu 0
```

### 7.4 Capacity Planning

**Growth Projections:**
- Current baseline: 28 containers, ~4GB RAM usage
- Agent activation: +7 real agents = ~2GB additional RAM
- Database growth: Plan for 10GB/month data growth
- Model scaling: TinyLlama → larger models = 2-8GB additional

**Scaling Triggers:**
- CPU > 80% sustained for 5 minutes
- Memory > 85% sustained for 2 minutes
- Disk > 90% of available space
- Response time > 5 seconds average

**Horizontal Scaling Plan:**
1. **Phase 1**: Add second backend instance (load balancer required)
2. **Phase 2**: Database read replicas for PostgreSQL
3. **Phase 3**: Ollama cluster for model serving
4. **Phase 4**: Agent service horizontal scaling

---

## 8. Security Operations

### 8.1 Security Monitoring

**Container Security Monitoring:**
```bash
# Check for vulnerable images (Trivy required)
trivy image --severity HIGH,CRITICAL $(docker images --format "{{.Repository}}:{{.Tag}}")

# Container privilege escalation check
docker inspect $(docker ps -q) | jq -r '.[] | "\(.Name): privileged=\(.HostConfig.Privileged)"'

# Network security assessment
docker network ls
docker network inspect sutazai-network | jq '.[] | .Containers'
```

**Access Control Monitoring:**
```bash
# Failed authentication attempts (if logs available)
docker-compose logs | grep -i "authentication\|unauthorized\|403\|401"

# Unusual access patterns
docker-compose logs nginx | grep -E "^[0-9.]+ - - \[.*\] \"(GET|POST)" | awk '{print $1}' | sort | uniq -c | sort -nr | head -10

# Container file system changes
docker diff [container-name]
```

### 8.2 Vulnerability Scanning

**Automated Security Scanning:**
```bash
#!/bin/bash
# security_scan.sh

SCAN_DIR="/opt/sutazaiapp/security-reports"
DATE=$(date +"%Y%m%d_%H%M%S")

mkdir -p $SCAN_DIR

echo "Starting security scan at $(date)"

# Scan all images
for image in $(docker images --format "{{.Repository}}:{{.Tag}}" | grep -v "<none>"); do
    echo "Scanning $image..."
    trivy image --format table --output "$SCAN_DIR/trivy_${image//[:\/]/_}_$DATE.table" $image
done

# Scan filesystem
trivy fs --format json --output "$SCAN_DIR/filesystem_scan_$DATE.json" /opt/sutazaiapp/

# Network security scan
nmap -sS -O localhost -oN "$SCAN_DIR/network_scan_$DATE.txt"

echo "Security scan complete. Results in $SCAN_DIR/"
```

### 8.3 Security Incident Response

**Security Alert Classification:**
- **Critical**: Active intrusion, data breach, privilege escalation
- **High**: Vulnerability exploitation, unauthorized access attempts
- **Medium**: Configuration drift, suspicious activity
- **Low**: Security policy violations, outdated components

**Incident Response Playbook:**
```bash
#!/bin/bash
# security_incident_response.sh

echo "SECURITY INCIDENT RESPONSE INITIATED $(date)"

# Immediate containment
echo "1. Isolating affected containers..."
# docker network disconnect sutazai-network [affected-container]

# Evidence collection
echo "2. Collecting evidence..."
docker-compose logs > "/tmp/security_logs_$(date +%s).txt"
docker inspect $(docker ps -q) > "/tmp/container_states_$(date +%s).json"

# System hardening
echo "3. Implementing emergency hardening..."
# Stop unnecessary services
docker-compose stop $(docker-compose ps --services | grep -v -E "postgres|redis|backend|frontend")

# Change default passwords
echo "4. Rotating credentials..."
# Manual step: Update all default passwords

echo "SECURITY INCIDENT RESPONSE COMPLETE"
echo "Next steps: Manual investigation and remediation required"
```

### 8.4 Compliance Monitoring

**Security Compliance Checklist:**
- [ ] All containers run as non-root user
- [ ] No hardcoded secrets in images or configs
- [ ] Network segmentation properly configured
- [ ] Security patches applied within 30 days
- [ ] Access logs retained for 90 days
- [ ] Backup encryption enabled
- [ ] Regular vulnerability scans completed

**Compliance Validation:**
```bash
#!/bin/bash
# compliance_check.sh

echo "=== SutazAI Security Compliance Check $(date) ==="

# Check for root processes
echo "Root Process Check:"
docker exec $(docker ps -q) ps aux | grep -v grep | grep "root" | wc -l

# Secrets scanning
echo "Secrets Check:"
grep -r -i "password\|secret\|key" /opt/sutazaiapp/ --exclude-dir=backups | head -5

# Network exposure check
echo "Network Exposure Check:"
netstat -tlnp | grep -E ":(22|80|443|3389)"

# File permissions
echo "File Permissions Check:"
find /opt/sutazaiapp/secrets_secure -type f -exec ls -la {} \;

echo "=== Compliance Check Complete ==="
```

---

## 9. Known Issues and Workarounds

### 9.1 TinyLlama vs gpt-oss Model Mismatch

**Issue:** Backend expects gpt-oss model but TinyLlama is loaded

**Symptoms:**
- Backend shows "degraded" status
- AI features not working properly
- Model not found errors

**Workaround 1 - Load gpt-oss model:**
```bash
# Download and load gpt-oss model (if available)
docker exec sutazai-ollama ollama pull gpt-oss

# Restart backend to detect new model
docker-compose restart backend
```

**Workaround 2 - Update backend configuration:**
```bash
# Edit backend configuration to use tinyllama
# File: /opt/sutazaiapp/backend/app/core/config.py
# Change: MODEL_NAME = "gpt-oss" 
# To: MODEL_NAME = "tinyllama"

# Rebuild and restart backend
docker-compose build backend
docker-compose restart backend
```

### 9.2 ChromaDB Connection Issues

**Issue:** ChromaDB container restarts frequently, connection timeouts

**Symptoms:**
- ChromaDB container status shows restarting
- Backend logs show ChromaDB connection errors
- Vector search features unavailable

**Workaround:**
```bash
# Stop ChromaDB and remove volume
docker-compose stop chromadb
docker volume rm sutazai-chromadb-data

# Restart with fresh volume
docker-compose up -d chromadb

# Wait for initialization (can take 60+ seconds)
sleep 60

# Verify health
curl http://127.0.0.1:10100/api/v1/heartbeat
```

### 9.3 Agent Stub Limitations

**Issue:** Agent services are Flask stubs with no real AI logic

**Current Behavior:**
- All agents return {"status": "healthy"} on /health
- Process endpoints return hardcoded JSON responses
- No inter-agent communication
- No actual task processing

**Temporary Workaround:**
```bash
# Acknowledge current limitations in monitoring
echo "Agent services are stubs - expect minimal functionality"

# Monitor stub health only
for port in 8589 8587 8588 8551 8002 11015; do
    curl -s http://127.0.0.1:$port/health | jq -r '.status // "ERROR"'
done
```

**Long-term Solution:**
- Implement real agent logic in /agents/*/app.py
- Add actual processing capabilities
- Enable inter-agent communication
- Integrate with task queue system

### 9.4 PostgreSQL Missing Tables

**Issue:** PostgreSQL database exists but contains no tables

**Symptoms:**
- Backend database queries fail
- No user or agent data stored
- Application features limited

**Resolution:**
```bash
# Create database schema (if schema file exists)
if [ -f "/opt/sutazaiapp/sql/init.sql" ]; then
    docker exec -i sutazai-postgres psql -U sutazai -d sutazai < /opt/sutazaiapp/sql/init.sql
fi

# Or run backend migrations (if available)
docker exec sutazai-backend python -m app.db.init_db

# Verify table creation
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "\dt"
```

### 9.5 Service Mesh Not Configured

**Issue:** Kong/Consul/RabbitMQ running but not configured

**Symptoms:**
- Kong has no API routes configured
- Consul shows no registered services
- RabbitMQ not used by applications

**Temporary Workaround:**
```bash
# Direct service access (bypass service mesh)
# Use container IP addresses directly
docker inspect sutazai-backend | jq -r '.[0].NetworkSettings.Networks."sutazai-network".IPAddress'

# Configure basic Kong route (if needed)
curl -X POST http://127.0.0.1:8001/services \
  --data name=backend \
  --data url=http://sutazai-backend:8000

curl -X POST http://127.0.0.1:8001/services/backend/routes \
  --data paths=/api
```

---

## 10. Operational Runbooks

### 10.1 Service Degradation Response

**Symptom:** High response times, 5xx errors, timeouts

**Investigation Steps:**
```bash
# Step 1: Check overall system health
docker ps | grep -v "Up.*healthy\|Up.*minutes\|Up.*seconds" 

# Step 2: Identify resource constraints
docker stats --no-stream | awk '$3 > 80 || $7 > 85 {print}'

# Step 3: Check application logs
docker-compose logs --tail=50 backend frontend | grep -i error

# Step 4: Database connectivity
docker exec sutazai-postgres pg_isready -U sutazai
docker exec sutazai-redis redis-cli ping
```

**Resolution Actions:**
```bash
# Action 1: Restart unhealthy containers
docker-compose restart $(docker ps --filter "health=unhealthy" --format "{{.Names}}" | sed 's/sutazai-//')

# Action 2: Clear cache if Redis issues
docker exec sutazai-redis redis-cli FLUSHALL

# Action 3: Scale backend if CPU bound
# Manual: Add second backend instance to docker-compose.yml

# Action 4: Database maintenance if needed
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "VACUUM ANALYZE;"
```

### 10.2 Database Connection Pool Exhaustion

**Symptom:** "Too many connections" errors, backend timeouts

**Investigation:**
```bash
# Check active connections
docker exec sutazai-postgres psql -U sutazai -c "SELECT count(*) FROM pg_stat_activity;"

# Check connection pool status in backend logs
docker-compose logs backend | grep -i "connection\|pool"

# Monitor connection patterns
docker exec sutazai-postgres psql -U sutazai -c "SELECT client_addr, count(*) FROM pg_stat_activity GROUP BY client_addr;"
```

**Resolution:**
```bash
# Immediate fix: Restart backend to reset connections
docker-compose restart backend

# Kill long-running queries (if any)
docker exec sutazai-postgres psql -U sutazai -c "SELECT pg_terminate_backend(pid) FROM pg_stat_activity WHERE state = 'active' AND query_start < now() - interval '5 minutes';"

# Long-term: Increase connection limits
# Edit postgresql.conf: max_connections = 200
```

### 10.3 Memory Leak Detection and Resolution

**Symptom:** Gradually increasing memory usage, OOM kills

**Detection:**
```bash
# Memory trend analysis
for i in {1..10}; do
    echo "$(date): $(docker stats --no-stream --format '{{.Name}} {{.MemUsage}}' | head -5)"
    sleep 30
done

# Check for OOM events
dmesg | grep -i "killed process\|out of memory"

# Container memory limits
docker stats --no-stream --format "table {{.Name}}\t{{.MemUsage}}\t{{.MemPerc}}"
```

**Resolution:**
```bash
# Restart affected containers
docker-compose restart [high-memory-container]

# Add memory limits to prevent system impact
# docker-compose.yml:
# services:
#   backend:
#     deploy:
#       resources:
#         limits:
#           memory: 1G

# Monitor heap dumps (if Java/Python apps)
# docker exec [container] jmap -dump:format=b,file=/tmp/heapdump.hprof [PID]
```

### 10.4 Disk Space Management

**Symptom:** Disk space warnings, write failures

**Assessment:**
```bash
# System disk usage
df -h

# Docker space usage
docker system df

# Large log files
find /var/lib/docker/containers -name "*.log" -size +100M -exec ls -lh {} \;

# Container volume usage
docker exec [container] df -h
```

**Cleanup Actions:**
```bash
# Clean Docker system (safe)
docker system prune -f

# Clean old log files
find /var/lib/docker/containers -name "*.log" -mtime +7 -exec truncate -s 0 {} \;

# Compress old backups
find /opt/sutazaiapp/backups -name "*.sql" -mtime +7 -exec gzip {} \;

# Emergency: Remove unused volumes (CAUTION: data loss possible)
docker volume prune -f
```

### 10.5 Network Connectivity Issues

**Symptom:** Service timeouts, DNS resolution failures

**Diagnosis:**
```bash
# Container network connectivity
docker exec sutazai-backend ping -c3 sutazai-postgres
docker exec sutazai-backend nslookup sutazai-redis

# Port accessibility
netstat -tlnp | grep -E ":(10010|10104|8589)"

# Docker network inspection
docker network inspect sutazai-network

# Service registration (Consul)
curl http://127.0.0.1:10006/v1/catalog/services
```

**Resolution:**
```bash
# Restart Docker daemon (if network issues)
sudo systemctl restart docker
docker-compose up -d

# Recreate Docker network
docker-compose down
docker network rm sutazai-network
docker network create sutazai-network
docker-compose up -d

# DNS resolution fix
echo "nameserver 8.8.8.8" > /tmp/resolv.conf
docker exec sutazai-backend cp /tmp/resolv.conf /etc/resolv.conf
```

---

## 11. Monitoring and Alerting

### 11.1 Prometheus Monitoring Queries

**System Health Queries:**
```promql
# Container CPU usage
rate(container_cpu_usage_seconds_total{name=~"sutazai-.*"}[5m]) * 100

# Container memory usage
container_memory_usage_bytes{name=~"sutazai-.*"} / container_spec_memory_limit_bytes{name=~"sutazai-.*"} * 100

# Container restart count
increase(container_start_time_seconds{name=~"sutazai-.*"}[1h])

# HTTP response time
histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))

# Error rate
rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) * 100
```

**Database Performance Queries:**
```promql
# PostgreSQL connection count
pg_stat_database_numbackends{datname="sutazai"}

# Redis memory usage
redis_memory_used_bytes / redis_memory_max_bytes * 100

# Database query duration
pg_stat_statements_mean_time_ms > 1000
```

### 11.2 Grafana Dashboard Usage

**Pre-configured Dashboards:**
- **System Overview**: Container resource usage, system health
- **Application Performance**: Response times, error rates
- **Database Monitoring**: Connection pools, query performance
- **Business Metrics**: User activity, feature usage

**Dashboard Access:**
```bash
# Grafana login
URL: http://127.0.0.1:10201
Username: admin
Password: admin (default)

# Import custom dashboards
curl -X POST http://admin:admin@127.0.0.1:10201/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/system-overview.json
```

### 11.3 Alert Response Procedures

**Alert Severity Levels:**

**Critical Alerts (Immediate Response):**
- System Down: All containers unhealthy
- Database Unavailable: Connection failures
- High Error Rate: >10% 5xx responses
- Disk Space Critical: >95% usage

**Warning Alerts (15-minute Response):**
- High CPU Usage: >80% for 5 minutes
- High Memory Usage: >85% for 5 minutes
- Slow Response Time: >5s average
- Container Restarts: >3 restarts in 1 hour

**Info Alerts (Daily Review):**
- Backup Failures: Scheduled backup missed
- Security Scan Alerts: New vulnerabilities
- Performance Degradation: 20% slower than baseline

**Alert Response Scripts:**
```bash
#!/bin/bash
# alert_response.sh [alert_type] [severity]

ALERT_TYPE="$1"
SEVERITY="$2"

case $SEVERITY in
    "critical")
        echo "CRITICAL ALERT: $ALERT_TYPE"
        # Immediate escalation
        # Send notifications
        # Execute emergency procedures
        ;;
    "warning")
        echo "WARNING ALERT: $ALERT_TYPE"
        # Investigate and resolve within 15 minutes
        # Log incident
        ;;
    "info")
        echo "INFO ALERT: $ALERT_TYPE"
        # Add to daily review queue
        ;;
esac
```

### 11.4 Log Analysis with Loki

**Loki Query Examples:**
```logql
# Error log aggregation
{container_name=~"sutazai-.*"} |~ "ERROR|error|Error"

# Backend API errors
{container_name="sutazai-backend"} |~ "5[0-9][0-9]"

# Database connection issues
{container_name=~"sutazai-(postgres|redis)"} |~ "connection|timeout"

# Agent activity (minimal expected)
{container_name=~"sutazai-.*-agent.*"} |~ "health|process"
```

**Log Analysis Dashboard:**
```bash
# Access Loki
URL: http://127.0.0.1:10202

# Query recent errors
curl -G -s "http://127.0.0.1:10202/loki/api/v1/query" \
  --data-urlencode 'query={container_name=~"sutazai-.*"} |~ "ERROR"' \
  --data-urlencode 'time=now-1h'
```

---

## 12. Change Management

### 12.1 Change Request Process

**Change Categories:**
- **Emergency**: Critical security/stability fixes
- **Standard**: Planned features, updates, maintenance
- **Normal**: Configuration changes, minor updates

**Change Request Template:**
```markdown
## Change Request [CR-YYYY-NNNN]

**Requester**: [Name]
**Date**: [YYYY-MM-DD]
**Category**: [Emergency/Standard/Normal]
**Priority**: [P0/P1/P2/P3]

### Change Description
[Detailed description of the change]

### Business Justification
[Why this change is needed]

### Technical Impact Assessment
[Systems/services affected]

### Risk Assessment
[Potential risks and mitigation strategies]

### Testing Plan
[How the change will be tested]

### Rollback Plan
[Steps to revert if issues occur]

### Implementation Schedule
[Planned implementation time and duration]

### Success Criteria
[How to measure successful implementation]
```

### 12.2 Testing Procedures

**Pre-deployment Testing:**
```bash
#!/bin/bash
# pre_deployment_test.sh

echo "Starting pre-deployment testing..."

# Unit tests (if available)
docker exec sutazai-backend python -m pytest tests/unit/

# Integration tests
docker exec sutazai-backend python -m pytest tests/integration/

# Health check tests
for service in backend frontend ollama postgres redis; do
    echo "Testing $service health..."
    # Service-specific health checks
done

# Performance baseline verification
# Load test with expected traffic patterns

echo "Pre-deployment testing complete"
```

**Post-deployment Verification:**
```bash
#!/bin/bash
# post_deployment_verify.sh

echo "Starting post-deployment verification..."

# Service health verification
curl -s http://127.0.0.1:10010/health | jq -r '.status'

# Database connectivity
docker exec sutazai-postgres pg_isready -U sutazai

# Key functionality tests
# Test critical user journeys

# Performance verification
# Compare against baseline metrics

echo "Post-deployment verification complete"
```

### 12.3 Deployment Approval Process

**Approval Matrix:**
- **Emergency Changes**: On-call engineer + supervisor
- **Standard Changes**: Team lead + operations manager
- **Normal Changes**: Peer review + team lead

**Approval Checklist:**
- [ ] Change request reviewed and approved
- [ ] Risk assessment completed
- [ ] Testing plan executed successfully
- [ ] Rollback plan documented and tested
- [ ] Maintenance window scheduled (if required)
- [ ] Stakeholder notification sent
- [ ] Monitoring alerts configured
- [ ] Documentation updated

### 12.4 Documentation Requirements

**Required Documentation Updates:**
- System architecture diagrams
- Configuration management records
- Operational procedures
- Monitoring and alerting configurations
- Security controls documentation
- User guides and training materials

**Documentation Maintenance:**
```bash
# Update changelog
echo "$(date): [CHANGE-ID] - Brief description of change" >> /opt/sutazaiapp/docs/CHANGELOG.md

# Update system documentation
# Edit relevant markdown files in /opt/sutazaiapp/docs/

# Update operational runbooks
# Modify procedures in this playbook if needed

# Version control
cd /opt/sutazaiapp
git add docs/
git commit -m "docs: Update operational documentation for [CHANGE-ID]"
```

---

## 13. Emergency Procedures

### 13.1 Emergency Contacts

**Escalation Chain:**
1. **Primary On-Call**: [Contact Info]
2. **Secondary On-Call**: [Contact Info]  
3. **Operations Manager**: [Contact Info]
4. **System Administrator**: [Contact Info]
5. **External Support**: [Vendor Contact Info]

**Communication Channels:**
- **Slack**: #sutazai-incidents (immediate)
- **Email**: ops-team@company.com (documentation)
- **Phone**: [Emergency number] (critical issues)
- **Status Page**: [URL] (customer communication)

### 13.2 Emergency System Shutdown

**Complete System Emergency Shutdown:**
```bash
#!/bin/bash
# emergency_shutdown.sh

echo "EMERGENCY SYSTEM SHUTDOWN INITIATED $(date)"

# Stop all services immediately
docker-compose down --timeout 10

# Force kill any remaining containers
docker ps -q | xargs -r docker kill

# Stop Docker daemon if necessary
# sudo systemctl stop docker

echo "EMERGENCY SHUTDOWN COMPLETE $(date)"
echo "All services stopped. Investigation required before restart."
```

**Selective Service Shutdown:**
```bash
#!/bin/bash
# selective_shutdown.sh [service_pattern]

SERVICE_PATTERN="$1"

if [ -z "$SERVICE_PATTERN" ]; then
    echo "Usage: $0 <service_pattern>"
    exit 1
fi

echo "Emergency shutdown of services matching: $SERVICE_PATTERN"

# Stop matching services
docker ps --format "{{.Names}}" | grep "$SERVICE_PATTERN" | xargs -r docker stop

echo "Services stopped: $(docker ps --format '{{.Names}}' | grep $SERVICE_PATTERN)"
```

### 13.3 Data Recovery Procedures

**Emergency Data Recovery:**
```bash
#!/bin/bash
# emergency_data_recovery.sh

echo "EMERGENCY DATA RECOVERY INITIATED $(date)"

# Stop all services to prevent data corruption
docker-compose down

# Backup current state before recovery
mkdir -p /tmp/emergency_backup_$(date +%s)
docker cp sutazai-postgres:/var/lib/postgresql/data /tmp/emergency_backup_*/postgres_data
docker cp sutazai-redis:/data /tmp/emergency_backup_*/redis_data

# Restore from most recent backup
LATEST_BACKUP=$(ls -t /opt/sutazaiapp/backups/postgres/postgres_backup_*.sql.gz | head -1)
gunzip -c "$LATEST_BACKUP" | docker exec -i sutazai-postgres psql -U sutazai -d sutazai

echo "Emergency data recovery attempt complete"
echo "Verification required before returning to service"
```

### 13.4 Security Incident Response

**Immediate Security Response:**
```bash
#!/bin/bash
# security_incident_response.sh

echo "SECURITY INCIDENT RESPONSE ACTIVATED $(date)"

# Isolate compromised systems
echo "Isolating affected containers..."
# docker network disconnect sutazai-network [compromised-container]

# Preserve evidence
echo "Preserving evidence..."
docker-compose logs > "/opt/sutazaiapp/security-reports/incident_logs_$(date +%s).txt"
docker inspect $(docker ps -aq) > "/opt/sutazaiapp/security-reports/container_states_$(date +%s).json"

# Stop compromised services
echo "Stopping potentially compromised services..."
# docker-compose stop [compromised-services]

# Implement emergency security measures
echo "Implementing emergency security measures..."
# Change all passwords
# Disable external access
# Enable additional logging

echo "SECURITY INCIDENT RESPONSE PHASE 1 COMPLETE"
echo "Manual investigation and remediation required"
```

---

## 14. Appendix

### 14.1 Port Registry Reference

**Core Services:**
```
10000: PostgreSQL Database
10001: Redis Cache  
10002: Neo4j Browser Interface
10003: Neo4j Bolt Protocol
10004: [Reserved]
10005: Kong API Gateway
10006: Consul Service Discovery
10007: RabbitMQ AMQP
10008: RabbitMQ Management UI
10009: [Reserved]
10010: Backend FastAPI
10011: Frontend Streamlit
```

**AI Services:**
```
10100: ChromaDB Vector DB
10101: Qdrant HTTP
10102: Qdrant gRPC  
10103: FAISS Vector Service
10104: Ollama LLM Server
```

**Monitoring Stack:**
```
10200: Prometheus Metrics
10201: Grafana Dashboards
10202: Loki Log Aggregation
10203: AlertManager
10220: Node Exporter
10221: cAdvisor Container Metrics
10229: Blackbox Exporter
```

**Agent Services (Stubs):**
```
8002: Hardware Resource Optimizer
8551: Task Assignment Coordinator
8587: Multi-Agent Coordinator
8588: Resource Arbitration Agent
8589: AI Agent Orchestrator
11015: Ollama Integration Specialist
11063: AI Metrics Exporter (UNHEALTHY)
```

### 14.2 Command Reference

**Daily Operations:**
```bash
# System status check
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Service health verification
curl -s http://127.0.0.1:10010/health | jq
curl -s http://127.0.0.1:10104/api/tags | jq

# Resource monitoring
docker stats --no-stream --format "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Log monitoring
docker-compose logs -f --tail=50 backend

# Database connections
docker exec -it sutazai-postgres psql -U sutazai -d sutazai
docker exec -it sutazai-redis redis-cli
```

**Emergency Commands:**
```bash
# Emergency restart
docker-compose down && docker-compose up -d

# Service isolation
docker network disconnect sutazai-network [container-name]

# Force container cleanup
docker container prune -f

# System resource check
free -h && df -h && top -bn1 | head -10
```

### 14.3 Configuration File Locations

**Docker Configuration:**
- `docker-compose.yml`: Main service definitions
- `config/`: Service-specific configuration files
- `nginx/nginx.conf`: Reverse proxy configuration
- `monitoring/prometheus-rules.yml`: Alerting rules

**Application Configuration:**
- `backend/app/core/config.py`: Backend settings
- `frontend/app.py`: Frontend configuration  
- `agents/*/app.py`: Agent stub configurations

**Security Configuration:**
- `secrets_secure/`: Encrypted secrets storage
- `ssl/`: SSL certificates
- `.env`: Environment variables

### 14.4 Troubleshooting Quick Reference

**Common Issues:**

| Symptom | Likely Cause | Quick Fix |
|---------|-------------|-----------|
| Backend shows "degraded" | Ollama model mismatch | Load gpt-oss or update config |
| ChromaDB restarting | Volume/permission issues | Remove volume, restart |
| High memory usage | Memory leaks | Restart affected containers |
| Database connection errors | Connection pool exhausted | Restart backend |
| Slow response times | Resource constraints | Check docker stats |
| Agent 404 errors | Stub services down | Restart agent containers |
| No monitoring data | Prometheus issues | Check port 10200 access |

**Debug Commands:**
```bash
# Container health check
docker inspect [container] | jq '.State.Health'

# Network connectivity test
docker exec [container] ping [target-container]

# Resource constraints check
docker inspect [container] | jq '.HostConfig | {Memory, CpuShares}'

# Process investigation
docker exec [container] top -bn1

# Disk space check
docker exec [container] df -h
```

---

## Document Control

**Document Version**: 1.0  
**Created**: August 8, 2025  
**Last Updated**: August 8, 2025  
**Next Review**: September 8, 2025  
**Approved By**: DevOps Team Lead  
**Classification**: Internal Use  

**Change History:**
- v1.0 (2025-08-08): Initial version based on CLAUDE.md system truth

**Related Documents:**
- `/opt/sutazaiapp/CLAUDE.md` - System Truth Document
- `/opt/sutazaiapp/docs/SYSTEM_REALITY_REPORT.md` - Current System State
- `/opt/sutazaiapp/docs/runbooks/` - Specific operational runbooks

---

*This playbook reflects the actual system state as documented in CLAUDE.md. Regular updates are required as the system evolves from stubs to production-ready components.*