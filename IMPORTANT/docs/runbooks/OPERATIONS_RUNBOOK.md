# Operations Runbook - Perfect Jarvis System

**Document Version:** 1.0  
**Last Updated:** 2025-08-08  
**Author:** System Operations Team  

## 🎯 Purpose

This runbook provides step-by-step operational procedures for the Perfect Jarvis system based on actual system components and verified functionality.

## 📋 System Overview

### Core Infrastructure
- **Backend API**: FastAPI v17.0.0 on port 10010
- **Frontend**: Streamlit UI on port 10011
- **Database**: PostgreSQL on port 10000
- **Cache**: Redis on port 10001
- **Graph DB**: Neo4j on ports 10002/10003
- **LLM Service**: Ollama (TinyLlama) on port 10104
- **Monitoring**: Prometheus (10200), Grafana (10201), Loki (10202)

### Service Architecture
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Ollama LLM    │
│   (Streamlit)   │───▶│   (FastAPI)     │───▶│   (TinyLlama)   │
│   Port: 10011   │    │   Port: 10010   │    │   Port: 10104   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │     Redis       │    │     Neo4j       │
│   Port: 10000   │    │   Port: 10001   │    │  Ports: 10002/  │
└─────────────────┘    └─────────────────┘    │      10003      │
                                              └─────────────────┘
```

## 🚀 System Startup Procedures

### 1. Pre-Startup Verification

**Check Docker Environment:**
```bash
# Verify Docker is running
docker --version
docker ps

# Check available disk space (minimum 10GB required)
df -h /var/lib/docker

# Verify network exists
docker network ls | grep sutazai-network
```

**Check System Resources:**
```bash
# Memory check (minimum 4GB recommended)
free -h

# CPU check
lscpu | grep "CPU(s)"

# Disk space check
df -h
```

### 2. Standard Startup Sequence

**Step 1: Create Docker Network**
```bash
docker network create sutazai-network 2>/dev/null || true
```

**Step 2: Start Core Infrastructure Services**
```bash
# Start in dependency order
docker-compose up -d sutazai-postgres
docker-compose up -d sutazai-redis
docker-compose up -d sutazai-neo4j

# Wait 30 seconds for database initialization
sleep 30
```

**Step 3: Start Application Services**
```bash
# Start Ollama LLM service
docker-compose up -d sutazai-ollama

# Wait for Ollama to initialize
sleep 45

# Start backend API
docker-compose up -d backend

# Start frontend (if needed)
docker-compose up -d frontend
```

**Step 4: Start Monitoring Stack**
```bash
docker-compose up -d sutazai-prometheus
docker-compose up -d sutazai-grafana
docker-compose up -d sutazai-loki
docker-compose up -d sutazai-alertmanager
```

**Step 5: Verify Startup**
```bash
# Check all containers are running
docker-compose ps

# Verify core services
curl -s http://127.0.0.1:10010/health
curl -s http://127.0.0.1:10104/api/tags
```

### 3. Emergency Fast Start

**Quick startup for urgent situations:**
```bash
#!/bin/bash
# emergency_start.sh
set -e

echo "🚨 Emergency startup initiated..."

# Ensure network exists
docker network create sutazai-network 2>/dev/null || true

# Start critical services only
docker-compose up -d sutazai-postgres sutazai-redis sutazai-ollama

# Wait for initialization
sleep 60

# Start backend
docker-compose up -d backend

# Quick health check
timeout 30 bash -c 'until curl -s http://127.0.0.1:10010/health; do sleep 2; done'

echo "✅ Emergency startup complete"
```

## 🛑 System Shutdown Procedures

### 1. Graceful Shutdown

**Step 1: Stop Application Services**
```bash
# Stop backend API (allows current requests to complete)
docker-compose stop backend

# Stop frontend
docker-compose stop frontend

# Stop monitoring (optional)
docker-compose stop sutazai-prometheus sutazai-grafana sutazai-loki
```

**Step 2: Stop Core Services**
```bash
# Stop Ollama (may take time to save model state)
docker-compose stop sutazai-ollama

# Stop databases
docker-compose stop sutazai-postgres sutazai-redis sutazai-neo4j
```

**Step 3: Verify Shutdown**
```bash
# Check no critical containers are running
docker ps --filter "name=sutazai" --format "table {{.Names}}\t{{.Status}}"
```

### 2. Emergency Shutdown

**Immediate shutdown for critical situations:**
```bash
#!/bin/bash
# emergency_shutdown.sh
echo "🚨 Emergency shutdown initiated..."

# Kill all sutazai containers immediately
docker kill $(docker ps -q --filter "name=sutazai") 2>/dev/null || true

# Remove containers to free resources
docker rm $(docker ps -aq --filter "name=sutazai") 2>/dev/null || true

echo "✅ Emergency shutdown complete"
```

## 📊 Health Monitoring and Alerting

### 1. Health Check Commands

**System Health Dashboard:**
```bash
#!/bin/bash
# health_dashboard.sh
echo "=== PERFECT JARVIS HEALTH DASHBOARD ==="
echo "Timestamp: $(date)"
echo

# Container Status
echo "📦 CONTAINER STATUS:"
docker ps --filter "name=sutazai" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | head -20

echo
echo "🌐 SERVICE ENDPOINTS:"
services=(
    "Backend API:http://127.0.0.1:10010/health"
    "Ollama LLM:http://127.0.0.1:10104/api/tags"
    "Prometheus:http://127.0.0.1:10200/-/ready"
    "Grafana:http://127.0.0.1:10201/api/health"
)

for service in "${services[@]}"; do
    name=$(echo $service | cut -d: -f1)
    url=$(echo $service | cut -d: -f2,3)
    
    if curl -s --max-time 5 "$url" > /dev/null 2>&1; then
        echo "✅ $name: HEALTHY"
    else
        echo "❌ $name: UNHEALTHY"
    fi
done

echo
echo "💾 RESOURCE USAGE:"
echo "Memory: $(free -h | awk 'NR==2{printf "%.1f%% (%s/%s)", $3*100/$2, $3, $2}')"
echo "CPU: $(top -bn1 | grep load | awk '{printf "%.2f\n", $(NF-2)}')"
echo "Disk: $(df -h / | awk 'NR==2{print $5 " (" $3 "/" $2 ")"}')"
```

### 2. Automated Health Monitoring

**Health Monitor Script:**
```bash
#!/bin/bash
# health_monitor.sh
HEALTH_LOG="/opt/sutazaiapp/logs/health_monitor.log"
ALERT_EMAIL="ops-team@company.com"  # Configure as needed

check_service() {
    local service_name=$1
    local endpoint=$2
    local timeout=${3:-5}
    
    if curl -s --max-time $timeout "$endpoint" > /dev/null 2>&1; then
        echo "$(date): ✅ $service_name HEALTHY" >> "$HEALTH_LOG"
        return 0
    else
        echo "$(date): ❌ $service_name UNHEALTHY" >> "$HEALTH_LOG"
        return 1
    fi
}

# Main monitoring loop
while true; do
    failed_services=()
    
    # Check critical services
    check_service "Backend API" "http://127.0.0.1:10010/health" || failed_services+=("Backend API")
    check_service "Ollama LLM" "http://127.0.0.1:10104/api/tags" || failed_services+=("Ollama LLM")
    
    # Check database connectivity
    if ! docker exec sutazai-postgres pg_isready -U sutazai > /dev/null 2>&1; then
        failed_services+=("PostgreSQL")
        echo "$(date): ❌ PostgreSQL UNHEALTHY" >> "$HEALTH_LOG"
    fi
    
    # Alert on failures
    if [ ${#failed_services[@]} -gt 0 ]; then
        alert_msg="🚨 ALERT: Services down: $(IFS=,; echo "${failed_services[*]}")"
        echo "$alert_msg" >> "$HEALTH_LOG"
        
        # Send alert (configure notification method)
        echo "$alert_msg" | logger -t "jarvis-monitor"
    fi
    
    sleep 60  # Check every minute
done
```

### 3. Performance Monitoring

**System Metrics Collection:**
```bash
#!/bin/bash
# collect_metrics.sh
METRICS_DIR="/opt/sutazaiapp/logs/metrics"
mkdir -p "$METRICS_DIR"

collect_metrics() {
    local timestamp=$(date '+%Y%m%d_%H%M%S')
    local metrics_file="$METRICS_DIR/metrics_${timestamp}.json"
    
    # Collect system metrics
    cat > "$metrics_file" << EOF
{
    "timestamp": "$(date -Iseconds)",
    "system": {
        "cpu_percent": $(top -bn1 | grep "Cpu(s)" | awk '{print $2}' | sed 's/%us,//'),
        "memory_percent": $(free | grep Mem | awk '{printf("%.1f"), $3/$2 * 100.0}'),
        "disk_percent": $(df -h / | awk 'NR==2{print $5}' | sed 's/%//'),
        "load_average": "$(uptime | awk -F'load average:' '{print $2}')"
    },
    "containers": {
        "running": $(docker ps --filter "name=sutazai" -q | wc -l),
        "total": $(docker ps -a --filter "name=sutazai" -q | wc -l)
    },
    "services": {
        "backend_healthy": $(curl -s http://127.0.0.1:10010/health > /dev/null && echo true || echo false),
        "ollama_healthy": $(curl -s http://127.0.0.1:10104/api/tags > /dev/null && echo true || echo false)
    }
}
EOF
    
    echo "Metrics collected: $metrics_file"
}

# Run collection
collect_metrics
```

## 🔧 Troubleshooting Common Issues

### 1. Backend API Issues

**Issue: Backend returns "degraded" status**

**Cause:** Ollama connection problem or model mismatch

**Solution:**
```bash
# Check Ollama status
curl -s http://127.0.0.1:10104/api/tags

# Check loaded models
docker exec sutazai-ollama ollama list

# Load required model if missing
docker exec sutazai-ollama ollama pull tinyllama

# Restart backend
docker-compose restart backend

# Verify fix
curl -s http://127.0.0.1:10010/health | jq '.status'
```

**Issue: Backend not responding**

**Diagnosis Steps:**
```bash
# Check if container is running
docker ps | grep backend

# Check container logs
docker-compose logs -f backend

# Check resource usage
docker stats sutazai-backend --no-stream

# Check port binding
netstat -tulpn | grep 10010
```

**Resolution:**
```bash
# Restart backend service
docker-compose restart backend

# If restart fails, recreate container
docker-compose up -d --force-recreate backend
```

### 2. Ollama LLM Issues

**Issue: No models available**

**Solution:**
```bash
# Check available models
docker exec sutazai-ollama ollama list

# Install TinyLlama (recommended for CPU)
docker exec sutazai-ollama ollama pull tinyllama

# Verify model loaded
curl -s http://127.0.0.1:10104/api/tags | jq '.models[].name'
```

**Issue: Slow response times**

**CPU Optimization:**
```bash
# Reduce context window in backend configuration
# Edit /opt/sutazaiapp/backend/app/main.py
# Change num_ctx from 2048 to 1024
# Reduce num_predict from 256 to 128

# Restart services
docker-compose restart backend sutazai-ollama
```

### 3. Database Issues

**Issue: PostgreSQL connection failed**

**Diagnosis:**
```bash
# Check PostgreSQL status
docker exec sutazai-postgres pg_isready -U sutazai

# Check database logs
docker-compose logs sutazai-postgres

# Test connection
docker exec -it sutazai-postgres psql -U sutazai -d sutazai -c "SELECT 1;"
```

**Solution:**
```bash
# Restart PostgreSQL
docker-compose restart sutazai-postgres

# Wait for initialization
sleep 30

# Verify connection
docker exec sutazai-postgres pg_isready -U sutazai
```

**Issue: Database storage full**

**Solution:**
```bash
# Check database size
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT pg_size_pretty(pg_database_size('sutazai'));"

# Clean up old logs if needed
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "VACUUM FULL;"

# Check available disk space
df -h /var/lib/docker
```

### 4. Memory Issues

**Issue: High memory usage**

**Diagnosis:**
```bash
# Check container memory usage
docker stats --no-stream

# Check system memory
free -h

# Check for memory leaks
ps aux --sort=-%mem | head -10
```

**Solution:**
```bash
# Restart memory-intensive services
docker-compose restart sutazai-ollama backend

# Clear Docker system cache
docker system prune -f

# Restart entire system if needed
docker-compose down && docker-compose up -d
```

## 🎛️ Performance Tuning Guidelines

### 1. Resource Optimization

**Memory Optimization:**
```bash
# Ollama memory settings (edit docker-compose.yml)
environment:
  - OLLAMA_KEEP_ALIVE=5m
  - OLLAMA_HOST=0.0.0.0
  - OLLAMA_NUM_PARALLEL=1
  - OLLAMA_MAX_LOADED_MODELS=1

# Backend memory settings
environment:
  - UVICORN_WORKERS=1  # Single worker for low memory
  - UVICORN_MAX_REQUESTS=100
```

**CPU Optimization:**
```bash
# Set CPU limits in docker-compose.yml
services:
  sutazai-ollama:
    cpus: '2.0'  # Limit CPU usage
  backend:
    cpus: '1.0'
```

### 2. Response Time Optimization

**Cache Configuration:**
```bash
# Redis cache settings
docker exec sutazai-redis redis-cli CONFIG SET maxmemory 512mb
docker exec sutazai-redis redis-cli CONFIG SET maxmemory-policy allkeys-lru
```

**API Optimization:**
- Enable response caching for health checks (30s cache in main.py)
- Use concurrent health checks for agents
- Implement connection pooling for database

### 3. Model Performance

**TinyLlama Optimization:**
```bash
# Optimal settings for CPU inference
curl -X POST http://127.0.0.1:10104/api/generate \
  -d '{
    "model": "tinyllama",
    "prompt": "test",
    "options": {
      "temperature": 0.7,
      "top_p": 0.9,
      "top_k": 40,
      "num_ctx": 1024,
      "num_predict": 128
    }
  }'
```

## 💾 Backup and Recovery Procedures

### 1. Database Backup

**PostgreSQL Backup:**
```bash
#!/bin/bash
# backup_postgres.sh
BACKUP_DIR="/opt/sutazaiapp/backups/postgres"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
BACKUP_FILE="$BACKUP_DIR/postgres_backup_$TIMESTAMP.sql"

mkdir -p "$BACKUP_DIR"

# Create backup
docker exec sutazai-postgres pg_dump -U sutazai sutazai > "$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_FILE"

echo "PostgreSQL backup created: ${BACKUP_FILE}.gz"

# Clean old backups (keep last 7 days)
find "$BACKUP_DIR" -name "*.gz" -mtime +7 -delete
```

**Redis Backup:**
```bash
#!/bin/bash
# backup_redis.sh
BACKUP_DIR="/opt/sutazaiapp/backups/redis"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')

mkdir -p "$BACKUP_DIR"

# Create Redis backup
docker exec sutazai-redis redis-cli BGSAVE
sleep 10
docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/redis_backup_$TIMESTAMP.rdb"

echo "Redis backup created: $BACKUP_DIR/redis_backup_$TIMESTAMP.rdb"
```

### 2. Configuration Backup

**System Configuration Backup:**
```bash
#!/bin/bash
# backup_config.sh
BACKUP_DIR="/opt/sutazaiapp/backups/config"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
CONFIG_BACKUP="$BACKUP_DIR/config_backup_$TIMESTAMP.tar.gz"

mkdir -p "$BACKUP_DIR"

# Backup critical configuration files
tar -czf "$CONFIG_BACKUP" \
  /opt/sutazaiapp/docker-compose.yml \
  /opt/sutazaiapp/config/ \
  /opt/sutazaiapp/.env \
  /opt/sutazaiapp/backend/app/

echo "Configuration backup created: $CONFIG_BACKUP"
```

### 3. Recovery Procedures

**PostgreSQL Recovery:**
```bash
#!/bin/bash
# restore_postgres.sh
BACKUP_FILE=$1

if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    exit 1
fi

# Stop backend to prevent connections
docker-compose stop backend

# Restore database
gunzip -c "$BACKUP_FILE" | docker exec -i sutazai-postgres psql -U sutazai sutazai

# Restart services
docker-compose start backend

echo "PostgreSQL restore completed"
```

**Complete System Recovery:**
```bash
#!/bin/bash
# disaster_recovery.sh
BACKUP_DATE=$1

if [ -z "$BACKUP_DATE" ]; then
    echo "Usage: $0 <YYYYMMDD>"
    exit 1
fi

echo "🚨 Starting disaster recovery for date: $BACKUP_DATE"

# Stop all services
docker-compose down

# Restore configuration
tar -xzf "/opt/sutazaiapp/backups/config/config_backup_${BACKUP_DATE}*.tar.gz" -C /

# Start core services
docker-compose up -d sutazai-postgres sutazai-redis

# Wait for initialization
sleep 60

# Restore PostgreSQL
POSTGRES_BACKUP=$(ls /opt/sutazaiapp/backups/postgres/postgres_backup_${BACKUP_DATE}*.sql.gz | head -1)
if [ -f "$POSTGRES_BACKUP" ]; then
    gunzip -c "$POSTGRES_BACKUP" | docker exec -i sutazai-postgres psql -U sutazai sutazai
fi

# Restore Redis
REDIS_BACKUP=$(ls /opt/sutazaiapp/backups/redis/redis_backup_${BACKUP_DATE}*.rdb | head -1)
if [ -f "$REDIS_BACKUP" ]; then
    docker cp "$REDIS_BACKUP" sutazai-redis:/data/dump.rdb
    docker-compose restart sutazai-redis
fi

# Start all services
docker-compose up -d

echo "✅ Disaster recovery completed"
```

## 📝 Log Analysis and Debugging

### 1. Log Locations

**Container Logs:**
```bash
# Backend API logs
docker-compose logs -f backend

# Ollama logs
docker-compose logs -f sutazai-ollama

# Database logs
docker-compose logs -f sutazai-postgres

# All services
docker-compose logs -f
```

**Application Logs:**
- Health monitoring: `/opt/sutazaiapp/logs/health_monitor.log`
- Deployment logs: `/opt/sutazaiapp/logs/deployment_*.log`
- Metrics: `/opt/sutazaiapp/logs/metrics/`

### 2. Log Analysis Commands

**Error Analysis:**
```bash
# Find errors in backend logs
docker-compose logs backend | grep -i error | tail -20

# Find critical issues
docker-compose logs | grep -i "critical\|fatal\|exception" | tail -10

# Check for memory issues
docker-compose logs | grep -i "memory\|oom" | tail -10
```

**Performance Analysis:**
```bash
# Check response times
docker-compose logs backend | grep "response_time" | tail -20

# Check database queries
docker-compose logs sutazai-postgres | grep "duration" | tail -10

# Monitor connection issues
docker-compose logs | grep -i "connection\|timeout" | tail -15
```

### 3. Debug Mode

**Enable Debug Logging:**
```bash
# Set debug environment variables
export SUTAZAI_DEBUG=true
export SUTAZAI_LOG_LEVEL=DEBUG

# Restart backend with debug mode
docker-compose restart backend

# Monitor debug logs
docker-compose logs -f backend | grep DEBUG
```

## 📊 Maintenance Tasks

### 1. Daily Maintenance

**Daily Health Check Script:**
```bash
#!/bin/bash
# daily_maintenance.sh
echo "=== DAILY MAINTENANCE - $(date) ==="

# Health check
./health_dashboard.sh

# Backup databases
./backup_postgres.sh
./backup_redis.sh

# Clean old logs (keep 7 days)
find /opt/sutazaiapp/logs -name "*.log" -mtime +7 -delete

# Docker cleanup
docker system prune -f --volumes

# Check disk space
df -h | grep -E '^/dev/'

echo "=== DAILY MAINTENANCE COMPLETE ==="
```

### 2. Weekly Maintenance

**Weekly Optimization:**
```bash
#!/bin/bash
# weekly_maintenance.sh
echo "=== WEEKLY MAINTENANCE - $(date) ==="

# Update models if needed
docker exec sutazai-ollama ollama pull tinyllama

# Database optimization
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "VACUUM ANALYZE;"

# Redis optimization
docker exec sutazai-redis redis-cli FLUSHDB

# Configuration backup
./backup_config.sh

# System update check
echo "System packages to update:"
apt list --upgradable 2>/dev/null | grep -v WARNING || echo "None"

echo "=== WEEKLY MAINTENANCE COMPLETE ==="
```

### 3. Monthly Maintenance

**Security and Updates:**
```bash
#!/bin/bash
# monthly_maintenance.sh
echo "=== MONTHLY MAINTENANCE - $(date) ==="

# Security audit
docker images --format "table {{.Repository}}\t{{.Tag}}\t{{.CreatedAt}}" | head -10

# Check for container updates
docker-compose pull

# Archive old backups (move to cold storage)
find /opt/sutazaiapp/backups -name "*.gz" -mtime +30 -exec mv {} /archive/sutazai/ \;

# Generate monthly report
echo "Monthly System Report - $(date)" > /opt/sutazaiapp/reports/monthly_$(date +%Y%m).txt
docker-compose ps >> /opt/sutazaiapp/reports/monthly_$(date +%Y%m).txt

echo "=== MONTHLY MAINTENANCE COMPLETE ==="
```

---

## 📞 Emergency Contacts

**System Operations Team:**
- Primary: ops-team@company.com
- Secondary: admin@company.com
- On-call: +1-xxx-xxx-xxxx

**Escalation Procedures:**
1. Level 1: Auto-restart services
2. Level 2: Manual intervention required
3. Level 3: System administrator contact
4. Level 4: Emergency escalation

---

*This runbook is based on the actual system implementation as of August 8, 2025. Update procedures as the system evolves.*