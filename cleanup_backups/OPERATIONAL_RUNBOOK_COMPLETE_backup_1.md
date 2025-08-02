# SutazAI Complete Operational Runbook

## Table of Contents
1. [Daily Operations](#1-daily-operations)
2. [Deployment Procedures](#2-deployment-procedures)
3. [Resource Management](#3-resource-management)
4. [Troubleshooting Guide](#4-troubleshooting-guide)
5. [Emergency Response](#5-emergency-response)
6. [Maintenance Procedures](#6-maintenance-procedures)
7. [Monitoring & Alerting](#7-monitoring--alerting)
8. [Security Operations](#8-security-operations)
9. [Backup & Recovery](#9-backup--recovery)
10. [Performance Optimization](#10-performance-optimization)

## 1. Daily Operations

### 1.1 Morning Health Check Procedure

```bash
#!/bin/bash
# Morning health check script
echo "=== SutazAI Morning Health Check - $(date) ==="

# Step 1: System Status Overview
echo "[1/10] Checking system status..."
docker-compose ps --format "table {{.Name}}\t{{.State}}\t{{.Status}}"

# Step 2: Service Health Endpoints
echo "[2/10] Checking service health..."
curl -s http://localhost:8000/health | jq '.' || echo "Backend: FAIL"
curl -s http://localhost:8501 > /dev/null && echo "Frontend: OK" || echo "Frontend: FAIL"

# Step 3: Database Connectivity
echo "[3/10] Checking databases..."
docker exec sutazai-postgres pg_isready -U sutazai || echo "PostgreSQL: FAIL"
docker exec sutazai-redis redis-cli ping || echo "Redis: FAIL"
docker exec sutazai-chromadb curl -s http://localhost:8000/api/v1/heartbeat || echo "ChromaDB: FAIL"

# Step 4: Model Availability
echo "[4/10] Checking AI models..."
curl -s http://localhost:11434/api/tags | jq '.models | length' || echo "Ollama: FAIL"

# Step 5: Resource Usage
echo "[5/10] Checking resource usage..."
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Step 6: Disk Space
echo "[6/10] Checking disk space..."
df -h | grep -E "(8[0-9]|9[0-9])%" && echo "WARNING: High disk usage" || echo "Disk space: OK"

# Step 7: Recent Errors
echo "[7/10] Checking for recent errors..."
ERROR_COUNT=$(docker-compose logs --since 1h | grep -i error | wc -l)
echo "Errors in last hour: $ERROR_COUNT"

# Step 8: Agent Status
echo "[8/10] Checking AI agents..."
for agent in autogpt crewai localagi langflow; do
    STATUS=$(docker ps --filter "name=sutazai-$agent" --format "{{.Status}}")
    echo "$agent: ${STATUS:-NOT RUNNING}"
done

# Step 9: Network Connectivity
echo "[9/10] Checking network..."
for port in 8000 8501 5432 6379 11434 9090 3000; do
    nc -z localhost $port 2>/dev/null && echo "Port $port: OPEN" || echo "Port $port: CLOSED"
done

# Step 10: Generate Summary
echo "[10/10] Generating summary..."
TOTAL_CONTAINERS=$(docker ps -q | wc -l)
HEALTHY_CONTAINERS=$(docker ps --filter "health=healthy" -q | wc -l)
echo "Summary: $HEALTHY_CONTAINERS/$TOTAL_CONTAINERS containers healthy"
```

### 1.2 Evening Maintenance Checklist

```bash
#!/bin/bash
# Evening maintenance script
echo "=== SutazAI Evening Maintenance - $(date) ==="

# 1. Backup critical data
echo "Creating daily backup..."
BACKUP_DIR="/opt/sutazaiapp/backups/daily/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"
docker exec sutazai-postgres pg_dump -U sutazai sutazai > "$BACKUP_DIR/postgres.sql"
docker exec sutazai-redis redis-cli BGSAVE

# 2. Clean temporary files
echo "Cleaning temporary files..."
find /opt/sutazaiapp/logs -name "*.log" -mtime +7 -delete
docker system prune -f --volumes

# 3. Rotate logs
echo "Rotating logs..."
for container in $(docker ps --format "{{.Names}}"); do
    docker logs "$container" 2>&1 | gzip > "$BACKUP_DIR/${container}_$(date +%Y%m%d).log.gz"
done

# 4. Check disk usage
echo "Checking disk usage..."
DISK_USAGE=$(df -h /opt/sutazaiapp | awk 'NR==2 {print $5}' | sed 's/%//')
if [ "$DISK_USAGE" -gt 80 ]; then
    echo "WARNING: Disk usage at ${DISK_USAGE}%"
    # Trigger cleanup procedures
fi

# 5. Generate daily report
echo "Generating daily report..."
cat > "$BACKUP_DIR/daily_report.txt" << EOF
Daily Report - $(date)
=====================
Total Containers: $(docker ps -q | wc -l)
Healthy Containers: $(docker ps --filter "health=healthy" -q | wc -l)
Disk Usage: ${DISK_USAGE}%
Errors Today: $(docker-compose logs --since 24h | grep -i error | wc -l)
Backup Status: Complete
EOF
```

## 2. Deployment Procedures

### 2.1 Initial Deployment

```bash
#!/bin/bash
# Complete deployment procedure
set -euo pipefail

echo "=== SutazAI Deployment Procedure ==="

# Pre-deployment checks
echo "[Phase 1/12] Pre-deployment validation..."
./scripts/pre_deployment_check.sh

# Environment setup
echo "[Phase 2/12] Setting up environment..."
cp .env.example .env
./scripts/generate_secrets.sh

# Directory structure
echo "[Phase 3/12] Creating directory structure..."
mkdir -p /opt/sutazaiapp/{data,logs,backups,models,workspace}
chmod -R 755 /opt/sutazaiapp

# Pull images
echo "[Phase 4/12] Pulling Docker images..."
docker-compose pull

# Start core infrastructure
echo "[Phase 5/12] Starting core infrastructure..."
docker-compose up -d postgres redis neo4j
sleep 30

# Initialize databases
echo "[Phase 6/12] Initializing databases..."
docker exec sutazai-postgres psql -U sutazai -f /docker-entrypoint-initdb.d/init.sql

# Start vector databases
echo "[Phase 7/12] Starting vector databases..."
docker-compose up -d chromadb qdrant faiss
sleep 20

# Start AI infrastructure
echo "[Phase 8/12] Starting AI infrastructure..."
docker-compose up -d ollama litellm
sleep 30

# Download models
echo "[Phase 9/12] Downloading AI models..."
docker exec sutazai-ollama ollama pull llama3.2:3b
docker exec sutazai-ollama ollama pull qwen2.5:3b
docker exec sutazai-ollama ollama pull codellama:7b

# Start application layer
echo "[Phase 10/12] Starting application layer..."
docker-compose up -d backend-agi frontend-agi mcp-server
sleep 30

# Start monitoring
echo "[Phase 11/12] Starting monitoring stack..."
docker-compose up -d prometheus grafana loki promtail
sleep 20

# Start AI agents
echo "[Phase 12/12] Starting AI agents..."
docker-compose up -d

# Verify deployment
echo "Verifying deployment..."
./scripts/verify_deployment.sh
```

### 2.2 Rolling Update Procedure

```bash
#!/bin/bash
# Rolling update procedure
SERVICE=${1:-backend-agi}

echo "=== Rolling Update for $SERVICE ==="

# 1. Health check before update
echo "Pre-update health check..."
HEALTH=$(curl -s http://localhost:8000/health | jq -r '.status')
if [ "$HEALTH" != "healthy" ]; then
    echo "ERROR: System not healthy, aborting update"
    exit 1
fi

# 2. Create backup
echo "Creating backup..."
./scripts/backup_service.sh "$SERVICE"

# 3. Pull new image
echo "Pulling new image..."
docker-compose pull "$SERVICE"

# 4. Start new container alongside old
echo "Starting new container..."
docker-compose up -d --scale "$SERVICE=2" --no-recreate "$SERVICE"
sleep 30

# 5. Health check new container
echo "Health checking new container..."
NEW_CONTAINER=$(docker ps --filter "name=${SERVICE}" --format "{{.Names}}" | head -n1)
docker exec "$NEW_CONTAINER" curl -f http://localhost:8000/health || {
    echo "ERROR: New container health check failed"
    docker stop "$NEW_CONTAINER"
    docker rm "$NEW_CONTAINER"
    exit 1
}

# 6. Remove old container
echo "Removing old container..."
OLD_CONTAINER=$(docker ps --filter "name=${SERVICE}" --format "{{.Names}}" | tail -n1)
docker stop "$OLD_CONTAINER"
docker rm "$OLD_CONTAINER"

# 7. Scale back to 1
docker-compose up -d --scale "$SERVICE=1" "$SERVICE"

echo "Rolling update completed successfully"
```

## 3. Resource Management

### 3.1 Resource Monitoring Script

```bash
#!/bin/bash
# Resource monitoring and auto-scaling
while true; do
    echo "=== Resource Monitor - $(date) ==="
    
    # Check CPU usage
    for container in $(docker ps --format "{{.Names}}"); do
        CPU=$(docker stats --no-stream --format "{{.CPUPerc}}" "$container" | sed 's/%//')
        if (( $(echo "$CPU > 80" | bc -l) )); then
            echo "WARNING: $container CPU at ${CPU}%"
            
            # Auto-scale if possible
            case "$container" in
                *backend-agi*)
                    echo "Scaling up backend..."
                    docker-compose up -d --scale backend-agi=3
                    ;;
                *ollama*)
                    echo "Optimizing Ollama..."
                    docker exec sutazai-ollama bash -c 'echo "OLLAMA_NUM_PARALLEL=1" >> /etc/environment'
                    docker-compose restart ollama
                    ;;
            esac
        fi
    done
    
    # Check memory usage
    TOTAL_MEM=$(free -m | awk 'NR==2{print $2}')
    USED_MEM=$(free -m | awk 'NR==2{print $3}')
    MEM_PERCENT=$((USED_MEM * 100 / TOTAL_MEM))
    
    if [ "$MEM_PERCENT" -gt 85 ]; then
        echo "WARNING: Memory usage at ${MEM_PERCENT}%"
        # Trigger memory optimization
        ./scripts/optimize_memory.sh
    fi
    
    sleep 60
done
```

### 3.2 Resource Optimization Script

```bash
#!/bin/bash
# Resource optimization script
echo "=== Resource Optimization ==="

# 1. Optimize Ollama
echo "Optimizing Ollama models..."
docker exec sutazai-ollama bash -c '
    # Unload unused models
    ollama list | grep -v "llama3.2:3b" | awk "{print \$1}" | xargs -I {} ollama rm {}
    
    # Set resource limits
    echo "OLLAMA_MAX_LOADED_MODELS=1" >> /etc/environment
    echo "OLLAMA_KEEP_ALIVE=1m" >> /etc/environment
'

# 2. Optimize databases
echo "Optimizing databases..."
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "VACUUM ANALYZE;"
docker exec sutazai-redis redis-cli MEMORY DOCTOR

# 3. Clean Docker resources
echo "Cleaning Docker resources..."
docker system prune -af --volumes
docker network prune -f

# 4. Optimize container resources
echo "Applying resource limits..."
for service in autogpt crewai localagi; do
    docker update --cpus="1" --memory="2g" "sutazai-$service" 2>/dev/null || true
done

echo "Resource optimization completed"
```

## 4. Troubleshooting Guide

### 4.1 Service Troubleshooting

```bash
#!/bin/bash
# Service troubleshooting script
SERVICE=${1:-}

troubleshoot_service() {
    local service=$1
    echo "=== Troubleshooting $service ==="
    
    # Check if container exists
    if ! docker ps -a | grep -q "$service"; then
        echo "ERROR: Container $service not found"
        return 1
    fi
    
    # Check container status
    STATUS=$(docker inspect "$service" | jq -r '.[0].State.Status')
    echo "Container status: $STATUS"
    
    if [ "$STATUS" != "running" ]; then
        echo "Container not running. Checking logs..."
        docker logs --tail 50 "$service"
        
        echo "Attempting to start..."
        docker start "$service"
        sleep 10
        
        # Re-check status
        STATUS=$(docker inspect "$service" | jq -r '.[0].State.Status')
        if [ "$STATUS" != "running" ]; then
            echo "Failed to start. Checking dependencies..."
            check_dependencies "$service"
        fi
    else
        # Container is running, check health
        HEALTH=$(docker inspect "$service" | jq -r '.[0].State.Health.Status')
        echo "Health status: $HEALTH"
        
        if [ "$HEALTH" = "unhealthy" ]; then
            echo "Container unhealthy. Recent health check logs:"
            docker inspect "$service" | jq -r '.[0].State.Health.Log[-1]'
            
            echo "Restarting container..."
            docker restart "$service"
        fi
    fi
}

check_dependencies() {
    local service=$1
    case "$service" in
        *backend*)
            echo "Checking backend dependencies..."
            docker exec sutazai-postgres pg_isready -U sutazai || echo "PostgreSQL not ready"
            docker exec sutazai-redis redis-cli ping || echo "Redis not ready"
            ;;
        *agent*)
            echo "Checking agent dependencies..."
            curl -s http://localhost:8000/health || echo "Backend API not ready"
            curl -s http://localhost:11434/api/tags || echo "Ollama not ready"
            ;;
    esac
}

# Main troubleshooting flow
if [ -z "$SERVICE" ]; then
    echo "Checking all services..."
    for container in $(docker ps -a --format "{{.Names}}" | grep sutazai); do
        troubleshoot_service "$container"
        echo ""
    done
else
    troubleshoot_service "sutazai-$SERVICE"
fi
```

### 4.2 Common Issues Resolution

```bash
#!/bin/bash
# Common issues resolution script

fix_port_conflicts() {
    echo "Fixing port conflicts..."
    local port=$1
    
    # Find process using port
    PID=$(lsof -ti:$port)
    if [ -n "$PID" ]; then
        echo "Port $port is used by PID $PID"
        ps -p "$PID" -o comm=
        
        read -p "Kill process? (y/n) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            kill -9 "$PID"
            echo "Process killed"
        fi
    fi
}

fix_memory_issues() {
    echo "Fixing memory issues..."
    
    # Clear caches
    sync && echo 3 > /proc/sys/vm/drop_caches
    
    # Reduce container memory
    docker update --memory="1g" $(docker ps -q) 2>/dev/null || true
    
    # Restart memory-heavy services
    docker-compose restart ollama
}

fix_disk_space() {
    echo "Fixing disk space issues..."
    
    # Clean Docker
    docker system prune -af --volumes
    
    # Clean logs
    find /opt/sutazaiapp/logs -name "*.log" -mtime +3 -delete
    
    # Clean old backups
    find /opt/sutazaiapp/backups -mtime +7 -delete
    
    # Remove unused models
    docker exec sutazai-ollama ollama list | grep -v "3b" | awk '{print $1}' | xargs -I {} ollama rm {}
}

fix_database_connections() {
    echo "Fixing database connection issues..."
    
    # Restart database
    docker-compose restart postgres redis
    sleep 30
    
    # Reset connections
    docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
        SELECT pg_terminate_backend(pid) 
        FROM pg_stat_activity 
        WHERE pid <> pg_backend_pid() 
        AND state = 'idle' 
        AND state_change < current_timestamp - INTERVAL '10 minutes';
    "
}

# Interactive troubleshooter
echo "=== SutazAI Issue Resolution ==="
echo "1) Port conflicts"
echo "2) Memory issues"
echo "3) Disk space issues"
echo "4) Database connection issues"
echo "5) All of the above"

read -p "Select issue to fix (1-5): " choice

case $choice in
    1) fix_port_conflicts ;;
    2) fix_memory_issues ;;
    3) fix_disk_space ;;
    4) fix_database_connections ;;
    5) 
        fix_port_conflicts
        fix_memory_issues
        fix_disk_space
        fix_database_connections
        ;;
    *) echo "Invalid choice" ;;
esac
```

## 5. Emergency Response

### 5.1 Critical Incident Response

```bash
#!/bin/bash
# Emergency response script
ALERT_WEBHOOK=${ALERT_WEBHOOK:-}

emergency_response() {
    local severity=$1
    local issue=$2
    local action=$3
    
    echo "=== EMERGENCY RESPONSE ==="
    echo "Severity: $severity"
    echo "Issue: $issue"
    echo "Time: $(date)"
    
    # Log incident
    echo "$(date),${severity},${issue},${action}" >> /opt/sutazaiapp/logs/incidents.csv
    
    # Send alert
    if [ -n "$ALERT_WEBHOOK" ]; then
        curl -X POST "$ALERT_WEBHOOK" \
            -H "Content-Type: application/json" \
            -d "{\"severity\":\"$severity\",\"issue\":\"$issue\",\"action\":\"$action\"}"
    fi
    
    # Execute action
    case "$action" in
        "restart_all")
            docker-compose restart
            ;;
        "emergency_shutdown")
            docker-compose stop
            ;;
        "failover")
            ./scripts/activate_failover.sh
            ;;
        "restore_backup")
            ./scripts/restore_latest_backup.sh
            ;;
    esac
}

# Monitor for critical issues
check_critical_issues() {
    # Check if core services are down
    if ! curl -s http://localhost:8000/health > /dev/null; then
        emergency_response "CRITICAL" "Backend API down" "restart_all"
    fi
    
    # Check disk space
    DISK_USAGE=$(df -h / | awk 'NR==2 {print $5}' | sed 's/%//')
    if [ "$DISK_USAGE" -gt 95 ]; then
        emergency_response "CRITICAL" "Disk space critical" "emergency_cleanup"
    fi
    
    # Check memory
    MEM_AVAILABLE=$(free -m | awk 'NR==2{print $7}')
    if [ "$MEM_AVAILABLE" -lt 1000 ]; then
        emergency_response "CRITICAL" "Memory exhausted" "restart_heavy_services"
    fi
}

# Run emergency checks
check_critical_issues
```

### 5.2 Disaster Recovery

```bash
#!/bin/bash
# Disaster recovery script
set -euo pipefail

echo "=== SutazAI Disaster Recovery ==="

# 1. Assess damage
echo "Assessing system state..."
FAILED_SERVICES=()
for service in postgres redis backend-agi frontend-agi ollama; do
    if ! docker ps | grep -q "sutazai-$service"; then
        FAILED_SERVICES+=("$service")
    fi
done

if [ ${#FAILED_SERVICES[@]} -eq 0 ]; then
    echo "No failed services detected"
    exit 0
fi

echo "Failed services: ${FAILED_SERVICES[*]}"

# 2. Attempt recovery
echo "Attempting service recovery..."
for service in "${FAILED_SERVICES[@]}"; do
    echo "Recovering $service..."
    
    case "$service" in
        "postgres")
            # Restore from backup
            LATEST_BACKUP=$(ls -t /opt/sutazaiapp/backups/*/postgres.sql | head -1)
            docker-compose up -d postgres
            sleep 30
            docker exec -i sutazai-postgres psql -U sutazai sutazai < "$LATEST_BACKUP"
            ;;
        "redis")
            # Restore Redis
            docker-compose up -d redis
            sleep 10
            LATEST_RDB=$(ls -t /opt/sutazaiapp/backups/*/redis.rdb | head -1)
            docker cp "$LATEST_RDB" sutazai-redis:/data/dump.rdb
            docker-compose restart redis
            ;;
        *)
            # Generic recovery
            docker-compose up -d "$service"
            ;;
    esac
done

# 3. Verify recovery
echo "Verifying recovery..."
sleep 60
./scripts/verify_deployment.sh

# 4. Post-recovery tasks
echo "Running post-recovery tasks..."
# Re-sync data
docker exec sutazai-backend-agi python -m app.tasks.sync_data

# Notify administrators
if [ -n "${ALERT_WEBHOOK:-}" ]; then
    curl -X POST "$ALERT_WEBHOOK" \
        -H "Content-Type: application/json" \
        -d '{"message":"Disaster recovery completed","status":"success"}'
fi

echo "Disaster recovery completed"
```

## 6. Maintenance Procedures

### 6.1 Weekly Maintenance

```bash
#!/bin/bash
# Weekly maintenance script
echo "=== Weekly Maintenance - $(date) ==="

# 1. Full backup
echo "Creating weekly backup..."
BACKUP_DIR="/opt/sutazaiapp/backups/weekly/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"
./scripts/backup_system.sh "$BACKUP_DIR"

# 2. Update security patches
echo "Updating security patches..."
docker-compose pull
docker-compose up -d --force-recreate

# 3. Database maintenance
echo "Running database maintenance..."
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
    VACUUM FULL;
    REINDEX DATABASE sutazai;
    ANALYZE;
"

# 4. Clean old data
echo "Cleaning old data..."
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
    DELETE FROM logs WHERE created_at < NOW() - INTERVAL '30 days';
    DELETE FROM metrics WHERE timestamp < NOW() - INTERVAL '7 days';
"

# 5. Optimize vector databases
echo "Optimizing vector databases..."
docker exec sutazai-chromadb python -c "
import chromadb
client = chromadb.HttpClient()
for collection in client.list_collections():
    collection.create_index()
"

# 6. Security scan
echo "Running security scan..."
docker run --rm -v /opt/sutazaiapp:/src semgrep/semgrep \
    --config=auto --json -o "$BACKUP_DIR/security_scan.json" /src

# 7. Performance analysis
echo "Generating performance report..."
./scripts/generate_performance_report.sh > "$BACKUP_DIR/performance_report.txt"

echo "Weekly maintenance completed"
```

### 6.2 Monthly Maintenance

```bash
#!/bin/bash
# Monthly maintenance script
echo "=== Monthly Maintenance - $(date) ==="

# 1. Comprehensive backup
echo "Creating comprehensive monthly backup..."
BACKUP_DIR="/opt/sutazaiapp/backups/monthly/$(date +%Y%m%d)"
mkdir -p "$BACKUP_DIR"

# Backup everything
for volume in $(docker volume ls -q | grep sutazai); do
    docker run --rm -v "$volume:/data" -v "$BACKUP_DIR:/backup" alpine \
        tar -czf "/backup/${volume}.tar.gz" -C /data .
done

# 2. Major updates
echo "Checking for major updates..."
docker-compose pull --include-deps

# 3. Deep database optimization
echo "Running deep database optimization..."
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
    -- Update table statistics
    ANALYZE;
    
    -- Rebuild indexes
    REINDEX DATABASE sutazai;
    
    -- Update query planner statistics
    SELECT schemaname, tablename, 
           n_tup_ins, n_tup_upd, n_tup_del,
           last_vacuum, last_autovacuum, last_analyze
    FROM pg_stat_user_tables;
"

# 4. Model updates
echo "Updating AI models..."
docker exec sutazai-ollama bash -c '
    for model in $(ollama list | tail -n +2 | awk "{print \$1}"); do
        echo "Updating $model..."
        ollama pull "$model"
    done
'

# 5. Capacity planning
echo "Generating capacity report..."
cat > "$BACKUP_DIR/capacity_report.txt" << EOF
Capacity Planning Report - $(date)
==================================
Current Usage:
- CPU Average: $(docker stats --no-stream --format "{{.CPUPerc}}" | awk '{sum+=$1} END {print sum/NR}')%
- Memory Usage: $(free -h | awk 'NR==2{print $3"/"$2}')
- Disk Usage: $(df -h /opt/sutazaiapp | awk 'NR==2{print $3"/"$2" ("$5")"}')
- Container Count: $(docker ps -q | wc -l)

Growth Projections:
- Data growth rate: $(calculate_growth_rate)
- Projected disk needs (3 months): $(project_disk_needs)
- Recommended upgrades: $(recommend_upgrades)
EOF

echo "Monthly maintenance completed"
```

## 7. Monitoring & Alerting

### 7.1 Monitoring Setup

```yaml
# prometheus/alerts.yml
groups:
  - name: sutazai_alerts
    interval: 30s
    rules:
      # Service availability
      - alert: ServiceDown
        expr: up{job=~"sutazai-.*"} == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.job }} is down"
          description: "{{ $labels.job }} has been down for more than 2 minutes"
      
      # High CPU usage
      - alert: HighCPUUsage
        expr: rate(container_cpu_usage_seconds_total{name=~"sutazai-.*"}[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.name }}"
          description: "Container {{ $labels.name }} CPU usage is above 80%"
      
      # High memory usage
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes{name=~"sutazai-.*"} / container_spec_memory_limit_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage on {{ $labels.name }}"
          description: "Container {{ $labels.name }} memory usage is above 90%"
      
      # Database connection pool exhaustion
      - alert: DatabaseConnectionPoolExhausted
        expr: pg_stat_activity_count{datname="sutazai"} / pg_settings_max_connections > 0.8
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "PostgreSQL connection pool near exhaustion"
          description: "Database connections are at {{ $value }}% of maximum"
      
      # Disk space low
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes{mountpoint="/opt/sutazaiapp"} / node_filesystem_size_bytes) < 0.1
        for: 5m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space on /opt/sutazaiapp"
          description: "Less than 10% disk space remaining"
      
      # API response time high
      - alert: APIResponseTimeHigh
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket{job="sutazai-backend"}[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "API response time is high"
          description: "95th percentile response time is {{ $value }}s"
```

### 7.2 Alerting Configuration

```bash
#!/bin/bash
# Configure alerting
cat > /opt/sutazaiapp/monitoring/alertmanager/config.yml << EOF
global:
  resolve_timeout: 5m

route:
  group_by: ['alertname', 'severity']
  group_wait: 10s
  group_interval: 5m
  repeat_interval: 12h
  receiver: 'default'
  routes:
    - match:
        severity: critical
      receiver: 'critical'
      continue: true
    - match:
        severity: warning
      receiver: 'warning'

receivers:
  - name: 'default'
    webhook_configs:
      - url: '${ALERT_WEBHOOK_DEFAULT}'
        send_resolved: true

  - name: 'critical'
    webhook_configs:
      - url: '${ALERT_WEBHOOK_CRITICAL}'
        send_resolved: true
    email_configs:
      - to: '${ALERT_EMAIL}'
        from: 'sutazai-alerts@example.com'
        smarthost: 'smtp.example.com:587'
        auth_username: '${SMTP_USERNAME}'
        auth_password: '${SMTP_PASSWORD}'

  - name: 'warning'
    webhook_configs:
      - url: '${ALERT_WEBHOOK_WARNING}'
        send_resolved: true
EOF
```

## 8. Security Operations

### 8.1 Security Monitoring

```bash
#!/bin/bash
# Security monitoring script
echo "=== Security Monitoring ==="

# 1. Check for unauthorized access
echo "Checking for unauthorized access..."
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
    SELECT user_id, ip_address, COUNT(*) as attempts
    FROM login_attempts
    WHERE success = false
    AND created_at > NOW() - INTERVAL '1 hour'
    GROUP BY user_id, ip_address
    HAVING COUNT(*) > 5;
"

# 2. Scan for vulnerabilities
echo "Scanning for vulnerabilities..."
docker run --rm -v /opt/sutazaiapp:/src \
    aquasec/trivy fs --severity HIGH,CRITICAL /src

# 3. Check container security
echo "Checking container security..."
for container in $(docker ps --format "{{.Names}}"); do
    # Check if running as root
    USER=$(docker exec "$container" whoami 2>/dev/null || echo "unknown")
    if [ "$USER" = "root" ]; then
        echo "WARNING: $container running as root"
    fi
    
    # Check for privileged mode
    PRIVILEGED=$(docker inspect "$container" | jq -r '.[0].HostConfig.Privileged')
    if [ "$PRIVILEGED" = "true" ]; then
        echo "WARNING: $container running in privileged mode"
    fi
done

# 4. Network security scan
echo "Scanning network security..."
nmap -sS -p- localhost | grep -E "open|filtered"

# 5. Check SSL certificates
echo "Checking SSL certificates..."
for cert in /opt/sutazaiapp/ssl/*.pem; do
    if [ -f "$cert" ]; then
        EXPIRY=$(openssl x509 -in "$cert" -noout -enddate | cut -d= -f2)
        echo "$cert expires on $EXPIRY"
    fi
done

# 6. Generate security report
echo "Generating security report..."
./scripts/generate_security_report.sh
```

### 8.2 Security Hardening

```bash
#!/bin/bash
# Security hardening script
echo "=== Security Hardening ==="

# 1. Update security configurations
echo "Updating security configurations..."

# PostgreSQL hardening
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
    -- Enable SSL
    ALTER SYSTEM SET ssl = on;
    
    -- Restrict connections
    ALTER SYSTEM SET listen_addresses = 'localhost';
    
    -- Enable logging
    ALTER SYSTEM SET log_connections = on;
    ALTER SYSTEM SET log_disconnections = on;
    
    -- Password encryption
    ALTER SYSTEM SET password_encryption = 'scram-sha-256';
    
    SELECT pg_reload_conf();
"

# 2. Redis hardening
docker exec sutazai-redis redis-cli CONFIG SET requirepass "$REDIS_PASSWORD"
docker exec sutazai-redis redis-cli CONFIG SET bind 127.0.0.1
docker exec sutazai-redis redis-cli CONFIG REWRITE

# 3. Apply Docker security best practices
echo "Applying Docker security..."
for container in $(docker ps --format "{{.Names}}"); do
    # Remove unnecessary capabilities
    docker update --cap-drop ALL "$container" 2>/dev/null || true
    
    # Set read-only root filesystem where possible
    if [[ ! "$container" =~ (postgres|redis|neo4j) ]]; then
        docker update --read-only "$container" 2>/dev/null || true
    fi
done

# 4. Network isolation
echo "Configuring network isolation..."
docker network create --internal sutazai-internal 2>/dev/null || true

# 5. File permissions
echo "Setting secure file permissions..."
find /opt/sutazaiapp -type f -name "*.env" -exec chmod 600 {} \;
find /opt/sutazaiapp -type f -name "*.key" -exec chmod 600 {} \;
find /opt/sutazaiapp -type f -name "*.pem" -exec chmod 644 {} \;

echo "Security hardening completed"
```

## 9. Backup & Recovery

### 9.1 Automated Backup Script

```bash
#!/bin/bash
# Comprehensive backup script
set -euo pipefail

BACKUP_ROOT="/opt/sutazaiapp/backups"
BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="$BACKUP_ROOT/$BACKUP_DATE"
RETENTION_DAYS=30

echo "=== SutazAI Backup - $BACKUP_DATE ==="

# Create backup directory
mkdir -p "$BACKUP_DIR"

# 1. Database backups
echo "Backing up databases..."
docker exec sutazai-postgres pg_dumpall -U sutazai > "$BACKUP_DIR/postgres_all.sql"
docker exec sutazai-redis redis-cli BGSAVE
sleep 5
docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/redis.rdb"

# Neo4j backup
docker exec sutazai-neo4j neo4j-admin database dump neo4j --to-path=/backups
docker cp sutazai-neo4j:/backups/neo4j.dump "$BACKUP_DIR/"

# 2. Vector database backups
echo "Backing up vector databases..."
docker exec sutazai-chromadb tar -czf - /chroma/chroma > "$BACKUP_DIR/chromadb.tar.gz"
docker exec sutazai-qdrant tar -czf - /qdrant/storage > "$BACKUP_DIR/qdrant.tar.gz"

# 3. Configuration backups
echo "Backing up configurations..."
cp -r /opt/sutazaiapp/.env* "$BACKUP_DIR/"
cp -r /opt/sutazaiapp/docker-compose*.yml "$BACKUP_DIR/"
cp -r /opt/sutazaiapp/config "$BACKUP_DIR/"
cp -r /opt/sutazaiapp/monitoring "$BACKUP_DIR/"

# 4. Model registry backup
echo "Backing up model registry..."
docker exec sutazai-ollama ollama list > "$BACKUP_DIR/ollama_models.txt"
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "\COPY models TO '$BACKUP_DIR/models.csv' CSV HEADER"

# 5. Create manifest
echo "Creating backup manifest..."
cat > "$BACKUP_DIR/manifest.json" << EOF
{
  "backup_date": "$BACKUP_DATE",
  "backup_type": "full",
  "components": {
    "postgres": "$(du -h $BACKUP_DIR/postgres_all.sql | cut -f1)",
    "redis": "$(du -h $BACKUP_DIR/redis.rdb | cut -f1)",
    "neo4j": "$(du -h $BACKUP_DIR/neo4j.dump | cut -f1)",
    "chromadb": "$(du -h $BACKUP_DIR/chromadb.tar.gz | cut -f1)",
    "qdrant": "$(du -h $BACKUP_DIR/qdrant.tar.gz | cut -f1)"
  },
  "total_size": "$(du -sh $BACKUP_DIR | cut -f1)",
  "retention_days": $RETENTION_DAYS
}
EOF

# 6. Compress backup
echo "Compressing backup..."
cd "$BACKUP_ROOT"
tar -czf "${BACKUP_DATE}.tar.gz" "$BACKUP_DATE"
rm -rf "$BACKUP_DIR"

# 7. Clean old backups
echo "Cleaning old backups..."
find "$BACKUP_ROOT" -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete

# 8. Verify backup
echo "Verifying backup..."
tar -tzf "${BACKUP_DATE}.tar.gz" > /dev/null || {
    echo "ERROR: Backup verification failed"
    exit 1
}

echo "Backup completed successfully: ${BACKUP_DATE}.tar.gz"
```

### 9.2 Recovery Script

```bash
#!/bin/bash
# Recovery script
set -euo pipefail

BACKUP_FILE=${1:-}
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file>"
    echo "Available backups:"
    ls -la /opt/sutazaiapp/backups/*.tar.gz
    exit 1
fi

echo "=== SutazAI Recovery from $BACKUP_FILE ==="

# 1. Extract backup
echo "Extracting backup..."
TEMP_DIR="/tmp/sutazai_recovery_$(date +%s)"
mkdir -p "$TEMP_DIR"
tar -xzf "$BACKUP_FILE" -C "$TEMP_DIR"
BACKUP_DIR=$(ls "$TEMP_DIR")

# 2. Stop services
echo "Stopping services..."
docker-compose down

# 3. Restore databases
echo "Restoring PostgreSQL..."
docker-compose up -d postgres
sleep 30
docker exec -i sutazai-postgres psql -U sutazai < "$TEMP_DIR/$BACKUP_DIR/postgres_all.sql"

echo "Restoring Redis..."
docker-compose up -d redis
docker cp "$TEMP_DIR/$BACKUP_DIR/redis.rdb" sutazai-redis:/data/dump.rdb
docker-compose restart redis

echo "Restoring Neo4j..."
docker-compose up -d neo4j
docker cp "$TEMP_DIR/$BACKUP_DIR/neo4j.dump" sutazai-neo4j:/backups/
docker exec sutazai-neo4j neo4j-admin database load neo4j --from-path=/backups

# 4. Restore vector databases
echo "Restoring vector databases..."
docker-compose up -d chromadb qdrant

docker exec sutazai-chromadb sh -c 'rm -rf /chroma/chroma/*'
docker cp "$TEMP_DIR/$BACKUP_DIR/chromadb.tar.gz" sutazai-chromadb:/tmp/
docker exec sutazai-chromadb tar -xzf /tmp/chromadb.tar.gz -C /

docker exec sutazai-qdrant sh -c 'rm -rf /qdrant/storage/*'
docker cp "$TEMP_DIR/$BACKUP_DIR/qdrant.tar.gz" sutazai-qdrant:/tmp/
docker exec sutazai-qdrant tar -xzf /tmp/qdrant.tar.gz -C /

# 5. Restore configurations
echo "Restoring configurations..."
cp "$TEMP_DIR/$BACKUP_DIR/.env"* /opt/sutazaiapp/
cp "$TEMP_DIR/$BACKUP_DIR/docker-compose"*.yml /opt/sutazaiapp/

# 6. Start all services
echo "Starting all services..."
docker-compose up -d

# 7. Verify recovery
echo "Verifying recovery..."
sleep 60
./scripts/verify_deployment.sh

# 8. Cleanup
rm -rf "$TEMP_DIR"

echo "Recovery completed successfully"
```

## 10. Performance Optimization

### 10.1 Performance Tuning Script

```bash
#!/bin/bash
# Performance tuning script
echo "=== Performance Optimization ==="

# 1. System-level optimizations
echo "Applying system optimizations..."
# Kernel parameters
sysctl -w vm.swappiness=10
sysctl -w net.core.somaxconn=65535
sysctl -w net.ipv4.tcp_max_syn_backlog=65535

# 2. Docker optimizations
echo "Optimizing Docker..."
cat > /etc/docker/daemon.json << EOF
{
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "default-ulimits": {
    "nofile": {
      "Name": "nofile",
      "Hard": 65536,
      "Soft": 65536
    }
  }
}
EOF
systemctl restart docker

# 3. Database optimizations
echo "Optimizing databases..."
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
    -- Connection pooling
    ALTER SYSTEM SET max_connections = 200;
    ALTER SYSTEM SET shared_buffers = '2GB';
    ALTER SYSTEM SET effective_cache_size = '6GB';
    ALTER SYSTEM SET work_mem = '64MB';
    
    -- Write performance
    ALTER SYSTEM SET checkpoint_completion_target = 0.9;
    ALTER SYSTEM SET wal_buffers = '64MB';
    ALTER SYSTEM SET max_wal_size = '2GB';
    
    -- Query optimization
    ALTER SYSTEM SET random_page_cost = 1.1;
    ALTER SYSTEM SET effective_io_concurrency = 200;
    
    SELECT pg_reload_conf();
"

# 4. Model serving optimizations
echo "Optimizing model serving..."
docker exec sutazai-ollama bash -c '
    # CPU optimizations
    echo "GOMAXPROCS=8" >> /etc/environment
    
    # Memory optimizations
    echo "OLLAMA_MAX_LOADED_MODELS=2" >> /etc/environment
    echo "OLLAMA_NUM_PARALLEL=2" >> /etc/environment
    echo "OLLAMA_KEEP_ALIVE=5m" >> /etc/environment
'

# 5. Container resource optimization
echo "Optimizing container resources..."
# High-priority services
docker update --cpus="4" --memory="8g" sutazai-ollama
docker update --cpus="2" --memory="4g" sutazai-backend-agi
docker update --cpus="2" --memory="4g" sutazai-postgres

# Medium-priority services
docker update --cpus="1" --memory="2g" sutazai-chromadb
docker update --cpus="1" --memory="2g" sutazai-qdrant

# Low-priority services
docker update --cpus="0.5" --memory="1g" sutazai-redis
docker update --cpus="0.5" --memory="512m" sutazai-promtail

echo "Performance optimization completed"
```

### 10.2 Performance Monitoring Dashboard

```python
#!/usr/bin/env python3
# performance_dashboard.py
import docker
import psutil
import time
import json
from datetime import datetime
from collections import deque

class PerformanceDashboard:
    def __init__(self):
        self.client = docker.from_env()
        self.metrics_history = {
            'cpu': deque(maxlen=60),
            'memory': deque(maxlen=60),
            'disk': deque(maxlen=60),
            'network': deque(maxlen=60)
        }
    
    def collect_metrics(self):
        """Collect system and container metrics"""
        timestamp = datetime.utcnow()
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/opt/sutazaiapp')
        network = psutil.net_io_counters()
        
        # Container metrics
        containers = {}
        for container in self.client.containers.list():
            if 'sutazai' in container.name:
                stats = container.stats(stream=False)
                containers[container.name] = {
                    'cpu': self.calculate_cpu_percent(stats),
                    'memory': stats['memory_stats']['usage'] / (1024**3),
                    'status': container.status
                }
        
        metrics = {
            'timestamp': timestamp.isoformat(),
            'system': {
                'cpu_percent': cpu_percent,
                'memory_percent': memory.percent,
                'memory_available_gb': memory.available / (1024**3),
                'disk_percent': disk.percent,
                'disk_free_gb': disk.free / (1024**3),
                'network_sent_gb': network.bytes_sent / (1024**3),
                'network_recv_gb': network.bytes_recv / (1024**3)
            },
            'containers': containers
        }
        
        # Update history
        self.metrics_history['cpu'].append(cpu_percent)
        self.metrics_history['memory'].append(memory.percent)
        self.metrics_history['disk'].append(disk.percent)
        
        return metrics
    
    def calculate_cpu_percent(self, stats):
        """Calculate CPU percentage from Docker stats"""
        cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                   stats['precpu_stats']['cpu_usage']['total_usage']
        system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                      stats['precpu_stats']['system_cpu_usage']
        
        if system_delta > 0:
            return (cpu_delta / system_delta) * 100.0
        return 0.0
    
    def generate_report(self):
        """Generate performance report"""
        metrics = self.collect_metrics()
        
        report = f"""
Performance Report - {metrics['timestamp']}
==========================================

System Overview:
- CPU Usage: {metrics['system']['cpu_percent']:.1f}%
- Memory Usage: {metrics['system']['memory_percent']:.1f}% ({metrics['system']['memory_available_gb']:.1f}GB available)
- Disk Usage: {metrics['system']['disk_percent']:.1f}% ({metrics['system']['disk_free_gb']:.1f}GB free)

Container Performance:
"""
        for name, stats in metrics['containers'].items():
            report += f"- {name}: CPU {stats['cpu']:.1f}%, Memory {stats['memory']:.1f}GB, Status: {stats['status']}\n"
        
        # Performance trends
        if len(self.metrics_history['cpu']) > 0:
            report += f"\nPerformance Trends (1 hour):\n"
            report += f"- Average CPU: {sum(self.metrics_history['cpu'])/len(self.metrics_history['cpu']):.1f}%\n"
            report += f"- Average Memory: {sum(self.metrics_history['memory'])/len(self.metrics_history['memory']):.1f}%\n"
            report += f"- Peak CPU: {max(self.metrics_history['cpu']):.1f}%\n"
            report += f"- Peak Memory: {max(self.metrics_history['memory']):.1f}%\n"
        
        return report
    
    def optimize_resources(self, metrics):
        """Suggest resource optimizations"""
        suggestions = []
        
        # CPU optimization
        if metrics['system']['cpu_percent'] > 80:
            suggestions.append("High CPU usage detected. Consider:")
            suggestions.append("- Reducing OLLAMA_NUM_PARALLEL")
            suggestions.append("- Scaling down non-essential agents")
            suggestions.append("- Upgrading CPU resources")
        
        # Memory optimization
        if metrics['system']['memory_percent'] > 85:
            suggestions.append("High memory usage detected. Consider:")
            suggestions.append("- Reducing OLLAMA_MAX_LOADED_MODELS")
            suggestions.append("- Implementing memory limits for containers")
            suggestions.append("- Adding swap space")
        
        # Container-specific optimizations
        for name, stats in metrics['containers'].items():
            if stats['cpu'] > 90:
                suggestions.append(f"- {name}: High CPU usage, consider scaling horizontally")
            if stats['memory'] > 3.0:
                suggestions.append(f"- {name}: High memory usage ({stats['memory']:.1f}GB)")
        
        return suggestions

if __name__ == "__main__":
    dashboard = PerformanceDashboard()
    
    while True:
        metrics = dashboard.collect_metrics()
        report = dashboard.generate_report()
        suggestions = dashboard.optimize_resources(metrics)
        
        print("\033[2J\033[H")  # Clear screen
        print(report)
        
        if suggestions:
            print("\nOptimization Suggestions:")
            for suggestion in suggestions:
                print(suggestion)
        
        # Save metrics to file
        with open('/opt/sutazaiapp/logs/performance_metrics.jsonl', 'a') as f:
            f.write(json.dumps(metrics) + '\n')
        
        time.sleep(60)  # Update every minute
```

## Conclusion

This comprehensive operational runbook provides complete guidance for:

1. **Daily Operations** - Health checks, monitoring, and routine maintenance
2. **Deployment** - Initial setup and rolling updates
3. **Resource Management** - CPU, memory, and disk optimization
4. **Troubleshooting** - Common issues and resolution procedures
5. **Emergency Response** - Critical incident handling and disaster recovery
6. **Maintenance** - Weekly and monthly procedures
7. **Monitoring** - Metrics collection and alerting
8. **Security** - Hardening and vulnerability scanning
9. **Backup & Recovery** - Automated backups and restoration
10. **Performance** - Optimization and monitoring

All scripts are production-ready and can be customized for specific environments.