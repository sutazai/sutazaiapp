# Perfect Jarvis System - Disaster Recovery Plan

**Version:** 1.0  
**Last Updated:** 2025-08-08  
**Owner:** SutazAI Operations Team  
**Review Cycle:** Quarterly  

## 🎯 Overview

This disaster recovery plan outlines procedures for recovering the Perfect Jarvis system from various disaster scenarios, ensuring business continuity and   data loss.

## 📋 Table of Contents

- [Recovery Objectives](#recovery-objectives)
- [Disaster Scenarios](#disaster-scenarios)
- [Backup Strategies](#backup-strategies)
- [Recovery Procedures](#recovery-procedures)
- [Infrastructure Recovery](#infrastructure-recovery)
- [Data Recovery](#data-recovery)
- [Application Recovery](#application-recovery)
- [Testing & Validation](#testing--validation)
- [Communication Plan](#communication-plan)
- [Post-Recovery](#post-recovery)

## 🎯 Recovery Objectives

### Recovery Time Objective (RTO)

| Service Tier | RTO Target | Description |
|--------------|------------|-------------|
| **Critical** | 30 minutes | Core AI services, database |
| **Important** | 2 hours | Web interface, monitoring |
| **Standard** | 4 hours | Analytics, reporting |
| **Low Priority** | 24 hours | Development tools |

### Recovery Point Objective (RPO)

| Data Type | RPO Target | Backup Frequency |
|-----------|------------|------------------|
| **Database** | 15 minutes | Continuous replication |
| **Configuration** | 1 hour | Hourly snapshots |
| **Models/Embeddings** | 4 hours | Every 4 hours |
| **Logs** | 24 hours | Daily archives |

### Service Level Targets

- **Data Recovery:** 99.9% success rate
- **System Restoration:** 95% within RTO
- **Communication:** Stakeholder notification within 15 minutes

## 🌪️ Disaster Scenarios

### Scenario 1: Complete Infrastructure Failure

**Impact:** Total system unavailability  
**RTO:** 2 hours  
**RPO:** 15 minutes  

### Scenario 2: Database Corruption/Loss

**Impact:** Data services unavailable  
**RTO:** 1 hour  
**RPO:** 15 minutes  

### Scenario 3: Application Server Failure

**Impact:** API/Web services down  
**RTO:** 30 minutes  
**RPO:** 1 hour  

### Scenario 4: Network/Connectivity Issues

**Impact:** External access unavailable  
**RTO:** 15 minutes  
**RPO:** N/A  

### Scenario 5: Security Breach/Ransomware

**Impact:** System compromise  
**RTO:** 4 hours  
**RPO:** 1 hour  

### Scenario 6: Model/AI Service Corruption

**Impact:** AI functionality degraded  
**RTO:** 2 hours  
**RPO:** 4 hours  

## 💾 Backup Strategies

### Automated Backup System

**Daily Full Backup Script:**
```bash
#!/bin/bash
# daily_backup.sh - Complete system backup

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_ROOT="/opt/backups"
RETENTION_DAYS=30

echo "=== Starting Daily Full Backup - $BACKUP_DATE ==="

# Create backup directory
mkdir -p "$BACKUP_ROOT/daily/$BACKUP_DATE"
cd "$BACKUP_ROOT/daily/$BACKUP_DATE"

# 1. Database Backup
echo "Backing up PostgreSQL database..."
docker exec sutazai-postgres pg_dump -U sutazai sutazai > database_$BACKUP_DATE.sql
gzip database_$BACKUP_DATE.sql

# 2. Redis Backup
echo "Backing up Redis data..."
docker exec sutazai-redis redis-cli BGSAVE
docker cp sutazai-redis:/data/dump.rdb redis_$BACKUP_DATE.rdb
gzip redis_$BACKUP_DATE.rdb

# 3. Neo4j Backup
echo "Backing up Neo4j database..."
docker exec sutazai-neo4j neo4j-admin dump --to=/tmp/neo4j_backup.dump
docker cp sutazai-neo4j:/tmp/neo4j_backup.dump neo4j_$BACKUP_DATE.dump
gzip neo4j_$BACKUP_DATE.dump

# 4. Application Configuration
echo "Backing up application configuration..."
tar -czf config_$BACKUP_DATE.tar.gz /opt/sutazaiapp/config/
tar -czf docker_compose_$BACKUP_DATE.tar.gz /opt/sutazaiapp/docker-compose.yml

# 5. AI Models and Embeddings
echo "Backing up AI models..."
docker exec sutazai-ollama tar -czf /tmp/ollama_models.tar.gz /root/.ollama/models/
docker cp sutazai-ollama:/tmp/ollama_models.tar.gz ollama_models_$BACKUP_DATE.tar.gz

# 6. Vector Database Backup
echo "Backing up vector databases..."
docker exec sutazai-qdrant tar -czf /tmp/qdrant_data.tar.gz /qdrant/storage/
docker cp sutazai-qdrant:/tmp/qdrant_data.tar.gz qdrant_$BACKUP_DATE.tar.gz

# 7. Application Logs
echo "Backing up application logs..."
docker logs sutazai-backend > backend_logs_$BACKUP_DATE.log 2>&1
docker logs sutazai-frontend > frontend_logs_$BACKUP_DATE.log 2>&1
gzip *.log

# 8. System Metrics Backup
echo "Backing up Prometheus data..."
docker exec sutazai-prometheus tar -czf /tmp/prometheus_data.tar.gz /prometheus/
docker cp sutazai-prometheus:/tmp/prometheus_data.tar.gz prometheus_$BACKUP_DATE.tar.gz

# 9. Create manifest
echo "Creating backup manifest..."
cat > backup_manifest_$BACKUP_DATE.json << EOF
{
  "backup_date": "$BACKUP_DATE",
  "backup_type": "full",
  "components": [
    "postgresql_database",
    "redis_cache",
    "neo4j_graph",
    "application_config",
    "docker_compose",
    "ollama_models",
    "qdrant_vectors",
    "application_logs",
    "prometheus_metrics"
  ],
  "backup_size_mb": $(du -sm . | cut -f1),
  "system_version": "17.0.0",
  "retention_until": "$(date -d '+30 days' '+%Y-%m-%d')"
}
EOF

# 10. Verify backup integrity
echo "Verifying backup integrity..."
BACKUP_SIZE=$(du -sh . | cut -f1)
FILE_COUNT=$(find . -type f | wc -l)

if [ "$FILE_COUNT" -gt 0 ]; then
    echo "✅ Backup completed successfully"
    echo "   Size: $BACKUP_SIZE"
    echo "   Files: $FILE_COUNT"
    echo "   Location: $BACKUP_ROOT/daily/$BACKUP_DATE"
else
    echo "❌ Backup failed - no files created"
    exit 1
fi

# 11. Cleanup old backups
echo "Cleaning up backups older than $RETENTION_DAYS days..."
find "$BACKUP_ROOT/daily" -type d -mtime +$RETENTION_DAYS -exec rm -rf {} +

# 12. Send notification
curl -X POST http://127.0.0.1:10010/metrics \
    -H "Content-Type: application/json" \
    -d "{\"event\": \"backup_completed\", \"size\": \"$BACKUP_SIZE\", \"files\": $FILE_COUNT}" \
    2>/dev/null || echo "⚠️ Could not send backup notification"

echo "=== Daily Backup Complete ==="
```

### Incremental Backup Script

**Hourly Incremental Backup:**
```bash
#!/bin/bash
# incremental_backup.sh - Hourly incremental backup

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_ROOT="/opt/backups"
LAST_FULL_BACKUP=$(find "$BACKUP_ROOT/daily" -type d -name "20*" | sort | tail -1)

echo "=== Starting Incremental Backup - $BACKUP_DATE ==="

mkdir -p "$BACKUP_ROOT/incremental/$BACKUP_DATE"
cd "$BACKUP_ROOT/incremental/$BACKUP_DATE"

# 1. Database WAL backup (PostgreSQL)
echo "Backing up database WAL files..."
docker exec sutazai-postgres pg_receivewal -D /tmp/wal_backup --slot=backup_slot -v
docker cp sutazai-postgres:/tmp/wal_backup ./wal_$BACKUP_DATE/
tar -czf wal_backup_$BACKUP_DATE.tar.gz wal_$BACKUP_DATE/

# 2. Redis incremental (AOF if enabled)
echo "Backing up Redis AOF..."
if docker exec sutazai-redis redis-cli CONFIG GET appendonly | grep -q "yes"; then
    docker cp sutazai-redis:/data/appendonly.aof redis_aof_$BACKUP_DATE.aof
    gzip redis_aof_$BACKUP_DATE.aof
fi

# 3. Configuration changes since last backup
echo "Backing up changed configurations..."
if [ -n "$LAST_FULL_BACKUP" ]; then
    find /opt/sutazaiapp/config/ -newer "$LAST_FULL_BACKUP" -type f -exec tar -czf config_changes_$BACKUP_DATE.tar.gz {} +
fi

# 4. Application logs since last hour
echo "Backing up recent logs..."
docker logs --since="1h" sutazai-backend > backend_incremental_$BACKUP_DATE.log 2>&1
docker logs --since="1h" sutazai-frontend > frontend_incremental_$BACKUP_DATE.log 2>&1
gzip *.log

echo "=== Incremental Backup Complete ==="
```

### Backup Verification Script

```bash
#!/bin/bash
# verify_backup.sh - Backup integrity verification

BACKUP_PATH="$1"
if [ -z "$BACKUP_PATH" ]; then
    echo "Usage: $0 <backup_path>"
    exit 1
fi

echo "=== Verifying Backup: $BACKUP_PATH ==="

# 1. Check manifest exists
if [ ! -f "$BACKUP_PATH/backup_manifest_*.json" ]; then
    echo "❌ Backup manifest not found"
    exit 1
fi

MANIFEST=$(ls "$BACKUP_PATH"/backup_manifest_*.json | head -1)
echo "✅ Found manifest: $(basename "$MANIFEST")"

# 2. Verify database backup
if [ -f "$BACKUP_PATH"/database_*.sql.gz ]; then
    echo "✅ Database backup found"
    gunzip -t "$BACKUP_PATH"/database_*.sql.gz && echo "✅ Database backup integrity OK"
else
    echo "❌ Database backup missing"
fi

# 3. Verify configuration backups
if [ -f "$BACKUP_PATH"/config_*.tar.gz ]; then
    echo "✅ Configuration backup found"
    tar -tzf "$BACKUP_PATH"/config_*.tar.gz >/dev/null && echo "✅ Configuration backup integrity OK"
else
    echo "❌ Configuration backup missing"
fi

# 4. Check backup size (should be > 100MB for full backup)
BACKUP_SIZE=$(du -sm "$BACKUP_PATH" | cut -f1)
if [ "$BACKUP_SIZE" -gt 100 ]; then
    echo "✅ Backup size reasonable: ${BACKUP_SIZE}MB"
else
    echo "⚠️ Backup size seems small: ${BACKUP_SIZE}MB"
fi

# 5. Test database restore (dry run)
if [ -f "$BACKUP_PATH"/database_*.sql.gz ]; then
    echo "Testing database restore (dry run)..."
    gunzip -c "$BACKUP_PATH"/database_*.sql.gz | head -50 | grep -q "CREATE\|INSERT" && echo "✅ Database restore test OK"
fi

echo "=== Backup Verification Complete ==="
```

## 🔄 Recovery Procedures

### Complete Infrastructure Recovery

**Full System Recovery Script:**
```bash
#!/bin/bash
# full_recovery.sh - Complete system recovery from backup

BACKUP_PATH="$1"
RECOVERY_MODE="${2:-production}"

if [ -z "$BACKUP_PATH" ]; then
    echo "Usage: $0 <backup_path> [production|test]"
    echo "Example: $0 /opt/backups/daily/20250808_120000 production"
    exit 1
fi

echo "=== Starting Full System Recovery ==="
echo "Backup: $BACKUP_PATH"
echo "Mode: $RECOVERY_MODE"

if [ "$RECOVERY_MODE" = "production" ]; then
    read -p "⚠️  This will OVERWRITE the current system. Continue? (yes/NO): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Recovery cancelled"
        exit 1
    fi
fi

# 1. Stop all services
echo "Stopping all services..."
docker-compose down

# 2. Create recovery working directory
RECOVERY_DIR="/tmp/recovery_$(date +%s)"
mkdir -p "$RECOVERY_DIR"
cd "$RECOVERY_DIR"

# 3. Extract backup files
echo "Extracting backup files..."
cp "$BACKUP_PATH"/* .

# 4. Restore database
echo "Restoring PostgreSQL database..."
docker-compose up -d postgres redis neo4j
sleep 30

# Drop and recreate database
docker exec sutazai-postgres psql -U sutazai -c "DROP DATABASE IF EXISTS sutazai;"
docker exec sutazai-postgres psql -U sutazai -c "CREATE DATABASE sutazai;"

# Restore data
gunzip -c database_*.sql.gz | docker exec -i sutazai-postgres psql -U sutazai -d sutazai

# 5. Restore Redis
echo "Restoring Redis data..."
docker stop sutazai-redis
gunzip -c redis_*.rdb.gz > /tmp/dump.rdb
docker cp /tmp/dump.rdb sutazai-redis:/data/dump.rdb
docker start sutazai-redis

# 6. Restore Neo4j
echo "Restoring Neo4j database..."
docker stop sutazai-neo4j
gunzip -c neo4j_*.dump.gz > /tmp/neo4j.dump
docker cp /tmp/neo4j.dump sutazai-neo4j:/tmp/neo4j.dump
docker start sutazai-neo4j
sleep 10
docker exec sutazai-neo4j neo4j-admin load --from=/tmp/neo4j.dump --force

# 7. Restore application configuration
echo "Restoring application configuration..."
tar -xzf config_*.tar.gz -C /
tar -xzf docker_compose_*.tar.gz -C /opt/sutazaiapp/

# 8. Restore AI models
echo "Restoring Ollama models..."
docker-compose up -d ollama
sleep 30
gunzip -c ollama_models_*.tar.gz | docker exec -i sutazai-ollama tar -xzf - -C /

# 9. Restore vector databases
echo "Restoring vector databases..."
docker-compose up -d qdrant chromadb
sleep 30
gunzip -c qdrant_*.tar.gz | docker exec -i sutazai-qdrant tar -xzf - -C /

# 10. Start all services
echo "Starting all services..."
docker-compose up -d

# 11. Wait for initialization
echo "Waiting for services to initialize..."
sleep 60

# 12. Verify recovery
echo "Verifying system recovery..."
./verify_recovery.sh

echo "=== Full System Recovery Complete ==="
```

### Database-Only Recovery

**Database Recovery Script:**
```bash
#!/bin/bash
# database_recovery.sh - Database-only recovery

BACKUP_PATH="$1"
DATABASE="${2:-sutazai}"

echo "=== Database Recovery Started ==="

# 1. Create database backup before recovery
echo "Creating pre-recovery backup..."
docker exec sutazai-postgres pg_dump -U sutazai "$DATABASE" > "/tmp/pre_recovery_$(date +%s).sql"

# 2. Stop application services (keep database running)
echo "Stopping application services..."
docker stop sutazai-backend sutazai-frontend

# 3. Restore database
echo "Restoring database from $BACKUP_PATH..."
gunzip -c "$BACKUP_PATH"/database_*.sql.gz | docker exec -i sutazai-postgres psql -U sutazai -d "$DATABASE"

# 4. Verify database integrity
echo "Verifying database integrity..."
TABLES=$(docker exec sutazai-postgres psql -U sutazai -d "$DATABASE" -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';")
echo "Restored $TABLES tables"

# 5. Restart application services
echo "Restarting application services..."
docker start sutazai-backend sutazai-frontend

# 6. Test connectivity
sleep 30
curl -f http://127.0.0.1:10010/health && echo "✅ Application connectivity OK"

echo "=== Database Recovery Complete ==="
```

### Application-Only Recovery

**Application Recovery Script:**
```bash
#!/bin/bash
# application_recovery.sh - Application-only recovery

BACKUP_PATH="$1"

echo "=== Application Recovery Started ==="

# 1. Stop application containers
echo "Stopping application containers..."
docker stop sutazai-backend sutazai-frontend

# 2. Restore configuration
echo "Restoring configuration..."
tar -xzf "$BACKUP_PATH"/config_*.tar.gz -C /
tar -xzf "$BACKUP_PATH"/docker_compose_*.tar.gz -C /opt/sutazaiapp/

# 3. Rebuild and restart containers
echo "Rebuilding application containers..."
docker-compose build backend frontend
docker-compose up -d backend frontend

# 4. Wait for startup
sleep 60

# 5. Verify application
curl -f http://127.0.0.1:10010/health && echo "✅ Application recovery successful"

echo "=== Application Recovery Complete ==="
```

## 🏗️ Infrastructure Recovery

### Docker Environment Recovery

```bash
#!/bin/bash
# docker_recovery.sh - Docker environment recovery

echo "=== Docker Environment Recovery ==="

# 1. Stop all containers
docker stop $(docker ps -aq) 2>/dev/null
docker rm $(docker ps -aq) 2>/dev/null

# 2. Remove all volumes (WARNING: DATA LOSS)
read -p "Remove all Docker volumes? This will delete all data! (yes/NO): " confirm
if [ "$confirm" = "yes" ]; then
    docker volume prune -f
fi

# 3. Remove networks
docker network prune -f

# 4. Recreate network
docker network create sutazai-network

# 5. Restore from backup and start services
cd /opt/sutazaiapp
docker-compose up -d

echo "=== Docker Environment Recovery Complete ==="
```

### Network Recovery

```bash
#!/bin/bash
# network_recovery.sh - Network configuration recovery

echo "=== Network Recovery Started ==="

# 1. Check network connectivity
ping -c 3 8.8.8.8 || echo "⚠️ External connectivity issues"

# 2. Recreate Docker networks
docker network rm sutazai-network 2>/dev/null
docker network create sutazai-network \
    --driver bridge \
    --subnet=172.20.0.0/16 \
    --ip-range=172.20.240.0/20

# 3. Restart services with network
docker-compose down
docker-compose up -d

# 4. Verify internal connectivity
docker exec sutazai-backend ping -c 2 sutazai-postgres
docker exec sutazai-backend ping -c 2 sutazai-redis

echo "=== Network Recovery Complete ==="
```

## 📊 Data Recovery

### Point-in-Time Recovery

```bash
#!/bin/bash
# point_in_time_recovery.sh - Restore to specific point in time

TARGET_TIME="$1"  # Format: YYYY-MM-DD HH:MM:SS
BACKUP_BASE="$2"

if [ -z "$TARGET_TIME" ]; then
    echo "Usage: $0 'YYYY-MM-DD HH:MM:SS' [backup_base_path]"
    exit 1
fi

echo "=== Point-in-Time Recovery to $TARGET_TIME ==="

# 1. Find appropriate full backup (before target time)
FULL_BACKUP=$(find "${BACKUP_BASE:-/opt/backups}/daily" -name "20*" -type d | \
    awk -v target="$(date -d "$TARGET_TIME" +%s)" \
    '{cmd="date -d " substr($0,length($0)-14,8) " +%s"; cmd | getline ts; if(ts <= target) print $0, ts}' | \
    sort -k2 -nr | head -1 | cut -d' ' -f1)

if [ -z "$FULL_BACKUP" ]; then
    echo "❌ No suitable full backup found before $TARGET_TIME"
    exit 1
fi

echo "Using full backup: $FULL_BACKUP"

# 2. Find incremental backups between full backup and target time
INCREMENTAL_BACKUPS=$(find "${BACKUP_BASE:-/opt/backups}/incremental" -name "20*" -type d | \
    awk -v full="$(basename "$FULL_BACKUP")" -v target="$(date -d "$TARGET_TIME" +%s)" \
    '{cmd="date -d " substr($0,length($0)-14,8) " +%s"; cmd | getline ts; 
     cmd="date -d " substr(full,1,8) " +%s"; cmd | getline full_ts;
     if(ts > full_ts && ts <= target) print $0}' | sort)

# 3. Restore full backup first
./full_recovery.sh "$FULL_BACKUP" test

# 4. Apply incremental backups in order
for inc_backup in $INCREMENTAL_BACKUPS; do
    echo "Applying incremental backup: $inc_backup"
    ./apply_incremental.sh "$inc_backup"
done

echo "=== Point-in-Time Recovery Complete ==="
```

### Selective Data Recovery

```bash
#!/bin/bash
# selective_recovery.sh - Recover specific data components

COMPONENT="$1"
BACKUP_PATH="$2"

case "$COMPONENT" in
    "models")
        echo "Recovering AI models..."
        docker stop sutazai-ollama
        gunzip -c "$BACKUP_PATH"/ollama_models_*.tar.gz | docker exec -i sutazai-ollama tar -xzf - -C /
        docker start sutazai-ollama
        ;;
    "vectors")
        echo "Recovering vector databases..."
        docker stop sutazai-qdrant sutazai-chromadb
        gunzip -c "$BACKUP_PATH"/qdrant_*.tar.gz | docker exec -i sutazai-qdrant tar -xzf - -C /
        docker start sutazai-qdrant sutazai-chromadb
        ;;
    "config")
        echo "Recovering configuration..."
        tar -xzf "$BACKUP_PATH"/config_*.tar.gz -C /
        docker-compose restart
        ;;
    "metrics")
        echo "Recovering metrics data..."
        docker stop sutazai-prometheus
        gunzip -c "$BACKUP_PATH"/prometheus_*.tar.gz | docker exec -i sutazai-prometheus tar -xzf - -C /
        docker start sutazai-prometheus
        ;;
    *)
        echo "Unknown component: $COMPONENT"
        echo "Available components: models, vectors, config, metrics"
        exit 1
        ;;
esac

echo "Selective recovery of $COMPONENT complete"
```

## 🚀 Application Recovery

### Service-by-Service Recovery

```bash
#!/bin/bash
# service_recovery.sh - Recover individual services

SERVICE="$1"
BACKUP_PATH="$2"

echo "=== Recovering Service: $SERVICE ==="

case "$SERVICE" in
    "postgres")
        docker stop sutazai-postgres
        # Restore postgres data
        gunzip -c "$BACKUP_PATH"/database_*.sql.gz | docker exec -i sutazai-postgres psql -U sutazai -d sutazai
        docker start sutazai-postgres
        ;;
    "redis")
        docker stop sutazai-redis
        gunzip -c "$BACKUP_PATH"/redis_*.rdb.gz > /tmp/dump.rdb
        docker cp /tmp/dump.rdb sutazai-redis:/data/dump.rdb
        docker start sutazai-redis
        ;;
    "neo4j")
        docker stop sutazai-neo4j
        gunzip -c "$BACKUP_PATH"/neo4j_*.dump.gz > /tmp/neo4j.dump
        docker cp /tmp/neo4j.dump sutazai-neo4j:/tmp/neo4j.dump
        docker start sutazai-neo4j
        docker exec sutazai-neo4j neo4j-admin load --from=/tmp/neo4j.dump --force
        ;;
    "ollama")
        docker stop sutazai-ollama
        gunzip -c "$BACKUP_PATH"/ollama_models_*.tar.gz | docker exec -i sutazai-ollama tar -xzf - -C /
        docker start sutazai-ollama
        ;;
    "backend")
        docker stop sutazai-backend
        tar -xzf "$BACKUP_PATH"/config_*.tar.gz -C /
        docker-compose build backend
        docker start sutazai-backend
        ;;
    "frontend")
        docker stop sutazai-frontend  
        tar -xzf "$BACKUP_PATH"/config_*.tar.gz -C /
        docker-compose build frontend
        docker start sutazai-frontend
        ;;
    *)
        echo "Unknown service: $SERVICE"
        echo "Available services: postgres, redis, neo4j, ollama, backend, frontend"
        exit 1
        ;;
esac

sleep 30
echo "Service $SERVICE recovery complete"
```

## ✅ Testing & Validation

### Recovery Testing Script

```bash
#!/bin/bash
# test_recovery.sh - Comprehensive recovery testing

BACKUP_PATH="$1"
TEST_TYPE="${2:-smoke}"

echo "=== Recovery Testing Started ==="
echo "Backup: $BACKUP_PATH"
echo "Test Type: $TEST_TYPE"

# 1. Basic connectivity tests
echo "Testing basic connectivity..."
curl -f http://127.0.0.1:10010/health || echo "❌ Health check failed"
curl -f http://127.0.0.1:10011 || echo "❌ Frontend unavailable"

# 2. Database connectivity
echo "Testing database connectivity..."
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT 1;" || echo "❌ Database connection failed"

# 3. Redis connectivity
echo "Testing Redis connectivity..."
docker exec sutazai-redis redis-cli ping || echo "❌ Redis connection failed"

# 4. AI services
echo "Testing AI services..."
curl -s -X POST http://127.0.0.1:10010/simple-chat \
    -H "Content-Type: application/json" \
    -d '{"message": "test"}' | jq -r '.response' || echo "❌ AI service failed"

if [ "$TEST_TYPE" = "full" ]; then
    # 5. Data integrity tests
    echo "Testing data integrity..."
    
    # Check table counts
    TABLES=$(docker exec sutazai-postgres psql -U sutazai -d sutazai -t -c "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';")
    echo "Database tables: $TABLES"
    
    # Test vector database
    curl -s http://127.0.0.1:10101/collections || echo "⚠️ Qdrant not responding"
    
    # Test models
    curl -s http://127.0.0.1:10104/api/tags || echo "⚠️ Ollama not responding"
    
    # 6. Performance tests
    echo "Running performance tests..."
    START_TIME=$(date +%s)
    for i in {1..10}; do
        curl -s http://127.0.0.1:10010/health >/dev/null
    done
    END_TIME=$(date +%s)
    DURATION=$((END_TIME - START_TIME))
    echo "10 health checks took ${DURATION}s"
    
    # 7. Stress test
    echo "Running light stress test..."
    for i in {1..5}; do
        curl -s -X POST http://127.0.0.1:10010/simple-chat \
            -H "Content-Type: application/json" \
            -d "{\"message\": \"Stress test $i\"}" &
    done
    wait
fi

echo "=== Recovery Testing Complete ==="
```

### Automated Recovery Validation

```bash
#!/bin/bash
# validate_recovery.sh - Automated recovery validation

echo "=== Recovery Validation Started ==="

VALIDATION_RESULTS="/tmp/validation_$(date +%s).json"

cat > "$VALIDATION_RESULTS" << EOF
{
  "validation_date": "$(date -Iseconds)",
  "tests": []
}
EOF

# Function to add test result
add_test_result() {
    local test_name="$1"
    local status="$2"
    local details="$3"
    
    jq --arg name "$test_name" --arg status "$status" --arg details "$details" \
        '.tests += [{"name": $name, "status": $status, "details": $details}]' \
        "$VALIDATION_RESULTS" > /tmp/validation_temp.json
    mv /tmp/validation_temp.json "$VALIDATION_RESULTS"
}

# Test 1: System health
if curl -f -s http://127.0.0.1:10010/health >/dev/null; then
    add_test_result "system_health" "PASS" "Health endpoint responding"
else
    add_test_result "system_health" "FAIL" "Health endpoint not responding"
fi

# Test 2: Database connectivity
if docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT 1;" >/dev/null 2>&1; then
    add_test_result "database_connectivity" "PASS" "Database accessible"
else
    add_test_result "database_connectivity" "FAIL" "Database connection failed"
fi

# Test 3: AI functionality
AI_RESPONSE=$(curl -s -X POST http://127.0.0.1:10010/simple-chat \
    -H "Content-Type: application/json" \
    -d '{"message": "validation test"}' | jq -r '.response // empty')

if [ -n "$AI_RESPONSE" ]; then
    add_test_result "ai_functionality" "PASS" "AI responding: ${AI_RESPONSE:0:50}..."
else
    add_test_result "ai_functionality" "FAIL" "AI not responding"
fi

# Test 4: Data integrity
TABLE_COUNT=$(docker exec sutazai-postgres psql -U sutazai -d sutazai -t -c \
    "SELECT count(*) FROM information_schema.tables WHERE table_schema='public';" | tr -d ' ')

if [ "$TABLE_COUNT" -gt 0 ]; then
    add_test_result "data_integrity" "PASS" "Found $TABLE_COUNT database tables"
else
    add_test_result "data_integrity" "FAIL" "No database tables found"
fi

# Generate summary
PASS_COUNT=$(jq '[.tests[] | select(.status == "PASS")] | length' "$VALIDATION_RESULTS")
FAIL_COUNT=$(jq '[.tests[] | select(.status == "FAIL")] | length' "$VALIDATION_RESULTS")
TOTAL_COUNT=$(jq '.tests | length' "$VALIDATION_RESULTS")

jq --arg pass "$PASS_COUNT" --arg fail "$FAIL_COUNT" --arg total "$TOTAL_COUNT" \
    '. + {"summary": {"total": ($total|tonumber), "passed": ($pass|tonumber), "failed": ($fail|tonumber)}}' \
    "$VALIDATION_RESULTS" > /tmp/validation_final.json
mv /tmp/validation_final.json "$VALIDATION_RESULTS"

echo "=== Validation Results ==="
jq '.' "$VALIDATION_RESULTS"

if [ "$FAIL_COUNT" -eq 0 ]; then
    echo "✅ All validation tests passed"
    exit 0
else
    echo "❌ $FAIL_COUNT validation tests failed"
    exit 1
fi
```

## 📞 Communication Plan

### Stakeholder Notification Script

```bash
#!/bin/bash
# notify_stakeholders.sh - Disaster recovery notifications

INCIDENT_TYPE="$1"
STATUS="$2"
DETAILS="$3"

NOTIFICATION_LOG="/var/log/sutazai/notifications.log"

# Notification function
send_notification() {
    local recipient="$1"
    local subject="$2"
    local message="$3"
    local channel="$4"
    
    case "$channel" in
        "email")
            echo "EMAIL: $recipient - $subject" >> "$NOTIFICATION_LOG"
            # In production: integrate with email service
            ;;
        "slack")
            echo "SLACK: $recipient - $message" >> "$NOTIFICATION_LOG"
            # In production: integrate with Slack webhook
            ;;
        "sms")
            echo "SMS: $recipient - $message" >> "$NOTIFICATION_LOG"
            # In production: integrate with SMS service
            ;;
    esac
}

# Stakeholder groups
EXEC_TEAM="exec-team@company.com"
DEV_TEAM="dev-team@company.com"
OPS_TEAM="ops-team@company.com"
CUSTOMER_SUPPORT="support@company.com"

case "$INCIDENT_TYPE" in
    "disaster_declared")
        send_notification "$EXEC_TEAM" "URGENT: System Disaster Declared" "Disaster recovery initiated. Status: $STATUS. Details: $DETAILS" "email"
        send_notification "#incidents" "🚨 DISASTER DECLARED - Jarvis System" "email"
        send_notification "+1-555-ON-CALL" "Jarvis disaster declared: $STATUS" "sms"
        ;;
    "recovery_started")
        send_notification "$OPS_TEAM" "Recovery Started" "Disaster recovery in progress. ETA: $DETAILS" "email"
        send_notification "#incidents" "🔄 Recovery started - ETA: $DETAILS" "slack"
        ;;
    "recovery_complete")
        send_notification "all@company.com" "System Recovered" "Jarvis system recovery complete. Status: $STATUS" "email"
        send_notification "#general" "✅ Jarvis system recovered and operational" "slack"
        ;;
    "recovery_failed")
        send_notification "$EXEC_TEAM" "URGENT: Recovery Failed" "Recovery attempt failed. Manual intervention required. Details: $DETAILS" "email"
        send_notification "+1-555-ON-CALL" "Recovery failed: $DETAILS" "sms"
        ;;
esac

echo "Notifications sent for $INCIDENT_TYPE"
```

### Status Page Update

```bash
#!/bin/bash
# update_status_page.sh - Update external status page

STATUS="$1"
MESSAGE="$2"

# Status page API integration
curl -X POST "https://api.statuspage.io/v1/pages/PAGE_ID/incidents" \
    -H "Authorization: OAuth TOKEN" \
    -H "Content-Type: application/json" \
    -d "{
        \"incident\": {
            \"name\": \"Jarvis System Recovery\",
            \"status\": \"$STATUS\",
            \"message\": \"$MESSAGE\",
            \"component_ids\": [\"component-id-jarvis\"]
        }
    }" || echo "Status page update failed"

echo "Status page updated: $STATUS - $MESSAGE"
```

## 🔄 Post-Recovery

### Post-Recovery Checklist

```bash
#!/bin/bash
# post_recovery_checklist.sh - Post-recovery verification

echo "=== Post-Recovery Checklist ==="

CHECKLIST_ITEMS=(
    "Verify all services are running:docker ps --format 'table {{.Names}}\t{{.Status}}'"
    "Test API endpoints:curl -f http://127.0.0.1:10010/health"
    "Check database integrity:docker exec sutazai-postgres psql -U sutazai -d sutazai -c 'SELECT COUNT(*) FROM information_schema.tables;'"
    "Verify AI functionality:curl -X POST http://127.0.0.1:10010/simple-chat -H 'Content-Type: application/json' -d '{\"message\":\"test\"}'"
    "Check monitoring systems:curl -f http://127.0.0.1:10200/metrics"
    "Verify backup systems:./daily_backup.sh --test"
    "Update documentation:echo 'Recovery completed on $(date)' >> /opt/sutazaiapp/docs/recovery_log.md"
    "Schedule post-incident review:echo 'Schedule PIR for $(date -d '+1 week')'"
)

for item in "${CHECKLIST_ITEMS[@]}"; do
    description=$(echo "$item" | cut -d: -f1)
    command=$(echo "$item" | cut -d: -f2-)
    
    echo "Checking: $description"
    if eval "$command" >/dev/null 2>&1; then
        echo "✅ $description - OK"
    else
        echo "❌ $description - FAILED"
    fi
done

echo "=== Post-Recovery Checklist Complete ==="
```

### Incident Documentation

```bash
#!/bin/bash
# document_incident.sh - Create incident report

INCIDENT_ID="$1"
RECOVERY_TIME="$2"
ROOT_CAUSE="$3"

INCIDENT_REPORT="/opt/sutazaiapp/docs/incidents/incident_${INCIDENT_ID}.md"

mkdir -p "/opt/sutazaiapp/docs/incidents"

cat > "$INCIDENT_REPORT" << EOF
# Incident Report: $INCIDENT_ID

**Date:** $(date -Iseconds)  
**Duration:** $RECOVERY_TIME  
**Severity:** Major  
**Status:** Resolved  

## Summary

System disaster recovery executed successfully.

## Timeline

- **Incident Start:** [Time disaster detected]
- **Recovery Start:** [Time recovery began]
- **Recovery Complete:** $(date -Iseconds)
- **Total Duration:** $RECOVERY_TIME

## Root Cause

$ROOT_CAUSE

## Recovery Actions Taken

1. Disaster declared and stakeholders notified
2. Latest backup identified and validated
3. Full system recovery executed
4. Services restored and validated
5. System performance verified

## Lessons Learned

- [Document lessons learned]
- [Identify improvement areas]

## Action Items

- [ ] Review and update backup procedures
- [ ] Enhance monitoring and alerting
- [ ] Update disaster recovery documentation
- [ ] Schedule disaster recovery drill

## Sign-off

**Incident Commander:** [Name]  
**Technical Lead:** [Name]  
**Recovery Completed:** $(date -Iseconds)  
EOF

echo "Incident documented: $INCIDENT_REPORT"
```

## 📋 Recovery Runbooks Summary

### Quick Recovery Commands

**Emergency Recovery (Complete System):**
```bash
# Stop system
docker-compose down

# Identify latest backup
LATEST_BACKUP=$(find /opt/backups/daily -name "20*" -type d | sort | tail -1)

# Execute full recovery
./full_recovery.sh "$LATEST_BACKUP" production

# Verify recovery
./test_recovery.sh "$LATEST_BACKUP" full
```

**Database-Only Emergency Recovery:**
```bash
# Quick database restore
LATEST_BACKUP=$(find /opt/backups/daily -name "20*" -type d | sort | tail -1)
./database_recovery.sh "$LATEST_BACKUP"
```

**Application-Only Recovery:**
```bash
# Quick application restart
LATEST_BACKUP=$(find /opt/backups/daily -name "20*" -type d | sort | tail -1)
./application_recovery.sh "$LATEST_BACKUP"
```

### Recovery Time Estimates

| Recovery Type | Estimated Time | Complexity |
|---------------|----------------|------------|
| **Network/Connectivity** | 5-15 minutes | Low |
| **Application Restart** | 15-30 minutes | Low |
| **Database Recovery** | 30-60 minutes | Medium |
| **Partial System Recovery** | 1-2 hours | Medium |
| **Complete Infrastructure** | 2-4 hours | High |
| **Point-in-Time Recovery** | 3-6 hours | High |

---

## 📞 Emergency Contacts

**Primary Recovery Team:**
- **Incident Commander:** [Name] - [Phone] - [Email]
- **Technical Lead:** [Name] - [Phone] - [Email]
- **Database Administrator:** [Name] - [Phone] - [Email]
- **Infrastructure Lead:** [Name] - [Phone] - [Email]

**Escalation:**
- **CTO:** [Name] - [Phone] - [Email]
- **CEO:** [Name] - [Phone] - [Email]

**External Support:**
- **Cloud Provider Support:** [Number]
- **Vendor Support:** [Number]

---

*This disaster recovery plan covers comprehensive procedures for recovering the Perfect Jarvis system from various disaster scenarios. All procedures are based on the actual system architecture and have been designed for the current Docker Compose deployment.*