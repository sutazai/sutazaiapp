# SutazAI Operational Runbook

## Daily Operations Checklist

### Morning Health Check (Start of Business)

```bash
# 1. System Status Overview
./scripts/deploy_complete_sutazai_agi_system.sh status

# 2. Check All Service Health
curl -s http://localhost:8000/health | jq '.'
curl -s http://localhost:8501 > /dev/null && echo "Frontend: OK" || echo "Frontend: FAIL"

# 3. Resource Usage Check
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.NetIO}}"

# 4. Database Health
docker exec sutazai-postgres pg_isready -U sutazai
docker exec sutazai-redis redis-cli ping
docker exec sutazai-neo4j cypher-shell "MATCH (n) RETURN count(n) LIMIT 1;"

# 5. Model Availability
curl -s http://localhost:11434/api/tags | jq '.models | length'

# 6. Log Review (Check for errors in last 24 hours)
docker-compose logs --since 24h | grep -i error | wc -l
```

### Evening Maintenance (End of Business)

```bash
# 1. Backup Critical Data
./scripts/backup_system.sh

# 2. Log Rotation and Cleanup
docker system prune -f
./scripts/cleanup_logs.sh

# 3. Update System Metrics
./scripts/generate_daily_report.sh

# 4. Check Disk Space
df -h | grep -E '(8[0-9]|9[0-9])%' || echo "Disk space OK"

# 5. Security Scan (if enabled)
./scripts/run_security_scan.sh
```

## Troubleshooting Procedures

### 1. Service Startup Issues

#### Problem: Container Fails to Start

```bash
# Diagnostic Steps
echo "=== Troubleshooting Container Startup ==="

# Step 1: Check container status
docker-compose ps

# Step 2: Check specific container logs
SERVICE_NAME="backend"  # Replace with failing service
echo "Checking logs for $SERVICE_NAME:"
docker-compose logs --tail=50 $SERVICE_NAME

# Step 3: Check resource availability
echo "System Resources:"
free -h
df -h
docker system df

# Step 4: Check port conflicts
echo "Checking port conflicts:"
netstat -tulpn | grep -E ':(8000|8501|5432|6379|11434)'

# Step 5: Verify dependencies
echo "Checking dependencies:"
docker-compose config --services | while read service; do
    echo "$service: $(docker-compose ps $service | tail -n +3 | awk '{print $4}')"
done

# Step 6: Restart with clean state
echo "Attempting clean restart:"
docker-compose down $SERVICE_NAME
docker-compose up -d $SERVICE_NAME

# Step 7: Check health after restart
sleep 30
docker-compose ps $SERVICE_NAME
```

#### Problem: Database Connection Issues

```bash
# PostgreSQL Connection Troubleshooting
echo "=== PostgreSQL Connection Diagnostics ==="

# Check if PostgreSQL is running
docker-compose ps postgres

# Check PostgreSQL logs
docker-compose logs --tail=20 postgres

# Test connection from host
PGPASSWORD=sutazai_password psql -h localhost -U sutazai -d sutazai -c "SELECT version();"

# Test connection from backend container
docker exec sutazai-backend python -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='postgres',
        database='sutazai',
        user='sutazai',
        password='sutazai_password'
    )
    print('PostgreSQL connection: OK')
    conn.close()
except Exception as e:
    print(f'PostgreSQL connection failed: {e}')
"

# Check connection limits
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
SELECT count(*) as current_connections, 
       setting as max_connections 
FROM pg_stat_activity, pg_settings 
WHERE name = 'max_connections';
"

# Redis Connection Troubleshooting
echo "=== Redis Connection Diagnostics ==="
docker exec sutazai-redis redis-cli ping
docker exec sutazai-backend python -c "
import redis
try:
    r = redis.Redis(host='redis', port=6379, password='redis_password')
    r.ping()
    print('Redis connection: OK')
except Exception as e:
    print(f'Redis connection failed: {e}')
"
```

### 2. Performance Issues

#### Problem: High CPU Usage

```bash
# CPU Usage Diagnostics
echo "=== CPU Usage Analysis ==="

# Identify top CPU consumers
echo "Top CPU consuming containers:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}" | sort -k2 -nr | head -10

# Check system load
echo "System Load Average:"
uptime

# Identify problematic processes
echo "Top processes by CPU:"
ps aux --sort=-%cpu | head -10

# Ollama specific diagnostics
echo "Ollama Model Status:"
curl -s http://localhost:11434/api/tags | jq '.models[] | {name: .name, size: .size}'

# CPU optimization script
cat > /tmp/cpu_optimize.sh << 'EOF'
#!/bin/bash
# Temporary CPU optimization
echo "Reducing Ollama parallelism..."
docker exec sutazai-ollama bash -c 'echo "OLLAMA_NUM_PARALLEL=1" >> /etc/environment'
docker-compose restart ollama

echo "Limiting AI agent concurrency..."
# Add logic to pause non-essential agents
docker-compose pause tabbyml semgrep pentestgpt 2>/dev/null || true
EOF

chmod +x /tmp/cpu_optimize.sh
/tmp/cpu_optimize.sh
```

#### Problem: High Memory Usage

```bash
# Memory Usage Diagnostics
echo "=== Memory Usage Analysis ==="

# System memory overview
echo "System Memory:"
free -h

# Container memory usage
echo "Container Memory Usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.MemUsage}}\t{{.MemPerc}}" | sort -k3 -nr

# Check for memory leaks
echo "Memory trends (if available):"
ps aux --sort=-%mem | head -10

# OOM killer logs
echo "Recent OOM events:"
dmesg | grep -i "killed process" | tail -5

# PostgreSQL memory analysis
echo "PostgreSQL Memory Settings:"
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
SELECT name, setting, unit 
FROM pg_settings 
WHERE name IN ('shared_buffers', 'work_mem', 'maintenance_work_mem', 'effective_cache_size');
"

# Memory optimization script
cat > /tmp/memory_optimize.sh << 'EOF'
#!/bin/bash
echo "Optimizing memory usage..."

# Reduce Ollama model cache
docker exec sutazai-ollama bash -c 'echo "OLLAMA_KEEP_ALIVE=30s" >> /etc/environment'
docker exec sutazai-ollama bash -c 'echo "OLLAMA_MAX_LOADED_MODELS=1" >> /etc/environment'

# Clear system caches
sync && echo 3 > /proc/sys/vm/drop_caches

# Restart memory-intensive services
docker-compose restart ollama backend

echo "Memory optimization completed"
EOF

chmod +x /tmp/memory_optimize.sh
/tmp/memory_optimize.sh
```

### 3. Network Connectivity Issues

#### Problem: API Endpoints Not Responding

```bash
# Network Connectivity Diagnostics
echo "=== Network Connectivity Analysis ==="

# Check all exposed ports
echo "Port Accessibility Test:"
ports=(8000 8501 5432 6379 7474 7687 11434 9090 3000)
for port in "${ports[@]}"; do
    if nc -z localhost $port 2>/dev/null; then
        echo "Port $port: OPEN"
    else
        echo "Port $port: CLOSED"
    fi
done

# Check Docker network
echo "Docker Network Status:"
docker network ls
docker network inspect sutazai-network | jq '.[0].Containers'

# Test inter-container connectivity
echo "Inter-container Connectivity:"
docker exec sutazai-backend curl -s -o /dev/null -w "%{http_code}" http://postgres:5432 || echo "Backend->Postgres: FAIL"
docker exec sutazai-backend curl -s -o /dev/null -w "%{http_code}" http://redis:6379 || echo "Backend->Redis: FAIL"
docker exec sutazai-backend curl -s -o /dev/null -w "%{http_code}" http://ollama:11434 || echo "Backend->Ollama: FAIL"

# Check firewall rules
echo "Firewall Status:"
ufw status verbose 2>/dev/null || iptables -L INPUT -n | grep -E ':(8000|8501)'

# Network troubleshooting script
cat > /tmp/network_fix.sh << 'EOF'
#!/bin/bash
echo "Attempting network fixes..."

# Recreate Docker network
docker-compose down
docker network prune -f
docker-compose up -d

# Wait and test
sleep 30
curl -f http://localhost:8000/health || echo "Backend still not responding"
curl -f http://localhost:8501 || echo "Frontend still not responding"

echo "Network fix completed"
EOF

chmod +x /tmp/network_fix.sh
/tmp/network_fix.sh
```

### 4. AI Model Issues

#### Problem: Ollama Model Loading Failures

```bash
# Ollama Diagnostics
echo "=== Ollama Model Diagnostics ==="

# Check Ollama service status
docker-compose ps ollama

# Check available models
echo "Available Models:"
curl -s http://localhost:11434/api/tags | jq '.models[] | {name: .name, size: .size, modified_at: .modified_at}'

# Check Ollama logs
echo "Recent Ollama Logs:"
docker-compose logs --tail=20 ollama

# Test model inference
echo "Testing Model Inference:"
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama3.2:3b",
    "prompt": "Hello, respond with just OK",
    "stream": false
  }' | jq '.response'

# Check disk space for models
echo "Ollama Data Directory Size:"
docker exec sutazai-ollama du -sh /root/.ollama

# Model recovery script
cat > /tmp/model_recovery.sh << 'EOF'
#!/bin/bash
echo "Attempting model recovery..."

# Stop Ollama gracefully
docker-compose stop ollama

# Clear any corrupted model files
docker exec sutazai-ollama find /root/.ollama -name "*.tmp" -delete 2>/dev/null || true

# Restart Ollama
docker-compose start ollama
sleep 30

# Re-pull essential models
echo "Re-downloading essential models..."
docker exec sutazai-ollama ollama pull llama3.2:3b
docker exec sutazai-ollama ollama pull qwen2.5:3b

# Test model availability
curl -s http://localhost:11434/api/tags | jq '.models | length'

echo "Model recovery completed"
EOF

chmod +x /tmp/model_recovery.sh
/tmp/model_recovery.sh
```

## Maintenance Procedures

### 1. Scheduled Maintenance Windows

#### Weekly Maintenance (Low Traffic Hours)

```bash
#!/bin/bash
# Weekly maintenance script
echo "=== Weekly Maintenance - $(date) ==="

# 1. Full system backup
echo "Creating full system backup..."
./scripts/backup_system.sh weekly

# 2. Update Docker images
echo "Updating Docker images..."
docker-compose pull

# 3. Database maintenance
echo "Database maintenance..."
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "VACUUM ANALYZE;"
docker exec sutazai-redis redis-cli BGREWRITEAOF

# 4. Log rotation
echo "Log rotation..."
find /opt/sutazaiapp/logs -name "*.log" -mtime +7 -delete
docker system prune -f

# 5. Security updates
echo "Security updates..."
./scripts/run_security_scan.sh

# 6. Performance analysis
echo "Generating performance report..."
./scripts/generate_performance_report.sh

# 7. Restart services with new images
echo "Rolling restart with updates..."
docker-compose up -d --force-recreate

# 8. Verify all services
echo "Post-maintenance verification..."
sleep 60
./scripts/verify_deployment.sh

echo "Weekly maintenance completed - $(date)"
```

#### Monthly Maintenance (Scheduled Downtime)

```bash
#!/bin/bash
# Monthly maintenance script
echo "=== Monthly Maintenance - $(date) ==="

# 1. Complete system backup
echo "Creating complete system backup..."
./scripts/backup_system.sh monthly

# 2. Database optimization
echo "Database optimization..."
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
  REINDEX DATABASE sutazai;
  VACUUM FULL;
  ANALYZE;
"

# 3. Vector database optimization
echo "Vector database optimization..."
docker exec sutazai-chromadb python -c "
import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
# Optimize collections
for collection in client.list_collections():
    print(f'Optimizing collection: {collection.name}')
"

# 4. Model updates
echo "Checking for model updates..."
docker exec sutazai-ollama bash -c "
  ollama list | tail -n +2 | while read model rest; do
    echo \"Updating model: \$model\"
    ollama pull \$model
  done
"

# 5. Configuration updates
echo "Updating configurations..."
cp docker-compose.yml docker-compose.yml.backup.$(date +%Y%m%d)
# Apply any configuration updates here

# 6. Security audit
echo "Running security audit..."
./scripts/run_comprehensive_security_audit.sh

# 7. Performance testing
echo "Running performance tests..."
./scripts/run_performance_tests.sh

echo "Monthly maintenance completed - $(date)"
```

### 2. Disaster Recovery Procedures

#### Complete System Recovery

```bash
#!/bin/bash
# Disaster recovery script
set -euo pipefail

BACKUP_DATE=${1:-$(date -d "yesterday" +%Y%m%d)}
BACKUP_DIR="/opt/sutazaiapp/backups/$BACKUP_DATE"

echo "=== Disaster Recovery - Restoring from $BACKUP_DATE ==="

if [[ ! -d "$BACKUP_DIR" ]]; then
    echo "ERROR: Backup directory $BACKUP_DIR not found"
    exit 1
fi

# 1. Stop all services
echo "Stopping all services..."
docker-compose down
docker system prune -af

# 2. Restore configuration files
echo "Restoring configuration files..."
cp "$BACKUP_DIR/.env" .
cp "$BACKUP_DIR/docker-compose.yml" .

# 3. Restore database volumes
echo "Restoring database volumes..."
docker volume rm sutazai_postgres_data 2>/dev/null || true
docker volume create sutazai_postgres_data
docker run --rm -v sutazai_postgres_data:/data -v "$BACKUP_DIR":/backup alpine \
  sh -c 'cd /data && tar -xzf /backup/postgres_data.tar.gz --strip-components=1'

# 4. Restore Redis data
echo "Restoring Redis data..."
docker volume rm sutazai_redis_data 2>/dev/null || true
docker volume create sutazai_redis_data
docker run --rm -v sutazai_redis_data:/data -v "$BACKUP_DIR":/backup alpine \
  sh -c 'cd /data && tar -xzf /backup/redis_data.tar.gz --strip-components=1'

# 5. Restore vector databases
echo "Restoring vector databases..."
docker volume rm sutazai_chromadb_data 2>/dev/null || true
docker volume create sutazai_chromadb_data
docker run --rm -v sutazai_chromadb_data:/data -v "$BACKUP_DIR":/backup alpine \
  sh -c 'cd /data && tar -xzf /backup/chromadb_data.tar.gz --strip-components=1'

# 6. Restore Ollama models
echo "Restoring Ollama models..."
docker volume rm sutazai_ollama_data 2>/dev/null || true
docker volume create sutazai_ollama_data
docker run --rm -v sutazai_ollama_data:/data -v "$BACKUP_DIR":/backup alpine \
  sh -c 'cd /data && tar -xzf /backup/ollama_data.tar.gz --strip-components=1'

# 7. Start core services
echo "Starting core services..."
docker-compose up -d postgres redis neo4j

# 8. Wait for databases to be ready
echo "Waiting for databases to initialize..."
sleep 60

# 9. Start remaining services
echo "Starting all services..."
docker-compose up -d

# 10. Verify recovery
echo "Verifying system recovery..."
sleep 120
./scripts/verify_deployment.sh

echo "=== Disaster Recovery Completed ==="
echo "System has been restored from backup: $BACKUP_DATE"
echo "Please verify all services are functioning correctly."
```

### 3. Performance Optimization Procedures

#### System Performance Tuning

```bash
#!/bin/bash
# Performance tuning script
echo "=== System Performance Tuning ==="

# 1. Kernel parameter optimization
echo "Optimizing kernel parameters..."
cat > /tmp/sysctl_optimizations.conf << EOF
# Network optimizations
net.core.rmem_max = 16777216
net.core.wmem_max = 16777216
net.ipv4.tcp_rmem = 4096 65536 16777216
net.ipv4.tcp_wmem = 4096 65536 16777216

# Memory optimizations
vm.swappiness = 10
vm.dirty_ratio = 15
vm.dirty_background_ratio = 5

# File system optimizations
fs.file-max = 2097152
EOF

sudo cp /tmp/sysctl_optimizations.conf /etc/sysctl.d/99-sutazai.conf
sudo sysctl -p /etc/sysctl.d/99-sutazai.conf

# 2. Docker daemon optimization
echo "Optimizing Docker daemon..."
sudo mkdir -p /etc/docker
cat > /tmp/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "3"
  },
  "storage-driver": "overlay2",
  "storage-opts": [
    "overlay2.override_kernel_check=true"
  ],
  "default-ulimits": {
    "nofile": {
      "Hard": 65536,
      "Name": "nofile",
      "Soft": 65536
    }
  }
}
EOF
sudo cp /tmp/daemon.json /etc/docker/daemon.json
sudo systemctl restart docker

# 3. Database performance tuning
echo "Tuning database performance..."
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
  -- Performance tuning
  ALTER SYSTEM SET shared_buffers = '2GB';
  ALTER SYSTEM SET effective_cache_size = '6GB';
  ALTER SYSTEM SET work_mem = '256MB';
  ALTER SYSTEM SET maintenance_work_mem = '512MB';
  
  -- Checkpoint tuning
  ALTER SYSTEM SET checkpoint_completion_target = 0.7;
  ALTER SYSTEM SET wal_buffers = '64MB';
  
  -- Query planner tuning
  ALTER SYSTEM SET random_page_cost = 1.1;
  ALTER SYSTEM SET effective_io_concurrency = 200;
  
  -- Parallel processing
  ALTER SYSTEM SET max_parallel_workers = 8;
  ALTER SYSTEM SET max_parallel_workers_per_gather = 4;
  
  SELECT pg_reload_conf();
"

# 4. Ollama optimization
echo "Optimizing Ollama performance..."
docker exec sutazai-ollama sh -c '
  echo "OLLAMA_NUM_PARALLEL=2" >> /etc/environment
  echo "OLLAMA_MAX_LOADED_MODELS=2" >> /etc/environment
  echo "OLLAMA_KEEP_ALIVE=5m" >> /etc/environment
'

# 5. Apply optimized Docker Compose configuration
echo "Applying optimized container configurations..."
cat > docker-compose.performance.yml << EOF
version: '3.8'

x-performance-config: &perf-config
  deploy:
    resources:
      reservations:
        memory: 50%

services:
  postgres:
    <<: *perf-config
    environment:
      POSTGRES_SHARED_BUFFERS: 2GB
      POSTGRES_EFFECTIVE_CACHE_SIZE: 6GB
      POSTGRES_WORK_MEM: 256MB

  ollama:
    <<: *perf-config
    environment:
      OLLAMA_NUM_PARALLEL: 2
      OLLAMA_MAX_LOADED_MODELS: 2
      OLLAMA_KEEP_ALIVE: 5m

  backend:
    <<: *perf-config
    environment:
      WORKERS: 4
      WORKER_CONNECTIONS: 1000
      KEEPALIVE: 5

  redis:
    <<: *perf-config
    command: redis-server --maxmemory 2gb --maxmemory-policy allkeys-lru --save 60 1000
EOF

# 6. Restart services with performance optimizations
echo "Restarting services with performance optimizations..."
docker-compose -f docker-compose.yml -f docker-compose.performance.yml up -d

echo "Performance tuning completed!"
echo "Monitor system performance and adjust parameters as needed."
```

## Emergency Response Procedures

### 1. Critical Service Failure

```bash
#!/bin/bash
# Emergency response for critical service failure
SERVICE_NAME=${1:-"backend"}
echo "=== EMERGENCY: $SERVICE_NAME Service Failure ==="

# 1. Immediate assessment
echo "1. Service Status Assessment:"
docker-compose ps $SERVICE_NAME
docker-compose logs --tail=50 $SERVICE_NAME

# 2. Quick restart attempt
echo "2. Attempting quick restart..."
docker-compose restart $SERVICE_NAME
sleep 30

# 3. Check if restart resolved the issue
if docker-compose ps $SERVICE_NAME | grep -q "Up"; then
    echo "✓ Service $SERVICE_NAME restored"
    curl -f http://localhost:8000/health && echo "✓ Health check passed"
    exit 0
fi

# 4. Escalated recovery
echo "3. Quick restart failed. Initiating escalated recovery..."

# Stop and remove container
docker-compose down $SERVICE_NAME
docker-compose up -d --force-recreate $SERVICE_NAME

# 5. If still failing, emergency fallback
sleep 60
if ! docker-compose ps $SERVICE_NAME | grep -q "Up"; then
    echo "4. CRITICAL: Initiating emergency fallback..."
    
    # Start minimal system
    docker-compose down
    docker-compose up -d postgres redis ollama
    sleep 30
    docker-compose up -d backend frontend
    
    echo "Emergency fallback initiated. Manual intervention required."
    echo "Contact: system-admin@company.com"
fi
```

### 2. Data Corruption Recovery

```bash
#!/bin/bash
# Emergency data corruption recovery
echo "=== EMERGENCY: Data Corruption Detected ==="

# 1. Immediate isolation
echo "1. Isolating affected services..."
docker-compose stop backend frontend

# 2. Create emergency backup of current state
echo "2. Creating emergency backup..."
EMERGENCY_BACKUP="/opt/sutazaiapp/emergency_backup_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EMERGENCY_BACKUP"
docker run --rm -v sutazai_postgres_data:/data -v "$EMERGENCY_BACKUP":/backup alpine \
  tar -czf /backup/postgres_corrupted.tar.gz -C /data .

# 3. Database integrity check
echo "3. Checking database integrity..."
docker run --rm -v sutazai_postgres_data:/data postgres:16.3-alpine \
  pg_dump --host=postgres --username=sutazai --dbname=sutazai --verbose

# 4. Attempt automatic repair
echo "4. Attempting automatic repair..."
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
  REINDEX DATABASE sutazai;
  VACUUM FULL;
"

# 5. If repair fails, restore from backup
if [[ $? -ne 0 ]]; then
    echo "5. Automatic repair failed. Restoring from latest backup..."
    LATEST_BACKUP=$(ls -t /opt/sutazaiapp/backups/ | head -1)
    ./scripts/disaster_recovery.sh "$LATEST_BACKUP"
fi

echo "Data corruption recovery completed. Verify system integrity."
```

This operational runbook provides comprehensive procedures for daily operations, troubleshooting, maintenance, and emergency response for the SutazAI system.