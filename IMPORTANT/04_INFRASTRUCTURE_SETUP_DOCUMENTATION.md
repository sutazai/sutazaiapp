# SutazAI Infrastructure Setup Documentation
## Complete Guide for Production-Ready Deployment

**Version:** 1.0  
**Date:** August 5, 2025  
**Status:** IMPLEMENTATION READY  
**Based On:** Research-backed findings and security audit results

---

## CRITICAL PREREQUISITES

### ⚠️ SECURITY FIRST - MUST COMPLETE BEFORE ANY SETUP

```bash
# DO NOT PROCEED WITHOUT COMPLETING THESE STEPS
1. Fix 715 critical security vulnerabilities (see COMPREHENSIVE_CODE_AUDIT_REPORT.md)
2. Remove all hardcoded credentials from source code
3. Implement proper secret management
4. Consolidate 71 docker-compose files into single file
5. Verify no fantasy implementations remain
```

---

## PART 1: SYSTEM REQUIREMENTS

### Hardware Requirements (Research-Validated)

```yaml
Minimum Requirements:
  CPU: 8 cores (x86_64)
  RAM: 16GB
  Storage: 50GB SSD
  Network: 100Mbps

Recommended (Current System):
  CPU: 12 cores
  RAM: 29GB  
  Storage: 100GB SSD
  Network: 1Gbps

Production Optimal:
  CPU: 16+ cores
  RAM: 32GB
  Storage: 200GB NVMe
  Network: 1Gbps
```

### Software Requirements

```yaml
Operating System:
  - Ubuntu 22.04 LTS (recommended)
  - Debian 11+
  - RHEL 8+
  - WSL2 (development only)

Container Runtime:
  - Docker: 24.0+
  - Docker Compose: 2.20+
  - containerd: 1.7+ (alternative)

System Packages:
  - Python: 3.11+
  - Git: 2.34+
  - curl, wget, jq
  - htop, iotop (monitoring)
```

---

## PART 2: INITIAL SYSTEM SETUP

### Step 1: Operating System Preparation

```bash
#!/bin/bash
# setup_os.sh - Run as root or with sudo

# Update system packages
apt-get update && apt-get upgrade -y

# Install required system packages
apt-get install -y \
    curl \
    wget \
    git \
    jq \
    htop \
    iotop \
    vim \
    build-essential \
    python3.11 \
    python3-pip \
    python3-venv \
    ca-certificates \
    gnupg \
    lsb-release

# Configure system limits for containers
cat >> /etc/sysctl.conf << EOF
# Container optimizations
fs.file-max = 2097152
fs.inotify.max_user_watches = 524288
net.core.somaxconn = 65535
net.ipv4.tcp_tw_reuse = 1
net.ipv4.ip_local_port_range = 1024 65535
vm.max_map_count = 262144
vm.swappiness = 10
EOF

# Apply sysctl changes
sysctl -p

# Configure ulimits
cat >> /etc/security/limits.conf << EOF
* soft nofile 65536
* hard nofile 65536
* soft nproc 32768
* hard nproc 32768
EOF

echo "OS preparation complete"
```

### Step 2: Docker Installation

```bash
#!/bin/bash
# install_docker.sh

# Remove old Docker versions
apt-get remove -y docker docker-engine docker.io containerd runc

# Add Docker's official GPG key
mkdir -m 0755 -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
    gpg --dearmor -o /etc/apt/keyrings/docker.gpg

# Set up repository
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  tee /etc/apt/sources.list.d/docker.list > /dev/null

# Install Docker Engine
apt-get update
apt-get install -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin

# Configure Docker daemon for production
cat > /etc/docker/daemon.json << EOF
{
  "log-driver": "json-file",
  "log-opts": {
    "max-size": "100m",
    "max-file": "5"
  },
  "storage-driver": "overlay2",
  "metrics-addr": "0.0.0.0:9323",
  "experimental": false,
  "features": {
    "buildkit": true
  },
  "default-ulimits": {
    "nofile": {
      "Hard": 65536,
      "Soft": 65536
    }
  }
}
EOF

# Start and enable Docker
systemctl restart docker
systemctl enable docker

# Verify installation
docker version
docker compose version
```

### Step 3: Storage Configuration

```bash
#!/bin/bash
# configure_storage.sh

# Create directory structure
mkdir -p /opt/sutazaiapp/{data,logs,backups,models,cache}
mkdir -p /opt/sutazaiapp/data/{postgres,redis,neo4j,vectors}
mkdir -p /opt/sutazaiapp/logs/{containers,applications,system}
mkdir -p /opt/sutazaiapp/models/ollama
mkdir -p /opt/sutazaiapp/cache/{redis,semantic}

# Set permissions
chown -R 1000:1000 /opt/sutazaiapp
chmod -R 755 /opt/sutazaiapp

# Create Docker volumes for persistence
docker volume create sutazai_postgres_data
docker volume create sutazai_redis_data
docker volume create sutazai_neo4j_data
docker volume create sutazai_ollama_models
docker volume create sutazai_vector_data
docker volume create sutazai_logs

echo "Storage configuration complete"
```

---

## PART 3: CORE SERVICES DEPLOYMENT

### Step 1: Create Production Docker Compose

```yaml
# /opt/sutazaiapp/docker-compose.production.yml
version: '3.8'

networks:
  sutazai_network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.28.0.0/16

volumes:
  postgres_data:
    external: true
    name: sutazai_postgres_data
  redis_data:
    external: true
    name: sutazai_redis_data
  neo4j_data:
    external: true
    name: sutazai_neo4j_data
  ollama_models:
    external: true
    name: sutazai_ollama_models

services:
  # Core Database Services
  postgres:
    image: postgres:15-alpine
    container_name: sutazai-postgres
    hostname: postgres
    restart: unless-stopped
    environment:
      POSTGRES_DB: ${POSTGRES_DB:-sutazai}
      POSTGRES_USER: ${POSTGRES_USER:-sutazai}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
      POSTGRES_MAX_CONNECTIONS: 200
      POSTGRES_SHARED_BUFFERS: 256MB
      POSTGRES_EFFECTIVE_CACHE_SIZE: 1GB
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./init-scripts/postgres:/docker-entrypoint-initdb.d
    ports:
      - "10000:5432"
    networks:
      sutazai_network:
        ipv4_address: 172.28.1.1
    deploy:
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
        reservations:
          cpus: '0.5'
          memory: 512M
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-sutazai}"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    container_name: sutazai-redis
    hostname: redis
    restart: unless-stopped
    command: 
      - redis-server
      - --maxmemory 2gb
      - --maxmemory-policy allkeys-lru
      - --appendonly yes
      - --appendfsync everysec
      - --requirepass ${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    ports:
      - "10001:6379"
    networks:
      sutazai_network:
        ipv4_address: 172.28.1.2
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 2G
        reservations:
          cpus: '0.25'
          memory: 256M
    healthcheck:
      test: ["CMD", "redis-cli", "--raw", "incr", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  # LLM Service (Critical for AI functionality)
  ollama:
    image: ollama/ollama:latest
    container_name: sutazai-ollama
    hostname: ollama
    restart: unless-stopped
    environment:
      OLLAMA_HOST: 0.0.0.0
      OLLAMA_ORIGINS: "*"
      OLLAMA_NUM_PARALLEL: 4
      OLLAMA_MAX_LOADED_MODELS: 2
      OLLAMA_KEEP_ALIVE: 5m
      OLLAMA_DEBUG: false
    volumes:
      - ollama_models:/root/.ollama
      - ./scripts/ollama:/scripts
    ports:
      - "11434:11434"
    networks:
      sutazai_network:
        ipv4_address: 172.28.1.10
    deploy:
      resources:
        limits:
          cpus: '4.0'
          memory: 8G
        reservations:
          cpus: '2.0'
          memory: 4G
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:11434/api/tags"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  # Monitoring Stack
  prometheus:
    image: prom/prometheus:latest
    container_name: sutazai-prometheus
    hostname: prometheus
    restart: unless-stopped
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
      - '--storage.tsdb.retention.time=30d'
      - '--web.enable-lifecycle'
    volumes:
      - ./monitoring/prometheus:/etc/prometheus
      - prometheus_data:/prometheus
    ports:
      - "10200:9090"
    networks:
      sutazai_network:
        ipv4_address: 172.28.2.1
    deploy:
      resources:
        limits:
          cpus: '1.0'
          memory: 1G

  grafana:
    image: grafana/grafana:latest
    container_name: sutazai-grafana
    hostname: grafana
    restart: unless-stopped
    environment:
      GF_SECURITY_ADMIN_USER: ${GRAFANA_USER:-admin}
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_PASSWORD}
      GF_INSTALL_PLUGINS: redis-datasource
    volumes:
      - ./monitoring/grafana/dashboards:/etc/grafana/provisioning/dashboards
      - ./monitoring/grafana/datasources:/etc/grafana/provisioning/datasources
      - grafana_data:/var/lib/grafana
    ports:
      - "10201:3000"
    networks:
      sutazai_network:
        ipv4_address: 172.28.2.2
    deploy:
      resources:
        limits:
          cpus: '0.5'
          memory: 512M
```

### Step 2: Environment Configuration

```bash
# /opt/sutazaiapp/.env.production
# NEVER commit this file to version control

# Database Credentials
POSTGRES_DB=sutazai
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=CHANGE_THIS_SECURE_PASSWORD_123!
REDIS_PASSWORD=CHANGE_THIS_REDIS_PASSWORD_456!

# Monitoring
GRAFANA_USER=admin
GRAFANA_PASSWORD=CHANGE_THIS_GRAFANA_PASSWORD_789!

# API Keys (if needed)
OPENAI_API_KEY=
ANTHROPIC_API_KEY=

# System Configuration
LOG_LEVEL=INFO
DEBUG_MODE=false
ENVIRONMENT=production

# Resource Limits
MAX_WORKERS=10
MAX_MEMORY_MB=20000
MAX_CPU_PERCENT=80
```

### Step 3: Initialize Core Services

```bash
#!/bin/bash
# initialize_services.sh

cd /opt/sutazaiapp

# Load environment variables
export $(cat .env.production | grep -v '^#' | xargs)

# Start core infrastructure
docker compose -f docker-compose.production.yml up -d postgres redis

# Wait for databases to be ready
echo "Waiting for databases to initialize..."
sleep 30

# Initialize PostgreSQL schema
docker exec sutazai-postgres psql -U $POSTGRES_USER -d $POSTGRES_DB << EOF
-- Create schemas
CREATE SCHEMA IF NOT EXISTS agents;
CREATE SCHEMA IF NOT EXISTS tasks;
CREATE SCHEMA IF NOT EXISTS monitoring;

-- Create base tables
CREATE TABLE IF NOT EXISTS agents.registry (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) UNIQUE NOT NULL,
    type VARCHAR(100),
    status VARCHAR(50),
    config JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS tasks.queue (
    id SERIAL PRIMARY KEY,
    task_id UUID DEFAULT gen_random_uuid(),
    type VARCHAR(100),
    payload JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    assigned_to VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS monitoring.metrics (
    id SERIAL PRIMARY KEY,
    metric_name VARCHAR(255),
    metric_value NUMERIC,
    tags JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes
CREATE INDEX idx_agents_status ON agents.registry(status);
CREATE INDEX idx_tasks_status ON tasks.queue(status);
CREATE INDEX idx_tasks_assigned ON tasks.queue(assigned_to);
CREATE INDEX idx_metrics_timestamp ON monitoring.metrics(timestamp);
EOF

# Start Ollama service
docker compose -f docker-compose.production.yml up -d ollama

# Wait for Ollama to be ready
echo "Waiting for Ollama to initialize..."
sleep 20

# Pull required models
echo "Pulling AI models (this may take several minutes)..."
docker exec sutazai-ollama ollama pull tinyllama:latest
docker exec sutazai-ollama ollama pull mistral:7b-instruct-q4_K_M

# Start monitoring stack
docker compose -f docker-compose.production.yml up -d prometheus grafana

echo "Core services initialized successfully"
```

---

## PART 4: AGENT FRAMEWORK SETUP

### Step 1: Install CrewAI Framework

```bash
#!/bin/bash
# setup_crewai.sh

# Create Python virtual environment
python3.11 -m venv /opt/sutazaiapp/venv
source /opt/sutazaiapp/venv/bin/activate

# Install CrewAI and dependencies
pip install --upgrade pip
pip install \
    crewai==0.30.0 \
    langchain==0.1.0 \
    langchain-community==0.1.0 \
    redis==5.0.1 \
    psycopg2-binary==2.9.9 \
    fastapi==0.110.0 \
    uvicorn==0.27.0 \
    pydantic==2.5.0 \
    prometheus-client==0.19.0 \
    python-dotenv==1.0.0

# Create requirements file
pip freeze > /opt/sutazaiapp/requirements.production.txt
```

### Step 2: Deploy Master Coordinator

```python
# /opt/sutazaiapp/agents/master_coordinator/app.py
import os
import json
import redis
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from crewai import Agent, Task, Crew
from typing import Dict, List, Optional
from datetime import datetime
import psycopg2
from prometheus_client import Counter, Histogram, generate_latest

# Load environment
from dotenv import load_dotenv
load_dotenv('/opt/sutazaiapp/.env.production')

# Initialize FastAPI
app = FastAPI(title="Master Coordinator", version="1.0.0")

# Metrics
task_counter = Counter('tasks_processed', 'Total tasks processed')
task_duration = Histogram('task_duration_seconds', 'Task processing duration')

# Redis connection
redis_client = redis.Redis(
    host='redis',
    port=6379,
    password=os.getenv('REDIS_PASSWORD'),
    decode_responses=True
)

# PostgreSQL connection
pg_conn = psycopg2.connect(
    host='postgres',
    database=os.getenv('POSTGRES_DB'),
    user=os.getenv('POSTGRES_USER'),
    password=os.getenv('POSTGRES_PASSWORD')
)

class TaskRequest(BaseModel):
    type: str
    description: str
    priority: Optional[int] = 5
    metadata: Optional[Dict] = {}

class MasterCoordinator:
    def __init__(self):
        # Initialize master agent with Ollama
        self.master = Agent(
            role='System Orchestrator',
            goal='Efficiently coordinate and route tasks',
            backstory='Master coordinator for SutazAI system',
            llm='ollama/tinyllama',
            max_iter=3,
            memory=True,
            verbose=False
        )
        
        # Team lead agents
        self.team_leads = self._initialize_team_leads()
        
    def _initialize_team_leads(self) -> List[Agent]:
        """Initialize team lead agents"""
        return [
            Agent(
                role='Development Lead',
                goal='Manage development and code tasks',
                llm='ollama/mistral:7b-instruct-q4_K_M',
                max_iter=3
            ),
            Agent(
                role='Analysis Lead',
                goal='Handle data analysis and reporting',
                llm='ollama/tinyllama',
                max_iter=3
            ),
            Agent(
                role='Operations Lead',
                goal='Manage deployment and infrastructure',
                llm='ollama/tinyllama',
                max_iter=3
            )
        ]
    
    async def process_task(self, task_request: TaskRequest) -> Dict:
        """Process incoming task request"""
        with task_duration.time():
            # Check cache
            cache_key = f"task:{hash(str(task_request.dict()))}"
            cached = redis_client.get(cache_key)
            if cached:
                task_counter.inc()
                return json.loads(cached)
            
            # Route to appropriate team lead
            if task_request.type in ['code', 'review', 'debug']:
                lead = self.team_leads[0]
            elif task_request.type in ['data', 'analysis', 'report']:
                lead = self.team_leads[1]
            else:
                lead = self.team_leads[2]
            
            # Create task and crew
            task = Task(
                description=task_request.description,
                agent=lead
            )
            
            crew = Crew(
                agents=[self.master, lead],
                tasks=[task],
                verbose=False
            )
            
            # Execute task
            result = crew.kickoff()
            
            # Store in cache
            redis_client.setex(
                cache_key,
                3600,
                json.dumps({'result': str(result), 'timestamp': datetime.utcnow().isoformat()})
            )
            
            # Log to database
            with pg_conn.cursor() as cur:
                cur.execute("""
                    INSERT INTO tasks.queue (type, payload, status, completed_at)
                    VALUES (%s, %s, %s, %s)
                """, (task_request.type, json.dumps(task_request.dict()), 'completed', datetime.utcnow()))
                pg_conn.commit()
            
            task_counter.inc()
            return {'result': str(result), 'timestamp': datetime.utcnow().isoformat()}

# Initialize coordinator
coordinator = MasterCoordinator()

@app.post("/task")
async def submit_task(task: TaskRequest):
    """Submit task for processing"""
    try:
        result = await coordinator.process_task(task)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "redis": redis_client.ping(),
        "postgres": pg_conn.status == 1
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10300)
```

### Step 3: Deploy Agent Container

```dockerfile
# /opt/sutazaiapp/agents/master_coordinator/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.production.txt .
RUN pip install --no-cache-dir -r requirements.production.txt

# Copy application
COPY agents/master_coordinator/app.py .

# Run application
CMD ["python", "app.py"]
```

---

## PART 5: NETWORK CONFIGURATION

### Service Mesh Setup

```yaml
# /opt/sutazaiapp/infrastructure/linkerd.yml
apiVersion: v1
kind: Namespace
metadata:
  name: sutazai
  annotations:
    linkerd.io/inject: enabled
---
apiVersion: v1
kind: Service
metadata:
  name: sutazai-mesh
spec:
  ports:
  - port: 80
    targetPort: 10300
  selector:
    app: master-coordinator
```

### Firewall Configuration

```bash
#!/bin/bash
# configure_firewall.sh

# Allow Docker networks
ufw allow from 172.28.0.0/16 to any

# Allow specific ports
ufw allow 22/tcp    # SSH
ufw allow 10000:10999/tcp  # SutazAI services
ufw allow 11434/tcp # Ollama

# Enable firewall
ufw --force enable
```

---

## PART 6: MONITORING & ALERTING

### Prometheus Configuration

```yaml
# /opt/sutazaiapp/monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

alerting:
  alertmanagers:
    - static_configs:
        - targets: ['alertmanager:9093']

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'docker'
    static_configs:
      - targets: ['172.17.0.1:9323']

  - job_name: 'master-coordinator'
    static_configs:
      - targets: ['master-coordinator:10300']
    metrics_path: /metrics

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']
```

### Alert Rules

```yaml
# /opt/sutazaiapp/monitoring/prometheus/alerts.yml
groups:
  - name: sutazai_alerts
    interval: 30s
    rules:
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.85
        for: 5m
        annotations:
          summary: "High memory usage detected"
          
      - alert: HighCPUUsage
        expr: 100 - (avg(rate(node_cpu_seconds_total{mode="idle"}[5m])) * 100) > 80
        for: 5m
        annotations:
          summary: "High CPU usage detected"
          
      - alert: ServiceDown
        expr: up == 0
        for: 1m
        annotations:
          summary: "Service {{ $labels.job }} is down"
          
      - alert: SlowTaskProcessing
        expr: histogram_quantile(0.95, rate(task_duration_seconds_bucket[5m])) > 5
        for: 5m
        annotations:
          summary: "Task processing is slow (P95 > 5s)"
```

---

## PART 7: BACKUP & RECOVERY

### Automated Backup Script

```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/backup.sh

BACKUP_DIR="/opt/sutazaiapp/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker exec sutazai-postgres pg_dumpall -U $POSTGRES_USER | \
    gzip > $BACKUP_DIR/postgres_backup.sql.gz

# Backup Redis
docker exec sutazai-redis redis-cli --rdb /data/dump.rdb BGSAVE
sleep 5
docker cp sutazai-redis:/data/dump.rdb $BACKUP_DIR/redis_backup.rdb

# Backup configurations
cp -r /opt/sutazaiapp/*.yml $BACKUP_DIR/
cp /opt/sutazaiapp/.env.production $BACKUP_DIR/

# Backup Docker volumes
for volume in $(docker volume ls -q | grep sutazai); do
    docker run --rm -v $volume:/data -v $BACKUP_DIR:/backup \
        alpine tar czf /backup/${volume}.tar.gz -C /data .
done

# Cleanup old backups (keep last 7 days)
find /opt/sutazaiapp/backups -type d -mtime +7 -exec rm -rf {} \;

echo "Backup completed: $BACKUP_DIR"
```

### Recovery Procedure

```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/restore.sh

if [ -z "$1" ]; then
    echo "Usage: ./restore.sh BACKUP_DIR"
    exit 1
fi

BACKUP_DIR=$1

# Stop services
docker compose -f docker-compose.production.yml down

# Restore PostgreSQL
gunzip < $BACKUP_DIR/postgres_backup.sql.gz | \
    docker exec -i sutazai-postgres psql -U $POSTGRES_USER

# Restore Redis
docker cp $BACKUP_DIR/redis_backup.rdb sutazai-redis:/data/dump.rdb
docker exec sutazai-redis redis-cli SHUTDOWN SAVE
docker restart sutazai-redis

# Restore configurations
cp $BACKUP_DIR/*.yml /opt/sutazaiapp/
cp $BACKUP_DIR/.env.production /opt/sutazaiapp/

# Restart services
docker compose -f docker-compose.production.yml up -d

echo "Recovery completed from: $BACKUP_DIR"
```

---

## PART 8: PRODUCTION DEPLOYMENT CHECKLIST

### Pre-Deployment

```markdown
## Security Checklist
- [ ] All 715 security vulnerabilities fixed
- [ ] No hardcoded credentials in code
- [ ] Environment variables properly configured
- [ ] Firewall rules applied
- [ ] TLS/SSL certificates installed
- [ ] Authentication middleware enabled

## Infrastructure Checklist
- [ ] Docker and Docker Compose installed
- [ ] System limits configured
- [ ] Storage volumes created
- [ ] Network configuration complete
- [ ] Monitoring stack operational

## Application Checklist
- [ ] Core services running and healthy
- [ ] Ollama models downloaded
- [ ] Master coordinator deployed
- [ ] Database schemas created
- [ ] Redis cache operational
```

### Deployment Verification

```bash
#!/bin/bash
# verify_deployment.sh

echo "=== SutazAI Deployment Verification ==="
echo

# Check Docker services
echo "Checking Docker services..."
docker compose -f docker-compose.production.yml ps

# Check service health
echo -e "\nChecking service health..."
for service in postgres redis ollama prometheus grafana; do
    STATUS=$(docker inspect sutazai-$service --format='{{.State.Health.Status}}' 2>/dev/null || echo "no healthcheck")
    echo "$service: $STATUS"
done

# Test Ollama
echo -e "\nTesting Ollama..."
curl -s -X POST http://localhost:11434/api/generate \
    -d '{"model": "tinyllama", "prompt": "Hello"}' | jq -r .response | head -1

# Test Master Coordinator
echo -e "\nTesting Master Coordinator..."
curl -s http://localhost:10300/health | jq

# Check resource usage
echo -e "\nResource usage:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Check logs for errors
echo -e "\nChecking for errors in logs..."
docker compose -f docker-compose.production.yml logs --tail=100 | grep -i error | wc -l
echo "errors found in last 100 log lines"

echo -e "\n=== Verification Complete ==="
```

---

## PART 9: TROUBLESHOOTING

### Common Issues and Solutions

```yaml
Issue: Ollama not responding
Solution:
  - Check memory allocation: docker stats sutazai-ollama
  - Restart service: docker restart sutazai-ollama
  - Check models: docker exec sutazai-ollama ollama list
  - Pull models again if needed

Issue: High memory usage
Solution:
  - Reduce OLLAMA_NUM_PARALLEL to 2
  - Decrease Redis maxmemory setting
  - Implement container memory limits
  - Add swap space if needed

Issue: Slow task processing
Solution:
  - Check cache hit rate in Redis
  - Monitor Ollama response times
  - Reduce agent max_iter parameter
  - Scale horizontally if needed

Issue: Database connection errors
Solution:
  - Check credentials in .env.production
  - Verify network connectivity
  - Check connection pool settings
  - Review PostgreSQL logs

Issue: Container restart loops
Solution:
  - Check container logs: docker logs sutazai-[service]
  - Verify resource limits aren't too restrictive
  - Check for port conflicts
  - Review health check configurations
```

---

## PART 10: MAINTENANCE PROCEDURES

### Daily Maintenance

```bash
#!/bin/bash
# daily_maintenance.sh

# Check system health
/opt/sutazaiapp/scripts/verify_deployment.sh

# Cleanup old logs
find /opt/sutazaiapp/logs -type f -mtime +7 -delete

# Vacuum PostgreSQL
docker exec sutazai-postgres psql -U $POSTGRES_USER -c "VACUUM ANALYZE;"

# Check disk usage
df -h /opt/sutazaiapp
```

### Weekly Maintenance

```bash
#!/bin/bash
# weekly_maintenance.sh

# Full backup
/opt/sutazaiapp/scripts/backup.sh

# Update container images (if needed)
docker compose -f docker-compose.production.yml pull

# Clean unused Docker resources
docker system prune -f

# Review monitoring metrics
echo "Review Grafana dashboards at http://localhost:10201"
```

---

## CONCLUSION

This infrastructure setup provides a production-ready foundation for the SutazAI system. Key points:

1. **Security First**: All security vulnerabilities must be fixed before deployment
2. **Resource Optimized**: Configured for CPU-only deployment with proper limits
3. **Monitoring Enabled**: Full observability with Prometheus and Grafana
4. **Backup Ready**: Automated backup and recovery procedures
5. **Production Hardened**: Proper error handling and health checks

**Next Steps:**
1. Complete security remediation
2. Follow deployment checklist
3. Verify all services are healthy
4. Begin agent deployment following Phase 2 plan

---

**Document Status:** READY FOR IMPLEMENTATION  
**Prerequisites:** Security remediation must be completed first  
**Support:** Create issues in project repository for help