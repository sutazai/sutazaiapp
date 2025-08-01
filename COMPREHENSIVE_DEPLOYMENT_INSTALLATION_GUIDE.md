# 🚀 SutazAI Comprehensive Deployment & Installation Guide

## Table of Contents
1. [Prerequisites and System Requirements](#prerequisites-and-system-requirements)
2. [Installation Procedures](#installation-procedures)
3. [Configuration Management](#configuration-management)
4. [Deployment Options](#deployment-options)
5. [Post-deployment Steps](#post-deployment-steps)
6. [Troubleshooting Guide](#troubleshooting-guide)
7. [Performance Tuning](#performance-tuning)
8. [Security Configuration](#security-configuration)
9. [Backup and Recovery](#backup-and-recovery)

---

## Prerequisites and System Requirements

### Hardware Requirements

#### Minimum Requirements (TinyLlama Configuration)
- **CPU**: 4 cores (x86_64 or ARM64)
- **RAM**: 8GB
- **Storage**: 20GB free disk space
- **Network**: Broadband internet (for initial setup)

#### Recommended Requirements (Full System)
- **CPU**: 8+ cores (x86_64 recommended)
- **RAM**: 32GB
- **Storage**: 100GB+ SSD
- **GPU**: Optional - NVIDIA GPU with 8GB+ VRAM
- **Network**: 100Mbps+ for model downloads

#### Production Requirements
- **CPU**: 16+ cores
- **RAM**: 64GB+
- **Storage**: 500GB+ NVMe SSD
- **GPU**: NVIDIA A100, H100, or multiple RTX 4090s
- **Network**: 1Gbps+ dedicated bandwidth

### Software Dependencies

#### Core Requirements
```bash
# Operating System
- Ubuntu 20.04+ / CentOS 8+ / RHEL 8+
- macOS 12+ (Intel/Apple Silicon)
- Windows 11 with WSL2

# Container Runtime
- Docker Engine 20.10+
- Docker Compose 2.0+

# System Tools
- curl
- wget
- git
- jq (for JSON processing)
```

#### Development Tools (Optional)
```bash
- Python 3.11+
- Node.js 18+
- kubectl (for Kubernetes)
- helm (for Kubernetes deployments)
- terraform (for infrastructure as code)
```

### Network Requirements

#### Ports Required
```yaml
Core Services:
  - 8000: Backend API
  - 8501: Frontend Interface
  - 5432: PostgreSQL
  - 6379: Redis
  - 11434: Ollama

Vector Databases:
  - 8001: ChromaDB
  - 6333: Qdrant
  - 8002: FAISS

Monitoring:
  - 3000: Grafana
  - 9090: Prometheus
  - 3100: Loki

AI Services:
  - 8090-8115: Various AI agents
  - 5678: N8N Workflow
```

#### Firewall Configuration
```bash
# Allow required ports
sudo ufw allow 8000
sudo ufw allow 8501
sudo ufw allow 3000
sudo ufw allow 9090

# For development only
sudo ufw allow 5432
sudo ufw allow 6379
```

---

## Installation Procedures

### Quick Start Installation

#### Option 1: One-Command Installation
```bash
# Clone repository
git clone https://github.com/yourusername/sutazai.git
cd sutazai

# Start minimal system (TinyLlama)
./scripts/deploy_complete_sutazai_agi_system.sh

# Access system
open http://localhost:8501
```

#### Option 2: Step-by-Step Installation
```bash
# 1. Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh
sudo usermod -aG docker $USER
newgrp docker

# 2. Install Docker Compose
sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
sudo chmod +x /usr/local/bin/docker-compose

# 3. Clone and setup
git clone https://github.com/yourusername/sutazai.git
cd sutazai
chmod +x scripts/*.sh

# 4. Choose deployment type
# Minimal (4GB RAM): 
docker compose -f docker-compose.minimal.yml up -d

# Full system (16GB+ RAM):
docker compose up -d

# TinyLlama only (2GB RAM):
docker compose -f docker-compose.tinyllama.yml up -d
```

### Production Deployment

#### Docker Compose Production
```bash
# 1. Prepare production environment
cp .env.example .env.production
vim .env.production  # Configure production settings

# 2. Generate secure secrets
openssl rand -hex 32 > secrets/jwt_secret.txt
openssl rand -base64 32 > secrets/postgres_password.txt
openssl rand -base64 32 > secrets/redis_password.txt
openssl rand -base64 16 > secrets/grafana_password.txt

# 3. Deploy production stack
docker compose -f deployment/docker-compose.production.yml up -d

# 4. Initialize system
./scripts/deploy-production.sh
```

#### Kubernetes Deployment
```bash
# 1. Prepare cluster
kubectl create namespace sutazai

# 2. Create secrets
kubectl create secret generic sutazai-secrets \
  --from-file=secrets/ \
  -n sutazai

# 3. Deploy with Kustomize
kubectl apply -k deployment/k8s/overlays/prod

# 4. Verify deployment
kubectl get pods -n sutazai
kubectl get services -n sutazai
```

### Development Environment Setup

```bash
# 1. Install development dependencies
pip install -r requirements.txt
npm install  # For frontend development

# 2. Setup development environment
cp .env.example .env
python -m venv venv
source venv/bin/activate

# 3. Start development services
docker compose -f docker-compose.minimal.yml up -d postgres redis ollama

# 4. Run backend in development mode
cd backend
uvicorn app.main:app --reload --port 8000

# 5. Run frontend in development mode
cd frontend
streamlit run app.py --server.port 8501
```

### Agent Deployment

#### Core Agents Only
```bash
# Deploy essential agents
docker compose up -d \
  senior-ai-engineer \
  testing-qa-validator \
  deployment-automation-master \
  infrastructure-devops-manager
```

#### Full Agent Ecosystem
```bash
# Deploy all 40+ agents
./scripts/deploy_all_agents.sh

# Check agent status
docker compose ps | grep -E "(autogpt|crewai|aider|localagi)"
```

---

## Configuration Management

### Environment Variables

#### Core Configuration (.env)
```bash
# System Settings
TZ=UTC
SUTAZAI_ENV=production
LOCAL_IP=localhost

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=secure_password_here
POSTGRES_DB=sutazai
DATABASE_URL=postgresql://sutazai:password@postgres:5432/sutazai

# Redis Configuration
REDIS_PASSWORD=secure_redis_password
REDIS_URL=redis://:password@redis:6379/0

# Security
SECRET_KEY=your-secret-key-here
JWT_SECRET=your-jwt-secret

# AI Models
DEFAULT_MODEL=tinyllama:latest
OLLAMA_BASE_URL=http://ollama:11434
ENABLE_GPU=false

# Monitoring
GRAFANA_PASSWORD=admin_password
PROMETHEUS_RETENTION=30d

# Feature Flags
ENABLE_MONITORING=true
ENABLE_SECURITY_SCAN=true
ENABLE_AUTO_BACKUP=true
```

#### Agent Configuration
```bash
# Agent-specific settings
MAX_CONCURRENT_AGENTS=10
AGENT_TIMEOUT=300
AGENT_MEMORY_LIMIT=2G
AGENT_CPU_LIMIT=1.0

# Model Settings
MAX_TOKENS=4096
TEMPERATURE=0.7
TOP_P=0.9
```

### Configuration Files

#### Docker Compose Override
```yaml
# docker-compose.override.yml
version: '3.8'

services:
  backend-agi:
    environment:
      - LOG_LEVEL=DEBUG
    volumes:
      - ./custom-config:/app/config

  ollama:
    environment:
      - OLLAMA_DEBUG=1
    deploy:
      resources:
        limits:
          memory: 16G
```

#### Nginx Configuration
```nginx
# nginx/nginx.conf
upstream backend {
    server backend-agi:8000;
}

upstream frontend {
    server frontend-agi:8501;
}

server {
    listen 80;
    server_name sutazai.local;

    location /api/ {
        proxy_pass http://backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }

    location / {
        proxy_pass http://frontend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Security Settings

#### Production Security Configuration
```bash
# Enable security features
ENABLE_HTTPS=true
REQUIRE_API_KEY=true
ENABLE_RATE_LIMITING=true
ENABLE_CORS_WHITELIST=true

# CORS Configuration
BACKEND_CORS_ORIGINS=["https://yourdomain.com", "https://api.yourdomain.com"]

# API Security
API_KEY_HEADER=X-API-Key
RATE_LIMIT_PER_MINUTE=60
MAX_REQUEST_SIZE=10MB
```

---

## Deployment Options

### 1. Single-Server Deployment (Recommended for Development)

```bash
# All services on one machine
docker compose up -d

# Resource allocation
total_memory: 16GB
total_cpu: 8 cores
estimated_load: Medium
```

### 2. Distributed Deployment

#### Multi-Server Setup
```bash
# Server 1: Core Services (8GB RAM)
docker compose up -d postgres redis neo4j

# Server 2: AI Services (16GB RAM)  
docker compose up -d ollama backend-agi

# Server 3: Agent Services (32GB RAM)
docker compose up -d autogpt crewai aider localagi

# Server 4: Monitoring (4GB RAM)
docker compose up -d prometheus grafana loki
```

### 3. Kubernetes Deployment

#### Cluster Configuration
```yaml
# k8s/namespace.yml
apiVersion: v1
kind: Namespace
metadata:
  name: sutazai
  labels:
    name: sutazai
```

```yaml
# k8s/backend-deployment.yml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-agi
  namespace: sutazai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend-agi
  template:
    metadata:
      labels:
        app: backend-agi
    spec:
      containers:
      - name: backend
        image: sutazai/backend:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: sutazai-secrets
              key: database-url
        resources:
          requests:
            memory: "2Gi"
            cpu: "1000m"
          limits:
            memory: "4Gi"
            cpu: "2000m"
```

### 4. Cloud Deployment

#### AWS ECS Configuration
```json
{
  "family": "sutazai-backend",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "2048",
  "memory": "4096",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "backend",
      "image": "sutazai/backend:latest",
      "portMappings": [
        {
          "containerPort": 8000,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "DATABASE_URL",
          "value": "postgresql://user:pass@rds-endpoint:5432/sutazai"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/sutazai",
          "awslogs-region": "us-west-2"
        }
      }
    }
  ]
}
```

---

## Post-deployment Steps

### Health Checks

#### Automated Health Validation
```bash
# Run comprehensive health check
./scripts/deploy_complete_sutazai_agi_system.sh health

# Individual service checks
curl http://localhost:8000/health
curl http://localhost:8501/healthz
curl http://localhost:11434/api/tags
curl http://localhost:3000/api/health
```

#### Manual Verification
```bash
# Check all services are running
docker compose ps

# Check logs for errors
docker compose logs --tail=50

# Test API endpoints
curl -X GET http://localhost:8000/api/v1/system/status
curl -X GET http://localhost:8000/docs
```

### Initial Configuration

#### Database Setup
```bash
# Initialize database schema
docker exec sutazai-backend-agi alembic upgrade head

# Create initial user (if applicable)
docker exec sutazai-backend-agi python -c "
from app.models import User
from app.core.database import SessionLocal
db = SessionLocal()
user = User(username='admin', email='admin@sutazai.com')  
db.add(user)
db.commit()
"
```

#### Model Loading
```bash
# Load essential AI models
docker exec sutazai-ollama ollama pull tinyllama:latest
docker exec sutazai-ollama ollama pull qwen2.5:3b
docker exec sutazai-ollama ollama pull codellama:7b
docker exec sutazai-ollama ollama pull nomic-embed-text:latest

# Verify models are loaded
docker exec sutazai-ollama ollama list
```

### Agent Registration

#### Register Core Agents
```bash
# Register agents with the system
curl -X POST http://localhost:8000/api/v1/agents/register \
  -H "Content-Type: application/json" \
  -d '{
    "name": "senior-ai-engineer",
    "type": "code-assistant", 
    "endpoint": "http://senior-ai-engineer:8080",
    "capabilities": ["code-review", "optimization", "debugging"]
  }'
```

#### Verify Agent Status
```bash
# Check agent registry
curl http://localhost:8000/api/v1/agents/list

# Test agent communication
curl -X POST http://localhost:8000/api/v1/agents/senior-ai-engineer/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, can you help me review some code?"}'
```

### Monitoring Setup

#### Configure Grafana Dashboards
```bash
# Access Grafana
open http://localhost:3000
# Login: admin / (check GRAFANA_PASSWORD in .env)

# Import dashboards
curl -X POST http://admin:password@localhost:3000/api/dashboards/db \
  -H "Content-Type: application/json" \
  -d @monitoring/grafana/dashboards/sutazai-system-overview.json
```

#### Setup Alerting
```bash
# Configure alert channels (Slack, Email, etc.)
curl -X POST http://admin:password@localhost:3000/api/alert-notifications \
  -H "Content-Type: application/json" \
  -d '{
    "name": "slack-alerts",
    "type": "slack",
    "settings": {
      "url": "YOUR_SLACK_WEBHOOK_URL",
      "channel": "#alerts"
    }
  }'
```

---

## Troubleshooting Guide

### Common Issues

#### 1. Services Not Starting

**Symptoms:**
- Containers exiting immediately
- "Connection refused" errors
- Health checks failing

**Solutions:**
```bash
# Check container logs
docker compose logs <service-name>

# Check resource usage
docker stats

# Verify network connectivity
docker network ls
docker network inspect sutazai-network

# Check port conflicts
netstat -tulpn | grep :8000
```

#### 2. Memory Issues

**Symptoms:**
- OOM (Out of Memory) kills
- Slow performance
- Services restarting frequently

**Solutions:**
```bash
# Check memory usage
free -h
docker stats --no-stream

# Reduce resource limits
# Edit docker-compose.yml:
deploy:
  resources:
    limits:
      memory: 2G  # Reduce from 4G

# Enable swap (if not already)
sudo swapon --show
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

#### 3. Model Loading Issues

**Symptoms:**
- Ollama returns "model not found"
- Very slow inference
- Connection timeouts

**Solutions:**
```bash
# Check available models
docker exec sutazai-ollama ollama list

# Pull models manually
docker exec sutazai-ollama ollama pull tinyllama:latest

# Check Ollama logs
docker logs sutazai-ollama

# Restart Ollama service
docker compose restart ollama
```

#### 4. Database Connection Issues

**Symptoms:**
- "Connection refused" to PostgreSQL
- Database migration failures
- Authentication errors

**Solutions:**
```bash
# Check PostgreSQL status
docker exec sutazai-postgres pg_isready -U sutazai

# Test connection
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT 1;"

# Reset database (CAUTION: Data loss)
docker compose down postgres
docker volume rm sutazai_postgres_data
docker compose up -d postgres
```

### Log Locations

```bash
# Container logs
docker compose logs -f <service-name>

# Application logs (if mounted)
tail -f logs/deployment_*.log
tail -f logs/backend.log
tail -f logs/agents/*.log

# System logs
journalctl -u docker.service
/var/log/docker.log
```

### Debug Procedures

#### Enable Debug Mode
```bash
# Backend debug
docker compose exec backend-agi python -c "
import logging
logging.basicConfig(level=logging.DEBUG)
"

# Ollama debug
docker compose exec ollama sh -c "
export OLLAMA_DEBUG=1
ollama serve
"
```

#### Network Debugging
```bash
# Test connectivity between services
docker exec sutazai-backend-agi ping postgres
docker exec sutazai-backend-agi ping ollama
docker exec sutazai-backend-agi ping redis

# Check DNS resolution
docker exec sutazai-backend-agi nslookup postgres
```

#### Performance Debugging
```bash
# Check resource usage over time
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Profile application
docker exec sutazai-backend-agi python -m cProfile -s cumulative app/main.py
```

---

## Performance Tuning

### Resource Optimization

#### Memory Optimization
```yaml
# docker-compose.override.yml
services:
  ollama:
    environment:
      - OLLAMA_NUM_PARALLEL=1  # Reduce parallel requests
      - OLLAMA_MAX_LOADED_MODELS=1  # Keep only one model in memory
    deploy:
      resources:
        limits:
          memory: 4G  # Adjust based on available RAM

  postgres:
    command: >
      postgres
      -c shared_buffers=256MB
      -c effective_cache_size=1GB
      -c maintenance_work_mem=64MB
      -c checkpoint_completion_target=0.9
```

#### CPU Optimization
```yaml
services:
  backend-agi:
    environment:
      - MAX_WORKERS=2  # Limit worker processes
      - WORKER_TIMEOUT=30
    deploy:
      resources:
        limits:
          cpus: '2.0'
```

### Network Optimization

```yaml
# Use host networking for performance (less secure)
services:
  backend-agi:
    network_mode: host
```

### Storage Optimization

```bash
# Use faster storage for databases
docker volume create --driver local \
  --opt type=tmpfs \
  --opt device=tmpfs \
  --opt o=size=2g \
  redis_tmpfs

# Enable compression
docker run --rm -v postgres_data:/data alpine sh -c "
  tar czf /data/backup.tar.gz /data/postgresql
"
```

---

## Security Configuration

### Network Security

#### Enable HTTPS
```yaml
# nginx/nginx.conf
server {
    listen 443 ssl http2;
    server_name sutazai.yourdomain.com;
    
    ssl_certificate /etc/ssl/certs/sutazai.crt;
    ssl_certificate_key /etc/ssl/private/sutazai.key;
    
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
}
```

#### Firewall Configuration
```bash
# UFW (Ubuntu/Debian)
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable

# iptables (manual)
iptables -A INPUT -p tcp --dport 8000 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 8000 -j DROP
```

### Application Security

#### API Security
```python
# backend/app/core/security.py
from fastapi import Security, Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def verify_api_key(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    return credentials.credentials
```

#### Database Security
```bash
# Enable PostgreSQL SSL
docker compose exec postgres psql -U postgres -c "
ALTER SYSTEM SET ssl = on;
ALTER SYSTEM SET ssl_cert_file = '/etc/ssl/certs/server.crt';
ALTER SYSTEM SET ssl_key_file = '/etc/ssl/private/server.key';
SELECT pg_reload_conf();
"
```

### Secrets Management

#### Docker Secrets
```yaml
# docker-compose.yml
secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
  jwt_secret:
    file: ./secrets/jwt_secret.txt

services:
  postgres:
    secrets:
      - postgres_password
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
```

#### External Secrets (Kubernetes)
```yaml
apiVersion: external-secrets.io/v1beta1
kind: SecretStore
metadata:
  name: vault-backend
spec:
  provider:
    vault:
      server: "https://vault.company.com"
      path: "secret"
      version: "v2"
      auth:
        kubernetes:
          mountPath: "kubernetes"
          role: "sutazai"
```

---

## Backup and Recovery

### Automated Backup

#### Database Backup
```bash
#!/bin/bash
# scripts/backup_database.sh
BACKUP_DIR="/opt/sutazai/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
docker exec sutazai-postgres pg_dump -U sutazai sutazai > $BACKUP_DIR/postgres_$DATE.sql

# Backup Redis
docker exec sutazai-redis redis-cli BGSAVE
docker cp sutazai-redis:/data/dump.rdb $BACKUP_DIR/redis_$DATE.rdb

# Cleanup old backups (keep 7 days)
find $BACKUP_DIR -name "*.sql" -mtime +7 -delete
find $BACKUP_DIR -name "*.rdb" -mtime +7 -delete
```

#### Volume Backup
```bash
#!/bin/bash
# scripts/backup_volumes.sh
docker run --rm \
  -v sutazai_postgres_data:/data \
  -v /opt/sutazai/backups:/backup \
  alpine tar czf /backup/postgres_data_$(date +%Y%m%d).tar.gz -C /data .

docker run --rm \
  -v sutazai_ollama_data:/data \
  -v /opt/sutazai/backups:/backup \
  alpine tar czf /backup/ollama_data_$(date +%Y%m%d).tar.gz -C /data .
```

### Recovery Procedures

#### Database Recovery
```bash
# Stop services
docker compose down

# Restore PostgreSQL
docker compose up -d postgres
docker exec -i sutazai-postgres psql -U sutazai sutazai < backup/postgres_20250101_120000.sql

# Restore Redis
docker cp backup/redis_20250101_120000.rdb sutazai-redis:/data/dump.rdb
docker compose restart redis
```

#### Full System Recovery
```bash
# 1. Restore configuration
cp backup/.env .env
cp -r backup/config/* config/

# 2. Restore data volumes
docker volume create sutazai_postgres_data
docker volume create sutazai_ollama_data

# 3. Extract backups
docker run --rm \
  -v sutazai_postgres_data:/data \
  -v /opt/sutazai/backups:/backup \
  alpine tar xzf /backup/postgres_data_20250101.tar.gz -C /data

# 4. Start services
docker compose up -d

# 5. Verify recovery
./scripts/deploy_complete_sutazai_agi_system.sh health
```

### Disaster Recovery Plan

#### Recovery Time Objectives (RTO)
- **Database**: 15 minutes
- **Application**: 30 minutes  
- **Full System**: 1 hour

#### Recovery Point Objectives (RPO)
- **Database**: 1 hour (hourly backups)
- **Configurations**: 24 hours (daily backups)
- **Models**: 7 days (weekly backups)

#### Disaster Recovery Checklist
```markdown
## Disaster Recovery Checklist

### Phase 1: Assessment (0-15 minutes)
- [ ] Identify scope of failure
- [ ] Determine if backup/restore or rebuild is needed
- [ ] Notify stakeholders
- [ ] Document incident start time

### Phase 2: Recovery (15-60 minutes)
- [ ] Stop affected services
- [ ] Restore from most recent backup
- [ ] Verify data integrity
- [ ] Restart services in dependency order
- [ ] Run health checks

### Phase 3: Validation (60-90 minutes)
- [ ] Test all critical functions
- [ ] Verify agent communications
- [ ] Check monitoring systems
- [ ] Validate recent data
- [ ] Performance testing

### Phase 4: Post-Recovery (90+ minutes)
- [ ] Document root cause
- [ ] Update recovery procedures
- [ ] Schedule post-mortem
- [ ] Implement preventive measures
```

---

## Monitoring and Maintenance

### Health Monitoring

#### Automated Health Checks
```bash
#!/bin/bash
# scripts/health_monitor.sh
SERVICES=(
  "http://localhost:8000/health|Backend API"
  "http://localhost:8501|Frontend"
  "http://localhost:11434/api/tags|Ollama"
  "http://localhost:3000/api/health|Grafana"
)

for service in "${SERVICES[@]}"; do
  IFS='|' read -r url name <<< "$service"
  if curl -f -s --max-time 10 "$url" >/dev/null 2>&1; then
    echo "✅ $name: OK"
  else
    echo "❌ $name: FAILED"
    # Send alert here
  fi
done
```

#### Prometheus Metrics
```python
# backend/app/core/metrics.py
from prometheus_client import Counter, Histogram, Gauge, start_http_server

REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint'])
REQUEST_DURATION = Histogram('http_request_duration_seconds', 'HTTP request duration')
ACTIVE_CONNECTIONS = Gauge('active_connections', 'Active database connections')

# Start metrics server
start_http_server(8080)
```

### Maintenance Tasks

#### Daily Maintenance
```bash
#!/bin/bash
# scripts/daily_maintenance.sh

# Check disk space
df -h | awk '$5 > 80 { print "WARNING: " $1 " is " $5 " full" }'

# Rotate logs
docker compose exec backend-agi logrotate /etc/logrotate.conf

# Update system metrics
docker system df
docker system prune -f --volumes=false

# Backup critical data
./scripts/backup_database.sh
```

#### Weekly Maintenance
```bash
#!/bin/bash
# scripts/weekly_maintenance.sh

# Update containers
docker compose pull
docker compose up -d

# Clean up unused images
docker image prune -f

# Update AI models
docker exec sutazai-ollama ollama pull tinyllama:latest
```

#### Monthly Maintenance
```bash
#!/bin/bash
# scripts/monthly_maintenance.sh

# Security scan
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image sutazai/backend:latest

# Performance analysis
docker stats --no-stream > performance_report_$(date +%Y%m).txt

# Update dependencies
pip list --outdated
npm audit

# Capacity planning
echo "Current resource usage:" > capacity_report_$(date +%Y%m).txt
docker stats --no-stream >> capacity_report_$(date +%Y%m).txt
```

This comprehensive guide covers all aspects of deploying and maintaining the SutazAI system, from basic installation to production operations. The documentation provides practical, actionable guidance for system administrators and developers at all levels.