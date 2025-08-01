# SutazAI Production Deployment Guide

## Quick Start Production Deployment

### Prerequisites Checklist

```bash
# System Requirements Verification
□ Ubuntu 20.04+ / RHEL 8+ / Docker Desktop
□ 8+ CPU cores (16 recommended)
□ 32GB+ RAM (64GB recommended)
□ 100GB+ SSD storage (200GB+ recommended)
□ Docker Engine 24.0+
□ Docker Compose V2
□ Network connectivity for model downloads
```

### 1. Rapid Production Deployment

```bash
# Clone and deploy in one command
git clone <repository-url> /opt/sutazaiapp
cd /opt/sutazaiapp
chmod +x scripts/deploy_complete_sutazai_agi_system.sh
./scripts/deploy_complete_sutazai_agi_system.sh
```

### 2. Verification Commands

```bash
# Check system status
docker-compose ps
curl http://localhost:8000/health
curl http://localhost:8501

# View live logs
./scripts/live_logs.sh

# Monitor resource usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"
```

## Production Configuration Profiles

### Small Production Environment (32GB RAM)

```yaml
# docker-compose.small-prod.yml
version: '3.8'

x-small-resource-limits: &small-limits
  deploy:
    resources:
      limits:
        cpus: '1'
        memory: 2G
      reservations:
        cpus: '0.5'
        memory: 1G

x-minimal-resource-limits: &minimal-limits
  deploy:
    resources:
      limits:
        cpus: '0.5'
        memory: 1G
      reservations:
        cpus: '0.25'
        memory: 512M

services:
  # Core services with reduced resources
  postgres:
    <<: *minimal-limits
    image: postgres:16.3-alpine
    environment:
      POSTGRES_SHARED_BUFFERS: 128MB
      POSTGRES_MAX_CONNECTIONS: 100

  ollama:
    <<: *small-limits
    environment:
      OLLAMA_MAX_LOADED_MODELS: 1
      OLLAMA_NUM_PARALLEL: 1
      OLLAMA_KEEP_ALIVE: 1m

  backend-agi:
    <<: *small-limits
    environment:
      MAX_WORKERS: 2
      MAX_CONCURRENT_REQUESTS: 10
```

### Large Production Environment (128GB+ RAM)

```yaml
# docker-compose.large-prod.yml
version: '3.8'

x-high-performance: &high-perf
  deploy:
    resources:
      limits:
        cpus: '8'
        memory: 16G
      reservations:
        cpus: '4'
        memory: 8G

services:
  ollama:
    <<: *high-perf
    environment:
      OLLAMA_MAX_LOADED_MODELS: 5
      OLLAMA_NUM_PARALLEL: 4
      OLLAMA_KEEP_ALIVE: 10m

  backend-agi:
    deploy:
      resources:
        limits:
          cpus: '4'
          memory: 8G
        reservations:
          cpus: '2'
          memory: 4G
      replicas: 3  # Load balanced backend
    environment:
      MAX_WORKERS: 8
      MAX_CONCURRENT_REQUESTS: 50
```

## Environment-Specific Configurations

### Development Environment

```bash
# .env.development
SUTAZAI_ENV=development
LOG_LEVEL=DEBUG
ENABLE_DEBUG_ENDPOINTS=true
ENABLE_PROFILING=true
MODEL_CACHE_SIZE=2GB
MAX_CONCURRENT_AGENTS=5
```

### Staging Environment

```bash
# .env.staging
SUTAZAI_ENV=staging
LOG_LEVEL=INFO
ENABLE_MONITORING=true
ENABLE_SECURITY_SCAN=true
MODEL_CACHE_SIZE=5GB
MAX_CONCURRENT_AGENTS=10
BACKUP_RETENTION_DAYS=7
```

### Production Environment

```bash
# .env.production
SUTAZAI_ENV=production
LOG_LEVEL=WARNING
ENABLE_MONITORING=true
ENABLE_SECURITY_SCAN=true
ENABLE_AUTO_BACKUP=true
MODEL_CACHE_SIZE=10GB
MAX_CONCURRENT_AGENTS=20
BACKUP_RETENTION_DAYS=30
HEALTH_CHECK_INTERVAL=30
```

## Load Balancing and High Availability

### NGINX Load Balancer Configuration

```nginx
# /opt/sutazaiapp/nginx/nginx.conf
upstream backend_pool {
    least_conn;
    server backend-agi-1:8000 max_fails=3 fail_timeout=30s;
    server backend-agi-2:8000 max_fails=3 fail_timeout=30s;
    server backend-agi-3:8000 max_fails=3 fail_timeout=30s;
}

upstream frontend_pool {
    ip_hash;  # Session affinity for Streamlit
    server frontend-agi-1:8501 max_fails=3 fail_timeout=30s;
    server frontend-agi-2:8501 max_fails=3 fail_timeout=30s;
}

server {
    listen 80;
    server_name sutazai.local;

    # Backend API
    location /api/ {
        proxy_pass http://backend_pool;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # Timeouts
        proxy_connect_timeout 5s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;
        
        # Health checks
        proxy_next_upstream error timeout http_500 http_502 http_503;
    }

    # Frontend Application
    location / {
        proxy_pass http://frontend_pool;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        
        # WebSocket support for Streamlit
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
    }
}
```

### Docker Swarm Deployment

```yaml
# docker-stack.yml for Docker Swarm
version: '3.8'

services:
  backend-agi:
    image: sutazai/backend:latest
    deploy:
      replicas: 3
      placement:
        constraints:
          - node.role == worker
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
      update_config:
        parallelism: 1
        delay: 10s
        failure_action: rollback
      rollback_config:
        parallelism: 1
        delay: 5s
        failure_action: pause
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 60s

  postgres:
    image: postgres:16.3-alpine
    deploy:
      replicas: 1
      placement:
        constraints:
          - node.labels.postgres == true
      restart_policy:
        condition: on-failure
    volumes:
      - postgres_data:/var/lib/postgresql/data
```

## Kubernetes Deployment

### Kubernetes Manifests

```yaml
# k8s/namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sutazai-system
---
# k8s/configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sutazai-config
  namespace: sutazai-system
data:
  SUTAZAI_ENV: "production"
  LOG_LEVEL: "INFO"
  ENABLE_MONITORING: "true"
---
# k8s/secret.yaml
apiVersion: v1
kind: Secret
metadata:
  name: sutazai-secrets
  namespace: sutazai-system
type: Opaque
stringData:
  postgres-password: "secure-random-password"
  redis-password: "secure-random-password"
  jwt-secret: "secure-jwt-secret"
---
# k8s/postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: sutazai-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: postgres:16.3-alpine
        env:
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: sutazai-secrets
              key: postgres-password
        - name: POSTGRES_USER
          value: "sutazai"
        - name: POSTGRES_DB
          value: "sutazai"
        resources:
          requests:
            memory: "2Gi"
            cpu: "500m"
          limits:
            memory: "4Gi"
            cpu: "1"
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        livenessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - sutazai
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          exec:
            command:
            - pg_isready
            - -U
            - sutazai
          initialDelaySeconds: 5
          periodSeconds: 5
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
# k8s/backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: backend-agi
  namespace: sutazai-system
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
          value: "postgresql://sutazai:$(POSTGRES_PASSWORD)@postgres:5432/sutazai"
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: sutazai-secrets
              key: postgres-password
        envFrom:
        - configMapRef:
            name: sutazai-config
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 60
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
---
# k8s/backend-service.yaml
apiVersion: v1
kind: Service
metadata:
  name: backend-agi
  namespace: sutazai-system
spec:
  selector:
    app: backend-agi
  ports:
  - protocol: TCP
    port: 8000
    targetPort: 8000
  type: ClusterIP
---
# k8s/ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sutazai-ingress
  namespace: sutazai-system
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
    nginx.ingress.kubernetes.io/ssl-redirect: "false"
spec:
  rules:
  - host: sutazai.local
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: backend-agi
            port:
              number: 8000
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-agi
            port:
              number: 8501
```

## Monitoring and Alerting Setup

### Prometheus Configuration

```yaml
# monitoring/prometheus/prometheus.yml
global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alerts.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'prometheus'
    static_configs:
      - targets: ['localhost:9090']

  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['backend-agi:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'docker-containers'
    static_configs:
      - targets: ['node-exporter:9100']

  - job_name: 'postgres'
    static_configs:
      - targets: ['postgres-exporter:9187']

  - job_name: 'redis'
    static_configs:
      - targets: ['redis-exporter:9121']

  - job_name: 'ollama'
    static_configs:
      - targets: ['ollama:11434']
    metrics_path: '/metrics'
    scrape_interval: 60s
```

### Grafana Dashboard Configuration

```json
{
  "dashboard": {
    "title": "SutazAI System Overview",
    "panels": [
      {
        "title": "System Health",
        "type": "stat",
        "targets": [
          {
            "expr": "up{job=\"sutazai-backend\"}",
            "legendFormat": "Backend Status"
          }
        ]
      },
      {
        "title": "API Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "container_memory_usage_bytes{name=~\"sutazai-.*\"} / 1024 / 1024 / 1024",
            "legendFormat": "{{name}}"
          }
        ]
      }
    ]
  }
}
```

## Security Hardening for Production

### SSL/TLS Configuration

```yaml
# docker-compose.prod.yml with SSL
version: '3.8'

services:
  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx/nginx-ssl.conf:/etc/nginx/nginx.conf
      - ./ssl/cert.pem:/etc/ssl/certs/sutazai.pem
      - ./ssl/key.pem:/etc/ssl/private/sutazai.key
    depends_on:
      - backend-agi
      - frontend-agi
```

```nginx
# nginx/nginx-ssl.conf
server {
    listen 443 ssl http2;
    server_name sutazai.local;

    ssl_certificate /etc/ssl/certs/sutazai.pem;
    ssl_certificate_key /etc/ssl/private/sutazai.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers ECDHE-RSA-AES256-GCM-SHA512:DHE-RSA-AES256-GCM-SHA512;
    ssl_prefer_server_ciphers off;

    # Security headers
    add_header Strict-Transport-Security "max-age=63072000" always;
    add_header X-Frame-Options DENY;
    add_header X-Content-Type-Options nosniff;
    add_header X-XSS-Protection "1; mode=block";

    location / {
        proxy_pass http://frontend-agi:8501;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### Secrets Management

```bash
# scripts/generate-secrets.sh
#!/bin/bash
set -euo pipefail

SECRETS_DIR="/opt/sutazaiapp/secrets"
mkdir -p "$SECRETS_DIR"

# Generate secure passwords
POSTGRES_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
REDIS_PASSWORD=$(openssl rand -base64 32 | tr -d "=+/" | cut -c1-25)
JWT_SECRET=$(openssl rand -hex 32)
GRAFANA_PASSWORD=$(openssl rand -base64 16 | tr -d "=+/" | cut -c1-16)

# Store secrets securely
echo "$POSTGRES_PASSWORD" | tee "$SECRETS_DIR/postgres_password.txt"
echo "$REDIS_PASSWORD" | tee "$SECRETS_DIR/redis_password.txt"
echo "$JWT_SECRET" | tee "$SECRETS_DIR/jwt_secret.txt"
echo "$GRAFANA_PASSWORD" | tee "$SECRETS_DIR/grafana_password.txt"

# Set secure permissions
chmod 600 "$SECRETS_DIR"/*
chown root:root "$SECRETS_DIR"/*

echo "Secrets generated and stored securely in $SECRETS_DIR"
```

## Performance Optimization

### Database Optimization

```sql
-- PostgreSQL optimization queries
-- /opt/sutazaiapp/config/postgres/performance.sql

-- Increase shared_buffers for better performance
ALTER SYSTEM SET shared_buffers = '2GB';

-- Optimize for AI workloads
ALTER SYSTEM SET work_mem = '256MB';
ALTER SYSTEM SET maintenance_work_mem = '512MB';
ALTER SYSTEM SET effective_cache_size = '6GB';

-- Connection pooling
ALTER SYSTEM SET max_connections = 200;

-- Enable parallel processing
ALTER SYSTEM SET max_parallel_workers = 8;
ALTER SYSTEM SET max_parallel_workers_per_gather = 4;

-- Optimize for write-heavy workloads
ALTER SYSTEM SET checkpoint_completion_target = 0.7;
ALTER SYSTEM SET wal_buffers = '64MB';

-- Reload configuration
SELECT pg_reload_conf();
```

### Redis Optimization

```conf
# config/redis/redis-prod.conf
# Memory optimization
maxmemory 4gb
maxmemory-policy allkeys-lru

# Persistence optimization
save 900 1
save 300 10
save 60 10000

# Network optimization
tcp-keepalive 300
timeout 0

# Performance tuning
tcp-backlog 511
databases 16
```

### Ollama Model Optimization

```bash
# scripts/optimize-ollama.sh
#!/bin/bash

# Pre-load essential models
docker exec sutazai-ollama ollama pull llama3.2:3b
docker exec sutazai-ollama ollama pull qwen2.5:3b
docker exec sutazai-ollama ollama pull codellama:7b

# Optimize model loading
docker exec sutazai-ollama sh -c '
echo "FROM llama3.2:3b
PARAMETER num_thread 8
PARAMETER num_gpu 0
PARAMETER num_ctx 4096" | ollama create llama3.2-optimized
'

# Warm up models
curl -X POST http://localhost:11434/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "llama3.2-optimized", "prompt": "Hello", "stream": false}'
```

This production deployment guide provides comprehensive instructions for deploying the SutazAI system in various production environments with proper security, monitoring, and performance optimization.