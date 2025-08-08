# SutazAI Infrastructure Setup Guide

**Version:** 1.0.0  
**Last Updated:** 2025-08-08  
**Maintainer:** Infrastructure Team  
**Based on:** CLAUDE.md System Reality Check

---

## Table of Contents

1. [Infrastructure Overview](#infrastructure-overview)
2. [Prerequisites](#prerequisites)
3. [Initial Setup](#initial-setup)
4. [Service Configuration](#service-configuration)
5. [Environment Configuration](#environment-configuration)
6. [Network Configuration](#network-configuration)
7. [Storage Configuration](#storage-configuration)
8. [Security Hardening](#security-hardening)
9. [Resource Management](#resource-management)
10. [Monitoring Infrastructure](#monitoring-infrastructure)
11. [Scaling Considerations](#scaling-considerations)
12. [Disaster Recovery](#disaster-recovery)
13. [Troubleshooting](#troubleshooting)
14. [Migration Paths](#migration-paths)

---

## Infrastructure Overview

The SutazAI system is built on a Docker Compose architecture with 59 defined services, of which 28 containers are currently running in a production environment. The system provides a comprehensive AI agent platform with full monitoring, vector databases, and distributed processing capabilities.

### Current Architecture (Verified)

- **Docker Compose Based**: Single-host orchestration with multi-container services
- **Network**: Custom bridge network (`sutazai-network`)
- **Service Count**: 28 active containers out of 59 defined services
- **Port Range**: 8000-11200 (see [Port Registry](#port-registry))
- **Storage**: Named volumes with persistent data
- **Monitoring**: Full Prometheus/Grafana/Loki stack

### Service Topology

```
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   Frontend      │  │   Backend       │  │   Ollama LLM    │
│  (Streamlit)    │◄─┤   (FastAPI)     │◄─┤  (TinyLlama)    │
│   Port 10011    │  │   Port 10010    │  │   Port 10104    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                     │                     │
    ┌────▼─────────────────────▼─────────────────────▼────┐
    │              Core Infrastructure                    │
    │  PostgreSQL   Redis    Neo4j    ChromaDB   Qdrant   │
    │    10000      10001    10002     10100     10101    │
    └──────────────────────────────────────────────────────┘
         │
    ┌────▼─────────────────────┐
    │    Service Mesh         │
    │  Kong   Consul  RabbitMQ │
    │  10005   10006   10007   │
    └─────────────────────────┘
         │
    ┌────▼─────────────────────┐
    │   Monitoring Stack      │
    │ Prometheus Grafana Loki │
    │   10200    10201  10202  │
    └─────────────────────────┘
```

### Network Architecture

- **Network Name**: `sutazai-network`
- **Network Type**: Docker bridge network
- **DNS Resolution**: Automatic service discovery via container names
- **External Access**: Port mappings for key services
- **Internal Communication**: Service-to-service via container names

---

## Prerequisites

### Hardware Requirements

#### Minimum Requirements
- **CPU**: 4 cores (x86_64)
- **RAM**: 8GB minimum, 16GB recommended
- **Disk**: 50GB free space (SSD recommended)
- **Network**: 100Mbps internet connection

#### Recommended Requirements
- **CPU**: 8+ cores (x86_64)
- **RAM**: 32GB (for full service stack)
- **Disk**: 100GB+ SSD storage
- **Network**: 1Gbps connection

#### Production Requirements
- **CPU**: 16+ cores
- **RAM**: 64GB+
- **Disk**: 500GB+ NVMe SSD
- **Network**: Dedicated network interface

### Software Requirements

#### Essential Components
```bash
# Docker Engine (tested with 24.x+)
docker --version
# Docker Compose v2 (required)
docker-compose --version
```

#### Operating System Compatibility
- **Linux**: Ubuntu 20.04+, CentOS 8+, RHEL 8+ (Recommended)
- **macOS**: 10.15+ with Docker Desktop
- **Windows**: Windows 10/11 with WSL2 and Docker Desktop

#### Network Requirements
- **Firewall**: Ports 8000-11200 range accessible
- **DNS**: Local resolution capabilities
- **Internet**: Access for Docker image pulls

---

## Initial Setup

### 1. Docker Installation

#### Ubuntu/Debian
```bash
# Remove old versions
sudo apt-get remove docker docker-engine docker.io containerd runc

# Install Docker Engine
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose v2
sudo apt-get install docker-compose-plugin
```

#### CentOS/RHEL
```bash
# Install Docker Engine
sudo yum install -y yum-utils
sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
sudo yum install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin

# Start Docker service
sudo systemctl start docker
sudo systemctl enable docker
```

### 2. System Configuration

#### Create Required Directories
```bash
# Create data directories
sudo mkdir -p /opt/sutazaiapp/{data,logs,configs,secrets}

# Set permissions
sudo chown -R $USER:docker /opt/sutazaiapp
sudo chmod -R 755 /opt/sutazaiapp
```

#### Configure System Limits
```bash
# Edit /etc/security/limits.conf
echo "*    soft    nofile    65536" | sudo tee -a /etc/security/limits.conf
echo "*    hard    nofile    65536" | sudo tee -a /etc/security/limits.conf

# Edit /etc/sysctl.conf for networking
echo "net.core.somaxconn=65535" | sudo tee -a /etc/sysctl.conf
echo "vm.max_map_count=262144" | sudo tee -a /etc/sysctl.conf

# Apply changes
sudo sysctl -p
```

### 3. Network Creation

#### Create Docker Network
```bash
# Create the sutazai network (idempotent)
docker network create sutazai-network 2>/dev/null || echo "Network already exists"

# Verify network creation
docker network ls | grep sutazai-network
docker network inspect sutazai-network
```

### 4. Directory Structure Setup

```bash
# Create required directory structure
mkdir -p /opt/sutazaiapp/{
    agents/{core,configs},
    backend/{app,data},
    frontend,
    monitoring/{grafana,prometheus,loki,alertmanager},
    data/{postgres,redis,neo4j,chromadb,qdrant,faiss},
    logs/{agents,backend,frontend,monitoring},
    configs/{kong,consul},
    secrets,
    backups
}
```

---

## Service Configuration

### Core Services

#### PostgreSQL Database
```yaml
# Primary database for application state
Service: sutazai-postgres
Port: 10000
Image: postgres:16.3-alpine
Resources:
  CPU: 2 cores limit, 0.5 cores reserved
  Memory: 2GB limit, 512MB reserved
Volume: postgres_data:/var/lib/postgresql/data
Health Check: pg_isready
```

**Configuration:**
```bash
# Database initialization
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=<secure_password>
POSTGRES_DB=sutazai
```

#### Redis Cache
```yaml
# In-memory cache and session store
Service: sutazai-redis
Port: 10001
Image: redis:7.2-alpine
Resources:
  CPU: 0.5 cores limit, 0.1 cores reserved
  Memory: 512MB limit, 128MB reserved
Configuration: --maxmemory 512mb --maxmemory-policy allkeys-lru
```

#### Neo4j Graph Database
```yaml
# Knowledge graph and relationships
Service: sutazai-neo4j
Ports: 10002 (HTTP), 10003 (Bolt)
Image: neo4j:5.13-community
Resources:
  CPU: 1.5 cores limit, 0.5 cores reserved
  Memory: 1GB limit, 512MB reserved
```

**Neo4j Optimization Settings:**
```bash
NEO4J_server_memory_heap_max__size=512m
NEO4J_server_memory_heap_initial__size=256m
NEO4J_server_memory_pagecache_size=256m
NEO4J_server_jvm_additional="-XX:+UseG1GC -XX:G1HeapRegionSize=4m"
```

### Application Services

#### Backend API
```yaml
# FastAPI application server
Service: sutazai-backend
Port: 10010
Build: ./backend/Dockerfile
Resources:
  CPU: 4 cores limit, 1 core reserved
  Memory: 4GB limit, 1GB reserved
Dependencies: postgres, redis, neo4j, ollama, chromadb, qdrant
```

#### Frontend Interface
```yaml
# Streamlit web interface
Service: sutazai-frontend
Port: 10011
Build: ./frontend/Dockerfile
Dependencies: backend (healthy)
Command: streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

#### Ollama LLM Service
```yaml
# Local language model server
Service: sutazai-ollama
Port: 10104
Image: ollama/ollama:latest
Resources:
  CPU: 10 cores limit, 4 cores reserved
  Memory: 20GB limit, 8GB reserved
Model: TinyLlama (637MB)
```

### Vector Databases

#### ChromaDB
```yaml
Service: sutazai-chromadb
Port: 10100
Image: chromadb/chroma:0.5.0
Status: Connection issues (known)
Resources:
  CPU: 1 core limit, 0.25 cores reserved
  Memory: 1GB limit, 256MB reserved
```

#### Qdrant
```yaml
Service: sutazai-qdrant
Ports: 10101 (HTTP), 10102 (gRPC)
Image: qdrant/qdrant:v1.9.2
Resources:
  CPU: 2 cores limit, 0.5 cores reserved
  Memory: 2GB limit, 512MB reserved
```

#### FAISS
```yaml
Service: sutazai-faiss
Port: 10103
Build: ./docker/faiss/Dockerfile
Resources:
  CPU: 1 core limit, 0.25 cores reserved
  Memory: 512MB limit, 128MB reserved
```

### Agent Services (Flask Stubs)

The system includes 7 active agent services, all currently implemented as Flask stubs:

```yaml
# Hardware Resource Optimizer
Port: 11110
Status: Stub with /health endpoint

# Jarvis Voice Interface
Port: 11150
Status: Stub with /health endpoint

# Jarvis Knowledge Management
Port: 11101
Status: Stub with /health endpoint

# Jarvis Automation Agent
Port: 11102
Status: Stub with /health endpoint

# Jarvis Multimodal AI
Port: 11103
Status: Stub with /health endpoint

# Jarvis Hardware Resource Optimizer
Port: 11104
Status: Stub with /health endpoint

# Ollama Integration
Port: 8090
Status: Functional integration layer
```

### Infrastructure Services

#### Kong API Gateway
```yaml
Service: sutazai-kong
Ports: 10005 (Proxy), 10015 (Admin)
Image: kong:3.5
Configuration: DB-less mode
Status: Running but no routes configured
```

#### Consul Service Discovery
```yaml
Service: sutazai-consul
Port: 10006
Image: hashicorp/consul:1.17
Configuration: Single-node server with UI
Status: Running but minimal usage
```

#### RabbitMQ Message Queue
```yaml
Service: sutazai-rabbitmq
Ports: 10007 (AMQP), 10008 (Management)
Image: rabbitmq:3.12-management-alpine
Status: Running but not actively used
```

---

## Environment Configuration

### Environment File Structure

#### Primary Environment File
```bash
# /opt/sutazaiapp/.env
SUTAZAI_ENV=production
TZ=UTC

# Database Configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=<secure_random_password>
POSTGRES_DB=sutazai
REDIS_PASSWORD=<redis_password>
NEO4J_PASSWORD=<neo4j_password>

# Security
SECRET_KEY=<64_char_hex_key>
JWT_SECRET=<64_char_hex_key>
GRAFANA_PASSWORD=<grafana_password>
CHROMADB_API_KEY=<chromadb_token>

# Optional Features (disabled by default)
ENABLE_FSDP=false
ENABLE_TABBY=false
```

#### Agent-Specific Environment
```bash
# /opt/sutazaiapp/.env.agents
OLLAMA_BASE_URL=http://ollama:10104
OLLAMA_API_KEY=local
OLLAMA_MODEL=tinyllama:latest
BACKEND_URL=http://backend:8000
REDIS_URL=redis://redis:6379/0
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
```

### Required Environment Variables

#### Core System Variables
```bash
SUTAZAI_ENV=production          # Environment identifier
TZ=UTC                          # Timezone setting
```

#### Database Variables
```bash
POSTGRES_USER=sutazai           # PostgreSQL username
POSTGRES_PASSWORD=<password>    # PostgreSQL password (required)
POSTGRES_DB=sutazai            # Database name
REDIS_PASSWORD=<password>      # Redis password (optional)
NEO4J_PASSWORD=<password>      # Neo4j password (required)
```

#### Security Variables
```bash
SECRET_KEY=<64_char_hex>       # Application secret key
JWT_SECRET=<64_char_hex>       # JWT signing secret
GRAFANA_PASSWORD=<password>    # Grafana admin password
CHROMADB_API_KEY=<token>       # ChromaDB authentication token
```

#### Service URLs
```bash
DATABASE_URL=postgresql://${POSTGRES_USER}:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}
REDIS_URL=redis://redis:6379/0
NEO4J_URI=bolt://neo4j:7687
OLLAMA_BASE_URL=http://ollama:10104
CHROMADB_URL=http://chromadb:8000
QDRANT_URL=http://qdrant:6333
```

### Secrets Management

#### Directory Structure
```bash
/opt/sutazaiapp/secrets_secure/
├── postgres_password.txt
├── redis_password.txt
├── neo4j_password.txt
├── jwt_secret.txt
└── grafana_password.txt
```

#### Password Generation
```bash
# Generate secure passwords
openssl rand -hex 32 > secrets_secure/postgres_password.txt
openssl rand -hex 32 > secrets_secure/redis_password.txt
openssl rand -hex 32 > secrets_secure/neo4j_password.txt
openssl rand -hex 64 > secrets_secure/jwt_secret.txt
openssl rand -hex 16 > secrets_secure/grafana_password.txt

# Set appropriate permissions
chmod 600 secrets_secure/*
```

---

## Network Configuration

### Docker Network Setup

#### Network Configuration
```bash
# Create custom bridge network
docker network create \
  --driver bridge \
  --subnet=172.20.0.0/16 \
  --ip-range=172.20.240.0/20 \
  sutazai-network
```

#### Network Inspection
```bash
# Verify network configuration
docker network inspect sutazai-network
docker network ls | grep sutazai
```

### Port Registry

#### Core Services
| Service | Internal Port | External Port | Protocol | Purpose |
|---------|---------------|---------------|----------|---------|
| PostgreSQL | 5432 | 10000 | TCP | Database |
| Redis | 6379 | 10001 | TCP | Cache |
| Neo4j HTTP | 7474 | 10002 | HTTP | Graph DB Browser |
| Neo4j Bolt | 7687 | 10003 | TCP | Graph DB Protocol |
| Kong Proxy | 8000 | 10005 | HTTP | API Gateway |
| Kong Admin | 8001 | 10015 | HTTP | Gateway Admin |
| Consul | 8500 | 10006 | HTTP | Service Discovery |
| RabbitMQ AMQP | 5672 | 10007 | TCP | Message Queue |
| RabbitMQ Mgmt | 15672 | 10008 | HTTP | Queue Management |

#### Application Services  
| Service | Internal Port | External Port | Protocol | Purpose |
|---------|---------------|---------------|----------|---------|
| Backend API | 8000 | 10010 | HTTP | FastAPI Application |
| Frontend UI | 8501 | 10011 | HTTP | Streamlit Interface |
| Ollama LLM | 11434 | 10104 | HTTP | Language Model Server |

#### Vector Databases
| Service | Internal Port | External Port | Protocol | Purpose |
|---------|---------------|---------------|----------|---------|
| ChromaDB | 8000 | 10100 | HTTP | Vector Database |
| Qdrant HTTP | 6333 | 10101 | HTTP | Vector Search |
| Qdrant gRPC | 6334 | 10102 | gRPC | Vector Search API |
| FAISS | 8000 | 10103 | HTTP | Vector Similarity |

#### Monitoring Stack
| Service | Internal Port | External Port | Protocol | Purpose |
|---------|---------------|---------------|----------|---------|
| Prometheus | 9090 | 10200 | HTTP | Metrics Collection |
| Grafana | 3000 | 10201 | HTTP | Visualization |
| Loki | 3100 | 10202 | HTTP | Log Aggregation |
| AlertManager | 9093 | 10203 | HTTP | Alert Routing |
| Blackbox Exporter | 9115 | 10204 | HTTP | Endpoint Monitoring |
| Node Exporter | 9100 | 10205 | HTTP | System Metrics |
| cAdvisor | 8080 | 10206 | HTTP | Container Metrics |
| Postgres Exporter | 9187 | 10207 | HTTP | DB Metrics |
| Redis Exporter | 9121 | 10208 | HTTP | Cache Metrics |

#### Agent Services (Active)
| Service | Internal Port | External Port | Protocol | Status |
|---------|---------------|---------------|----------|--------|
| Hardware Resource Optimizer | 8080 | 11110 | HTTP | Stub |
| Jarvis Voice Interface | 8080 | 11150 | HTTP | Stub |
| Jarvis Knowledge Management | 8080 | 11101 | HTTP | Stub |
| Jarvis Automation Agent | 8080 | 11102 | HTTP | Stub |
| Jarvis Multimodal AI | 8080 | 11103 | HTTP | Stub |
| Jarvis Hardware Resource Optimizer | 8080 | 11104 | HTTP | Stub |
| Ollama Integration | 8090 | 8090 | HTTP | Functional |

### Service Discovery

#### Container Name Resolution
All services communicate using container names as hostnames:
```bash
# Examples of internal service URLs
http://backend:8000      # Backend API
http://postgres:5432     # PostgreSQL database  
http://redis:6379        # Redis cache
http://ollama:11434      # Ollama LLM server
```

#### DNS Configuration
Docker automatically provides DNS resolution for container names within the `sutazai-network`.

### Load Balancing Considerations

#### Current State
- Single-host deployment (no load balancing)
- Kong Gateway available but not configured
- Direct service-to-service communication

#### Future Scaling Options
- Kong Gateway route configuration
- HAProxy integration
- Nginx reverse proxy
- Container orchestration migration

---

## Storage Configuration

### Volume Management

#### Named Volumes
The system uses Docker named volumes for persistent data:

```yaml
volumes:
  # Database Storage
  postgres_data:           # PostgreSQL data
  redis_data:             # Redis persistence
  neo4j_data:             # Neo4j graph data
  
  # Vector Database Storage  
  chromadb_data:          # ChromaDB vectors
  qdrant_data:            # Qdrant vectors
  faiss_data:             # FAISS indices
  
  # LLM Storage
  ollama_data:            # Ollama models
  models_data:            # Shared model storage
  
  # Application Storage
  agent_workspaces:       # Agent working directories
  agent_outputs:          # Agent output files
  
  # Monitoring Storage
  prometheus_data:        # Metrics data
  grafana_data:          # Dashboards and config
  loki_data:             # Log storage
  alertmanager_data:     # Alert state
  
  # Service Mesh Storage
  consul_data:           # Service discovery data
  rabbitmq_data:         # Message queue data
```

#### Host Bind Mounts
Specific directories mounted from host:

```yaml
# Configuration Files
./monitoring/grafana:/etc/grafana/provisioning
./monitoring/prometheus:/etc/prometheus
./config/kong/kong.yml:/etc/kong/kong.yml

# Application Code (Development)
./backend:/app          # Backend source
./frontend:/app         # Frontend source

# System Access
/var/run/docker.sock:/var/run/docker.sock  # Docker API
/proc:/host/proc:ro     # System metrics
/sys:/host/sys:ro       # Hardware info
```

### Persistent Data Locations

#### Volume Inspection
```bash
# List all volumes
docker volume ls | grep sutazai

# Inspect volume location
docker volume inspect postgres_data
docker volume inspect prometheus_data

# Volume backup location (typically)
/var/lib/docker/volumes/
```

#### Data Directories
```bash
# Host data directories
/opt/sutazaiapp/data/
├── postgres/           # Database backups
├── redis/              # Redis dumps
├── neo4j/              # Graph exports
├── chromadb/           # Vector backups
├── qdrant/             # Vector snapshots
├── faiss/              # Index backups
├── ollama/             # Model files
└── monitoring/         # Metrics exports
```

### Backup Strategies

#### Automated Backup Script
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/backup-data.sh

BACKUP_DIR="/opt/sutazaiapp/backups/$(date +%Y-%m-%d)"
mkdir -p "$BACKUP_DIR"

# PostgreSQL backup
docker exec sutazai-postgres pg_dump -U sutazai sutazai > "$BACKUP_DIR/postgres.sql"

# Redis backup
docker exec sutazai-redis redis-cli SAVE
docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/"

# Neo4j backup  
docker exec sutazai-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" \
  "CALL apoc.export.cypher.all('$BACKUP_DIR/neo4j-export.cypher', {})"

# Vector database snapshots
docker exec sutazai-qdrant curl -X POST http://localhost:6333/snapshots
docker exec sutazai-chromadb cp -r /chroma/chroma "$BACKUP_DIR/chromadb/"

# Prometheus data
docker exec sutazai-prometheus promtool tsdb create-blocks-from -r /prometheus "$BACKUP_DIR/prometheus/"
```

#### Backup Schedule
```bash
# Add to crontab
0 2 * * * /opt/sutazaiapp/scripts/backup-data.sh
0 2 * * 0 /opt/sutazaiapp/scripts/cleanup-old-backups.sh
```

### Data Retention Policies

#### Metrics Retention
```yaml
# Prometheus configuration
--storage.tsdb.retention.time=7d
--storage.tsdb.retention.size=1GB
```

#### Log Retention
```yaml
# Loki configuration
retention_deletes_enabled: true
retention_period: 168h  # 7 days
```

#### Database Retention
- PostgreSQL: No automatic cleanup (manual management)
- Redis: LRU eviction policy with 512MB limit
- Neo4j: Manual cleanup required

---

## Security Hardening

### Container Security

#### Security Context
All containers run with appropriate security constraints:

```yaml
# Example security configuration
security_opt:
  - no-new-privileges:true
  - seccomp:unconfined
read_only: true
tmpfs:
  - /tmp
  - /var/tmp
```

#### Privileged Access Control
Only specific containers require privileged access:

```yaml
# Containers requiring privileged access
hardware-resource-optimizer:
  privileged: true          # System monitoring
  pid: host                 # Process monitoring
  volumes:
    - /proc:/host/proc:ro   # System metrics
    - /sys:/host/sys:ro     # Hardware info

cadvisor:
  privileged: true          # Container metrics
```

### Network Isolation

#### Internal Network
```bash
# Services communicate via internal network only
docker network create --internal sutazai-internal

# Expose only necessary ports externally
```

#### Firewall Configuration
```bash
# UFW rules for production
sudo ufw default deny incoming
sudo ufw default allow outgoing

# Allow essential services
sudo ufw allow 22/tcp           # SSH
sudo ufw allow 80/tcp           # HTTP
sudo ufw allow 443/tcp          # HTTPS
sudo ufw allow 10010/tcp        # Backend API
sudo ufw allow 10011/tcp        # Frontend
sudo ufw allow 10201/tcp        # Grafana

# Deny direct database access
sudo ufw deny 10000/tcp         # PostgreSQL
sudo ufw deny 10001/tcp         # Redis
sudo ufw deny 10002/tcp         # Neo4j
```

### Access Controls

#### Authentication Requirements
- Grafana: Username/password authentication
- Neo4j: Database authentication
- ChromaDB: Token-based authentication
- Kong: JWT authentication (when configured)

#### API Security
```yaml
# Backend API security
BACKEND_CORS_ORIGINS: '["http://localhost:10011"]'
JWT_SECRET: <secure_secret>
```

### Secret Management

#### Environment Variables
Never store secrets in Docker images or docker-compose.yml:

```bash
# Use environment files
env_file:
  - .env.production
  - .env.secrets
```

#### Secret Storage
```bash
# Secure secret files
/opt/sutazaiapp/secrets_secure/
├── postgres_password.txt    (600 permissions)
├── redis_password.txt       (600 permissions)
├── neo4j_password.txt       (600 permissions)
└── jwt_secret.txt          (600 permissions)
```

### Vulnerability Scanning

#### Container Image Scanning
```bash
# Trivy security scanning
docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image sutazai-backend:latest

# Scan all running containers
docker images --format "table {{.Repository}}:{{.Tag}}" | \
  grep sutazai | xargs -I {} docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
  aquasec/trivy image {}
```

#### Regular Security Updates
```bash
# Update base images regularly
docker-compose pull
docker-compose up -d
```

### Compliance Frameworks

#### Security Standards
- CIS Docker Benchmark compliance
- OWASP Container Security verification
- NIST Cybersecurity Framework alignment

#### Audit Logging
All security events logged to centralized system:
```yaml
# Loki log aggregation
logging:
  driver: "json-file"
  options:
    labels: "service,security"
```

---

## Resource Management

### CPU Limits and Requests

#### Resource Allocation Strategy
```yaml
# High-priority services (guaranteed resources)
ollama:                  # 10 CPU limit, 4 CPU reserved
backend:                 # 4 CPU limit, 1 CPU reserved
hardware-resource-optimizer: # 2 CPU limit, 0.5 CPU reserved

# Medium-priority services  
postgres:                # 2 CPU limit, 0.5 CPU reserved
neo4j:                   # 1.5 CPU limit, 0.5 CPU reserved
qdrant:                  # 2 CPU limit, 0.5 CPU reserved

# Low-priority services
redis:                   # 0.5 CPU limit, 0.1 CPU reserved
grafana:                 # 1 CPU limit, 0.25 CPU reserved
prometheus:              # 1 CPU limit, 0.25 CPU reserved
```

### Memory Allocation

#### Memory Management
```yaml
# Memory limits and reservations
ollama:
  limits: 20G
  reservations: 8G

backend:
  limits: 4G  
  reservations: 1G

postgres:
  limits: 2G
  reservations: 512M

redis:
  limits: 512M
  reservations: 128M
  
# JVM-based services (Neo4j)
neo4j:
  limits: 1G
  reservations: 512M
  heap_max: 512m
  heap_initial: 256m
```

#### Memory Monitoring
```bash
# Monitor memory usage
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"

# Check for memory pressure
docker system df
docker system prune --volumes
```

### Disk Space Management

#### Storage Allocation
```bash
# Monitor disk usage
df -h /var/lib/docker/volumes/
docker system df -v

# Clean up unused resources
docker system prune -a --volumes
```

#### Volume Size Limits
```yaml
# Logical volume size limits (if using LVM)
postgres_data: 50GB
prometheus_data: 10GB  
grafana_data: 1GB
ollama_data: 100GB
models_data: 200GB
```

### Container Restart Policies

#### Restart Configuration
```yaml
# Restart policies by service type
core_services:
  restart: unless-stopped    # PostgreSQL, Redis, Neo4j

application_services:
  restart: unless-stopped    # Backend, Frontend, Ollama

monitoring_services:
  restart: unless-stopped    # Prometheus, Grafana, Loki

agent_services:
  restart: unless-stopped    # All agent containers

optional_services:
  restart: "no"              # Semgrep, one-time tools
```

#### Health Check Configuration
```yaml
# Health check intervals
backend:
  interval: 60s
  timeout: 30s
  retries: 5
  start_period: 120s

postgres:
  interval: 10s
  timeout: 5s  
  retries: 5
  start_period: 60s
```

### Resource Monitoring

#### System Resource Tracking
```bash
# Node Exporter metrics
http://localhost:10205/metrics

# cAdvisor container metrics  
http://localhost:10206/metrics

# Prometheus queries for resource usage
rate(container_cpu_usage_seconds_total[5m])
container_memory_usage_bytes / container_spec_memory_limit_bytes
```

---

## Monitoring Infrastructure

### Prometheus Configuration

#### Metrics Collection
```yaml
# Prometheus targets
scrape_configs:
  # Application metrics
  - job_name: 'backend'
    static_configs:
      - targets: ['backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  # System metrics
  - job_name: 'node-exporter'
    static_configs:
      - targets: ['node-exporter:9100']
    scrape_interval: 15s

  # Container metrics
  - job_name: 'cadvisor'
    static_configs:
      - targets: ['cadvisor:8080']
    scrape_interval: 30s

  # Database metrics
  - job_name: 'postgres-exporter'
    static_configs:
      - targets: ['postgres-exporter:9187']
    scrape_interval: 30s

  # Cache metrics
  - job_name: 'redis-exporter'
    static_configs:
      - targets: ['redis-exporter:9121']
    scrape_interval: 30s
```

#### Data Retention
```yaml
# Storage configuration
--storage.tsdb.retention.time=7d
--storage.tsdb.retention.size=1GB
--storage.tsdb.max-block-duration=2h
--storage.tsdb.min-block-duration=2h
```

### Grafana Dashboard Setup

#### Dashboard Categories
```bash
# Available dashboards
/monitoring/grafana/dashboards/
├── sutazai-system-overview.json      # System health overview
├── sutazai-performance.json          # Performance metrics
├── sutazai-database-metrics.json     # Database monitoring
├── sutazai-ollama-metrics.json       # LLM performance
├── sutazai-agent-performance.json    # Agent monitoring
└── sutazai-resource-utilization.json # Resource usage
```

#### Grafana Configuration
```bash
# Access Grafana
http://localhost:10201
Username: admin
Password: <from GRAFANA_PASSWORD env var>

# Import dashboards automatically
docker-compose exec grafana grafana-cli admin reset-admin-password <new_password>
```

### Loki Log Aggregation

#### Log Collection
```yaml
# Promtail configuration
clients:
  - url: http://loki:3100/loki/api/v1/push

scrape_configs:
  # Container logs
  - job_name: containers
    static_configs:
      - targets:
          - localhost
        labels:
          job: containerlogs
          __path__: /var/lib/docker/containers/*/*log

  # Application logs  
  - job_name: sutazai-logs
    static_configs:
      - targets:
          - localhost
        labels:
          job: sutazai
          __path__: /app/logs/*.log
```

#### Log Retention
```yaml
# Loki configuration
retention_deletes_enabled: true
retention_period: 168h  # 7 days
```

### AlertManager Rules

#### Critical Alerts
```yaml
# Production alert rules
groups:
  - name: sutazai-critical
    rules:
      # Service availability
      - alert: ServiceDown
        expr: up == 0
        for: 2m
        labels:
          severity: critical
        annotations:
          summary: "Service {{ $labels.instance }} is down"

      # High CPU usage
      - alert: HighCPUUsage
        expr: rate(container_cpu_usage_seconds_total[5m]) > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High CPU usage on {{ $labels.name }}"

      # High memory usage
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes / container_spec_memory_limit_bytes > 0.9
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "High memory usage on {{ $labels.name }}"

      # Disk space
      - alert: DiskSpaceWarning
        expr: (node_filesystem_size_bytes - node_filesystem_free_bytes) / node_filesystem_size_bytes > 0.8
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "Disk space usage high on {{ $labels.mountpoint }}"
```

#### Notification Channels
```yaml
# AlertManager configuration
receivers:
  - name: 'slack-alerts'
    slack_configs:
      - api_url: '${SLACK_WEBHOOK_URL}'
        channel: '#sutazai-alerts'
        title: 'SutazAI Alert'
        text: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'

  - name: 'pagerduty-critical'
    pagerduty_configs:
      - service_key: '${PAGERDUTY_SERVICE_KEY}'
        description: '{{ range .Alerts }}{{ .Annotations.summary }}{{ end }}'
```

### Health Check Endpoints

#### System Health Monitoring
```bash
# Backend health check
curl http://localhost:10010/health
# Returns: {"status": "degraded", "details": {...}}

# Individual service health
curl http://localhost:10104/api/tags  # Ollama models
curl http://localhost:10000/health    # PostgreSQL via proxy
curl http://localhost:10001/ping      # Redis ping
```

#### Blackbox Monitoring
```yaml
# External endpoint monitoring
modules:
  http_2xx:
    prober: http
    http:
      method: GET
      valid_status_codes: [200, 201, 202]
      
targets:
  - http://localhost:10010/health   # Backend API
  - http://localhost:10011/_stcore/health  # Streamlit
  - http://localhost:10201/api/health      # Grafana
```

---

## Scaling Considerations

### Horizontal Scaling Patterns

#### Stateless Services (Easily Scalable)
```yaml
# Services that can be horizontally scaled
backend:
  scale: 3                  # Multiple API instances
  load_balancer: kong       # Route distribution

frontend:  
  scale: 2                  # Multiple UI instances
  session_store: redis      # Shared session state

agents:
  scale: 5                  # Agent pool scaling
  task_queue: rabbitmq      # Work distribution
```

#### Stateful Services (Scaling Challenges)
```yaml
# Services requiring special scaling approaches
postgres:
  primary_replica: true     # Read replicas
  connection_pooling: pgbouncer
  
redis:
  cluster_mode: true        # Redis Cluster
  sentinel: true            # High availability

neo4j:
  causal_cluster: true      # Neo4j clustering (Enterprise)
  read_replicas: 2
```

### Vertical Scaling

#### Resource Optimization
```bash
# Current resource utilization
docker stats --no-stream

# Identify resource bottlenecks
container_cpu_usage_seconds_total
container_memory_usage_bytes
container_fs_usage_bytes
```

#### Scaling Guidelines
```yaml
# Scale up indicators
cpu_usage > 70% sustained      # Add CPU cores
memory_usage > 80%             # Add RAM  
disk_io_wait > 20%             # Add faster storage
network_rx/tx > 80%            # Add bandwidth

# Scale out indicators
queue_length > 100             # Add worker instances
response_time > 2s             # Add API instances
concurrent_users > 500         # Add frontend instances
```

### Auto-scaling Implementation

#### Container Orchestration Migration
```yaml
# Kubernetes deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sutazai-backend
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sutazai-backend
  template:
    spec:
      containers:
      - name: backend
        image: sutazai-backend:latest
        resources:
          requests:
            cpu: 1000m
            memory: 1Gi
          limits:
            cpu: 4000m
            memory: 4Gi
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: sutazai-backend-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sutazai-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
```

### Load Balancing

#### Kong Gateway Configuration
```yaml
# Kong services and routes
services:
  - name: backend-service
    url: http://backend:8000
    
routes:
  - name: api-route
    service: backend-service
    paths: ["/api"]
    
upstreams:
  - name: backend-upstream
    targets:
      - target: backend-1:8000
        weight: 100
      - target: backend-2:8000  
        weight: 100
      - target: backend-3:8000
        weight: 100
```

#### HAProxy Alternative
```bash
# HAProxy configuration for load balancing
backend sutazai_backend
    balance roundrobin
    server backend1 backend-1:8000 check
    server backend2 backend-2:8000 check
    server backend3 backend-3:8000 check
```

### Performance Optimization

#### Database Optimization
```sql
-- PostgreSQL optimization
VACUUM ANALYZE;
REINDEX DATABASE sutazai;

-- Connection pooling
max_connections = 200
shared_buffers = 512MB  
effective_cache_size = 2GB
```

#### Cache Optimization
```bash
# Redis optimization  
maxmemory-policy allkeys-lru
maxmemory 1gb
save ""                    # Disable persistence for cache
```

#### LLM Optimization
```bash
# Ollama performance tuning
OLLAMA_NUM_PARALLEL=50     # Concurrent requests
OLLAMA_NUM_THREADS=10      # CPU threads
OLLAMA_FLASH_ATTENTION=1   # Memory optimization
OLLAMA_MAX_LOADED_MODELS=3 # Model cache
```

---

## Disaster Recovery

### Backup Procedures

#### Automated Backup System
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/disaster-recovery-backup.sh

BACKUP_DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/opt/sutazaiapp/backups/dr_$BACKUP_DATE"
REMOTE_BACKUP="/mnt/remote-backup"

mkdir -p "$BACKUP_DIR"

echo "Starting disaster recovery backup: $BACKUP_DATE"

# 1. Database backups with compression
echo "Backing up PostgreSQL..."
docker exec sutazai-postgres pg_dump -U sutazai -Fc sutazai > "$BACKUP_DIR/postgres.dump"

echo "Backing up Redis..."  
docker exec sutazai-redis redis-cli --rdb > "$BACKUP_DIR/redis.rdb"

echo "Backing up Neo4j..."
docker exec sutazai-neo4j neo4j-admin dump --database=sutazai --to="/backups/neo4j_$BACKUP_DATE.dump"
docker cp sutazai-neo4j:/backups/neo4j_$BACKUP_DATE.dump "$BACKUP_DIR/"

# 2. Vector database snapshots
echo "Backing up vector databases..."
docker exec sutazai-qdrant curl -X POST http://localhost:6333/snapshots
docker exec sutazai-chromadb tar -czf "/tmp/chromadb_$BACKUP_DATE.tar.gz" -C /chroma chroma
docker cp sutazai-chromadb:/tmp/chromadb_$BACKUP_DATE.tar.gz "$BACKUP_DIR/"

# 3. Application data and configuration
echo "Backing up application data..."
tar -czf "$BACKUP_DIR/configs.tar.gz" /opt/sutazaiapp/configs/
tar -czf "$BACKUP_DIR/secrets.tar.gz" /opt/sutazaiapp/secrets_secure/
tar -czf "$BACKUP_DIR/logs.tar.gz" /opt/sutazaiapp/logs/

# 4. Docker volumes
echo "Backing up Docker volumes..."
docker run --rm -v ollama_data:/data -v "$BACKUP_DIR":/backup alpine tar -czf /backup/ollama_data.tar.gz -C /data .
docker run --rm -v models_data:/data -v "$BACKUP_DIR":/backup alpine tar -czf /backup/models_data.tar.gz -C /data .
docker run --rm -v grafana_data:/data -v "$BACKUP_DIR":/backup alpine tar -czf /backup/grafana_data.tar.gz -C /data .

# 5. System configuration
echo "Backing up system configuration..."
cp /opt/sutazaiapp/docker-compose.yml "$BACKUP_DIR/"
cp /opt/sutazaiapp/.env* "$BACKUP_DIR/"

# 6. Create manifest
echo "Creating backup manifest..."
cat > "$BACKUP_DIR/manifest.txt" << EOF
Backup Date: $BACKUP_DATE
System: SutazAI Infrastructure
Components:
- PostgreSQL database dump
- Redis data snapshot  
- Neo4j database dump
- ChromaDB vector data
- Qdrant vector snapshots
- Application configurations
- Docker volumes
- System configuration files

Verification Commands:
pg_restore --list postgres.dump
redis-cli --latency-history -h localhost
neo4j-admin load --database=sutazai --from=neo4j_$BACKUP_DATE.dump
EOF

# 7. Sync to remote backup location
if [ -d "$REMOTE_BACKUP" ]; then
    echo "Syncing to remote backup location..."
    rsync -av "$BACKUP_DIR/" "$REMOTE_BACKUP/dr_$BACKUP_DATE/"
fi

# 8. Cleanup old backups (keep 7 days locally, 30 days remote)
find /opt/sutazaiapp/backups/ -name "dr_*" -type d -mtime +7 -exec rm -rf {} \;
find "$REMOTE_BACKUP/" -name "dr_*" -type d -mtime +30 -exec rm -rf {} \; 2>/dev/null

echo "Disaster recovery backup completed: $BACKUP_DIR"
```

#### Backup Schedule
```bash
# Crontab configuration
# Full system backup daily at 2 AM
0 2 * * * /opt/sutazaiapp/scripts/disaster-recovery-backup.sh

# Quick application backup every 4 hours
0 */4 * * * /opt/sutazaiapp/scripts/quick-backup.sh

# Weekly validation of backup integrity
0 3 * * 0 /opt/sutazaiapp/scripts/validate-backups.sh
```

### Recovery Procedures

#### Complete System Recovery
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/disaster-recovery-restore.sh

BACKUP_DIR="$1"
if [ -z "$BACKUP_DIR" ]; then
    echo "Usage: $0 <backup_directory>"
    exit 1
fi

echo "Starting disaster recovery from: $BACKUP_DIR"

# 1. Verify backup integrity
echo "Verifying backup integrity..."
if [ ! -f "$BACKUP_DIR/manifest.txt" ]; then
    echo "ERROR: Backup manifest not found!"
    exit 1
fi

# 2. Stop all services
echo "Stopping all services..."
docker-compose down

# 3. Clean existing data (DESTRUCTIVE - use with caution)
echo "WARNING: This will destroy existing data. Continue? (yes/no)"
read confirm
if [ "$confirm" != "yes" ]; then
    echo "Recovery cancelled"
    exit 1
fi

docker volume rm $(docker volume ls -q | grep sutazai) 2>/dev/null || true

# 4. Restore configuration
echo "Restoring configuration files..."
cp "$BACKUP_DIR/docker-compose.yml" /opt/sutazaiapp/
cp "$BACKUP_DIR/.env"* /opt/sutazaiapp/
tar -xzf "$BACKUP_DIR/configs.tar.gz" -C /
tar -xzf "$BACKUP_DIR/secrets.tar.gz" -C /

# 5. Start core services
echo "Starting core services..."
docker-compose up -d postgres redis neo4j

# Wait for services to be ready
sleep 30

# 6. Restore databases
echo "Restoring PostgreSQL..."
cat "$BACKUP_DIR/postgres.dump" | docker exec -i sutazai-postgres pg_restore -U sutazai -d sutazai

echo "Restoring Redis..."
docker cp "$BACKUP_DIR/redis.rdb" sutazai-redis:/data/dump.rdb
docker-compose restart redis

echo "Restoring Neo4j..."
docker exec sutazai-neo4j neo4j-admin load --database=sutazai --from="/backups/$(basename $BACKUP_DIR/neo4j_*.dump)"

# 7. Restore vector databases
echo "Restoring vector databases..."
docker-compose up -d chromadb qdrant

# Wait for services
sleep 30

# Restore ChromaDB
docker cp "$BACKUP_DIR/chromadb_*.tar.gz" sutazai-chromadb:/tmp/
docker exec sutazai-chromadb sh -c "cd /chroma && tar -xzf /tmp/chromadb_*.tar.gz"

# Restore Qdrant (snapshots restored automatically on startup)

# 8. Restore Docker volumes
echo "Restoring Docker volumes..."
docker run --rm -v ollama_data:/data -v "$BACKUP_DIR":/backup alpine tar -xzf /backup/ollama_data.tar.gz -C /data
docker run --rm -v models_data:/data -v "$BACKUP_DIR":/backup alpine tar -xzf /backup/models_data.tar.gz -C /data  
docker run --rm -v grafana_data:/data -v "$BACKUP_DIR":/backup alpine tar -xzf /backup/grafana_data.tar.gz -C /data

# 9. Start all services
echo "Starting all services..."
docker-compose up -d

# 10. Verify recovery
echo "Verifying system recovery..."
sleep 60

# Check service health
./scripts/health-check.sh

echo "Disaster recovery completed from: $BACKUP_DIR"
echo "Please verify all services are functioning correctly"
```

### High Availability Setup

#### Service Redundancy
```yaml
# Multi-instance deployment
version: '3.8'
services:
  # Load balanced backend
  backend-1:
    <<: *backend-service
    container_name: sutazai-backend-1
  
  backend-2:
    <<: *backend-service  
    container_name: sutazai-backend-2
    
  backend-3:
    <<: *backend-service
    container_name: sutazai-backend-3

  # Load balancer
  haproxy:
    image: haproxy:2.4
    ports:
      - "10010:80"
    volumes:
      - ./config/haproxy.cfg:/usr/local/etc/haproxy/haproxy.cfg
    depends_on:
      - backend-1
      - backend-2  
      - backend-3
```

#### Database High Availability
```yaml
# PostgreSQL with read replicas
postgres-primary:
  image: postgres:16.3-alpine
  environment:
    POSTGRES_REPLICATION_MODE: master
    POSTGRES_REPLICATION_USER: replicator
    POSTGRES_REPLICATION_PASSWORD: ${REPLICATION_PASSWORD}

postgres-replica:
  image: postgres:16.3-alpine  
  environment:
    POSTGRES_REPLICATION_MODE: slave
    POSTGRES_MASTER_SERVICE: postgres-primary
    POSTGRES_REPLICATION_USER: replicator
    POSTGRES_REPLICATION_PASSWORD: ${REPLICATION_PASSWORD}
```

### Failover Strategies

#### Automated Failover Script
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/automated-failover.sh

HEALTH_CHECK_URL="http://localhost:10010/health"
FAILOVER_TRIGGERED="/tmp/sutazai-failover"

while true; do
    # Check primary system health
    if ! curl -f "$HEALTH_CHECK_URL" >/dev/null 2>&1; then
        if [ ! -f "$FAILOVER_TRIGGERED" ]; then
            echo "Primary system unhealthy, triggering failover..."
            
            # Mark failover as triggered
            touch "$FAILOVER_TRIGGERED"
            
            # Stop unhealthy services
            docker-compose stop backend frontend
            
            # Start backup instances
            docker-compose up -d backend-backup frontend-backup
            
            # Update load balancer configuration
            ./scripts/update-lb-config.sh backup
            
            # Send alert
            curl -X POST -H 'Content-type: application/json' \
                --data '{"text":"SutazAI failover triggered - backup systems active"}' \
                "$SLACK_WEBHOOK_URL"
        fi
    else
        if [ -f "$FAILOVER_TRIGGERED" ]; then
            echo "Primary system healthy, ending failover..."
            
            # Remove failover marker
            rm "$FAILOVER_TRIGGERED"
            
            # Restore primary services
            docker-compose up -d backend frontend
            docker-compose stop backend-backup frontend-backup
            
            # Update load balancer
            ./scripts/update-lb-config.sh primary
        fi
    fi
    
    sleep 30
done
```

---

## Troubleshooting

### Common Infrastructure Issues

#### 1. Container Restart Loops

**Symptoms:**
```bash
docker ps | grep "Restarting"
docker logs sutazai-backend | tail -20
```

**Common Causes & Solutions:**

```bash
# Issue: Out of Memory
# Solution: Check memory limits and usage
docker stats --no-stream | grep sutazai
# Increase memory limits in docker-compose.yml

# Issue: Port conflicts  
# Solution: Check port bindings
netstat -tulpn | grep :10010
# Update port mappings if conflicts exist

# Issue: Missing environment variables
# Solution: Verify .env files
docker-compose config | grep -A 5 -B 5 "environment:"
# Add missing variables to .env files

# Issue: Volume mount failures
# Solution: Check permissions
ls -la /var/lib/docker/volumes/
sudo chown -R $USER:docker /opt/sutazaiapp/
```

#### 2. Database Connection Issues

**PostgreSQL Connection Problems:**
```bash
# Check PostgreSQL health
docker exec sutazai-postgres pg_isready -U sutazai

# Verify connection from backend
docker exec sutazai-backend python -c "
import psycopg2
conn = psycopg2.connect('postgresql://sutazai:password@postgres:5432/sutazai')
print('Connection successful')
conn.close()
"

# Check PostgreSQL logs
docker logs sutazai-postgres | tail -50

# Common fixes
docker-compose restart postgres
docker exec sutazai-postgres createdb -U sutazai sutazai  # If DB doesn't exist
```

**Redis Connection Problems:**
```bash
# Test Redis connectivity
docker exec sutazai-redis redis-cli ping

# Check from backend
docker exec sutazai-backend python -c "
import redis
r = redis.Redis(host='redis', port=6379)
print(r.ping())
"

# Clear Redis if corrupted
docker exec sutazai-redis redis-cli FLUSHALL
```

**Neo4j Connection Problems:**
```bash
# Check Neo4j browser access
curl http://localhost:10002/browser/

# Test Bolt connection
docker exec sutazai-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "MATCH (n) RETURN count(n);"

# Common Neo4j issues
# - Insufficient memory: Increase heap size
# - Authentication failure: Reset password
docker exec sutazai-neo4j cypher-shell -u neo4j -p "$NEO4J_PASSWORD" "CALL dbms.security.changePassword('newpassword');"
```

#### 3. ChromaDB Connection Issues (Known Issue)

**Current Status:** ChromaDB has known connection issues

```bash
# Check ChromaDB logs
docker logs sutazai-chromadb | tail -50

# Restart ChromaDB
docker-compose restart chromadb

# Verify ChromaDB API
curl -f http://localhost:10100/api/v1/heartbeat

# Temporary workaround: Use Qdrant instead
# Update application configuration to use Qdrant as primary vector DB
```

#### 4. Ollama Model Issues

**TinyLlama vs gpt-oss Mismatch:**
```bash
# Check loaded models
curl http://localhost:10104/api/tags

# Expected: TinyLlama (current)
# Application expects: gpt-oss

# Solutions:
# Option 1: Load gpt-oss model
docker exec sutazai-ollama ollama pull gpt-oss

# Option 2: Update application to use TinyLlama
# Edit backend configuration to use "tinyllama:latest"
```

#### 5. Network Connectivity Issues

**Service Discovery Problems:**
```bash
# Test internal network connectivity
docker exec sutazai-backend ping -c 3 postgres
docker exec sutazai-backend ping -c 3 redis
docker exec sutazai-backend ping -c 3 ollama

# Check network configuration
docker network inspect sutazai-network

# Recreate network if needed
docker network rm sutazai-network
docker network create sutazai-network
docker-compose up -d
```

**Port Binding Conflicts:**
```bash
# Find conflicting processes
sudo netstat -tulpn | grep :10010
sudo lsof -i :10010

# Kill conflicting processes
sudo kill $(sudo lsof -t -i:10010)

# Or change port mapping in docker-compose.yml
```

### Container Debugging

#### Log Analysis
```bash
# View logs for specific service
docker logs sutazai-backend --tail 100 --follow

# View logs with timestamps
docker logs sutazai-backend --timestamps

# Search logs for specific errors
docker logs sutazai-backend 2>&1 | grep -i error

# Export logs for analysis
docker logs sutazai-backend > /tmp/backend-logs-$(date +%Y%m%d).txt
```

#### Interactive Debugging
```bash
# Execute shell in running container
docker exec -it sutazai-backend /bin/bash

# Start stopped container for debugging
docker run -it --rm --entrypoint /bin/bash sutazai-backend

# Check filesystem inside container
docker exec sutazai-backend ls -la /app/
docker exec sutazai-backend ps aux
docker exec sutazai-backend netstat -tulpn
```

#### Resource Debugging
```bash
# Real-time resource usage
docker stats

# Detailed container inspection
docker inspect sutazai-backend | jq '.[]'

# Check container limits
docker inspect sutazai-backend | jq '.[].HostConfig.Memory'
docker inspect sutazai-backend | jq '.[].HostConfig.CpuQuota'
```

### Performance Troubleshooting

#### Slow Response Times
```bash
# Check Grafana dashboards
http://localhost:10201/d/sutazai-performance

# Manual performance testing
curl -w "@curl-format.txt" -o /dev/null -s http://localhost:10010/health

# Create curl-format.txt:
cat > curl-format.txt << 'EOF'
     time_namelookup:  %{time_namelookup}\n
        time_connect:  %{time_connect}\n
     time_appconnect:  %{time_appconnect}\n
    time_pretransfer:  %{time_pretransfer}\n
       time_redirect:  %{time_redirect}\n
  time_starttransfer:  %{time_starttransfer}\n
                     ----------\n
          time_total:  %{time_total}\n
EOF
```

#### High Resource Usage
```bash
# Identify resource-heavy containers
docker stats --no-stream | sort -k3 -hr  # Sort by CPU
docker stats --no-stream | sort -k4 -hr  # Sort by Memory

# Check for memory leaks
docker exec sutazai-backend python -c "
import psutil
process = psutil.Process()
print(f'Memory: {process.memory_info().rss / 1024 / 1024:.2f} MB')
print(f'CPU: {process.cpu_percent()}%')
"
```

### Monitoring Troubleshooting

#### Prometheus Issues
```bash
# Check Prometheus targets
curl http://localhost:10200/api/v1/targets

# Verify Prometheus configuration  
docker exec sutazai-prometheus promtool check config /etc/prometheus/prometheus.yml

# Restart Prometheus
docker-compose restart prometheus
```

#### Grafana Issues
```bash
# Reset admin password
docker exec sutazai-grafana grafana-cli admin reset-admin-password newpassword

# Import dashboards manually
curl -X POST \
  http://admin:password@localhost:10201/api/dashboards/db \
  -H 'Content-Type: application/json' \
  -d @monitoring/grafana/dashboards/sutazai-system-overview.json
```

### Recovery Commands

#### Emergency System Recovery
```bash
#!/bin/bash
# Emergency recovery script

echo "Starting emergency recovery..."

# 1. Stop all services
docker-compose down

# 2. Clean up problematic containers
docker container prune -f
docker volume prune -f  # CAUTION: This removes unused volumes

# 3. Restart core services only
docker-compose up -d postgres redis neo4j ollama

# 4. Wait for databases to be ready
sleep 30

# 5. Restart application services
docker-compose up -d backend frontend

# 6. Restart monitoring
docker-compose up -d prometheus grafana loki

# 7. Check system health
sleep 60
./scripts/health-check.sh

echo "Emergency recovery completed"
```

#### Data Recovery Commands
```bash
# Recover from backup
./scripts/disaster-recovery-restore.sh /opt/sutazaiapp/backups/latest/

# Reset specific service data
docker-compose stop postgres
docker volume rm postgres_data
docker-compose up -d postgres

# Import SQL backup
cat backup.sql | docker exec -i sutazai-postgres psql -U sutazai -d sutazai
```

---

## Migration Paths

### Docker Compose to Kubernetes

The current Docker Compose setup can be migrated to Kubernetes for better scalability and management. Here's a migration strategy:

#### Phase 1: Containerization Audit
```bash
# Analyze current containers
docker images | grep sutazai
docker-compose ps --services

# Export configurations
docker inspect $(docker-compose ps -q) > containers-config.json
```

#### Phase 2: Kubernetes Manifests

**Namespace:**
```yaml
# namespace.yaml
apiVersion: v1
kind: Namespace
metadata:
  name: sutazai-production
  labels:
    name: sutazai-production
    environment: production
```

**ConfigMaps:**
```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: sutazai-config
  namespace: sutazai-production
data:
  SUTAZAI_ENV: "production"
  TZ: "UTC"
  POSTGRES_DB: "sutazai"
  POSTGRES_USER: "sutazai"
  REDIS_URL: "redis://redis-service:6379/0"
  NEO4J_URI: "bolt://neo4j-service:7687"
  OLLAMA_BASE_URL: "http://ollama-service:11434"
```

**Secrets:**
```yaml
# secrets.yaml
apiVersion: v1
kind: Secret
metadata:
  name: sutazai-secrets
  namespace: sutazai-production
type: Opaque
data:
  postgres-password: <base64_encoded_password>
  neo4j-password: <base64_encoded_password>
  jwt-secret: <base64_encoded_secret>
  grafana-password: <base64_encoded_password>
```

**Persistent Volumes:**
```yaml
# storage.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: postgres-pvc
  namespace: sutazai-production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 50Gi
  storageClassName: fast-ssd
---
apiVersion: v1  
kind: PersistentVolumeClaim
metadata:
  name: ollama-models-pvc
  namespace: sutazai-production
spec:
  accessModes:
    - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
```

**Core Services:**
```yaml
# postgres-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: postgres
  namespace: sutazai-production
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
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          valueFrom:
            configMapKeyRef:
              name: sutazai-config
              key: POSTGRES_DB
        - name: POSTGRES_USER  
          valueFrom:
            configMapKeyRef:
              name: sutazai-config
              key: POSTGRES_USER
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: sutazai-secrets
              key: postgres-password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
        resources:
          requests:
            cpu: 500m
            memory: 512Mi
          limits:
            cpu: 2000m
            memory: 2Gi
      volumes:
      - name: postgres-storage
        persistentVolumeClaim:
          claimName: postgres-pvc
---
apiVersion: v1
kind: Service  
metadata:
  name: postgres-service
  namespace: sutazai-production
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  type: ClusterIP
```

**Backend Application:**
```yaml
# backend-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sutazai-backend
  namespace: sutazai-production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: sutazai-backend
  template:
    metadata:
      labels:
        app: sutazai-backend
    spec:
      containers:
      - name: backend
        image: sutazai-backend:latest
        ports:
        - containerPort: 8000
        envFrom:
        - configMapRef:
            name: sutazai-config
        - secretRef:
            name: sutazai-secrets
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            cpu: 1000m
            memory: 1Gi
          limits:
            cpu: 4000m
            memory: 4Gi
---
apiVersion: v1
kind: Service
metadata:
  name: backend-service
  namespace: sutazai-production
spec:
  selector:
    app: sutazai-backend
  ports:
  - port: 8000
    targetPort: 8000
  type: ClusterIP
---
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: backend-hpa
  namespace: sutazai-production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: sutazai-backend
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
```

**Ingress Controller:**
```yaml
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: sutazai-ingress
  namespace: sutazai-production
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
    nginx.ingress.kubernetes.io/rate-limit: "100"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
spec:
  tls:
  - hosts:
    - api.sutazai.com
    - app.sutazai.com
    secretName: sutazai-tls
  rules:
  - host: api.sutazai.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: backend-service
            port:
              number: 8000
  - host: app.sutazai.com  
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: frontend-service
            port:
              number: 8501
```

#### Phase 3: Migration Process

**Migration Script:**
```bash
#!/bin/bash
# migrate-to-kubernetes.sh

echo "Starting Kubernetes migration..."

# 1. Export current data
./scripts/disaster-recovery-backup.sh

# 2. Build and push container images
docker build -t registry.company.com/sutazai-backend:v1.0.0 ./backend/
docker build -t registry.company.com/sutazai-frontend:v1.0.0 ./frontend/

docker push registry.company.com/sutazai-backend:v1.0.0
docker push registry.company.com/sutazai-frontend:v1.0.0

# 3. Create Kubernetes resources
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/secrets.yaml
kubectl apply -f k8s/storage.yaml

# 4. Deploy core services
kubectl apply -f k8s/postgres-deployment.yaml
kubectl apply -f k8s/redis-deployment.yaml
kubectl apply -f k8s/neo4j-deployment.yaml

# Wait for databases to be ready
kubectl wait --for=condition=ready pod -l app=postgres -n sutazai-production --timeout=300s

# 5. Restore data to Kubernetes
kubectl exec -n sutazai-production deployment/postgres -- pg_restore -U sutazai -d sutazai < backup/postgres.dump

# 6. Deploy application services
kubectl apply -f k8s/backend-deployment.yaml
kubectl apply -f k8s/frontend-deployment.yaml

# 7. Deploy monitoring
kubectl apply -f k8s/prometheus-deployment.yaml
kubectl apply -f k8s/grafana-deployment.yaml

# 8. Configure ingress
kubectl apply -f k8s/ingress.yaml

# 9. Verify migration
kubectl get pods -n sutazai-production
kubectl get services -n sutazai-production

echo "Kubernetes migration completed"
```

#### Phase 4: Monitoring Migration

**Prometheus Operator:**
```yaml
# monitoring.yaml
apiVersion: monitoring.coreos.com/v1
kind: Prometheus
metadata:
  name: sutazai-prometheus
  namespace: sutazai-production
spec:
  replicas: 1
  retention: "7d"
  storage:
    volumeClaimTemplate:
      spec:
        accessModes: ["ReadWriteOnce"]
        resources:
          requests:
            storage: 10Gi
  serviceMonitorSelector:
    matchLabels:
      app: sutazai
---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: sutazai-backend-monitor
  namespace: sutazai-production
  labels:
    app: sutazai
spec:
  selector:
    matchLabels:
      app: sutazai-backend
  endpoints:
  - port: metrics
    path: /metrics
    interval: 30s
```

### Cloud Migration Options

#### AWS EKS Deployment
```yaml
# aws-eks-cluster.yaml
apiVersion: eksctl.io/v1alpha5
kind: ClusterConfig

metadata:
  name: sutazai-production
  region: us-west-2
  version: "1.28"

managedNodeGroups:
  - name: sutazai-workers
    instanceType: m5.xlarge
    desiredCapacity: 3
    minSize: 2
    maxSize: 10
    volumeSize: 100
    ssh:
      enableSsm: true

addons:
  - name: vpc-cni
  - name: coredns
  - name: kube-proxy
  - name: aws-ebs-csi-driver

iam:
  withOIDC: true
```

#### Google GKE Deployment
```bash
# Create GKE cluster
gcloud container clusters create sutazai-production \
    --num-nodes=3 \
    --machine-type=n1-standard-4 \
    --disk-size=100GB \
    --enable-autorepair \
    --enable-autoupgrade \
    --enable-autoscaling \
    --max-nodes=10 \
    --min-nodes=2 \
    --zone=us-central1-a
```

#### Azure AKS Deployment
```bash
# Create AKS cluster
az aks create \
    --resource-group sutazai-rg \
    --name sutazai-production \
    --node-count 3 \
    --node-vm-size Standard_D4s_v3 \
    --generate-ssh-keys \
    --enable-cluster-autoscaler \
    --min-count 2 \
    --max-count 10
```

### Hybrid Cloud Strategy

#### Multi-Cloud Deployment
```yaml
# Use GitOps with ArgoCD for multi-cloud deployment
apiVersion: argoproj.io/v1alpha1
kind: Application
metadata:
  name: sutazai-production
  namespace: argocd
spec:
  project: default
  source:
    repoURL: https://github.com/company/sutazai-k8s-manifests
    targetRevision: main
    path: overlays/production
  destination:
    server: https://kubernetes.default.svc
    namespace: sutazai-production
  syncPolicy:
    automated:
      prune: true
      selfHeal: true
    syncOptions:
    - CreateNamespace=true
```

This comprehensive infrastructure setup guide provides the foundation for deploying, managing, and scaling the SutazAI system. The guide is based on the actual system state documented in CLAUDE.md and provides practical, tested procedures for infrastructure management.

---

**Next Steps:**
1. Review and customize configurations for your specific environment
2. Test deployment procedures in a staging environment
3. Implement monitoring and alerting
4. Plan disaster recovery tests
5. Consider migration to container orchestration platforms

**Maintenance:**
- Review and update this guide monthly
- Validate backup and recovery procedures quarterly
- Update security configurations based on latest best practices
- Monitor resource usage and optimize as needed