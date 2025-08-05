# SutazAI Production Deployment Guide

**Version:** 4.0.0  
**Last Updated:** January 8, 2025  
**Target Audience:** System operators and DevOps engineers  

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Prerequisites](#prerequisites)
4. [Pre-Deployment Checklist](#pre-deployment-checklist)
5. [Step-by-Step Deployment](#step-by-step-deployment)
6. [Configuration Management](#configuration-management)
7. [Service Startup Sequence](#service-startup-sequence)
8. [Health Validation](#health-validation)
9. [Common Issues & Troubleshooting](#common-issues--troubleshooting)
10. [Rollback Procedures](#rollback-procedures)
11. [Post-Deployment Checklist](#post-deployment-checklist)
12. [Monitoring & Maintenance](#monitoring--maintenance)
13. [Security Considerations](#security-considerations)
14. [Performance Tuning](#performance-tuning)
15. [Appendices](#appendices)

---

## Overview

SutazAI is a comprehensive AI automation platform featuring 69 specialized AI agents, vector databases, monitoring systems, and distributed AI model serving capabilities. This guide provides step-by-step instructions for deploying SutazAI in production environments.

### Key Components
- **Core Infrastructure**: PostgreSQL, Redis, Neo4j
- **Vector Databases**: ChromaDB, Qdrant, FAISS
- **AI Models**: Ollama with TinyLlama, Qwen2.5, and embedding models
- **Application Layer**: FastAPI backend, Streamlit frontend
- **AI Agents**: 69 specialized agents for various automation tasks
- **Monitoring Stack**: Prometheus, Grafana, Loki, AlertManager
- **External Integrations**: LangFlow, FlowiseAI, Dify, n8n

---

## System Requirements

### Minimum Requirements

| Component | Specification |
|-----------|---------------|
| **CPU** | 4 cores (x86_64) |
| **Memory** | 16 GB RAM |
| **Storage** | 100 GB available disk space |
| **Network** | Stable internet connection (for model downloads) |
| **OS** | Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+) |

### Recommended Requirements

| Component | Specification |
|-----------|---------------|
| **CPU** | 8+ cores (x86_64) |
| **Memory** | 32 GB RAM |
| **Storage** | 500 GB SSD |
| **Network** | High-bandwidth connection |
| **GPU** | NVIDIA GPU with 8GB+ VRAM (optional) |

### Port Requirements

The following ports must be available:

| Service | Port | Protocol | Access |
|---------|------|----------|--------|
| Frontend | 8501 | HTTP | External |
| Backend API | 8000 | HTTP | External |
| PostgreSQL | 5432 | TCP | Internal |
| Redis | 6379 | TCP | Internal |
| Neo4j | 7474, 7687 | HTTP/Bolt | Internal |
| Ollama API | 11434 | HTTP | Internal |
| ChromaDB | 8001 | HTTP | Internal |
| Qdrant | 6333 | HTTP | Internal |
| Prometheus | 9090 | HTTP | Internal |
| Grafana | 3000 | HTTP | External |
| AlertManager | 9093 | HTTP | Internal |

---

## Prerequisites

### Required Software

1. **Docker Engine** (v24.0+)
   ```bash
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   sudo usermod -aG docker $USER
   ```

2. **Docker Compose** (v2.0+)
   ```bash
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

3. **System Tools**
   ```bash
   # Ubuntu/Debian
   sudo apt update && sudo apt install -y curl wget git jq htop vim

   # CentOS/RHEL/Fedora
   sudo yum install -y curl wget git jq htop vim
   ```

### System Configuration

1. **Increase File Limits**
   ```bash
   echo "* soft nofile 65536" | sudo tee -a /etc/security/limits.conf
   echo "* hard nofile 65536" | sudo tee -a /etc/security/limits.conf
   ```

2. **Configure Kernel Parameters**
   ```bash
   echo "net.core.somaxconn = 65535" | sudo tee -a /etc/sysctl.conf
   echo "vm.max_map_count = 262144" | sudo tee -a /etc/sysctl.conf
   sudo sysctl -p
   ```

3. **Create Application Directory**
   ```bash
   sudo mkdir -p /opt/sutazaiapp
   sudo chown $USER:$USER /opt/sutazaiapp
   cd /opt/sutazaiapp
   ```

---

## Pre-Deployment Checklist

### Environment Preparation

- [ ] Verify system meets minimum requirements
- [ ] Confirm all required ports are available
- [ ] Install Docker and Docker Compose
- [ ] Create application directory structure
- [ ] Configure system limits and kernel parameters
- [ ] Verify internet connectivity for model downloads

### Security Preparation

- [ ] Generate SSL certificates (self-signed or CA-issued)
- [ ] Prepare secure passwords for databases
- [ ] Configure firewall rules (if applicable)
- [ ] Set up log rotation policies
- [ ] Review security compliance requirements

### Resource Planning

- [ ] Plan resource allocation for AI models
- [ ] Determine monitoring retention policies
- [ ] Configure backup storage locations
- [ ] Plan for horizontal scaling (if needed)

---

## Step-by-Step Deployment

### Step 1: Initial Setup and Validation

1. **Clone or extract SutazAI to `/opt/sutazaiapp`**

2. **Validate deployment system**:
   ```bash
   cd /opt/sutazaiapp
   chmod +x validate_deployment.sh
   ./validate_deployment.sh
   ```

   **Expected Output**:
   ```
   SutazAI Deployment System Validation
   ====================================

   [SUCCESS] Found: deploy.sh
   [SUCCESS] Found: docker-compose.yml
   [SUCCESS] All required files present
   [SUCCESS] Executable: deploy.sh
   [SUCCESS] Valid: docker-compose.yml
   🎉 All validation checks passed!
   ```

### Step 2: System Detection and Dependency Installation

3. **Run system detection**:
   ```bash
   ./deploy.sh deploy local
   ```

   **Expected Behavior**:
   - Automatically detects system capabilities
   - Installs missing dependencies
   - Validates minimum requirements
   - Creates necessary directory structure

   **Sample Output**:
   ```
   ═══════════════════════════════════════════════════
   🚀 PHASE: SYSTEM_DETECTION
   ═══════════════════════════════════════════════════

   [INFO] System detected: 8 cores, 32GB RAM, 500GB disk
   [INFO] Container runtime: docker
   [INFO] GPU available: true
   [INFO] Internet connectivity: true
   [SUCCESS] System requirements validation completed
   ```

### Step 3: Security Configuration

4. **Security setup (automatic)**:
   The deployment script automatically:
   - Generates secure passwords for all databases
   - Creates SSL certificates
   - Sets proper file permissions
   - Configures firewall rules (production only)

   **Manual Security Review**:
   ```bash
   # Review generated secrets
   ls -la /opt/sutazaiapp/secrets/
   
   # Expected files:
   # postgres_password.txt
   # redis_password.txt  
   # neo4j_password.txt
   # jwt_secret.txt
   # grafana_password.txt
   ```

### Step 4: Environment Configuration

5. **Environment variables setup**:
   ```bash
   # Review the generated .env file
   cat /opt/sutazaiapp/.env
   ```

   **Key Configuration Areas**:
   - Database connections
   - Security tokens
   - Resource limits
   - Feature flags

### Step 5: Infrastructure Deployment

6. **Deploy core infrastructure**:
   ```bash
   # The deployment continues automatically
   # Monitor progress through the logs
   tail -f /opt/sutazaiapp/logs/deployment_deploy_$(date +%Y%m%d)_*.log
   ```

   **Infrastructure Services Startup Order**:
   1. PostgreSQL (with health checks)
   2. Redis (with connectivity validation)
   3. Neo4j (with startup verification)
   4. ChromaDB (vector database)
   5. Qdrant (vector search engine)
   6. FAISS (similarity search)
   7. Ollama (AI model server)

### Step 6: Application Services Deployment

7. **Deploy application layer**:
   The deployment automatically proceeds to:
   - Build custom Docker images
   - Start backend API service
   - Start frontend application
   - Initialize database schemas

   **Health Check Validation**:
   ```bash
   # Check service health during deployment
   curl -f http://localhost:8000/health
   curl -f http://localhost:8501/healthz
   ```

### Step 7: AI Agent Deployment

8. **Deploy AI agents**:
   The system deploys 69 specialized AI agents:
   - Development agents (senior engineers, architects)
   - Testing and QA agents
   - Infrastructure and DevOps agents
   - Security and compliance agents
   - Business analysis agents

### Step 8: Monitoring Stack Deployment

9. **Deploy monitoring services**:
   - Prometheus (metrics collection)
   - Grafana (visualization dashboards) 
   - Loki (log aggregation)
   - AlertManager (alert routing)
   - Health monitors

---

## Configuration Management

### Core Configuration Files

| File | Purpose | Critical Settings |
|------|---------|-------------------|
| `.env` | Environment variables | Database passwords, API keys |
| `docker-compose.yml` | Service definitions | Resource limits, port mappings |
| `secrets/` | Sensitive data | Database passwords, certificates |
| `config/ollama.yaml` | AI model configuration | Model parameters, performance tuning |
| `monitoring/prometheus/prometheus.yml` | Monitoring config | Scrape targets, retention |

### Environment Variables

**Critical Variables**:
```bash
# Database Configuration
POSTGRES_PASSWORD=<auto-generated>
REDIS_PASSWORD=<auto-generated>
NEO4J_PASSWORD=<auto-generated>

# Security
JWT_SECRET=<auto-generated>
SECRET_KEY=<auto-generated>

# Performance Tuning
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=2
MAX_WORKERS=4
CONNECTION_POOL_SIZE=20

# Feature Flags
ENABLE_GPU=auto
ENABLE_MONITORING=true
ENABLE_LOGGING=true
```

### Resource Allocation

**Service Resource Limits**:
```yaml
# Database services
postgres: 2 CPU cores, 2GB RAM
redis: 1 CPU core, 1GB RAM
neo4j: 3 CPU cores, 4GB RAM

# AI services
ollama: 10 CPU cores, 20GB RAM
backend: 4 CPU cores, 4GB RAM
frontend: 1 CPU core, 1GB RAM

# Vector databases
chromadb: 2 CPU cores, 2GB RAM
qdrant: 2 CPU cores, 2GB RAM
```

---

## Service Startup Sequence

### Phase 1: Infrastructure Services (0-120 seconds)

1. **PostgreSQL** 
   - Health check: `pg_isready -U sutazai`
   - Expected: Service healthy within 30 seconds
   
2. **Redis**
   - Health check: `redis-cli ping`
   - Expected: Service healthy within 10 seconds

3. **Neo4j**
   - Health check: HTTP endpoint availability
   - Expected: Service healthy within 60 seconds

### Phase 2: Vector Databases (120-240 seconds)

4. **ChromaDB**
   - Health check: `GET /api/v1/heartbeat`
   - Expected: Service healthy within 30 seconds

5. **Qdrant**
   - Health check: TCP port 6333 connectivity
   - Expected: Service healthy within 30 seconds

6. **FAISS**
   - Health check: HTTP endpoint `/health`
   - Expected: Service healthy within 30 seconds

### Phase 3: AI Model Services (240-480 seconds)

7. **Ollama**
   - Health check: `ollama list`
   - Model downloads: TinyLlama, Qwen2.5, nomic-embed
   - Expected: Service healthy within 120 seconds
   - Model downloads: Additional 5-10 minutes

### Phase 4: Application Services (480-600 seconds)

8. **Backend API**
   - Health check: `GET /health`
   - Database migrations run automatically
   - Expected: Service healthy within 60 seconds

9. **Frontend Application**
   - Health check: `GET /healthz`
   - Expected: Service healthy within 30 seconds

### Phase 5: AI Agents (600-900 seconds)

10. **Specialized AI Agents**
    - 69 agents deploy in parallel
    - Health checks: Container status
    - Expected: 80%+ agents healthy within 300 seconds

### Phase 6: Monitoring Stack (900-1080 seconds)

11. **Monitoring Services**
    - Prometheus, Grafana, Loki, AlertManager
    - Dashboard provisioning
    - Expected: All monitoring healthy within 180 seconds

---

## Health Validation

### Automated Health Checks

The deployment script includes comprehensive health validation:

```bash
# Run health checks manually
./deploy.sh health

# Or use the dedicated health check script
./health_check.sh
```

### Manual Validation Steps

1. **Infrastructure Health**:
   ```bash
   # PostgreSQL
   docker exec sutazai-postgres pg_isready -U sutazai
   
   # Redis
   docker exec sutazai-redis redis-cli ping
   
   # Neo4j
   curl -f http://localhost:7474/db/system/tx/commit
   ```

2. **Application Health**:
   ```bash
   # Backend API
   curl -f http://localhost:8000/health
   curl -f http://localhost:8000/api/v1/health
   
   # Frontend
   curl -f http://localhost:8501/healthz
   ```

3. **AI Services Health**:
   ```bash
   # Ollama models
   curl -s http://localhost:11434/api/tags | jq '.models[].name'
   
   # Test AI inference
   curl -X POST http://localhost:11434/api/generate \
     -H "Content-Type: application/json" \
     -d '{"model": "tinyllama", "prompt": "Hello", "stream": false}'
   ```

4. **Vector Database Health**:
   ```bash
   # ChromaDB
   curl -f http://localhost:8001/api/v1/heartbeat
   
   # Qdrant
   curl -f http://localhost:6333/collections
   ```

### Expected Health Check Results

**Healthy System Output**:
```
=== SutazAI System Health Check ===
✓ PostgreSQL is healthy
✓ Redis is healthy  
✓ Ollama is healthy
✓ Backend API is healthy
✓ Frontend is healthy
✓ TinyLlama model is working

System Status: ALL SERVICES HEALTHY (8/8)
```

---

## Common Issues & Troubleshooting

### Issue 1: Insufficient System Resources

**Symptoms**:
- Services failing to start
- Out of memory errors
- Slow response times

**Solutions**:
```bash
# Check system resources
free -h
df -h
docker stats

# Deploy in lightweight mode
LIGHTWEIGHT_MODE=true ./deploy.sh deploy optimized

# Reduce resource limits
export OLLAMA_MAX_LOADED_MODELS=1
export MAX_WORKERS=2
```

### Issue 2: Port Conflicts

**Symptoms**:
- Services fail to bind to ports
- Connection refused errors

**Solutions**:
```bash
# Check port usage
netstat -tulpn | grep -E ':(8000|8501|5432|6379|7474|7687|11434)'

# Stop conflicting services
sudo systemctl stop nginx  # Example
sudo systemctl stop apache2  # Example

# Use alternative ports (modify docker-compose.yml)
ports:
  - "18501:8501"  # Alternative frontend port
```

### Issue 3: Docker Daemon Issues

**Symptoms**:
- Cannot connect to Docker daemon
- Permission denied errors

**Solutions**:
```bash
# Start Docker daemon
sudo systemctl start docker
sudo systemctl enable docker

# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Test Docker access
docker ps
```

### Issue 4: Model Download Failures

**Symptoms**:
- Ollama models not downloading
- AI inference failures

**Solutions**:
```bash
# Check internet connectivity
curl -I https://ollama.ai

# Manual model download
docker exec sutazai-ollama ollama pull tinyllama
docker exec sutazai-ollama ollama pull qwen2.5:3b

# Verify model availability
docker exec sutazai-ollama ollama list
```

### Issue 5: Database Connection Failures

**Symptoms**:
- Backend API health checks fail
- Database connection errors in logs

**Solutions**:
```bash
# Check database container status
docker ps | grep sutazai-postgres

# Verify database accessibility
docker exec sutazai-postgres pg_isready -U sutazai

# Check connection from backend
docker exec sutazai-backend nc -zv postgres 5432

# Review database logs
docker logs sutazai-postgres
```

### Issue 6: Memory Leaks and Resource Exhaustion

**Symptoms**:
- Gradually increasing memory usage
- System becoming unresponsive

**Solutions**:
```bash
# Monitor resource usage
docker stats --no-stream

# Restart resource-intensive services
docker restart sutazai-ollama
docker restart sutazai-backend

# Check for memory leaks
docker exec sutazai-backend ps aux --sort=-%mem | head

# Enable memory limits (in docker-compose.yml)
deploy:
  resources:
    limits:
      memory: 4G
```

---

## Rollback Procedures

### Understanding Rollback Points

The deployment system automatically creates rollback points at critical phases:
- Before infrastructure deployment
- Before services deployment  
- Before AI agent deployment
- Before configuration changes

### Manual Rollback

1. **List Available Rollback Points**:
   ```bash
   ls -la /opt/sutazaiapp/logs/rollback/
   ```

2. **Rollback to Latest Point**:
   ```bash
   ./deploy.sh rollback latest
   ```

3. **Rollback to Specific Point**:
   ```bash
   ./deploy.sh rollback rollback_infrastructure_1704723600
   ```

### Emergency Stop and Rollback

```bash
# Emergency stop all services
docker compose down --remove-orphans

# Clean up volumes (CAUTION: This removes all data)
docker compose down --volumes

# Rollback to last known good state
./deploy.sh rollback latest

# Restart with rollback configuration
docker compose up -d
```

### Rollback Validation

After rollback, validate system health:
```bash
# Check service status
./deploy.sh status

# Run health checks
./health_check.sh

# Verify critical functionality
curl -f http://localhost:8000/health
curl -f http://localhost:8501/healthz
```

---

## Post-Deployment Checklist

### Immediate Post-Deployment (0-30 minutes)

- [ ] All core services are healthy
- [ ] Web interfaces are accessible
- [ ] AI models are responding
- [ ] Database connections are working
- [ ] Monitoring dashboards are populated

### Short-term Validation (30 minutes - 2 hours)

- [ ] Run comprehensive health checks
- [ ] Test AI agent functionality
- [ ] Verify monitoring and alerting
- [ ] Check log aggregation
- [ ] Validate backup systems

### Documentation and Access

- [ ] Record deployment details
- [ ] Document access credentials
- [ ] Share access URLs with stakeholders
- [ ] Update operational runbooks

### Access Information

After successful deployment, access the system via:

| Service | URL | Default Credentials |
|---------|-----|-------------------|
| **Main Application** | http://localhost:8501 | No authentication |
| **Backend API** | http://localhost:8000 | No authentication |
| **API Documentation** | http://localhost:8000/docs | No authentication |
| **Grafana** | http://localhost:3000 | admin / [generated password] |
| **Prometheus** | http://localhost:9090 | No authentication |
| **Neo4j Browser** | http://localhost:7474 | neo4j / [generated password] |

**Credential Locations**:
```bash
# Database passwords
cat /opt/sutazaiapp/secrets/postgres_password.txt
cat /opt/sutazaiapp/secrets/redis_password.txt
cat /opt/sutazaiapp/secrets/neo4j_password.txt

# Monitoring passwords
cat /opt/sutazaiapp/secrets/grafana_password.txt

# Full access information
cat /opt/sutazaiapp/logs/ACCESS_INFO_*.txt
```

---

## Monitoring & Maintenance

### Monitoring Dashboard Access

1. **Grafana Dashboard**: http://localhost:3000
   - Username: `admin`
   - Password: Check `/opt/sutazaiapp/secrets/grafana_password.txt`

2. **Key Dashboards**:
   - SutazAI System Overview
   - AI Agent Performance
   - Infrastructure Metrics
   - Ollama Model Performance
   - Database Health

### Log Management

**Log Locations**:
```bash
# Deployment logs
/opt/sutazaiapp/logs/deployment_*.log

# Health check reports
/opt/sutazaiapp/logs/health_report_*.json

# Service logs (via Docker)
docker logs sutazai-backend
docker logs sutazai-frontend
docker logs sutazai-ollama
```

**Log Rotation**:
```bash
# Configure log rotation
sudo nano /etc/logrotate.d/sutazai

# Add configuration:
/opt/sutazaiapp/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    create 644 sutazai sutazai
}
```

### Backup Procedures

1. **Database Backups**:
   ```bash
   # PostgreSQL backup
   docker exec sutazai-postgres pg_dump -U sutazai sutazai > sutazai_backup_$(date +%Y%m%d).sql
   
   # Neo4j backup
   docker exec sutazai-neo4j neo4j-admin backup --backup-dir=/backups --name=sutazai_$(date +%Y%m%d)
   ```

2. **Configuration Backup**:
   ```bash
   # Backup configuration and secrets
   tar -czf sutazai_config_backup_$(date +%Y%m%d).tar.gz \
     /opt/sutazaiapp/.env \
     /opt/sutazaiapp/secrets/ \
     /opt/sutazaiapp/config/
   ```

3. **Automated Backup Script**:
   ```bash
   # Use the built-in backup script (created during deployment)
   /opt/sutazaiapp/scripts/backup_system.sh
   ```

### Regular Maintenance Tasks

**Daily**:
- [ ] Check system health via `./health_check.sh`
- [ ] Review error logs
- [ ] Monitor resource usage

**Weekly**:
- [ ] Review monitoring dashboards
- [ ] Check for security updates
- [ ] Validate backup integrity
- [ ] Clean up old logs and temporary files

**Monthly**:
- [ ] Update AI models
- [ ] Review and update monitoring alerts
- [ ] Perform security audit
- [ ] Update system documentation

---

## Security Considerations

### Network Security

1. **Firewall Configuration**:
   ```bash
   # Allow necessary ports only
   sudo ufw allow 22/tcp    # SSH
   sudo ufw allow 8501/tcp  # Frontend
   sudo ufw allow 8000/tcp  # Backend API
   sudo ufw allow 3000/tcp  # Grafana (if external access needed)
   sudo ufw enable
   ```

2. **Internal Network Isolation**:
   - All internal services communicate via Docker network
   - Database ports are not exposed externally
   - AI model endpoints are internal-only

### Data Protection

1. **Secrets Management**:
   - All passwords are auto-generated and stored securely
   - Secrets are not logged or exposed in environment variables
   - File permissions are set to 600 for sensitive files

2. **Encryption**:
   - SSL certificates are generated for internal communication
   - Database connections use encryption where supported
   - API communications can be secured with reverse proxy

### Access Control

1. **Container Security**:
   - Containers run with minimal privileges
   - Host filesystem access is limited to necessary paths
   - Security scanning is performed on container images

2. **Authentication** (Future Enhancement):
   - Currently, services use network-level access control
   - Consider implementing OAuth2/JWT for API access
   - RBAC for different user roles

---

## Performance Tuning

### System-Level Optimization

1. **Kernel Parameters**:
   ```bash
   # Already configured during deployment
   echo "net.core.somaxconn = 65535" >> /etc/sysctl.conf
   echo "vm.max_map_count = 262144" >> /etc/sysctl.conf
   sysctl -p
   ```

2. **File System Optimization**:
   ```bash
   # Use SSD for database storage
   # Mount with noatime for better performance
   /dev/sdb1 /opt/sutazaiapp/data ext4 defaults,noatime 0 0
   ```

### Application Tuning

1. **Database Optimization**:
   ```bash
   # PostgreSQL settings (in docker-compose.yml)
   environment:
     - POSTGRES_SHARED_BUFFERS=1GB
     - POSTGRES_EFFECTIVE_CACHE_SIZE=4GB
     - POSTGRES_WORK_MEM=16MB
   ```

2. **AI Model Optimization**:
   ```bash
   # Ollama performance settings
   OLLAMA_NUM_PARALLEL=2          # Adjust based on available CPU
   OLLAMA_MAX_LOADED_MODELS=2     # Limit concurrent models
   OLLAMA_KEEP_ALIVE=10m          # Model cache timeout
   ```

3. **Backend API Optimization**:
   ```bash
   # FastAPI/Uvicorn settings
   MAX_WORKERS=4                  # Adjust based on CPU cores
   CONNECTION_POOL_SIZE=20        # Database connection pool
   CACHE_TTL=3600                # API response caching
   ```

### Resource Monitoring

Use the monitoring dashboards to track:
- CPU and memory usage per service
- Database query performance
- AI model inference latency
- Network I/O patterns
- Disk usage and growth rates

---

## Appendices

### Appendix A: Service Port Reference

| Service | Internal Port | External Port | Protocol | Purpose |
|---------|---------------|---------------|----------|---------|
| sutazai-postgres | 5432 | 10000 | TCP | Database |
| sutazai-redis | 6379 | 10001 | TCP | Cache |
| sutazai-neo4j | 7474, 7687 | 10002, 10003 | HTTP/Bolt | Graph DB |
| sutazai-chromadb | 8000 | 10100 | HTTP | Vector DB |
| sutazai-qdrant | 6333 | 10101 | HTTP | Vector Search |
| sutazai-faiss | 8000 | 10103 | HTTP | Similarity Search |
| sutazai-ollama | 11434 | 10104 | HTTP | AI Models |
| sutazai-backend | 8000 | 10010 | HTTP | API Server |
| sutazai-frontend | 8501 | 10011 | HTTP | Web UI |
| sutazai-prometheus | 9090 | 10200 | HTTP | Metrics |
| sutazai-grafana | 3000 | 10201 | HTTP | Dashboards |
| sutazai-langflow | 7860 | 10400 | HTTP | AI Workflows |
| sutazai-flowise | 3000 | 10401 | HTTP | AI Chains |

### Appendix B: Environment Variables Reference

**Core Configuration**:
```bash
# System
TZ=UTC
SUTAZAI_ENV=production
LOCAL_IP=<auto-detected>

# Database
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=<auto-generated>
POSTGRES_DB=sutazai
REDIS_PASSWORD=<auto-generated>
NEO4J_PASSWORD=<auto-generated>

# Security
SECRET_KEY=<auto-generated>
JWT_SECRET=<auto-generated>

# AI Models
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*
OLLAMA_NUM_PARALLEL=2
OLLAMA_MAX_LOADED_MODELS=2

# Performance
MAX_WORKERS=4
CONNECTION_POOL_SIZE=20
CACHE_TTL=3600

# Features
ENABLE_GPU=auto
ENABLE_MONITORING=true
ENABLE_LOGGING=true
```

### Appendix C: Troubleshooting Commands

**System Information**:
```bash
# System resources
free -h
df -h
nproc
lscpu

# Docker information
docker version
docker info
docker stats --no-stream

# Service status
docker ps -a
docker compose ps
docker compose logs <service>

# Network connectivity
netstat -tulpn | grep LISTEN
curl -I http://localhost:8000/health
curl -I http://localhost:8501/healthz
```

**Log Analysis**:
```bash
# Deployment logs
tail -f /opt/sutazaiapp/logs/deployment_*.log

# Service logs
docker logs sutazai-backend --tail 100
docker logs sutazai-ollama --tail 100

# Health check history
ls -la /opt/sutazaiapp/logs/health_report_*.json
```

### Appendix D: Support and Resources

- **Project Repository**: Internal SutazAI repository
- **Documentation**: `/opt/sutazaiapp/docs/`
- **Log Files**: `/opt/sutazaiapp/logs/`
- **Configuration**: `/opt/sutazaiapp/config/`
- **Monitoring**: http://localhost:3000 (Grafana)

**Emergency Contacts**:
- System Administrator: [Your contact information]
- DevOps Team: [Team contact information]
- On-call Support: [Support contact information]

---

**Document Information**:
- **Created**: January 8, 2025
- **Version**: 4.0.0
- **Last Updated**: January 8, 2025
- **Next Review**: February 8, 2025

This deployment guide provides comprehensive instructions for deploying SutazAI in production environments. Follow all steps carefully and maintain proper documentation of your specific deployment configuration.