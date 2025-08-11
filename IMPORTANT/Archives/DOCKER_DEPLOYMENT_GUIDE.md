# SutazAI Consolidated Docker Deployment Guide

## Overview

This guide covers the **new consolidated Docker deployment system** for SutazAI. The old deployment chaos has been replaced with a single, working configuration that only includes services that actually exist and can be built.

## What Changed

### ❌ REMOVED (Phantom Services)
- 100+ conflicting Docker Compose files
- conceptual network configurations
- Non-existent service definitions
- Broken port mappings
- Restart loops and dependency hell
- AGI/quantum/advanced features that don't exist

### ✅ KEPT (Working Services Only)
- **Core Infrastructure**: PostgreSQL, Redis, Neo4j
- **Vector Databases**: ChromaDB, Qdrant  
- **LLM Service**: Ollama with tinyllama
- **Application**: Backend (FastAPI) + Frontend (Streamlit)
- **Monitoring**: Prometheus, Grafana, Loki
- **Health Monitoring**: Real container health checks

## Quick Start

### 1. Clean Up Old Configuration
```bash
# Remove all old Docker compose files and containers
./cleanup-old-docker-files.sh

# This will:
# - Stop all old containers
# - Archive old compose files 
# - Remove orphaned containers
# - Clean up unused images
```

### 2. Deploy Everything
```bash
# Single command deployment
./deploy-consolidated.sh

# This will:
# - Check prerequisites
# - Create network and environment
# - Pull images and build services
# - Start services in correct order
# - Wait for health checks
# - Show access information
```

### 3. Validate Deployment
```bash
# Full validation suite
./validate-deployment.sh

# Quick health check
./validate-deployment.sh quick

# Test specific components
./validate-deployment.sh endpoints
./validate-deployment.sh ports
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      SutazAI Consolidated                   │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit)     │  Backend (FastAPI)              │
│  Port: 10011             │  Port: 10010                    │
├─────────────────────────────────────────────────────────────┤
│           LLM Service (Ollama) - Port: 10104               │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL │  Redis  │  Neo4j  │ ChromaDB │  Qdrant      │
│  Port: 10000│Port:10001│Port:10002│Port:10100│Port: 10101   │
├─────────────────────────────────────────────────────────────┤
│     Monitoring Stack (Prometheus, Grafana, Loki)           │
│     Ports: 10200-10202                                     │
└─────────────────────────────────────────────────────────────┘
```

## Port Allocation

All services follow the standardized port registry:

### Infrastructure (10000-10199)
- **10000**: PostgreSQL Database
- **10001**: Redis Cache
- **10002**: Neo4j HTTP (Browser)
- **10003**: Neo4j Bolt (Driver)
- **10010**: Backend API
- **10011**: Frontend Application
- **10100**: ChromaDB Vector Database
- **10101**: Qdrant Vector Database (HTTP)
- **10102**: Qdrant Vector Database (gRPC)
- **10104**: Ollama LLM Service

### Monitoring (10200-10299)
- **10200**: Prometheus Metrics
- **10201**: Grafana Dashboards
- **10202**: Loki Log Aggregation
- **10205**: Node Exporter (System Metrics)
- **10206**: cAdvisor (Container Metrics)
- **10210**: Health Monitor

## Services

### Core Infrastructure

#### PostgreSQL (`sutazai-postgres`)
- **Image**: `postgres:16.3-alpine`
- **Port**: `10000:5432`
- **Purpose**: Main application database
- **Health Check**: `pg_isready` command
- **Resources**: 2GB RAM limit, 512MB reserved

#### Redis (`sutazai-redis`)
- **Image**: `redis:7.2-alpine`
- **Port**: `10001:6379`
- **Purpose**: Caching and session storage
- **Health Check**: `redis-cli ping`
- **Resources**: 1GB RAM limit, 256MB reserved

#### Neo4j (`sutazai-neo4j`)
- **Image**: `neo4j:5.13-community`
- **Ports**: `10002:7474` (HTTP), `10003:7687` (Bolt)
- **Purpose**: Knowledge graph database
- **Plugins**: APOC, Graph Data Science
- **Resources**: 4GB RAM limit, 1GB reserved

### Vector Databases

#### ChromaDB (`sutazai-chromadb`)
- **Image**: `chromadb/chroma:0.5.0`
- **Port**: `10100:8000`
- **Purpose**: Vector embeddings storage
- **Auth**: Token-based authentication
- **Resources**: 2GB RAM limit, 512MB reserved

#### Qdrant (`sutazai-qdrant`)
- **Image**: `qdrant/qdrant:v1.9.2`
- **Ports**: `10101:6333` (HTTP), `10102:6334` (gRPC)
- **Purpose**: High-performance vector search
- **Resources**: 2GB RAM limit, 512MB reserved

### LLM Service

#### Ollama (`sutazai-ollama`)
- **Image**: `ollama/ollama:latest`
- **Port**: `10104:10104`
- **Purpose**: Local LLM inference
- **Models**: tinyllama (auto-downloaded)
- **Resources**: 20GB RAM limit, 8GB reserved
- **Features**: Flash attention, multi-threading

### Application Layer

#### Backend (`sutazai-backend`)
- **Build**: `./backend/Dockerfile`
- **Port**: `10010:8000`
- **Purpose**: FastAPI application server
- **Dependencies**: All databases + Ollama
- **Health Check**: TCP connection test
- **Resources**: 4GB RAM limit, 1GB reserved

#### Frontend (`sutazai-frontend`)
- **Build**: `./frontend/Dockerfile`
- **Port**: `10011:8501`
- **Purpose**: Streamlit web interface
- **Dependencies**: Backend API
- **Health Check**: TCP connection test

### Monitoring Stack

#### Prometheus (`sutazai-prometheus`)
- **Image**: `prom/prometheus:latest`
- **Port**: `10200:9090`
- **Purpose**: Metrics collection and storage
- **Retention**: 30 days
- **Config**: `/monitoring/prometheus/prometheus.yml`

#### Grafana (`sutazai-grafana`)
- **Image**: `grafana/grafana:latest`
- **Port**: `10201:3000`
- **Purpose**: Metrics visualization
- **Credentials**: `admin` / `${GRAFANA_PASSWORD}`
- **Plugins**: Piechart, Worldmap, Clock

#### Loki (`sutazai-loki`)
- **Image**: `grafana/loki:2.9.0`
- **Port**: `10202:3100`
- **Purpose**: Log aggregation and storage
- **Retention**: 30 days
- **Config**: `/monitoring/loki/config.yml`

### System Monitoring

#### Node Exporter (`sutazai-node-exporter`)
- **Image**: `prom/node-exporter:latest`
- **Port**: `10205:9100`
- **Purpose**: Host system metrics

#### cAdvisor (`sutazai-cadvisor`)
- **Image**: `gcr.io/cadvisor/cadvisor:v0.47.0`
- **Port**: `10206:8080`
- **Purpose**: Container resource metrics
- **Privileges**: Required for system access

#### Health Monitor (`sutazai-health-monitor`)
- **Build**: `./docker/health-monitor/Dockerfile`
- **Port**: `10210:8000`
- **Purpose**: Service health monitoring and alerting
- **Monitors**: All critical services

## Environment Configuration

Create a `.env` file with these variables:

```bash
# Database Configuration
POSTGRES_DB=sutazai
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=your_secure_password_here
NEO4J_PASSWORD=your_neo4j_password_here

# Vector Database
CHROMADB_API_KEY=your_chroma_token_here

# Application Security
SECRET_KEY=your-32-character-secret-key-here
JWT_SECRET=your-32-character-jwt-secret-here

# Monitoring
GRAFANA_PASSWORD=your_grafana_password

# Application Settings
SUTAZAI_ENV=production
TZ=UTC

# Optional
HEALTH_ALERT_WEBHOOK=your_webhook_url
REDIS_PASSWORD=optional_redis_password
```

## Usage Commands

### Deployment Commands
```bash
# Deploy everything
./deploy-consolidated.sh deploy

# Stop all services
./deploy-consolidated.sh stop

# Restart all services
./deploy-consolidated.sh restart

# Check service status
./deploy-consolidated.sh status

# View logs
./deploy-consolidated.sh logs [service_name]

# Health check
./deploy-consolidated.sh health

# Complete cleanup (destructive)
./deploy-consolidated.sh clean
```

### Validation Commands
```bash
# Full validation
./validate-deployment.sh validate

# Quick health check
./validate-deployment.sh quick

# Test endpoints only
./validate-deployment.sh endpoints

# Test ports only
./validate-deployment.sh ports
```

### Docker Compose Commands
```bash
# View service status
docker-compose -f docker-compose.consolidated.yml ps

# View logs
docker-compose -f docker-compose.consolidated.yml logs -f [service]

# Scale a service
docker-compose -f docker-compose.consolidated.yml up -d --scale backend=2

# Restart specific service
docker-compose -f docker-compose.consolidated.yml restart backend

# Execute command in container
docker-compose -f docker-compose.consolidated.yml exec backend bash
```

## Health Checks

All services include comprehensive health checks:

- **Databases**: Connection tests and readiness probes
- **Applications**: HTTP endpoint checks
- **Monitoring**: Service-specific health endpoints
- **Resources**: Memory and CPU monitoring

Health check intervals:
- **Critical services**: 10-30 seconds
- **Applications**: 30-60 seconds
- **Monitoring**: 60 seconds

## Resource Requirements

### Minimum System Requirements
- **RAM**: 8GB (16GB recommended)
- **CPU**: 4 cores (8 cores recommended)
- **Disk**: 50GB available space
- **OS**: Linux with Docker support

### Container Resource Allocation
- **Total RAM Usage**: ~15GB under full load
- **CPU Usage**: Up to 20 cores for Ollama/ML workloads
- **Storage**: Persistent volumes for data

## Troubleshooting

### Common Issues

#### Services Won't Start
```bash
# Check Docker service
sudo systemctl status docker

# Check logs
./deploy-consolidated.sh logs [service_name]

# Restart specific service
docker-compose -f docker-compose.consolidated.yml restart [service]
```

#### Port Conflicts
```bash
# Check what's using a port
sudo lsof -i :10010

# Kill process using port
sudo kill -9 $(sudo lsof -t -i:10010)
```

#### Memory Issues
```bash
# Check container resource usage
docker stats

# Reduce Ollama resource limits in compose file
# Restart with lower memory allocation
```

#### Database Connection Issues
```bash
# Test PostgreSQL connection
docker exec sutazai-postgres psql -U sutazai -c '\l'

# Test Redis connection
docker exec sutazai-redis redis-cli ping

# Check Neo4j status
curl http://localhost:10002
```

### Logs and Debugging

```bash
# View all service logs
docker-compose -f docker-compose.consolidated.yml logs

# Follow specific service logs
docker-compose -f docker-compose.consolidated.yml logs -f backend

# Check container stats
docker stats --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Execute shell in container
docker exec -it sutazai-backend bash
```

## Security Considerations

1. **Change Default Passwords**: All default passwords must be changed in production
2. **Network Security**: All services communicate over isolated Docker network
3. **Secret Management**: Use Docker secrets or external secret management
4. **Resource Limits**: All containers have memory and CPU limits
5. **Health Monitoring**: Continuous health monitoring and alerting

## Backup and Recovery

### Database Backups
```bash
# PostgreSQL backup
docker exec sutazai-postgres pg_dump -U sutazai sutazai > backup.sql

# Neo4j backup
docker exec sutazai-neo4j neo4j-admin database dump sutazai

# Vector database backups are stored in persistent volumes
```

### Volume Management
```bash
# List all volumes
docker volume ls --filter name=sutazai

# Backup volume data
docker run --rm -v sutazai_postgres_data:/data -v $(pwd):/backup alpine tar czf /backup/postgres_backup.tar.gz /data
```

## Performance Tuning

### Ollama Optimization
- Adjust `OLLAMA_NUM_PARALLEL` for concurrent requests
- Modify `OLLAMA_NUM_THREADS` based on CPU cores
- Use `OLLAMA_MAX_LOADED_MODELS` to limit memory usage

### Database Tuning
- PostgreSQL: Adjust shared_buffers and work_mem
- Neo4j: Configure heap size and page cache
- Redis: Set maxmemory and eviction policies

### Monitoring Optimization
- Adjust scrape intervals in Prometheus config
- Configure log retention in Loki
- Set up alerting rules for critical metrics

## Migration from Old System

1. **Stop old services**: `./cleanup-old-docker-files.sh`
2. **Backup data**: Export databases and important data
3. **Deploy new system**: `./deploy-consolidated.sh`
4. **Validate deployment**: `./validate-deployment.sh`
5. **Import data**: Restore databases from backups

## Support

For support with the consolidated deployment:

1. Run validation: `./validate-deployment.sh`
2. Check logs: `./deploy-consolidated.sh logs`
3. Review this guide for common issues
4. Check container resource usage: `docker stats`

The consolidated deployment eliminates the complexity and conflicts of the previous system while providing a stable, monitored, and scalable foundation for SutazAI.