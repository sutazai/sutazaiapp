# Local Development Environment Setup

**Last Updated:** 2025-09-03  
**Version:** 1.0.0  
**Maintainer:** Development Team  

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [System Requirements](#system-requirements)
3. [Initial Setup](#initial-setup)
4. [Docker Environment](#docker-environment)
5. [Service Deployment](#service-deployment)
6. [Verification Steps](#verification-steps)
7. [Common Issues](#common-issues)
8. [Development Workflow](#development-workflow)
9. [IDE Configuration](#ide-configuration)
10. [Related Documentation](#related-documentation)

## Prerequisites

### Required Software

| Software | Minimum Version | Recommended Version | Purpose |
|----------|-----------------|---------------------|---------|  
| Docker | 24.0.0 | Latest | Container runtime |
| Docker Compose | 2.20.0 | Latest | Service orchestration |
| Python | 3.11 | 3.12 | Backend development |
| Node.js | 18.0.0 | 20.0.0 | Frontend tooling |
| Git | 2.30.0 | Latest | Version control |
| PostgreSQL Client | 14.0 | 16.0 | Database access |
| Redis CLI | 6.2 | 7.0 | Cache debugging |

### Hardware Requirements

- **CPU**: Minimum 4 cores, recommended 8 cores
- **RAM**: Minimum 16GB, recommended 32GB
- **Storage**: 50GB free space for Docker images and data
- **Network**: Stable internet connection for package downloads

## System Requirements

### Operating System Support

- **Linux**: Ubuntu 20.04+, Debian 11+, RHEL 8+
- **macOS**: 12.0+ (Monterey or later)
- **Windows**: WSL2 with Ubuntu 20.04+

### Network Ports

Ensure the following ports are available:

```bash
# Core Services
10000  # PostgreSQL
10001  # Redis
10002-10003  # Neo4j
10004-10005  # RabbitMQ
10008-10009  # Kong API Gateway

# Application Services
10200  # Backend API
11000  # Frontend UI
11100  # MCP Bridge

# Vector Databases
10100  # ChromaDB
10101-10102  # Qdrant
10103  # FAISS

# AI Agents
11401-11801  # Agent pool
```

## Initial Setup

### 1. Clone Repository

```bash
# Clone the repository
git clone https://github.com/sutazai/sutazaiapp.git /opt/sutazaiapp
cd /opt/sutazaiapp

# Checkout development branch
git checkout v118
```

### 2. Environment Configuration

```bash
# Copy example environment file
cp .env.example .env

# Edit environment variables
nano .env
```

#### Required Environment Variables

```bash
# Database Configuration
POSTGRES_USER=jarvis
POSTGRES_PASSWORD=sutazai_secure_2024
POSTGRES_DB=jarvis_ai
POSTGRES_PORT=10000

# Redis Configuration
REDIS_HOST=sutazai-redis
REDIS_PORT=6379
REDIS_PASSWORD=sutazai_redis_2024

# Neo4j Configuration
NEO4J_AUTH=neo4j/sutazai_secure_2024
NEO4J_BOLT_PORT=10002
NEO4J_HTTP_PORT=10003

# RabbitMQ Configuration
RABBITMQ_DEFAULT_USER=sutazai
RABBITMQ_DEFAULT_PASS=sutazai_secure_2024

# Application Settings
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG
```

### 3. Create Docker Network

```bash
# Create custom bridge network
docker network create \
  --driver bridge \
  --subnet=172.20.0.0/16 \
  --gateway=172.20.0.1 \
  sutazai-network
```

### 4. Build Base Images

```bash
# Build all services
docker compose build --no-cache

# Or build specific services
docker compose build backend
docker compose build frontend
```

## Docker Environment

### Service Architecture

```yaml
# docker-compose-core.yml - Infrastructure services
services:
  postgres:
    container_name: sutazai-postgres
    networks:
      sutazai-network:
        ipv4_address: 172.20.0.10

  redis:
    container_name: sutazai-redis
    networks:
      sutazai-network:
        ipv4_address: 172.20.0.11

  neo4j:
    container_name: sutazai-neo4j
    networks:
      sutazai-network:
        ipv4_address: 172.20.0.12

  rabbitmq:
    container_name: sutazai-rabbitmq
    networks:
      sutazai-network:
        ipv4_address: 172.20.0.13
```

### Volume Management

```bash
# Create persistent volumes
docker volume create sutazai-postgres-data
docker volume create sutazai-redis-data
docker volume create sutazai-neo4j-data
docker volume create sutazai-rabbitmq-data

# List volumes
docker volume ls | grep sutazai
```

## Service Deployment

### Full Stack Deployment

```bash
# Deploy all services
cd /opt/sutazaiapp
./deploy.sh

# Or deploy individually
docker compose -f docker-compose-core.yml up -d
docker compose -f docker-compose-backend.yml up -d
docker compose -f docker-compose-frontend.yml up -d
docker compose -f docker-compose-vectors.yml up -d
```

### Service Health Checks

```bash
# Check service status
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Verify core services
curl http://localhost:10200/health  # Backend API
curl http://localhost:11000/_stcore/health  # Frontend
curl http://localhost:11100/health  # MCP Bridge

# Database connectivity
PGPASSWORD=sutazai_secure_2024 psql -h localhost -p 10000 -U jarvis -d jarvis_ai -c "\l"
redis-cli -h localhost -p 10001 ping
```

## Verification Steps

### 1. Service Connectivity

```bash
# Test PostgreSQL
docker exec sutazai-backend pg_isready -h sutazai-postgres -p 5432

# Test Redis
docker exec sutazai-backend redis-cli -h sutazai-redis ping

# Test RabbitMQ
curl -u sutazai:sutazai_secure_2024 http://localhost:10005/api/overview

# Test Neo4j
curl http://localhost:10003
```

### 2. API Endpoints

```bash
# Health check
curl http://localhost:10200/health

# API documentation
open http://localhost:10200/docs  # Swagger UI
open http://localhost:10200/redoc  # ReDoc

# Frontend
open http://localhost:11000  # Streamlit UI
```

### 3. MCP Servers

```bash
# Test MCP server connectivity
for wrapper in /opt/sutazaiapp/scripts/mcp/wrappers/*.sh; do
    echo "Testing $(basename "$wrapper" .sh)..."
    "$wrapper" --selfcheck
done
```

## Common Issues

### Port Conflicts

```bash
# Find process using port
sudo lsof -i :10000
sudo netstat -tulpn | grep 10000

# Kill process
sudo kill -9 <PID>
```

### Docker Issues

```bash
# Clean Docker system
docker system prune -a --volumes

# Reset Docker
sudo systemctl restart docker

# Check Docker logs
journalctl -u docker.service -f
```

### Database Connection Issues

```bash
# Check container logs
docker logs sutazai-postgres --tail 50
docker logs sutazai-backend --tail 50

# Restart specific service
docker compose restart sutazai-postgres
docker compose restart sutazai-backend
```

### Memory Issues

```bash
# Check memory usage
free -h
docker stats --no-stream

# Adjust Docker memory limits
# Edit docker-compose.yml
mem_limit: 2g
memswap_limit: 2g
```

## Development Workflow

### 1. Branch Management

```bash
# Create feature branch
git checkout -b feature/your-feature

# Keep branch updated
git fetch origin
git rebase origin/v118
```

### 2. Code Changes

```bash
# Backend development
cd backend
source venv/bin/activate
pip install -r requirements.txt
python -m pytest tests/

# Frontend development
cd frontend
source venv/bin/activate
streamlit run app.py --server.port 11000
```

### 3. Testing

```bash
# Run backend tests
cd backend
./venv/bin/pytest tests/ -v --cov=app

# Run integration tests
docker exec sutazai-backend pytest tests/integration/

# Run MCP server tests
./scripts/test_mcp_servers.sh
```

### 4. Debugging

```bash
# Live logs
./scripts/monitoring/live_logs.sh live

# Specific service logs
docker logs -f sutazai-backend --tail 100

# Enable debug mode
export DEBUG=true
export LOG_LEVEL=DEBUG
```

## IDE Configuration

### VS Code

```json
// .vscode/settings.json
{
  "python.defaultInterpreterPath": "./backend/venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "editor.formatOnSave": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true
  }
}
```

### PyCharm

1. Set Python interpreter: `backend/venv/bin/python`
2. Mark directories:
   - `backend/app` as Sources Root
   - `backend/tests` as Test Sources Root
3. Enable Docker integration
4. Configure remote debugging

### Docker Desktop

1. Allocate resources:
   - CPUs: 4+
   - Memory: 8GB+
   - Swap: 2GB
   - Disk: 50GB+

2. Enable Kubernetes (optional)
3. Configure registry mirrors

## Related Documentation

- [Environment Configuration](./environments.md)
- [Dependencies Management](./dependencies.md)
- [Troubleshooting Guide](./troubleshooting.md)
- [Development Tools](./tools.md)
- [Docker Architecture](../architecture/system_design.md)
- [API Development](../development/coding_standards.md)
- [Testing Strategy](../development/testing_strategy.md)

---

*For additional support, refer to the [Troubleshooting Guide](./troubleshooting.md) or contact the development team via Slack #sutazai-dev.*