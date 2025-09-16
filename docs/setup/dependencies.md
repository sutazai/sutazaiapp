# System Requirements and Dependencies

**Last Updated**: 2025-01-03  
**Version**: 1.0.0  
**Maintainer**: DevOps Team

## Table of Contents

1. [System Requirements](#system-requirements)
2. [Core Dependencies](#core-dependencies)
3. [Python Dependencies](#python-dependencies)
4. [Node.js Dependencies](#node-js-dependencies)
5. [Docker Dependencies](#docker-dependencies)
6. [Database Dependencies](#database-dependencies)
7. [Installation Guide](#installation-guide)
8. [Version Compatibility Matrix](#version-compatibility-matrix)
9. [Validation](#validation)

## System Requirements

### Minimum Requirements

```yaml
CPU: 4 cores (2.4 GHz)
RAM: 16 GB
Storage: 50 GB SSD
OS: Ubuntu 20.04+ / Debian 11+ / RHEL 8+ / macOS 11+
Network: 100 Mbps
```

### Recommended Requirements

```yaml
CPU: 8 cores (3.0 GHz)
RAM: 32 GB
Storage: 200 GB NVMe SSD
OS: Ubuntu 22.04 LTS
Network: 1 Gbps
GPU: NVIDIA RTX 3060+ (for ML workloads)
```

## Core Dependencies

### Docker & Container Runtime

```bash
# Docker Engine
Docker CE: 24.0+
Docker Compose: 2.20+
containerd: 1.6+

# Installation
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER

# Verify
docker --version
docker compose version
```

### System Packages

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    git \
    curl \
    wget \
    make \
    gcc \
    g++ \
    python3-dev \
    libssl-dev \
    libffi-dev \
    libpq-dev \
    redis-tools \
    postgresql-client \
    netcat \
    jq

# RHEL/CentOS
sudo yum groupinstall -y 'Development Tools'
sudo yum install -y \
    git \
    curl \
    wget \
    python3-devel \
    openssl-devel \
    postgresql-devel \
    redis
```

## Python Dependencies

### Backend Requirements

```txt
# Core Framework
fastapi==0.109.0
uvicorn[standard]==0.27.0
pydantic==2.5.3
pydantic-settings==2.1.0

# Database
sqlalchemy==2.0.25
asyncpg==0.29.0
alembic==1.13.1
redis==5.0.1
neo4j==5.16.0
motor==3.3.2

# Authentication & Security
python-jose[cryptography]==3.3.0
passlib[bcrypt]==1.7.4
python-multipart==0.0.6
cryptography==41.0.7

# Message Queue
aio-pika==9.4.0
celery[redis]==5.3.4

# Vector Stores
chromadb==0.4.22
qdrant-client==1.7.0
faiss-cpu==1.7.4

# AI/ML
openai==1.6.1
anthropic==0.8.1
langchain==0.1.0
transformers==4.36.2
torch==2.1.2

# Utilities
httpx==0.25.2
python-dateutil==2.8.2
pytz==2023.3
pyyaml==6.0.1
python-dotenv==1.0.0

# Testing
pytest==7.4.3
pytest-asyncio==0.23.2
pytest-cov==4.1.0
httpx-mock==0.25.0
```

### Frontend Requirements

```txt
# Core
streamlit==1.29.0
streamlit-aggrid==0.3.4
streamlit-extras==0.3.6

# Data Processing
pandas==2.1.4
numpy==1.26.2
plotly==5.18.0
altair==5.2.0

# API Client
requests==2.31.0
websockets==12.0
aiohttp==3.9.1
```

## Node.js Dependencies

### MCP Server Requirements

```json
{
  "dependencies": {
    "@modelcontextprotocol/sdk": "^0.5.0",
    "typescript": "^5.3.3",
    "tsx": "^4.7.0",
    "zod": "^3.22.4",
    "winston": "^3.11.0",
    "dotenv": "^16.3.1"
  },
  "devDependencies": {
    "@types/node": "^20.10.5",
    "nodemon": "^3.0.2",
    "jest": "^29.7.0",
    "@types/jest": "^29.5.11",
    "eslint": "^8.56.0",
    "@typescript-eslint/parser": "^6.15.0",
    "@typescript-eslint/eslint-plugin": "^6.15.0"
  }
}
```

## Database Dependencies

### PostgreSQL

```yaml
Version: 16.1
Extensions:
  - pg_stat_statements
  - uuid-ossp
  - pgcrypto
  - pg_trgm
  - btree_gin
  - btree_gist
Configuration:
  max_connections: 200
  shared_buffers: 2GB
  effective_cache_size: 6GB
```

### Redis

```yaml
Version: 7.2.3
Modules:
  - RedisJSON
  - RediSearch
  - RedisTimeSeries
Configuration:
  maxmemory: 4gb
  maxmemory-policy: allkeys-lru
```

### Neo4j

```yaml
Version: 5.15.0
Edition: Community
Plugins:
  - APOC: 5.15.0
  - Graph Data Science: 2.5.0
Configuration:
  heap_size: 8g
  pagecache_size: 4g
```

### RabbitMQ

```yaml
Version: 3.12.10
Plugins:
  - rabbitmq_management
  - rabbitmq_prometheus
  - rabbitmq_shovel
Configuration:
  vm_memory_high_watermark: 0.6
  disk_free_limit: 10GB
```

## Installation Guide

### Step 1: System Preparation

```bash
# Update system packages
sudo apt-get update && sudo apt-get upgrade -y

# Install prerequisites
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER
newgrp docker

# Install Python 3.12
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update
sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

# Install Node.js 20
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt-get install -y nodejs

# Install pnpm
npm install -g pnpm
```

### Step 2: Clone Repository

```bash
git clone https://github.com/sutazai/sutazaiapp.git /opt/sutazaiapp
cd /opt/sutazaiapp
git checkout v118
```

### Step 3: Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Generate secure passwords
export DB_PASSWORD=$(openssl rand -base64 32)
export REDIS_PASSWORD=$(openssl rand -base64 32)
export JWT_SECRET=$(openssl rand -base64 64)

# Update .env file
sed -i "s/DB_PASSWORD=.*/DB_PASSWORD=$DB_PASSWORD/" .env
sed -i "s/REDIS_PASSWORD=.*/REDIS_PASSWORD=$REDIS_PASSWORD/" .env
sed -i "s/JWT_SECRET_KEY=.*/JWT_SECRET_KEY=$JWT_SECRET/" .env
```

### Step 4: Install Dependencies

```bash
# Backend Python dependencies
cd backend
python3.12 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

# Frontend dependencies
cd ../frontend
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# MCP server dependencies
cd ../mcp-servers
for server in */; do
    cd "$server"
    pnpm install
    pnpm build
    cd ..
done
```

### Step 5: Database Initialization

```bash
# Start core services
docker compose -f docker-compose-core.yml up -d

# Wait for databases
./scripts/wait-for-databases.sh

# Run migrations
cd backend
alembic upgrade head

# Seed initial data
python scripts/seed_database.py
```

## Version Compatibility Matrix

| Component | Min Version | Recommended | Max Tested |
|-----------|------------|-------------|------------|
| Python | 3.10 | 3.12 | 3.12 |
| Node.js | 18.0 | 20.11 | 21.0 |
| Docker | 23.0 | 24.0 | 25.0 |
| PostgreSQL | 15.0 | 16.1 | 16.1 |
| Redis | 6.2 | 7.2 | 7.2 |
| Neo4j | 5.0 | 5.15 | 5.15 |
| RabbitMQ | 3.11 | 3.12 | 3.13 |

## Validation

### Dependency Check Script

```bash
#!/bin/bash
# scripts/check_dependencies.sh

echo "Checking system dependencies..."

# Function to check command existence
check_command() {
    if command -v "$1" &> /dev/null; then
        echo "✓ $1: $(eval "$2")"
    else
        echo "✗ $1: NOT INSTALLED"
        return 1
    fi
}

# Check core tools
check_command "docker" "docker --version"
check_command "docker-compose" "docker compose version"
check_command "python3" "python3 --version"
check_command "node" "node --version"
check_command "npm" "npm --version"
check_command "pnpm" "pnpm --version"
check_command "git" "git --version"
check_command "psql" "psql --version"
check_command "redis-cli" "redis-cli --version"

# Check Python packages
echo -e "\nChecking Python packages..."
python3 -c "
import sys
packages = [
    'fastapi', 'uvicorn', 'sqlalchemy', 'redis',
    'neo4j', 'pydantic', 'alembic', 'pytest'
]
for pkg in packages:
    try:
        __import__(pkg)
        print(f'✓ {pkg}')
    except ImportError:
        print(f'✗ {pkg}: NOT INSTALLED')
"

# Check service connectivity
echo -e "\nChecking service connectivity..."
nc -zv localhost 10000 2>/dev/null && echo "✓ PostgreSQL" || echo "✗ PostgreSQL"
nc -zv localhost 10001 2>/dev/null && echo "✓ Redis" || echo "✗ Redis"
nc -zv localhost 10002 2>/dev/null && echo "✓ Neo4j" || echo "✗ Neo4j"
nc -zv localhost 10004 2>/dev/null && echo "✓ RabbitMQ" || echo "✗ RabbitMQ"
```

### Run Validation

```bash
chmod +x scripts/check_dependencies.sh
./scripts/check_dependencies.sh
```

## Related Documentation

- [Troubleshooting Guide](./troubleshooting.md)
- [Tools Setup](./tools.md)
- [System Architecture](../architecture/system_design.md)
- [Development Setup](../development/coding_standards.md)

## Support

For dependency issues:
1. Check [Troubleshooting Guide](./troubleshooting.md)
2. Review system logs: `./scripts/monitoring/live_logs.sh`
3. Contact DevOps team: devops@sutazai.com