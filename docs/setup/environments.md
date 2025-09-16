# Environment Configuration Guide

**Last Updated:** 2025-09-03  
**Version:** 1.0.0  
**Maintainer:** DevOps Team  

## Table of Contents

1. [Overview](#overview)
2. [Environment Types](#environment-types)
3. [Configuration Files](#configuration-files)
4. [Environment Variables](#environment-variables)
5. [Secrets Management](#secrets-management)
6. [Service Configuration](#service-configuration)
7. [Network Configuration](#network-configuration)
8. [Validation](#validation)
9. [Migration Between Environments](#migration-between-environments)
10. [Best Practices](#best-practices)

## Overview

This guide covers environment configuration for all deployment stages of the SutazAI application, from local development to production.

### Configuration Hierarchy

```
.env (base configuration)
  └── .env.local (local overrides)
      └── .env.development (dev environment)
          └── .env.staging (staging environment)
              └── .env.production (production environment)
```

## Environment Types

### Local Development

- **Purpose**: Individual developer workstations
- **Characteristics**: Debug mode, verbose logging, mock services
- **Data**: Sample data, test accounts

### Development

- **Purpose**: Shared development server
- **Characteristics**: Latest features, frequent deployments
- **Data**: Development database, test integrations

### Staging

- **Purpose**: Pre-production testing
- **Characteristics**: Production-like, performance testing
- **Data**: Production snapshot, sanitized data

### Production

- **Purpose**: Live system
- **Characteristics**: Optimized, monitored, secured
- **Data**: Live data, real users

## Configuration Files

### Base Configuration (.env)

```bash
# Application
APP_NAME=SutazAI
APP_VERSION=1.0.0
ENVIRONMENT=development

# Server
HOST=0.0.0.0
BACKEND_PORT=10200
FRONTEND_PORT=11000

# Database - PostgreSQL
POSTGRES_HOST=sutazai-postgres
POSTGRES_PORT=5432
POSTGRES_USER=jarvis
POSTGRES_PASSWORD=${POSTGRES_PASSWORD}
POSTGRES_DB=jarvis_ai
DATABASE_URL=postgresql+asyncpg://${POSTGRES_USER}:${POSTGRES_PASSWORD}@${POSTGRES_HOST}:${POSTGRES_PORT}/${POSTGRES_DB}

# Cache - Redis
REDIS_HOST=sutazai-redis
REDIS_PORT=6379
REDIS_PASSWORD=${REDIS_PASSWORD}
REDIS_URL=redis://:${REDIS_PASSWORD}@${REDIS_HOST}:${REDIS_PORT}/0

# Message Queue - RabbitMQ
RABBITMQ_HOST=sutazai-rabbitmq
RABBITMQ_PORT=5672
RABBITMQ_USER=sutazai
RABBITMQ_PASSWORD=${RABBITMQ_PASSWORD}
RABBITMQ_VHOST=/

# Graph Database - Neo4j
NEO4J_HOST=sutazai-neo4j
NEO4J_BOLT_PORT=7687
NEO4J_HTTP_PORT=7474
NEO4J_AUTH=neo4j/${NEO4J_PASSWORD}

# Security
SECRET_KEY=${SECRET_KEY}
JWT_SECRET_KEY=${JWT_SECRET_KEY}
JWT_ALGORITHM=HS256
JWT_EXPIRATION_MINUTES=1440

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
LOG_FILE=/var/log/sutazai/app.log

# CORS
CORS_ORIGINS=http://localhost:3000,http://localhost:11000
CORS_CREDENTIALS=true
CORS_METHODS=GET,POST,PUT,DELETE,OPTIONS
CORS_HEADERS=*
```

### Development Environment (.env.development)

```bash
# Inherits from .env
ENVIRONMENT=development
DEBUG=true
LOG_LEVEL=DEBUG

# Development Database
POSTGRES_HOST=localhost
POSTGRES_PORT=10000

# Hot Reload
RELOAD=true
WATCH_FILES=true

# Testing
TEST_MODE=true
MOCK_EXTERNAL_SERVICES=true
```

### Staging Environment (.env.staging)

```bash
# Inherits from .env
ENVIRONMENT=staging
DEBUG=false
LOG_LEVEL=INFO

# Staging Database
POSTGRES_HOST=staging-db.sutazai.internal
POSTGRES_PORT=5432

# Performance
WORKERS=4
THREAD_POOL_SIZE=10

# Monitoring
METRICS_ENABLED=true
TRACING_ENABLED=true
```

### Production Environment (.env.production)

```bash
# Inherits from .env
ENVIRONMENT=production
DEBUG=false
LOG_LEVEL=WARNING

# Production Database
POSTGRES_HOST=prod-db-cluster.sutazai.internal
POSTGRES_PORT=5432
POSTGRES_SSL_MODE=require

# Performance
WORKERS=8
THREAD_POOL_SIZE=20
CONNECTION_POOL_SIZE=50

# Security
FORCE_HTTPS=true
SECURE_COOKIES=true
SESSION_COOKIE_SECURE=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=strict

# Rate Limiting
RATE_LIMIT_ENABLED=true
RATE_LIMIT_PER_MINUTE=100
RATE_LIMIT_PER_HOUR=1000
```

## Environment Variables

### Required Variables

| Variable | Description | Example |
|----------|-------------|---------|  
| `ENVIRONMENT` | Deployment environment | development |
| `SECRET_KEY` | Application secret | 32-char random string |
| `DATABASE_URL` | PostgreSQL connection | postgresql://... |
| `REDIS_URL` | Redis connection | redis://... |
| `JWT_SECRET_KEY` | JWT signing key | 64-char random string |

### Optional Variables

| Variable | Description | Default |
|----------|-------------|---------|  
| `DEBUG` | Debug mode | false |
| `LOG_LEVEL` | Logging level | INFO |
| `WORKERS` | Worker processes | 4 |
| `PORT` | Application port | 10200 |

### Service-Specific Variables

```bash
# MCP Bridge
MCP_BRIDGE_HOST=localhost
MCP_BRIDGE_PORT=11100
MCP_BRIDGE_WORKERS=4

# Vector Databases
CHROMA_HOST=localhost
CHROMA_PORT=10100
QDRANT_HOST=localhost
QDRANT_PORT=10101
FAISS_HOST=localhost
FAISS_PORT=10103

# AI Agents
AGENT_POOL_SIZE=10
AGENT_TIMEOUT=300
AGENT_MAX_RETRIES=3
```

## Secrets Management

### Local Development

```bash
# Generate secrets
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Store in .env.local (git-ignored)
SECRET_KEY=your_generated_secret
JWT_SECRET_KEY=your_jwt_secret
```

### Production Secrets

#### Using Docker Secrets

```bash
# Create secrets
echo "production_password" | docker secret create postgres_password -
echo "jwt_secret_key" | docker secret create jwt_secret -

# Reference in docker-compose.yml
secrets:
  postgres_password:
    external: true
  jwt_secret:
    external: true
```

#### Using HashiCorp Vault

```bash
# Store secrets
vault kv put secret/sutazai/prod \
  postgres_password="secure_password" \
  jwt_secret="jwt_secret_key"

# Retrieve secrets
vault kv get -field=postgres_password secret/sutazai/prod
```

#### Using AWS Secrets Manager

```bash
# Create secret
aws secretsmanager create-secret \
  --name sutazai/prod/database \
  --secret-string '{"password":"secure_password"}'

# Retrieve secret
aws secretsmanager get-secret-value \
  --secret-id sutazai/prod/database
```

## Service Configuration

### Database Configuration

```python
# backend/app/core/database.py
from sqlalchemy.ext.asyncio import create_async_engine
from app.core.config import settings

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=settings.DB_POOL_SIZE,
    max_overflow=settings.DB_MAX_OVERFLOW,
    pool_pre_ping=True,
    pool_recycle=3600,
    echo=settings.DEBUG
)
```

### Redis Configuration

```python
# backend/app/core/cache.py
import redis.asyncio as redis
from app.core.config import settings

redis_client = redis.from_url(
    settings.REDIS_URL,
    decode_responses=True,
    max_connections=settings.REDIS_MAX_CONNECTIONS
)
```

### RabbitMQ Configuration

```python
# backend/app/core/messaging.py
import aio_pika
from app.core.config import settings

async def get_rabbitmq_connection():
    return await aio_pika.connect_robust(
        f"amqp://{settings.RABBITMQ_USER}:{settings.RABBITMQ_PASSWORD}@"
        f"{settings.RABBITMQ_HOST}:{settings.RABBITMQ_PORT}/{settings.RABBITMQ_VHOST}"
    )
```

## Network Configuration

### Docker Network

```yaml
# docker-compose.yml
networks:
  sutazai-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1
```

### Service Discovery

```yaml
# Service naming convention
services:
  postgres:
    container_name: sutazai-postgres
    hostname: sutazai-postgres
    networks:
      sutazai-network:
        aliases:
          - postgres
          - database
```

## Validation

### Configuration Validation Script

```bash
#!/bin/bash
# scripts/validate_env.sh

# Check required variables
required_vars=(
  "ENVIRONMENT"
  "SECRET_KEY"
  "DATABASE_URL"
  "REDIS_URL"
  "JWT_SECRET_KEY"
)

for var in "${required_vars[@]}"; do
  if [ -z "${!var}" ]; then
    echo "Error: $var is not set"
    exit 1
  fi
done

# Test database connection
PGPASSWORD="$POSTGRES_PASSWORD" psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1"

# Test Redis connection
redis-cli -h "$REDIS_HOST" -p "$REDIS_PORT" -a "$REDIS_PASSWORD" ping

echo "Configuration validation successful"
```

### Health Checks

```python
# backend/app/api/health.py
from fastapi import APIRouter
from app.core.database import engine
from app.core.cache import redis_client

router = APIRouter()

@router.get("/health/config")
async def check_configuration():
    checks = {}
    
    # Database check
    try:
        async with engine.connect() as conn:
            await conn.execute("SELECT 1")
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {e}"
    
    # Redis check
    try:
        await redis_client.ping()
        checks["redis"] = "healthy"
    except Exception as e:
        checks["redis"] = f"unhealthy: {e}"
    
    return checks
```

## Migration Between Environments

### Promoting to Staging

```bash
# 1. Export development config
docker compose config > dev-config.yml

# 2. Update for staging
cp .env.development .env.staging
vim .env.staging  # Update staging-specific values

# 3. Deploy to staging
ENVIRONMENT=staging docker compose up -d
```

### Promoting to Production

```bash
# 1. Backup current production
./scripts/backup_production.sh

# 2. Update configuration
cp .env.staging .env.production
vim .env.production  # Update production values

# 3. Deploy with zero downtime
./scripts/deploy_production.sh --rolling
```

## Best Practices

### Security

1. **Never commit secrets**: Add `.env*` to `.gitignore`
2. **Use strong passwords**: Minimum 32 characters for secrets
3. **Rotate secrets regularly**: Quarterly for production
4. **Encrypt sensitive data**: Use TLS for all connections
5. **Audit access**: Log all configuration changes

### Organization

1. **Use hierarchical configs**: Base → Environment-specific
2. **Document all variables**: Purpose and expected values
3. **Version control templates**: Keep `.env.example` updated
4. **Validate before deployment**: Run configuration checks
5. **Maintain parity**: Keep environments as similar as possible

### Performance

1. **Tune connection pools**: Based on load testing
2. **Set appropriate timeouts**: Prevent hanging connections
3. **Configure caching**: Redis TTLs and eviction policies
4. **Monitor resource usage**: Adjust limits as needed
5. **Use environment-specific optimizations**: Debug off in production

## Related Documentation

- [Local Development Setup](./local_dev.md)
- [Secrets Manager Setup](../operations/infrastructure/storage.md)
- [Deployment Procedures](../operations/deployment/procedures.md)
- [Security Policies](../compliance/security_policies.md)
- [Monitoring Setup](../operations/monitoring/observability.md)

---

*For questions about environment configuration, contact the DevOps team via Slack #sutazai-ops.*