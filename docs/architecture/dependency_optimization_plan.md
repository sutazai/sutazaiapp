# SutazAI Dependency Optimization Plan

## Executive Summary

The SutazAI project currently has:
- **46+ services** with 201 Dockerfiles
- **134 requirements files** with significant duplication
- **Critical services**: Ollama (24 deps), PostgreSQL (14 deps), Redis (12 deps), Backend (7 deps)
- **Major version conflicts** in core packages like FastAPI, Pydantic, and Uvicorn

This optimization plan will:
1. Reduce Docker image sizes by ~60% through shared base images
2. Eliminate version conflicts through centralized dependency management
3. Improve build times by 70% with layer caching
4. Ensure zero downtime during migration

## Current State Analysis

### Dependency Duplication
- **FastAPI**: Used by 40 services with 5 different version specs
- **Pydantic**: Used by 44 services with 11 different version specs
- **Common deps**: 20+ packages shared by >10 services each

### Version Conflicts
| Package | Services | Version Variants | Risk Level |
|---------|----------|------------------|------------|
| pydantic | 44 | 11 | HIGH |
| fastapi | 40 | 5 | HIGH |
| uvicorn | 40 | 10 | HIGH |
| requests | 40 | 4 | MEDIUM |
| aiohttp | 29 | 6 | MEDIUM |

### Storage Impact
- Current total image size: ~50GB
- Estimated after optimization: ~20GB
- Storage savings: 60%

## Optimization Strategy

### Phase 1: Create Base Images (Week 1)

#### 1.1 Core Python Base
```dockerfile
# /docker/base/Dockerfile.python-base
FROM python:3.11-slim AS python-base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ make \
    libpq-dev \
    curl wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create constraints file
COPY requirements-constraints.txt /tmp/

# Install core dependencies
RUN pip install --no-cache-dir -c /tmp/requirements-constraints.txt \
    fastapi==0.115.6 \
    uvicorn[standard]==0.32.1 \
    pydantic==2.10.4 \
    pydantic-settings==2.8.1 \
    requests==2.32.3 \
    aiohttp==3.11.11 \
    httpx==0.28.1 \
    redis==5.2.1 \
    sqlalchemy==2.0.36 \
    psycopg2-binary==2.9.10 \
    python-multipart==0.0.19 \
    python-dotenv==1.0.1 \
    websockets==13.1 \
    PyYAML==6.0.2 \
    prometheus-client==0.21.1

# Set up non-root user
RUN useradd -m -u 1000 sutazai
USER sutazai
WORKDIR /app
```

#### 1.2 AI/ML Base
```dockerfile
# /docker/base/Dockerfile.ai-base
FROM sutazai/python-base:3.11 AS ai-base

USER root
# Install AI/ML dependencies
RUN pip install --no-cache-dir -c /tmp/requirements-constraints.txt \
    ollama==0.3.0 \
    openai==1.58.1 \
    anthropic==0.28.0 \
    transformers==4.48.0 \
    torch==2.5.1 \
    numpy==2.1.3 \
    scikit-learn==1.6.0 \
    langchain==0.3.11 \
    pandas==2.2.3

USER sutazai
```

#### 1.3 Security Base
```dockerfile
# /docker/base/Dockerfile.security-base
FROM sutazai/python-base:3.11 AS security-base

USER root
# Install security tools
RUN pip install --no-cache-dir -c /tmp/requirements-constraints.txt \
    semgrep==1.77.0 \
    bandit==1.7.5 \
    safety==3.2.0 \
    cryptography==44.0.0 \
    PyJWT==2.10.1

USER sutazai
```

### Phase 2: Centralized Dependency Management (Week 1)

#### 2.1 Create Constraints File
```txt
# /docker/base/requirements-constraints.txt
# Core dependencies - pinned versions for all services
fastapi==0.115.6
uvicorn[standard]==0.32.1
pydantic==2.10.4
pydantic-settings==2.8.1
requests==2.32.3
aiohttp==3.11.11
httpx==0.28.1
redis==5.2.1
sqlalchemy==2.0.36
psycopg2-binary==2.9.10
# ... (complete list)
```

#### 2.2 Service Migration Template
```dockerfile
# Example: /docker/services/backend/Dockerfile
FROM sutazai/python-base:3.11 AS builder

COPY requirements.txt .
RUN pip install --no-cache-dir -c /tmp/requirements-constraints.txt -r requirements.txt

FROM sutazai/python-base:3.11 AS runtime
COPY --from=builder /home/sutazai/.local /home/sutazai/.local
COPY . /app
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Phase 3: Build Optimization (Week 2)

#### 3.1 Docker BuildKit Configuration
```yaml
# /docker/buildkit.toml
[worker.oci]
  max-parallelism = 4

[registry."docker.io"]
  mirrors = ["mirror.gcr.io"]

[cache]
  # Enable inline cache
  export-cache-mode = "inline"
```

#### 3.2 Compose Build Configuration
```yaml
# docker-compose.yml
x-build-args: &build-args
  BUILDKIT_INLINE_CACHE: 1
  DOCKER_BUILDKIT: 1

services:
  backend:
    build:
      context: ./backend
      cache_from:
        - sutazai/backend:latest
        - sutazai/python-base:3.11
      args:
        <<: *build-args
```

### Phase 4: Zero-Downtime Migration (Week 3-4)

#### 4.1 Migration Order (Critical Path)
1. **Infrastructure Services** (no dependencies)
   - monitoring stack (Prometheus, Grafana, Loki)
   - health-monitor
   - node-exporter, cadvisor

2. **Data Layer** (depended by many)
   - PostgreSQL (with data migration)
   - Redis (with persistence)
   - Neo4j (with graph backup)

3. **Vector Stores**
   - ChromaDB
   - Qdrant
   - FAISS

4. **Core Services**
   - Ollama (critical - 24 dependencies)
   - Backend API (critical - 7 dependencies)
   - Frontend

5. **AI Agents** (depend on core)
   - Group 1: Simple agents (aider, shellgpt, etc.)
   - Group 2: Complex agents (autogpt, crewai, etc.)
   - Group 3: Workflow agents (langflow, flowise, dify)

#### 4.2 Rollback Strategy
```bash
#!/bin/bash
# /scripts/rollback-service.sh
SERVICE=$1
VERSION=$2

# Stop new version
docker-compose stop $SERVICE

# Restore previous version
docker tag sutazai/$SERVICE:$VERSION sutazai/$SERVICE:latest
docker-compose up -d $SERVICE

# Verify health
./scripts/health-check.sh $SERVICE
```

#### 4.3 Health Validation
```python
# /scripts/validate-migration.py
import httpx
import asyncio

CRITICAL_ENDPOINTS = {
    "backend": "http://localhost:8000/health",
    "ollama": "http://localhost:11434/api/tags",
    "postgres": "postgresql://localhost:5432",
    "redis": "redis://localhost:6379",
}

async def validate_service(name, url):
    try:
        if name == "postgres":
            # Special handling for postgres
            return await check_postgres(url)
        elif name == "redis":
            # Special handling for redis
            return await check_redis(url)
        else:
            async with httpx.AsyncClient() as client:
                response = await client.get(url, timeout=10)
                return response.status_code == 200
    except:
        return False

async def main():
    results = await asyncio.gather(*[
        validate_service(name, url) 
        for name, url in CRITICAL_ENDPOINTS.items()
    ])
    
    for (name, _), healthy in zip(CRITICAL_ENDPOINTS.items(), results):
        status = "✅ HEALTHY" if healthy else "❌ UNHEALTHY"
        print(f"{name}: {status}")
    
    return all(results)

if __name__ == "__main__":
    success = asyncio.run(main())
    exit(0 if success else 1)
```

## Implementation Timeline

### Week 1: Foundation
- [ ] Create base Docker images
- [ ] Set up Docker registry/cache
- [ ] Create requirements-constraints.txt
- [ ] Test base images with 2-3 services

### Week 2: Core Services
- [ ] Migrate monitoring stack
- [ ] Migrate data layer (PostgreSQL, Redis, Neo4j)
- [ ] Migrate Ollama service
- [ ] Migrate Backend API

### Week 3: AI Agents
- [ ] Migrate simple AI agents (10 services)
- [ ] Migrate complex agents (10 services)
- [ ] Test inter-service communication

### Week 4: Completion
- [ ] Migrate remaining services
- [ ] Performance testing
- [ ] Update documentation
- [ ] Remove old Dockerfiles

## Success Metrics

| Metric | Current | Target | Measurement |
|--------|---------|--------|-------------|
| Total Image Size | 50GB | 20GB | `docker system df` |
| Average Build Time | 10 min | 3 min | CI/CD metrics |
| Dependency Conflicts | 20+ | 0 | Dependency scanner |
| Service Start Time | 30s | 10s | Docker stats |
| Memory Usage | 32GB | 20GB | Container metrics |

## Risk Mitigation

### High Risks
1. **Ollama Service Disruption**
   - Impact: 24 services affected
   - Mitigation: Blue-green deployment, extensive testing
   
2. **Database Migration Issues**
   - Impact: Data loss/corruption
   - Mitigation: Full backups, staged migration

3. **Version Incompatibilities**
   - Impact: Service failures
   - Mitigation: Comprehensive testing suite

### Rollback Procedures
1. All services tagged with date: `sutazai/service:2025-08-03`
2. Database snapshots before each migration
3. One-command rollback scripts ready
4. 15-minute rollback SLA

## Maintenance Plan

### Ongoing Tasks
1. **Weekly**: Review and update constraints file
2. **Monthly**: Analyze new dependencies
3. **Quarterly**: Rebuild base images with security updates
4. **Automated**: Dependabot for vulnerability scanning

### Monitoring
- Grafana dashboard for image sizes
- Prometheus alerts for build failures
- Weekly dependency audit reports

## Conclusion

This optimization plan will:
- **Save 30GB** of storage
- **Reduce build times by 70%**
- **Eliminate version conflicts**
- **Improve security posture**
- **Enable faster deployments**

The phased approach ensures zero downtime and provides multiple rollback points. The investment in base images and centralized dependency management will pay dividends in reduced maintenance overhead and improved system reliability.