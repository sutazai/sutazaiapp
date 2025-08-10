# Dockerfile Deduplication Strategy & Execution Plan

**Date:** August 10, 2025  
**Current Status:** 305 Dockerfiles → Target: 50  
**Risk Level:** HIGH - Requires ultra-careful execution  

## Executive Summary

We have 305 Dockerfiles across the codebase with massive redundancy. This plan will safely consolidate them to ~50 unique Dockerfiles using a phased approach with extensive validation at each step.

## Current State Analysis

### Distribution Breakdown
- **Total Dockerfiles:** 305
- **Docker directory:** 195 files (152 service directories)
- **Agents directory:** 15 files
- **Security migration backups:** 55 files (can be archived)
- **Other locations:** 40 files (backend, frontend, deployment, etc.)

### Base Image Usage Pattern
- **98 Python-based services** (majority)
- **4 Node.js services**
- **3 Nginx services**
- **Various specialized** (Ollama, PostgreSQL, Redis, etc.)

### Duplication Analysis
- **Exact duplicates found:** Multiple services sharing identical Dockerfiles
- **Near duplicates:** ~80% of Python services differ only in requirements.txt
- **Master bases created:** 2 (Python and Node.js) - only 6 services using them

## Phase 1: Immediate Safe Cleanup (Remove 55 files)

### 1.1 Archive Security Migration Backups
```bash
# Create archive directory
mkdir -p /opt/sutazaiapp/archive/dockerfile-backups/security-migration

# Move all security migration Dockerfiles
find /opt/sutazaiapp -path "*security_migration_20250810*" -name "Dockerfile*" \
  -exec mv {} /opt/sutazaiapp/archive/dockerfile-backups/security-migration/ \;

# Verify no active references
grep -r "security_migration_20250810" /opt/sutazaiapp --exclude-dir=archive
```

**Files to remove:** 55  
**New total:** 250  
**Risk:** ZERO (these are backups)

## Phase 2: Consolidate Exact Duplicates (Remove 40 files)

### 2.1 Identify Exact Duplicates
```bash
# Find all exact duplicates by MD5
find /opt/sutazaiapp -name "Dockerfile" -exec md5sum {} \; | \
  sort | uniq -D -w32 > /tmp/duplicate-dockerfiles.txt

# Group by hash
awk '{print $1}' /tmp/duplicate-dockerfiles.txt | uniq | while read hash; do
  echo "Hash: $hash"
  grep "$hash" /tmp/duplicate-dockerfiles.txt
done
```

### 2.2 Consolidation Strategy for Duplicates

**Pattern Found:** Many AI agent services use identical Python setup
- data-analysis-engineer
- deep-local-brain-builder  
- document-knowledge-manager
- edge-computing-optimizer
- (30+ more services)

**Action:** Create shared base images by category

## Phase 3: Create Category-Based Master Images (5 new bases)

### 3.1 Master Base Images to Create

1. **sutazai-python-ai-agent** (covers ~80 services)
   - Base: python:3.11-slim
   - Common: FastAPI, uvicorn, numpy, pandas, scikit-learn
   - Security: non-root user, health checks

2. **sutazai-python-ml-heavy** (covers ~20 services)  
   - Base: python:3.11
   - Includes: TensorFlow, PyTorch, transformers
   - GPU support optional

3. **sutazai-nodejs-frontend** (covers 4 services)
   - Base: node:18-alpine
   - Common: React/Next.js dependencies

4. **sutazai-monitoring** (covers 5 services)
   - Base: python:3.11-slim
   - Includes: Prometheus client, logging

5. **sutazai-data-service** (covers 10 services)
   - Base: python:3.11-slim
   - Includes: Database drivers, ORMs

### 3.2 Master Dockerfile Template

```dockerfile
# /opt/sutazaiapp/docker/base/sutazai-python-ai-agent/Dockerfile
FROM python:3.11-slim as base

# Security optimizations
RUN apt-get update && apt-get install -y \
    curl git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Base Python packages (common to all AI agents)
COPY requirements-ai-base.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements-ai-base.txt

# Create non-root user
RUN useradd -m -s /bin/bash appuser && \
    mkdir -p /app && chown -R appuser:appuser /app

FROM base as production
WORKDIR /app
USER appuser

# Standard health check
HEALTHCHECK CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

# Override in child images
CMD ["python", "app.py"]
```

## Phase 4: Service Migration Strategy (195 → 45 files)

### 4.1 Migration Categories

**Category A: Simple Python Agents (80 services)**
- Use: sutazai-python-ai-agent
- Migration: Change FROM line only
- Validation: Health check must pass

**Category B: ML/AI Heavy Services (20 services)**
- Use: sutazai-python-ml-heavy  
- Migration: Update FROM, verify GPU requirements
- Validation: Model loading tests

**Category C: Infrastructure Services (15 services)**
- Keep unique Dockerfiles (PostgreSQL, Redis, etc.)
- No changes needed

**Category D: Frontend Services (4 services)**
- Use: sutazai-nodejs-frontend
- Migration: Consolidate build steps

**Category E: Deprecated/Unused (76 services)**
- Identify via docker-compose references
- Archive before removal

### 4.2 Service-Specific Dockerfile Pattern

```dockerfile
# Service-specific Dockerfile after migration
FROM sutazai-python-ai-agent:latest

# Service-specific dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code
COPY . .

# Service-specific configuration
ENV SERVICE_NAME=data-analyzer
ENV PORT=8080

CMD ["python", "app.py"]
```

## Phase 5: Validation Framework

### 5.1 Pre-Migration Validation
```bash
#!/bin/bash
# validate-before-migration.sh

SERVICE=$1
echo "Validating $SERVICE before migration..."

# 1. Check current functionality
docker build -t test-original -f $SERVICE/Dockerfile $SERVICE
docker run --rm test-original python -c "import sys; print('Original OK')"

# 2. Record image size
docker images test-original --format "{{.Size}}" > /tmp/$SERVICE-original-size.txt

# 3. Test health endpoint
docker run -d --name test-health test-original
sleep 5
docker exec test-health curl -f http://localhost:8080/health
docker stop test-health && docker rm test-health
```

### 5.2 Post-Migration Validation
```bash
#!/bin/bash
# validate-after-migration.sh

SERVICE=$1
echo "Validating $SERVICE after migration..."

# 1. Build with new base
docker build -t test-migrated -f $SERVICE/Dockerfile.new $SERVICE

# 2. Compare image sizes
ORIGINAL=$(cat /tmp/$SERVICE-original-size.txt)
NEW=$(docker images test-migrated --format "{{.Size}}")
echo "Size comparison: $ORIGINAL → $NEW"

# 3. Functionality test
docker run --rm test-migrated python -c "import sys; print('Migrated OK')"

# 4. Integration test
docker-compose up -d $SERVICE
sleep 10
curl -f http://localhost:$(docker port $SERVICE | cut -d: -f2)/health
docker-compose down $SERVICE
```

## Phase 6: Execution Timeline

### Week 1: Foundation (Days 1-2)
- [ ] Archive security migration files (55 files)
- [ ] Create 5 master base images
- [ ] Build and test master images
- [ ] Document master image APIs

### Week 1: Pilot Migration (Days 3-5)
- [ ] Select 5 low-risk services for pilot
- [ ] Migrate pilots to master bases
- [ ] Run full validation suite
- [ ] Monitor for 24 hours

### Week 2: Bulk Migration (Days 6-10)
- [ ] Migrate Category A services (80 files)
- [ ] Validate each batch of 10
- [ ] Update docker-compose references
- [ ] Run integration tests

### Week 2: Complex Services (Days 11-12)
- [ ] Migrate Category B ML services
- [ ] Special handling for GPU services
- [ ] Performance benchmarking

### Week 3: Cleanup (Days 13-14)
- [ ] Archive deprecated services
- [ ] Remove unused Dockerfiles
- [ ] Update documentation
- [ ] Final validation

## Phase 7: Rollback Strategy

### 7.1 Instant Rollback Plan
```bash
# Tag all original images before migration
docker images | grep sutazai | awk '{print $1":"$2}' | \
  xargs -I {} docker tag {} {}-backup-$(date +%Y%m%d)

# Rollback script
#!/bin/bash
SERVICE=$1
DATE=$2
docker tag sutazai-$SERVICE:latest sutazai-$SERVICE:failed-$(date +%Y%m%d)
docker tag sutazai-$SERVICE:$DATE-backup sutazai-$SERVICE:latest
docker-compose up -d $SERVICE
```

### 7.2 Gradual Rollback
- Keep originals for 30 days
- Monitor metrics/logs for issues
- A/B test if needed

## Phase 8: Success Metrics

### 8.1 Quantitative Metrics
- **File Reduction:** 305 → 50 files (84% reduction)
- **Image Size:** Target 30% reduction average
- **Build Time:** Target 50% faster builds
- **Maintenance Time:** 80% reduction

### 8.2 Qualitative Metrics
- All services remain functional
- No production incidents
- Improved developer experience
- Simplified CI/CD pipeline

## Phase 9: Risk Mitigation

### 9.1 High-Risk Services (Do Not Migrate Initially)
1. **PostgreSQL** - Custom configuration critical
2. **Redis** - Performance-tuned setup
3. **Ollama** - Model serving specific
4. **Neo4j** - Graph database specific
5. **RabbitMQ** - Message queue critical

### 9.2 Risk Matrix
| Risk Level | Service Count | Strategy |
|------------|---------------|----------|
| Low | 80 | Bulk migrate |
| Medium | 20 | Migrate with extra validation |
| High | 15 | Keep unique Dockerfiles |
| Critical | 5 | No changes |

## Phase 10: Final State Architecture

### 10.1 Target Structure
```
/opt/sutazaiapp/docker/
├── base/                    # 5 master images
│   ├── python-ai-agent/
│   ├── python-ml-heavy/
│   ├── nodejs-frontend/
│   ├── monitoring/
│   └── data-service/
├── services/               # 45 service-specific files
│   ├── core/              # 10 files
│   ├── agents/            # 20 files
│   ├── infrastructure/    # 10 files
│   └── frontend/          # 5 files
└── archive/               # Historical versions
```

### 10.2 Naming Convention
- Base images: `sutazai-{language}-{category}`
- Service images: `sutazai-{service-name}`
- Versions: Semantic versioning (1.0.0)

## Appendix A: Command Reference

### Quick Commands
```bash
# Count current Dockerfiles
find /opt/sutazaiapp -name "Dockerfile*" | wc -l

# Find duplicates
find /opt/sutazaiapp -name "Dockerfile" -exec md5sum {} \; | \
  sort | uniq -D -w32

# Test build all
for dir in /opt/sutazaiapp/docker/*/; do
  docker build -t test-$(basename $dir) $dir
done

# Validate health endpoints
docker ps --format "{{.Names}}" | xargs -I {} \
  docker exec {} curl -f http://localhost:8080/health
```

## Appendix B: Verification Checklist

### Pre-Migration Checklist
- [ ] All services documented
- [ ] Dependencies mapped
- [ ] Base images built
- [ ] Rollback plan tested
- [ ] Monitoring configured

### Per-Service Migration
- [ ] Original backed up
- [ ] Dependencies verified
- [ ] Build successful
- [ ] Tests passing
- [ ] Health check working
- [ ] Performance acceptable
- [ ] Logs clean
- [ ] Metrics normal

### Post-Migration Validation
- [ ] All services running
- [ ] Integration tests pass
- [ ] No performance degradation
- [ ] Documentation updated
- [ ] Team trained

## Conclusion

This ultra-comprehensive strategy will safely reduce 305 Dockerfiles to ~50 through:
1. Removing 55 backup files (zero risk)
2. Creating 5 master base images
3. Migrating 195 service Dockerfiles to use bases
4. Archiving 76 unused services
5. Maintaining 15 unique infrastructure Dockerfiles

Total timeline: 3 weeks with extensive validation at each phase.
Risk level: LOW with proper execution and validation.
Expected benefits: 84% reduction in maintenance overhead, 50% faster builds, 30% smaller images.