# Migration Guide: From Fantasy to Reality

## Overview

This guide helps you migrate from the current bloated 90+ service setup to a realistic, working system with 5-9 services.

## Current State Assessment

### Step 1: Check What's Actually Running
```bash
# See what containers are running
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check resource usage
docker stats --no-stream

# See disk usage
docker system df
```

### Step 2: Identify Critical Data
```bash
# Backup critical volumes
docker run --rm -v sutazai-postgres_data:/data -v $(pwd)/backup:/backup alpine tar czf /backup/postgres_backup.tar.gz /data
docker run --rm -v sutazai-redis_data:/data -v $(pwd)/backup:/backup alpine tar czf /backup/redis_backup.tar.gz /data
docker run --rm -v sutazai-ollama_data:/data -v $(pwd)/backup:/backup alpine tar czf /backup/ollama_backup.tar.gz /data
```

## Migration Process

### Phase 1: Stop Everything
```bash
# Stop all services
docker-compose down

# Keep the network and volumes
docker network ls | grep sutazai
docker volume ls | grep sutazai
```

### Phase 2: Clean Up Unused Resources
```bash
# Remove stopped containers
docker container prune -f

# Remove unused images (be careful!)
docker image prune -a

# Remove unused volumes (DANGER - this deletes data!)
# docker volume prune -f  # Only if you're sure!
```

### Phase 3: Archive the Old Configuration
```bash
# Backup old docker-compose files
mkdir -p archive/docker-compose-backup-$(date +%Y%m%d)
cp docker-compose*.yml archive/docker-compose-backup-$(date +%Y%m%d)/

# Keep the main one for reference
mv docker-compose.yml docker-compose.old.yml
```

### Phase 4: Deploy Simplified Stack
```bash
# Use the new simplified compose file
cp docker-compose.simple.yml docker-compose.yml

# Or use it directly
docker-compose -f docker-compose.simple.yml up -d
```

## Service-by-Service Migration

### Core Services (Keep These)

#### PostgreSQL
```yaml
# Old: Multiple database dependencies, complex setup
# New: Single postgres instance with proper health checks
postgres:
  image: postgres:16.3-alpine
  # Simplified configuration
```

#### Redis
```yaml
# Old: Used by 50+ services that don't exist
# New: Cache for backend only
redis:
  image: redis:7.2-alpine
  # Minimal configuration
```

#### Ollama
```yaml
# Old: Complex environment with unused features
# New: Essential settings only
ollama:
  environment:
    OLLAMA_HOST: 0.0.0.0
    OLLAMA_ORIGINS: '*'
    OLLAMA_KEEP_ALIVE: 10m
```

#### Backend
```yaml
# Old: 30+ environment variables, many unused
# New: Only required settings
backend:
  environment:
    DATABASE_URL: postgresql://...
    REDIS_URL: redis://...
    OLLAMA_BASE_URL: http://ollama:11434
```

#### Frontend
```yaml
# Old: Complex multi-framework setup
# New: Simple Streamlit app
frontend:
  command: streamlit run app.py
```

### Services to Remove

#### Fantasy AI Agents (Remove All)
- agentzero-coordinator
- agent-creator
- agent-debugger
- adversarial-attack-detector
- cognitive-architecture-designer
- quantum-computing-optimizer
- neuromorphic-computing-expert
- ... (80+ more)

**Why**: These are all stub implementations that return placeholder data.

#### Service Mesh Components
- kong (API gateway)
- consul (service discovery)
- rabbitmq (message queue)

**Why**: Not integrated, just consuming resources.

#### ML Training Services
- pytorch
- tensorflow
- jax

**Why**: No actual model training happening.

#### Workflow Tools
- langflow
- flowise
- n8n
- dify

**Why**: Not integrated with the system.

### Optional Services (Keep if Actually Using)

#### Neo4j
Keep if you're actually using graph database features.
```bash
# Check if Neo4j has any data
docker exec sutazai-neo4j cypher-shell -u neo4j -p $NEO4J_PASSWORD "MATCH (n) RETURN count(n)"
```

#### Monitoring Stack
Keep if you need system monitoring.
- prometheus
- grafana
- loki

## Configuration Changes

### Environment Variables

#### Old .env (100+ variables)
```env
# Complex, mostly unused
POSTGRES_PASSWORD=...
NEO4J_PASSWORD=...
GRAFANA_PASSWORD=...
JWT_SECRET=...
SECRET_KEY=...
REDIS_PASSWORD=...
CHROMADB_API_KEY=...
QDRANT_API_KEY=...
... (many more)
```

#### New .env (minimal)
```env
# Only what's actually used
POSTGRES_PASSWORD=your_secure_password
NEO4J_PASSWORD=your_neo4j_password
GRAFANA_PASSWORD=admin
SECRET_KEY=your_secret_key
JWT_SECRET=your_jwt_secret
SUTAZAI_ENV=production
```

### Port Mappings

#### Ports to Keep
- 10000: PostgreSQL
- 10001: Redis
- 10010: Backend API
- 10011: Frontend UI
- 10104: Ollama
- 10200-10202: Monitoring (optional)

#### Ports to Release
- 10300-10999: Agent services
- 11000-11999: More agents
- All other random ports

## Verification Steps

### After Migration

1. **Check Core Services**
```bash
# Verify services are running
docker-compose ps

# Test database connection
docker exec sutazai-postgres psql -U sutazai -c "SELECT 1"

# Test Redis
docker exec sutazai-redis redis-cli ping

# Test Ollama
curl http://localhost:10104/api/tags
```

2. **Test Application**
```bash
# Backend health
curl http://localhost:10010/health

# Frontend access
open http://localhost:10011
```

3. **Download Ollama Model**
```bash
# If not already present
docker exec sutazai-ollama ollama pull gpt-oss
```

## Rollback Plan

If something goes wrong:

```bash
# Stop new setup
docker-compose -f docker-compose.simple.yml down

# Restore old setup
mv docker-compose.old.yml docker-compose.yml
docker-compose up -d

# Restore data if needed
docker run --rm -v sutazai-postgres_data:/data -v $(pwd)/backup:/backup alpine tar xzf /backup/postgres_backup.tar.gz -C /
```

## Performance Improvements

### Before Migration
- **Containers**: 90+
- **RAM Usage**: 20+ GB
- **CPU Usage**: High constant load
- **Disk Usage**: 50+ GB
- **Startup Time**: 10+ minutes

### After Migration
- **Containers**: 5-9
- **RAM Usage**: 4-6 GB
- **CPU Usage**: Low, spike only during LLM inference
- **Disk Usage**: 5-10 GB
- **Startup Time**: 1-2 minutes

## Troubleshooting

### Common Issues

1. **"Cannot connect to backend"**
   - Ensure postgres and redis are healthy first
   - Check backend logs: `docker-compose logs backend`

2. **"Ollama not responding"**
   - Check if model is downloaded: `docker exec sutazai-ollama ollama list`
   - Verify Ollama is running: `docker ps | grep ollama`

3. **"Import errors in backend"**
   - Many imports are for non-existent enterprise features
   - Backend has try/catch blocks, should start anyway

4. **"Frontend won't load"**
   - Backend must be healthy first
   - Check: `curl http://localhost:10010/health`

## Next Steps

After successful migration:

1. **Clean up old files**
```bash
# Remove unused agent directories
rm -rf agents/*/  # Be careful!

# Remove unused Docker build contexts
rm -rf docker/*/  # Except the ones you need

# Clean up documentation of fantasy features
mkdir archive/old-docs
mv docs/*.md archive/old-docs/
```

2. **Update documentation**
- Replace fantasy documentation with real capabilities
- Update README to reflect actual system
- Remove references to non-existent features

3. **Implement one feature properly**
- Choose ONE agent to implement fully
- Make it actually work before adding more
- Test thoroughly

## Summary

The migration from 90+ services to 5-9 services will:
- Reduce resource usage by 80%
- Improve startup time by 90%
- Eliminate restart loops and crashes
- Make the system actually maintainable
- Allow focus on making features work vs managing complexity

Remember: **Less is more when it comes to actually working software!**