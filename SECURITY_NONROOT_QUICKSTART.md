# Quick Start Guide: Non-Root Container Security

## Current Status
- **8 of 15 containers running as root** (Critical security risk)
- **Security Score: 46%** (Needs immediate attention)
- 2 containers running in privileged mode (hardware optimizers)

## Containers Requiring Migration
| Container | Current User | Target User | Priority |
|-----------|-------------|-------------|----------|
| sutazai-postgres | root | postgres (70:70) | HIGH |
| sutazai-redis | root | redis (999:999) | HIGH |
| sutazai-rabbitmq | root | rabbitmq (999:999) | HIGH |
| sutazai-neo4j | root | neo4j (7474:7474) | MEDIUM |
| sutazai-ollama | root | ollama (1002:1002) | MEDIUM |
| sutazai-chromadb | root | chroma (1003:1003) | MEDIUM |
| sutazai-qdrant | root | qdrant (1004:1004) | MEDIUM |
| sutazai-ai-agent-orchestrator | root | appuser (1001:1001) | LOW |

## Quick Implementation Steps

### Option 1: Automated Migration (Recommended)
```bash
# Run the automated migration script
sudo /opt/sutazaiapp/scripts/security/migrate_to_nonroot.sh

# Select option 1 for gradual migration (safer)
# or option 2 for full migration (faster)
```

### Option 2: Manual Migration Per Service

#### Step 1: Build Secure Images
```bash
# Build all secure Docker images
cd /opt/sutazaiapp

# PostgreSQL
docker build -t sutazai-postgres-secure:latest docker/postgres-secure/

# Redis
docker build -t sutazai-redis-secure:latest docker/redis-secure/

# Ollama
docker build -t sutazai-ollama-secure:latest docker/ollama-secure/

# ChromaDB
docker build -t sutazai-chromadb-secure:latest docker/chromadb-secure/

# Qdrant
docker build -t sutazai-qdrant-secure:latest docker/qdrant-secure/

# AI Agent Orchestrator
docker build -t sutazai-ai-agent-orchestrator-secure:latest \
  -f agents/ai_agent_orchestrator/Dockerfile.secure \
  agents/ai_agent_orchestrator/
```

#### Step 2: Update docker-compose.yml
Replace service definitions with secure versions:

```yaml
# Example for PostgreSQL
postgres:
  image: sutazai-postgres-secure:latest
  user: "70:70"
  security_opt:
    - no-new-privileges:true
  # ... rest of configuration

# Example for Redis  
redis:
  image: sutazai-redis-secure:latest
  user: "999:999"
  security_opt:
    - no-new-privileges:true
  # ... rest of configuration
```

#### Step 3: Fix Volume Permissions
```bash
# Fix volume ownership before restarting
docker run --rm -v sutazaiapp_postgres_data:/data alpine \
  chown -R 70:70 /data

docker run --rm -v sutazaiapp_redis_data:/data alpine \
  chown -R 999:999 /data

docker run --rm -v sutazaiapp_ollama_data:/data alpine \
  chown -R 1002:1002 /data

docker run --rm -v sutazaiapp_chromadb_data:/data alpine \
  chown -R 1003:1003 /data

docker run --rm -v sutazaiapp_qdrant_data:/data alpine \
  chown -R 1004:1004 /data
```

#### Step 4: Restart Services
```bash
# Stop and remove old containers
docker-compose down

# Start with secure configuration
docker-compose -f docker-compose.secure.yml up -d
```

## Validation

### Check Implementation Success
```bash
# Run validation script
/opt/sutazaiapp/scripts/security/validate_nonroot.sh

# Expected output:
# - All containers should show non-root users
# - Security score should be > 90%
```

### Quick Security Check
```bash
# Check specific container
docker exec sutazai-postgres whoami
# Should output: postgres (not root)

# Check all containers at once
for c in $(docker ps --format "{{.Names}}"); do
  echo "$c: $(docker exec $c whoami 2>/dev/null)"
done
```

## Troubleshooting

### Container Won't Start
```bash
# Check logs
docker logs sutazai-[service-name]

# Common fix: permissions on volumes
docker run --rm -v [volume-name]:/data alpine \
  chown -R [uid]:[gid] /data
```

### Service Connection Issues
```bash
# Verify service is running
docker ps | grep sutazai-[service-name]

# Check health status
docker inspect sutazai-[service-name] | grep -A5 Health

# Test connectivity
docker exec sutazai-backend curl http://[service]:port/health
```

### Database Access Issues
```bash
# PostgreSQL permission fix
docker exec sutazai-postgres chown -R postgres:postgres /var/lib/postgresql/data

# Redis permission fix
docker exec sutazai-redis chown -R redis:redis /data
```

## Rollback Procedure

If issues occur:
```bash
# Stop all services
docker-compose down

# Restore original configuration
cp /opt/sutazaiapp/backups/security-migration-[timestamp]/docker-compose.yml.backup \
   /opt/sutazaiapp/docker-compose.yml

# Restart with original configuration
docker-compose up -d
```

## Success Criteria

✅ All containers running with non-root users (validate_nonroot.sh shows 100%)
✅ All services healthy and responding
✅ No functionality degradation
✅ Security score improved to > 90%
✅ Volume permissions correctly set

## Next Steps After Migration

1. **Enable Additional Security Features**
   ```yaml
   security_opt:
     - no-new-privileges:true
     - apparmor:docker-default
   read_only: true  # Where applicable
   ```

2. **Implement Network Policies**
   - Restrict inter-container communication
   - Use internal networks for service mesh

3. **Regular Security Scans**
   ```bash
   # Install Trivy
   docker run --rm -v /var/run/docker.sock:/var/run/docker.sock \
     aquasec/trivy image sutazai-postgres-secure:latest
   ```

4. **Monitor and Audit**
   - Set up alerts for root user detection
   - Regular security validation runs
   - Audit container capabilities

## Support

- Logs: `/opt/sutazaiapp/logs/security-migration-*.log`
- Validation Reports: `/opt/sutazaiapp/logs/security-validation-*.txt`
- Backups: `/opt/sutazaiapp/backups/security-migration-*/`

## Important Notes

⚠️ **Hardware Optimizers**: Currently require privileged mode for system monitoring. Consider:
- Using specific capabilities instead of full privileged mode
- Implementing monitoring via external agents
- Using read-only bind mounts for /proc and /sys

⚠️ **Testing**: Always test in development environment first

⚠️ **Backup**: Ensure backups exist before migration

## Quick Reference

| Service | Non-Root User | UID:GID | Port |
|---------|--------------|---------|------|
| PostgreSQL | postgres | 70:70 | 10000 |
| Redis | redis | 999:999 | 10001 |
| Neo4j | neo4j | 7474:7474 | 10002/10003 |
| Ollama | ollama | 1002:1002 | 10104 |
| ChromaDB | chroma | 1003:1003 | 10100 |
| Qdrant | qdrant | 1004:1004 | 10101/10102 |
| RabbitMQ | rabbitmq | 999:999 | 10007/10008 |
| AI Orchestrator | appuser | 1001:1001 | 8589 |