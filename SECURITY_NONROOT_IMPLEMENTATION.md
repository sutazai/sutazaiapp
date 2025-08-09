# Non-Root User Security Implementation Plan

## Current Security Audit Results

### Containers Running as ROOT (Critical Security Risk)
1. **sutazai-ai-agent-orchestrator** - Port 8589
2. **sutazai-rabbitmq** - Ports 10007/10008  
3. **sutazai-ollama** - Port 10104
4. **sutazai-redis** - Port 10001
5. **sutazai-postgres** - Port 10000
6. **sutazai-qdrant** - Ports 10101/10102
7. **sutazai-chromadb** - Port 10100
8. **sutazai-neo4j** - Ports 10002/10003

### Containers Already Using Non-Root Users (Compliant)
1. **sutazai-jarvis-automation-agent** - appuser
2. **sutazai-hardware-resource-optimizer** - appuser
3. **sutazai-jarvis-hardware-resource-optimizer** - appuser
4. **sutazai-ollama-integration** - appuser
5. **sutazai-grafana** - grafana
6. **sutazai-loki** - loki
7. **sutazai-prometheus** - nobody

## Implementation Strategy

### Phase 1: Create Secure Dockerfile Templates

#### For Python-based Services (AI Agent Orchestrator)
```dockerfile
FROM python:3.11-slim

# Create non-root user early
RUN groupadd -r appuser && useradd -r -g appuser -u 1001 appuser

WORKDIR /app

# Install dependencies as root
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY --chown=appuser:appuser . .

# Create necessary directories with proper permissions
RUN mkdir -p /app/logs /app/data && \
    chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

EXPOSE 8589
CMD ["python", "app.py"]
```

#### For Database Services

##### PostgreSQL (Custom Dockerfile)
```dockerfile
FROM postgres:16.3-alpine

# PostgreSQL already runs as 'postgres' user (UID 70)
# Ensure proper permissions on volumes
RUN mkdir -p /var/lib/postgresql/data && \
    chown -R postgres:postgres /var/lib/postgresql

USER postgres
```

##### Redis (Custom Dockerfile)
```dockerfile
FROM redis:7.2-alpine

# Create redis user if not exists
RUN addgroup -g 1001 -S redis 2>/dev/null || true && \
    adduser -S -H -u 1001 -G redis redis 2>/dev/null || true

# Fix permissions
RUN mkdir -p /data && chown -R redis:redis /data

USER redis
```

##### Neo4j (docker-compose adjustment)
```yaml
neo4j:
  image: neo4j:5.18
  user: "7474:7474"  # Neo4j's default non-root user
```

### Phase 2: Services Requiring Special Handling

#### Ollama (Requires GPU/System Access)
```dockerfile
FROM ollama/ollama:latest

# Create ollama user with specific UID
RUN groupadd -r ollama -g 1002 && \
    useradd -r -g ollama -u 1002 ollama

# Ensure model directory permissions
RUN mkdir -p /root/.ollama/models /tmp && \
    chown -R ollama:ollama /root/.ollama /tmp

USER ollama
```

#### RabbitMQ (Already has rabbitmq user)
```yaml
rabbitmq:
  image: rabbitmq:3.12-management-alpine
  user: "999:999"  # RabbitMQ's default non-root user
```

#### ChromaDB
```dockerfile
FROM chromadb/chroma:0.5.0

# Create chroma user
RUN groupadd -r chroma -g 1003 && \
    useradd -r -g chroma -u 1003 chroma

# Fix permissions for data directory
RUN mkdir -p /chroma/data && \
    chown -R chroma:chroma /chroma

USER chroma
```

#### Qdrant
```dockerfile
FROM qdrant/qdrant:v1.9.2

# Create qdrant user
RUN groupadd -r qdrant -g 1004 && \
    useradd -r -g qdrant -u 1004 qdrant

# Fix storage permissions
RUN mkdir -p /qdrant/storage && \
    chown -R qdrant:qdrant /qdrant

USER qdrant
```

## Implementation Steps

### Step 1: Create Custom Dockerfiles
Create custom Dockerfiles for services that don't have them:
- `/opt/sutazaiapp/docker/postgres-secure/Dockerfile`
- `/opt/sutazaiapp/docker/redis-secure/Dockerfile`
- `/opt/sutazaiapp/docker/ollama-secure/Dockerfile`
- `/opt/sutazaiapp/docker/chromadb-secure/Dockerfile`
- `/opt/sutazaiapp/docker/qdrant-secure/Dockerfile`

### Step 2: Update docker-compose.yml
Modify service definitions to use custom Dockerfiles or specify user directly:

```yaml
postgres:
  build:
    context: ./docker/postgres-secure
    dockerfile: Dockerfile
  # OR use user directive
  user: "999:999"

redis:
  build:
    context: ./docker/redis-secure
    dockerfile: Dockerfile
  # OR
  user: "999:999"
```

### Step 3: Fix Volume Permissions
Update volume mounts to ensure proper ownership:

```bash
# Fix PostgreSQL data
docker exec sutazai-postgres chown -R postgres:postgres /var/lib/postgresql/data

# Fix Redis data  
docker exec sutazai-redis chown -R redis:redis /data

# Fix other volumes
docker exec sutazai-neo4j chown -R neo4j:neo4j /data
docker exec sutazai-qdrant chown -R 1004:1004 /qdrant/storage
```

### Step 4: Testing Strategy

1. **Gradual Migration**: Update one service at a time
2. **Health Check Validation**: Ensure all health checks pass
3. **Functionality Testing**: Verify service operations
4. **Rollback Plan**: Keep original configurations backed up

## Security Benefits

1. **Reduced Attack Surface**: Containers can't escalate to root privileges
2. **Defense in Depth**: Additional layer of security
3. **Compliance**: Meets security best practices and audit requirements
4. **Container Escape Mitigation**: Limits damage if container is compromised

## Rollout Schedule

1. **Day 1**: Implement for stateless services (AI agents)
2. **Day 2**: Implement for cache/queue services (Redis, RabbitMQ)
3. **Day 3**: Implement for databases (PostgreSQL, Neo4j)
4. **Day 4**: Implement for vector databases (ChromaDB, Qdrant)
5. **Day 5**: Implement for LLM service (Ollama)

## Monitoring & Validation

### Pre-Implementation Checks
```bash
# Check current user for each container
for container in $(docker ps --format "{{.Names}}"); do 
  echo "$container: $(docker exec $container whoami 2>/dev/null)"
done
```

### Post-Implementation Validation
```bash
# Verify non-root implementation
for container in $(docker ps --format "{{.Names}}"); do 
  user=$(docker exec $container whoami 2>/dev/null)
  if [ "$user" = "root" ]; then
    echo "WARNING: $container still running as root"
  else
    echo "SUCCESS: $container running as $user"
  fi
done
```

## Common Issues & Solutions

### Issue 1: Permission Denied on Volumes
**Solution**: Pre-create volumes with correct ownership or use init containers

### Issue 2: Service Won't Start
**Solution**: Check if service requires specific UIDs/GIDs

### Issue 3: Cannot Bind to Privileged Ports
**Solution**: Use ports > 1024 or add CAP_NET_BIND_SERVICE capability

### Issue 4: Docker Socket Access Needed
**Solution**: Add user to docker group (security trade-off) or use Docker-in-Docker

## Exceptions & Justifications

Some services may require root for legitimate reasons:
- **Hardware monitoring agents**: Need access to /proc, /sys
- **Network services**: Binding to ports < 1024
- **Docker management**: Need Docker socket access

For these cases, implement compensating controls:
- Read-only root filesystem
- Dropped capabilities
- Security profiles (AppArmor/SELinux)
- Network policies

## Success Criteria

✅ All containers running with non-root users (except documented exceptions)
✅ All services remain functional after migration
✅ Health checks passing for all services
✅ No performance degradation
✅ Security scan shows reduced vulnerabilities