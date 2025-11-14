# Portainer Stack Deployment Guide
**SutazAI Platform - Production Deployment**

## Prerequisites

### 1. System Requirements
- Docker Engine 20.10+ installed
- Portainer CE/BE 2.19+ installed
- Minimum 4GB RAM available
- 20GB free disk space
- Network access to ports 10000-11434

### 2. External Services
Ensure Ollama is running on the **host machine** (not in Docker):

```bash
# Check Ollama status
curl http://localhost:11434/api/version

# If not running, start Ollama
ollama serve &

# Pull required model
ollama pull tinyllama:latest

# Verify model loaded
ollama list | grep tinyllama
```

### 3. Docker Network Setup
Create the external network before deploying:

```bash
# Check if network exists
docker network ls | grep sutazaiapp_sutazai-network

# If not present, create it
docker network create \
  --driver bridge \
  --subnet 172.20.0.0/16 \
  sutazaiapp_sutazai-network

# Verify creation
docker network inspect sutazaiapp_sutazai-network
```

---

## Deployment Steps

### Option A: Deploy via Portainer UI (Recommended)

1. **Access Portainer**
   - Open web browser
   - Navigate to `http://localhost:9000` (or your Portainer URL)
   - Log in with admin credentials

2. **Navigate to Stacks**
   - Click "Stacks" in left sidebar
   - Click "+ Add stack" button

3. **Configure Stack**
   - **Name**: `sutazai-platform`
   - **Build method**: Choose "Upload" or "Web editor"
   
   **If Upload**:
   - Click "Upload"
   - Select `/opt/sutazaiapp/docker-compose-portainer.yml`
   
   **If Web Editor**:
   - Copy contents of `docker-compose-portainer.yml`
   - Paste into editor

4. **Environment Variables** (Optional)
   No additional variables needed - all configured in compose file

5. **Deploy Stack**
   - Scroll to bottom
   - Click "Deploy the stack"
   - Wait for deployment (2-3 minutes)

6. **Verify Deployment**
   - Click on `sutazai-platform` stack
   - Check all 11 containers show "running" status
   - Wait for health checks (green indicators)

### Option B: Deploy via Portainer API

```bash
# Get auth token (replace credentials)
TOKEN=$(curl -s -X POST "http://localhost:9000/api/auth" \
  -H "Content-Type: application/json" \
  -d '{"Username":"admin","Password":"YOUR_PASSWORD"}' | \
  jq -r '.jwt')

# Deploy stack
curl -X POST "http://localhost:9000/api/stacks?type=1&method=file&endpointId=1" \
  -H "X-API-Key: $TOKEN" \
  -F "Name=sutazai-platform" \
  -F "file=@/opt/sutazaiapp/docker-compose-portainer.yml"
```

### Option C: Deploy via Docker Compose CLI

```bash
# Change to project directory
cd /opt/sutazaiapp

# Deploy using Portainer-ready compose file
docker-compose -f docker-compose-portainer.yml up -d

# Check status
docker-compose -f docker-compose-portainer.yml ps
```

---

## Post-Deployment Verification

### 1. Container Health Checks

```bash
# Check all containers running
docker ps --filter "name=sutazai-" --format "table {{.Names}}\t{{.Status}}"

# Expected output: 11 containers, all "Up" with "(healthy)" status
```

### 2. Service Health Endpoints

```bash
# Backend health
curl http://localhost:10200/health/detailed

# Expected: {"status":"healthy","services":{"postgres":"connected",...}}

# Frontend health
curl http://localhost:11000/_stcore/health

# Expected: ok
```

### 3. Integration Test Suite

```bash
cd /opt/sutazaiapp
bash tests/integration/test_integration.sh

# Expected: ✅ ALL INTEGRATION TESTS PASSED - PRODUCTION READY
```

### 4. Web Interface Access

**Frontend UI**:
- URL: `http://localhost:11000`
- Expected: JARVIS interface loads with 4 tabs

**Backend API Docs**:
- URL: `http://localhost:10200/docs`
- Expected: FastAPI Swagger UI

**Consul Dashboard**:
- URL: `http://localhost:10006`
- Expected: Consul service registry

**RabbitMQ Management**:
- URL: `http://localhost:10005`
- Credentials: `jarvis` / `sutazai2024`

---

## Service Port Reference

| Service | Internal Port | External Port | URL |
|---------|---------------|---------------|-----|
| PostgreSQL | 5432 | 10000 | postgres://172.20.0.10:5432 |
| Redis | 6379 | 10001 | redis://172.20.0.11:6379 |
| Neo4j HTTP | 7474 | 10002 | http://localhost:10002 |
| Neo4j Bolt | 7687 | 10003 | bolt://172.20.0.12:7687 |
| RabbitMQ AMQP | 5672 | 10004 | amqp://172.20.0.13:5672 |
| RabbitMQ Mgmt | 15672 | 10005 | http://localhost:10005 |
| Consul HTTP | 8500 | 10006 | http://localhost:10006 |
| Consul DNS | 8600 | 10007 | dns://localhost:10007 |
| Kong Proxy | 8000 | 10008 | http://localhost:10008 |
| Kong Admin | 8001 | 10009 | http://localhost:10009 |
| ChromaDB | 8000 | 10100 | http://localhost:10100 |
| Qdrant HTTP | 6333 | 10101 | http://localhost:10101 |
| Qdrant gRPC | 6334 | 10102 | grpc://localhost:10102 |
| FAISS | 8000 | 10103 | http://localhost:10103 |
| Backend API | 8000 | 10200 | http://localhost:10200 |
| Frontend UI | 11000 | 11000 | http://localhost:11000 |
| Ollama (host) | 11434 | 11434 | http://localhost:11434 |

---

## Troubleshooting

### Issue: Network not found

**Error**: `network sutazaiapp_sutazai-network not found`

**Solution**:
```bash
docker network create --driver bridge --subnet 172.20.0.0/16 sutazaiapp_sutazai-network
```

### Issue: Ollama not accessible

**Error**: Backend logs show "Connection refused to Ollama"

**Solution**:
```bash
# Ensure Ollama running on host
ollama serve &

# Test from container
docker exec sutazai-backend wget -q -O- http://host.docker.internal:11434/api/version
```

### Issue: Container fails to start

**Check logs**:
```bash
docker logs sutazai-<service-name> --tail 50
```

**Common fixes**:
- Ensure all `depends_on` services are healthy first
- Check port conflicts: `netstat -tulpn | grep <port>`
- Verify volume permissions
- Restart unhealthy dependency services

### Issue: Health check fails

**Check health status**:
```bash
docker inspect sutazai-<service> --format '{{json .State.Health}}' | jq
```

**Common causes**:
- Service needs more start time (increase `start_period` in compose)
- Missing dependencies (check `depends_on`)
- Configuration error (check environment variables)

### Issue: Port already in use

**Find conflicting process**:
```bash
sudo lsof -i :<port>
# or
sudo netstat -tulpn | grep :<port>
```

**Solution**:
- Stop conflicting service
- Or change port in `docker-compose-portainer.yml`

---

## Stack Management

### Update Stack

**Via Portainer UI**:
1. Navigate to Stacks → `sutazai-platform`
2. Click "Editor"
3. Modify compose content
4. Click "Update the stack"

**Via CLI**:
```bash
cd /opt/sutazaiapp
docker-compose -f docker-compose-portainer.yml up -d --force-recreate
```

### Stop Stack

**Via Portainer UI**:
1. Navigate to Stacks → `sutazai-platform`
2. Click "Stop this stack"

**Via CLI**:
```bash
docker-compose -f docker-compose-portainer.yml stop
```

### Remove Stack

**Via Portainer UI**:
1. Navigate to Stacks → `sutazai-platform`
2. Click "Delete this stack"
3. Select "Remove associated volumes" (optional)

**Via CLI**:
```bash
# Stop and remove containers
docker-compose -f docker-compose-portainer.yml down

# Also remove volumes (WARNING: deletes all data)
docker-compose -f docker-compose-portainer.yml down -v
```

### View Logs

**Via Portainer UI**:
1. Navigate to Containers
2. Click on container name
3. Click "Logs" tab
4. Adjust log level/auto-refresh

**Via CLI**:
```bash
# Specific service
docker logs -f sutazai-backend

# All services
docker-compose -f docker-compose-portainer.yml logs -f

# Last 100 lines
docker logs sutazai-frontend --tail 100
```

### Scale Services

**Not recommended** for this stack (all services are singletons), but if needed:

```bash
# Via Portainer UI:
# Containers → Select container → "Duplicate/Edit" → Change replica count

# Via CLI (requires swarm mode):
docker service scale sutazai-platform_backend=3
```

---

## Backup & Restore

### Backup Volumes

```bash
#!/bin/bash
# Create backup directory
mkdir -p /opt/sutazaiapp/backups/$(date +%Y%m%d)

# Backup PostgreSQL
docker exec sutazai-postgres pg_dump -U jarvis jarvis_ai > \
  /opt/sutazaiapp/backups/$(date +%Y%m%d)/postgres_backup.sql

# Backup Neo4j
docker exec sutazai-neo4j neo4j-admin dump --database=neo4j \
  --to=/var/lib/neo4j/data/neo4j-backup.dump
docker cp sutazai-neo4j:/var/lib/neo4j/data/neo4j-backup.dump \
  /opt/sutazaiapp/backups/$(date +%Y%m%d)/

# Backup volumes
docker run --rm -v postgres_data:/data -v /opt/sutazaiapp/backups/$(date +%Y%m%d):/backup \
  alpine tar czf /backup/postgres_data.tar.gz /data
```

### Restore from Backup

```bash
# Stop stack first
docker-compose -f docker-compose-portainer.yml down

# Restore PostgreSQL
docker run --rm -v postgres_data:/data -v /opt/sutazaiapp/backups/YYYYMMDD:/backup \
  alpine tar xzf /backup/postgres_data.tar.gz -C /

# Restart stack
docker-compose -f docker-compose-portainer.yml up -d
```

---

## Monitoring

### Portainer Built-in Stats

1. Navigate to "Containers"
2. View real-time CPU/Memory/Network graphs per container

### Prometheus/Grafana (Future)

When Phase 9 monitoring stack deployed:
- Prometheus: http://localhost:10300
- Grafana: http://localhost:10301 (default: admin/admin)

### Manual Health Checks

```bash
# Create monitoring script
cat > /opt/sutazaiapp/health_check.sh <<'EOF'
#!/bin/bash
echo "=== SutazAI Health Check ===" 
echo ""
echo "Container Status:"
docker ps --filter "name=sutazai-" --format "{{.Names}}: {{.Status}}"
echo ""
echo "Backend Health:"
curl -s http://localhost:10200/health | jq
echo ""
echo "Frontend Health:"
curl -s http://localhost:11000/_stcore/health
EOF

chmod +x /opt/sutazaiapp/health_check.sh

# Run it
./health_check.sh
```

---

## Security Hardening

### Change Default Credentials

**PostgreSQL**:
```bash
docker exec -it sutazai-postgres psql -U jarvis -d jarvis_ai -c \
  "ALTER USER jarvis WITH PASSWORD 'new_secure_password';"

# Update in docker-compose-portainer.yml:
# POSTGRES_PASSWORD: new_secure_password
```

**RabbitMQ**:
```bash
docker exec -it sutazai-rabbitmq rabbitmqctl change_password jarvis new_secure_password

# Update in docker-compose-portainer.yml:
# RABBITMQ_DEFAULT_PASS: new_secure_password
```

**Neo4j**:
```bash
docker exec -it sutazai-neo4j cypher-shell -u neo4j -p sutazai2024 \
  "ALTER CURRENT USER SET PASSWORD FROM 'sutazai2024' TO 'new_secure_password';"
```

### JWT Secret Rotation

```bash
# Generate new secret
NEW_SECRET=$(openssl rand -hex 32)

# Update in docker-compose-portainer.yml:
# SECRET_KEY: $NEW_SECRET

# Restart backend
docker restart sutazai-backend
```

---

## Support & Resources

### Documentation
- Production Validation Report: `/opt/sutazaiapp/PRODUCTION_VALIDATION_REPORT.md`
- Port Registry: `/opt/sutazaiapp/IMPORTANT/ports/PortRegistry.md`
- TODO Status: `/opt/sutazaiapp/TODO.md`
- DeepWiki: https://deepwiki.com/sutazai/sutazaiapp

### Test Suites
- Integration Tests: `bash /opt/sutazaiapp/tests/integration/test_integration.sh`
- E2E Tests: `cd /opt/sutazaiapp/frontend && npx playwright test`

### Contact
- Project Repository: (add your GitHub/GitLab URL)
- Issue Tracker: (add your issue tracker URL)

---

**Deployment Guide Version**: 1.0  
**Last Updated**: 2025-11-13 20:45:00 UTC  
**Compatible With**: SutazAI Platform v1.0 (Phase 8 Complete)
