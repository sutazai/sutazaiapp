# 🚨 EMERGENCY FIX PLAN - RESTORE SYSTEM INTEGRITY

## CRITICAL ISSUE: Backend Service Not Running

### IMMEDIATE ACTION REQUIRED
The backend service (FastAPI on port 10010) is completely down, breaking:
- All API functionality
- MCP integration 
- Service mesh connectivity
- Kong gateway routing

### Step 1: Start Backend Service (PRIORITY 0)

#### Option A: Using docker-compose.yml
```bash
cd /opt/sutazaiapp/docker
docker-compose -f docker-compose.yml up -d sutazai-backend
```

#### Option B: Direct Docker Run
```bash
docker run -d \
  --name sutazai-backend \
  --network sutazai-network \
  -p 10010:8000 \
  -e DATABASE_URL=postgresql://sutazai:sutazai@sutazai-postgres:5432/sutazaidb \
  -e REDIS_URL=redis://sutazai-redis:6379/0 \
  -e OLLAMA_BASE_URL=http://sutazai-ollama:11434 \
  -v /opt/sutazaiapp/backend:/app \
  docker_backend:latest
```

### Step 2: Verify Backend Health
```bash
# Check if running
docker ps | grep backend

# Test health endpoint
curl http://localhost:10010/health

# Check logs if issues
docker logs sutazai-backend
```

### Step 3: Fix Docker Consolidation

#### Current Chaos (19 Files):
```
/docker/
├── docker-compose.yml (main)
├── docker-compose.base.yml
├── docker-compose.optimized.yml
├── docker-compose.secure.yml
├── docker-compose.ultra-performance.yml
├── docker-compose.memory-optimized.yml
├── docker-compose.minimal.yml
├── docker-compose.mcp.yml
├── docker-compose.mcp-fix.yml
├── docker-compose.mcp-monitoring.yml
├── docker-compose.override.yml
└── [9 more files...]
```

#### Consolidation Plan:
1. **Identify Active Configuration**
   - Check which file(s) are actually being used
   - Document current startup command

2. **Create True Consolidated File**
   ```bash
   # Merge all necessary services into one file
   cp docker-compose.yml docker-compose.consolidated.yml
   
   # Add backend service if missing
   # Add MCP orchestrator config
   # Add monitoring stack
   ```

3. **Archive Legacy Files**
   ```bash
   mkdir -p /opt/sutazaiapp/docker/archive/2025-08-18
   mv docker-compose.*-legacy.yml archive/2025-08-18/
   mv docker-compose.override-legacy.yml archive/2025-08-18/
   ```

### Step 4: Fix Service Mesh Integration

1. **Update Kong Routes for Backend**
   ```bash
   # Check current backend service in Kong
   curl http://localhost:10015/services/backend
   
   # Update if needed
   curl -X PATCH http://localhost:10015/services/backend \
     -H "Content-Type: application/json" \
     -d '{"url": "http://sutazai-backend:8000"}'
   ```

2. **Register Backend in Consul**
   ```bash
   curl -X PUT http://localhost:10006/v1/agent/service/register \
     -H "Content-Type: application/json" \
     -d '{
       "ID": "backend",
       "Name": "backend",
       "Port": 8000,
       "Address": "sutazai-backend",
       "Check": {
         "HTTP": "http://sutazai-backend:8000/health",
         "Interval": "10s"
       }
     }'
   ```

### Step 5: Fix Network Fragmentation

#### Current Networks (5 - Should be 1):
- sutazai-network (main)
- docker_sutazai-network (duplicate)
- docker_mcp-internal (should not exist)
- dind_sutazai-dind-internal (DinD internal)
- mcp-bridge (unnecessary)

#### Consolidation:
```bash
# Move all containers to main network
docker network connect sutazai-network [container-name]

# Remove duplicate networks (after verification)
docker network rm docker_mcp-internal
docker network rm mcp-bridge
```

### Step 6: Update Documentation with Truth

#### Files to Update:
1. `/opt/sutazaiapp/CLAUDE.md`
   - Remove false claims about consolidation
   - Update actual port mappings
   - Document real startup procedures

2. `/opt/sutazaiapp/IMPORTANT/diagrams/PortRegistry.md`
   - Add missing services
   - Update backend status

3. `/opt/sutazaiapp/README.md`
   - Update quick start commands
   - Document actual architecture

### Step 7: Implement Verification

Create verification script:
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/verify_claims.sh

echo "=== System Reality Check ==="

# Check backend
echo -n "Backend API (10010): "
curl -s http://localhost:10010/health >/dev/null 2>&1 && echo "✅ UP" || echo "❌ DOWN"

# Check MCP
echo -n "MCP Containers: "
docker exec sutazai-mcp-orchestrator docker ps -q | wc -l

# Check Docker files
echo -n "Docker Compose Files: "
find /opt/sutazaiapp/docker -name "docker-compose*.yml" | wc -l

# Check networks
echo -n "Docker Networks: "
docker network ls | grep -c sutazai

echo "=== End Reality Check ==="
```

### Step 8: Emergency Monitoring

Set up immediate monitoring:
```bash
# Watch backend logs
docker logs -f sutazai-backend

# Monitor system health
watch -n 5 'curl -s http://localhost:10010/health'

# Check service mesh
watch -n 10 'curl -s http://localhost:10005/api/v1/health'
```

## EXPECTED OUTCOMES

After completing these steps:
1. ✅ Backend running on port 10010
2. ✅ MCP API endpoints functional
3. ✅ Service mesh routing working
4. ✅ Single consolidated Docker configuration
5. ✅ Documentation reflects reality
6. ✅ Automated verification in place

## TIMELINE

- **Hour 1:** Start backend, verify health
- **Hour 2:** Consolidate Docker files
- **Hour 3:** Fix service mesh
- **Hour 4:** Update documentation
- **Day 1:** Complete consolidation
- **Day 2:** Implement CI/CD verification

## SUCCESS CRITERIA

System is considered fixed when:
- [ ] Backend responds on port 10010
- [ ] MCP API endpoints return data
- [ ] Kong routes to backend successfully
- [ ] Single docker-compose.consolidated.yml exists
- [ ] All legacy files archived
- [ ] Documentation updated with truth
- [ ] Verification script shows all green

---
**Priority:** P0 - EMERGENCY
**Owner:** DevOps Team
**Deadline:** 2025-08-18 18:00 UTC