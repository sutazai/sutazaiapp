# 🚨 EMERGENCY REMEDIATION PLAN - SYSTEM RECOVERY
## Immediate Actions Required for Critical System Failures

**Created**: 2025-08-17 22:45:00 UTC  
**Priority**: P0 CRITICAL  
**Estimated Recovery Time**: 2-4 hours  
**Dependencies**: Docker, Python, System Administration Access  

---

## 🎯 CRITICAL PRIORITY ACTIONS (Complete within 1 hour)

### 1. **BACKEND API EMERGENCY RESTART** (15 minutes)
```bash
# Step 1: Diagnose backend container
cd /opt/sutazaiapp
docker logs sutazai-backend --tail 100 > backend_failure_log.txt

# Step 2: Check application startup issues
docker exec sutazai-backend ps aux | grep python
docker exec sutazai-backend netstat -tulpn | grep 8000

# Step 3: Force restart with debug mode
docker restart sutazai-backend

# Step 4: Monitor startup
docker logs sutazai-backend --follow &

# Step 5: Test recovery
sleep 30
curl -v http://localhost:10010/health
```

### 2. **CONTAINER CHAOS CLEANUP** (20 minutes)
```bash
# Step 1: List all orphaned containers
docker ps -a --filter "name=amazing_" --filter "name=fervent_" --filter "name=infallible_" --filter "name=suspicious_" --format "table {{.Names}}\t{{.Status}}\t{{.Image}}"

# Step 2: Stop orphaned containers
docker stop $(docker ps -q --filter "name=amazing_" --filter "name=fervent_" --filter "name=infallible_" --filter "name=suspicious_")

# Step 3: Remove orphaned containers
docker rm $(docker ps -aq --filter "name=amazing_" --filter "name=fervent_" --filter "name=infallible_" --filter "name=suspicious_")

# Step 4: Clean up unused networks
docker network prune -f

# Step 5: Clean up unused volumes
docker volume prune -f
```

### 3. **SERVICE DISCOVERY REPAIR** (10 minutes)
```bash
# Step 1: Restart Consul
docker restart sutazai-consul

# Step 2: Wait for stabilization
sleep 15

# Step 3: Verify Consul health
curl http://localhost:10006/v1/status/leader

# Step 4: Check agent list
curl http://localhost:10006/v1/agent/members

# Step 5: Restart MCP manager
docker restart sutazai-mcp-manager
```

### 4. **MCP SERVICES VERIFICATION** (15 minutes)
```bash
# Step 1: Check actual MCP containers
docker ps --filter "name=mcp" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Step 2: Restart failed MCP services
docker restart mcp-unified-dev-container
docker restart mcp-unified-memory

# Step 3: Deploy missing MCP containers using DinD
cd /opt/sutazaiapp/docker/dind/mcp-containers
docker-compose -f docker-compose.mcp-services.yml up -d

# Step 4: Verify DinD orchestrator
docker exec sutazai-mcp-orchestrator docker ps

# Step 5: Test MCP API endpoints
curl http://localhost:10010/api/v1/mcp/status
```

---

## 🔧 HIGH PRIORITY FIXES (Complete within 2 hours)

### 5. **CONFIGURATION CONSOLIDATION ENFORCEMENT** (30 minutes)
```bash
# Step 1: Move to project root
cd /opt/sutazaiapp

# Step 2: Stop all services
docker-compose down

# Step 3: Use ONLY the consolidated configuration
docker-compose -f docker/docker-compose.consolidated.yml down
docker-compose -f docker/docker-compose.consolidated.yml up -d

# Step 4: Archive scattered configs
mkdir -p /tmp/scattered_configs_$(date +%Y%m%d_%H%M%S)
find . -name "*docker-compose*" -not -path "./docker/docker-compose.consolidated.yml" -exec mv {} /tmp/scattered_configs_$(date +%Y%m%d_%H%M%S)/ \;

# Step 5: Verify single configuration
find /opt/sutazaiapp -name "*docker-compose*" | wc -l  # Should be 1
```

### 6. **NETWORK TOPOLOGY CLEANUP** (20 minutes)
```bash
# Step 1: Identify active networks
docker network ls | grep sutazai

# Step 2: Connect all services to primary network
docker network connect sutazai-network sutazai-backend
docker network connect sutazai-network sutazai-mcp-manager

# Step 3: Remove redundant networks
docker network rm docker_sutazai-network dind_sutazai-dind-internal

# Step 4: Verify network consolidation
docker network ls | grep sutazai  # Should show only sutazai-network

# Step 5: Test inter-service connectivity
docker exec sutazai-backend ping -c 3 sutazai-consul
```

### 7. **DinD MCP DEPLOYMENT** (40 minutes)
```bash
# Step 1: Access DinD orchestrator
docker exec -it sutazai-mcp-orchestrator sh

# Step 2: Inside DinD - Deploy MCP services
cd /mcp-manifests
docker-compose -f files-mcp.yml up -d
docker-compose -f postgres-mcp.yml up -d

# Step 3: Verify MCP containers inside DinD
docker ps  # Should show MCP services

# Step 4: Exit DinD and verify external access
exit
docker exec sutazai-mcp-orchestrator docker ps | wc -l  # Should be > 1

# Step 5: Test MCP service communication
curl http://localhost:4001/health  # Unified dev service
curl http://localhost:3009/health  # Unified memory service
```

### 8. **HEALTH CHECK SYSTEM REPAIR** (30 minutes)
```bash
# Step 1: Update backend health check
docker exec sutazai-backend python3 -c "
import requests
try:
    response = requests.get('http://localhost:8000/health', timeout=5)
    print(f'Health check: {response.status_code}')
except Exception as e:
    print(f'Health check failed: {e}')
"

# Step 2: Fix backend application issues
docker exec sutazai-backend cat /app/logs/startup.log | tail -20

# Step 3: Restart with proper environment
docker exec sutazai-backend env | grep -E "(PATH|PYTHON|APP)"

# Step 4: Test all endpoints
curl http://localhost:10010/health
curl http://localhost:10010/docs
curl http://localhost:10010/api/v1/mcp/status

# Step 5: Verify sustained health
sleep 60
curl http://localhost:10010/health
```

---

## 🔍 VERIFICATION PROCEDURES

### Post-Remediation Validation Checklist:

```bash
# 1. Backend API Functional
curl -f http://localhost:10010/health || echo "❌ Backend still down"

# 2. MCP API Responsive
curl -f http://localhost:10010/api/v1/mcp/status || echo "❌ MCP API still broken"

# 3. Container Count Correct
CONTAINER_COUNT=$(docker ps | wc -l)
if [ $CONTAINER_COUNT -lt 20 ]; then echo "❌ Insufficient containers: $CONTAINER_COUNT"; fi

# 4. No Orphaned Containers
ORPHANED=$(docker ps --filter "name=amazing_" --filter "name=fervent_" --format "{{.Names}}" | wc -l)
if [ $ORPHANED -gt 0 ]; then echo "❌ Orphaned containers still exist: $ORPHANED"; fi

# 5. Single Docker Configuration
CONFIG_COUNT=$(find /opt/sutazaiapp -name "*docker-compose*" | wc -l)
if [ $CONFIG_COUNT -gt 1 ]; then echo "❌ Multiple configs remain: $CONFIG_COUNT"; fi

# 6. DinD Contains MCP Services
DIND_CONTAINERS=$(docker exec sutazai-mcp-orchestrator docker ps | wc -l)
if [ $DIND_CONTAINERS -lt 5 ]; then echo "❌ DinD insufficient containers: $DIND_CONTAINERS"; fi

# 7. Service Discovery Working
curl -f http://localhost:10006/v1/status/leader || echo "❌ Consul still broken"

# 8. Network Consolidation Complete
NETWORK_COUNT=$(docker network ls | grep sutazai | wc -l)
if [ $NETWORK_COUNT -gt 1 ]; then echo "❌ Multiple networks remain: $NETWORK_COUNT"; fi
```

---

## 🚨 ROLLBACK PROCEDURES (If remediation fails)

### Emergency Rollback Plan:
```bash
# 1. Stop all services
docker-compose down
docker stop $(docker ps -q)

# 2. Restore from veteran backup
cd /opt/sutazaiapp/docker/veteran_backup_20250817_233351
cp -r archived_configs_20250817_final/* /opt/sutazaiapp/docker/

# 3. Restart with known good configuration
docker-compose -f /opt/sutazaiapp/docker/docker-compose.consolidated.yml up -d

# 4. Restore previous container state
docker start sutazai-backend sutazai-consul sutazai-mcp-manager

# 5. Verify basic functionality
curl http://localhost:10010/health
```

---

## 📞 ESCALATION CONTACTS

If remediation fails:
1. **System Administrator**: Check system resources and permissions
2. **Docker Specialist**: Investigate container orchestration issues  
3. **Network Engineer**: Resolve connectivity and routing problems
4. **Application Developer**: Fix backend application startup issues

---

## 📋 SUCCESS CRITERIA

**Remediation considered successful when:**
- ✅ Backend API responds to HTTP requests (< 2 second response time)
- ✅ All 21 MCP services deployed and accessible  
- ✅ Single authoritative docker-compose configuration
- ✅ DinD orchestrator contains actual MCP containers
- ✅ Service discovery (Consul) functional
- ✅ No orphaned containers with random names
- ✅ Unified network topology
- ✅ Documentation updated to reflect actual system state

**Estimated Full Recovery**: 2-4 hours with dedicated engineering effort

---

*Emergency plan prepared by Elite Senior Debugging Specialist*  
*Status: READY FOR IMMEDIATE EXECUTION*  
*Last Updated: 2025-08-17 22:45:00 UTC*