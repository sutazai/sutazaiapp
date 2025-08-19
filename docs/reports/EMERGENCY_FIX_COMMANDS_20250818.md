# EMERGENCY INFRASTRUCTURE FIX COMMANDS
**Priority:** P0 CRITICAL  
**Execution Required:** IMMEDIATELY  
**Expected Downtime:** 15-30 minutes  

## 🚨 PRE-EXECUTION CHECKLIST
- [ ] System backup completed
- [ ] Emergency access to containers confirmed  
- [ ] Network admin privileges verified
- [ ] Database data backup confirmed

## ⚡ IMMEDIATE P0 FIXES

### 1. Fix Database Services (5 minutes)
```bash
# Stop and remove failed database containers
docker stop $(docker ps -q --filter "name=postgres") 2>/dev/null || true
docker stop $(docker ps -q --filter "name=neo4j") 2>/dev/null || true
docker rm $(docker ps -aq --filter "name=postgres") 2>/dev/null || true
docker rm $(docker ps -aq --filter "name=neo4j") 2>/dev/null || true

# Start PostgreSQL with proper configuration
docker run -d --name sutazai-postgres \
  --network sutazai-network \
  --env-file /opt/sutazaiapp/docker/.env \
  -p 10000:5432 \
  -v postgres_data:/var/lib/postgresql/data \
  --restart unless-stopped \
  postgres:16.3-alpine3.20

# Start Neo4j with proper configuration  
docker run -d --name sutazai-neo4j \
  --network sutazai-network \
  -e NEO4J_AUTH=neo4j/neo4j_secure_password_2025 \
  -p 10002:7474 -p 10003:7687 \
  -v neo4j_data:/data \
  --restart unless-stopped \
  neo4j:5.15.0-community

# Wait for databases to initialize
echo "Waiting for databases to initialize..."
sleep 30
```

### 2. Create MCP Network Bridge (5 minutes)
```bash
# Create bridge network for MCP access
docker network create --driver bridge \
  --subnet=172.25.0.0/16 \
  --gateway=172.25.0.1 \
  mcp-host-bridge || echo "Network exists"

# Connect DinD orchestrator to bridge network
docker network connect mcp-host-bridge sutazai-mcp-orchestrator || echo "Already connected"

# Configure iptables rules for MCP port forwarding
for port in {3001..3019}; do
  docker_ip=$(docker exec sutazai-mcp-orchestrator docker inspect mcp-$(echo "claude-flow files context7 ruv-swarm ddg http-fetch sequentialthinking nx-mcp extended-memory mcp-ssh ultimatecoder playwright-mcp memory-bank-mcp knowledge-graph-mcp compass-mcp github http language-server claude-task-runner" | cut -d' ' -f$((port-3000))) --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' 2>/dev/null || echo "172.17.0.$((port-3000))")
  
  # Add port forwarding rule
  iptables -t nat -C DOCKER -p tcp --dport $port -j DNAT --to-destination $docker_ip:$port 2>/dev/null || \
  iptables -t nat -A DOCKER -p tcp --dport $port -j DNAT --to-destination $docker_ip:$port
done
```

### 3. Restart Backend Service (2 minutes)
```bash
# Wait for databases and restart backend
echo "Restarting backend service..."
docker restart sutazai-backend

# Wait for backend to initialize
sleep 20
```

### 4. Verify Fixes (3 minutes)
```bash
# Test database connections
echo "Testing PostgreSQL..."
docker exec sutazai-postgres pg_isready -U sutazai

echo "Testing Neo4j..."  
curl -f http://localhost:10002 || echo "Neo4j starting..."

echo "Testing Backend..."
curl -f http://localhost:10010/health || echo "Backend initializing..."

echo "Testing MCP Services..."
for port in {3001..3005}; do
  echo -n "Port $port: "
  curl -s --connect-timeout 2 http://localhost:$port/health && echo "OK" || echo "FAIL"
done
```

## 🔧 ALTERNATIVE QUICK FIX (If above fails)

### Emergency Service Restart (10 minutes)
```bash
# Nuclear option: restart all services
cd /opt/sutazaiapp
docker-compose down
docker system prune -f --volumes
docker-compose up -d

# Wait for full initialization
sleep 60

# Check critical services
docker ps --filter "health=healthy"
```

## ⚠️ VALIDATION CHECKLIST

After executing fixes, verify:
- [ ] PostgreSQL responding on port 10000
- [ ] Neo4j responding on ports 10002/10003  
- [ ] Backend healthy on port 10010
- [ ] At least 5 MCP services accessible (ports 3001-3005)
- [ ] Kong gateway can reach backend
- [ ] Consul shows services as healthy

## 📞 ESCALATION CONTACTS

If fixes fail:
1. **System Administrator**: Check container resource limits
2. **Network Administrator**: Verify firewall/routing rules  
3. **Database Administrator**: Check data integrity
4. **DevOps Lead**: Consider full system rebuild

## 🚨 ROLLBACK PROCEDURE

If system becomes unstable:
```bash
# Emergency rollback
docker-compose down
git checkout HEAD~1  # Roll back to previous version
docker-compose up -d
```

---
**Document Created:** 2025-08-18 14:30:00 UTC  
**Priority:** P0 CRITICAL  
**Estimated Fix Time:** 15-30 minutes