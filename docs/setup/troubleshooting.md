# Troubleshooting Guide

**Last Updated**: 2025-01-03  
**Version**: 1.0.0  
**Maintainer**: DevOps Team

## Table of Contents

1. [Common Setup Issues](#common-setup-issues)
2. [Docker Problems](#docker-problems)
3. [Database Connection Issues](#database-connection-issues)
4. [Service Startup Failures](#service-startup-failures)
5. [MCP Server Issues](#mcp-server-issues)
6. [Network Problems](#network-problems)
7. [Performance Issues](#performance-issues)
8. [Authentication Errors](#authentication-errors)
9. [Debugging Tools](#debugging-tools)
10. [Emergency Recovery](#emergency-recovery)

## Common Setup Issues

### Issue: Permission Denied Errors

**Symptoms**: 
```bash
docker: permission denied while trying to connect to the Docker daemon socket
```

**Solution**:
```bash
# Add user to docker group
sudo usermod -aG docker $USER
newgrp docker

# Fix socket permissions
sudo chmod 666 /var/run/docker.sock

# Restart Docker service
sudo systemctl restart docker
```

### Issue: Port Already in Use

**Symptoms**:
```bash
Error: bind: address already in use :10000
```

**Solution**:
```bash
# Find process using port
sudo lsof -i :10000
# or
sudo netstat -tulpn | grep 10000

# Kill the process
sudo kill -9 <PID>

# Alternative: Change port in .env
sed -i 's/POSTGRES_PORT=10000/POSTGRES_PORT=10010/' .env
```

### Issue: Insufficient Resources

**Symptoms**:
```bash
Cannot allocate memory
No space left on device
```

**Solution**:
```bash
# Check available resources
df -h
free -h
docker system df

# Clean up Docker resources
docker system prune -a --volumes
docker builder prune -a

# Remove unused images
docker images -q -f dangling=true | xargs docker rmi

# Clear logs
truncate -s 0 /var/lib/docker/containers/*/*-json.log
```

## Docker Problems

### Issue: Docker Daemon Not Running

**Symptoms**:
```bash
Cannot connect to the Docker daemon at unix:///var/run/docker.sock
```

**Solution**:
```bash
# Check Docker status
sudo systemctl status docker

# Start Docker
sudo systemctl start docker
sudo systemctl enable docker

# Restart Docker
sudo systemctl restart docker

# Check logs
sudo journalctl -u docker.service -n 100
```

### Issue: Container Keeps Restarting

**Symptoms**: Container in restart loop

**Solution**:
```bash
# Check container logs
docker logs sutazai-backend --tail 100

# Inspect container
docker inspect sutazai-backend

# Check health status
docker inspect sutazai-backend --format='{{json .State.Health}}'

# Force recreate
docker compose -f docker-compose-backend.yml up -d --force-recreate

# Remove and rebuild
docker compose -f docker-compose-backend.yml down
docker compose -f docker-compose-backend.yml build --no-cache
docker compose -f docker-compose-backend.yml up -d
```

### Issue: Network Conflicts

**Symptoms**:
```bash
Error: network sutazai-network already exists
Subnet conflicts with existing network
```

**Solution**:
```bash
# List networks
docker network ls

# Remove conflicting network
docker network rm sutazai-network

# Prune unused networks
docker network prune

# Create with different subnet
docker network create --subnet=172.21.0.0/16 sutazai-network-new

# Update docker-compose files
find . -name "docker-compose*.yml" -exec sed -i 's/sutazai-network/sutazai-network-new/g' {} \;
```

## Database Connection Issues

### PostgreSQL Connection Failed

**Symptoms**:
```bash
psycopg2.OperationalError: could not connect to server
FATAL: password authentication failed for user "jarvis"
```

**Solution**:
```bash
# Check PostgreSQL status
docker exec sutazai-postgres pg_isready

# Verify credentials
docker exec sutazai-postgres psql -U jarvis -c "SELECT 1"

# Reset password
docker exec sutazai-postgres psql -U postgres -c "ALTER USER jarvis PASSWORD 'new_password';"

# Check connection from backend
docker exec sutazai-backend python -c "
from app.core.database import engine
from sqlalchemy import text
with engine.connect() as conn:
    result = conn.execute(text('SELECT 1'))
    print('Connected:', result.scalar())
"

# Fix permission issues
docker exec sutazai-postgres chmod 600 /var/lib/postgresql/data/pg_hba.conf
docker exec sutazai-postgres sed -i 's/md5/trust/g' /var/lib/postgresql/data/pg_hba.conf
docker restart sutazai-postgres
```

### Redis Connection Refused

**Symptoms**:
```bash
redis.exceptions.ConnectionError: Error 111 connecting to localhost:10001
```

**Solution**:
```bash
# Test Redis connection
docker exec sutazai-backend redis-cli -h sutazai-redis -p 6379 ping

# Check Redis logs
docker logs sutazai-redis --tail 50

# Fix config issues
docker exec sutazai-redis redis-cli CONFIG GET bind
docker exec sutazai-redis redis-cli CONFIG SET bind "0.0.0.0"

# Flush problematic data
docker exec sutazai-redis redis-cli FLUSHALL
```

### Neo4j Authentication Failed

**Symptoms**:
```bash
Neo4j.exceptions.AuthError: The client is unauthorized due to authentication failure
```

**Solution**:
```bash
# Reset Neo4j password
docker exec -it sutazai-neo4j cypher-shell -u neo4j -p neo4j \
  "ALTER CURRENT USER SET PASSWORD FROM 'neo4j' TO 'sutazai_secure_2024';"

# Verify connection
docker exec sutazai-backend python -c "
from neo4j import GraphDatabase
driver = GraphDatabase.driver('bolt://sutazai-neo4j:7687', 
                              auth=('neo4j', 'sutazai_secure_2024'))
driver.verify_connectivity()
print('Neo4j connected successfully')
"
```

## Service Startup Failures

### Backend API Won't Start

**Symptoms**: FastAPI service fails to start

**Diagnostic Steps**:
```bash
# Check detailed logs
docker logs sutazai-backend --tail 200

# Run startup validation
docker exec sutazai-backend python -c "
from app.core.config import settings
print('Settings loaded:', settings.dict())
"

# Test database migrations
docker exec sutazai-backend alembic current
docker exec sutazai-backend alembic check

# Manual startup for debugging
docker exec -it sutazai-backend bash
python -m uvicorn app.main:app --host 0.0.0.0 --port 10200 --reload
```

### Frontend Streamlit Errors

**Symptoms**: Streamlit app crashes or shows errors

**Solution**:
```bash
# Check Streamlit logs
docker logs sutazai-frontend --tail 100

# Validate config
docker exec sutazai-frontend python -c "
from config.settings import settings
print('API URL:', settings.API_URL)
print('WebSocket URL:', settings.WS_URL)
"

# Clear cache
docker exec sutazai-frontend rm -rf /root/.streamlit/cache

# Restart with debug mode
docker exec -it sutazai-frontend bash
streamlit run app.py --server.enableCORS false --server.enableXsrfProtection false
```

## MCP Server Issues

### MCP Server Not Responding

**Symptoms**: MCP server timeouts or connection refused

**Solution**:
```bash
# Test individual MCP server
/opt/sutazaiapp/scripts/mcp/wrappers/filesystem.sh --selfcheck

# Debug with verbose output
export DEBUG=mcp:*
/opt/sutazaiapp/scripts/mcp/wrappers/memory.sh 2>&1 | tee debug.log

# Check Node.js installation
which node
node --version

# Rebuild MCP server
cd /opt/sutazaiapp/mcp-servers/filesystem
pnpm install
pnpm build
```

### MCP Bridge Communication Errors

**Symptoms**: Agent orchestration failures

**Solution**:
```bash
# Check MCP Bridge health
curl http://localhost:11100/health

# Restart MCP Bridge
docker restart sutazai-mcp-bridge

# Verify agent registry
curl http://localhost:11100/agents

# Test agent connectivity
for port in {11401..11801}; do
    echo "Testing port $port:"
    nc -zv localhost $port 2>&1 | grep -q succeeded && echo "✓ Connected" || echo "✗ Failed"
done
```

## Network Problems

### Container Communication Issues

**Symptoms**: Containers can't reach each other

**Solution**:
```bash
# Verify network configuration
docker network inspect sutazai-network

# Test container-to-container connectivity
docker exec sutazai-backend ping -c 3 sutazai-postgres
docker exec sutazai-backend nc -zv sutazai-redis 6379

# Fix DNS resolution
docker exec sutazai-backend cat /etc/hosts
docker exec sutazai-backend nslookup sutazai-postgres

# Recreate network
docker compose down
docker network rm sutazai-network
docker network create sutazai-network --subnet=172.20.0.0/16
docker compose up -d
```

### External API Connection Failed

**Symptoms**: Cannot reach external services

**Solution**:
```bash
# Check DNS resolution
docker exec sutazai-backend nslookup api.openai.com
docker exec sutazai-backend dig google.com

# Test external connectivity
docker exec sutazai-backend curl -I https://api.openai.com

# Fix proxy settings if needed
docker exec sutazai-backend bash -c 'echo "export HTTP_PROXY=http://proxy:8080" >> ~/.bashrc'
docker exec sutazai-backend bash -c 'echo "export HTTPS_PROXY=http://proxy:8080" >> ~/.bashrc'
```

## Performance Issues

### High Memory Usage

**Symptoms**: System running slow, OOM errors

**Solution**:
```bash
# Check memory usage
docker stats --no-stream

# Limit container memory
docker update --memory="2g" --memory-swap="2g" sutazai-backend

# Optimize Neo4j memory
docker exec sutazai-neo4j bash -c "
echo 'dbms.memory.heap.initial_size=2g' >> /var/lib/neo4j/conf/neo4j.conf
echo 'dbms.memory.heap.max_size=4g' >> /var/lib/neo4j/conf/neo4j.conf
echo 'dbms.memory.pagecache.size=2g' >> /var/lib/neo4j/conf/neo4j.conf
"
docker restart sutazai-neo4j

# Clear caches
docker exec sutazai-redis redis-cli FLUSHALL
docker exec sutazai-backend rm -rf /tmp/*
```

### Slow Database Queries

**Symptoms**: API responses taking too long

**Solution**:
```bash
# Analyze slow queries
docker exec sutazai-postgres psql -U jarvis -c "
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;"

# Add missing indexes
docker exec sutazai-postgres psql -U jarvis -d jarvis_ai -c "
CREATE INDEX CONCURRENTLY idx_users_email ON users(email);
CREATE INDEX CONCURRENTLY idx_sessions_user_id ON sessions(user_id);
CREATE INDEX CONCURRENTLY idx_messages_created_at ON messages(created_at DESC);
"

# Vacuum and analyze
docker exec sutazai-postgres vacuumdb -U jarvis -d jarvis_ai -z
```

## Authentication Errors

### JWT Token Invalid

**Symptoms**: 401 Unauthorized errors

**Solution**:
```bash
# Verify JWT secret
docker exec sutazai-backend python -c "
from app.core.config import settings
print('JWT Secret exists:', bool(settings.JWT_SECRET_KEY))
print('JWT Algorithm:', settings.JWT_ALGORITHM)
"

# Generate new JWT secret
export NEW_JWT_SECRET=$(openssl rand -base64 64)
docker exec sutazai-backend bash -c "echo 'JWT_SECRET_KEY=$NEW_JWT_SECRET' >> .env"
docker restart sutazai-backend

# Clear invalid sessions
docker exec sutazai-redis redis-cli --scan --pattern "session:*" | xargs docker exec sutazai-redis redis-cli DEL
```

## Debugging Tools

### Log Analysis Script

```bash
#!/bin/bash
# scripts/analyze_logs.sh

echo "=== ERROR SUMMARY ==="
docker compose logs --tail 1000 | grep -i error | tail -20

echo -e "\n=== WARNING SUMMARY ==="
docker compose logs --tail 1000 | grep -i warning | tail -20

echo -e "\n=== FAILED CONNECTIONS ==="
docker compose logs --tail 1000 | grep -i "connection refused\|connection failed" | tail -10

echo -e "\n=== SERVICE HEALTH ==="
for service in backend frontend postgres redis neo4j rabbitmq; do
    status=$(docker inspect sutazai-$service --format='{{.State.Status}}' 2>/dev/null)
    echo "sutazai-$service: ${status:-not found}"
done
```

### Health Check Script

```bash
#!/bin/bash
# scripts/health_check.sh

check_service() {
    local name=$1
    local url=$2
    if curl -f -s "$url" > /dev/null; then
        echo "✓ $name: Healthy"
    else
        echo "✗ $name: Unhealthy"
    fi
}

check_service "Backend API" "http://localhost:10200/health"
check_service "Frontend" "http://localhost:11000/_stcore/health"
check_service "MCP Bridge" "http://localhost:11100/health"
check_service "ChromaDB" "http://localhost:10100/api/v1/heartbeat"
check_service "Qdrant" "http://localhost:10101/health"
```

## Emergency Recovery

### Full System Reset

```bash
#!/bin/bash
# scripts/emergency_reset.sh

echo "⚠️  WARNING: This will delete all data!"
read -p "Are you sure? (yes/no): " confirm
[[ "$confirm" != "yes" ]] && exit 1

# Stop all services
docker compose down -v

# Remove all containers and volumes
docker rm -f $(docker ps -aq) 2>/dev/null
docker volume rm $(docker volume ls -q) 2>/dev/null

# Clean Docker system
docker system prune -a --volumes -f

# Remove data directories
sudo rm -rf /opt/sutazaiapp/data/*
sudo rm -rf /opt/sutazaiapp/logs/*

# Recreate directories
mkdir -p /opt/sutazaiapp/{data,logs}/{postgres,redis,neo4j,rabbitmq}

# Rebuild everything
./deploy.sh --clean
```

### Database Recovery

```bash
#!/bin/bash
# scripts/recover_database.sh

# Backup current state
docker exec sutazai-postgres pg_dump -U jarvis jarvis_ai > backup_$(date +%Y%m%d_%H%M%S).sql

# Check database integrity
docker exec sutazai-postgres psql -U jarvis -d jarvis_ai -c "
SELECT schemaname, tablename, 
       pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables 
WHERE schemaname NOT IN ('pg_catalog', 'information_schema')
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;"

# Repair corrupted tables
docker exec sutazai-postgres psql -U jarvis -d jarvis_ai -c "REINDEX DATABASE jarvis_ai;"
docker exec sutazai-postgres psql -U jarvis -d jarvis_ai -c "VACUUM FULL ANALYZE;"

# Reset sequences
docker exec sutazai-postgres psql -U jarvis -d jarvis_ai -c "
SELECT setval(pg_get_serial_sequence('users', 'id'), MAX(id)) FROM users;
SELECT setval(pg_get_serial_sequence('sessions', 'id'), MAX(id)) FROM sessions;
"
```

## Related Documentation

- [Dependencies Guide](./dependencies.md)
- [Tools Setup](./tools.md)
- [System Architecture](../architecture/system_design.md)
- [Performance Tuning](../development/performance_tuning.md)

## Support Contacts

- **DevOps Team**: devops@sutazai.com
- **Emergency Hotline**: +1-555-SUTAZAI
- **Slack Channel**: #sutazai-support
- **Issue Tracker**: https://github.com/sutazai/sutazaiapp/issues