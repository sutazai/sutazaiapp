# MCP Network Troubleshooting Guide

## Quick Diagnostics

### Health Check Commands
```bash
# Overall system health
curl -s http://localhost:11091/health | jq '.'

# Individual service health
curl -s http://localhost:11100/health | jq '.'  # PostgreSQL
curl -s http://localhost:11101/health | jq '.'  # Files
curl -s http://localhost:11102/health | jq '.'  # HTTP

# Infrastructure health
curl -s http://localhost:11090/v1/status/leader  # Consul
curl -s http://localhost:11099/stats | head -20  # HAProxy
```

### Container Status Check
```bash
# Check all MCP containers
docker ps --filter "name=mcp" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check network membership
docker network inspect sutazai-network | jq '.[] | .Containers | keys'
docker network inspect mcp-internal | jq '.[] | .Containers | keys'
```

## Common Issues and Solutions

### 1. Service Discovery Problems

#### Issue: Services not appearing in Consul
**Symptoms:**
- Empty service list in Consul UI
- Service registration errors in logs
- 404 errors when querying services

**Diagnosis:**
```bash
# Check Consul connectivity
curl -s http://localhost:11090/v1/status/leader

# Check service registration logs
docker logs sutazai-mcp-consul | grep -i error

# Verify container network access
docker exec sutazai-mcp-postgres ping mcp-consul-agent
```

**Solutions:**
1. **Restart Consul agent:**
   ```bash
   docker restart sutazai-mcp-consul
   ```

2. **Manual service registration:**
   ```bash
   curl -X PUT http://localhost:11090/v1/agent/service/register -d '{
     "ID": "mcp-postgres",
     "Name": "mcp-postgres", 
     "Port": 11100,
     "Address": "mcp-postgres",
     "Check": {"HTTP": "http://mcp-postgres:11100/health", "Interval": "10s"}
   }'
   ```

3. **Check network connectivity:**
   ```bash
   docker exec sutazai-mcp-consul nslookup mcp-postgres
   ```

### 2. Load Balancer Issues

#### Issue: 503 Service Unavailable errors
**Symptoms:**
- HAProxy returns 503 errors
- Backend servers showing as DOWN
- Connection refused errors

**Diagnosis:**
```bash
# Check HAProxy backend status
curl -s http://localhost:11099/stats | grep -A10 "postgres-servers"

# Test backend connectivity
docker exec sutazai-mcp-haproxy nc -zv mcp-postgres 11100

# Check backend health
curl -s http://localhost:11100/health
```

**Solutions:**
1. **Restart unhealthy backends:**
   ```bash
   docker restart sutazai-mcp-postgres
   ```

2. **Check HAProxy configuration:**
   ```bash
   docker exec sutazai-mcp-haproxy cat /usr/local/etc/haproxy/haproxy.cfg | grep -A5 postgres-servers
   ```

3. **Manual backend testing:**
   ```bash
   docker exec sutazai-mcp-haproxy curl -f http://mcp-postgres:11100/health
   ```

### 3. Network Connectivity Issues

#### Issue: Services unreachable or timeouts
**Symptoms:**
- Connection timeouts
- DNS resolution failures
- Network isolation problems

**Diagnosis:**
```bash
# Test network connectivity
docker exec sutazai-mcp-postgres ping mcp-consul-agent
docker exec sutazai-mcp-postgres ping sutazai-backend

# Check DNS resolution
docker exec sutazai-mcp-postgres nslookup mcp-consul-agent
docker exec sutazai-mcp-postgres nslookup sutazai-postgres

# Verify network configuration
docker network inspect mcp-internal | jq '.[] | .Config'
```

**Solutions:**
1. **Recreate network:**
   ```bash
   docker-compose -f docker/docker-compose.mcp-network.yml down
   docker network rm mcp-internal
   docker-compose -f docker/docker-compose.mcp-network.yml up -d
   ```

2. **Check container network membership:**
   ```bash
   docker inspect sutazai-mcp-postgres | jq '.[] | .NetworkSettings.Networks'
   ```

3. **Test port accessibility:**
   ```bash
   nc -zv localhost 11100  # From host
   docker exec sutazai-mcp-haproxy nc -zv mcp-postgres 11100  # From load balancer
   ```

### 4. Service Health Check Failures

#### Issue: Health checks failing but service appears running
**Symptoms:**
- Services marked as unhealthy in monitoring
- HAProxy marking backends as DOWN
- Health endpoint returning errors

**Diagnosis:**
```bash
# Check service logs
docker logs sutazai-mcp-postgres | tail -50

# Test health endpoint directly
docker exec sutazai-mcp-postgres curl -f http://localhost:11100/health

# Check service process
docker exec sutazai-mcp-postgres ps aux
```

**Solutions:**
1. **Restart service:**
   ```bash
   docker restart sutazai-mcp-postgres
   ```

2. **Check service configuration:**
   ```bash
   docker exec sutazai-mcp-postgres env | grep MCP_
   ```

3. **Validate health endpoint:**
   ```bash
   docker exec sutazai-mcp-postgres curl -v http://localhost:11100/health
   ```

### 5. Port Conflicts

#### Issue: Ports already in use
**Symptoms:**
- Container startup failures
- Port binding errors
- Address already in use errors

**Diagnosis:**
```bash
# Check port usage
ss -tulpn | grep -E ":(111[0-9][0-9])"

# Find process using port
lsof -i :11100

# Check Docker port mappings
docker ps --format "table {{.Names}}\t{{.Ports}}" | grep 111
```

**Solutions:**
1. **Stop conflicting services:**
   ```bash
   # Find and stop process using port
   sudo kill $(lsof -t -i:11100)
   ```

2. **Use alternative ports:**
   ```bash
   # Edit docker-compose.yml to use different ports
   # Then redeploy
   docker-compose -f docker/docker-compose.mcp-network.yml up -d
   ```

3. **Check for orphaned containers:**
   ```bash
   docker ps -a | grep 111
   docker rm -f $(docker ps -aq --filter "name=mcp")
   ```

## Performance Issues

### High Response Times

**Diagnosis:**
```bash
# Check response times in monitoring
curl -s http://localhost:11091/metrics | jq '.services'

# Test direct service response time
time curl -s http://localhost:11100/health

# Check service resource usage
docker stats --no-stream sutazai-mcp-postgres
```

**Solutions:**
1. **Scale services:**
   ```bash
   docker-compose -f docker/docker-compose.mcp-network.yml up -d --scale mcp-postgres=2
   ```

2. **Optimize health check intervals:**
   ```bash
   # Edit HAProxy config to reduce check frequency
   # interval 30s (instead of 10s)
   ```

3. **Check underlying service performance:**
   ```bash
   docker exec sutazai-mcp-postgres top
   docker exec sutazai-postgres htop  # Check database performance
   ```

### High Resource Usage

**Diagnosis:**
```bash
# Check container resource usage
docker stats --no-stream $(docker ps --filter "name=mcp" -q)

# Check host resources
free -h
df -h
htop
```

**Solutions:**
1. **Set resource limits:**
   ```yaml
   # In docker-compose.yml
   services:
     mcp-postgres:
       deploy:
         resources:
           limits:
             memory: 512M
             cpus: '0.5'
   ```

2. **Optimize service configuration:**
   ```bash
   # Review service-specific configurations
   # Adjust timeout values, connection pools, etc.
   ```

## Emergency Procedures

### Complete System Reset

```bash
# Stop all MCP network services
docker-compose -f docker/docker-compose.mcp-network.yml down

# Remove all MCP containers
docker rm -f $(docker ps -aq --filter "name=mcp")

# Remove MCP network
docker network rm mcp-internal

# Clean up volumes (if needed)
docker volume rm $(docker volume ls -q --filter "name=mcp")

# Redeploy from scratch
./scripts/network/deploy-mcp-network.sh
```

### Service Recovery

```bash
# Stop specific service
docker stop sutazai-mcp-postgres

# Remove container
docker rm sutazai-mcp-postgres

# Recreate service
docker-compose -f docker/docker-compose.mcp-network.yml up -d mcp-postgres

# Verify recovery
curl -s http://localhost:11100/health
```

### Network Recovery

```bash
# Recreate networks
docker network rm mcp-internal
docker-compose -f docker/docker-compose.mcp-network.yml up -d

# Verify connectivity
docker exec sutazai-mcp-postgres ping mcp-consul-agent
```

## Monitoring and Alerting

### Key Metrics to Monitor

1. **Service Availability**
   - Health check success rate > 95%
   - Response time < 1000ms average
   - Zero 5xx errors for > 99% of requests

2. **Network Performance**
   - TCP connection success rate > 99%
   - DNS resolution time < 100ms
   - Network latency < 50ms

3. **Infrastructure Health**
   - Consul cluster healthy
   - HAProxy backend servers UP
   - Container memory usage < 80%

### Alerting Setup

**Prometheus Queries:**
```promql
# Service availability
up{job="mcp-services"} == 0

# High response times
http_request_duration_seconds{quantile="0.95"} > 1

# High error rates
rate(http_requests_total{status=~"5.."}[5m]) > 0.01
```

**Alert Rules:**
```yaml
groups:
- name: mcp-network
  rules:
  - alert: MCPServiceDown
    expr: up{job="mcp-services"} == 0
    for: 1m
    labels:
      severity: critical
    annotations:
      summary: "MCP service {{ $labels.instance }} is down"
```

## Log Analysis

### Important Log Locations

```bash
# Service logs
docker logs sutazai-mcp-postgres
docker logs sutazai-mcp-consul
docker logs sutazai-mcp-haproxy

# Deployment logs
tail -f /tmp/mcp-network-deploy.log

# System logs
journalctl -u docker -f
```

### Log Patterns to Watch

**Error Patterns:**
- `connection refused`
- `timeout`
- `service registration failed`
- `health check failed`
- `backend server down`

**Success Patterns:**
- `service registered`
- `health check passed`
- `backend server up`
- `connection established`

### Log Analysis Commands

```bash
# Count errors in last hour
docker logs sutazai-mcp-postgres --since=1h | grep -i error | wc -l

# Find connection issues
docker logs sutazai-mcp-haproxy | grep -i "connection refused"

# Monitor real-time logs
docker logs -f sutazai-mcp-consul | grep -i service
```

## Contact and Escalation

### Escalation Procedures

1. **Level 1**: Try basic diagnostics and restart procedures
2. **Level 2**: Analyze logs and performance metrics
3. **Level 3**: Complete system reset and redeploy
4. **Level 4**: Contact system administrators

### Support Information

- **Documentation**: `/opt/sutazaiapp/docs/network/`
- **Logs**: `/tmp/mcp-network-deploy.log`
- **Configuration**: `/opt/sutazaiapp/docker/config/`
- **Scripts**: `/opt/sutazaiapp/scripts/network/`