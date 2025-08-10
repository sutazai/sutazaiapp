# Container Optimization Plan - Production Ready & K3s Compatible

**Created By:** Ultra Container Specialist  
**Date:** August 10, 2025  
**System Version:** SutazAI v76  
**Objective:** Optimize all container configurations for production deployment with K3s

## Executive Summary

This plan addresses critical container optimizations needed for production readiness, focusing on security hardening, resource optimization, proper health checks, and K3s deployment compatibility.

## Current State Analysis

### ✅ Strengths Identified
- Basic resource limits configured for most services
- Health checks present (but need optimization)
- Non-root user configurations in secure compose file
- Volume management properly configured
- Network isolation with custom bridge network

### 🔴 Critical Issues Requiring Immediate Fix

1. **Security Vulnerabilities**
   - 3 containers still running as root (Neo4j, Ollama, RabbitMQ)
   - Missing security capabilities restrictions
   - No read-only root filesystems where applicable
   - Insufficient privilege dropping

2. **Resource Management Issues**
   - Excessive resource allocations (Ollama: 16GB RAM)
   - Missing memory swap limits
   - No PID limits configured
   - Inefficient CPU allocations

3. **Health Check Problems**
   - Inconsistent health check intervals (30s to 60s)
   - Some health checks using inefficient methods
   - Missing dependency-aware health checks
   - No startup ordering based on health

4. **Networking Gaps**
   - All services exposed on host ports (security risk)
   - Missing internal-only network configurations
   - No network policies defined
   - Lack of service mesh integration points

5. **K3s Compatibility Issues**
   - No Kubernetes manifest generation
   - Missing liveness/readiness probe configurations
   - No ConfigMap/Secret management patterns
   - Lack of horizontal pod autoscaling configs

## Optimization Strategy

### Phase 1: Security Hardening (Priority 1)

#### 1.1 Convert Remaining Root Containers

**Neo4j Security Migration:**
```yaml
neo4j:
  image: neo4j:5.18-community
  container_name: sutazai-neo4j
  user: "7474:7474"
  security_opt:
    - no-new-privileges:true
    - apparmor:unconfined
    - seccomp:unconfined
  cap_drop:
    - ALL
  cap_add:
    - CHOWN
    - SETUID
    - SETGID
    - DAC_OVERRIDE
```

**Ollama Security Migration:**
```yaml
ollama:
  build:
    context: ./docker/ollama-secure
    dockerfile: Dockerfile
  user: "1002:1002"
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  cap_add:
    - SYS_RESOURCE  # For model loading
```

**RabbitMQ Security Migration:**
```yaml
rabbitmq:
  image: rabbitmq:3.12-management-alpine
  user: "999:999"
  security_opt:
    - no-new-privileges:true
  cap_drop:
    - ALL
  cap_add:
    - CHOWN
    - SETUID
    - SETGID
```

#### 1.2 Implement Read-Only Root Filesystems

For services that don't need to write to the filesystem:
```yaml
services:
  prometheus:
    read_only: true
    tmpfs:
      - /tmp
      - /prometheus  # Writable data directory
```

### Phase 2: Resource Optimization (Priority 2)

#### 2.1 Right-Size Resource Allocations

**Optimized Resource Matrix:**

| Service | Current Memory | Optimized Memory | Current CPU | Optimized CPU |
|---------|---------------|------------------|-------------|---------------|
| Ollama | 16GB | 8GB | 8 cores | 4 cores |
| PostgreSQL | 2GB | 1.5GB | 2 cores | 1.5 cores |
| Backend | 4GB | 2GB | 4 cores | 2 cores |
| Neo4j | 1GB | 768MB | 1.5 cores | 1 core |
| Qdrant | 2GB | 1GB | 2 cores | 1 core |

#### 2.2 Implement Swap and PID Limits

```yaml
deploy:
  resources:
    limits:
      memory: 1G
      pids: 200
    reservations:
      memory: 512M
  restart_policy:
    condition: on-failure
    delay: 5s
    max_attempts: 3
    window: 120s
```

### Phase 3: Health Check Optimization (Priority 3)

#### 3.1 Standardized Health Check Configuration

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:${PORT}/health"]
  interval: 30s
  timeout: 5s
  retries: 3
  start_period: 60s
```

#### 3.2 Dependency-Aware Health Checks

```yaml
backend:
  depends_on:
    postgres:
      condition: service_healthy
    redis:
      condition: service_healthy
    ollama:
      condition: service_started
```

### Phase 4: Network Security (Priority 4)

#### 4.1 Internal Network Configuration

```yaml
networks:
  sutazai-internal:
    driver: bridge
    internal: true
  sutazai-external:
    driver: bridge

services:
  postgres:
    networks:
      - sutazai-internal  # Only internal access
  
  backend:
    networks:
      - sutazai-internal
      - sutazai-external  # Needs external access
```

#### 4.2 Remove Unnecessary Port Exposures

Only expose ports for services that need external access:
- Frontend (10011)
- Backend API (10010)
- Monitoring dashboards (Grafana: 10201)

### Phase 5: K3s Migration Preparation (Priority 5)

#### 5.1 Generate Kubernetes Manifests

Create Helm charts for deployment flexibility:
```yaml
# values.yaml
global:
  storageClass: local-path
  nodeSelector:
    node-role.kubernetes.io/worker: "true"
  
postgresql:
  enabled: true
  persistence:
    size: 10Gi
  resources:
    limits:
      memory: 1.5Gi
      cpu: 1500m
    requests:
      memory: 512Mi
      cpu: 500m
```

#### 5.2 Implement Proper Probes

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8080
  initialDelaySeconds: 60
  periodSeconds: 30
  
readinessProbe:
  httpGet:
    path: /ready
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 5
```

## Implementation Roadmap

### Week 1: Security Hardening
- Day 1-2: Convert remaining root containers
- Day 3-4: Implement security capabilities
- Day 5: Test security configurations

### Week 2: Resource & Health Optimization
- Day 1-2: Right-size resource allocations
- Day 3-4: Optimize health checks
- Day 5: Performance testing

### Week 3: Network & K3s Preparation
- Day 1-2: Implement network segmentation
- Day 3-4: Create K3s manifests
- Day 5: Deploy to K3s test cluster

## Monitoring & Validation

### Key Metrics to Track
- Container restart frequency
- Memory/CPU utilization
- Health check success rate
- Network latency between services
- Security scan results

### Validation Commands
```bash
# Security validation
docker-compose run --rm security-scanner

# Resource monitoring
docker stats --no-stream

# Health check status
docker-compose ps

# Network connectivity
docker-compose exec backend curl http://postgres:5432
```

## Risk Mitigation

### Potential Issues & Solutions

1. **Service startup failures after security hardening**
   - Solution: Gradually apply restrictions, test each change
   
2. **Performance degradation with resource limits**
   - Solution: Monitor metrics, adjust limits based on actual usage
   
3. **Health check failures during high load**
   - Solution: Implement circuit breakers, adjust timeouts

## Success Criteria

- ✅ 100% containers running as non-root
- ✅ All services have proper resource limits
- ✅ Health checks succeed 99.9% of the time
- ✅ Internal services not accessible from host
- ✅ Successfully deployed to K3s cluster
- ✅ Security scan shows no critical vulnerabilities

## Next Steps

1. Review and approve this plan
2. Create backup of current configuration
3. Implement Phase 1 (Security Hardening)
4. Validate changes in staging environment
5. Proceed with subsequent phases

## Appendix: Quick Reference

### Security Checklist
- [ ] Non-root user configured
- [ ] Capabilities dropped (ALL)
- [ ] Only necessary capabilities added
- [ ] no-new-privileges enabled
- [ ] Read-only root filesystem (where applicable)
- [ ] tmpfs for temporary files

### Resource Checklist
- [ ] Memory limits set
- [ ] CPU limits set
- [ ] PID limits configured
- [ ] Swap accounting enabled
- [ ] Restart policy configured

### Health Check Checklist
- [ ] Consistent intervals (30s)
- [ ] Appropriate timeouts (5s)
- [ ] Sufficient retries (3)
- [ ] Adequate start period
- [ ] Efficient health check method

### Network Checklist
- [ ] Internal network for databases
- [ ] External network for APIs
- [ ] Minimal port exposure
- [ ] Network policies defined
- [ ] Service discovery configured