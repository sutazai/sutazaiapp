# SutazAI Troubleshooting Guide

## Overview

This comprehensive troubleshooting guide provides systematic approaches to diagnose and resolve common issues in SutazAI. Follow the structured diagnostic procedures to quickly identify and fix problems.

## Table of Contents

1. [General Troubleshooting Methodology](#general-troubleshooting-methodology)
2. [System Startup Issues](#system-startup-issues)
3. [Agent Communication Problems](#agent-communication-problems)
4. [Performance Issues](#performance-issues)
5. [Security and Authentication Issues](#security-and-authentication-issues)
6. [Resource and Scaling Problems](#resource-and-scaling-problems)
7. [Data and Storage Issues](#data-and-storage-issues)
8. [Network and Connectivity Problems](#network-and-connectivity-problems)
9. [Advanced Debugging Techniques](#advanced-debugging-techniques)
10. [Escalation Procedures](#escalation-procedures)

---

## General Troubleshooting Methodology

### The SUDAZAI Diagnostic Framework

**S**ystem Health Check  
**U**nderstand the Problem  
**D**iagnose Root Cause  
**A**nalyze Impact  
**Z**ero-in on Solution  
**A**pply Fix  
**I**mprove and Document

### Step 1: System Health Check (2 minutes)

```bash
# Quick system overview
curl http://localhost:8000/health

# Container status
docker-compose ps

# Resource usage
docker stats --no-stream

# Recent logs
docker-compose logs --tail=20
```

**Expected Results:**
- Health endpoint returns 200 OK
- All containers show "Up" status
- Resource usage under 80%
- No critical errors in logs

### Step 2: Collect Diagnostic Information

```bash
# Comprehensive health report
./scripts/comprehensive-agent-health-monitor.py > health_report.txt

# System validation
./scripts/validate-complete-system.py > system_validation.txt

# Performance metrics
curl http://localhost:8000/metrics > metrics.json
```

### Step 3: Identify Problem Category

Use this decision tree:

```
Problem Category Decision Tree:
├── System won't start → [System Startup Issues](#system-startup-issues)
├── Agents not responding → [Agent Communication Problems](#agent-communication-problems)
├── Slow performance → [Performance Issues](#performance-issues)
├── Authentication failures → [Security and Authentication Issues](#security-and-authentication-issues)
├── Resource exhaustion → [Resource and Scaling Problems](#resource-and-scaling-problems)
├── Data corruption/loss → [Data and Storage Issues](#data-and-storage-issues)
└── Network timeouts → [Network and Connectivity Problems](#network-and-connectivity-problems)
```

---

## System Startup Issues

### Problem: System Won't Start

#### Symptoms
- `docker-compose up` fails
- Services exit immediately
- Port binding errors
- Container startup loops

#### Diagnostic Steps

1. **Check Docker Status**
```bash
# Verify Docker is running
systemctl status docker

# Check Docker version compatibility
docker version
docker-compose version
```

2. **Port Conflict Detection**
```bash
# Check port usage
netstat -tulpn | grep -E ":(8000|8001|8002|8003|11434)"

# Find conflicting processes
lsof -i :8000
```

3. **Configuration Validation**
```bash
# Validate Docker Compose file
docker-compose config

# Check environment variables
./scripts/validate-configuration.sh
```

#### Common Solutions

**Port Conflicts:**
```bash
# Kill conflicting processes
sudo kill -9 $(lsof -t -i:8000)

# Or change port configuration
export SUTAZAI_API_PORT=8080
```

**Permission Issues:**
```bash
# Fix Docker permissions
sudo usermod -aG docker $USER
newgrp docker

# Fix file permissions
sudo chown -R $USER:$USER /opt/sutazaiapp
```

**Resource Constraints:**
```bash
# Check available resources
free -h
df -h

# Reduce resource allocation
export SUTAZAI_SCALE=small
```

### Problem: Services Start But Exit Immediately

#### Diagnostic Steps

1. **Check Container Logs**
```bash
# Individual service logs
docker-compose logs [service-name]

# All service logs with timestamps
docker-compose logs -t
```

2. **Check Health Checks**
```bash
# Container health status
docker inspect [container-name] | grep -A 20 "Health"

# Manual health check
curl -v http://localhost:8000/health
```

#### Common Solutions

**Missing Dependencies:**
```bash
# Rebuild with dependencies
docker-compose build --no-cache
docker-compose up -d
```

**Configuration Errors:**
```bash
# Reset to default configuration
./scripts/reset-configuration.sh

# Validate configuration syntax
./scripts/validate-configuration.sh
```

**Database Connection Issues:**
```bash
# Check database container
docker-compose logs postgres

# Reset database
docker-compose down -v
docker-compose up -d postgres
# Wait 30 seconds
docker-compose up -d
```

---

## Agent Communication Problems

### Problem: Agents Not Responding

#### Symptoms
- HTTP 503 Service Unavailable
- Agent timeout errors
- "Agent not found" messages
- Empty agent registry

#### Diagnostic Steps

1. **Agent Registry Status**
```bash
# Check agent registration
curl http://localhost:8000/agents/list

# Verify specific agent
curl http://localhost:8000/agents/senior-ai-engineer/health
```

2. **Network Connectivity**
```bash
# Check internal networking
docker network ls
docker network inspect sutazaiapp_default

# Test container-to-container communication
docker exec -it sutazaiapp_api-gateway_1 ping senior-ai-engineer
```

3. **Agent Health Verification**
```bash
# Individual agent health
./scripts/test-agent-health.py --agent senior-ai-engineer

# All agents health
./scripts/comprehensive-agent-health-monitor.py
```

#### Common Solutions

**Agent Registration Issues:**
```bash
# Force agent re-registration
./scripts/force-agent-registration.sh

# Restart agent registry
docker-compose restart agent-registry
```

**Network Configuration Problems:**
```bash
# Recreate network
docker-compose down
docker network prune -f
docker-compose up -d
```

**Service Discovery Issues:**
```bash
# Update service discovery
./scripts/update-service-discovery.sh

# Check DNS resolution
docker exec -it sutazaiapp_api-gateway_1 nslookup senior-ai-engineer
```

### Problem: Intermittent Agent Failures

#### Diagnostic Steps

1. **Pattern Analysis**
```bash
# Analyze failure patterns
./scripts/analyze-agent-failures.py --timeframe 24h

# Check load distribution
curl http://localhost:8000/agents/load-distribution
```

2. **Resource Monitoring**
```bash
# Real-time resource monitoring
./scripts/monitor-agent-resources.py --real-time

# Memory leak detection
./scripts/detect-memory-leaks.py
```

#### Common Solutions

**Load Balancing Issues:**
```bash
# Rebalance agent load
./scripts/rebalance-agent-load.py

# Increase agent instances
docker-compose up --scale senior-ai-engineer=3
```

**Memory Leaks:**
```bash
# Restart problematic agents
./scripts/restart-leaky-agents.py

# Implement memory limits
./scripts/apply-memory-limits.sh
```

---

## Performance Issues

### Problem: Slow Response Times

#### Symptoms
- API responses > 5 seconds
- Agent task timeouts
- High CPU/memory usage
- Queue backlogs

#### Diagnostic Steps

1. **Performance Profiling**
```bash
# System performance analysis
./scripts/performance-profiler-suite.py --duration 300

# Agent performance breakdown
./scripts/agent-performance-analysis.py
```

2. **Resource Utilization**
```bash
# Real-time resource monitoring
htop

# Docker resource usage
docker stats

# Disk I/O monitoring
iotop
```

3. **Queue Analysis**
```bash
# Check task queues
curl http://localhost:8000/queues/status

# Analyze queue backlogs
./scripts/analyze-queue-backlogs.py
```

#### Common Solutions

**CPU Bottlenecks:**
```bash
# Scale horizontally
docker-compose up --scale api-gateway=3

# Optimize CPU allocation
./scripts/optimize-cpu-allocation.py
```

**Memory Issues:**
```bash
# Clear caches
./scripts/clear-system-caches.py

# Implement memory optimization
./scripts/memory-optimization.py
```

**Database Performance:**
```bash
# Optimize database queries
./scripts/optimize-database-queries.py

# Database maintenance
./scripts/database-maintenance.py
```

### Problem: High Resource Usage

#### Diagnostic Steps

1. **Resource Analysis**
```bash
# Identify resource hogs
./scripts/identify-resource-hogs.py

# Memory usage breakdown
./scripts/memory-usage-analysis.py
```

2. **Process Analysis**
```bash
# Top processes by resource usage
ps aux --sort=-%cpu | head -20
ps aux --sort=-%mem | head -20
```

#### Common Solutions

**Resource Optimization:**
```bash
# Apply resource limits
./scripts/apply-resource-limits.sh

# Optimize resource allocation
./scripts/optimize-resource-allocation.py
```

**Garbage Collection:**
```bash
# Run garbage collection
./scripts/garbage-collection-system.py

# Schedule automatic cleanup
./scripts/schedule-garbage-collection.sh
```

---

## Security and Authentication Issues

### Problem: Authentication Failures

#### Symptoms
- 401 Unauthorized errors
- Token validation failures
- Access denied messages
- SSL/TLS handshake failures

#### Diagnostic Steps

1. **Authentication Status**
```bash
# Check authentication service
curl http://localhost:8000/auth/status

# Validate JWT tokens
./scripts/validate-jwt-tokens.py
```

2. **Certificate Validation**
```bash
# Check SSL certificates
openssl s_client -connect localhost:8000

# Certificate expiration
./scripts/check-certificate-expiry.py
```

#### Common Solutions

**Token Issues:**
```bash
# Regenerate JWT secrets
./scripts/regenerate-jwt-secrets.sh

# Clear token cache
./scripts/clear-token-cache.py
```

**Certificate Problems:**
```bash
# Renew certificates
./scripts/renew-certificates.sh

# Update certificate configuration
./scripts/update-certificate-config.py
```

### Problem: Security Scan Failures

#### Diagnostic Steps

1. **Security Service Status**
```bash
# Check security agents
curl http://localhost:8000/agents/security-pentesting-specialist/health

# Validate security configuration
./scripts/validate-security-config.py
```

#### Common Solutions

**Security Agent Issues:**
```bash
# Restart security services
docker-compose restart security-pentesting-specialist

# Update security rules
./scripts/update-security-rules.py
```

---

## Resource and Scaling Problems

### Problem: Out of Memory Errors

#### Symptoms
- Container OOMKilled status
- Java heap space errors
- Memory allocation failures
- System swap usage

#### Diagnostic Steps

1. **Memory Analysis**
```bash
# System memory status
free -h

# Container memory usage
docker stats --no-stream

# Memory leaks detection
./scripts/detect-memory-leaks.py
```

2. **Application Memory Profiling**
```bash
# JVM heap analysis (if applicable)
./scripts/analyze-jvm-heap.py

# Python memory profiling
./scripts/python-memory-profiler.py
```

#### Common Solutions

**Immediate Relief:**
```bash
# Restart high-memory containers
docker-compose restart [high-memory-service]

# Clear memory caches
./scripts/clear-memory-caches.py
```

**Long-term Solutions:**
```bash
# Increase memory limits
./scripts/increase-memory-limits.sh

# Implement memory optimization
./scripts/implement-memory-optimization.py
```

### Problem: Scaling Issues

#### Diagnostic Steps

1. **Scaling Status**
```bash
# Current scaling configuration
docker-compose ps

# Scaling history
./scripts/scaling-history.py
```

2. **Load Distribution**
```bash
# Load balancer status
curl http://localhost:8000/load-balancer/status

# Traffic distribution
./scripts/analyze-traffic-distribution.py
```

#### Common Solutions

**Horizontal Scaling:**
```bash
# Scale services
docker-compose up --scale api-gateway=3 --scale senior-ai-engineer=2

# Verify scaling
docker-compose ps
```

**Load Balancing:**
```bash
# Reconfigure load balancer
./scripts/reconfigure-load-balancer.py

# Test load distribution
./scripts/test-load-distribution.py
```

---

## Data and Storage Issues

### Problem: Database Connection Failures

#### Symptoms
- Database connection refused
- Connection timeout errors
- Transaction failures
- Data inconsistency

#### Diagnostic Steps

1. **Database Status**
```bash
# Database container status
docker-compose logs postgres

# Connection test
./scripts/test-database-connection.py
```

2. **Database Health**
```bash
# Database health check
./scripts/database-health-check.py

# Query performance analysis
./scripts/analyze-query-performance.py
```

#### Common Solutions

**Connection Issues:**
```bash
# Restart database
docker-compose restart postgres

# Reset database connections
./scripts/reset-database-connections.py
```

**Performance Issues:**
```bash
# Database optimization
./scripts/optimize-database.py

# Index maintenance
./scripts/database-index-maintenance.py
```

### Problem: Storage Space Issues

#### Diagnostic Steps

1. **Storage Analysis**
```bash
# Disk usage
df -h

# Large file identification
du -sh /opt/sutazaiapp/* | sort -hr

# Docker storage usage
docker system df
```

#### Common Solutions

**Cleanup:**
```bash
# Clean Docker resources
docker system prune -a -f

# Clean application logs
./scripts/clean-application-logs.py

# Archive old data
./scripts/archive-old-data.py
```

---

## Network and Connectivity Problems

### Problem: Network Timeouts

#### Symptoms
- Connection timeout errors
- DNS resolution failures
- Inter-service communication issues
- External API failures

#### Diagnostic Steps

1. **Network Connectivity**
```bash
# Test network connectivity
./scripts/test-network-connectivity.py

# DNS resolution test
nslookup api-gateway
```

2. **Port Accessibility**
```bash
# Port scanning
nmap -p 8000-8010 localhost

# Service accessibility
./scripts/test-service-accessibility.py
```

#### Common Solutions

**Network Configuration:**
```bash
# Reset network configuration
docker network prune -f
docker-compose down && docker-compose up -d
```

**DNS Issues:**
```bash
# Update DNS configuration
./scripts/update-dns-config.py

# Clear DNS cache
sudo systemctl flush-dns
```

---

## Advanced Debugging Techniques

### Debug Mode Activation

```bash
# Enable debug logging
export SUTAZAI_LOG_LEVEL=DEBUG
docker-compose restart

# Enable agent debug mode
./scripts/enable-agent-debug-mode.py --all
```

### Deep System Analysis

```bash
# Complete system audit
./scripts/complete-system-audit.py

# Performance bottleneck analysis
./scripts/bottleneck-analysis.py --deep

# Security audit
./scripts/security-audit.py --comprehensive
```

### Log Analysis Tools

```bash
# Log aggregation and analysis
./scripts/aggregate-logs.py --timeframe 24h

# Error pattern detection
./scripts/detect-error-patterns.py

# Performance trend analysis
./scripts/analyze-performance-trends.py
```

### Remote Debugging

```bash
# Enable remote debugging
./scripts/enable-remote-debugging.py --port 5005

# Remote profiling
./scripts/remote-profiling.py --target api-gateway
```

---

## Escalation Procedures

### Level 1: Self-Service Resolution (0-30 minutes)

1. Follow quick diagnostic steps
2. Apply common solutions
3. Check documentation and FAQs
4. Use automated repair tools

**Tools:**
- Quick reference cards
- Automated diagnostic scripts
- Self-healing mechanisms

### Level 2: Team Support (30 minutes - 2 hours)

1. Contact team lead or senior engineer
2. Provide diagnostic information
3. Share logs and error messages
4. Document troubleshooting steps taken

**Information to Provide:**
- System health report
- Error logs (last 1 hour)
- Recent changes made
- Business impact assessment

### Level 3: Expert Support (2-8 hours)

1. Escalate to system architects
2. Engage vendor support if applicable
3. Consider emergency procedures
4. Prepare for potential downtime

**Escalation Criteria:**
- Critical system failure
- Data loss or corruption
- Security breach
- Extended service outage

### Level 4: Emergency Response (Immediate)

1. Activate incident response team
2. Implement emergency procedures
3. Consider system isolation
4. Engage executive stakeholders

**Emergency Situations:**
- Complete system failure
- Security incident
- Data breach
- Customer-impacting outage

### Documentation Requirements

After resolving any issue:

1. **Document the problem** in the incident log
2. **Record the solution** for future reference
3. **Update troubleshooting guides** if needed
4. **Share learnings** with the team
5. **Implement preventive measures** if possible

### Contact Information Template

```
Primary Contact: [System Administrator]
Email: admin@company.com
Phone: +1-XXX-XXX-XXXX

Secondary Contact: [DevOps Lead]
Email: devops@company.com
Phone: +1-XXX-XXX-XXXX

Emergency Contact: [CTO/Technical Director]
Email: cto@company.com
Phone: +1-XXX-XXX-XXXX
```

---

## Prevention and Maintenance

### Proactive Monitoring

```bash
# Set up automated monitoring
./scripts/setup-proactive-monitoring.sh

# Configure alerting
./scripts/configure-alerting.py

# Health check automation
./scripts/automate-health-checks.py
```

### Regular Maintenance Schedule

**Daily:**
- Automated health checks
- Log rotation
- Basic cleanup

**Weekly:**
- Performance analysis
- Security scans
- Configuration backup

**Monthly:**
- Comprehensive system audit
- Capacity planning review
- Update dependency analysis

### Knowledge Base Maintenance

1. **Update troubleshooting guides** based on new issues
2. **Add new solutions** to the knowledge base
3. **Review and update** contact information
4. **Test escalation procedures** regularly
5. **Train team members** on new procedures

---

This troubleshooting guide should be your first reference when issues arise. Remember to always document your findings and solutions to help improve the system and assist future troubleshooting efforts.