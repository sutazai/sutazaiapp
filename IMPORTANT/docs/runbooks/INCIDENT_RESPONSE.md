# Incident Response Runbook - Perfect Jarvis System

**Document Version:** 1.0  
**Last Updated:** 2025-08-08  
**Author:** Incident Response Team  

## 🚨 Purpose

This runbook provides structured incident response procedures for the Perfect Jarvis system, ensuring rapid identification, containment, and resolution of system issues.

## 📊 Incident Severity Classification

### Severity Levels

| Level | Impact | Response Time | Examples |
|-------|--------|---------------|----------|
| **Critical (P0)** | System completely down, data loss risk | 15 minutes | Complete system outage, database corruption |
| **High (P1)** | Major functionality impaired | 1 hour | Backend API down, Ollama service failure |
| **Medium (P2)** | Partial functionality affected | 4 hours | Single agent failure, performance degradation |
| **Low (P3)** | Minor issues, workarounds available | 24 hours | UI glitches, non-critical warnings |

### Impact Assessment Matrix

```
HIGH IMPACT     │ P1  │ P0  │ P0  │
MEDIUM IMPACT   │ P2  │ P1  │ P0  │  
LOW IMPACT      │ P3  │ P2  │ P1  │
                └─────┴─────┴─────┘
                 LOW   MED   HIGH
                  URGENCY LEVEL
```

## 👥 Response Team Roles and Responsibilities

### Incident Commander (IC)
**Primary:** System Administrator  
**Backup:** Senior DevOps Engineer  

**Responsibilities:**
- Overall incident coordination
- Decision making authority
- Communication with stakeholders
- Post-incident review leadership

**Key Contacts:**
- Phone: +1-xxx-xxx-xxxx
- Email: incident-commander@company.com
- Slack: @incident-commander

### Technical Lead (TL)
**Primary:** Backend Developer  
**Backup:** Full-Stack Developer  

**Responsibilities:**
- Technical investigation and diagnosis
- System recovery implementation
- Root cause analysis
- Technical communication with IC

### Operations Lead (OL)
**Primary:** DevOps Engineer  
**Backup:** System Administrator  

**Responsibilities:**
- Infrastructure monitoring
- Service deployment and rollback
- System health verification
- Performance optimization

### Communications Lead (CL)
**Primary:** Product Manager  
**Backup:** Technical Lead  

**Responsibilities:**
- Stakeholder communication
- Status page updates
- Customer notifications
- Documentation coordination

## 📞 Communication Protocols

### Internal Communication Channels

**Slack Channels:**
- `#incidents-critical` - P0/P1 incidents
- `#incidents-general` - P2/P3 incidents
- `#jarvis-ops` - Operational updates
- `#jarvis-dev` - Development discussions

**Email Lists:**
- `incident-response@company.com` - Core response team
- `engineering@company.com` - All engineering staff
- `management@company.com` - Executive updates

### External Communication

**Customer Communication:**
- Status page: `status.company.com`
- Support email: `support@company.com`
- Customer Slack: `#customer-support`

**Vendor/Partner Communication:**
- Cloud provider support
- Third-party service providers
- External contractors

## 🚀 Incident Response Procedures

### Phase 1: Detection and Alert (0-5 minutes)

#### Automated Detection Sources
1. **Monitoring Alerts**
   ```bash
   # Prometheus alerts
   curl -s http://127.0.0.1:10200/api/v1/alerts | jq '.data[].labels'
   
   # Grafana alerts
   curl -s http://admin:admin@127.0.0.1:10201/api/alerts
   ```

2. **Health Check Failures**
   ```bash
   # Backend health check
   curl -s http://127.0.0.1:10010/health | jq '.status'
   
   # Service-specific checks
   curl -s http://127.0.0.1:10104/api/tags  # Ollama
   ```

3. **Log Analysis**
   ```bash
   # Critical errors in logs
   docker-compose logs | grep -i "critical\|fatal\|error" | tail -20
   ```

#### Manual Detection
- User reports via support channels
- Monitoring dashboard observations
- Scheduled health checks

#### Initial Response Actions
1. **Acknowledge Alert** (within 2 minutes)
   ```bash
   echo "$(date): ALERT ACKNOWLEDGED - $(whoami)" >> /opt/sutazaiapp/logs/incident.log
   ```

2. **Assess Severity**
   - Check system dashboard
   - Verify user impact
   - Classify incident level

3. **Initiate Response**
   - Alert response team based on severity
   - Create incident channel
   - Start incident log

### Phase 2: Initial Assessment (5-15 minutes)

#### Quick Triage Checklist
```bash
#!/bin/bash
# incident_triage.sh
echo "=== INCIDENT TRIAGE - $(date) ==="

# Check core services
echo "1. Core Service Status:"
docker-compose ps --services | xargs -I {} sh -c 'echo -n "{}: "; docker-compose ps {} | tail -n +2 | awk "{print \$NF}"'

# Check system resources
echo "2. System Resources:"
echo "   Memory: $(free -h | awk 'NR==2{print $3"/"$2}')"
echo "   CPU: $(top -bn1 | grep "Cpu(s)" | awk '{print $2}')"
echo "   Disk: $(df -h / | awk 'NR==2{print $5}')"

# Check network connectivity
echo "3. Network Status:"
ping -c 1 8.8.8.8 > /dev/null && echo "   Internet: OK" || echo "   Internet: FAIL"

# Check recent errors
echo "4. Recent Errors:"
docker-compose logs --since=10m | grep -i error | wc -l | xargs echo "   Error count (10min):"

echo "=== TRIAGE COMPLETE ==="
```

#### Service-Specific Checks

**Backend API Issues:**
```bash
# Check backend status and logs
curl -s http://127.0.0.1:10010/health | jq
docker-compose logs backend | tail -50

# Check backend dependencies
curl -s http://127.0.0.1:10104/api/tags  # Ollama
docker exec sutazai-postgres pg_isready -U sutazai  # PostgreSQL
docker exec sutazai-redis redis-cli ping  # Redis
```

**Database Issues:**
```bash
# PostgreSQL health
docker exec sutazai-postgres pg_isready -U sutazai
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT 1;"

# Check database size and connections
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
  SELECT 
    pg_size_pretty(pg_database_size('sutazai')) as db_size,
    count(*) as connections 
  FROM pg_stat_activity;"
```

**LLM Service Issues:**
```bash
# Ollama health and models
curl -s http://127.0.0.1:10104/api/tags | jq '.models | length'
docker exec sutazai-ollama ollama list

# Test model inference
curl -X POST http://127.0.0.1:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "test", "stream": false}' | jq '.response'
```

### Phase 3: Containment (15-30 minutes)

#### Immediate Containment Actions

**Stop the Bleeding:**
1. **Traffic Redirection** (if applicable)
   ```bash
   # Redirect traffic to maintenance page
   # Configure load balancer or reverse proxy
   ```

2. **Resource Protection**
   ```bash
   # Stop non-essential services to preserve resources
   docker-compose stop sutazai-grafana sutazai-loki  # Monitoring can be paused
   
   # Limit resource usage
   docker update --memory=2g --cpus="1.0" sutazai-ollama
   ```

3. **Data Protection**
   ```bash
   # Emergency database backup
   docker exec sutazai-postgres pg_dump -U sutazai sutazai > emergency_backup_$(date +%s).sql
   
   # Protect Redis data
   docker exec sutazai-redis redis-cli BGSAVE
   ```

#### Service-Specific Containment

**Backend API Failure:**
```bash
# Quick restart attempt
docker-compose restart backend

# If restart fails, recreate with clean state
docker-compose stop backend
docker-compose rm -f backend
docker-compose up -d backend

# Monitor startup
docker-compose logs -f backend
```

**Database Issues:**
```bash
# PostgreSQL emergency procedures
if ! docker exec sutazai-postgres pg_isready -U sutazai; then
    echo "PostgreSQL unresponsive - attempting recovery"
    
    # Stop dependent services
    docker-compose stop backend
    
    # Restart PostgreSQL
    docker-compose restart sutazai-postgres
    
    # Wait for recovery
    timeout 60 bash -c 'until docker exec sutazai-postgres pg_isready -U sutazai; do sleep 2; done'
    
    # Restart dependencies
    docker-compose start backend
fi
```

**Memory/Resource Issues:**
```bash
# Emergency resource cleanup
docker system prune -f
docker volume prune -f

# Kill memory-intensive processes
docker stats --no-stream | awk 'NR>1 && $3+0 > 80 {print $2}' | xargs docker restart

# Clear application caches
docker exec sutazai-redis redis-cli FLUSHDB
```

### Phase 4: Investigation (30-60 minutes)

#### Log Analysis
```bash
#!/bin/bash
# incident_investigation.sh
INCIDENT_ID=$1
INVESTIGATION_DIR="/opt/sutazaiapp/logs/incidents/$INCIDENT_ID"

mkdir -p "$INVESTIGATION_DIR"

echo "=== INCIDENT INVESTIGATION: $INCIDENT_ID ==="

# Collect system state
docker-compose ps > "$INVESTIGATION_DIR/container_status.txt"
docker stats --no-stream > "$INVESTIGATION_DIR/resource_usage.txt"
docker system df > "$INVESTIGATION_DIR/disk_usage.txt"

# Collect logs from all services
for service in backend sutazai-ollama sutazai-postgres sutazai-redis; do
    docker-compose logs --since=2h $service > "$INVESTIGATION_DIR/${service}_logs.txt"
done

# Collect system metrics
curl -s http://127.0.0.1:10010/health > "$INVESTIGATION_DIR/backend_health.json"
curl -s http://127.0.0.1:10104/api/tags > "$INVESTIGATION_DIR/ollama_models.json"

# Collect OS-level information
top -bn1 > "$INVESTIGATION_DIR/system_processes.txt"
free -h > "$INVESTIGATION_DIR/memory_status.txt"
df -h > "$INVESTIGATION_DIR/disk_status.txt"

echo "Investigation data collected in: $INVESTIGATION_DIR"
```

#### Root Cause Analysis Framework

**The 5 Whys Method:**
1. **Why did the incident occur?**
   - Document the immediate trigger

2. **Why did that trigger cause a problem?**
   - Identify the underlying weakness

3. **Why wasn't it prevented?**
   - Examine prevention mechanisms

4. **Why didn't monitoring catch it sooner?**
   - Review detection capabilities

5. **Why wasn't the system resilient?**
   - Analyze recovery mechanisms

**Common Root Causes:**
- Resource exhaustion (memory, CPU, disk)
- Configuration errors
- Dependency failures
- Code bugs or regressions
- Infrastructure issues
- Human error

### Phase 5: Recovery (1-4 hours)

#### Recovery Strategies

**Service Restart Recovery:**
```bash
#!/bin/bash
# service_recovery.sh
SERVICE=$1

echo "Starting recovery for service: $SERVICE"

# Stop service gracefully
docker-compose stop $SERVICE

# Remove container to ensure clean state
docker-compose rm -f $SERVICE

# Pull latest image (if needed)
docker-compose pull $SERVICE

# Start service
docker-compose up -d $SERVICE

# Verify recovery
timeout 300 bash -c "
  while ! docker-compose ps $SERVICE | grep -q 'Up'; do
    echo 'Waiting for $SERVICE to start...'
    sleep 10
  done
"

echo "Service $SERVICE recovery completed"
```

**Database Recovery:**
```bash
#!/bin/bash
# database_recovery.sh
BACKUP_FILE=$1

echo "Starting database recovery..."

# Stop backend to prevent connections
docker-compose stop backend

# Restore from backup if provided
if [ -n "$BACKUP_FILE" ]; then
    echo "Restoring from backup: $BACKUP_FILE"
    cat "$BACKUP_FILE" | docker exec -i sutazai-postgres psql -U sutazai sutazai
fi

# Verify database integrity
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT version();"

# Start backend
docker-compose start backend

# Verify backend connectivity
timeout 60 bash -c 'until curl -s http://127.0.0.1:10010/health; do sleep 2; done'

echo "Database recovery completed"
```

**Complete System Recovery:**
```bash
#!/bin/bash
# full_system_recovery.sh
echo "=== FULL SYSTEM RECOVERY ==="

# Stop all services
docker-compose down

# Clean up resources
docker system prune -f

# Start core services first
docker-compose up -d sutazai-postgres sutazai-redis sutazai-neo4j
sleep 60

# Start LLM service
docker-compose up -d sutazai-ollama
sleep 30

# Start backend
docker-compose up -d backend
sleep 30

# Start monitoring
docker-compose up -d sutazai-prometheus sutazai-grafana

# Verify all services
./health_dashboard.sh

echo "=== SYSTEM RECOVERY COMPLETE ==="
```

### Phase 6: Verification (15-30 minutes)

#### Health Verification Checklist
```bash
#!/bin/bash
# post_recovery_verification.sh
echo "=== POST-RECOVERY VERIFICATION ==="

# Test core functionality
echo "1. Testing Backend Health:"
curl -s http://127.0.0.1:10010/health | jq '.status'

echo "2. Testing Model Inference:"
curl -s -X POST http://127.0.0.1:10010/simple-chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello, system test"}' | jq '.response'

echo "3. Testing Database:"
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT 'Database OK';"

echo "4. Testing Redis:"
docker exec sutazai-redis redis-cli ping

echo "5. Performance Check:"
curl -s http://127.0.0.1:10010/public/metrics | jq '.system'

echo "=== VERIFICATION COMPLETE ==="
```

## ⚡ Escalation Procedures

### Escalation Matrix

| Time Elapsed | P0 Critical | P1 High | P2 Medium |
|--------------|-------------|---------|-----------|
| 15 minutes | → Level 2 | Initial response | Monitor |
| 1 hour | → Level 3 | → Level 2 | Initial response |
| 4 hours | → Level 4 | → Level 3 | → Level 2 |
| 24 hours | Executive | → Level 4 | → Level 3 |

### Escalation Contacts

**Level 1: On-Call Engineer**
- Primary: on-call@company.com
- Phone: +1-xxx-xxx-1111
- Response: Immediate

**Level 2: Engineering Manager**
- Primary: eng-manager@company.com
- Phone: +1-xxx-xxx-2222
- Response: Within 30 minutes

**Level 3: Technical Director**
- Primary: tech-director@company.com
- Phone: +1-xxx-xxx-3333
- Response: Within 1 hour

**Level 4: VP Engineering**
- Primary: vp-eng@company.com
- Phone: +1-xxx-xxx-4444
- Response: Within 2 hours

### Escalation Triggers

**Automatic Escalation:**
- Incident duration exceeds time thresholds
- Multiple system failures
- Data loss detected
- Security breach suspected

**Manual Escalation:**
- Technical team lacks expertise
- External vendor involvement needed
- Customer escalation received
- Media/PR implications

## 🔍 Root Cause Analysis Process

### Post-Incident Analysis Framework

#### 1. Timeline Reconstruction
```markdown
## Incident Timeline

| Time | Event | Action Taken | Decision Maker |
|------|-------|-------------|----------------|
| 14:23 | Alert received | Acknowledged | On-call engineer |
| 14:25 | Severity assessed | P1 declared | Incident commander |
| 14:30 | Service restart attempted | Failed | Technical lead |
| 14:45 | Database issue identified | Recovery initiated | Operations lead |
| 15:20 | Service restored | Verification started | Technical lead |
| 15:35 | Full recovery confirmed | Incident closed | Incident commander |
```

#### 2. Impact Assessment
```markdown
## Impact Summary

**Duration:** X hours Y minutes
**Services Affected:** Backend API, Chat functionality
**Users Impacted:** ~X users
**Revenue Impact:** $X estimated
**SLA Impact:** Missed 99.9% uptime target for month
```

#### 3. Root Cause Analysis

**Immediate Cause:**
- What directly caused the incident?

**Underlying Cause:**
- What system weakness allowed this to happen?

**Root Cause:**
- What organizational/process gap enabled this?

### Action Items Framework

```markdown
## Action Items

| ID | Description | Owner | Due Date | Priority |
|----|-------------|-------|----------|----------|
| AI-001 | Implement better monitoring for X | DevOps | 2025-08-15 | P0 |
| AI-002 | Update runbook for Y scenario | SRE | 2025-08-20 | P1 |
| AI-003 | Review alert thresholds | On-call | 2025-08-25 | P2 |
```

## 📋 Post-Incident Review Template

### Meeting Agenda (1 hour)
1. **Timeline Review** (15 minutes)
   - Walk through incident timeline
   - Identify decision points

2. **Root Cause Analysis** (20 minutes)
   - Discuss findings
   - Validate root cause

3. **Response Evaluation** (15 minutes)
   - What went well?
   - What could be improved?

4. **Action Planning** (10 minutes)
   - Define specific action items
   - Assign owners and dates

### Blameless Postmortem Template

```markdown
# Post-Incident Review: [Incident ID]

**Date:** YYYY-MM-DD
**Incident Commander:** Name
**Attendees:** List of participants

## Executive Summary
Brief summary of incident impact and resolution.

## Incident Details
- **Start Time:** YYYY-MM-DD HH:MM UTC
- **Resolution Time:** YYYY-MM-DD HH:MM UTC
- **Duration:** X hours Y minutes
- **Severity:** P1

## What Went Well
- Quick detection and response
- Effective communication
-   data loss

## What Could Be Improved
- Detection could be faster
- Recovery time longer than expected
- Communication gaps

## Root Cause
Detailed explanation of the root cause.

## Action Items
List of specific actions to prevent recurrence.

## Lessons Learned
Key insights for future incidents.
```

## 📊 Common Incident Scenarios and Resolutions

### Scenario 1: Backend API Unresponsive

**Symptoms:**
- Health check failures
- 502/503 errors
- High response times

**Common Causes:**
- Memory exhaustion
- Database connection pool exhaustion
- Ollama service unavailable

**Resolution Steps:**
```bash
# 1. Check service status
docker-compose ps backend
curl -s http://127.0.0.1:10010/health

# 2. Check dependencies
curl -s http://127.0.0.1:10104/api/tags
docker exec sutazai-postgres pg_isready -U sutazai

# 3. Check resources
docker stats backend --no-stream

# 4. Restart if needed
docker-compose restart backend
```

### Scenario 2: Database Connection Failures

**Symptoms:**
- Database connection errors
- Backend degraded status
- Transaction failures

**Common Causes:**
- PostgreSQL service down
- Connection pool exhaustion
- Disk space full

**Resolution Steps:**
```bash
# 1. Check PostgreSQL status
docker exec sutazai-postgres pg_isready -U sutazai

# 2. Check connections
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT count(*) FROM pg_stat_activity;"

# 3. Check disk space
df -h

# 4. Restart PostgreSQL if needed
docker-compose restart sutazai-postgres
```

### Scenario 3: Model Inference Failures

**Symptoms:**
- "No models available" errors
- Slow response times
- Model loading failures

**Common Causes:**
- Ollama service down
- Model not loaded
- Insufficient memory

**Resolution Steps:**
```bash
# 1. Check Ollama status
curl -s http://127.0.0.1:10104/api/tags

# 2. Check loaded models
docker exec sutazai-ollama ollama list

# 3. Load required model
docker exec sutazai-ollama ollama pull tinyllama

# 4. Restart services
docker-compose restart sutazai-ollama backend
```

### Scenario 4: High Resource Usage

**Symptoms:**
- System slow response
- High CPU/memory usage
- Out of memory errors

**Common Causes:**
- Model memory leak
- Large dataset processing
- Inefficient queries

**Resolution Steps:**
```bash
# 1. Check resource usage
docker stats --no-stream

# 2. Identify resource-heavy containers
docker stats --no-stream | sort -k3 -nr

# 3. Emergency resource cleanup
docker system prune -f
docker restart $(docker stats --no-stream | awk 'NR>1 && $3+0 > 80 {print $2}')

# 4. Optimize configuration
# Reduce model context window
# Limit concurrent requests
```

---

## 🚨 Emergency Contact Information

**24/7 On-Call:**
- Phone: +1-xxx-xxx-xxxx
- Email: oncall@company.com
- PagerDuty: https://company.pagerduty.com

**Incident Commander:**
- Primary: +1-xxx-xxx-1111
- Secondary: +1-xxx-xxx-2222

**Executive Escalation:**
- VP Engineering: +1-xxx-xxx-3333
- CTO: +1-xxx-xxx-4444

**External Vendors:**
- Cloud Provider: 1-800-xxx-xxxx
- Support Vendor: 1-800-yyy-yyyy

---

*This incident response runbook is based on actual system architecture and common failure patterns. Update procedures as the system evolves and lessons are learned from real incidents.*