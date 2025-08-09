# Workshop Day 3: Administration and Maintenance

**Duration:** 8 hours (with breaks)  
**Target Audience:** System Administrators, DevOps Engineers, Technical Managers  
**Prerequisites:** Days 1-2 or equivalent experience, technical background

## 🎯 Learning Objectives

By the end of Day 3, participants will:
- Master system administration and configuration management
- Implement comprehensive monitoring and alerting
- Execute backup and recovery procedures
- Optimize system performance and resource usage
- Troubleshoot complex technical issues
- Plan and execute system upgrades and maintenance
- Establish operational procedures and documentation

## 📅 Schedule

| Time | Topic | Duration |
|------|-------|----------|
| 9:00-9:30 | Administrator Introduction & System Architecture | 30 min |
| 9:30-10:45 | System Configuration & Management | 75 min |
| 10:45-11:00 | **Break** | 15 min |
| 11:00-12:15 | Monitoring, Logging & Alerting | 75 min |
| 12:15-13:15 | **Lunch Break** | 60 min |
| 13:15-14:30 | Backup, Recovery & Disaster Planning | 75 min |
| 14:30-14:45 | **Break** | 15 min |
| 14:45-16:00 | Performance Optimization & Troubleshooting | 75 min |
| 16:00-16:15 | **Break** | 15 min |
| 16:15-16:45 | Security & Compliance | 30 min |
| 16:45-17:00 | Operational Procedures & Documentation | 15 min |

---

## 🏗️ Session 1: Administrator Introduction & System Architecture (30 min)

### Administrator Role Overview (10 min)

#### Core Responsibilities
**System Administration Scope**:
- **Service Management**: Ensuring all containers and services run smoothly
- **Performance Monitoring**: Tracking system health and resource usage
- **Data Management**: Backup, recovery, and data integrity
- **Security**: Access control, vulnerability management, compliance
- **Maintenance**: Updates, patches, and system optimization
- **Documentation**: Procedures, configurations, and troubleshooting guides

#### Perfect Jarvis Administrator Context
**Current System Reality**:
- 59 services defined in docker-compose.yml
- 28 containers actually running
- 7 functional agent services (currently stubs)
- Local TinyLlama model (637MB)
- Comprehensive monitoring stack operational

### Deep Architecture Review (20 min)

#### Container Architecture Details
```
┌─────────────────────────────────────────────────────────────┐
│                    Production Infrastructure                 │
├─────────────────────────────────────────────────────────────┤
│  Frontend (Streamlit) │  Backend (FastAPI) │ Agent Services │
│      Port 10011       │     Port 10010      │   Various      │
├─────────────────────────────────────────────────────────────┤
│  PostgreSQL │  Redis  │  Neo4j  │ Ollama/TinyLlama │ Vector │
│   10000     │  10001  │ 10002/3 │      10104       │   DB   │
├─────────────────────────────────────────────────────────────┤
│  Prometheus │ Grafana │  Loki   │ AlertManager │  Exporters │
│    10200    │  10201  │ 10202   │    10203     │   10220+   │
└─────────────────────────────────────────────────────────────┘
```

#### Service Dependencies
**Critical Path Analysis**:
1. **Core Services**: PostgreSQL → Redis → Backend API
2. **AI Services**: Ollama → Backend → Frontend
3. **Monitoring**: Prometheus → Grafana → AlertManager
4. **Optional**: Vector DBs, Agent Services, Service Mesh

#### Resource Requirements
**Per-Service Resource Usage**:
- **PostgreSQL**: 512MB RAM, 2GB storage
- **Redis**: 256MB RAM, 100MB storage
- **Ollama/TinyLlama**: 1GB RAM, 1GB storage
- **Backend API**: 512MB RAM, 100MB storage
- **Frontend**: 256MB RAM, 50MB storage
- **Monitoring Stack**: 1GB RAM, 5GB storage

---

## 🔧 Session 2: System Configuration & Management (75 min)

### Configuration Management (30 min)

#### Exercise 1: Configuration File Audit (15 min)
**Configuration Inventory**:
```bash
# List all configuration files
find /opt/sutazaiapp -name "*.yml" -o -name "*.yaml" -o -name "*.json" | head -20

# Key configuration files to examine:
cat /opt/sutazaiapp/docker-compose.yml | head -50
cat /opt/sutazaiapp/config/port-registry.yaml
cat /opt/sutazaiapp/config/services.yaml
cat /opt/sutazaiapp/config/ollama.yaml
```

**Configuration Categories**:
- **Core Services**: database, cache, API settings
- **AI/ML Models**: Ollama configuration, model parameters
- **Monitoring**: Prometheus rules, Grafana dashboards
- **Networking**: Port mappings, service discovery
- **Security**: Authentication, authorization, encryption

#### Exercise 2: Environment Variable Management (15 min)
**Environment Configuration Review**:
```bash
# Check current environment variables
docker-compose config | grep -A 10 -B 10 environment

# Critical environment variables:
- POSTGRES_PASSWORD
- REDIS_PASSWORD  
- API_SECRET_KEY
- OLLAMA_HOST
- MONITORING_ENABLED
```

**Best Practices**:
- Use `.env` files for sensitive data
- Document all required variables
- Implement variable validation
- Use different configs for dev/staging/prod

### Service Management (25 min)

#### Exercise 3: Service Lifecycle Management (15 min)
**Complete Service Management**:
```bash
# System startup sequence
docker network create sutazai-network 2>/dev/null
docker-compose up -d

# Check service dependencies
docker-compose ps
docker-compose logs --tail=50 backend
docker-compose logs --tail=50 ollama

# Rolling restart strategy
docker-compose restart redis
sleep 10
docker-compose restart backend
sleep 10
docker-compose restart frontend
```

**Service Health Verification**:
```bash
# Health check endpoints
curl -s http://localhost:10010/health | jq
curl -s http://localhost:10104/api/tags | jq
curl -s http://localhost:10200/api/v1/targets | jq
```

#### Exercise 4: Resource Management (10 min)
**Container Resource Monitoring**:
```bash
# Real-time resource usage
docker stats --no-stream

# Memory usage by service
docker exec sutazai-postgres cat /proc/meminfo | grep MemTotal
docker exec sutazai-redis redis-cli info memory
docker system df
```

**Resource Optimization**:
```yaml
# docker-compose.yml resource limits example
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.0"
        reservations:
          memory: 1G
          cpus: "0.5"
```

### Database Administration (20 min)

#### Exercise 5: PostgreSQL Administration (10 min)
**Database Health and Maintenance**:
```bash
# Connect to PostgreSQL
docker exec -it sutazai-postgres psql -U sutazai -d sutazai

# Inside PostgreSQL:
\dt                           # List tables
\d+ users                     # Describe users table
SELECT COUNT(*) FROM users;   # Count records
\q                           # Exit
```

**Performance Monitoring Queries**:
```sql
-- Active connections
SELECT * FROM pg_stat_activity WHERE state = 'active';

-- Slow queries (if pg_stat_statements enabled)
SELECT query, mean_time, calls, total_time 
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 10;

-- Database size
SELECT pg_size_pretty(pg_database_size('sutazai'));
```

#### Exercise 6: Redis Administration (10 min)
**Redis Monitoring and Management**:
```bash
# Connect to Redis
docker exec -it sutazai-redis redis-cli

# Redis commands:
INFO memory                   # Memory usage
INFO stats                    # Statistics
CONFIG GET maxmemory          # Memory limit
SLOWLOG GET 10               # Slow queries
KEYS *                       # List keys (use carefully)
```

**Redis Optimization**:
```bash
# Set memory policy
CONFIG SET maxmemory-policy allkeys-lru

# Monitor performance
redis-cli --latency-history -i 1
```

---

## 📊 Session 3: Monitoring, Logging & Alerting (75 min)

### Comprehensive Monitoring Setup (30 min)

#### Exercise 7: Prometheus Configuration (15 min)
**Prometheus Targets Review**:
```bash
# Check Prometheus configuration
curl -s http://localhost:10200/api/v1/targets | jq '.data.activeTargets[] | {job, health, lastScrape}'

# Key metrics to monitor
curl -s "http://localhost:10200/api/v1/query?query=up" | jq
curl -s "http://localhost:10200/api/v1/query?query=container_memory_usage_bytes" | jq
```

**Custom Metrics Configuration**:
```yaml
# /monitoring/prometheus-rules.yml
groups:
  - name: jarvis_alerts
    rules:
      - alert: JarvisBackendDown
        expr: up{job="backend"} == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "JARVIS Backend is down"
          
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, http_request_duration_seconds) > 10
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
```

#### Exercise 8: Grafana Dashboard Management (15 min)
**Access and Configure Grafana**:
1. **Navigate to**: http://localhost:10201
2. **Login**: admin/admin (change password on first login)
3. **Import Dashboard**: Use pre-built JSON configs
4. **Create Custom Dashboard**: Add panels for specific metrics

**Key Dashboard Panels**:
- **System Overview**: CPU, memory, disk usage
- **Service Health**: Up/down status for all services  
- **Response Times**: API endpoint performance
- **Database Performance**: Query times, connections
- **LLM Usage**: Model inference statistics

### Advanced Logging (25 min)

#### Exercise 9: Centralized Log Management (15 min)
**Loki and Log Aggregation**:
```bash
# Check Loki status
curl -s http://localhost:10202/ready

# Query logs via API
curl -G -s "http://localhost:10202/loki/api/v1/query" \
  --data-urlencode 'query={container_name="sutazai-backend"}' | jq
```

**Log Analysis in Grafana**:
1. Add Loki as data source
2. Create log dashboard
3. Set up log-based alerts
4. Configure log retention policies

#### Exercise 10: Application Logging (10 min)
**Backend Application Logs**:
```bash
# Application-specific logs
docker-compose logs -f --tail=100 backend | grep ERROR
docker-compose logs -f --tail=100 ollama | grep -i "error\|warn"

# Log rotation and cleanup
docker system prune -f
docker volume prune -f
```

**Log Structure Best Practices**:
```python
# Example structured logging in backend
import logging
import json

logger = logging.getLogger(__name__)

def log_api_request(endpoint, user_id, response_time):
    log_data = {
        "timestamp": datetime.utcnow().isoformat(),
        "endpoint": endpoint,
        "user_id": user_id,
        "response_time": response_time,
        "component": "api"
    }
    logger.info(json.dumps(log_data))
```

### Alerting and Notification (20 min)

#### Exercise 11: AlertManager Configuration (10 min)
**Alert Routing Setup**:
```yaml
# /monitoring/alertmanager.yml
global:
  smtp_smarthost: 'localhost:587'
  smtp_from: 'alerts@jarvis.local'

route:
  group_by: ['alertname']
  group_wait: 10s
  group_interval: 10s
  repeat_interval: 1h
  receiver: 'web.hook'

receivers:
  - name: 'web.hook'
    email_configs:
      - to: 'admin@jarvis.local'
        subject: 'JARVIS Alert: {{ .GroupLabels.alertname }}'
        body: |
          {{ range .Alerts }}
          Alert: {{ .Annotations.summary }}
          Description: {{ .Annotations.description }}
          {{ end }}
```

#### Exercise 12: Custom Alert Testing (10 min)
**Alert Simulation and Testing**:
```bash
# Simulate service failure
docker stop sutazai-backend
sleep 60  # Wait for alert to fire
docker start sutazai-backend

# Check alert status
curl -s http://localhost:10203/api/v1/alerts | jq
```

**Health Check Scripts**:
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/health-check.sh

services=("backend" "postgres" "redis" "ollama")
failed_services=()

for service in "${services[@]}"; do
    if ! docker-compose ps $service | grep -q "Up"; then
        failed_services+=($service)
    fi
done

if [ ${#failed_services[@]} -ne 0 ]; then
    echo "CRITICAL: Services down: ${failed_services[*]}"
    exit 2
else
    echo "OK: All services running"
    exit 0
fi
```

---

## 💾 Session 4: Backup, Recovery & Disaster Planning (75 min)

### Comprehensive Backup Strategy (30 min)

#### Exercise 13: Database Backup Procedures (15 min)
**PostgreSQL Backup**:
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/backup-postgres.sh

BACKUP_DIR="/opt/sutazaiapp/backups"
DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_FILE="$BACKUP_DIR/postgres_backup_$DATE.sql"

mkdir -p "$BACKUP_DIR"

# Full database backup
docker exec sutazai-postgres pg_dump -U sutazai -d sutazai > "$BACKUP_FILE"

# Compress backup
gzip "$BACKUP_FILE"

# Verify backup
if [ -f "$BACKUP_FILE.gz" ]; then
    echo "Backup completed: $BACKUP_FILE.gz"
    # Optional: Upload to remote storage
else
    echo "Backup failed!"
    exit 1
fi
```

**Redis Backup**:
```bash
#!/bin/bash
# Redis backup script
BACKUP_DIR="/opt/sutazaiapp/backups"
DATE=$(date +%Y%m%d_%H%M%S)

# Create Redis backup
docker exec sutazai-redis redis-cli BGSAVE
docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/redis_backup_$DATE.rdb"

echo "Redis backup completed: redis_backup_$DATE.rdb"
```

#### Exercise 14: Configuration and Application Backup (15 min)
**Complete System Backup**:
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/full-backup.sh

BACKUP_DIR="/opt/sutazaiapp/backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Database backups
./backup-postgres.sh
./backup-redis.sh

# Configuration backup
tar -czf "$BACKUP_DIR/configurations.tar.gz" \
    config/ \
    docker-compose.yml \
    .env

# Application data backup
tar -czf "$BACKUP_DIR/application_data.tar.gz" \
    logs/ \
    data/ \
    monitoring/

# Document backup contents
echo "Backup completed: $BACKUP_DIR" > "$BACKUP_DIR/backup_manifest.txt"
echo "Timestamp: $(date)" >> "$BACKUP_DIR/backup_manifest.txt"
echo "Contents:" >> "$BACKUP_DIR/backup_manifest.txt"
ls -la "$BACKUP_DIR" >> "$BACKUP_DIR/backup_manifest.txt"
```

### Recovery Procedures (25 min)

#### Exercise 15: Database Recovery Testing (15 min)
**PostgreSQL Recovery**:
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/restore-postgres.sh

BACKUP_FILE="$1"
if [ -z "$BACKUP_FILE" ]; then
    echo "Usage: $0 <backup_file.sql.gz>"
    exit 1
fi

# Stop backend to prevent connections
docker-compose stop backend

# Restore database
gunzip -c "$BACKUP_FILE" | docker exec -i sutazai-postgres psql -U sutazai -d sutazai

# Restart services
docker-compose start backend

# Verify restoration
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "\dt"
echo "Database restoration completed"
```

**Recovery Validation**:
```bash
# Test system functionality after recovery
curl -s http://localhost:10010/health | jq
curl -s http://localhost:10011 | head -10
```

#### Exercise 16: Disaster Recovery Simulation (10 min)
**Complete System Recovery**:
```bash
# Simulate disaster: Stop all services
docker-compose down

# Remove data (CAUTION: Test environment only!)
docker volume rm $(docker volume ls -q | grep sutazai)

# Restore from backup
./restore-complete-system.sh /path/to/backup/20250808_120000

# Verify recovery
./health-check.sh
```

### Disaster Recovery Planning (20 min)

#### Exercise 17: DR Plan Development (10 min)
**Disaster Recovery Checklist**:
```markdown
# JARVIS Disaster Recovery Plan

## Recovery Time Objective (RTO): 2 hours
## Recovery Point Objective (RPO): 1 hour

### Phase 1: Assessment (15 minutes)
- [ ] Identify affected services
- [ ] Determine scope of failure
- [ ] Locate most recent backups
- [ ] Notify stakeholders

### Phase 2: Recovery (1 hour 30 minutes)
- [ ] Restore infrastructure
- [ ] Recover databases from backup
- [ ] Restore configuration files
- [ ] Restart services in dependency order
- [ ] Validate system functionality

### Phase 3: Verification (15 minutes)
- [ ] Test all critical endpoints
- [ ] Verify data integrity
- [ ] Check monitoring systems
- [ ] Document recovery process
```

#### Exercise 18: Backup Testing and Validation (10 min)
**Automated Backup Testing**:
```bash
#!/bin/bash
# /opt/sutazaiapp/scripts/test-backup.sh

# Create test backup
./full-backup.sh

# Create temporary test environment
docker-compose -f docker-compose.test.yml up -d

# Restore backup to test environment
./restore-complete-system.sh /path/to/latest/backup

# Run validation tests
./validate-system.sh

# Cleanup test environment
docker-compose -f docker-compose.test.yml down

echo "Backup validation completed"
```

---

## ⚡ Session 5: Performance Optimization & Troubleshooting (75 min)

### System Performance Analysis (30 min)

#### Exercise 19: Performance Baseline Establishment (15 min)
**System Metrics Collection**:
```bash
#!/bin/bash
# Performance baseline script

echo "=== System Performance Baseline ==="
echo "Timestamp: $(date)"

# CPU and Memory
echo "CPU Usage:"
top -b -n1 | grep "Cpu(s)"

echo "Memory Usage:"
free -h

echo "Disk Usage:"
df -h

# Container Performance
echo "Container Stats:"
docker stats --no-stream --format "table {{.Container}}\t{{.CPUPerc}}\t{{.MemUsage}}"

# Service Response Times
echo "Service Response Times:"
time curl -s http://localhost:10010/health > /dev/null
time curl -s http://localhost:10104/api/tags > /dev/null
```

#### Exercise 20: Performance Bottleneck Identification (15 min)
**Load Testing and Analysis**:
```bash
# Simple load test
for i in {1..10}; do
    curl -s -w "%{time_total}\n" -o /dev/null \
        -X POST http://localhost:10010/api/v1/chat \
        -H "Content-Type: application/json" \
        -d '{"message": "Performance test ' $i '"}' &
done
wait

# Monitor during load
watch -n 1 'docker stats --no-stream | head -10'
```

**Resource Utilization Analysis**:
```bash
# Database performance
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
SELECT schemaname,tablename,attname,n_distinct,correlation 
FROM pg_stats 
WHERE tablename = 'users';"

# Redis performance
docker exec sutazai-redis redis-cli --latency-history -i 1 | head -20
```

### Optimization Implementation (25 min)

#### Exercise 21: Database Optimization (10 min)
**PostgreSQL Performance Tuning**:
```sql
-- Connect to PostgreSQL and run optimization queries
-- docker exec -it sutazai-postgres psql -U sutazai -d sutazai

-- Check for missing indexes
SELECT schemaname, tablename, attname, n_distinct, correlation
FROM pg_stats
WHERE schemaname = 'public';

-- Analyze table statistics
ANALYZE;

-- Check slow queries (if enabled)
SELECT query, calls, mean_time, total_time
FROM pg_stat_statements 
ORDER BY mean_time DESC 
LIMIT 5;
```

**Redis Optimization**:
```bash
# Redis memory optimization
docker exec sutazai-redis redis-cli CONFIG SET maxmemory 512mb
docker exec sutazai-redis redis-cli CONFIG SET maxmemory-policy allkeys-lru

# Monitor memory usage
docker exec sutazai-redis redis-cli INFO memory | grep used_memory_human
```

#### Exercise 22: Application Performance Optimization (15 min)
**Container Resource Optimization**:
```yaml
# Optimized docker-compose.yml settings
version: '3.8'
services:
  backend:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: "1.5"
        reservations:
          memory: 1G
          cpus: "0.5"
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 40s

  postgres:
    command: >
      postgres
      -c shared_buffers=256MB
      -c max_connections=100
      -c effective_cache_size=1GB
      -c maintenance_work_mem=64MB
```

**Caching Strategy**:
```bash
# Implement Redis caching for API responses
# Configure cache TTL based on content type
docker exec sutazai-redis redis-cli CONFIG SET timeout 300
```

### Advanced Troubleshooting (20 min)

#### Exercise 23: Log Analysis and Debugging (10 min)
**Systematic Log Analysis**:
```bash
# Error pattern analysis
docker-compose logs backend | grep -E "(ERROR|CRITICAL|EXCEPTION)" | tail -20

# Performance issue identification  
docker-compose logs backend | grep -E "slow|timeout|latency" | tail -20

# Connection issues
docker-compose logs postgres | grep -E "(connection|authentication)" | tail -20
```

**Advanced Debugging Techniques**:
```bash
# Container resource limits investigation
docker inspect sutazai-backend | jq '.HostConfig.Memory'
docker inspect sutazai-backend | jq '.HostConfig.CpuShares'

# Network connectivity testing
docker exec sutazai-backend ping -c 3 sutazai-postgres
docker exec sutazai-backend nslookup sutazai-ollama
```

#### Exercise 24: Root Cause Analysis (10 min)
**Systematic Problem Resolution**:
```bash
#!/bin/bash
# Troubleshooting checklist script

echo "=== JARVIS Troubleshooting Analysis ==="

# 1. Service Status Check
echo "1. Service Status:"
docker-compose ps

# 2. Resource Usage Check
echo "2. Resource Usage:"
docker stats --no-stream | grep -E "(sutazai|NAME)"

# 3. Network Connectivity
echo "3. Network Connectivity:"
curl -s -o /dev/null -w "%{http_code} %{time_total}s" http://localhost:10010/health
curl -s -o /dev/null -w "%{http_code} %{time_total}s" http://localhost:10104/api/tags

# 4. Error Logs (last 5 minutes)
echo "4. Recent Errors:"
docker-compose logs --since=5m | grep -i error | tail -5

# 5. Disk Space
echo "5. Disk Space:"
df -h | grep -E "(Filesystem|/dev)"
```

---

## 🔒 Session 6: Security & Compliance (30 min)

### Security Configuration (15 min)

#### Exercise 25: Security Audit (10 min)
**Security Checklist Review**:
```bash
#!/bin/bash
# Security audit script

echo "=== JARVIS Security Audit ==="

# 1. Check default passwords
echo "1. Password Security:"
docker-compose config | grep -i password | head -5

# 2. Network exposure
echo "2. Network Exposure:"
netstat -tlnp | grep -E "(10000|10001|10010|10011|10104)"

# 3. File permissions
echo "3. File Permissions:"
ls -la /opt/sutazaiapp/ | head -10

# 4. Container security
echo "4. Container Security:"
docker inspect sutazai-backend | jq '.HostConfig.Privileged'
```

**Security Hardening**:
```yaml
# Security-focused docker-compose.yml additions
version: '3.8'
services:
  backend:
    security_opt:
      - no-new-privileges:true
    read_only: true
    tmpfs:
      - /tmp:rw,size=100M
    
  postgres:
    environment:
      POSTGRES_PASSWORD_FILE: /run/secrets/postgres_password
    secrets:
      - postgres_password

secrets:
  postgres_password:
    file: ./secrets/postgres_password.txt
```

#### Exercise 26: Access Control Implementation (5 min)
**Basic Access Control**:
```bash
# Network access restrictions (iptables example)
# Allow only localhost access to sensitive ports
sudo iptables -A INPUT -p tcp --dport 10000 -s 127.0.0.1 -j ACCEPT
sudo iptables -A INPUT -p tcp --dport 10000 -j DROP

# Container network isolation
docker network create --driver bridge --internal sutazai-internal
```

### Compliance and Documentation (15 min)

#### Exercise 27: Compliance Monitoring (10 min)
**Audit Trail Implementation**:
```bash
# Enable audit logging
echo "audit_log_enabled = true" >> /opt/sutazaiapp/config/backend.conf

# Log access patterns
docker-compose logs backend | grep -E "(POST|GET|PUT|DELETE)" | tail -20

# Monitor data access
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
SELECT query, calls, total_time 
FROM pg_stat_statements 
WHERE query LIKE '%SELECT%' 
ORDER BY calls DESC 
LIMIT 10;"
```

#### Exercise 28: Documentation Standards (5 min)
**Operational Documentation Requirements**:
- **System Architecture**: Current state documentation
- **Configuration Management**: All settings documented
- **Security Procedures**: Access control and audit procedures
- **Incident Response**: Escalation procedures and contacts
- **Change Management**: Approval and rollback procedures

---

## 📋 Session 7: Operational Procedures & Documentation (15 min)

### Standard Operating Procedures (10 min)

#### SOP Template Development
**Daily Operations Checklist**:
```markdown
# Daily JARVIS Operations Checklist

## Morning Health Check (5 minutes)
- [ ] Check system status dashboard
- [ ] Verify all services running
- [ ] Review overnight alerts
- [ ] Check disk space usage
- [ ] Validate backup completion

## Midday Performance Review (10 minutes)
- [ ] Monitor response times
- [ ] Check resource utilization
- [ ] Review error logs
- [ ] Validate user access
- [ ] Update system documentation

## Evening Maintenance (15 minutes)
- [ ] Run backup procedures
- [ ] Clear temporary files
- [ ] Review security logs
- [ ] Plan next day activities
- [ ] Update maintenance log
```

### Documentation Standards (5 min)

#### Critical Documentation Requirements
**Mandatory Documentation**:
1. **System Configuration**: All settings and parameters
2. **Operational Procedures**: Step-by-step instructions
3. **Troubleshooting Guides**: Known issues and solutions
4. **Emergency Procedures**: Contact information and escalation
5. **Change Log**: All modifications and their impact

**Documentation Locations**:
- `/opt/sutazaiapp/docs/runbooks/`: Operational procedures
- `/opt/sutazaiapp/docs/architecture/`: System design
- `/opt/sutazaiapp/IMPORTANT/`: Critical system information
- `/opt/sutazaiapp/config/`: Configuration documentation

---

## 🎓 Workshop Completion & Certification (30 min)

### Skills Assessment (15 min)

#### Practical Administrator Exam
**Hands-on Assessment Tasks**:
1. **System Health Check**: Demonstrate comprehensive system monitoring
2. **Service Recovery**: Simulate and recover from service failure
3. **Performance Analysis**: Identify and resolve performance bottleneck
4. **Backup/Restore**: Execute complete backup and recovery procedure
5. **Security Audit**: Conduct security assessment and implement improvement

#### Knowledge Verification
**Technical Questions**:
- How do you determine if system performance is degraded?
- What steps would you take if PostgreSQL becomes unresponsive?
- How would you implement a new monitoring alert?
- What is your approach to disaster recovery planning?
- How do you ensure system security and compliance?

### Certification and Next Steps (15 min)

#### Administrator Certification
**Perfect Jarvis System Administrator Certificate**:
- Demonstrates competency in system administration
- Validates troubleshooting and maintenance skills
- Certifies backup and recovery procedures knowledge
- Confirms security and compliance understanding

#### Continuing Education Path
**Advanced Administrator Training**:
- **Enterprise Deployment**: Multi-node clustering and scaling
- **Advanced Security**: Zero-trust architecture and compliance
- **Automation**: Infrastructure as code and CI/CD integration
- **Performance Engineering**: Advanced optimization and tuning

#### Professional Development
**Administrator Community**:
- Join administrator forums and knowledge sharing
- Contribute to documentation and best practices
- Mentor new administrators
- Participate in system improvement initiatives

---

## 📚 Resources and Reference Materials

### Technical References
- **System Architecture Documentation**: `/opt/sutazaiapp/IMPORTANT/`
- **Operational Runbooks**: `/opt/sutazaiapp/docs/runbooks/`
- **Configuration References**: `/opt/sutazaiapp/config/`
- **Monitoring Dashboards**: Grafana templates and configurations

### Emergency Contacts
- **System Owner**: [contact-info]
- **Development Team**: [contact-info]  
- **Security Team**: [contact-info]
- **Infrastructure Support**: [contact-info]

### Useful Commands Reference
```bash
# System Management
docker-compose ps                    # Service status
docker-compose logs -f [service]     # View logs
docker-compose restart [service]     # Restart service
docker system prune -f              # Clean up resources

# Health Checks
curl http://localhost:10010/health   # Backend health
curl http://localhost:10104/api/tags # Ollama status
docker stats --no-stream           # Resource usage

# Database Access
docker exec -it sutazai-postgres psql -U sutazai -d sutazai
docker exec -it sutazai-redis redis-cli

# Monitoring
http://localhost:10201              # Grafana dashboards
http://localhost:10200              # Prometheus metrics
http://localhost:10202              # Loki logs
```

---

**🎉 Congratulations on becoming a certified Perfect Jarvis System Administrator!**

You now have the comprehensive skills needed to maintain, optimize, and troubleshoot the Perfect Jarvis system. Use this knowledge to ensure reliable, secure, and high-performance AI assistant capabilities for your organization.

**Ready to maintain enterprise-grade AI infrastructure!**