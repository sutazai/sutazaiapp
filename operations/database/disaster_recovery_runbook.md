# SutazAI Database Disaster Recovery Runbook

**Document Version:** 1.0  
**Author:** DBA Administrator  
**Date:** 2025-08-09  
**Classification:** CONFIDENTIAL - FOR OPERATIONAL USE ONLY  

## Emergency Contacts

| Role | Contact | Phone | Backup |
|------|---------|-------|--------|
| Primary DBA | DBA Admin | +1-xxx-xxx-xxxx | dba@sutazai.local |
| Infrastructure Lead | DevOps Lead | +1-xxx-xxx-xxxx | devops@sutazai.local |
| System Administrator | SysAdmin | +1-xxx-xxx-xxxx | sysadmin@sutazai.local |

## Recovery Time/Point Objectives

- **RTO (Recovery Time Objective):** 30 minutes
- **RPO (Recovery Point Objective):** 1 hour
- **Business Impact:** Critical - AI agents cannot function without database

## Quick Reference Commands

### Emergency Status Check
```bash
# Container status
docker ps --filter name=sutazai-postgres

# Database connectivity
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT current_timestamp, version();"

# Backup verification
ls -la /opt/sutazaiapp/backups/database/
```

### Emergency Backup
```bash
# Immediate backup
/opt/sutazaiapp/scripts/database/backup_database.sh

# Manual backup
docker exec sutazai-postgres pg_dump -U sutazai -d sutazai > emergency_backup_$(date +%Y%m%d_%H%M%S).sql
```

## Disaster Scenarios & Recovery Procedures

### Scenario 1: Database Container Stopped/Crashed

**Symptoms:**
- Backend API returns database errors
- Container not running in `docker ps`
- Connection refused errors

**Recovery Steps:**

1. **Immediate Assessment** (2 minutes)
   ```bash
   # Check container status
   docker ps -a --filter name=sutazai-postgres
   
   # Check Docker daemon
   systemctl status docker
   
   # Check available disk space
   df -h
   ```

2. **Restart Attempt** (3 minutes)
   ```bash
   # Simple restart
   docker start sutazai-postgres
   
   # Verify startup
   docker logs sutazai-postgres --tail 50
   
   # Test connectivity
   docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT 1;"
   ```

3. **If Restart Fails** (5 minutes)
   ```bash
   # Remove corrupted container
   docker stop sutazai-postgres
   docker rm sutazai-postgres
   
   # Recreate from compose
   cd /opt/sutazaiapp
   docker-compose up -d postgres
   ```

**Expected Recovery Time:** 5-10 minutes  
**Data Loss Risk:** None (data in volumes)

### Scenario 2: Database Corruption

**Symptoms:**
- Database connection works but queries fail
- Integrity constraint errors
- Index corruption messages in logs

**Recovery Steps:**

1. **Immediate Isolation** (2 minutes)
   ```bash
   # Stop the database to prevent further corruption
   docker stop sutazai-postgres
   
   # Create emergency backup if possible
   docker start sutazai-postgres
   docker exec sutazai-postgres pg_dump -U sutazai -d sutazai > corruption_emergency_backup.sql || true
   docker stop sutazai-postgres
   ```

2. **Database Recovery** (10-15 minutes)
   ```bash
   # Start in single-user mode for repair
   docker run -it --rm \
     --volumes-from sutazai-postgres \
     postgres:16.3-alpine \
     postgres --single -D /var/lib/postgresql/data sutazai
   
   # Run consistency checks
   # (In single-user mode)
   REINDEX DATABASE sutazai;
   ```

3. **If Repair Fails - Restore from Backup** (15-20 minutes)
   ```bash
   # Destroy corrupted database
   docker volume rm sutazai_postgres_data
   
   # Recreate container
   cd /opt/sutazaiapp
   docker-compose up -d postgres
   
   # Wait for initialization
   sleep 30
   
   # Restore from latest backup
   latest_backup=$(ls -t /opt/sutazaiapp/backups/database/sutazai_backup_*.sql.gz | head -1)
   gunzip -c "$latest_backup" | docker exec -i sutazai-postgres psql -U sutazai -d sutazai
   ```

**Expected Recovery Time:** 15-25 minutes  
**Data Loss Risk:** Up to 1 hour (based on backup frequency)

### Scenario 3: Complete Data Volume Loss

**Symptoms:**
- Container starts but database is empty
- Volume mount errors
- Storage hardware failure

**Recovery Steps:**

1. **Confirm Data Loss** (3 minutes)
   ```bash
   # Check volume status
   docker volume inspect sutazai_postgres_data
   
   # Check if database exists
   docker exec sutazai-postgres psql -U sutazai -l
   ```

2. **Full Database Restore** (20-30 minutes)
   ```bash
   # Ensure clean environment
   docker-compose down
   docker volume rm sutazai_postgres_data
   
   # Recreate infrastructure
   docker-compose up -d postgres
   
   # Wait for PostgreSQL to initialize
   sleep 60
   
   # Restore from latest backup
   latest_backup=$(ls -t /opt/sutazaiapp/backups/database/sutazai_backup_*.sql.gz | head -1)
   
   if [[ -n "$latest_backup" ]]; then
       echo "Restoring from: $latest_backup"
       gunzip -c "$latest_backup" | docker exec -i sutazai-postgres psql -U sutazai -d sutazai
   else
       echo "ERROR: No backup found!"
       exit 1
   fi
   
   # Verify restoration
   docker exec sutazai-postgres psql -U sutazai -d sutazai -c "
   SELECT 
       schemaname, 
       tablename,
       (SELECT count(*) FROM pg_stat_user_tables WHERE relname = tablename) as row_count
   FROM pg_tables 
   WHERE schemaname = 'public'
   ORDER BY tablename;"
   ```

3. **Restart Dependent Services** (5 minutes)
   ```bash
   # Restart backend to clear connection errors
   docker restart sutazai-backend
   
   # Restart agent services
   docker restart sutazai-hardware-resource-optimizer
   docker restart sutazai-jarvis-automation-agent
   
   # Verify system health
   curl http://localhost:10010/health
   ```

**Expected Recovery Time:** 25-35 minutes  
**Data Loss Risk:** Up to 1 hour (based on backup frequency)

### Scenario 4: Network/Connectivity Issues

**Symptoms:**
- Intermittent connection failures
- Timeout errors
- High response times

**Recovery Steps:**

1. **Network Diagnostics** (3 minutes)
   ```bash
   # Check Docker network
   docker network inspect sutazai-network
   
   # Test internal connectivity
   docker exec sutazai-backend ping sutazai-postgres
   
   # Check port accessibility
   telnet localhost 10000
   ```

2. **Service Restart** (2 minutes)
   ```bash
   # Restart database container
   docker restart sutazai-postgres
   
   # Restart backend
   docker restart sutazai-backend
   ```

3. **Network Reset** (if needed)
   ```bash
   # Recreate network
   docker network rm sutazai-network
   docker-compose up -d
   ```

**Expected Recovery Time:** 5-10 minutes  
**Data Loss Risk:** None

## Post-Recovery Verification Checklist

### Database Health Verification
- [ ] Container is running: `docker ps --filter name=sutazai-postgres`
- [ ] Database accepts connections: `docker exec sutazai-postgres psql -U sutazai -d sutazai -c "SELECT 1;"`
- [ ] All tables present: Check table count matches expected
- [ ] Data integrity: Run sample queries on key tables
- [ ] Recent data present: Check latest timestamps in tables

### Application Health Verification
- [ ] Backend API responding: `curl http://localhost:10010/health`
- [ ] Database status healthy in API response
- [ ] Agent services can query database
- [ ] New user registration works (if applicable)
- [ ] Authentication system functional

### System Monitoring Verification
- [ ] Prometheus scraping database metrics
- [ ] Grafana dashboards showing data
- [ ] Log aggregation working (Loki)
- [ ] Alerting system functional

## Backup Strategy

### Automated Backups
- **Frequency:** Every 4 hours
- **Retention:** 30 days local, 90 days archive
- **Script:** `/opt/sutazaiapp/scripts/database/backup_database.sh`
- **Schedule:** Cron job at 00:00, 06:00, 12:00, 18:00

### Manual Backup Process
```bash
# Full database backup
docker exec sutazai-postgres pg_dump -U sutazai -d sutazai --clean --no-owner > manual_backup.sql

# Schema-only backup
docker exec sutazai-postgres pg_dump -U sutazai -d sutazai --schema-only > schema_backup.sql

# Data-only backup
docker exec sutazai-postgres pg_dump -U sutazai -d sutazai --data-only > data_backup.sql
```

### Backup Verification
```bash
# Test backup integrity
/opt/sutazaiapp/scripts/database/backup_database.sh

# Manual verification
gunzip -t backup_file.sql.gz
```

## Connection Pool Configuration

### PgBouncer Configuration (Production)
```ini
# pgbouncer.ini
[databases]
sutazai = host=sutazai-postgres port=5432 dbname=sutazai

[pgbouncer]
listen_port = 6432
listen_addr = *
auth_type = md5
auth_file = /etc/pgbouncer/userlist.txt
admin_users = sutazai
pool_mode = transaction
max_client_conn = 100
default_pool_size = 20
max_db_connections = 50
```

### Application Connection Settings
```python
# Recommended connection pool settings
DATABASE_CONFIG = {
    'host': 'localhost',
    'port': 10000,
    'database': 'sutazai',
    'user': 'sutazai',
    'min_size': 5,
    'max_size': 20,
    'command_timeout': 60,
    'server_settings': {
        'application_name': 'sutazai_backend',
        'jit': 'off'
    }
}
```

## Monitoring Thresholds

### Critical Alerts (Immediate Response)
- Database unavailable for > 2 minutes
- Connection pool exhausted
- Disk space < 10% free
- Replication lag > 5 minutes

### Warning Alerts (Response within 30 minutes)
- Connection usage > 80%
- Slow queries > 5 seconds
- Disk space < 20% free
- Lock contention detected

### Informational (Daily review)
- Backup completion status
- Database size growth
- Query performance trends
- Connection usage patterns

## Recovery Testing Schedule

### Monthly Recovery Drills
- [ ] Test backup restoration to isolated environment
- [ ] Verify all recovery procedures
- [ ] Time recovery operations
- [ ] Update procedures based on findings

### Quarterly DR Exercises
- [ ] Full disaster simulation
- [ ] Cross-team coordination test
- [ ] Communication protocol verification
- [ ] Documentation updates

## Emergency Response Process

1. **Incident Detection** (0-2 minutes)
   - Automated alerts trigger
   - Manual discovery reported
   - Impact assessment initiated

2. **Initial Response** (2-5 minutes)
   - Incident commander assigned
   - Stakeholder notification sent
   - Recovery team assembled

3. **Recovery Execution** (5-30 minutes)
   - Execute appropriate recovery procedure
   - Monitor progress and adjust as needed
   - Document all actions taken

4. **Verification & Communication** (30-45 minutes)
   - Verify full system recovery
   - Notify stakeholders of resolution
   - Begin post-incident review

5. **Post-Incident Activities** (Within 24 hours)
   - Complete post-incident review
   - Document lessons learned
   - Update procedures if necessary
   - Schedule follow-up improvements

## Documentation Updates

This runbook should be reviewed and updated:
- After each incident
- Monthly during routine reviews
- When system architecture changes
- When backup/recovery procedures change

**Last Updated:** 2025-08-09  
**Next Review Date:** 2025-09-09  
**Version:** 1.0