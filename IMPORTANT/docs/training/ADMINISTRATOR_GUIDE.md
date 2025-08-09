# Perfect JARVIS Administrator Training Guide

## Table of Contents
1. [System Architecture Overview](#system-architecture-overview)
2. [Administration Dashboard](#administration-dashboard)
3. [User Management](#user-management)
4. [Configuration Management](#configuration-management)
5. [Monitoring & Alerting](#monitoring--alerting)
6. [Backup & Recovery](#backup--recovery)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)
9. [Hands-on Exercises](#hands-on-exercises)

## System Architecture Overview

### Core Components
The Perfect JARVIS system consists of 28 running containers organized into functional layers:

```
┌─────────────────────────────────────────────────────┐
│                   User Interface                      │
│         (Streamlit Frontend - Port 10011)            │
└─────────────────────────────────────────────────────┘
                           │
┌─────────────────────────────────────────────────────┐
│                    API Gateway                        │
│          (FastAPI Backend - Port 10010)              │
└─────────────────────────────────────────────────────┘
                           │
┌──────────────┬──────────────┬──────────────────────┐
│  PostgreSQL  │    Redis     │       Neo4j          │
│  Port 10000  │  Port 10001  │   Ports 10002/3      │
└──────────────┴──────────────┴──────────────────────┘
                           │
┌─────────────────────────────────────────────────────┐
│                  Ollama LLM Server                   │
│            (TinyLlama - Port 10104)                  │
└─────────────────────────────────────────────────────┘
```

### Service Categories
- **Core Services**: PostgreSQL, Redis, Neo4j, Ollama
- **Application Layer**: Backend API, Frontend UI
- **Vector Databases**: ChromaDB, Qdrant, FAISS
- **Monitoring Stack**: Prometheus, Grafana, Loki, AlertManager
- **Agent Services**: 7 Flask-based agent stubs

## Administration Dashboard

### Accessing Grafana
```bash
# Open Grafana dashboard
open http://localhost:10201

# Default credentials
Username: admin
Password: admin
```

### Key Dashboards
1. **JARVIS System Overview**: Overall system health and metrics
2. **Request Performance**: API latency and throughput
3. **Error Tracking**: Error rates and types
4. **Resource Usage**: CPU, memory, disk utilization

### Creating Custom Dashboards
```sql
-- Example Prometheus query for request rate
sum(rate(jarvis_requests_total[5m])) by (service)

-- Latency percentiles
histogram_quantile(0.95, sum(rate(jarvis_latency_seconds_bucket[5m])) by (le, service))
```

## User Management

### Managing Database Users
```bash
# Connect to PostgreSQL
docker exec -it sutazai-postgres psql -U sutazai -d sutazai

# Create new user
CREATE USER jarvis_admin WITH PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE sutazai TO jarvis_admin;

# List users
\du

# Revoke permissions
REVOKE ALL PRIVILEGES ON DATABASE sutazai FROM username;
```

### API Access Control
```python
# backend/app/api/auth.py - JWT token generation
from datetime import datetime, timedelta
from jose import jwt

def create_access_token(data: dict, expires_delta: timedelta = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm="HS256")
```

## Configuration Management

### Environment Variables
```bash
# View current configuration
docker exec sutazai-backend env | grep SUTAZAI

# Update configuration
vim /opt/sutazaiapp/.env

# Key variables
SUTAZAI_OLLAMA_HOST=sutazai-ollama
SUTAZAI_OLLAMA_PORT=11434
SUTAZAI_DEFAULT_MODEL=tinyllama
SUTAZAI_POSTGRES_HOST=sutazai-postgres
SUTAZAI_REDIS_HOST=sutazai-redis
```

### Docker Compose Management
```bash
# Update service configuration
docker-compose -f docker-compose.yml up -d --no-deps backend

# Scale services
docker-compose scale backend=3

# View service logs
docker-compose logs -f backend --tail=100
```

## Monitoring & Alerting

### Prometheus Configuration
```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'sutazai-backend'
    static_configs:
      - targets: ['sutazai-backend:8000']
    metrics_path: '/metrics'
    scrape_interval: 15s
```

### Setting Up Alerts
```yaml
# alert_rules.yml
groups:
  - name: jarvis_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(jarvis_errors_total[5m]) > 0.05
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ $value }} errors per second"
```

### Log Analysis
```bash
# View backend logs
docker logs sutazai-backend --tail=100 --follow

# Search logs with Loki
curl -G -s "http://localhost:10202/loki/api/v1/query_range" \
  --data-urlencode 'query={container="sutazai-backend"} |= "ERROR"'

# Export logs for analysis
docker logs sutazai-backend > backend_logs.txt 2>&1
```

## Backup & Recovery

### Database Backup
```bash
#!/bin/bash
# backup-databases.sh

# PostgreSQL backup
docker exec sutazai-postgres pg_dump -U sutazai sutazai > \
  /backups/postgres_$(date +%Y%m%d_%H%M%S).sql

# Redis backup
docker exec sutazai-redis redis-cli BGSAVE
docker cp sutazai-redis:/data/dump.rdb /backups/redis_$(date +%Y%m%d_%H%M%S).rdb

# Neo4j backup
docker exec sutazai-neo4j neo4j-admin backup \
  --database=neo4j --to=/backups/neo4j_$(date +%Y%m%d_%H%M%S)
```

### Recovery Procedures
```bash
# Restore PostgreSQL
docker exec -i sutazai-postgres psql -U sutazai sutazai < backup.sql

# Restore Redis
docker cp backup.rdb sutazai-redis:/data/dump.rdb
docker restart sutazai-redis

# Restore Neo4j
docker exec sutazai-neo4j neo4j-admin restore \
  --database=neo4j --from=/backups/neo4j_backup
```

## Performance Optimization

### Container Resource Limits
```yaml
# docker-compose.yml
services:
  backend:
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```

### Database Optimization
```sql
-- PostgreSQL query optimization
EXPLAIN ANALYZE SELECT * FROM conversations WHERE user_id = 1;

-- Create indexes
CREATE INDEX idx_conversations_user_id ON conversations(user_id);
CREATE INDEX idx_messages_timestamp ON messages(timestamp);

-- Vacuum and analyze
VACUUM ANALYZE conversations;
```

### Ollama Model Optimization
```bash
# Reduce memory usage
docker exec sutazai-ollama ollama run tinyllama --num-gpu 0 --num-thread 4

# Monitor model performance
curl http://localhost:10104/api/tags | jq '.models[] | {name, size}'
```

## Troubleshooting

### Common Issues and Solutions

#### 1. Backend Shows "Degraded" Status
```bash
# Check Ollama connection
curl http://localhost:10104/api/tags

# Verify model is loaded
docker exec sutazai-ollama ollama list

# Restart backend
docker-compose restart backend
```

#### 2. High Memory Usage
```bash
# Check container stats
docker stats --no-stream

# Identify memory-heavy containers
docker ps --format "table {{.Names}}\t{{.Status}}" | grep -E "(Restarting|OOMKilled)"

# Apply memory limits
docker update --memory="1g" --memory-swap="1g" container_name
```

#### 3. Database Connection Issues
```bash
# Test PostgreSQL connection
docker exec sutazai-backend python -c "
from sqlalchemy import create_engine
engine = create_engine('postgresql://sutazai:password@sutazai-postgres:5432/sutazai')
print('Connection successful:', engine.connect())
"

# Check Redis connectivity
docker exec sutazai-backend redis-cli -h sutazai-redis ping
```

## Hands-on Exercises

### Exercise 1: Deploy Configuration Change
1. Update the Ollama model from `tinyllama` to another model
2. Restart necessary services
3. Verify the change through the API

**Solution:**
```bash
# 1. Update configuration
echo "SUTAZAI_DEFAULT_MODEL=llama2" >> .env

# 2. Restart backend
docker-compose restart backend

# 3. Verify
curl http://localhost:10010/api/v1/models
```

### Exercise 2: Create Monitoring Alert
1. Create an alert for high latency (>2 seconds)
2. Configure email notification
3. Test the alert

**Solution:**
```yaml
# Add to prometheus/alert_rules.yml
- alert: HighLatency
  expr: histogram_quantile(0.95, rate(jarvis_latency_seconds_bucket[5m])) > 2
  for: 5m
  labels:
    severity: critical
```

### Exercise 3: Perform Backup and Recovery
1. Create a full system backup
2. Simulate data loss
3. Restore from backup
4. Verify data integrity

**Solution:**
```bash
# 1. Backup
./scripts/backup-all.sh

# 2. Simulate loss
docker exec sutazai-postgres psql -U sutazai -c "DROP TABLE conversations;"

# 3. Restore
./scripts/restore-all.sh

# 4. Verify
docker exec sutazai-postgres psql -U sutazai -c "SELECT COUNT(*) FROM conversations;"
```

### Exercise 4: Performance Tuning
1. Identify slow queries
2. Add appropriate indexes
3. Measure performance improvement

**Solution:**
```sql
-- Enable query logging
ALTER SYSTEM SET log_statement = 'all';
SELECT pg_reload_conf();

-- Analyze slow queries
SELECT query, mean_exec_time, calls 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;

-- Add index
CREATE INDEX CONCURRENTLY idx_messages_user_session 
ON messages(user_id, session_id);
```

## Best Practices

### Security
- Rotate passwords quarterly
- Use environment-specific secrets
- Enable SSL/TLS for all services
- Implement rate limiting
- Regular security scans

### Monitoring
- Set up proactive alerts
- Review logs daily
- Track key metrics trends
- Maintain runbooks updated
- Document incidents

### Maintenance
- Schedule regular maintenance windows
- Test backups monthly
- Keep dependencies updated
- Monitor disk usage
- Clean up old logs

## Quick Reference Commands

```bash
# Service Management
docker-compose ps                     # List services
docker-compose logs -f service_name   # View logs
docker-compose restart service_name   # Restart service
docker-compose exec service_name bash # Enter container

# Health Checks
curl http://localhost:10010/health    # Backend health
curl http://localhost:10104/api/tags  # Ollama models
docker exec sutazai-postgres pg_isready # PostgreSQL status

# Performance
docker stats --no-stream              # Resource usage
docker system df                      # Disk usage
docker system prune -a                # Clean unused resources

# Debugging
docker inspect container_name         # Container details
docker logs container_name --tail=50  # Recent logs
docker exec container_name env        # Environment variables
```

## Support Resources

- **Documentation**: `/opt/sutazaiapp/docs/`
- **Runbooks**: `/opt/sutazaiapp/docs/runbooks/`
- **API Reference**: `/opt/sutazaiapp/docs/api/`
- **Monitoring**: http://localhost:10201 (Grafana)
- **Metrics**: http://localhost:10200 (Prometheus)

---

*This guide is based on the actual running Perfect JARVIS system with 28 containers and real TinyLlama LLM via Ollama.*