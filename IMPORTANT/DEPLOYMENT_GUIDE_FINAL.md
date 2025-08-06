# Production Deployment Guide

> **📋 Complete Technology Stack**: See `TECHNOLOGY_STACK_REPOSITORY_INDEX.md` for comprehensive technology inventory and verified components.

## Prerequisites
- Docker 24.0+
- Docker Compose 2.0+
- 4GB RAM minimum
- 20GB disk space

## Deployment Steps

### 1. Environment Setup
```bash
# Create .env file
cat > .env << EOF
POSTGRES_PASSWORD=secure_password_here
JWT_SECRET=your_jwt_secret_here
REDIS_PASSWORD=redis_password_here
ENVIRONMENT=production
EOF

# Set permissions
chmod 600 .env
```

### 2. Build Images
```bash
# Build all services
docker-compose -f docker-compose.clean.yml build

# Pull base images
docker pull postgres:15-alpine
docker pull redis:7-alpine
docker pull ollama/ollama:latest
```

### 3. Initialize Database
```bash
# Start database only
docker-compose -f docker-compose.clean.yml up -d postgres

# Wait for postgres
sleep 10

# Run migrations
docker-compose -f docker-compose.clean.yml run --rm backend python -m app.database.migrate
```

### 4. Deploy Ollama Model (VERIFIED WORKING)
```bash
# Start Ollama
docker-compose up -d ollama

# TinyLlama model is currently loaded (VERIFIED)
# To verify current model:
docker exec sutazai-ollama ollama list
# Should show: tinyllama:latest
# Note: GPT-OSS mentioned for production but not currently loaded
```

### 5. Start All Services (VERIFIED 26 CONTAINERS)
```bash
# Start everything using main compose file
docker-compose up -d

# Check status (should show 26+ containers)
docker-compose ps

# Verify service mesh is running
curl -f http://localhost:10006/v1/status/leader  # Consul
curl -f http://localhost:10005/                 # Kong
curl -f http://localhost:10008/                 # RabbitMQ Management UI

# View logs for troubleshooting
docker-compose logs -f --tail=50
```

### 6. Health Verification (ACTUAL WORKING PORTS)
```bash
# Check backend health (VERIFIED WORKING)
curl http://localhost:10010/health

# Check frontend (VERIFIED WORKING) 
curl http://localhost:10011

# Check Ollama (VERIFIED PORT)
curl http://localhost:10104/api/tags

# Check service mesh (VERIFIED WORKING)
curl http://localhost:10006/v1/status/leader  # Consul healthy
curl http://localhost:10005/                 # Kong gateway
curl http://localhost:10008/api/overview     # RabbitMQ management

# Check agent orchestration (VERIFIED WORKING)
curl http://localhost:8589/health            # AI Agent Orchestrator
curl http://localhost:8587/health            # Multi-Agent Coordinator
curl http://localhost:8588/health            # Resource Arbitration Agent
```

## Monitoring

### Prometheus Metrics
- URL: http://localhost:10200
- Metrics: CPU, Memory, Request rates

### Grafana Dashboards
- URL: http://localhost:10201
- Default login: admin/admin

### Log Aggregation
```bash
# View all logs
docker-compose logs

# View specific service
docker-compose logs backend

# Follow logs
docker-compose logs -f --tail 100
```

## Backup & Recovery

### Backup Database
```bash
docker exec sutazai-postgres pg_dump -U sutazai sutazai > backup.sql
```

### Restore Database
```bash
docker exec -i sutazai-postgres psql -U sutazai sutazai < backup.sql
```

## Scaling

### Horizontal Scaling
```yaml
# docker-compose.scale.yml
services:
  backend:
    deploy:
      replicas: 3
```

### Load Balancing
```nginx
upstream backend {
    server backend1:8000;
    server backend2:8000;
    server backend3:8000;
}
```

## Security Checklist
- [ ] Change all default passwords
- [ ] Enable SSL/TLS
- [ ] Configure firewall rules
- [ ] Set up regular backups
- [ ] Enable audit logging
- [ ] Implement rate limiting
- [ ] Regular security updates

## Troubleshooting

### Container Won't Start
```bash
docker-compose logs [service_name]
docker-compose restart [service_name]
```

### Database Connection Issues
```bash
docker exec -it sutazai-postgres psql -U sutazai
```

### Performance Issues
```bash
docker stats
docker-compose top
```
