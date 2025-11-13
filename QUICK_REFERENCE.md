# SutazaiApp Quick Reference Card

## 🚀 Deployment

### First Time Setup
```bash
git clone https://github.com/sutazai/sutazaiapp.git
cd sutazaiapp
./deploy-portainer.sh
```

### Using Portainer
1. Open http://localhost:9000
2. Create admin account
3. Select "Local" environment
4. Stacks → Add stack → Upload `portainer-stack.yml`

## 🔍 Health Checks

```bash
# Full system check
./scripts/health-check.sh

# Individual service checks
docker exec sutazai-postgres pg_isready -U jarvis
docker exec sutazai-redis redis-cli ping
curl http://localhost:10200/health              # Backend
curl http://localhost:11000/_stcore/health      # Frontend
curl http://localhost:11434/api/tags            # Ollama
```

## 📊 Service URLs

| Service | URL | Credentials |
|---------|-----|-------------|
| Portainer | http://localhost:9000 | Set on first login |
| JARVIS UI | http://localhost:11000 | None |
| API Docs | http://localhost:10200/docs | JWT required |
| Grafana | http://localhost:10201 | admin/sutazai_secure_2024 |
| RabbitMQ | http://localhost:10005 | sutazai/sutazai_secure_2024 |
| Neo4j | http://localhost:10002 | neo4j/sutazai_secure_2024 |
| Consul | http://localhost:10006 | None |

## 🛠️ Common Commands

### Start/Stop Services
```bash
# Start all
docker compose -f portainer-stack.yml up -d

# Stop all
docker compose -f portainer-stack.yml down

# Restart specific service
docker restart sutazai-backend

# View logs
docker logs -f sutazai-backend
```

### Ollama Operations
```bash
# Pull model
docker exec sutazai-ollama ollama pull tinyllama

# List models
docker exec sutazai-ollama ollama list

# Test model
docker exec sutazai-ollama ollama run tinyllama "Hello"
```

### Database Access
```bash
# PostgreSQL
docker exec -it sutazai-postgres psql -U jarvis -d jarvis_ai

# Redis
docker exec -it sutazai-redis redis-cli

# Neo4j (use browser: http://localhost:10002)
```

## 📈 Monitoring

### Container Stats
```bash
docker stats $(docker ps --filter "name=sutazai-" --format "{{.Names}}")
```

### View Metrics
- Prometheus: http://localhost:10202
- Grafana: http://localhost:10201

### Log Aggregation
```bash
# All services
docker compose -f portainer-stack.yml logs -f

# Specific service
docker logs sutazai-backend --tail 100 -f

# Search logs
docker logs sutazai-backend 2>&1 | grep ERROR
```

## 🔧 Troubleshooting

### Service Won't Start
```bash
# Check logs
docker logs sutazai-[service-name]

# Check resource usage
docker stats

# Restart service
docker restart sutazai-[service-name]

# Rebuild if needed
docker compose -f portainer-stack.yml up -d --build sutazai-[service-name]
```

### Port Conflicts
```bash
# Find what's using port
sudo lsof -i :PORT
sudo netstat -tulpn | grep :PORT

# Kill process
sudo kill -9 PID
```

### Reset Everything
```bash
# WARNING: Deletes all data
docker compose -f portainer-stack.yml down -v
docker compose -f portainer-stack.yml up -d
```

## 💾 Backup & Restore

### Backup Volumes
```bash
# Backup all volumes
for vol in $(docker volume ls --filter name=sutazaiapp_ -q); do
  docker run --rm \
    -v $vol:/data:ro \
    -v $(pwd)/backups:/backup \
    alpine tar czf /backup/$vol.tar.gz -C /data .
done
```

### Restore Volumes
```bash
# Restore single volume
docker run --rm \
  -v sutazaiapp_postgres_data:/data \
  -v $(pwd)/backups:/backup:ro \
  alpine sh -c "rm -rf /data/* && tar xzf /backup/sutazaiapp_postgres_data.tar.gz -C /data"
```

## 🔒 Security Checklist

Before production:
- [ ] Change all default passwords
- [ ] Generate new JWT secret
- [ ] Enable SSL/TLS
- [ ] Configure firewall
- [ ] Set up automated backups
- [ ] Enable audit logging
- [ ] Review security groups

## 📚 Documentation

- Full Guide: `docs/PORTAINER_DEPLOYMENT_GUIDE.md`
- Port Registry: `IMPORTANT/ports/PortRegistry.md`
- TODO List: `TODO.md`
- Changelog: `CHANGELOG.md`
- Wiki: https://deepwiki.com/sutazai/sutazaiapp

## 🆘 Quick Help

```bash
# View service status
docker ps --filter "name=sutazai-"

# View service IPs
docker network inspect sutazaiapp_sutazai-network

# View volumes
docker volume ls --filter name=sutazaiapp_

# Check system resources
free -h
df -h
nproc
```

---
**Version**: 1.0.0  
**Updated**: 2025-11-13 21:40:00 UTC
