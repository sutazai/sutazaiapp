# Portainer Quick Start Guide
**SutazAI Platform - Portainer Stack Management**

## 🚀 Quick Migration (3 Steps)

### Step 1: Run Migration Script
```bash
cd /opt/sutazaiapp
./migrate-to-portainer.sh
```

The script will:
- ✅ Verify prerequisites (Docker, Portainer, network, Ollama)
- ✅ Create backup of current state
- ✅ Stop existing docker-compose services
- ✅ Guide you through Portainer UI deployment
- ✅ Verify all containers are healthy
- ✅ Generate migration report

### Step 2: Deploy in Portainer UI
When prompted by the script:

1. **Open Portainer**: http://localhost:9000
2. **Create Admin Account** (if first time):
   - Username: `admin`
   - Password: Cristi45!!!!!!
3. **Navigate**: Home → Stacks → "Add stack"
4. **Configure**:
   - Name: `sutazai-platform`
   - Build method: "Upload"
   - File: Select `/opt/sutazaiapp/docker-compose-portainer.yml`
5. **Deploy**: Click "Deploy the stack"
6. **Return to terminal** and press Enter to continue verification

### Step 3: Verify Deployment
The script will automatically verify:
- ✅ All 11 containers running
- ✅ Health checks passing
- ✅ Service endpoints accessible

---

## 📊 Daily Operations

### View Stack Status
```bash
# Open Portainer
http://localhost:9000

# Navigate
Stacks → sutazai-platform → View details
```

### Start/Stop Individual Containers
```bash
# Via Portainer UI
Containers → Select container(s) → Start/Stop/Restart

# Or via CLI
sudo docker stop sutazai-backend
sudo docker start sutazai-backend
```

### View Logs
```bash
# Via Portainer UI
Containers → [container-name] → Logs → Auto-refresh

# Or via CLI
sudo docker logs -f sutazai-backend
```

### Update Stack Configuration
```bash
# 1. Edit the compose file
nano /opt/sutazaiapp/docker-compose-portainer.yml

# 2. Update in Portainer
Portainer → Stacks → sutazai-platform → Editor
→ Paste updated content or upload file
→ "Update the stack"

# 3. Verify changes
Stacks → sutazai-platform → View containers
```

### Scale Services
```bash
# Via Portainer UI (only for stateless services)
Stacks → sutazai-platform → Editor
→ Add "deploy.replicas: 3" under service
→ Update the stack
```

---

## 🔧 Common Tasks

### Restart Entire Stack
```bash
# Option 1: Via Portainer UI
Stacks → sutazai-platform → Stop → Start

# Option 2: Via CLI
cd /opt/sutazaiapp
sudo docker-compose -f docker-compose-portainer.yml restart
```

### View Resource Usage
```bash
# Via Portainer UI
Containers → sutazai-platform → Stats (real-time graphs)

# Or via CLI
sudo docker stats --filter "name=sutazai-"
```

### Backup Volumes
```bash
# Create backup directory
mkdir -p /opt/sutazaiapp/backups/volumes-$(date +%Y%m%d)

# Backup PostgreSQL
sudo docker exec sutazai-postgres pg_dump -U jarvis jarvis_ai > \
  /opt/sutazaiapp/backups/volumes-$(date +%Y%m%d)/postgres.sql

# Backup Neo4j
sudo docker exec sutazai-neo4j neo4j-admin database dump neo4j \
  --to-path=/backups 2>/dev/null || \
  sudo docker cp sutazai-neo4j:/data \
  /opt/sutazaiapp/backups/volumes-$(date +%Y%m%d)/neo4j-data

# Backup all volumes (safer method)
sudo docker run --rm \
  -v sutazaiapp_postgres_data:/source:ro \
  -v /opt/sutazaiapp/backups/volumes-$(date +%Y%m%d):/backup \
  alpine tar czf /backup/postgres_data.tar.gz -C /source .
```

### Restore Volumes
```bash
# Stop stack first
Portainer → Stacks → sutazai-platform → Stop

# Restore PostgreSQL data
sudo docker run --rm \
  -v sutazaiapp_postgres_data:/target \
  -v /opt/sutazaiapp/backups/volumes-YYYYMMDD:/backup \
  alpine tar xzf /backup/postgres_data.tar.gz -C /target

# Start stack
Portainer → Stacks → sutazai-platform → Start
```

### Monitor Health Checks
```bash
# Via Portainer UI
Containers → Filter by "unhealthy" status

# Or via CLI
sudo docker ps --filter "health=unhealthy"

# Detailed health info
sudo docker inspect sutazai-backend | jq '.[0].State.Health'
```

---

## 🔍 Troubleshooting

### Container Won't Start
```bash
# 1. Check logs
Portainer → Containers → [container] → Logs

# 2. Check dependency services
Portainer → Stacks → sutazai-platform → View all containers
→ Verify all dependencies are "healthy"

# 3. Recreate container
Portainer → Containers → [container] → Duplicate/Edit
→ Deploy

# Or via CLI
sudo docker-compose -f docker-compose-portainer.yml up -d --force-recreate [service-name]
```

### Health Check Failing
```bash
# 1. Check health command
sudo docker inspect sutazai-backend | jq '.[0].Config.Healthcheck'

# 2. Manually run health check
sudo docker exec sutazai-backend wget -q --spider http://localhost:8000/health

# 3. Increase health check intervals
Edit docker-compose-portainer.yml:
  healthcheck:
    start_period: 60s  # Increase from 45s
    interval: 30s      # Increase from 15s
```

### Network Issues
```bash
# 1. Verify network exists
sudo docker network inspect sutazaiapp_sutazai-network

# 2. Check container network connections
sudo docker inspect sutazai-backend | jq '.[0].NetworkSettings.Networks'

# 3. Reconnect container to network
sudo docker network connect sutazaiapp_sutazai-network sutazai-backend

# 4. Test connectivity between containers
sudo docker exec sutazai-backend ping -c 3 172.20.0.10
sudo docker exec sutazai-backend curl http://172.20.0.10:5432
```

### Ollama Not Accessible
```bash
# 1. Verify Ollama running on host
curl http://localhost:11434/api/version

# 2. Check host.docker.internal mapping
sudo docker exec sutazai-backend ping -c 3 host.docker.internal

# 3. Test from backend container
sudo docker exec sutazai-backend curl http://host.docker.internal:11434/api/version

# 4. If fails, use host IP instead
# Find host IP
ip addr show docker0 | grep 'inet ' | awk '{print $2}' | cut -d/ -f1

# Update docker-compose-portainer.yml
environment:
  OLLAMA_BASE_URL: http://172.17.0.1:11434  # Use actual host IP
```

### Port Conflicts
```bash
# 1. Check what's using the port
sudo netstat -tlnp | grep :10200

# 2. Stop conflicting service
sudo systemctl stop [service-name]

# 3. Or change port in docker-compose-portainer.yml
ports:
  - "10201:8000"  # Change host port
```

---

## 📈 Performance Monitoring

### Enable Prometheus/Grafana (Optional)
```bash
# Deploy monitoring stack
cd /opt/sutazaiapp
sudo docker-compose -f agents/docker-compose-phase9.yml up -d

# Access Grafana
http://localhost:10310
# Default: admin/admin

# Add Prometheus datasource
http://172.20.0.60:9090
```

### Resource Limits
Current limits in `docker-compose-portainer.yml`:
```yaml
deploy:
  resources:
    limits:
      memory: 1024M  # Adjust as needed
      cpus: '1.0'    # Adjust as needed
```

To modify:
```bash
# Edit docker-compose-portainer.yml
nano /opt/sutazaiapp/docker-compose-portainer.yml

# Update in Portainer
Stacks → sutazai-platform → Editor → Update
```

---

## 🔐 Security Hardening

### Change Default Credentials
```bash
# 1. PostgreSQL
sudo docker exec -it sutazai-postgres psql -U jarvis -d jarvis_ai
ALTER USER jarvis WITH PASSWORD 'new_secure_password';

# Update in docker-compose-portainer.yml
environment:
  POSTGRES_PASSWORD: new_secure_password

# 2. Neo4j
sudo docker exec -it sutazai-neo4j cypher-shell -u neo4j -p sutazai2024
CALL dbms.security.changePassword('new_password');

# Update in docker-compose-portainer.yml
environment:
  NEO4J_AUTH: neo4j/new_password

# 3. Update stack
Portainer → Stacks → sutazai-platform → Editor → Update
```

### Enable TLS/SSL
```bash
# Generate certificates
mkdir -p /opt/sutazaiapp/certs
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
  -keyout /opt/sutazaiapp/certs/key.pem \
  -out /opt/sutazaiapp/certs/cert.pem

# Update docker-compose-portainer.yml
# Add volume mount and env vars for SSL
```

### Restrict Portainer Access
```bash
# Via Portainer UI
Settings → Users → Add user with limited permissions
Settings → Teams → Create team with specific access

# Enable HTTPS for Portainer
sudo docker stop portainer
sudo docker rm portainer
sudo docker run -d \
  -p 9443:9443 \
  --name portainer \
  --restart=always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  -v /opt/sutazaiapp/certs:/certs \
  portainer/portainer-ce:latest \
  --sslcert /certs/cert.pem \
  --sslkey /certs/key.pem

# Access via HTTPS
https://localhost:9443
```

---

## 📚 Additional Resources

### Portainer Documentation
- Official Docs: https://docs.portainer.io/
- Stack Management: https://docs.portainer.io/user/docker/stacks
- API Reference: https://docs.portainer.io/api/

### SutazAI Documentation
- Architecture: `/opt/sutazaiapp/docs/architecture/`
- Port Registry: `/opt/sutazaiapp/IMPORTANT/ports/PortRegistry.md`
- Validation Report: `/opt/sutazaiapp/PRODUCTION_VALIDATION_REPORT.md`
- Deployment Guide: `/opt/sutazaiapp/PORTAINER_DEPLOYMENT_GUIDE.md`

### Quick Commands Reference
```bash
# View all sutazai containers
sudo docker ps --filter "name=sutazai-"

# View stack via Portainer API (get API key from UI)
curl -H "X-API-Key: YOUR_KEY" http://localhost:9000/api/stacks

# Restart specific service
sudo docker restart sutazai-backend

# View real-time logs
sudo docker logs -f sutazai-backend

# Check health status
sudo docker ps --filter "name=sutazai-" --format "{{.Names}}: {{.Status}}"

# Run integration tests
bash /opt/sutazaiapp/tests/integration/test_integration.sh

# Run E2E tests
cd /opt/sutazaiapp/frontend && npx playwright test
```

---

## ✅ Post-Migration Checklist

After migration, verify:

- [ ] All 11 containers running: `sudo docker ps --filter "name=sutazai-"`
- [ ] Health checks passing: Check Portainer UI container status
- [ ] Frontend accessible: http://localhost:11000
- [ ] Backend API accessible: http://localhost:10200/docs
- [ ] Integration tests passing: `bash tests/integration/test_integration.sh`
- [ ] E2E tests passing: `cd frontend && npx playwright test`
- [ ] Portainer accessible: http://localhost:9000
- [ ] Stack visible in Portainer: Navigate to Stacks → sutazai-platform
- [ ] Volume backups configured
- [ ] Monitoring setup (optional)
- [ ] Security hardening applied
- [ ] Team access configured in Portainer

---

**Need Help?**
- Check migration report: `/opt/sutazaiapp/PORTAINER_MIGRATION_REPORT.md`
- View deployment guide: `/opt/sutazaiapp/PORTAINER_DEPLOYMENT_GUIDE.md`
- Review logs: Portainer UI → Containers → [service] → Logs
