# SutazaiApp Portainer Stack Deployment Guide
## Complete System Management via Portainer

**Version**: 1.0.0  
**Created**: 2025-11-13 21:30:00 UTC  
**Last Updated**: 2025-11-13 21:30:00 UTC  
**Author**: SutazaiApp Development Team  

---

## Overview

This guide provides comprehensive instructions for deploying and managing the entire SutazaiApp platform using Portainer. The unified stack approach simplifies deployment, monitoring, and management of all services through a single interface.

### What is Included

The Portainer stack (`portainer-stack.yml`) consolidates all SutazaiApp services:

- **Container Management**: Portainer CE for web-based Docker management
- **Core Infrastructure**: PostgreSQL, Redis, Neo4j, RabbitMQ, Consul
- **API Gateway**: Kong with automatic database migrations
- **Vector Databases**: ChromaDB, Qdrant, FAISS
- **AI Services**: Ollama for local LLM inference
- **Application**: Backend API (FastAPI) and Frontend UI (Streamlit/JARVIS)
- **Monitoring**: Prometheus and Grafana for observability

---

## Prerequisites

### System Requirements

**Minimum Requirements:**
- **CPU**: 8 cores
- **RAM**: 16GB
- **Disk**: 100GB free space
- **OS**: Linux (Ubuntu 20.04+, Debian 11+, CentOS 8+)
- **Docker**: 20.10+ with Docker Compose v2
- **Network**: Internet access for pulling images

**Recommended Requirements:**
- **CPU**: 16+ cores
- **RAM**: 32GB+
- **Disk**: 500GB SSD
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional, for enhanced AI performance)

### Required Software

1. **Docker Engine** (v20.10+)
   ```bash
   curl -fsSL https://get.docker.com | sh
   sudo usermod -aG docker $USER
   ```

2. **Docker Compose** (v2.0+)
   ```bash
   # Usually included with Docker Engine
   docker compose version
   ```

3. **Git** (for cloning repository)
   ```bash
   sudo apt-get install git -y  # Ubuntu/Debian
   sudo yum install git -y      # CentOS/RHEL
   ```

---

## Installation Methods

### Method 1: Portainer Web Interface (Recommended)

#### Step 1: Install Portainer

```bash
# Create Portainer volume
docker volume create portainer_data

# Deploy Portainer
docker run -d \
  -p 9000:9000 \
  -p 9443:9443 \
  --name portainer \
  --restart=always \
  -v /var/run/docker.sock:/var/run/docker.sock \
  -v portainer_data:/data \
  portainer/portainer-ce:latest
```

#### Step 2: Access Portainer

1. Open browser and navigate to: `http://localhost:9000` or `https://localhost:9443`
2. Create admin account on first access
3. Select "Local" environment
4. Click "Connect"

#### Step 3: Deploy Stack

1. Navigate to **Stacks** in the left sidebar
2. Click **+ Add stack**
3. Enter stack name: `sutazaiapp`
4. Choose deployment method:

   **Option A: Git Repository (Recommended)**
   - Select "Repository" tab
   - Repository URL: `https://github.com/sutazai/sutazaiapp`
   - Repository reference: `copilot/fix-issues-and-improve-performance`
   - Compose path: `portainer-stack.yml`
   - Click **Deploy the stack**

   **Option B: Web Editor**
   - Select "Web editor" tab
   - Copy contents of `portainer-stack.yml`
   - Paste into editor
   - Click **Deploy the stack**

   **Option C: Upload**
   - Select "Upload" tab
   - Choose `portainer-stack.yml` file
   - Click **Deploy the stack**

#### Step 4: Monitor Deployment

1. Stack deployment begins automatically
2. Monitor progress in the **Containers** section
3. Check container health status
4. Wait for all services to reach "healthy" status (5-10 minutes)

---

### Method 2: Command Line Deployment

#### Step 1: Clone Repository

```bash
# Clone the repository
git clone https://github.com/sutazai/sutazaiapp.git
cd sutazaiapp

# Checkout correct branch
git checkout copilot/fix-issues-and-improve-performance
```

#### Step 2: Deploy Stack

```bash
# Deploy the complete stack
docker compose -f portainer-stack.yml up -d

# Monitor logs
docker compose -f portainer-stack.yml logs -f

# Check service status
docker compose -f portainer-stack.yml ps
```

#### Step 3: Access Portainer

```bash
# Portainer should be accessible at:
# http://localhost:9000 (HTTP)
# https://localhost:9443 (HTTPS)

# Create admin account on first access
```

---

## Post-Deployment Configuration

### 1. Initialize Ollama Models

```bash
# Access Ollama container
docker exec -it sutazai-ollama sh

# Pull TinyLlama model (lightweight, 637MB)
ollama pull tinyllama

# Optional: Pull larger models if you have resources
# ollama pull llama2
# ollama pull mistral
# ollama pull codellama

# Verify models
ollama list

# Exit container
exit
```

### 2. Create Kong API Routes (Optional)

Access Kong Admin API to configure routes:

```bash
# Example: Register backend service
curl -X POST http://localhost:10009/services \
  -d name=backend-api \
  -d url='http://sutazai-backend:8000'

# Create route for backend
curl -X POST http://localhost:10009/services/backend-api/routes \
  -d name=backend-route \
  -d 'paths[]=/api'
```

### 3. Configure Grafana Dashboards

1. Access Grafana: `http://localhost:10201`
2. Login with credentials:
   - Username: `admin`
   - Password: `sutazai_secure_2024`
3. Dashboards are auto-provisioned from Prometheus datasource

---

## Service Access URLs

After successful deployment, access services via:

| Service | URL | Credentials |
|---------|-----|-------------|
| **Portainer** | http://localhost:9000 | Create on first access |
| **JARVIS Frontend** | http://localhost:11000 | None required |
| **Backend API** | http://localhost:10200 | JWT auth required |
| **API Docs** | http://localhost:10200/docs | Interactive Swagger UI |
| **Kong Proxy** | http://localhost:10008 | Configured routes |
| **Kong Admin** | http://localhost:10009 | None (internal) |
| **RabbitMQ Management** | http://localhost:10005 | sutazai/sutazai_secure_2024 |
| **Neo4j Browser** | http://localhost:10002 | neo4j/sutazai_secure_2024 |
| **Consul UI** | http://localhost:10006 | None required |
| **Prometheus** | http://localhost:10200 | None required |
| **Grafana** | http://localhost:10201 | admin/sutazai_secure_2024 |
| **Ollama API** | http://localhost:11434 | None required |

---

## Health Checks

### Quick Health Check Script

```bash
#!/bin/bash
# health-check.sh - Verify all services are healthy

echo "SutazaiApp Health Check"
echo "======================="

services=(
  "sutazai-portainer:9000"
  "sutazai-postgres:10000"
  "sutazai-redis:10001"
  "sutazai-neo4j:10002"
  "sutazai-rabbitmq:10005"
  "sutazai-consul:10006"
  "sutazai-kong:10008"
  "sutazai-chromadb:10100"
  "sutazai-qdrant:10102"
  "sutazai-faiss:10103"
  "sutazai-backend:10200"
  "sutazai-frontend:11000"
  "sutazai-ollama:11434"
  "sutazai-prometheus:10200"
  "sutazai-grafana:10201"
)

for service in "${services[@]}"; do
  name="${service%%:*}"
  port="${service##*:}"
  
  if docker ps --filter "name=$name" --filter "status=running" | grep -q "$name"; then
    echo "✓ $name is running"
  else
    echo "✗ $name is NOT running"
  fi
done

echo ""
echo "Container Status:"
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}" | grep sutazai
```

### Manual Health Checks

```bash
# Check PostgreSQL
docker exec sutazai-postgres pg_isready -U jarvis -d jarvis_ai

# Check Redis
docker exec sutazai-redis redis-cli ping

# Check RabbitMQ
docker exec sutazai-rabbitmq rabbitmq-diagnostics ping

# Check Neo4j
curl -f http://localhost:10002

# Check Backend API
curl -f http://localhost:10200/health

# Check Frontend
curl -f http://localhost:11000/_stcore/health

# Check Ollama
curl -f http://localhost:11434/api/tags
```

---

## Troubleshooting

### Common Issues

#### 1. Port Conflicts

**Problem**: Port already in use  
**Solution**:
```bash
# Find process using port
sudo lsof -i :9000
# Or
sudo netstat -tulpn | grep :9000

# Stop conflicting service or change port in stack file
```

#### 2. Memory Issues

**Problem**: Services restarting due to OOM  
**Solution**:
```bash
# Check system memory
free -h

# Reduce resource limits in portainer-stack.yml
# Adjust deploy.resources.limits.memory for services

# Restart stack
docker compose -f portainer-stack.yml restart
```

#### 3. Network Issues

**Problem**: Services can't communicate  
**Solution**:
```bash
# Check network exists
docker network ls | grep sutazai

# Inspect network
docker network inspect sutazaiapp_sutazai-network

# Restart network stack
docker compose -f portainer-stack.yml down
docker compose -f portainer-stack.yml up -d
```

#### 4. Ollama Models Not Loading

**Problem**: Ollama can't find models  
**Solution**:
```bash
# Access container
docker exec -it sutazai-ollama sh

# Check models directory
ls -la /root/.ollama/models

# Pull model again
ollama pull tinyllama

# Verify
ollama list
```

#### 5. Build Failures

**Problem**: Custom images fail to build  
**Solution**:
```bash
# Build images manually first
cd backend
docker build -t sutazai/backend:latest .

cd ../services/faiss
docker build -t sutazai/faiss-service:latest .

cd ../frontend
docker build -t sutazai/frontend:latest .

# Then deploy stack
cd ../..
docker compose -f portainer-stack.yml up -d
```

---

## Updating the Stack

### Update via Portainer

1. Navigate to **Stacks** → **sutazaiapp**
2. Click **Editor**
3. Make changes to stack configuration
4. Click **Update the stack**
5. Select "Re-pull images and redeploy"
6. Click **Update**

### Update via Command Line

```bash
# Pull latest changes
git pull origin copilot/fix-issues-and-improve-performance

# Pull latest images
docker compose -f portainer-stack.yml pull

# Restart stack
docker compose -f portainer-stack.yml up -d

# Remove old images
docker image prune -f
```

---

## Backup and Restore

### Backup

```bash
#!/bin/bash
# backup.sh - Backup all volumes

BACKUP_DIR="/backup/sutazaiapp/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Stop services
docker compose -f portainer-stack.yml stop

# Backup volumes
for volume in $(docker volume ls --format "{{.Name}}" | grep sutazaiapp); do
  echo "Backing up $volume..."
  docker run --rm \
    -v "$volume":/source:ro \
    -v "$BACKUP_DIR":/backup \
    alpine tar czf "/backup/${volume}.tar.gz" -C /source .
done

# Start services
docker compose -f portainer-stack.yml start

echo "Backup completed: $BACKUP_DIR"
```

### Restore

```bash
#!/bin/bash
# restore.sh - Restore volumes from backup

BACKUP_DIR="$1"

if [ -z "$BACKUP_DIR" ]; then
  echo "Usage: $0 <backup_directory>"
  exit 1
fi

# Stop services
docker compose -f portainer-stack.yml stop

# Restore volumes
for archive in "$BACKUP_DIR"/*.tar.gz; do
  volume=$(basename "$archive" .tar.gz)
  echo "Restoring $volume..."
  
  docker run --rm \
    -v "$volume":/target \
    -v "$BACKUP_DIR":/backup:ro \
    alpine sh -c "rm -rf /target/* && tar xzf /backup/${volume}.tar.gz -C /target"
done

# Start services
docker compose -f portainer-stack.yml start

echo "Restore completed"
```

---

## Performance Optimization

### Resource Tuning

Adjust resource limits based on your hardware in `portainer-stack.yml`:

```yaml
deploy:
  resources:
    limits:
      memory: 2G      # Increase for heavy workloads
      cpus: '2.0'     # Increase for better performance
    reservations:
      memory: 512M    # Minimum guaranteed
      cpus: '0.5'
```

### Database Optimization

**PostgreSQL:**
```sql
-- Access PostgreSQL
docker exec -it sutazai-postgres psql -U jarvis -d jarvis_ai

-- Check database size
SELECT pg_size_pretty(pg_database_size('jarvis_ai'));

-- Vacuum and analyze
VACUUM ANALYZE;
```

**Redis:**
```bash
# Monitor Redis memory
docker exec sutazai-redis redis-cli INFO memory

# Flush if needed (WARNING: deletes all data)
# docker exec sutazai-redis redis-cli FLUSHALL
```

---

## Security Considerations

### Change Default Passwords

**Important**: Change all default passwords before production deployment!

Edit in `portainer-stack.yml` or set via environment variables:

```yaml
environment:
  POSTGRES_PASSWORD: "YOUR_SECURE_PASSWORD"
  NEO4J_PASSWORD: "YOUR_SECURE_PASSWORD"
  RABBITMQ_DEFAULT_PASS: "YOUR_SECURE_PASSWORD"
  GF_SECURITY_ADMIN_PASSWORD: "YOUR_SECURE_PASSWORD"
  CHROMA_SERVER_AUTH_CREDENTIALS: "YOUR_SECURE_TOKEN"
```

### Enable SSL/TLS

For production, enable HTTPS:

1. Obtain SSL certificates (Let's Encrypt recommended)
2. Configure Kong or use reverse proxy (nginx/Traefik)
3. Update port mappings to use HTTPS

### Firewall Configuration

```bash
# Allow only necessary ports
sudo ufw allow 9000/tcp   # Portainer
sudo ufw allow 11000/tcp  # Frontend
sudo ufw allow 10200/tcp  # Backend API
sudo ufw enable
```

---

## Monitoring and Alerts

### Grafana Dashboards

Pre-configured dashboards monitor:
- Container resource usage
- Service health status
- API request rates
- Database performance
- LLM inference latency

### Prometheus Alerts

Configure alerting rules in `monitoring/prometheus.yml`:

```yaml
rule_files:
  - "/etc/prometheus/alerts/*.yml"
```

---

## Support and Documentation

### Additional Resources

- **Project Repository**: https://github.com/sutazai/sutazaiapp
- **Documentation Wiki**: https://deepwiki.com/sutazai/sutazaiapp
- **Port Registry**: `/opt/sutazaiapp/IMPORTANT/ports/PortRegistry.md`
- **TODO List**: `/opt/sutazaiapp/TODO.md`
- **Changelog**: `/opt/sutazaiapp/CHANGELOG.md`

### Getting Help

For issues or questions:
1. Check this guide's troubleshooting section
2. Review container logs: `docker logs <container-name>`
3. Check Portainer logs in the web interface
4. Consult project documentation and TODO.md

---

## License

This deployment configuration is part of the SutazaiApp project.  
© 2025 SutazaiApp Development Team
