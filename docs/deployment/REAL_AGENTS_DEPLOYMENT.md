# SutazAI Real Agents Deployment Guide

## 🎯 Overview

This guide provides complete automation for deploying the SutazAI AGI/ASI system with **real AI agents only**. No mock implementations - everything is functional and production-ready.

## 🚀 Quick Start (Automated)

```bash
# One-command deployment of entire system with real agents
./deploy_automated_sutazai_system.sh
```

**That's it!** The script will automatically:
- Deploy all real AI agents
- Set up all models (DeepSeek R1, Qwen3, Llama 3.2)
- Configure infrastructure (PostgreSQL, Redis, Qdrant, ChromaDB)
- Start monitoring and health checks
- Verify everything is working

## 🏗️ System Architecture

### Real AI Agents Deployed
- **OpenWebUI** - Advanced chat interface with real model integration
- **TabbyML** - Real code completion and suggestions
- **LangFlow** - Visual workflow orchestration
- **Dify** - App development platform
- **Browserless Chrome** - Real web automation
- **Enhanced Agent Orchestrator** - Coordinates all agents

### AI Models
- **DeepSeek R1 8B** - Advanced reasoning and problem-solving
- **Qwen3 8B** - Multilingual capabilities
- **Llama 3.2 1B** - Fast general-purpose model

### Infrastructure
- **PostgreSQL** - Primary database
- **Redis** - Caching and message queuing
- **Qdrant** - Vector database for embeddings
- **ChromaDB** - Vector memory store
- **Prometheus/Grafana** - Monitoring

## 📋 Prerequisites

### System Requirements
- **RAM**: Minimum 8GB, Recommended 16GB+
- **Disk**: Minimum 50GB free space
- **CPU**: Multi-core recommended
- **GPU**: Optional (NVIDIA for acceleration)

### Software Requirements
- Docker 20.10+
- Docker Compose 2.0+
- Git
- curl
- Python 3.8+

### Installation
```bash
# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh

# Install Docker Compose
sudo apt-get install docker-compose-plugin

# Verify installation
docker --version
docker compose version
```

## 🛠️ Manual Deployment (Step by Step)

If you prefer manual control over the automated script:

### 1. Clone and Setup
```bash
git clone <your-repo>
cd sutazaiapp
```

### 2. Environment Configuration
```bash
# Copy and customize environment
cp .env.example .env
# Edit .env with your specific configurations
```

### 3. Deploy Infrastructure
```bash
# Start core services
docker-compose -f docker-compose-real-agents.yml up -d postgres redis qdrant chromadb ollama

# Wait for services to be ready
sleep 30
```

### 4. Deploy Real Agents
```bash
# Start all real AI agents
docker-compose -f docker-compose-real-agents.yml up -d tabby open-webui langflow dify-api browserless

# Wait for agents to initialize
sleep 45
```

### 5. Deploy Application
```bash
# Start backend and frontend
docker-compose -f docker-compose-real-agents.yml up -d sutazai-backend sutazai-streamlit

# Start monitoring
docker-compose -f docker-compose-real-agents.yml up -d prometheus grafana
```

### 6. Setup AI Models
```bash
# Pull real AI models
docker exec sutazai-ollama ollama pull llama3.2:1b
docker exec sutazai-ollama ollama pull tinyllama
docker exec sutazai-ollama ollama pull qwen3:8b
```

## 🔍 Verification and Testing

### Health Check Commands
```bash
# Check all container status
docker-compose -f docker-compose-real-agents.yml ps

# Test backend API
curl http://localhost:8000/health

# Test agent status
curl http://localhost:8000/api/external_agents/status

# Test models
curl http://localhost:11434/api/tags
```

### Access Points
- **Main UI**: http://localhost:8501
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs
- **OpenWebUI**: http://localhost:8089
- **TabbyML**: http://localhost:8081
- **LangFlow**: http://localhost:7860
- **Dify**: http://localhost:5001
- **Monitoring**: http://localhost:3000

## 🔧 Management Commands

### Daily Operations
```bash
# Start entire system
docker-compose -f docker-compose-real-agents.yml up -d

# Stop entire system
docker-compose -f docker-compose-real-agents.yml down

# Restart specific service
docker-compose -f docker-compose-real-agents.yml restart sutazai-backend

# View logs
docker-compose -f docker-compose-real-agents.yml logs -f

# Scale specific service
docker-compose -f docker-compose-real-agents.yml up -d --scale tabby=2
```

### Maintenance
```bash
# Update all images
docker-compose -f docker-compose-real-agents.yml pull

# Rebuild specific service
docker-compose -f docker-compose-real-agents.yml build sutazai-backend

# Clean up unused resources
docker system prune -f

# Backup data volumes
docker run --rm -v sutazaiapp_postgres-data:/data -v $(pwd):/backup alpine tar czf /backup/postgres-backup.tar.gz /data
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Service Won't Start
```bash
# Check logs
docker-compose -f docker-compose-real-agents.yml logs <service-name>

# Restart service
docker-compose -f docker-compose-real-agents.yml restart <service-name>

# Rebuild if needed
docker-compose -f docker-compose-real-agents.yml build <service-name>
```

#### 2. Models Not Loading
```bash
# Check Ollama status
docker exec sutazai-ollama ollama list

# Re-pull models
docker exec sutazai-ollama ollama pull llama3.2:1b

# Check disk space
df -h
```

#### 3. Agent Communication Issues
```bash
# Check network
docker network ls
docker network inspect sutazaiapp_sutazai-network

# Test connectivity
docker exec sutazai-backend curl http://tabby:8080/health
```

#### 4. Performance Issues
```bash
# Check resource usage
docker stats

# Check system resources
free -h
df -h

# Scale down if needed
docker-compose -f docker-compose-real-agents.yml down
# Edit docker-compose-real-agents.yml to reduce resource limits
docker-compose -f docker-compose-real-agents.yml up -d
```

### Log Locations
- **Deployment logs**: `sutazai_deployment_*.log`
- **Container logs**: `docker-compose logs <service>`
- **Application logs**: `/opt/sutazaiapp/data/logs/`

## 🔄 Updates and Maintenance

### Regular Updates
```bash
# 1. Pull latest code
git pull origin main

# 2. Update images
docker-compose -f docker-compose-real-agents.yml pull

# 3. Recreate services
docker-compose -f docker-compose-real-agents.yml up -d --force-recreate

# 4. Update models
docker exec sutazai-ollama ollama pull llama3.2:1b
```

### Backup Strategy
```bash
# Create backup script
cat > backup.sh << 'EOF'
#!/bin/bash
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$BACKUP_DIR"

# Backup databases
docker exec sutazai-postgres pg_dump -U sutazai sutazai > "$BACKUP_DIR/postgres.sql"
docker exec sutazai-redis redis-cli BGSAVE
docker cp sutazai-redis:/data/dump.rdb "$BACKUP_DIR/"

# Backup vector databases
docker cp sutazai-qdrant:/qdrant/storage "$BACKUP_DIR/qdrant"
docker cp sutazai-chromadb:/chroma/chroma "$BACKUP_DIR/chromadb"

# Backup models
docker cp sutazai-ollama:/root/.ollama "$BACKUP_DIR/ollama"

echo "Backup completed: $BACKUP_DIR"
EOF

chmod +x backup.sh
```

## 🔐 Security Considerations

### Production Security
1. **Change default passwords** in `.env` file
2. **Enable SSL/TLS** for public deployments
3. **Configure firewall** to restrict access
4. **Use secrets management** for sensitive data
5. **Enable container security scanning**

### Network Security
```bash
# Create custom network with isolation
docker network create --driver bridge --internal sutazai-internal

# Use environment variables for secrets
# Never commit secrets to git
```

## 📊 Monitoring and Observability

### Built-in Monitoring
- **Prometheus**: Metrics collection (http://localhost:9090)
- **Grafana**: Dashboards and visualization (http://localhost:3000)
- **Health checks**: All services have automated health monitoring
- **Logging**: Centralized logging with rotation

### Custom Metrics
```bash
# Add custom metrics to Prometheus
# Edit monitoring/prometheus/prometheus.yml

# Create custom Grafana dashboards
# Import dashboards via UI at http://localhost:3000
```

## 🚀 Advanced Configuration

### Resource Limits
Edit `docker-compose-real-agents.yml` to adjust resource limits:

```yaml
services:
  service-name:
    deploy:
      resources:
        limits:
          memory: 2G
          cpus: '1.0'
        reservations:
          memory: 1G
          cpus: '0.5'
```

### Custom Agent Development
1. Create new agent in `agents/` directory
2. Add Dockerfile and configuration
3. Update `docker-compose-real-agents.yml`
4. Register with orchestrator in backend

## 📝 API Reference

### Key Endpoints
- `GET /health` - System health check
- `GET /api/agents/` - List all agents
- `POST /api/chat` - Chat with AI
- `GET /api/external_agents/status` - Agent status
- `POST /api/orchestrator/tasks` - Submit tasks

### Example API Usage
```bash
# Chat with AI
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "model": "llama3.2:1b"}'

# Get agent status
curl http://localhost:8000/api/external_agents/status | jq
```

## 🎯 Best Practices

### Development
- Use automated deployment script for consistency
- Test changes in staging environment first
- Monitor resource usage regularly
- Keep models updated
- Implement proper error handling

### Production
- Use container orchestration (Kubernetes) for scale
- Implement load balancing
- Set up automated backups
- Monitor and alert on failures
- Document custom configurations

## 📞 Support

### Getting Help
1. Check logs first: `docker-compose logs`
2. Review this documentation
3. Check GitHub issues
4. Run health checks: `./deploy_automated_sutazai_system.sh --verify`

### Contributing
1. Fork the repository
2. Create feature branch
3. Test with automated deployment
4. Submit pull request with documentation updates

---

## 🔄 Automation Summary

The `deploy_automated_sutazai_system.sh` script provides complete automation:

✅ **Automated Prerequisites Check**
✅ **Real Agent Container Creation**  
✅ **AI Model Deployment**
✅ **Infrastructure Setup**
✅ **Health Verification**
✅ **Documentation Generation**
✅ **Error Handling and Recovery**

**Result**: One command deploys the entire SutazAI system with real, functional AI agents.