# Local Development Setup

## Prerequisites

### System Requirements
- **Operating System**: Linux, macOS, or Windows with WSL2
- **RAM**: 8GB minimum (16GB recommended)
- **Storage**: 10GB for models and data
- **CPU**: 4+ cores recommended
- **Docker**: 20.0+
- **Docker Compose**: 2.0+

### Required Tools

1. **Docker & Docker Compose**
   ```bash
   # Install Docker (Ubuntu/Debian)
   curl -fsSL https://get.docker.com -o get-docker.sh
   sudo sh get-docker.sh
   
   # Add user to docker group
   sudo usermod -aG docker $USER
   newgrp docker
   
   # Install Docker Compose
   sudo curl -L "https://github.com/docker/compose/releases/latest/download/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
   sudo chmod +x /usr/local/bin/docker-compose
   ```

2. **Git**
   ```bash
   sudo apt-get update
   sudo apt-get install git
   ```

3. **Python 3.11+** (for local scripts)
   ```bash
   sudo apt-get install python3 python3-pip python3-venv
   ```

## Quick Setup

### 1. Clone Repository
```bash
git clone <repository-url>
cd sutazai
```

### 2. Configure Environment
```bash
# Copy example environment file
cp .env.example .env

# Edit environment variables
nano .env
```

### 3. One-Command Deployment
```bash
# Standard development deployment
./scripts/deploy_complete_system.sh deploy --profile standard

# Minimal deployment for testing
./scripts/deploy_complete_system.sh deploy --profile minimal
```

## Development Profiles

### Minimal Profile (2-3GB RAM)
```bash
./scripts/deploy_complete_system.sh deploy --profile minimal
```
**Includes:**
- Core services: PostgreSQL, Redis, Ollama
- Backend API + Frontend
- 3 essential agents
- Basic monitoring

### Standard Profile (6-8GB RAM) - Recommended
```bash
./scripts/deploy_complete_system.sh deploy --profile standard
```
**Includes:**
- Everything in minimal
- Vector databases: ChromaDB, Qdrant, Neo4j
- 6 specialized agents
- Enhanced monitoring

### Full Profile (12-15GB RAM)
```bash
./scripts/deploy_complete_system.sh deploy --profile full
```
**Includes:**
- Everything in standard
- ML frameworks: PyTorch, TensorFlow, JAX
- Workflow tools: n8n, LangFlow, Flowise
- 40+ additional agents
- Complete monitoring stack

## Development Workflow

### Starting Development
```bash
# Start services
./scripts/deploy_complete_system.sh deploy --profile standard

# Verify deployment
./scripts/deploy_complete_system.sh health

# Access development interface
open http://localhost:8501
```

### Daily Development
```bash
# Check system status
./scripts/deploy_complete_system.sh status

# View logs
./scripts/live_logs.sh

# Monitor resources
docker stats
```

### Code Changes
```bash
# Restart specific service after code changes
docker restart sutazai-backend
docker restart sutazai-frontend

# Full restart if needed
./scripts/deploy_complete_system.sh restart
```

## Local Model Management

### Installing Models
```bash
# List available models
docker exec sutazai-ollama ollama list

# Install additional models
docker exec sutazai-ollama ollama pull llama2:7b
docker exec sutazai-ollama ollama pull codellama:7b
docker exec sutazai-ollama ollama pull mistral:7b
```

### Model Configuration
```bash
# Edit model configurations
nano /opt/sutazaiapp/config/ollama_models.yaml

# Restart Ollama service
docker restart sutazai-ollama
```

## Development Tools

### IDE Setup
1. **VS Code Extensions**:
   - Python
   - Docker
   - YAML
   - REST Client

2. **PyCharm Configuration**:
   - Set Python interpreter to project venv
   - Configure Docker integration
   - Enable code formatting (Black, isort)

### Debugging
```bash
# Enable debug mode
export DEBUG=true

# View detailed logs
docker logs sutazai-backend -f --tail 100

# Access Python debugger
docker exec -it sutazai-backend python -m pdb
```

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python tests/integration/test_api.py

# Run health checks
curl http://localhost:8000/health
```

## Database Development

### PostgreSQL Access
```bash
# Connect to database
docker exec -it sutazai-postgres psql -U sutazai_user -d sutazai_db

# Run migrations
docker exec sutazai-backend python -m alembic upgrade head

# View database schema
docker exec sutazai-backend python -c "from app.database import engine; print(engine.table_names())"
```

### Redis Access
```bash
# Connect to Redis
docker exec -it sutazai-redis redis-cli -a redis_password

# Monitor commands
docker exec sutazai-redis redis-cli -a redis_password monitor

# View keys
docker exec sutazai-redis redis-cli -a redis_password keys "*"
```

## Common Development Tasks

### Adding New Agents
1. Create agent directory: `agents/new-agent/`
2. Add Dockerfile and requirements.txt
3. Update agent registry: `agents/agent_registry.json`
4. Test with: `docker build agents/new-agent/`

### Updating Dependencies
```bash
# Update Python dependencies
pip-compile requirements.in

# Update Docker images
docker-compose pull

# Update system packages
sudo apt-get update && sudo apt-get upgrade
```

### Performance Monitoring
```bash
# System resources
htop
docker stats

# Service health
./scripts/deploy_complete_system.sh health

# Custom monitoring
python scripts/health_monitor.py
```

## Troubleshooting

### Common Issues

1. **Port Conflicts**
   ```bash
   # Find process using port
   sudo lsof -i :8000
   
   # Stop conflicting service
   sudo systemctl stop <service>
   ```

2. **Memory Issues**
   ```bash
   # Check memory usage
   free -h
   
   # Use minimal profile
   ./scripts/deploy_complete_system.sh deploy --profile minimal
   ```

3. **Docker Issues**
   ```bash
   # Cleanup Docker
   docker system prune -f
   
   # Reset Docker daemon
   sudo systemctl restart docker
   ```

### Getting Help
1. Check logs: `./scripts/live_logs.sh`
2. Run diagnostics: `./scripts/deploy_complete_system.sh health`
3. Review documentation in `/docs/`
4. Check system status: `./scripts/deploy_complete_system.sh status`

## Best Practices

### Development Workflow
- Use feature branches for development
- Test changes with minimal profile first
- Run health checks before committing
- Keep environment files updated

### Resource Management
- Monitor memory usage regularly
- Stop unused services to free resources
- Use appropriate deployment profile
- Clean up Docker resources periodically

### Security
- Change default passwords in `.env`
- Keep Docker and system updated
- Use non-root users for development
- Regularly update dependencies