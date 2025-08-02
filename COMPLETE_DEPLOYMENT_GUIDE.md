# SutazAI Complete Deployment Guide

## Overview
This guide reflects all lessons learned from the deployment process and provides the most efficient way to deploy the SutazAI Multi-Agent Task Automation System.

## Quick Start

```bash
# Standard deployment (recommended)
./scripts/deploy_complete_system.sh deploy --profile standard

# Minimal deployment (testing/development)
./scripts/deploy_complete_system.sh deploy --profile minimal

# Full deployment (all services)
./scripts/deploy_complete_system.sh deploy --profile full
```

## Deployment Profiles

### 1. **Minimal** (8-10 services)
- Core: PostgreSQL, Redis, Ollama
- Backend: API + Frontend
- Basic Agents: 3 essential agents
- Memory: ~2-3GB
- Best for: Development, testing

### 2. **Standard** (20-25 services) - RECOMMENDED
- Everything in Minimal +
- Vector DBs: ChromaDB, Qdrant, Neo4j
- Core Agents: 6 specialized agents
- Memory: ~6-8GB
- Best for: Production use

### 3. **Full** (70+ services)
- Everything in Standard +
- ML Frameworks: PyTorch, TensorFlow, JAX
- Monitoring: Prometheus, Grafana, Loki
- Workflow Tools: n8n, LangFlow, Flowise
- All 40+ additional agents
- Memory: ~12-15GB
- Best for: Enterprise deployments

## Key Improvements Made

### 1. **Fixed Redis Health Issues**
- Added special handling for Redis authentication
- Health check now properly validates connectivity

### 2. **Profile-Based Deployment**
- Intelligent service selection based on profile
- Automatic resource optimization

### 3. **Network Cleanup**
- Automatic cleanup of conflicting networks
- Prevents deployment failures

### 4. **Enhanced Health Checks**
- Service-specific health validation
- Detailed status reporting

### 5. **Better Error Handling**
- Failed services are tracked
- Automatic cleanup on failure
- Detailed error logging

## Management Commands

### Status and Health
```bash
# Check all services
./scripts/deploy_complete_system.sh status

# Run health checks
./scripts/deploy_complete_system.sh health

# View logs
./scripts/deploy_complete_system.sh logs
./scripts/deploy_complete_system.sh logs backend
```

### Service Control
```bash
# Stop all services
./scripts/deploy_complete_system.sh stop

# Restart all services
./scripts/deploy_complete_system.sh restart

# Clean system (removes containers/networks)
./scripts/deploy_complete_system.sh clean
```

### Live Monitoring
```bash
# Interactive monitoring menu
./scripts/live_logs.sh

# Option 10: Unified logs from all services
# Option 2: Individual service logs
# Option 4: Container statistics
```

## Common Issues and Solutions

### 1. **Redis "unhealthy" Status**
- **Issue**: Redis health check fails with authentication
- **Solution**: Script now handles this automatically
- **Manual Fix**: `docker exec sutazai-redis redis-cli -a redis_password ping`

### 2. **Network Conflicts**
- **Issue**: "network already exists" errors
- **Solution**: Script auto-cleans networks
- **Manual Fix**: `docker network prune -f`

### 3. **Agent Registration Failures**
- **Issue**: Agents show 404 on coordinator endpoint
- **Expected**: Normal without full orchestration service
- **Impact**: Minimal - agents work independently

### 4. **Memory Constraints**
- **Issue**: Services failing on low-memory systems
- **Solution**: Use minimal or standard profile
- **Check**: `free -h` before deployment

## Architecture Overview

### Core Infrastructure
- **PostgreSQL**: Primary database (sutazai_db)
- **Redis**: Caching and message queue
- **Ollama**: Local LLM serving

### Vector Databases (Standard+)
- **ChromaDB**: Document embeddings
- **Qdrant**: High-performance vector search
- **Neo4j**: Graph database for relationships

### AI Agents
- **Currently**: 71 configured agents
- **Active**: 34-40 depending on profile
- **Categories**: 
  - Task automation
  - Code generation
  - Security analysis
  - Document processing
  - Workflow orchestration

### Monitoring (Full profile)
- **Prometheus**: Metrics collection
- **Grafana**: Visualization dashboards
- **Loki**: Log aggregation

## Post-Deployment Steps

### 1. Verify Services
```bash
# Check health endpoint
curl http://localhost:8000/health | jq .

# Check Ollama models
curl http://localhost:11434/api/tags | jq .

# Access frontend
open http://localhost:8501
```

### 2. Load Additional Models
```bash
# Add more models to Ollama
docker exec sutazai-ollama ollama pull llama2:7b
docker exec sutazai-ollama ollama pull codellama:7b
```

### 3. Configure Agents
- Agent configs in `/opt/sutazaiapp/agents/configs/`
- Modify as needed for your use case
- Restart specific agents after changes

## Resource Optimization

### Memory Management
- Minimal: 2-3GB (50% of 8GB system)
- Standard: 6-8GB (50% of 16GB system)
- Full: 12-15GB (50% of 32GB system)

### CPU Optimization
- Ollama limited to 6 cores max
- Each agent limited to 1-2 cores
- Backend services share 4 cores

### Disk Usage
- Base install: ~5GB
- With models: ~10-20GB
- Full deployment: ~30-50GB

## Security Considerations

1. **Change Default Passwords**
   - Edit `.env` file before deployment
   - Update PostgreSQL, Redis, Neo4j passwords

2. **Network Security**
   - Services bound to localhost by default
   - Use reverse proxy for external access
   - Enable firewall rules as needed

3. **API Keys**
   - Add your own API keys to `.env`
   - Required for external service integration

## Troubleshooting

### Debug Mode
```bash
# Enable verbose logging
export DEBUG=true
./scripts/deploy_complete_system.sh deploy

# Check specific service logs
docker logs sutazai-backend -f
docker logs sutazai-ollama -f
```

### Health Check Details
```bash
# Detailed system status
curl http://localhost:8000/health | jq .

# Agent status
curl http://localhost:8000/api/v1/agents | jq .
```

### Reset Everything
```bash
# Complete cleanup (WARNING: Deletes all data)
./scripts/deploy_complete_system.sh clean
# Answer 'y' to remove volumes
```

## Next Steps

1. **Deploy with standard profile** (recommended)
2. **Access the UI** at http://localhost:8501
3. **Check API docs** at http://localhost:8000/docs
4. **Monitor services** with `./scripts/live_logs.sh`
5. **Load your data** and start automating tasks!

## Support

- Logs: `/opt/sutazaiapp/logs/`
- State files: `/opt/sutazaiapp/logs/deployment_state_*.json`
- Health endpoint: http://localhost:8000/health
- Live monitoring: `./scripts/live_logs.sh`