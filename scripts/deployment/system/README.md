# SutazAI Multi-Agent Task Automation System Deployment

This directory contains production-ready deployment scripts for the SutazAI multi-agent task automation system.

## deploy_complete_system.sh

A comprehensive deployment script that orchestrates the entire SutazAI system deployment with proper error handling, logging, and validation.

### Features

- **Complete System Deployment**: Deploys PostgreSQL, Redis, Ollama, backend services, and AI agents
- **Production-Ready**: Comprehensive error handling, logging, and validation
- **Modular Design**: Services are deployed in dependency order with proper health checks
- **Resource Monitoring**: Checks system resources and validates minimum requirements
- **State Tracking**: Maintains deployment state and provides detailed reporting
- **Multiple Commands**: Support for deploy, stop, restart, status, health checks, and cleanup

### Usage

```bash
# Deploy the complete system (default)
./deploy_complete_system.sh

# Or explicitly
./deploy_complete_system.sh deploy

# Show help
./deploy_complete_system.sh help

# Check system status
./deploy_complete_system.sh status

# Run health checks
./deploy_complete_system.sh health

# View logs
./deploy_complete_system.sh logs [service-name]

# Stop all services
./deploy_complete_system.sh stop

# Restart system
./deploy_complete_system.sh restart

# Clean up system
./deploy_complete_system.sh clean
```

### Services Deployed

#### Core Infrastructure
- **PostgreSQL**: Primary database for system state and data persistence
- **Redis**: In-memory cache and session storage
- **Ollama**: Local LLM inference service with tinyllama model

#### Backend Services
- **backend**: Main API backend service

#### AI Agents
- **senior-ai-engineer**: Senior development and architecture agent
- **deployment-automation-master**: Deployment and automation specialist
- **infrastructure-devops-manager**: Infrastructure and DevOps management
- **ollama-integration-specialist**: Ollama model integration and optimization
- **testing-qa-validator**: Quality assurance and testing automation

#### Optional Services (if enabled)
- **ChromaDB**: Vector database for semantic search
- **Prometheus**: Metrics collection and monitoring
- **Grafana**: Visualization and dashboards

### Environment Variables

```bash
# Enable monitoring services
DEPLOY_MONITORING=true

# Set environment mode
SUTAZAI_ENV=production

# Database configuration
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=sutazai_password
POSTGRES_DB=sutazai

# Redis configuration
REDIS_PASSWORD=redis_password

# Ollama configuration
OLLAMA_MODELS=tinyllama
OLLAMA_HOST=0.0.0.0
OLLAMA_ORIGINS=*
```

### Prerequisites

- Docker and Docker Compose installed
- Minimum 4 CPU cores (recommended)
- Minimum 8GB RAM (recommended)
- Minimum 20GB available disk space
- Network access for pulling Docker images and models

### Access Points

After successful deployment:

- **Backend API**: http://localhost:8000
- **Ollama API**: http://localhost:10104
- **PostgreSQL**: localhost:5432
- **Redis**: localhost:6379
- **Grafana** (if enabled): http://localhost:3000
- **Prometheus** (if enabled): http://localhost:9090

### Logging and State

The script provides comprehensive logging and state tracking:

- **Deployment Logs**: `/opt/sutazaiapp/logs/deployment_TIMESTAMP.log`
- **State File**: `/opt/sutazaiapp/logs/deployment_state_TIMESTAMP.json`
- **Error Tracking**: All errors are logged and counted
- **Success Tracking**: Successful deployments are tracked and reported

### Error Handling

The script includes robust error handling:

- **Prerequisites Validation**: Checks system requirements before deployment
- **Service Health Checks**: Validates each service after deployment
- **Automatic Cleanup**: Cleans up failed deployments automatically
- **Rollback Support**: Can restore from previous successful deployments
- **Detailed Error Reporting**: All errors are logged with timestamps and context

### Troubleshooting

1. **Check Prerequisites**: Run `./deploy_complete_system.sh health` to verify system status
2. **View Logs**: Use `./deploy_complete_system.sh logs [service-name]` to debug issues
3. **Check Resources**: Ensure sufficient CPU, memory, and disk space
4. **Docker Issues**: Verify Docker daemon is running and accessible
5. **Network Issues**: Check network connectivity for image pulls and API calls

### Production Considerations

- Set strong passwords in the environment file
- Configure proper firewall rules
- Set up SSL/TLS for production endpoints
- Configure log rotation
- Set up automated backups
- Monitor resource usage and scale as needed
- Configure proper DNS and load balancing for high availability

### Support

For issues and support:
1. Check the deployment logs in `/opt/sutazaiapp/logs/`
2. Run health checks with `./deploy_complete_system.sh health`
3. Review the system status with `./deploy_complete_system.sh status`
4. Check individual service logs with `docker-compose logs [service-name]`