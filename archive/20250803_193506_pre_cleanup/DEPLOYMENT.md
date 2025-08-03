# SutazAI Deployment Guide

## Overview

This document provides comprehensive deployment instructions for the SutazAI AI automation platform. Following CLAUDE.md codebase hygiene standards, this is the canonical deployment documentation.

## Prerequisites

### System Requirements

**Minimum Requirements:**
- 16GB RAM
- 100GB disk space
- 4 CPU cores
- Docker and Docker Compose v2
- Linux, macOS, or WSL2 on Windows

**Recommended Requirements:**
- 32GB RAM
- 500GB disk space
- 8+ CPU cores
- GPU (NVIDIA with CUDA support)
- Internet connectivity for model downloads

### Software Dependencies

```bash
# Install Docker (if not already installed)
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Install Docker Compose v2
sudo apt-get update
sudo apt-get install docker-compose-plugin

# Install additional tools
sudo apt-get install curl jq openssl
```

## Deployment Options

### 1. Local Development

```bash
# Deploy to local environment
./deploy.sh deploy local

# With debug logging
DEBUG=true ./deploy.sh deploy local

# Force deployment (skip safety checks)
FORCE_DEPLOY=true ./deploy.sh deploy local
```

### 2. Staging Environment

```bash
# Deploy to staging
./deploy.sh deploy staging

# With monitoring enabled
ENABLE_MONITORING=true ./deploy.sh deploy staging
```

### 3. Production Environment

```bash
# Deploy to production
./deploy.sh deploy production

# With all features enabled
ENABLE_MONITORING=true ENABLE_GPU=true ./deploy.sh deploy production
```

### 4. Fresh Installation

```bash
# Fresh system setup (installs all dependencies)
./deploy.sh deploy fresh

# Fresh installation with custom configuration
POSTGRES_PASSWORD=secure_password REDIS_PASSWORD=secure_redis ./deploy.sh deploy fresh
```

## Configuration

### Environment Variables

The deployment script supports the following environment variables:

| Variable | Description | Default | Example |
|----------|-------------|---------|---------|
| `SUTAZAI_ENV` | Deployment environment | `local` | `production` |
| `POSTGRES_PASSWORD` | PostgreSQL password | auto-generated | `secure_password` |
| `REDIS_PASSWORD` | Redis password | auto-generated | `secure_redis` |
| `NEO4J_PASSWORD` | Neo4j password | auto-generated | `secure_neo4j` |
| `DEBUG` | Enable debug logging | `false` | `true` |
| `FORCE_DEPLOY` | Skip safety checks | `false` | `true` |
| `AUTO_ROLLBACK` | Auto-rollback on failure | `true` | `false` |
| `ENABLE_MONITORING` | Enable monitoring stack | `true` | `false` |
| `ENABLE_GPU` | GPU support | `auto` | `true` |
| `LOG_LEVEL` | Logging level | `INFO` | `DEBUG` |

### GPU Support

Enable GPU acceleration for AI workloads:

```bash
# Auto-detect GPU
./deploy.sh deploy local

# Force GPU usage
ENABLE_GPU=true ./deploy.sh deploy local

# Deploy with GPU override file
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### CPU-Only Mode

For resource-constrained environments:

```bash
# Deploy with CPU optimizations
docker compose -f docker-compose.yml -f docker-compose.cpu-only.yml up -d

# Or via environment variable
ENABLE_GPU=false ./deploy.sh deploy local
```

## Service Management

### Checking Status

```bash
# Check overall system status
./deploy.sh status

# Check specific service
docker compose ps [service_name]

# View running containers
docker ps --filter "name=sutazai-"
```

### Viewing Logs

```bash
# View all service logs
./deploy.sh logs

# View specific service logs
./deploy.sh logs backend
./deploy.sh logs ollama

# Follow logs in real-time
docker compose logs -f [service_name]
```

### Health Checks

```bash
# Run comprehensive health checks
./deploy.sh health

# Check specific endpoints
curl http://localhost:8000/health  # Backend API
curl http://localhost:11434/api/tags  # Ollama
curl http://localhost:8501  # Frontend
```

## Monitoring and Observability

### Access Monitoring Services

| Service | URL | Default Credentials |
|---------|-----|-------------------|
| Grafana | http://localhost:3000 | admin / [generated] |
| Prometheus | http://localhost:9090 | none |
| Loki | http://localhost:3100 | none |

### Key Metrics

- **CPU/Memory Usage**: Available in Grafana dashboards
- **Service Health**: Backend `/health` endpoint
- **Model Performance**: Ollama metrics
- **Database Performance**: PostgreSQL/Redis metrics

## Troubleshooting

### Common Issues

**1. Port Conflicts**
```bash
# Check port usage
sudo lsof -i :8000
sudo lsof -i :11434

# Stop conflicting services
sudo systemctl stop nginx
sudo killall python
```

**2. Insufficient Resources**
```bash
# Check available resources
free -h
df -h
docker system df

# Clean up space
docker system prune -a --volumes
```

**3. Service Startup Failures**
```bash
# Check service logs
docker compose logs [service_name]

# Restart specific service
docker compose restart [service_name]

# Rebuild and restart
docker compose up -d --build [service_name]
```

**4. Database Connection Issues**
```bash
# Check database status
docker exec sutazai-postgres pg_isready
docker exec sutazai-redis redis-cli ping

# Reset database
docker compose down
docker volume rm sutazai_postgres_data
./deploy.sh deploy local
```

### Rollback Procedures

```bash
# Rollback to latest checkpoint
./deploy.sh rollback latest

# Rollback to specific point
./deploy.sh rollback rollback_infrastructure_20240803_120000

# List available rollback points
ls -la logs/rollback/
```

### Getting Help

```bash
# Show help
./deploy.sh help

# Show detailed usage
./deploy.sh --help

# Check script version
./deploy.sh --version
```

## Security Considerations

### Secrets Management

- Passwords are auto-generated and stored in `secrets/`
- Use environment variables for custom passwords
- Never commit secrets to version control
- Rotate passwords regularly in production

### Network Security

- Default deployment binds to localhost only
- Production deployments should use proper firewall rules
- Use SSL/TLS certificates for external access
- Implement proper authentication for monitoring services

### Container Security

- All images use non-root users where possible
- Resource limits prevent container abuse
- Regular security updates via base image updates
- Vulnerability scanning with integrated tools

## Production Considerations

### High Availability

- Use external load balancers
- Implement database clustering
- Set up automated backups
- Monitor service health continuously

### Performance Optimization

- Use GPU acceleration where available
- Optimize container resource limits
- Implement caching strategies
- Monitor and tune database performance

### Backup and Recovery

```bash
# Manual backup
./scripts/backup_system.sh

# Automated backups (production)
# Set up via cron job:
# 0 2 * * * /opt/sutazaiapp/scripts/backup_system.sh
```

## Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Frontend     │    │     Backend     │    │   AI Services   │
│   (Streamlit)   │────│    (FastAPI)    │────│   (Ollama/etc)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│    Database     │    │   Vector Store  │    │   Monitoring    │
│ (PostgreSQL/    │    │ (ChromaDB/      │    │ (Prometheus/    │
│  Redis/Neo4j)   │    │  Qdrant/FAISS)  │    │  Grafana/Loki)  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Development Workflow

### Local Development

1. Fork and clone the repository
2. Run `./deploy.sh deploy local`
3. Make your changes
4. Test with `./deploy.sh health`
5. Submit pull request

### Testing Changes

```bash
# Run deployment validation
./deploy.sh health

# Run specific tests
docker compose exec backend python -m pytest

# Check code quality
docker compose exec backend python -m flake8
```

## Support and Contributing

### Getting Support

- Check this documentation first
- Review logs in `logs/` directory
- Create GitHub issue with reproduction steps
- Include deployment state file for debugging

### Contributing

- Follow CLAUDE.md codebase hygiene standards
- Use the canonical deployment script only
- Test all changes thoroughly
- Update documentation for new features

### Reporting Issues

Include the following information:
- Operating system and version
- Docker version
- Error messages and logs
- Steps to reproduce
- System resources (CPU, RAM, disk)

---

*This deployment guide follows CLAUDE.md codebase hygiene standards and is the canonical documentation for SutazAI deployment.*