# Blue/Green Deployment Guide for Perfect Jarvis System

This guide provides comprehensive instructions for implementing and managing zero-downtime deployments using the Blue/Green deployment strategy for the Perfect Jarvis system.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Deployment Process](#deployment-process)
- [Monitoring and Health Checks](#monitoring-and-health-checks)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

Blue/Green deployment is a technique that reduces downtime and risk by running two identical production environments called Blue and Green. At any time, only one environment is live, serving production traffic. The other environment is idle, ready to receive the new deployment.

### Benefits
- **Zero Downtime**: Traffic switches instantly between environments
- **Fast Rollback**: Immediate rollback capability if issues occur
- **Testing in Production**: Test new versions with production data
- **Risk Reduction**: Isolated deployment testing before traffic switch

## Architecture

```
┌─────────────────┐    ┌─────────────────┐
│   HAProxy LB    │    │  GitHub Actions │
│   (Traffic      │    │  (CI/CD)        │
│    Router)      │    │                 │
└─────┬───────────┘    └─────────────────┘
      │
      ├─── Blue Environment ───┐
      │    ┌─────────────────┐  │
      │    │  Backend API    │  │
      │    │  Frontend UI    │  │
      │    │  Jarvis Agents  │  │
      │    └─────────────────┘  │
      │                         │
      └─── Green Environment ──┤
           ┌─────────────────┐  │
           │  Backend API    │  │
           │  Frontend UI    │  │
           │  Jarvis Agents  │  │
           └─────────────────┘  │
                                │
      ┌─────────────────────────┘
      │
      └─── Shared Services ────┐
           ┌─────────────────┐  │
           │  PostgreSQL     │  │
           │  Redis          │  │
           │  Neo4j          │  │
           │  Ollama         │  │
           │  Monitoring     │  │
           └─────────────────┘  │
```

## Prerequisites

Before implementing Blue/Green deployment, ensure you have:

1. **Docker and Docker Compose** installed
2. **HAProxy** for load balancing
3. **Python 3.8+** for management scripts
4. **Required Python packages**:
   ```bash
   pip install requests dataclasses
   ```
5. **System packages**:
   ```bash
   # Ubuntu/Debian
   sudo apt-get install socat curl jq
   
   # CentOS/RHEL
   sudo yum install socat curl jq
   ```

## Quick Start

### 1. Initial Setup

```bash
# Clone the repository
cd /opt/sutazaiapp

# Make scripts executable
chmod +x scripts/deploy/blue-green-deploy.sh
chmod +x scripts/deploy/health-checks.sh
chmod +x scripts/deploy/manage-environments.py

# Create the network
docker network create sutazai-network

# Copy secrets template and configure
cp config/deploy/secrets.env.template config/deploy/secrets.env
# Edit secrets.env with your actual values
```

### 2. Start Shared Services

```bash
# Start shared infrastructure
docker-compose -f docker/docker-compose.blue-green.yml up -d postgres redis neo4j ollama chromadb qdrant faiss prometheus grafana loki haproxy
```

### 3. Deploy Blue Environment

```bash
# Deploy to blue environment
./scripts/deploy/blue-green-deploy.sh --target-color blue --deployment-tag v1.0.0
```

### 4. Deploy Green Environment

```bash
# Deploy to green environment
./scripts/deploy/blue-green-deploy.sh --target-color green --deployment-tag v1.1.0
```

### 5. Switch Traffic

```bash
# Switch traffic to green environment
python3 scripts/deploy/manage-environments.py --switch-to green

# Or use automatic switching during deployment
./scripts/deploy/blue-green-deploy.sh --target-color green --deployment-tag v1.1.0 --auto-switch
```

## Configuration

### Environment Variables

#### Shared Configuration (`config/deploy/shared.env`)
- Database connections
- Ollama configuration (following CLAUDE.md Rule 16: TinyLlama)
- Monitoring settings
- Network configuration

#### Blue Environment (`config/deploy/blue.env`)
- Blue-specific settings
- Load balancer weights
- Logging configuration

#### Green Environment (`config/deploy/green.env`)
- Green-specific settings
- Feature flags for testing
- Experimental features

#### Secrets (`config/deploy/secrets.env`)
- Database passwords
- API keys
- SSL certificates
- Third-party credentials

### HAProxy Configuration

The HAProxy configuration supports:
- Automatic health checks
- Traffic switching via admin socket
- SSL termination
- Rate limiting
- Monitoring endpoints

Key endpoints:
- `:80` - HTTP (redirects to HTTPS)
- `:443` - HTTPS frontend
- `:8404` - HAProxy statistics
- `:20010` - Blue/Green API routing
- `:20011` - Blue/Green Frontend routing
- `:21010` - Blue environment direct access
- `:21011` - Green environment direct access

## Deployment Process

### Automated Deployment

```bash
# Full deployment with all safety checks
./scripts/deploy/blue-green-deploy.sh \
  --target-color green \
  --deployment-tag v2.0.0 \
  --auto-switch

# Quick deployment (skip tests and backup)
./scripts/deploy/blue-green-deploy.sh \
  --target-color green \
  --deployment-tag v2.0.0 \
  --skip-tests \
  --skip-backup \
  --auto-switch
```

### Manual Process

1. **Pre-deployment**:
   ```bash
   # Check current status
   python3 scripts/deploy/manage-environments.py --status
   
   # Create backup
   ./scripts/deploy/blue-green-deploy.sh --target-color green --skip-tests --skip-switch
   ```

2. **Deployment**:
   ```bash
   # Deploy to target environment
   docker-compose -f docker/docker-compose.blue-green.yml up -d green-backend green-frontend
   ```

3. **Testing**:
   ```bash
   # Run health checks
   ./scripts/deploy/health-checks.sh --environment green
   
   # Test endpoints directly
   curl http://localhost:21011/health
   ```

4. **Traffic Switch**:
   ```bash
   # Switch traffic
   python3 scripts/deploy/manage-environments.py --switch-to green
   ```

5. **Verification**:
   ```bash
   # Verify switch was successful
   python3 scripts/deploy/manage-environments.py --status
   
   # Run post-deployment health checks
   ./scripts/deploy/health-checks.sh --environment green
   ```

### Rollback

```bash
# Automatic rollback to previous environment
python3 scripts/deploy/manage-environments.py --rollback

# Or use deployment script rollback
./scripts/deploy/blue-green-deploy.sh --rollback
```

## Monitoring and Health Checks

### Health Check Types

1. **Service Availability**: Check if containers are running
2. **HTTP Endpoints**: Validate API responses
3. **Database Connectivity**: Test database connections
4. **Model Loading**: Verify Ollama models are available
5. **Performance**: Response time validation

### Health Check Commands

```bash
# Check specific environment
./scripts/deploy/health-checks.sh --environment blue
./scripts/deploy/health-checks.sh --environment green

# Quick health check
./scripts/deploy/health-checks.sh --environment all --quick

# Verbose health check with debug info
./scripts/deploy/health-checks.sh --environment blue --verbose

# Check shared services only
./scripts/deploy/health-checks.sh --environment shared
```

### Monitoring Endpoints

- **HAProxy Stats**: http://localhost:8404/stats
- **Prometheus Metrics**: http://localhost:10200
- **Grafana Dashboards**: http://localhost:10201
- **Health Status API**: python3 scripts/deploy/manage-environments.py --status

## Troubleshooting

### Common Issues

#### 1. Traffic Not Switching

```bash
# Check HAProxy backend status
echo "show stat" | socat stdio /var/run/haproxy/admin.sock

# Manually set weights
echo "set weight api_backend/green-api 100" | socat stdio /var/run/haproxy/admin.sock
echo "set weight api_backend/blue-api 0" | socat stdio /var/run/haproxy/admin.sock
```

#### 2. Service Health Check Failures

```bash
# Debug specific service
docker logs sutazai-green-backend
docker exec sutazai-green-backend curl -f http://localhost:8000/health

# Check network connectivity
docker network inspect sutazai-green-network
```

#### 3. Database Connection Issues

```bash
# Test database connectivity
docker exec sutazai-postgres pg_isready -U sutazai

# Check database logs
docker logs sutazai-postgres
```

#### 4. Ollama Model Issues

```bash
# Check available models (Rule 16: TinyLlama)
curl http://localhost:10104/api/tags

# Load TinyLlama model if missing
docker exec sutazai-ollama ollama pull tinyllama
```

### Debugging Scripts

```bash
# Enable verbose logging
export VERBOSE=true

# Check deployment state
python3 scripts/deploy/manage-environments.py --export-state > debug-state.json

# Analyze Docker containers
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check resource usage
docker stats --no-stream
```

## Best Practices

### 1. Deployment Safety

- Always run health checks before switching traffic
- Maintain automated rollback capabilities  
- Test deployments in staging environment first
- Use feature flags for gradual rollouts

### 2. Configuration Management

- Keep environment configurations in version control
- Use secrets management for sensitive data
- Validate configuration before deployment
- Document all configuration changes

### 3. Monitoring and Alerting

- Monitor both environments continuously
- Set up alerts for health check failures
- Track deployment metrics and success rates
- Maintain deployment logs for troubleshooting

### 4. Database Considerations

- Use shared databases between environments
- Plan for schema migrations carefully
- Backup data before major deployments
- Test database connectivity in both environments

### 5. Security

- Secure HAProxy admin socket access
- Use SSL/TLS for all communications
- Regularly rotate secrets and credentials
- Monitor for security vulnerabilities

### 6. Performance

- Monitor response times during switches
- Load test both environments
- Optimize resource allocation
- Plan for traffic spikes during switches

## GitHub Actions Integration

The deployment pipeline is fully automated through GitHub Actions:

```yaml
# Manual deployment trigger
name: Deploy to Production
on:
  workflow_dispatch:
    inputs:
      target_environment:
        type: choice
        options: [blue, green]
      auto_switch:
        type: boolean
        default: false
```

### Deployment Stages

1. **Validation**: Check prerequisites and compliance
2. **Build**: Create and push Docker images
3. **Staging**: Deploy to staging environment
4. **Gate**: Manual approval for production
5. **Production**: Deploy to production environment
6. **Cleanup**: Clean up old resources

## Conclusion

The Blue/Green deployment strategy provides a robust, zero-downtime deployment solution for the Perfect Jarvis system. By following this guide and the provided scripts, you can achieve reliable, automated deployments with immediate rollback capabilities.

For additional support or questions, refer to the troubleshooting section or examine the detailed logs generated by the deployment scripts.