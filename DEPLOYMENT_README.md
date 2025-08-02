# SutazAI Universal Deployment System

This document describes the comprehensive, bulletproof deployment system for SutazAI that implements **Rule 12: One-Command Universal Deployment**.

## 🚀 Quick Start

### Single Command Deployment

```bash
# Local development deployment
./deploy.sh

# Production deployment with all safety checks
./scripts/deploy-production.sh

# Force deployment on fresh system
FORCE_DEPLOY=true ./deploy.sh deploy fresh
```

## 📋 Deployment Targets

| Target | Description | Use Case |
|--------|-------------|----------|
| `local` | Development environment | Local testing and development |
| `staging` | Staging environment | Pre-production testing |
| `production` | Production environment | Live system deployment |
| `fresh` | Fresh system installation | Clean slate deployment |

## 🛠️ System Requirements

### Minimum Requirements
- **CPU**: 4 cores
- **Memory**: 16GB RAM
- **Storage**: 100GB available space
- **OS**: Linux (Ubuntu 20.04+, CentOS 8+, RHEL 8+)

### Recommended Requirements
- **CPU**: 8+ cores
- **Memory**: 32GB+ RAM
- **Storage**: 500GB+ SSD
- **Network**: Broadband internet connection

### Production Requirements
- **CPU**: 16+ cores
- **Memory**: 64GB+ RAM
- **Storage**: 1TB+ NVMe SSD
- **Network**: Enterprise-grade connectivity

## 🔧 Features

### Intelligent System Detection
- Automatic platform detection (Linux distro, architecture)
- Hardware capability assessment (CPU, RAM, GPU)
- Container runtime detection (Docker, Podman)
- Network connectivity validation

### Zero-Assumption Deployment
- Works on fresh OS installations
- Automatic dependency installation
- Platform-specific package management
- Self-configuring environment setup

### Bulletproof Error Handling
- Comprehensive rollback system
- Automatic recovery mechanisms
- State tracking and resume capability
- Detailed error reporting

### Security-First Approach
- Automatic secret generation
- SSL certificate management
- Firewall configuration (production)
- File permission security

### Production-Ready Monitoring
- Prometheus metrics collection
- Grafana visualization dashboards
- Log aggregation with Loki
- Health check automation
- Alert management

## 📁 File Structure

```
/opt/sutazaiapp/
├── deploy.sh                          # Master deployment script
├── scripts/
│   ├── deploy-production.sh           # Production deployment wrapper
│   └── production_maintenance.sh      # Automated maintenance
├── docker-compose.yml                 # Main service definitions
├── docker-compose.cpu-only.yml        # CPU-only optimizations
├── docker-compose.gpu.yml             # GPU acceleration config
├── docker-compose.monitoring.yml      # Monitoring stack
├── monitoring/
│   ├── prometheus/
│   │   ├── prometheus.yml             # Metrics collection config
│   │   └── alert_rules.yml            # Alert definitions
│   ├── grafana/                       # Dashboard configurations
│   └── loki/                          # Log aggregation config
├── secrets/                           # Auto-generated secrets
├── ssl/                               # SSL certificates
├── logs/
│   ├── deployment_state/              # Deployment state tracking
│   └── rollback/                      # Rollback point storage
└── .env                               # Environment configuration
```

## 🚀 Deployment Commands

### Basic Commands

```bash
# Deploy to local environment
./deploy.sh deploy local

# Deploy to production (with safety checks)
./scripts/deploy-production.sh

# Check system status
./deploy.sh status

# View service logs
./deploy.sh logs [service_name]

# Run health checks
./deploy.sh health

# Rollback to previous state
./deploy.sh rollback latest
```

## 🔐 Production Deployment

### Prerequisites

1. **Authorization Token**: Set `PRODUCTION_DEPLOY_TOKEN` environment variable
2. **System Resources**: Meet production requirements
3. **Network Access**: Ensure internet connectivity
4. **Privileges**: Root or sudo access for system configuration

### Production Safety Features

- **Authorization verification** with secure token validation
- **Resource validation** against production requirements
- **Pre-deployment backup** creation
- **Network and security checks**
- **SSL certificate validation**
- **Firewall configuration**
- **Post-deployment validation**
- **Automated maintenance scheduling**

## 📊 Monitoring & Health Checks

### Built-in Health Checks

- **Infrastructure**: Database connectivity, cache availability
- **Applications**: API health, frontend responsiveness
- **AI Services**: Model availability, inference capability
- **Integration**: End-to-end functionality tests

### Monitoring Stack

| Service | Purpose | Port | URL |
|---------|---------|------|-----|
| Prometheus | Metrics collection | 9090 | http://localhost:9090 |
| Grafana | Visualization | 3000 | http://localhost:3000 |
| Loki | Log aggregation | 3100 | http://localhost:3100 |
| AlertManager | Alert handling | 9093 | http://localhost:9093 |

## 🐛 Troubleshooting

### Common Issues

#### Docker Not Running
```bash
# Check Docker status
sudo systemctl status docker

# Start Docker
sudo systemctl start docker
```

#### Insufficient Permissions
```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Apply group changes
newgrp docker
```

#### Memory Issues
```bash
# Check available memory
free -h

# Clean Docker cache
docker system prune -f
```

## 🔧 Customization

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DEPLOYMENT_TARGET` | `local` | Target environment |
| `FORCE_DEPLOY` | `false` | Skip safety checks |
| `AUTO_ROLLBACK` | `true` | Enable automatic rollback |
| `ENABLE_MONITORING` | `true` | Deploy monitoring stack |
| `DEBUG` | `false` | Enable debug output |

## 📞 Support

### Getting Help

1. **Check this documentation** for common solutions
2. **Review deployment logs** for specific errors
3. **Use debug mode** for detailed diagnostics
4. **Contact support** for complex issues

---

This deployment system implements the highest standards of reliability, security, and automation to ensure your SutazAI system deploys flawlessly every time.
EOF < /dev/null
