# SutazAI Master Deployment Script

## Rule 12 Compliance

This repository follows **CLAUDE.md Rule 12** which mandates:
> **"One Self-Updating, Intelligent, End-to-End Deployment Script"**

## The Single Deploy Script

**Location:** `/opt/sutazaiapp/deploy.sh` (symlink to `scripts/deployment/deploy.sh`)  
**Version:** 5.0.0  
**Status:** ✅ Rule 12 Compliant

### Features

- ✅ **Self-Updating**: Automatically pulls latest changes before execution
- ✅ **Comprehensive**: Handles all environments (dev, staging, production)  
- ✅ **Self-Sufficient**: Contains all deployment logic in one place
- ✅ **Intelligent**: Detects system capabilities and adjusts accordingly
- ✅ **Error Handling**: Robust error handling with rollback capabilities
- ✅ **Secure**: No hardcoded passwords, generates secure secrets
- ✅ **Logged**: Comprehensive logging with deployment state tracking

## Quick Start

### Basic Deployment

```bash
# Deploy to local development environment
./deploy.sh deploy local

# Deploy to production
./deploy.sh deploy production

# Deploy with auto-scaling
./deploy.sh deploy autoscaling

# Optimized lightweight deployment
./deploy.sh deploy optimized
```

### System Management

```bash
# Check system status
./deploy.sh status

# View logs
./deploy.sh logs backend      # Specific service
./deploy.sh logs              # All services

# Run health checks
./deploy.sh health

# Clean up resources
./deploy.sh cleanup

# Rollback to previous state
./deploy.sh rollback latest
```

### Advanced Options

```bash
# Skip self-update (not recommended)
SKIP_UPDATE=true ./deploy.sh deploy local

# Force deployment despite warnings
FORCE_DEPLOY=true ./deploy.sh deploy production

# Enable debug output
DEBUG=true ./deploy.sh deploy local

# Use specific git branch for updates
BRANCH=develop ./deploy.sh deploy staging

# Deploy with GPU support
ENABLE_GPU=true ./deploy.sh deploy local

# Lightweight mode for resource-constrained systems
LIGHTWEIGHT_MODE=true ./deploy.sh deploy optimized
```

## Self-Update Mechanism

The script **automatically updates itself** before execution:

1. Fetches latest changes from git repository
2. Checks if deploy.sh has updates
3. If updated, re-executes with new version
4. Maintains all original arguments

To disable self-update (not recommended):
```bash
SKIP_UPDATE=true ./deploy.sh [command]
```

## Deployment Targets

| Target | Description |
|--------|-------------|
| `local` | Local development environment |
| `staging` | Staging environment with full features |
| `production` | Production environment with high availability |
| `fresh` | Fresh system installation from scratch |
| `autoscaling` | Deploy with auto-scaling capabilities |
| `optimized` | Optimized for resource-constrained environments |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `SUTAZAI_ENV` | Deployment environment | local |
| `DEBUG` | Enable debug output | false |
| `FORCE_DEPLOY` | Force deployment despite warnings | false |
| `AUTO_ROLLBACK` | Auto-rollback on failure | true |
| `ENABLE_MONITORING` | Enable monitoring stack | true |
| `ENABLE_GPU` | Enable GPU support | auto |
| `ENABLE_AUTOSCALING` | Enable auto-scaling | false |
| `LIGHTWEIGHT_MODE` | Use lightweight configs | false |
| `PLATFORM` | Container platform | compose |
| `PARALLEL_BUILD` | Enable parallel builds | auto |
| `SKIP_UPDATE` | Skip self-update check | false |
| `BRANCH` | Git branch for updates | main |

## Deployment Phases

The script executes deployment in these phases:

1. **Initialize** - Set up logging and state management
2. **System Detection** - Detect hardware and software capabilities
3. **Dependency Check** - Install required dependencies
4. **Security Setup** - Generate secrets and SSL certificates
5. **Environment Prepare** - Create directories and configurations
6. **Infrastructure Deploy** - Deploy databases and core services
7. **Services Deploy** - Deploy application and AI services
8. **Health Validation** - Validate deployment health
9. **Post Deployment** - Run optimizations and generate access info
10. **Finalize** - Complete deployment and show access information

## Rollback System

The script creates rollback points at critical phases:

```bash
# Rollback to latest checkpoint
./deploy.sh rollback latest

# Rollback to specific point
./deploy.sh rollback rollback_infrastructure_1234567890
```

Rollback points are stored in: `/opt/sutazaiapp/logs/rollback/`

## Logging

All deployment activities are logged:

- **Deployment logs**: `/opt/sutazaiapp/logs/deployment_*.log`
- **State files**: `/opt/sutazaiapp/logs/deployment_state/*.json`
- **Health reports**: `/opt/sutazaiapp/logs/health_report_*.json`
- **Access info**: `/opt/sutazaiapp/logs/ACCESS_INFO_*.txt`

## Security

The script follows security best practices:

- ✅ Generates secure passwords automatically
- ✅ Creates SSL certificates for HTTPS
- ✅ Sets proper file permissions
- ✅ Configures firewall rules (production)
- ✅ No hardcoded credentials
- ✅ Secrets stored in `/opt/sutazaiapp/secrets/`

## System Requirements

### Minimum
- 16GB RAM
- 100GB disk space
- 4 CPU cores
- Docker and Docker Compose v2

### Recommended
- 32GB RAM
- 500GB disk space
- 8 CPU cores
- GPU (optional, for AI workloads)

## Troubleshooting

### Common Issues

**Issue**: "Insufficient resources" error  
**Solution**: Use `FORCE_DEPLOY=true` or `LIGHTWEIGHT_MODE=true`

**Issue**: Docker daemon not running  
**Solution**: Script will attempt to start Docker automatically

**Issue**: Self-update fails  
**Solution**: Use `SKIP_UPDATE=true` temporarily, fix git issues

**Issue**: Services fail health checks  
**Solution**: Check logs with `./deploy.sh logs [service]`

## Rule 12 Compliance Statement

⚠️ **IMPORTANT**: This is the ONLY deployment script that should exist in the repository.

If you find other deployment scripts:
1. **DO NOT USE THEM** - They violate Rule 12
2. **Report them** for consolidation or removal
3. **Use only** `./deploy.sh` for all deployments

The script is designed to be:
- **Self-sufficient**: No need for other scripts
- **Self-updating**: Always uses latest version
- **Comprehensive**: Handles all deployment scenarios
- **Maintainable**: Single point of truth for deployment

## Support

For issues or questions:
1. Check deployment logs in `/opt/sutazaiapp/logs/`
2. Run `./deploy.sh help` for usage information
3. Review this documentation
4. Check CLAUDE.md for system rules

---

**Remember**: One script to rule them all, per Rule 12 of CLAUDE.md! 🚀