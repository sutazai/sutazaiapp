# SutazAI Tiered Deployment Quick Reference Guide

## Overview
The SutazAI system has been redesigned with a tiered architecture to optimize resource usage and provide flexible deployment options.

## Available Tiers

### 🟢 Minimal Tier
- **Containers**: 5 (PostgreSQL, Redis, Ollama, Backend, Frontend)
- **Resources**: 2 CPU cores, 4GB RAM
- **Use Case**: Development, testing, demos
- **Features**: Core functionality only

### 🔵 Standard Tier  
- **Containers**: 10 (Minimal + Qdrant, Prometheus, Grafana, Loki, Node-Exporter)
- **Resources**: 4 CPU cores, 8GB RAM
- **Use Case**: Production deployments
- **Features**: Core + Vector DB + Monitoring

### 🟣 Full Tier
- **Containers**: 15-20 (Standard + selected additional services)
- **Resources**: 8+ CPU cores, 16GB+ RAM
- **Use Case**: Enterprise deployments
- **Features**: All capabilities + HA features

## Quick Start Commands

### Deploy a Tier
```bash
# Deploy minimal tier (recommended to start)
./scripts/deploy-tier.sh minimal up

# Deploy standard tier
./scripts/deploy-tier.sh standard up

# Deploy full tier
./scripts/deploy-tier.sh full up
```

### Manage Deployments
```bash
# Check status
./scripts/deploy-tier.sh [tier] status

# View logs
./scripts/deploy-tier.sh [tier] logs
./scripts/deploy-tier.sh [tier] logs [service-name]

# Stop tier
./scripts/deploy-tier.sh [tier] down

# Restart services
./scripts/deploy-tier.sh [tier] restart

# Clean everything (CAUTION!)
./scripts/deploy-tier.sh [tier] clean
```

## Migration from Current System

### Step 1: Analyze Current State
```bash
./scripts/migrate-to-tiered.sh analyze
```

### Step 2: Create Backup
```bash
./scripts/migrate-to-tiered.sh backup
```

### Step 3: Perform Migration
```bash
./scripts/migrate-to-tiered.sh migrate
```

### Step 4: Deploy New Tier
```bash
# Start with minimal
./scripts/deploy-tier.sh minimal up

# Test functionality
curl http://localhost:10010/health
curl http://localhost:10011

# If stable, upgrade to standard
./scripts/deploy-tier.sh standard up
```

## Service Endpoints by Tier

### Minimal Tier Endpoints
| Service | Port | URL |
|---------|------|-----|
| PostgreSQL | 10000 | `postgresql://localhost:10000` |
| Redis | 10001 | `redis://localhost:10001` |
| Ollama | 10104 | `http://localhost:10104` |
| Backend API | 10010 | `http://localhost:10010` |
| Frontend UI | 10011 | `http://localhost:10011` |

### Standard Tier Additional Endpoints
| Service | Port | URL |
|---------|------|-----|
| Qdrant | 10101 | `http://localhost:10101` |
| Prometheus | 10200 | `http://localhost:10200` |
| Grafana | 10201 | `http://localhost:10201` |
| Loki | 10202 | `http://localhost:10202` |
| Node Exporter | 10205 | `http://localhost:10205/metrics` |

## Resource Optimization Achieved

### Before (Current System)
- **Services Defined**: 75
- **Running Containers**: 16 (unoptimized)
- **Memory Allocated**: ~50GB
- **CPU Allocation**: ~40 cores
- **Issues**: Neo4j high CPU, cAdvisor crashes, unused services

### After (Tiered System)
| Metric | Minimal | Standard | Full |
|--------|---------|----------|------|
| Containers | 5 | 10 | 15-20 |
| Memory | 4GB | 8GB | 16GB |
| CPU Cores | 2 | 4 | 8+ |
| Startup Time | <1 min | <2 min | <3 min |

## Troubleshooting

### Issue: Services not starting
```bash
# Check logs
./scripts/deploy-tier.sh [tier] logs

# Verify prerequisites
docker --version
docker-compose --version
docker network ls | grep sutazai

# Check resource availability
docker system df
free -h
```

### Issue: High resource usage
```bash
# Check individual container resources
docker stats --no-stream

# Apply resource limits manually
docker update --memory="512m" --cpus="0.5" [container-name]

# Use minimal tier instead
./scripts/deploy-tier.sh minimal up
```

### Issue: Service connectivity
```bash
# Verify network
docker network inspect sutazai-network

# Check service health
docker ps --format "table {{.Names}}\t{{.Status}}"

# Test endpoints
curl http://localhost:10010/health
```

## Rollback Instructions

If you need to revert to the original configuration:

```bash
# Stop current tier
./scripts/deploy-tier.sh [tier] down

# Restore original docker-compose.yml from backup
cp backups/[timestamp]/docker-compose.yml .

# Start original configuration
docker-compose up -d
```

## Configuration Files

### Core Files
- `docker-compose.minimal.yml` - Minimal tier definition
- `docker-compose.standard.yml` - Standard tier overlay
- `docker-compose.full.yml` - Full tier overlay (to be created as needed)

### Scripts
- `scripts/deploy-tier.sh` - Tier deployment manager
- `scripts/migrate-to-tiered.sh` - Migration assistant

### Documentation
- `docs/architecture-redesign-v59.md` - Complete architectural analysis
- `docs/tiered-deployment-guide.md` - This guide

## Best Practices

1. **Start Small**: Begin with minimal tier and scale up as needed
2. **Monitor Resources**: Use `docker stats` to track actual usage
3. **Test Thoroughly**: Validate each tier before production use
4. **Document Changes**: Keep track of customizations
5. **Regular Cleanup**: Remove unused volumes and images periodically

## Environment Variables

Ensure your `.env` file contains:
```env
# Required
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=<secure-password>
POSTGRES_DB=sutazai
SECRET_KEY=<secure-key>
JWT_SECRET=<secure-jwt>

# Optional (for standard/full tiers)
GRAFANA_PASSWORD=<grafana-password>
NEO4J_PASSWORD=<neo4j-password>
```

## Support & Maintenance

### Daily Operations
```bash
# Morning health check
./scripts/deploy-tier.sh [tier] status

# Evening backup (if needed)
./scripts/migrate-to-tiered.sh backup
```

### Weekly Maintenance
```bash
# Update images
./scripts/deploy-tier.sh [tier] pull

# Clean unused resources
docker system prune -f
docker volume prune -f
```

### Monthly Review
- Analyze resource usage trends
- Consider tier adjustments
- Update documentation
- Review and remove unused services

## Summary

The tiered deployment system provides:
- ✅ 70-80% resource reduction
- ✅ Flexible deployment options
- ✅ Improved stability
- ✅ Easier maintenance
- ✅ Clear scaling path

Choose the tier that matches your needs and scale as your requirements grow!