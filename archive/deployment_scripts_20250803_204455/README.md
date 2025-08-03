# Archived Deployment Scripts

**Archive Date:** 2025-08-03 20:44:55  
**Reason:** Consolidation into single master deploy.sh following Rule 12

## Archived Scripts

### 1. deploy_optimized.sh
- **Source:** `/opt/sutazaiapp/deploy_optimized.sh`
- **Version:** 1.0.0
- **Purpose:** Optimized deployment with staged builds and resource management
- **Key Features:**
  - Lightweight resource configurations
  - Dependency-aware service deployment ordering
  - BuildKit optimization
  - Progress indicators

### 2. deploy-autoscaling.sh
- **Source:** `/opt/sutazaiapp/deployment/autoscaling/scripts/deploy-autoscaling.sh`
- **Purpose:** Auto-scaling deployment for Kubernetes, Docker Swarm, and Docker Compose
- **Key Features:**
  - Multi-platform autoscaling support
  - HPA/VPA configuration for Kubernetes
  - Docker Swarm service autoscaling
  - Compose simulation for development

## Consolidation Summary

All functionality from these scripts has been integrated into the master `deploy.sh` v4.0.0:

- **Optimized deployment:** Available via `./deploy.sh deploy optimized`
- **Auto-scaling:** Available via `./deploy.sh deploy autoscaling` or `./deploy.sh autoscale`
- **Service ordering:** Integrated dependency-aware deployment
- **Resource optimization:** Automatic lightweight mode detection
- **Multi-platform support:** Environment variable controlled platform selection

## Migration

Replace usage of old scripts:

```bash
# OLD
./deploy_optimized.sh deploy
PLATFORM=kubernetes ./deployment/autoscaling/scripts/deploy-autoscaling.sh

# NEW
./deploy.sh deploy optimized
PLATFORM=kubernetes ./deploy.sh autoscale
```

## Compliance

This archival follows CLAUDE.md Rule 12: "One Self-Updating, Intelligent, End-to-End Deployment Script"
- Maintains single canonical deployment script
- Preserves historical functionality
- Eliminates script duplication
- Ensures maintainability and consistency