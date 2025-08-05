# Docker Compose Consolidation Plan

## Current State
- **56 docker-compose files** in root directory
- **71+ total** including subdirectories
- Massive confusion about which to use
- Port conflicts everywhere
- Duplicate service definitions

## Categories Found

### 1. Main Configuration
- `docker-compose.yml` - Main file (should be primary)
- `docker-compose-optimized.yml`
- `docker-compose.optimized.yml`
- `docker-compose.override.yml`

### 2. Agent-Related (8 files - ALL STUBS)
- `docker-compose.agents.yml`
- `docker-compose.agents-20.yml`
- `docker-compose.agents-critical-fixed.yml`
- `docker-compose.agents-deploy.yml`
- `docker-compose.agents-final.yml`
- `docker-compose.agents-fix.yml`
- `docker-compose.agents-fixed.yml`
- `docker-compose.orchestration-agents.yml`

### 3. Phase-Based Deployment (4 files)
- `docker-compose.phase1-activation.yml`
- `docker-compose.phase1-critical-activation.yml`
- `docker-compose.phase1-critical.yml`
- `docker-compose.phase2-specialized.yml`
- `docker-compose.phase3-auxiliary.yml`

### 4. Health & Monitoring (12 files)
- `docker-compose.health-final.yml`
- `docker-compose.health-fix.yml`
- `docker-compose.health-fixed.yml`
- `docker-compose.health-override.yml`
- `docker-compose.healthfix-override.yml`
- `docker-compose.healthfix.yml`
- `docker-compose.simple-health.yml`
- `docker-compose.monitoring.yml`
- `docker-compose.hygiene-monitor.yml`
- `docker-compose.hygiene-standalone.yml`
- `docker-compose.self-healing.yml`
- `docker-compose.self-healing-critical.yml`

### 5. Infrastructure & Services
- `docker-compose.infrastructure.yml`
- `docker-compose.service-mesh.yml`
- `docker-compose.missing-services.yml`
- `docker-compose.external-integration.yml`
- `docker-compose.auth.yml`

### 6. Ollama-Related (5 files)
- `docker-compose.ollama-cluster.yml`
- `docker-compose.ollama-cluster-optimized.yml`
- `docker-compose.ollama-fix.yml`
- `docker-compose.ollama-optimized.yml`
- `docker-compose.distributed-ollama.yml`

### 7. Hardware/Resource Variants
- `docker-compose.cpu-only.yml`
- `docker-compose.gpu.yml`
- `docker-compose.minimal.yml`
- `docker-compose.resource-optimized.yml`

### 8. Distributed/Scaling
- `docker-compose.distributed.yml`
- `docker-compose.distributed-ai.yml`
- `docker-compose.autoscaling.yml`

### 9. Security
- `docker-compose.secure.yml`
- `docker-compose.security.yml`
- `docker-compose.network-secure.yml`

### 10. Special Purpose
- `docker-compose.jarvis.yml`
- `docker-compose.jarvis-simple.yml`
- `docker-compose.fusion.yml`
- `docker-compose.production.yml`
- `docker-compose.critical-immediate.yml`

### 11. Missing/Fix Files
- `docker-compose.missing-agents.yml`
- `docker-compose.missing-agents-optimized.yml`

## Consolidation Strategy

### 1. Keep Only These Files:
- `docker-compose.yml` - Main services (infrastructure + basic agents)
- `docker-compose.override.yml` - Local development overrides
- `docker-compose.production.yml` - Production-specific settings
- `docker-compose.monitoring.yml` - Optional monitoring stack

### 2. Archive All Others:
- Move to `/archive/docker-compose-cleanup/`
- Keep for reference but mark as deprecated

### 3. Merge Important Services:
- Consolidate all infrastructure into main file
- Remove duplicate agent definitions
- Fix port conflicts
- Remove fantasy services

### 4. Document Clearly:
- One README explaining which file to use when
- Clear port allocation strategy
- No duplicate services