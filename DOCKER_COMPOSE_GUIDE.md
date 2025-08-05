# Docker Compose Files Guide

## ✅ CLEANED UP - Only 6 Essential Files Remain

We've consolidated from 71+ docker-compose files down to 6 essential ones.

### Main Files:

1. **docker-compose.yml** (PRIMARY)
   - Core infrastructure services
   - Main application containers
   - Use this for normal development/deployment
   - Command: `docker-compose up -d`

2. **docker-compose.override.yml** 
   - Local development overrides
   - Automatically loaded with docker-compose.yml
   - Add local customizations here

3. **docker-compose.production.yml**
   - Production-specific configurations
   - Use for production deployments
   - Command: `docker-compose -f docker-compose.yml -f docker-compose.production.yml up -d`

4. **docker-compose.monitoring.yml**
   - Optional monitoring stack (Prometheus, Grafana, etc.)
   - Use if you need monitoring
   - Command: `docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d`

5. **docker-compose.cpu-only.yml**
   - CPU-optimized configuration
   - Use on systems without GPU
   - Command: `docker-compose -f docker-compose.yml -f docker-compose.cpu-only.yml up -d`

6. **docker-compose.agents.yml**
   - Agent containers (WARNING: All are stubs!)
   - Only use if you need the stub services
   - Command: `docker-compose -f docker-compose.yml -f docker-compose.agents.yml up -d`

### ⚠️ Important Notes:

- **Agent Warning**: The agents in docker-compose.agents.yml are STUB implementations
- They return "Hello, I am [agent name]" responses only
- No actual AI functionality implemented

### Common Commands:

```bash
# Start core services
docker-compose up -d

# Start with monitoring
docker-compose -f docker-compose.yml -f docker-compose.monitoring.yml up -d

# Stop all services
docker-compose down

# View logs
docker-compose logs -f [service-name]

# Check status
docker-compose ps
```

### Archived Files:

All 50+ duplicate docker-compose files have been moved to:
`/opt/sutazaiapp/archive/docker-compose-cleanup-20250805_211331/`

These included:
- Multiple agent variations (agents-fix, agents-final, etc.)
- Phase deployments (phase1, phase2, phase3)
- Health check variations
- Ollama cluster configs
- Security variations
- And many more duplicates

### Port Strategy:

Consolidated port allocation:
- 10000-10999: Infrastructure services
- 11000-11999: Agent services (if used)
- No more conflicts!