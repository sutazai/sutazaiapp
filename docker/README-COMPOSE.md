# Docker Compose Configuration Consolidation

## Overview
The Docker Compose configuration has been restructured for better organization and maintainability.

## File Structure

### Primary Files
- `docker-compose.yml` - Main configuration with all services
- `docker-compose.override.yml` - Development overrides (auto-loaded)
- `.env.master` - Consolidated environment configuration

### Profile-Based Configurations
Instead of multiple variant files, use Docker Compose profiles:

```bash
# Core services only
docker-compose --profile core up

# Full stack with monitoring
docker-compose --profile core --profile monitoring up

# Performance optimized
docker-compose --profile core --profile performance up

# Security hardened
docker-compose --profile core --profile security up

# Blue-green deployment
docker-compose --profile blue up  # or --profile green
```

### Service Profiles
- **core**: Essential services (postgres, redis, backend, frontend)
- **monitoring**: Monitoring stack (prometheus, grafana, loki)
- **agents**: AI agent services
- **vector**: Vector databases (chromadb, qdrant, faiss)
- **performance**: Performance-optimized configurations
- **security**: Security-hardened configurations
- **blue/green**: Blue-green deployment profiles

## Migration from Multiple Files

### Old Structure
```
docker-compose.yml
docker-compose.base.yml
docker-compose.minimal.yml
docker-compose.optimized.yml
docker-compose.performance.yml
docker-compose.secure.yml
docker-compose.standard.yml
docker-compose.ultra-performance.yml
...
```

### New Structure
```
docker-compose.yml          # All services with profiles
docker-compose.override.yml # Development overrides
docker-compose.prod.yml     # Production overrides (optional)
```

## Usage Examples

### Development
```bash
# Uses docker-compose.yml + docker-compose.override.yml automatically
docker-compose up
```

### Production
```bash
# Core services with security
docker-compose --profile core --profile security up -d

# Full stack
docker-compose --profile core --profile monitoring --profile agents up -d
```

### Specific Configurations
```bash
# setup
docker-compose --profile core up

# With monitoring
docker-compose --profile core --profile monitoring up

# Performance optimized
docker-compose --profile core --profile performance up
```

## Benefits
1. Single source of truth for service definitions
2. Profile-based activation instead of file proliferation
3. Easier maintenance and updates
4. Clear separation of concerns
5. Reduced configuration duplication