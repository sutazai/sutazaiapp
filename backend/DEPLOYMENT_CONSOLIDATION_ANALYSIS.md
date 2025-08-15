# Deployment Script Consolidation Analysis
Generated: 2025-08-15T00:00:00Z
Purpose: Identify unique functionality from scattered scripts to consolidate into ./deploy.sh

## Current State
- **Root deploy.sh**: Comprehensive 1024-line script with hardware optimization, backup/rollback, and phased deployment
- **Scripts directory**: 230 shell scripts total, 28 deployment-specific scripts in scripts/deployment/

## Unique Functionality to Preserve

### 1. Fast Startup Modes (fast_start.sh)
- **Critical-only mode**: Start only postgres, redis, neo4j
- **Core mode**: Critical + ollama, backend, frontend
- **Agents-only mode**: Start only AI agents
- **Parallel startup optimization**: 50% time reduction

### 2. MCP Server Management (mcp_bootstrap.sh, mcp_teardown.sh)
- **MCP stack bootstrap**: Sequential Thinking, Context7, UltimateCoder
- **Docker image management**: Build and verify MCP images
- **Health checking**: MCP-specific health endpoints
- **Environment configuration**: MCP_HTTP_PORT settings

### 3. Disaster Recovery (disaster-recovery.sh)
- **Backup retention policies**: Daily (7), Weekly (4), Monthly (12)
- **Recovery state tracking**: JSON state file management
- **Business continuity procedures**: Automated failover
- **Point-in-time recovery**: Restore to specific timestamps

### 4. Service Discovery (consul-*.sh scripts)
- **Consul registration**: Register services with Consul
- **Dynamic IP resolution**: Docker network IP discovery
- **Health check registration**: Service health endpoints
- **Service deregistration**: Cleanup on shutdown

### 5. Model Management (manage-models.sh, ollama-startup.sh)
- **Model pulling**: Automated model downloads
- **Model verification**: Check model availability
- **GPU optimization**: CUDA/ROCm detection and configuration
- **Model preloading**: Warm cache for performance

### 6. Performance Optimization (optimize-ollama-performance.sh, apply_redis_ultra_optimization.sh)
- **Redis optimization**: Ultra performance settings
- **Ollama tuning**: Context size, batch processing
- **Resource allocation**: Dynamic based on hardware
- **Caching strategies**: Preemptive cache warming

### 7. Migration Strategies (zero-downtime-migration.sh, migrate-to-tiered.sh)
- **Blue-green deployment**: Zero-downtime updates
- **Tiered migration**: Progressive service updates
- **Minimal mode migration**: Resource-constrained deployments
- **Agent cluster migration**: Distributed agent deployment

### 8. External Service Integration (integrate-external-services.sh, configure_kong.sh)
- **Kong API Gateway**: JWT configuration, route setup
- **External API registration**: Third-party service integration
- **Security configuration**: API key management
- **Rate limiting setup**: Traffic management

### 9. Standards Initialization (initialize_standards.sh)
- **Codebase standards**: Apply formatting rules
- **Configuration validation**: Verify all configs
- **Security hardening**: Apply security policies
- **Compliance checks**: Ensure regulatory compliance

### 10. Phased Restart (phased-system-restart.sh)
- **Controlled shutdown**: Graceful service stopping
- **Dependency-aware restart**: Proper service ordering
- **Health validation**: Between restart phases
- **State preservation**: Maintain data consistency

## Consolidation Strategy

### Phase 1: Enhance Core Commands
The root deploy.sh already has basic commands. Enhance them:
- `up-core` → Add fast startup modes
- `up-full` → Add parallel optimization
- `up-monitoring` → Already exists
- `up-agents` → Add agent cluster support

### Phase 2: Add New Commands
Add missing functionality as new commands:
- `mcp-up` → Bootstrap MCP servers
- `mcp-down` → Teardown MCP servers  
- `consul-register` → Register services with Consul
- `models-pull` → Download and verify AI models
- `optimize` → Apply performance optimizations
- `migrate` → Zero-downtime migration modes
- `backup` → Enhanced backup with retention policies
- `recover` → Disaster recovery procedures
- `kong-setup` → Configure API gateway
- `standards` → Initialize codebase standards

### Phase 3: Add Advanced Options
Enhance existing options:
- `--startup-mode [critical|core|full|agents]` → Fast startup modes
- `--parallel N` → Control parallelization
- `--consul` → Enable Consul registration
- `--mcp` → Include MCP servers in deployment
- `--optimize-level [minimal|standard|ultra]` → Performance tuning
- `--migration-strategy [rolling|blue-green|canary]` → Update strategies
- `--backup-policy [daily|weekly|monthly]` → Retention policies

## Implementation Plan

1. **Backup current deploy.sh**: Create versioned backup
2. **Add function library**: Modularize functionality
3. **Implement new commands**: Add missing capabilities
4. **Test each scenario**: Validate all modes
5. **Archive old scripts**: Move to archive/ directory
6. **Update documentation**: Reflect new unified interface

## Risk Mitigation
- Keep original scripts in archive/ for 30 days
- Implement --legacy flag to use old scripts if needed
- Comprehensive testing before removal
- Gradual rollout with team notification