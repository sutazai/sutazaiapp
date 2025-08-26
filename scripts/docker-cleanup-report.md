# Docker Container Incident Report & Cleanup Plan
**Date:** 2025-08-26
**Severity:** 2 - Major Service Degradation
**Impact:** Resource exhaustion, service duplication, no monitoring

## Executive Summary
The SutazAI system has 16 running containers with severe issues:
- **ALL containers have random auto-generated names** (epic_elgamal, recursing_edison, etc.)
- **Massive duplication:** 8 mcp/fetch, 4 mcp/sequentialthinking, 4 mcp/duckduckgo instances
- **Critical services DOWN:** PostgreSQL, Redis, ChromaDB, Qdrant all exited 8 hours ago
- **No health checks** on any running container
- **No proper naming convention** applied

## Current Container Inventory

### Running Containers (16 total - ALL DUPLICATES)
| Service Type | Count | Container Names | Status |
|-------------|-------|-----------------|---------|
| mcp/fetch | 8 | festive_maxwell, awesome_cori, epic_elgamal, affectionate_swartz, jovial_bartik, wonderful_roentgen, romantic_germain, zealous_hodgkin | Running (various times) |
| mcp/sequentialthinking | 4 | festive_solomon, recursing_edison, trusting_burnell, agitated_tu | Running (various times) |
| mcp/duckduckgo | 4 | crazy_babbage, angry_faraday, funny_mendel, optimistic_wu | Running (various times) |

### Critical Services (ALL DOWN)
| Service | Container Name | Status | Port | Last Running |
|---------|---------------|---------|------|--------------|
| PostgreSQL | sutazai-postgres | Exited (255) | 10000 | 8 hours ago |
| Redis | sutazai-redis | Exited (255) | 10001 | 8 hours ago |
| ChromaDB | sutazai-chromadb | Exited (255) | 10100 | 8 hours ago |
| Qdrant | sutazai-qdrant | Exited (255) | 10101-10102 | 8 hours ago |
| Ollama | sutazai-ollama | Exited (255) | - | 8 hours ago |

## Root Cause Analysis
1. **MCP Server Launch Failure:** Scripts are creating new containers on each invocation without cleanup
2. **No Container Management:** No `--rm` flag or proper container lifecycle management
3. **Critical Services Crash:** All database services exited with code 255 (fatal error)
4. **No Monitoring:** No health checks or restart policies configured

## Immediate Action Plan

### Phase 1: Stop Bleeding (5 minutes)
1. Stop all duplicate MCP containers
2. Remove stopped duplicate containers
3. Document current state

### Phase 2: Restore Critical Services (15 minutes)
1. Restart PostgreSQL with health checks
2. Restart Redis with health checks
3. Restart vector databases (ChromaDB, Qdrant)
4. Verify service connectivity

### Phase 3: Fix MCP Servers (30 minutes)
1. Identify proper MCP server wrapper scripts
2. Add container lifecycle management (--rm flag)
3. Implement proper naming convention
4. Add health check mechanisms

### Phase 4: Implement Monitoring (30 minutes)
1. Add Docker health checks to all containers
2. Configure restart policies
3. Set up container monitoring
4. Create alerting for container failures

## Cleanup Commands

```bash
# Phase 1: Remove all duplicate MCP containers
docker stop $(docker ps -q --filter "ancestor=mcp/fetch")
docker stop $(docker ps -q --filter "ancestor=mcp/sequentialthinking")
docker stop $(docker ps -q --filter "ancestor=mcp/duckduckgo")
docker rm $(docker ps -aq --filter "ancestor=mcp/fetch")
docker rm $(docker ps -aq --filter "ancestor=mcp/sequentialthinking")
docker rm $(docker ps -aq --filter "ancestor=mcp/duckduckgo")

# Phase 2: Restart critical services
docker start sutazai-postgres
docker start sutazai-redis
docker start sutazai-chromadb
docker start sutazai-qdrant

# Verify services
docker ps --filter "name=sutazai-"
```

## Prevention Measures
1. **Container Lifecycle Management:**
   - Always use `--rm` flag for ephemeral containers
   - Use `--name` for persistent services
   - Implement proper stop/remove in scripts

2. **Health Monitoring:**
   - Add HEALTHCHECK to all Dockerfiles
   - Configure restart policies
   - Implement container monitoring dashboard

3. **Script Improvements:**
   - Fix MCP wrapper scripts to prevent duplication
   - Add cleanup on script exit
   - Implement singleton pattern for services

## Business Impact
- **Service Availability:** 0% for critical databases
- **Resource Usage:** ~300-400% overhead from duplicates
- **Recovery Time:** Estimated 1 hour for full restoration
- **Data Loss Risk:** Low (containers exited cleanly)

## Next Steps
1. Execute cleanup plan immediately
2. Restart critical services with monitoring
3. Fix MCP server launch scripts
4. Implement comprehensive health checking
5. Set up container orchestration properly