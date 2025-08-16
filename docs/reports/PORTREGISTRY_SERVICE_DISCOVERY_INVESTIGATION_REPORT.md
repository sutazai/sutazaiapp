# PortRegistry and Service Discovery Investigation Report

**Date**: 2025-08-16  
**Investigator**: Backend API Architect Agent  
**Severity**: HIGH  
**Status**: CRITICAL ARCHITECTURE ISSUES IDENTIFIED

## Executive Summary

Investigation reveals significant architectural inconsistencies and configuration conflicts in the PortRegistry and service discovery systems. The system has **4 competing service discovery patterns**, **3 different port registries**, and critical mismatches between documented ports and actual runtime configurations.

## Key Findings

### 1. Multiple Competing Service Registries

The system has **THREE separate service registries** with conflicting information:

1. **PortRegistry.md** (`/opt/sutazaiapp/IMPORTANT/diagrams/PortRegistry.md`)
   - Documents 31 services with port allocations
   - Shows 6 services as "DEFINED BUT NOT RUNNING"
   - Port range: 10000-11201

2. **service_registry.py** (`/opt/sutazaiapp/backend/app/mesh/service_registry.py`)
   - Defines 16 REAL_SERVICES
   - Uses dynamic port detection (container vs host)
   - Registers with Consul

3. **core/service_registry.py** (`/opt/sutazaiapp/backend/app/core/service_registry.py`)
   - Lists 61 services (mostly fantasy agents)
   - Hardcoded URLs with incorrect ports
   - Uses service names instead of container names

### 2. Port Configuration Mismatches

**Critical Discrepancies Found:**

| Service | PortRegistry.md | service_registry.py | core/service_registry.py | Docker Reality |
|---------|----------------|-------------------|------------------------|----------------|
| Ollama | 10104 (external) | 10104 (external) | **10104 (wrong)** | ✅ 10104 |
| Backend | 10010 | 10010 | Not listed | ✅ 10010 |
| PostgreSQL | 10000 | 10000 | 5432 (internal) | ✅ 10000 |
| Redis | 10001 | 10001 | 6379 (internal) | ✅ 10001 |

**Issue**: core/service_registry.py uses internal ports with wrong hostnames:
```python
"ollama": {"url": "http://ollama:10104", ...}  # WRONG - should be 11434 internally
"postgres": {"url": "postgresql://...@postgres:5432/..."}  # Correct internal port
```

### 3. Service Discovery Architecture Conflicts

**Four Competing Patterns Identified:**

1. **Consul-Based Discovery** (service_mesh.py)
   - Properly integrated with Consul at port 10006
   - Supports health checks and dynamic registration
   - Has fallback to local cache

2. **Kong API Gateway** (partially configured)
   - Admin API at port 10015
   - Proxy at port 10005
   - Upstream configuration attempted but incomplete

3. **Static Service Registry** (core/service_registry.py)
   - Hardcoded service URLs
   - No dynamic discovery
   - Incorrect port mappings

4. **MCP Bridge Discovery** (mcp_bridge.py)
   - Separate registry for MCP services
   - Configuration-based discovery
   - No integration with Consul

### 4. Container vs Host Networking Issues

**Dynamic Port Detection Logic:**
```python
if is_container:
    address = service_def.container_name
    port = service_def.internal_port
else:
    address = "localhost"
    port = service_def.external_port
```

**Problem**: This assumes all services can detect container environment correctly, but:
- Some services run on host network
- MCP servers run as separate processes
- Port mappings not consistent across all services

### 5. Fantasy Services vs Reality

**PortRegistry.md Status Analysis:**
- **22 containers actually running** (verified via docker ps)
- **6 services marked as "DEFINED BUT NOT RUNNING"**
- **39 fantasy agent services** in core/service_registry.py that don't exist

**Non-existent services in core registry:**
- autogpt, crewai, letta, aider, gpt-engineer
- localagi, tabbyml, semgrep, autogen, agentzero
- bigagi, browser-use, skyvern, dify, agentgpt
- (and 24 more...)

### 6. Consul Integration Issues

**Current Consul Status:**
- ✅ Consul running at port 10006
- ✅ 14 services registered (verified via API)
- ❌ But service_mesh.py only registers 16 services
- ❌ Core services missing from Consul registration

**Registered in Consul:**
```json
{
  "backend-api-sutazai-backend-8000": { "Address": "sutazai-backend", "Port": 8000 },
  "frontend-ui-sutazai-frontend-8501": { "Address": "sutazai-frontend", "Port": 8501 },
  "grafana-dashboards-sutazai-grafana-3000": { "Address": "sutazai-grafana", "Port": 3000 }
  // ... others
}
```

**Issue**: Using internal ports in Consul, not external mappings.

### 7. Service Mesh Implementation Gaps

**ServiceMesh class issues:**
1. **Circuit Breaker**: Implemented but not properly tracking service failures
2. **Load Balancer**: Round-robin counter not persisted
3. **Health Checks**: Only checking `/health` endpoint, many services use different paths
4. **Kong Integration**: Attempted but incomplete - upstreams not properly configured

### 8. MCP Integration Conflicts

**MCP Services Discovery:**
- 21 MCP services defined in .mcp.json
- Separate from main service discovery
- No Consul registration for MCP services
- Bridge pattern creates isolation from mesh

## Root Causes

### 1. Architectural Fragmentation
- Multiple teams/phases created different service discovery mechanisms
- No single source of truth for service configuration
- Competing patterns never reconciled

### 2. Port Configuration Management
- No centralized port allocation management
- Mix of internal/external port references
- Container networking not properly abstracted

### 3. Fantasy-Driven Development
- Registry contains 61 services, only 22 actually exist
- Configuration for non-existent services creates confusion
- No validation between configuration and reality

### 4. Incomplete Integration
- Service mesh partially implemented
- Kong gateway configured but not fully integrated
- Consul running but not authoritative

## Impact Assessment

### Critical Issues
1. **Service Discovery Failures**: Services may not find each other reliably
2. **Port Conflicts**: Risk of binding conflicts with misconfigured ports
3. **Health Check Failures**: Incorrect health endpoints cause false negatives
4. **MCP Isolation**: MCP services not participating in service mesh

### Operational Issues
1. **Debugging Complexity**: Multiple registries make troubleshooting difficult
2. **Configuration Drift**: Reality diverging from documentation
3. **Scaling Problems**: Cannot reliably scale with current discovery chaos
4. **Monitoring Gaps**: Some services not properly monitored

## Recommendations

### Immediate Actions (Priority 1)

1. **Consolidate Service Registries**
   - Choose ONE authoritative registry (recommend Consul-based)
   - Remove fantasy services from all registries
   - Update all code to use single registry

2. **Fix Port Mappings**
   - Create single source of truth for port allocations
   - Separate internal and external port configurations
   - Validate all services against actual Docker configuration

3. **Standardize Service Discovery**
   - All services must register with Consul
   - Remove static service URLs
   - Implement proper service discovery clients

### Short-term Fixes (Priority 2)

1. **Complete Service Mesh Integration**
   - Finish Kong upstream configuration
   - Implement proper health check endpoints
   - Add MCP services to mesh

2. **Remove Fantasy Services**
   - Delete all non-existent service configurations
   - Update documentation to reflect reality
   - Add validation to prevent future fantasy services

3. **Fix Container Networking**
   - Standardize on service names for internal communication
   - Use Consul DNS for service resolution
   - Document networking architecture

### Long-term Improvements (Priority 3)

1. **Implement Service Catalog**
   - Central service definition repository
   - Automated validation against running services
   - API for service registration and discovery

2. **Enhanced Monitoring**
   - Service discovery metrics
   - Port conflict detection
   - Configuration drift alerts

3. **Architecture Documentation**
   - Clear service discovery patterns
   - Network topology diagrams
   - Port allocation strategy

## Evidence

### Files Analyzed
- `/opt/sutazaiapp/IMPORTANT/diagrams/PortRegistry.md`
- `/opt/sutazaiapp/backend/app/mesh/service_registry.py`
- `/opt/sutazaiapp/backend/app/core/service_registry.py`
- `/opt/sutazaiapp/backend/app/mesh/service_mesh.py`
- `/opt/sutazaiapp/backend/app/api/v1/endpoints/mcp.py`

### Commands Executed
- `docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"`
- `curl -s http://localhost:10006/v1/agent/services`

### Key Code Snippets

**Problem 1: Wrong Port in core/service_registry.py**
```python
"ollama": {"url": "http://ollama:10104", "type": "llm", "priority": 1},
# Should be: http://ollama:11434 (internal) or http://localhost:10104 (external)
```

**Problem 2: Fantasy Services**
```python
"autogpt": {"url": "http://autogpt:8080", ...},  # Service doesn't exist
"crewai": {"url": "http://crewai:8080", ...},   # Service doesn't exist
```

**Problem 3: Competing Discovery**
```python
# Three different ways to find services:
service_registry.get_service("ollama")  # Static registry
await discovery.discover_services("ollama")  # Consul
bridge.get_service_info("ollama")  # MCP Bridge
```

## Conclusion

The PortRegistry and service discovery architecture is in a **CRITICAL** state with multiple competing systems, configuration mismatches, and fantasy services. The system works partially due to fallback mechanisms and hardcoded values, but is fragile and will fail under load or when scaling.

**Immediate action required** to consolidate service discovery into a single, authoritative system based on actual running services, not fantasy configurations.

## Next Steps

1. Emergency meeting to decide on single service discovery pattern
2. Create migration plan to consolidate registries
3. Implement validation to ensure configuration matches reality
4. Remove all fantasy service configurations
5. Update documentation to reflect actual architecture

---

**Report Status**: COMPLETE  
**Action Required**: IMMEDIATE  
**Risk Level**: HIGH  
**Estimated Fix Time**: 40-60 hours for full consolidation