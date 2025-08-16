# DOCUMENTATION UPDATE REQUIREMENTS
## Specific Changes Required for CLAUDE.md and AGENTS.md
**Generated**: 2025-08-15
**Priority**: URGENT
**Implementation Timeline**: Immediate

---

## CLAUDE.md - REQUIRED UPDATES

### Section 1: Project Overview
**CURRENT**:
```markdown
- **Agent System**: 7+ operational AI agents for various automation tasks
```

**UPDATE TO**:
```markdown
- **Agent System**: 93 defined AI agents with 8 containerized services for automation tasks
- **Service Mesh**: Kong API Gateway with Consul service discovery and circuit breaking
- **Message Broker**: RabbitMQ for async agent coordination and task distribution
```

### Section 2: Architecture Overview - Port Allocation
**ADD AFTER LINE 106**:
```markdown
**Infrastructure Services (10000-10099)**
- 10000: PostgreSQL database
- 10001: Redis cache  
- 10002-10003: Neo4j graph database (bolt/http)
- 10005: Kong API Gateway (proxy)        # ← ADD THIS
- 10006: Consul service discovery        # ← ADD THIS
- 10007-10008: RabbitMQ (amqp/mgmt)      # ← ADD THIS
- 10010: FastAPI backend
- 10011: Streamlit frontend
- 10015: Kong Admin API                  # ← ADD THIS
```

### Section 3: Container Architecture
**CURRENT** (Line 317):
```markdown
The system runs 25 operational containers organized in tiers:
```

**UPDATE TO**:
```markdown
The system runs 33+ operational containers organized in tiers:
```

**UPDATE TIER 4** (Line 338):
```markdown
**Tier 4: Monitoring Stack (9 containers)**
- Prometheus, Grafana, Loki, AlertManager
- Node Exporter, cAdvisor, Blackbox Exporter
- PostgreSQL Exporter, Redis Exporter
```

### Section 4: API Documentation
**UPDATE** (Line 344):
```markdown
**Primary API Endpoints:**
- `POST /api/v1/chat/` - Chat with AI models
- `GET /api/v1/models/` - List available models
- `GET /api/v1/agents/` - Agent management and status
- `GET /api/v1/documents/` - Document management operations
- `GET /api/v1/system/` - System status and information
- `GET /api/v1/hardware/` - Hardware resource monitoring
- `GET /api/v1/cache/` - Cache management operations
- `GET /api/v1/cache-optimized/` - Optimized cache operations
- `GET /api/v1/circuit-breaker/` - Circuit breaker status
- `POST /api/v1/mesh/enqueue` - Legacy task queue operations
- `GET /api/v1/mesh/results` - Legacy task results
- `POST /api/v1/mesh/v2/register` - Service mesh registration  # ← ADD THIS
- `GET /api/v1/mesh/v2/services` - Service discovery         # ← ADD THIS
- `GET /api/v1/features/` - Feature flags and configuration
- `GET /health` - System health status
- `GET /metrics` - Prometheus metrics
```

### Section 5: New Section - Service Mesh Architecture
**ADD AFTER MCP Server Management**:
```markdown
## Service Mesh Architecture

The system implements a comprehensive service mesh using Kong and Consul:

### Kong API Gateway (Ports 10005, 10015)
- Centralized API management and routing
- Rate limiting and authentication
- Request/response transformation
- Circuit breaking and retries
- Admin API for dynamic configuration

### Consul Service Discovery (Port 10006)
- Dynamic service registration
- Health checking and monitoring
- DNS and HTTP interfaces
- Key/value store for configuration
- Multi-datacenter support

### RabbitMQ Message Broker (Ports 10007-10008)
- Async agent communication
- Task queue management
- Event-driven architecture
- Message persistence and reliability
- Management UI for monitoring

### Service Mesh Implementation
- Legacy mesh API (`/api/v1/mesh/`) - Redis-based task queue
- Service mesh v2 (`/api/v1/mesh/v2/`) - Full service discovery and registration
```

---

## AGENTS.md - REQUIRED UPDATES

### Section 1: System Architecture Overview
**CURRENT** (Line 65):
```markdown
SutazAI operates a **comprehensive multi-agent architecture** with enterprise-grade AI automation capabilities, featuring 50+ specialized agents organized in three distinct tiers
```

**UPDATE TO**:
```markdown
SutazAI operates a **comprehensive multi-agent architecture** with enterprise-grade AI automation capabilities, featuring:
- **93 total agents** defined in the agent registry
- **8 containerized agents** actively running in production
- **15-20 fully implemented agents** with production-ready code
- **Three-tier hierarchy** for agent organization and coordination
```

### Section 2: Add Agent Implementation Status
**ADD AFTER Agent Hierarchy**:
```markdown
### Agent Implementation Status

| Category | Defined | Containerized | Implemented | Status |
|----------|---------|---------------|-------------|---------|
| Ultra Lead Architects | 2 | 2 | 2 | ✅ Operational |
| Core Operational | 10 | 6 | 8 | ✅ Operational |
| Domain Specialists | 15 | 0 | 10 | ⚠️ Partial |
| Task Executors | 20 | 0 | 0 | 📋 Planned |
| Monitoring Agents | 15 | 0 | 0 | 📋 Planned |
| Support Utilities | 31 | 0 | 0 | 📋 Planned |
| **TOTAL** | **93** | **8** | **20** | - |
```

### Section 3: Update Agent Port Allocations
**ADD Port Information**:
```markdown
### Containerized Agent Services

| Agent Name | Port | Status | Implementation |
|------------|------|---------|----------------|
| ollama-integration | 11071 | 🔴 Defined but not running | Full |
| hardware-resource-optimizer | 11019 | 🔴 Defined but not running | Full |
| task-assignment-coordinator | 11069 | 🔴 Defined but not running | Full |
| ultra-system-architect | 11200 | ✅ Running | Full |
| ultra-frontend-ui-architect | 11201 | 🔴 Defined but not running | Full |
```

### Section 4: Add Communication Architecture
**ADD NEW SECTION**:
```markdown
## Agent Communication Architecture

### Message Broker Integration
All agents communicate through RabbitMQ message broker:
- **Queue Management**: Dedicated queues per agent type
- **Topic Exchange**: Event-driven communication patterns
- **Priority Queues**: Task prioritization support
- **Dead Letter Handling**: Failed message recovery

### Agent Coordination Patterns
1. **Direct Communication**: HTTP REST between agents
2. **Async Messaging**: RabbitMQ for task distribution
3. **Service Discovery**: Consul for dynamic agent discovery
4. **Health Monitoring**: Standardized health check endpoints
5. **Circuit Breaking**: Fault tolerance for agent failures
```

### Section 5: Clarify 500-Agent Claims
**ADD CLARIFICATION**:
```markdown
## Agent Scaling Architecture

### Current vs Future State
- **Current State**: 8 containerized agents, 93 defined
- **Scaling Capability**: Architecture supports 500+ agents
- **Ultra-System-Architect**: Designed to coordinate up to 500 agents
- **Note**: References to "500 agents" indicate architectural capacity, not current deployment
```

---

## IMPLEMENTATION CHECKLIST

### Immediate Actions (Today)
- [ ] Update CLAUDE.md with service mesh documentation
- [ ] Correct agent counts in AGENTS.md
- [ ] Add Kong, Consul, RabbitMQ to port registry
- [ ] Document mesh v2 API endpoints
- [ ] Update container count to 33+

### Follow-up Actions (This Week)
- [ ] Create service dependency diagram
- [ ] Document agent health check endpoints
- [ ] Update monitoring dashboards
- [ ] Create agent capability matrix
- [ ] Document message queue patterns

### Validation Steps
1. Verify all port numbers against docker-compose.yml
2. Confirm agent counts against agent_registry.json
3. Test all documented API endpoints
4. Validate service health checks
5. Review with architecture team

---

**Priority**: URGENT - Documentation accuracy critical for team operations
**Owner**: System Architecture Team
**Review Date**: 2025-08-17