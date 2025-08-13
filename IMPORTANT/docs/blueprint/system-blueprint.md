---
title: SutazAI System Architecture Blueprint
version: 1.0.0
last_updated: 2025-08-08
author: System Architecture Team
review_status: Official
next_review: 2025-09-07
related_docs:
  - /opt/sutazaiapp/CLAUDE.md
  - /opt/sutazaiapp/IMPORTANT/10_canonical/INDEX.md
  - /opt/sutazaiapp/IMPORTANT/SUTAZAI_PRD.md
---

# SutazAI System Architecture Blueprint

## Executive Overview

SutazAI is a Docker Compose-based AI orchestration platform designed for local deployment and execution of AI workloads using locally-hosted language models. This blueprint provides the authoritative technical architecture documentation, consolidating the actual system state, design principles, and evolution roadmap.

### Current Reality Assessment
- **Architecture Pattern**: Modular monolith transitioning to service-oriented architecture
- **Deployment Model**: Docker Compose orchestration (28 running containers, 31 additional defined)
- **AI Model**: TinyLlama 637MB via Ollama (not GPT-OSS as originally planned)
- **Maturity Level**: Proof of Concept with production-ready monitoring infrastructure
- **Primary Use Case**: Local AI inference and agent orchestration prototype

### Key Architectural Decisions
- Local-first deployment using Docker Compose
- Ollama for LLM management with TinyLlama as default model
- PostgreSQL as primary datastore with Redis caching layer
- Prometheus/Grafana stack for comprehensive observability
- Flask-based agent stubs awaiting implementation

## System Architecture Principles

Based on Architecture Decision Records (ADRs) in `/opt/sutazaiapp/IMPORTANT/10_canonical/standards/`:

### 1. Data Consistency Principle
**ADR-0001**: Use UUID primary keys everywhere for global uniqueness and distributed system readiness
- All entities use UUID v4 with `gen_random_uuid()`
- Foreign keys properly indexed for performance
- Enables future sharding and replication strategies

### 2. Local-First AI Principle
**ADR-0002**: Consolidate LLM operations through Ollama
- No external API dependencies for core AI functionality
- TinyLlama as default model (637MB, efficient for PoC)
- Support for model switching without code changes

### 3. Observability-First Principle
**ADR-0003**: Comprehensive monitoring from day one
- Every service exposes Prometheus metrics
- Structured logging via Loki
- Distributed tracing preparation (Jaeger defined but not deployed)

### 4. Evolutionary Architecture Principle
**ADR-0004**: Design for incremental improvement
- Start with monolith, evolve to services
- Feature flags for gradual rollout
- Backward compatibility for all changes

## Component Inventory

### Core Infrastructure (Running - Verified Healthy)

| Component | Container Name | Port(s) | Purpose | Status | Dependencies |
|-----------|---------------|---------|---------|--------|--------------|
| PostgreSQL 16.3 | sutazai-postgres | 10000 | Primary datastore | ✅ HEALTHY | None |
| Redis 7 | sutazai-redis | 10001 | Cache & session store | ✅ HEALTHY | None |
| Neo4j 5 | sutazai-neo4j | 10002-10003 | Graph database | ✅ HEALTHY | None |
| Ollama | sutazai-ollama | 10104 (11434 internal) | LLM server | ✅ HEALTHY | None |

### Application Layer (Running)

| Component | Container Name | Port | Purpose | Status | Dependencies |
|-----------|---------------|------|---------|--------|--------------|
| Backend API | sutazai-backend | 10010 | FastAPI v17.0.0 | ✅ HEALTHY | PostgreSQL, Redis, Ollama |
| Frontend UI | sutazai-frontend | 10011 | Streamlit interface | ⚠️ STARTING | Backend API |

### Service Mesh Infrastructure (Running -   Configuration)

| Component | Container Name | Port(s) | Purpose | Status | Configuration |
|-----------|---------------|---------|---------|--------|---------------|
| Kong Gateway | sutazai-kong | 10005, 8001 | API gateway | ✅ RUNNING | No routes configured |
| Consul | sutazai-consul | 10006 | Service discovery | ✅ RUNNING |   usage |
| RabbitMQ | sutazai-rabbitmq | 10007-10008 | Message queue | ✅ RUNNING | Not actively used |

### Vector Databases (Running - Not Integrated)

| Component | Container Name | Port(s) | Purpose | Status | Integration |
|-----------|---------------|---------|---------|--------|-------------|
| Qdrant | sutazai-qdrant | 10101-10102 | Vector search | ✅ HEALTHY | Not integrated |
| FAISS | sutazai-faiss | 10103 | Vector similarity | ✅ HEALTHY | Not integrated |
| ChromaDB | sutazai-chromadb | 10100 | Embeddings store | ⚠️ ISSUES | Connection problems |

### Monitoring Stack (Running - Fully Operational)

| Component | Container Name | Port | Purpose | Status | Coverage |
|-----------|---------------|------|---------|--------|----------|
| Prometheus | sutazai-prometheus | 10200 | Metrics collection | ✅ HEALTHY | All services |
| Grafana | sutazai-grafana | 10201 | Visualization | ✅ HEALTHY | 3 dashboards |
| Loki | sutazai-loki | 10202 | Log aggregation | ✅ HEALTHY | All containers |
| AlertManager | sutazai-alertmanager | 10203 | Alert routing | ✅ HEALTHY | Basic rules |
| Node Exporter | sutazai-node-exporter | 10220 | System metrics | ✅ HEALTHY | Host metrics |
| cAdvisor | sutazai-cadvisor | 10221 | Container metrics | ✅ HEALTHY | All containers |

### Agent Services (Running - Stub Implementations)

| Agent | Container Name | Port | Actual Functionality | Owner | Priority |
|-------|---------------|------|---------------------|-------|----------|
| AI Agent Orchestrator | sutazai-ai-agent-orchestrator | 8589 | Health endpoint only | Core Team | P0 |
| Multi-Agent Coordinator | sutazai-multi-agent-coordinator | 8587 | Coordination stub | Core Team | P0 |
| Resource Arbitration | sutazai-resource-arbitration | 8588 | Resource stub | Platform Team | P1 |
| Task Assignment | sutazai-task-assignment | 8551 | Task routing stub | Core Team | P0 |
| Hardware Optimizer | sutazai-hardware-optimizer | 8002 | Hardware stub | Platform Team | P2 |
| Ollama Integration | sutazai-ollama-integration | 11015 | Ollama wrapper | AI Team | P0 |
| AI Metrics Exporter | sutazai-ai-metrics-exporter | 11063 | Metrics stub | Platform Team | P1 |

### Services Defined but Not Running (31 Total)

Categories of non-running services in docker-compose.yml:
- **Security Services**: HashiCorp Vault, OAuth2 Proxy
- **Advanced Monitoring**: Jaeger, Elasticsearch, Kibana
- **ML Infrastructure**: MLflow, Kubeflow components
- **Additional Agents**: 24 specialized AI agents
- **Development Tools**: Jupyter, Code Server

## Service Boundaries and Domain Model

### Domain-Driven Design Context Map

```
┌─────────────────────────────────────────────────────────────┐
│                     CORE DOMAIN                              │
│  ┌─────────────────┐        ┌──────────────────┐           │
│  │  Orchestrator   │◄──────►│   RAG Engine     │           │
│  │   (Backend)     │        │  (Vector DBs)    │           │
│  └─────────────────┘        └──────────────────┘           │
│           ▲                          ▲                       │
│           │                          │                       │
│  ┌─────────────────┐        ┌──────────────────┐           │
│  │  Agent Registry │        │  Task Manager    │           │
│  │   (PostgreSQL)  │        │   (PostgreSQL)   │           │
│  └─────────────────┘        └──────────────────┘           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                   SUPPORTING DOMAIN                          │
│  ┌─────────────────┐        ┌──────────────────┐           │
│  │  Authentication │        │  Documentation   │           │
│  │     (JWT)       │        │   (Frontend)     │           │
│  └─────────────────┘        └──────────────────┘           │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│                    GENERIC DOMAIN                            │
│  ┌─────────────────┐        ┌──────────────────┐           │
│  │  Observability  │        │   API Gateway    │           │
│  │  (Monitoring)   │        │     (Kong)       │           │
│  └─────────────────┘        └──────────────────┘           │
└─────────────────────────────────────────────────────────────┘
```

### Service Ownership Matrix

| Service Domain | Owner Team | Responsibilities | SLA Target |
|----------------|------------|------------------|------------|
| Core Orchestration | Core Team | Agent coordination, task routing | 99.9% |
| AI Infrastructure | AI Team | LLM management, embeddings | 99.5% |
| Platform Services | Platform Team | Databases, caching, messaging | 99.9% |
| Observability | Platform Team | Monitoring, logging, alerting | 99.95% |
| Frontend | Product Team | UI/UX, user workflows | 99.0% |
| Agent Implementation | Feature Teams | Business logic per agent | 95.0% |

## Integration Architecture

### Communication Patterns

```yaml
Synchronous (HTTP/REST):
  - Frontend → Backend API (port 10010)
  - Backend → Ollama (port 10104)
  - Backend → Agent Services (ports 8xxx)
  - Monitoring → All services (/metrics endpoints)

Asynchronous (Message Queue):
  - RabbitMQ available (port 10007) but not actively used
  - Future: Agent → Agent communication via RabbitMQ

Database Connections:
  - Backend → PostgreSQL (port 10000)
  - Backend → Redis (port 10001)
  - Future: Backend → Neo4j (port 10003) for graph operations
  - Future: Backend → Vector DBs for RAG

Service Discovery:
  - Consul available (port 10006) but not integrated
  - Currently using Docker DNS for service resolution
```

### API Contract Standards

Based on `/opt/sutazaiapp/IMPORTANT/10_canonical/api_contracts/contracts.md`:

```yaml
API Versioning:
  - Pattern: /api/v1/{resource}
  - Backward compatibility required
  - Deprecation notice: 2 releases minimum

Request Standards:
  - Pagination: Cursor-based with limit parameter
  - Idempotency: Required for all POST operations
  - Authentication: Bearer JWT tokens

Response Standards:
  - Format: JSON with consistent structure
  - Errors: {code, message, details}
  - Status Codes: Standard HTTP semantics

Key Endpoints:
  - POST /api/v1/generate - LLM text generation
  - POST /api/v1/documents - Document ingestion
  - GET /api/v1/query - RAG query interface
  - POST /api/v1/agents/execute - Agent execution
  - GET /api/v1/agents/list - Agent discovery
```

## Data Architecture

### Primary Data Stores

```yaml
PostgreSQL (Transactional Data):
  Tables:
    - users: System users and authentication
    - agents: Agent registry and capabilities
    - tasks: Task queue and execution history
    - documents: Document metadata
    - audit_logs: System audit trail
  
  Schema Standards:
    - UUID primary keys (gen_random_uuid())
    - Timestamp columns: created_at, updated_at
    - Soft deletes with deleted_at
    - Foreign key constraints enforced

Redis (Caching & Sessions):
  Use Cases:
    - Session storage (TTL: 24 hours)
    - API response caching (TTL: 5 minutes)
    - Rate limiting counters
    - Distributed locks

Neo4j (Graph Relationships):
  Planned Use Cases:
    - Agent dependency graphs
    - Task workflow definitions
    - Knowledge graph for RAG
  Status: Deployed but not integrated

Vector Databases (Embeddings):
  ChromaDB: Document embeddings (connection issues)
  Qdrant: Alternative vector store (not integrated)
  FAISS: Local vector search (not integrated)
```

### Data Flow Architecture

```
User Request → Frontend → Backend API → Service Layer
                                ↓
                          Data Access Layer
                    ↙         ↓           ↘
              PostgreSQL    Redis    Vector DBs
                    ↘         ↓           ↙
                          Response Layer
                                ↓
                          Backend API → Frontend → User
```

## Security Architecture

### Security Layers

Based on STRIDE threat model from `/opt/sutazaiapp/IMPORTANT/10_canonical/security/security_privacy.md`:

```yaml
Authentication & Authorization:
  - JWT tokens with 24-hour expiry
  - RBAC with scopes per endpoint
  - Service-to-service tokens for internal calls
  - No hardcoded credentials (use .env files)

Network Security:
  - All services internal to Docker network
  - Kong Gateway as single ingress point
  - TLS termination at gateway (when configured)
  - No direct external access to databases

Container Security:
  - Non-root containers where possible
  - Read-only root filesystems (planned)
  - Resource limits defined
  - Security scanning in CI/CD (planned)

Data Security:
  - Encryption in transit (TLS at gateway)
  - Encryption at rest (disk encryption)
  - PII handling compliance (GDPR considerations)
  - Audit logging for all state changes

Secrets Management:
  - Environment variables via .env files
  - No secrets in code or configuration
  - Rotation schedule: Quarterly
  - Future: HashiCorp Vault integration
```

### Security Checklist

- [x] JWT authentication implemented
- [x] No hardcoded secrets in codebase
- [x] Audit logging configured
- [x] Network isolation via Docker
- [ ] TLS configuration at gateway
- [ ] Container security scanning
- [ ] Secrets rotation automation
- [ ] Penetration testing

## Operational Architecture

### Deployment Topology

```yaml
Current State (Docker Compose):
  Host Requirements:
    - CPU: 8+ cores recommended
    - RAM: 16GB minimum, 32GB recommended
    - Disk: 100GB SSD minimum
    - OS: Linux (Ubuntu 22.04 LTS preferred)
  
  Network Configuration:
    - Custom bridge network: sutazai-network
    - Internal DNS via Docker
    - Port exposure via host binding

Target State (Kubernetes):
  Cluster Configuration:
    - 3 master nodes (HA control plane)
    - 5+ worker nodes
    - CNI: Calico or Cilium
    - CSI: Local volumes or Rook/Ceph
  
  Namespace Strategy:
    - sutazai-core: Core services
    - sutazai-agents: Agent deployments
    - sutazai-monitoring: Observability stack
    - sutazai-data: Databases and caches
```

### Monitoring & Observability

```yaml
Metrics Collection (Prometheus):
  Scrape Interval: 15 seconds
  Retention: 30 days
  Key Metrics:
    - Request rate, error rate, duration (RED)
    - CPU, memory, disk, network (USE)
    - Business metrics (tasks processed, documents indexed)

Log Aggregation (Loki):
  Retention: 7 days
  Log Levels: DEBUG, INFO, WARN, ERROR, FATAL
  Structured Logging: JSON format required
  Correlation: Request ID propagation

Visualization (Grafana):
  Dashboards:
    - System Overview (CPU, memory, disk)
    - Application Performance (latency, throughput)
    - Business Metrics (usage, success rates)
  
Alerting (AlertManager):
  Severity Levels: Critical, Warning, Info
  Routing: Email, Slack (when configured)
  Key Alerts:
    - Service down > 5 minutes
    - Error rate > 5%
    - Disk usage > 80%
    - Memory usage > 90%
```

### Disaster Recovery

```yaml
Backup Strategy:
  PostgreSQL:
    - Daily automated backups
    - Point-in-time recovery capability
    - Retention: 30 days
  
  Configuration:
    - Git repository for all config
    - Environment files backed up separately
    - Container images in registry

Recovery Objectives:
  - RTO (Recovery Time Objective): 4 hours
  - RPO (Recovery Point Objective): 24 hours
  - MTTR (Mean Time To Recovery): 2 hours

Failure Scenarios:
  1. Container Failure:
     - Auto-restart via Docker
     - Health checks for detection
     - Recovery: < 1 minute
  
  2. Host Failure:
     - Manual migration required
     - Recovery: 2-4 hours
  
  3. Data Corruption:
     - Restore from backup
     - Recovery: 4-8 hours
  
  4. Complete System Loss:
     - Rebuild from git + backups
     - Recovery: 8-24 hours
```

## Cross-Cutting Concerns

### Logging Strategy

```yaml
Standards:
  - Format: Structured JSON
  - Fields: timestamp, level, service, message, request_id, user_id
  - Levels: TRACE, DEBUG, INFO, WARN, ERROR, FATAL
  
Implementation:
  - Python: structlog library
  - Node.js: winston or pino
  - Correlation: X-Request-ID header propagation
  
Storage:
  - Stdout/stderr to Docker logs
  - Loki for aggregation
  - 7-day retention minimum
```

### Error Handling

```yaml
Client Errors (4xx):
  - 400: Bad Request - validation errors
  - 401: Unauthorized - missing/invalid auth
  - 403: Forbidden - insufficient permissions
  - 404: Not Found - resource doesn't exist
  - 429: Too Many Requests - rate limit exceeded

Server Errors (5xx):
  - 500: Internal Server Error - catch-all
  - 502: Bad Gateway - upstream service error
  - 503: Service Unavailable - maintenance/overload
  - 504: Gateway Timeout - upstream timeout

Error Response Format:
  {
    "error": {
      "code": "VALIDATION_ERROR",
      "message": "User-friendly message",
      "details": {
        "field": "specific error details"
      },
      "request_id": "uuid-for-tracking"
    }
  }
```

### Performance Requirements

```yaml
Latency Targets:
  - API Response: p50 < 100ms, p99 < 1s
  - LLM Generation: p50 < 2s, p99 < 10s
  - Database Queries: p50 < 10ms, p99 < 100ms
  
Throughput Targets:
  - API Requests: 1000 req/s
  - Concurrent Users: 100
  - Document Processing: 10 docs/minute
  
Resource Limits:
  - CPU per container: 2 cores max
  - Memory per container: 4GB max
  - Disk I/O: 100 MB/s max
```

## Evolution Roadmap

### Phase 1: Foundation Stabilization (Current)
**Timeline**: Q1 2025
**Status**: In Progress

- [x] Docker Compose deployment
- [x] Basic monitoring setup
- [x] PostgreSQL schema creation
- [ ] Fix model configuration (TinyLlama vs GPT-OSS)
- [ ] Implement one real agent
- [ ] Fix ChromaDB integration
- [ ] Configure Kong routing

### Phase 2: Core Functionality (Next)
**Timeline**: Q2 2025
**Status**: Planned

- [ ] Implement RAG pipeline
- [ ] Agent orchestration logic
- [ ] Task queue processing
- [ ] Vector database integration
- [ ] Basic UI workflows
- [ ] Authentication/authorization

### Phase 3: Production Readiness
**Timeline**: Q3 2025
**Status**: Future

- [ ] Kubernetes migration
- [ ] Horizontal scaling
- [ ] Advanced monitoring
- [ ] Security hardening
- [ ] Performance optimization
- [ ] Disaster recovery testing

### Phase 4: Advanced Features
**Timeline**: Q4 2025
**Status**: Future

- [ ] Multi-model support
- [ ] Complex agent workflows
- [ ] Graph-based orchestration
- [ ] Advanced RAG features
- [ ] Plugin architecture
- [ ] External integrations

## Governance Model

### Architecture Review Process

```yaml
Change Categories:
  Minor: Bug fixes, dependency updates
    - Review: Tech lead approval
    - Timeline: Same day
  
  Medium: New features, API changes
    - Review: Architecture team
    - Timeline: 2-3 days
  
  Major: New services, breaking changes
    - Review: Architecture board
    - Timeline: 1 week
    - Requires: ADR documentation

Review Criteria:
  - Alignment with principles
  - Security implications
  - Performance impact
  - Operational complexity
  - Technical debt assessment
```

### Technical Debt Management

```yaml
Debt Categories:
  Critical: Security vulnerabilities, data loss risk
    - Resolution: Immediate
  
  High: Performance issues, stability problems
    - Resolution: Current sprint
  
  Medium: Code quality, missing tests
    - Resolution: Next quarter
  
  Low: Nice-to-have improvements
    - Resolution: As capacity allows

Tracking:
  - Location: GitHub Issues with 'tech-debt' label
  - Review: Monthly architecture meeting
  - Budget: 20% of sprint capacity
```

### Change Management

```yaml
Process:
  1. Proposal: GitHub issue or RFC
  2. Review: Architecture team assessment
  3. Approval: Based on change category
  4. Implementation: Feature branch
  5. Testing: Automated + manual verification
  6. Deployment: Staged rollout
  7. Monitoring: 24-hour observation period

Documentation Requirements:
  - Update relevant documentation
  - Create/update ADR if needed
  - Update this blueprint if architecture changes
  - Announce in team channels
```

## Appendices

### A. Port Registry

```yaml
# Core Services
10000: PostgreSQL database
10001: Redis cache
10002: Neo4j browser
10003: Neo4j bolt
10005: Kong API Gateway
10006: Consul service discovery
10007: RabbitMQ AMQP
10008: RabbitMQ management
10010: Backend FastAPI
10011: Frontend Streamlit
10104: Ollama LLM server

# Vector Databases
10100: ChromaDB
10101: Qdrant HTTP
10102: Qdrant gRPC
10103: FAISS service

# Monitoring
10200: Prometheus
10201: Grafana
10202: Loki
10203: AlertManager
10220: Node Exporter
10221: cAdvisor

# Agent Services
8002: Hardware Optimizer
8551: Task Assignment
8587: Multi-Agent Coordinator
8588: Resource Arbitration
8589: AI Agent Orchestrator
11015: Ollama Integration
11063: AI Metrics Exporter
```

### B. Environment Variables

```bash
# Database
POSTGRES_HOST=sutazai-postgres
POSTGRES_PORT=10000
POSTGRES_DB=sutazai
POSTGRES_USER=sutazai
POSTGRES_PASSWORD=<secure>

# Redis
REDIS_HOST=sutazai-redis
REDIS_PORT=10001

# Ollama
OLLAMA_HOST=sutazai-ollama
OLLAMA_PORT=10104
OLLAMA_MODEL=tinyllama

# Security
JWT_SECRET=<secure>
JWT_EXPIRY=86400

# Monitoring
PROMETHEUS_PORT=10200
GRAFANA_PORT=10201
```

### C. Key Commands

```bash
# System Management
docker-compose up -d                    # Start all services
docker-compose ps                       # Check status
docker-compose logs -f [service]        # View logs
docker-compose restart [service]        # Restart service
docker-compose down                     # Stop all

# Health Checks
curl http://localhost:10010/health      # Backend health
curl http://localhost:10104/api/tags    # Ollama models
curl http://localhost:8589/health       # Agent health

# Database Access
docker exec -it sutazai-postgres psql -U sutazai -d sutazai
docker exec -it sutazai-redis redis-cli

# Monitoring Access
http://localhost:10200                  # Prometheus
http://localhost:10201                  # Grafana (admin/admin)
http://localhost:10002                  # Neo4j Browser
```

### D. References

- [CLAUDE.md](/opt/sutazaiapp/CLAUDE.md) - System reality documentation
- [Canonical Architecture](/opt/sutazaiapp/IMPORTANT/10_canonical/INDEX.md) - Source of truth
- [Product Requirements](/opt/sutazaiapp/IMPORTANT/SUTAZAI_PRD.md) - Business requirements
- [Engineering Standards](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/engineering_standards.md)
- [API Contracts](/opt/sutazaiapp/IMPORTANT/10_canonical/api_contracts/contracts.md)
- [Security & Privacy](/opt/sutazaiapp/IMPORTANT/10_canonical/security/security_privacy.md)

---

## Document Control

**Version**: 1.0.0
**Status**: Official
**Owner**: System Architecture Team
**Review Cycle**: Monthly
**Last Review**: 2025-08-08
**Next Review**: 2025-09-07

### Change Log
| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0.0 | 2025-08-08 | Architecture Team | Initial comprehensive blueprint |

### Approval
- Technical Lead: Pending
- Architecture Board: Pending
- Engineering Manager: Pending

---

*This document represents the authoritative system architecture for SutazAI. All development must align with the principles, standards, and patterns defined herein.*