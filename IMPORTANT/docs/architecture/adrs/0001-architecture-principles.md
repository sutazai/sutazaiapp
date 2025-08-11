---
title: Core Architecture Principles and Patterns
version: 2.0.0
last_updated: 2025-08-08
author: System Architect
review_status: Accepted
next_review: 2025-09-07
related_docs:
  - IMPORTANT/10_canonical/standards/ADR-0001.md
  - IMPORTANT/10_canonical/standards/engineering_standards.md
  - IMPORTANT/10_canonical/standards/codebase_rules.md
  - IMPORTANT/10_canonical/target_state/mvp_architecture.md
---

# ADR-0001: Core Architecture Principles and Patterns

## Status
**Accepted**

## Date
2025-08-08

## Context
SutazAI is an ambitious AI orchestration platform that started with complex goals but needs pragmatic architectural decisions to deliver value. After extensive cleanup (v56 removed 200+ conceptual documentation files), we need clear architectural principles that:

- Guide development toward realistic, working implementations
- Prevent regression to over-engineered conceptual features
- Support incremental development from PoC to production
- Align with the reality of what's actually deployed (28 containers running, 7 agent stubs)
- Enable future growth without requiring complete rewrites

Current challenges include:
- Mismatch between documentation promises and actual implementation
- Agent services that are stubs returning hardcoded responses
- Database schema that exists but lacks proper structure (PostgreSQL has tables but needs migrations)
- Service mesh components (Kong, Consul, RabbitMQ) running but not integrated
- No clear boundaries between components leading to potential coupling

## Decision Drivers
- [x] Maintainability - Code must be understandable and modifiable
- [x] Scalability needs - Support growth from PoC to production loads
- [x] Security constraints - Local-first execution, no external dependencies
- [x] Cost considerations - Optimize resource usage for containerized deployment
- [x] Team expertise - Align with Python/FastAPI/Docker skillsets
- [x] Time to market - Deliver working features incrementally
- [x] Integration requirements - Multiple databases and services must interoperate
- [x] Compliance requirements - Follow CLAUDE.md rules strictly

## Considered Options

### Option 1: Monolithic Architecture
**Description:** Single application containing all business logic
**Pros:**
- Simple deployment and debugging
- No network overhead between components
- Easy transaction management

**Cons:**
- Difficult to scale individual components
- Technology lock-in
- Harder to maintain as complexity grows
- Conflicts with existing multi-container setup

### Option 2: Full Microservices Architecture
**Description:** Highly distributed system with fine-grained services
**Pros:**
- Independent scaling and deployment
- Technology diversity possible
- Fault isolation

**Cons:**
- Complex orchestration required
- Network latency and reliability concerns
- Difficult distributed transaction management
- Over-engineering for current needs

### Option 3: Modular Microservices with Clear Boundaries
**Description:** Service-oriented architecture with well-defined module boundaries, event-driven communication, and pragmatic service sizing
**Pros:**
- Balanced complexity vs functionality
- Clear separation of concerns
- Incremental migration path
- Aligns with current Docker Compose setup

**Cons:**
- Still requires service coordination
- Some network overhead
- Need to define clear contracts

## Decision

We adopt **Option 3: Modular Microservices with Clear Boundaries** with the following specific principles:

### 1. UUID Primary Keys Everywhere
- **Decision:** All database tables use UUID primary keys with `gen_random_uuid()`
- **Rationale:** Enables distributed ID generation, data portability, and service autonomy
- **Implementation:** PostgreSQL native UUID support, indexed foreign keys
- **Reference:** IMPORTANT/10_canonical/standards/ADR-0001.md

### 2. Service Boundaries Based on Business Capabilities
- **Core Services:**
  - Backend API (FastAPI) - Central orchestration and API gateway
  - Agent Services - Specialized AI processing (currently stubs, to be implemented)
  - Frontend (Streamlit) - User interface
- **Data Services:**
  - PostgreSQL - Transactional data and system state
  - Redis - Caching and session management
  - Neo4j - Graph relationships between entities
  - Vector DBs (Qdrant, FAISS, ChromaDB) - Similarity search
- **Infrastructure Services:**
  - Kong - API gateway (to be configured)
  - Consul - Service discovery (to be utilized)
  - RabbitMQ - Async messaging (to be integrated)

### 3. Event-Driven Architecture (Future State)
- **Current:** Direct HTTP calls between services
- **Target:** Asynchronous message passing via RabbitMQ
- **Migration:** Implement incrementally, starting with non-critical flows

### 4. Local-First Processing
- **Decision:** All AI/ML operations use local models via Ollama
- **No external API dependencies** for core functionality
- **Reference:** IMPORTANT/10_canonical/standards/ADR-0002.md

### 5. Containerization as First-Class Citizen
- **Every service runs in Docker containers**
- **Docker Compose for local development and testing**
- **Multi-architecture support (amd64/arm64)**
- **Reference:** IMPORTANT/10_canonical/standards/ADR-0003.md

### 6. Database-Per-Service Pattern (Logical)
- Services own their data schemas
- Shared PostgreSQL instance acceptable in PoC/MVP
- Future: Physical separation as services mature

### 7. API-First Design
- All service interactions through well-defined APIs
- OpenAPI/Swagger documentation mandatory
- Version APIs from day one

### 8. Observability Built-In
- Prometheus metrics for all services
- Structured logging to Loki
- Distributed tracing ready (Jaeger placeholder)
- Health checks mandatory

### 9. Security by Design
- Authentication/authorization at API gateway
- Service-to-service authentication (mTLS future)
- Secrets management (currently environment variables, Vault future)
- Vulnerability scanning as release gate
- **Reference:** IMPORTANT/10_canonical/standards/ADR-0004.md

### 10. Progressive Enhancement
- Start with working stubs
- Implement real functionality incrementally
- Feature flags for gradual rollout
- No "big bang" replacements

## Consequences

### Positive Consequences
- Clear separation of concerns enables parallel development
- Services can be scaled independently based on load
- Technology choices can be optimized per service
- Failures are isolated to service boundaries
- Aligns with existing container infrastructure
- Supports incremental migration from stubs to real implementations
- UUID keys enable distributed development without coordination

### Negative Consequences
- Network calls between services add latency
- Distributed system complexity (eventual consistency, partial failures)
- Need for service discovery and coordination
- More complex deployment and monitoring
- Initial overhead in setting up service contracts
- UUID keys require more storage space than integers

### Neutral Consequences
- Requires investment in DevOps tooling and practices
- Team needs to understand distributed systems concepts
- More components to monitor and maintain
- Trade-off between autonomy and consistency

## Implementation Plan

1. **Phase 1: Foundation (Current State)**
   - ✅ Container infrastructure established
   - ✅ Core services running (even if stubs)
   - ✅ Monitoring stack operational
   - 🔄 Fix database schemas to use UUIDs consistently

2. **Phase 2: Service Contracts (Next 2 weeks)**
   - Define OpenAPI specs for all services
   - Implement health checks consistently
   - Document service boundaries and responsibilities
   - Configure Kong API gateway routes

3. **Phase 3: First Real Implementation (Next month)**
   - Pick one agent service to implement fully
   - Establish patterns for service communication
   - Integrate with message queue for async operations
   - Implement proper error handling and retries

4. **Phase 4: Incremental Enhancement (Ongoing)**
   - Replace stubs with real implementations one by one
   - Add service mesh features (circuit breakers, retries)
   - Implement distributed tracing
   - Optimize based on observed patterns

## Validation
Success metrics:
- Service health checks all green
- API response times < 500ms for 95th percentile
- Zero single points of failure
- Deployment of new service version < 5 minutes
- Test coverage > 80% per service

Monitoring approach:
- Grafana dashboards for service metrics
- Alert on service degradation
- Weekly architecture review meetings

Review timeline:
- Monthly assessment of principle adherence
- Quarterly architecture review for adjustments

## References
- IMPORTANT/10_canonical/standards/ADR-0001.md (UUID Primary Keys)
- IMPORTANT/10_canonical/standards/ADR-0002.md (Local Execution)
- IMPORTANT/10_canonical/standards/ADR-0003.md (Multi-Architecture)
- IMPORTANT/10_canonical/standards/ADR-0004.md (Security Scanning)
- IMPORTANT/10_canonical/standards/ADR-0005.md (Documentation)
- IMPORTANT/10_canonical/target_state/mvp_architecture.md
- IMPORTANT/10_canonical/standards/engineering_standards.md
- CLAUDE.md (System truth document)
- Martin Fowler's Microservices articles
- The Twelve-Factor App methodology
- C4 Model for architecture documentation

## Change Log
- 2025-08-08: Comprehensive rewrite incorporating all architectural decisions from IMPORTANT/10_canonical/standards/
- 2025-08-08: Added UUID primary key decision from ADR-0001
- 2025-08-08: Aligned with post-v56 cleanup reality
- 2025-08-08: Added specific implementation phases