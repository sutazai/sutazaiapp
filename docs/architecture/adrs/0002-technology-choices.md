---
title: Technology Stack and Platform Decisions
version: 2.0.0
last_updated: 2025-08-08
author: System Architect
review_status: Accepted
next_review: 2025-09-07
related_docs:
  - IMPORTANT/10_canonical/standards/ADR-0002.md
  - IMPORTANT/10_canonical/standards/ADR-0003.md
  - IMPORTANT/TECHNOLOGY_STACK_REPOSITORY_INDEX.md
  - CLAUDE.md
---

# ADR-0002: Technology Stack and Platform Decisions

## Status
**Accepted**

## Date
2025-08-08

## Context
SutazAI requires a technology stack that balances several competing concerns:

1. **Local-First AI Processing**: Absolute requirement for 100% local execution with no external API dependencies
2. **Resource Constraints**: Must run efficiently on developer machines and modest server hardware
3. **Team Expertise**: Current team has Python/FastAPI/Docker experience
4. **Existing Infrastructure**: 28 containers already deployed with specific technology choices
5. **Reality vs Ambition**: Documentation claims GPT-OSS but TinyLlama (637MB) is what's actually loaded

The system currently has:
- **Working**: PostgreSQL, Redis, Neo4j, Ollama (with TinyLlama), FastAPI backend, Streamlit frontend
- **Running but not integrated**: Kong, Consul, RabbitMQ, vector databases
- **Stubs only**: 7 agent services returning hardcoded JSON

Technology decisions must support the transition from current PoC state to production-ready system without requiring complete rewrites.

## Decision Drivers
- [x] Performance requirements - Sub-second response times for UI interactions
- [x] Scalability needs - Support 100+ concurrent users in production
- [x] Security constraints - No external dependencies, local execution only
- [x] Cost considerations - Minimize infrastructure and licensing costs
- [x] Team expertise - Leverage existing Python/Docker skills
- [x] Time to market - Use proven, stable technologies
- [x] Maintainability - Choose technologies with good documentation and community
- [x] Integration requirements - Must support multiple databases and message queuing

## Considered Options

### Option 1: Full Python Stack
**Description:** Python for everything - FastAPI backend, Dash/Gradio frontend, Python agents
**Pros:**
- Single language reduces context switching
- Strong AI/ML ecosystem
- Team already knows Python
- Excellent library support

**Cons:**
- Python performance limitations for high-throughput scenarios
- GIL limitations for true parallelism
- Memory usage can be high
- Frontend options less mature than JavaScript alternatives

### Option 2: Polyglot Microservices
**Description:** Best tool for each job - Go for performance-critical services, Python for AI, TypeScript for frontend
**Pros:**
- Optimal performance per service
- Use each language's strengths
- Better frontend tooling with React/Vue

**Cons:**
- Increased complexity and maintenance burden
- Need expertise in multiple languages
- More complex build and deployment
- Harder debugging across language boundaries

### Option 3: Python-First with Strategic Exceptions
**Description:** Python as default, with specific exceptions for UI (Streamlit) and future performance-critical paths
**Pros:**
- Balanced complexity and capability
- Leverages team expertise
- Allows optimization where needed
- Aligns with current implementation

**Cons:**
- Still some language diversity
- Streamlit limitations for complex UIs
- May need frontend rewrite for production

## Decision

We adopt **Option 3: Python-First with Strategic Exceptions** with these specific technology choices:

### Core Language and Frameworks

#### Backend API Layer
- **Language**: Python 3.11+
- **Framework**: FastAPI v0.100+
- **Rationale**: 
  - Async support for high concurrency
  - Automatic OpenAPI documentation
  - Pydantic for data validation
  - Already implemented and working
- **Trade-offs**: Not as fast as Go/Rust, but fast enough for our needs

#### Frontend
- **Current**: Streamlit
- **Rationale**: Rapid prototyping, Python-native, good for MVP
- **Future Migration Path**: React/Next.js when UI complexity demands it
- **Trade-offs**: Limited customization, but perfect for current needs

#### Agent Services
- **Language**: Python 3.11+
- **Framework**: Flask (current) → FastAPI (target)
- **Rationale**: Consistency with backend, AI/ML library ecosystem
- **Migration**: Replace Flask stubs with FastAPI incrementally

### AI/ML Platform

#### Local LLM Execution
- **Platform**: Ollama
- **Default Model**: TinyLlama (637MB)
- **Rationale**:
  - 100% local execution requirement (per ADR-0002 in standards)
  - TinyLlama is resource-efficient and "good enough" for most tasks
  - Ollama provides consistent API across models
- **Configuration**:
  ```yaml
  models:
    default: tinyllama
    available:
      - tinyllama  # 637MB, fast, general purpose
      - codellama  # 3.8GB, code-specific tasks
      - llama2     # 3.8GB, better reasoning (future)
  ```
- **Trade-offs**: Lower capability than GPT-4, but ensures data privacy

#### Vector Databases
- **Primary**: Qdrant (already running on ports 10101/10102)
- **Alternatives**: FAISS (simpler), ChromaDB (having issues)
- **Rationale**: Native Python client, good performance, persistent storage
- **Use Cases**: RAG, similarity search, embeddings storage

### Data Layer

#### Primary Database
- **Technology**: PostgreSQL 15+
- **Extensions**: pgvector, uuid-ossp
- **Rationale**:
  - Robust, proven technology
  - UUID support for distributed systems
  - JSON support for flexible schemas
  - Already configured and running
- **Schema Strategy**: UUID primary keys everywhere (per ADR-0001)

#### Cache Layer
- **Technology**: Redis 7+
- **Use Cases**: Session storage, API caching, rate limiting
- **Rationale**: Simple, fast, well-understood

#### Graph Database
- **Technology**: Neo4j Community Edition
- **Use Cases**: Agent relationships, knowledge graphs
- **Rationale**: Best-in-class graph database, Cypher query language
- **Trade-offs**: Another query language to learn

#### Message Queue
- **Technology**: RabbitMQ 3.12+
- **Rationale**: Reliable, feature-rich, good Python support
- **Current State**: Running but not integrated
- **Target State**: Event backbone for agent communication

### Infrastructure and Deployment

#### Containerization
- **Technology**: Docker 24+ with Docker Compose
- **Multi-arch Support**: linux/amd64 and linux/arm64 (per ADR-0003)
- **Build Tool**: Docker Buildx for multi-platform images
- **Rationale**: 
  - Already implemented and working
  - Simplifies deployment
  - Good for development and small-scale production

#### Service Mesh (Light)
- **API Gateway**: Kong (already running on port 10005)
- **Service Discovery**: Consul (already running on port 10006)
- **Rationale**: Already deployed, provides growth path
- **Current**: Not configured
- **Target**: Basic routing and health checking

#### Orchestration
- **Current**: Docker Compose
- **Future**: Kubernetes (when scale demands it)
- **Rationale**: Simple for current needs, clear upgrade path

### Monitoring and Observability

#### Metrics
- **Technology**: Prometheus + Grafana
- **Rationale**: Industry standard, already deployed and working
- **Dashboards**: System metrics, API performance, agent health

#### Logging
- **Technology**: Loki + Promtail
- **Rationale**: Integrates with Grafana, lightweight
- **Format**: Structured JSON logging

#### Tracing (Future)
- **Technology**: Jaeger (when needed)
- **Rationale**: OpenTelemetry compatible, proven technology

### Development Tools

#### API Documentation
- **Technology**: OpenAPI/Swagger (auto-generated by FastAPI)
- **Rationale**: Industry standard, automatic from code

#### Testing
- **Unit Tests**: pytest with pytest-asyncio
- **Integration Tests**: pytest with testcontainers
- **API Tests**: httpx for async testing
- **Coverage Target**: 80% minimum

#### Code Quality
- **Linting**: flake8, pylint
- **Formatting**: black, isort
- **Type Checking**: mypy
- **Security**: bandit, safety
- **Pre-commit Hooks**: Enforce all checks

#### CI/CD
- **Current**: Docker Compose based deployment
- **Target**: GitHub Actions / GitLab CI
- **Security Gate**: Trivy scanning (per ADR-0004)

## Consequences

### Positive Consequences
- Leverages team's existing Python expertise
- All technologies are open source (no licensing costs)
- Proven, stable technology stack reduces risk
- Clear migration paths for components that need upgrading
- Local execution ensures data privacy and security
- Consistent technology reduces cognitive load

### Negative Consequences
- Python performance ceiling may require future optimization
- Streamlit limitations will eventually require frontend rewrite
- TinyLlama has lower capability than larger models
- Multiple databases increase operational complexity
- Need to maintain competency in multiple data stores

### Neutral Consequences
- Standard technology stack (nothing exotic to maintain)
- Good hiring pool for these technologies
- Extensive documentation and community support
- Some components (Kong, Consul) may be overkill for current scale

## Implementation Plan

1. **Immediate Actions (Week 1)**
   - Update backend configuration to use `tinyllama` instead of expecting `gpt-oss`
   - Document API endpoints using OpenAPI
   - Fix ChromaDB connection issues or officially deprecate

2. **Short Term (Weeks 2-4)**
   - Migrate agent services from Flask to FastAPI
   - Configure Kong API gateway with basic routes
   - Implement proper Redis caching for API responses
   - Set up structured logging with Loki

3. **Medium Term (Months 2-3)**
   - Integrate RabbitMQ for agent communication
   - Implement vector search with Qdrant
   - Add comprehensive test coverage
   - Configure Consul for service discovery

4. **Long Term (Months 4-6)**
   - Evaluate frontend alternatives if Streamlit becomes limiting
   - Consider adding specialized models to Ollama (codellama, llama2)
   - Implement distributed tracing with Jaeger
   - Plan Kubernetes migration if scale demands it

## Validation

### Success Metrics
- API response time < 200ms for 95th percentile
- Model inference time < 1 second for standard prompts
- System supports 100 concurrent users
- Zero external API calls during operation
- All services have > 80% test coverage

### Technology Health Checks
- Weekly review of technology pain points
- Monthly assessment of performance metrics
- Quarterly evaluation of technology choices
- Annual major technology review

## Migration Strategies

### If Python Performance Becomes Limiting
1. Profile and optimize Python code first
2. Consider PyPy or Cython for hot paths
3. Rewrite specific services in Go/Rust if needed
4. Keep AI/ML workloads in Python regardless

### If Streamlit Becomes Limiting
1. Start with Streamlit components for specific features
2. Build React components for complex interactions
3. Run both in parallel during transition
4. Complete migration when business value justifies it

### If TinyLlama Becomes Insufficient
1. Load additional models in Ollama as needed
2. Use model routing based on task complexity
3. Consider quantized versions of larger models
4. Maintain TinyLlama as fallback for resource-constrained environments

## References
- IMPORTANT/10_canonical/standards/ADR-0002.md (Local Execution)
- IMPORTANT/10_canonical/standards/ADR-0003.md (Multi-Architecture)
- IMPORTANT/10_canonical/standards/ADR-0004.md (Security Scanning)
- CLAUDE.md (System truth - current deployment state)
- FastAPI Documentation: https://fastapi.tiangolo.com/
- Ollama Documentation: https://ollama.ai/docs
- Docker Compose Specification: https://docs.docker.com/compose/
- PostgreSQL UUID Documentation: https://www.postgresql.org/docs/current/datatype-uuid.html
- The Twelve-Factor App: https://12factor.net/

## Change Log
- 2025-08-08: Complete rewrite based on actual system state and IMPORTANT/10_canonical/standards/
- 2025-08-08: Acknowledged TinyLlama as current reality vs GPT-OSS fiction
- 2025-08-08: Added specific migration strategies
- 2025-08-08: Aligned with post-v56 cleanup state