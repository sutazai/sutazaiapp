# SUTAZAI System - Epics and User Stories

## Executive Summary
This document contains the complete product backlog for the SUTAZAI system, organized into epics and user stories based on the comprehensive architecture analysis. Stories are prioritized as P0 (Critical Blockers), P1 (Core Functionality), and P2 (Technical Debt & Cleanup).

## Priority Definitions
- **P0 - Critical Blockers**: System cannot function without these. Must be fixed immediately.
- **P1 - Core Functionality**: Essential features for basic operation. Sprint 1 priority.
- **P2 - Technical Debt**: Cleanup and optimization. Can be deferred but should be scheduled.

---

## Epic 1: Model & LLM Alignment
**Goal**: Fix the model mismatch between backend expectations and loaded models

### Story 1.1: Fix TinyLlama Configuration Mismatch [P0]
**As a** System Administrator  
**I want** the backend to correctly reference the loaded TinyLlama model  
**So that** LLM operations work without connection errors

**Acceptance Criteria:**
- Backend configuration uses "tinyllama" instead of "gpt-oss"
- All references to non-existent models are updated
- Health checks show Ollama as "connected" not "degraded"
- Test generation endpoint returns actual LLM responses

**Dependencies:** None  
**Effort:** 2 story points

### Story 1.2: Implement Model Hot-Swap Capability [P1]
**As a** DevOps Engineer  
**I want** to switch between different Ollama models without restarting services  
**So that** we can test different models easily

**Acceptance Criteria:**
- Configuration supports multiple model definitions
- API endpoint exists to switch active model
- Model switching doesn't require service restart
- Current model status visible in health endpoint

**Dependencies:** Story 1.1  
**Effort:** 3 story points

### Story 1.3: Add Model Performance Monitoring [P2]
**As a** System Administrator  
**I want** to monitor model inference performance  
**So that** I can optimize resource allocation

**Acceptance Criteria:**
- Prometheus metrics for inference time, token count, queue depth
- Grafana dashboard showing model performance
- Alert rules for slow inference or high queue depth

**Dependencies:** Story 1.1  
**Effort:** 3 story points

---

## Epic 2: Agent Implementation
**Goal**: Transform stub agents into functional AI services with real business logic

### Story 2.1: Implement Task Assignment Coordinator Logic [P0]
**As a** System Architect  
**I want** the Task Assignment Coordinator to actually route tasks  
**So that** work is distributed across agents

**Acceptance Criteria:**
- Agent reads from RabbitMQ task.assign queue
- Routes tasks based on agents.yaml configuration
- Implements round-robin and priority assignment
- Sends assignment confirmations/failures
- Integration tests pass with 5 concurrent assignments

**Dependencies:** Epic 4 (RabbitMQ setup)  
**Effort:** 5 story points

### Story 2.2: Implement AI Agent Orchestrator Core Logic [P0]
**As a** Developer  
**I want** the AI Agent Orchestrator to coordinate multi-agent workflows  
**So that** complex tasks can be decomposed and executed

**Acceptance Criteria:**
- Orchestrator receives high-level tasks via API
- Decomposes tasks into subtasks
- Assigns subtasks to appropriate agents
- Aggregates results from multiple agents
- Returns consolidated response

**Dependencies:** Story 2.1  
**Effort:** 8 story points

### Story 2.3: Implement Hardware Resource Optimizer [P1]
**As a** System Administrator  
**I want** the Hardware Optimizer to monitor and optimize resource usage  
**So that** system resources are efficiently utilized

**Acceptance Criteria:**
- Monitors CPU, memory, disk usage via host metrics
- Suggests resource allocation improvements
- Implements basic auto-scaling logic
- Exposes metrics to Prometheus

**Dependencies:** None  
**Effort:** 5 story points

### Story 2.4: Implement Multi-Agent Coordinator [P1]
**As a** Developer  
**I want** agents to coordinate on complex tasks  
**So that** we can handle workflows requiring multiple specialists

**Acceptance Criteria:**
- Coordinator manages agent dependencies
- Handles sequential and parallel task execution
- Implements retry logic for failed subtasks
- Provides workflow status updates

**Dependencies:** Story 2.1, 2.2  
**Effort:** 5 story points

### Story 2.5: Implement Resource Arbitration Agent [P1]
**As a** System Architect  
**I want** intelligent resource allocation between competing agents  
**So that** critical tasks get priority access to resources

**Acceptance Criteria:**
- Tracks resource requests from all agents
- Implements priority-based allocation
- Prevents resource starvation
- Handles resource conflicts

**Dependencies:** Story 2.3  
**Effort:** 5 story points

---

## Epic 3: Inter-Agent Communication
**Goal**: Establish robust messaging between agents using RabbitMQ

### Story 3.1: Create Message Schema Definitions [P0]
**As a** Developer  
**I want** standardized message schemas  
**So that** agents can communicate reliably

**Acceptance Criteria:**
- Pydantic schemas in /schemas/ for all message types
- Schema validation on send and receive
- Documentation of all message formats
- Version management for schema changes

**Dependencies:** None  
**Effort:** 3 story points

### Story 3.2: Implement RabbitMQ Connection Pool [P0]
**As a** Backend Developer  
**I want** a shared connection pool for RabbitMQ  
**So that** we don't exhaust connections

**Acceptance Criteria:**
- Connection pool with configurable size
- Automatic reconnection on failure
- Connection health monitoring
- Shared across all agent services

**Dependencies:** None  
**Effort:** 3 story points

### Story 3.3: Implement Agent Heartbeat System [P1]
**As a** System Administrator  
**I want** to know which agents are alive and their load  
**So that** I can monitor system health

**Acceptance Criteria:**
- Agents send heartbeats every 30 seconds
- Heartbeat includes agent_id, load, timestamp
- Stale agents detected after 120 seconds
- Dead agent alerts via AlertManager

**Dependencies:** Story 3.1, 3.2  
**Effort:** 3 story points

### Story 3.4: Implement Message Tracing [P1]
**As a** Developer  
**I want** to trace messages through the system  
**So that** I can debug complex workflows

**Acceptance Criteria:**
- Trace IDs added to all messages
- Trace ID propagated through agent calls
- Centralized trace logging in Loki
- Trace visualization in Grafana

**Dependencies:** Story 3.1  
**Effort:** 3 story points

---

## Epic 4: Database Integration
**Goal**: Create database schemas and integrate vector databases

### Story 4.1: Create PostgreSQL Schema and Migrations [P0]
**As a** Backend Developer  
**I want** database tables for users, agents, and tasks  
**So that** we can persist application data

**Acceptance Criteria:**
- SQLAlchemy models for all entities
- Alembic migrations configured
- Initial schema deployed
- Seed data loaded

**Dependencies:** None  
**Effort:** 3 story points

### Story 4.2: Integrate Qdrant Vector Database [P1]
**As a** AI Developer  
**I want** to store and search embeddings  
**So that** we can implement semantic search

**Acceptance Criteria:**
- Qdrant client integrated in backend
- Collection created for document embeddings
- Search endpoint implemented
- Performance benchmarks documented

**Dependencies:** None  
**Effort:** 5 story points

### Story 4.3: Implement Redis Caching Layer [P1]
**As a** Backend Developer  
**I want** to cache frequently accessed data  
**So that** we reduce database load

**Acceptance Criteria:**
- Redis client configured
- Cache-aside pattern implemented
- TTL strategy defined
- Cache hit rate metrics

**Dependencies:** Story 4.1  
**Effort:** 3 story points

### Story 4.4: Integrate Neo4j for Graph Operations [P2]
**As a** Data Scientist  
**I want** to model relationships as graphs  
**So that** we can analyze complex connections

**Acceptance Criteria:**
- Neo4j driver integrated
- Graph schema defined
- Basic CRUD operations
- Cypher query examples

**Dependencies:** None  
**Effort:** 5 story points

---

## Epic 5: Service Mesh Configuration
**Goal**: Properly configure Kong, Consul, and service discovery

### Story 5.1: Configure Kong API Routes [P1]
**As a** DevOps Engineer  
**I want** Kong to route API requests  
**So that** we have a single entry point

**Acceptance Criteria:**
- Routes defined for all backend services
- Authentication plugin configured
- Rate limiting enabled
- Route documentation

**Dependencies:** None  
**Effort:** 3 story points

### Story 5.2: Fix Consul Health Checks [P1]
**As a** System Administrator  
**I want** accurate health status in Consul  
**So that** we know service availability

**Acceptance Criteria:**
- All services registered with correct IPs
- Health checks return accurate status
- No phantom services
- Auto-deregistration of dead services

**Dependencies:** None  
**Effort:** 2 story points

### Story 5.3: Implement Service Discovery in Agents [P1]
**As a** Developer  
**I want** agents to discover services via Consul  
**So that** we don't hardcode endpoints

**Acceptance Criteria:**
- Consul client in agent base class
- Service lookup by name
- Fallback to environment variables
- Cache discovered endpoints

**Dependencies:** Story 5.2  
**Effort:** 3 story points

---

## Epic 6: Technical Debt Cleanup
**Goal**: Remove fantasy code and consolidate duplicated components

### Story 6.1: Consolidate Requirements Files [P2]
**As a** DevOps Engineer  
**I want** three requirements files (base, dev, prod)  
**So that** dependency management is simple

**Acceptance Criteria:**
- 75+ requirements files merged into 3
- Version conflicts resolved
- All agents use shared requirements
- Documentation updated

**Dependencies:** None  
**Effort:** 5 story points

### Story 6.2: Remove Non-Running Service Definitions [P2]
**As a** System Administrator  
**I want** docker-compose.yml to only define real services  
**So that** configuration matches reality

**Acceptance Criteria:**
- 31 phantom services removed
- Docker-compose validates without errors
- All defined services can start
- Port registry accurate

**Dependencies:** None  
**Effort:** 2 story points

### Story 6.3: Clean Script Directory [P2]
**As a** Developer  
**I want** organized, documented scripts  
**So that** automation is maintainable

**Acceptance Criteria:**
- Scripts organized by category
- Duplicate scripts removed
- All scripts have headers and documentation
- Executable permissions set correctly

**Dependencies:** None  
**Effort:** 3 story points

### Story 6.4: Remove Fantasy Documentation [P2]
**As a** Technical Writer  
**I want** only accurate documentation  
**So that** developers aren't misled

**Acceptance Criteria:**
- Quantum computing references removed
- AGI/ASI claims eliminated
- Documentation matches implementation
- CLAUDE.md is single source of truth

**Dependencies:** None  
**Effort:** 2 story points

---

## Epic 7: Documentation & Testing
**Goal**: Establish comprehensive documentation and testing practices

### Story 7.1: Create API Documentation [P1]
**As a** Frontend Developer  
**I want** complete API documentation  
**So that** I can integrate correctly

**Acceptance Criteria:**
- OpenAPI/Swagger spec for all endpoints
- Request/response examples
- Authentication documentation
- Postman collection

**Dependencies:** None  
**Effort:** 3 story points

### Story 7.2: Implement Integration Test Suite [P1]
**As a** QA Engineer  
**I want** automated integration tests  
**So that** we catch breaking changes

**Acceptance Criteria:**
- Tests for all critical paths
- Docker-compose test environment
- CI/CD integration
- 80% code coverage

**Dependencies:** Story 2.1, 2.2  
**Effort:** 5 story points

### Story 7.3: Create Operational Runbook [P1]
**As a** Operations Engineer  
**I want** runbooks for common operations  
**So that** incidents are resolved quickly

**Acceptance Criteria:**
- Startup/shutdown procedures
- Troubleshooting guides
- Disaster recovery steps
- Performance tuning guide

**Dependencies:** None  
**Effort:** 3 story points

### Story 7.4: Implement E2E Testing [P2]
**As a** QA Engineer  
**I want** end-to-end user journey tests  
**So that** we validate complete workflows

**Acceptance Criteria:**
- Playwright/Selenium tests for UI
- API workflow tests
- Performance benchmarks
- Nightly test runs

**Dependencies:** Story 7.2  
**Effort:** 5 story points

---

## Summary Statistics

### By Priority:
- **P0 (Critical)**: 6 stories, 21 story points
- **P1 (Core)**: 17 stories, 71 story points  
- **P2 (Cleanup)**: 8 stories, 30 story points

### By Epic:
1. **Model & LLM**: 3 stories, 8 points
2. **Agent Implementation**: 5 stories, 28 points
3. **Inter-Agent Comm**: 4 stories, 12 points
4. **Database**: 4 stories, 16 points
5. **Service Mesh**: 3 stories, 8 points
6. **Technical Debt**: 4 stories, 12 points
7. **Documentation**: 4 stories, 16 points

### Total: 31 stories, 122 story points

## Next Steps
1. Review and refine story estimates with team
2. Map P0 and P1 stories to Sprint 1 (see SUTAZAI_SPRINT_PLAN.md)
3. Create detailed technical designs for P0 stories
4. Set up tracking in project management tool