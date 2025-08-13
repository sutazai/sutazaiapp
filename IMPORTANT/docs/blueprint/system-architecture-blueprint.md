# SutazAI System Architecture Blueprint

**Version**: 2.0.0  
**Date**: 2025-08-08  
**Status**: Authoritative Architecture Document  
**Based on**: CLAUDE.md Reality Check & IMPORTANT/10_canonical Documentation

---

## 1. Executive Summary

### 1.1 System Purpose and Vision

SutazAI is a Docker Compose-based AI orchestration platform designed for **local deployment** of AI workloads using **locally-hosted language models**. The system aims to provide a foundation for multi-agent AI systems while maintaining complete data privacy through local execution.

### 1.2 Current State vs Target State

| Aspect | Current State (Reality) | Target State (6 weeks) |
|--------|------------------------|----------------------|
| **Architecture** | Distributed stubs with 1 working agent | 3 functional agents with real orchestration |
| **Database** | Empty PostgreSQL, no schema | UUID-based schema with full migrations |
| **Authentication** | None (open access) | JWT-based with RBAC |
| **AI Models** | TinyLlama only (637MB) | Multiple models via Ollama |
| **Agent Count** | 7 stubs, 1 functional | 3 fully functional agents |
| **API Gateway** | Unconfigured Kong | Configured routes with rate limiting |
| **Vector DB** | Running but isolated | Integrated RAG pipeline |
| **Monitoring** | Basic metrics collection | Custom dashboards with alerting |

### 1.3 Key Architectural Decisions

Per ADRs in `/opt/sutazaiapp/IMPORTANT/10_canonical/standards/`:

1. **ADR-0001**: UUID primary keys everywhere for distributed readiness
2. **ADR-0002**: 100% local execution with Ollama models
3. **ADR-0003**: Multi-architecture container images (amd64/arm64)
4. **ADR-0004**: Trivy vulnerability scans as release gates

---

## 2. System Context (C4 Level 1)

```mermaid
graph TB
    subgraph "External Actors"
        User[Users<br/>Web Browser]
        Admin[Administrators<br/>CLI/SSH]
        Dev[Developers<br/>IDE/Git]
    end
    
    subgraph "SutazAI System Boundary"
        System[SutazAI Platform<br/>Local AI Orchestration]
    end
    
    subgraph "External Systems"
        Git[Git Repository<br/>Source Control]
        Docker[Docker Hub<br/>Container Registry]
        Model[Model Registry<br/>Ollama Library]
    end
    
    User -->|HTTP/WebSocket| System
    Admin -->|Docker CLI/API| System
    Dev -->|Git Push/PR| Git
    System -->|Pull Images| Docker
    System -->|Download Models| Model
    Git -->|CI/CD Deploy| System
```

### 2.1 External Actors
- **Users**: Interact via Streamlit UI on port 10011
- **Administrators**: Manage via Docker CLI and monitoring dashboards
- **Developers**: Deploy through Git workflows and direct container access

### 2.2 System Boundaries
- All processing occurs locally within Docker network `sutazai-network`
- No external API calls for core functionality (per ADR-0002)
- Data never leaves the local environment

### 2.3 Key Interactions
- User tasks submitted through REST API (port 10010)
- Agent coordination via internal message passing
- Model inference through Ollama API (port 10104)

---

## 3. Container Architecture (C4 Level 2)

### 3.1 Running Containers Overview

```mermaid
graph TB
    subgraph "User Interface Layer"
        Frontend[Frontend<br/>Streamlit:10011<br/>⚠️ STARTING]
    end
    
    subgraph "API Layer"
        Backend[Backend API<br/>FastAPI:10010<br/>✅ HEALTHY]
        Kong[Kong Gateway<br/>:10005/8001<br/>✅ RUNNING]
    end
    
    subgraph "Agent Services (7 Stubs)"
        Orchestrator[AI Orchestrator<br/>:8589<br/>⚠️ Stub]
        TaskCoord[Task Coordinator<br/>:8551<br/>✅ Functional]
        MultiAgent[Multi-Agent<br/>:8587<br/>⚠️ Stub]
        Resource[Resource Arbitration<br/>:8588<br/>⚠️ Stub]
        Hardware[Hardware Optimizer<br/>:8002<br/>⚠️ Stub]
        OllamaInt[Ollama Integration<br/>:11015<br/>⚠️ Stub]
        Metrics[AI Metrics<br/>:11063<br/>❌ UNHEALTHY]
    end
    
    subgraph "Data Layer"
        Postgres[PostgreSQL<br/>:10000<br/>✅ HEALTHY<br/>(No Tables)]
        Redis[Redis<br/>:10001<br/>✅ HEALTHY]
        Neo4j[Neo4j<br/>:10002-3<br/>✅ HEALTHY]
    end
    
    subgraph "AI/ML Layer"
        Ollama[Ollama<br/>:10104<br/>✅ TinyLlama]
        ChromaDB[ChromaDB<br/>:10100<br/>⚠️ STARTING]
        Qdrant[Qdrant<br/>:10101-2<br/>✅ HEALTHY]
        FAISS[FAISS<br/>:10103<br/>✅ HEALTHY]
    end
    
    subgraph "Infrastructure"
        Consul[Consul<br/>:10006<br/>✅ RUNNING]
        RabbitMQ[RabbitMQ<br/>:10007-8<br/>✅ RUNNING]
    end
    
    subgraph "Monitoring Stack"
        Prometheus[Prometheus<br/>:10200<br/>✅ RUNNING]
        Grafana[Grafana<br/>:10201<br/>✅ admin/admin]
        Loki[Loki<br/>:10202<br/>✅ RUNNING]
        AlertManager[AlertManager<br/>:10203<br/>✅ RUNNING]
    end
    
    Frontend --> Backend
    Backend --> Kong
    Kong --> Orchestrator
    Orchestrator --> TaskCoord
    Orchestrator --> MultiAgent
    Backend --> Postgres
    Backend --> Redis
    Backend --> Ollama
    TaskCoord --> RabbitMQ
    Metrics --> Prometheus
```

### 3.2 Service Responsibilities

| Service | Port | Responsibility | Current Status |
|---------|------|---------------|----------------|
| **PostgreSQL** | 10000 | Primary data store | Empty, needs schema |
| **Redis** | 10001 | Caching & sessions | Functional |
| **Neo4j** | 10002-3 | Graph relationships | Not integrated |
| **Kong** | 10005 | API gateway | No routes configured |
| **Consul** | 10006 | Service discovery |   usage |
| **RabbitMQ** | 10007-8 | Message queue | Used by Task Coordinator |
| **Backend** | 10010 | REST API | Partially functional |
| **Frontend** | 10011 | Web UI | Basic implementation |
| **Ollama** | 10104 | LLM inference | TinyLlama loaded |
| **Prometheus** | 10200 | Metrics collection | Operational |
| **Grafana** | 10201 | Visualization | Needs custom dashboards |

### 3.3 Communication Patterns

```yaml
# Internal service communication
Backend → Redis: Session management (Redis protocol)
Backend → PostgreSQL: Data persistence (PostgreSQL protocol)
Backend → Ollama: Model inference (HTTP REST)
TaskCoordinator → RabbitMQ: Task queuing (AMQP)
Agents → Backend: Status updates (HTTP REST)
All Services → Prometheus: Metrics export (HTTP /metrics)
```

---

## 4. Component Architecture (C4 Level 3)

### 4.1 Backend API Components

```mermaid
graph LR
    subgraph "FastAPI Backend"
        API[API Routes]
        Auth[Auth Module<br/>❌ Not Implemented]
        Core[Core Config]
        DB[Database<br/>❌ No Schema]
        Services[Services]
        
        API --> Auth
        API --> Services
        Services --> DB
        Services --> Core
    end
    
    subgraph "External Dependencies"
        Ollama[Ollama API]
        PG[(PostgreSQL)]
        Redis[(Redis)]
    end
    
    Services --> Ollama
    DB --> PG
    Auth --> Redis
```

### 4.2 Agent Service Components

```python
# Current Agent Structure (Stub)
class BaseAgent:
    def __init__(self):
        self.app = Flask(__name__)
        
    @app.route('/health')
    def health():
        return {"status": "healthy"}
    
    @app.route('/process', methods=['POST'])
    def process():
        # Stub implementation
        return {"status": "processed", "result": {}}

# Target Agent Structure
class FunctionalAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.capabilities = []
        self.message_queue = RabbitMQ()
        self.ollama = OllamaClient()
    
    async def process(self, task):
        # Real implementation
        validation = self.validate_task(task)
        if not validation.valid:
            return {"error": validation.error}
        
        result = await self.execute_task(task)
        await self.publish_result(result)
        return result
```

### 4.3 Data Layer Components

- **PostgreSQL**: Requires UUID-based schema implementation
- **Redis**: Key-value caching for sessions and temporary data
- **Neo4j**: Graph database for relationship mapping (unused)
- **Vector DBs**: ChromaDB/Qdrant for embeddings (not integrated)

### 4.4 Infrastructure Components

- **Kong Gateway**: API routing and rate limiting (needs configuration)
- **Consul**: Service registry (  current usage)
- **RabbitMQ**: Message broker for agent communication
- **Docker Network**: `sutazai-network` for container isolation

---

## 5. Data Architecture

### 5.1 PostgreSQL Schema (Needs Implementation)

```sql
-- Core domain tables with UUID primary keys (per ADR-0001)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agents registry
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    type VARCHAR(50) NOT NULL,
    endpoint VARCHAR(255) NOT NULL,
    capabilities JSONB DEFAULT '[]'::jsonb,
    status VARCHAR(20) DEFAULT 'inactive',
    health_check_url VARCHAR(255),
    last_health_check TIMESTAMP,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    agent_id UUID REFERENCES agents(id),
    type VARCHAR(50) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    priority INTEGER DEFAULT 5,
    payload JSONB NOT NULL,
    result JSONB,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP
);

-- Agent communications
CREATE TABLE agent_messages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_agent_id UUID REFERENCES agents(id),
    target_agent_id UUID REFERENCES agents(id),
    task_id UUID REFERENCES tasks(id),
    message_type VARCHAR(50) NOT NULL,
    payload JSONB NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP
);

-- Knowledge documents for RAG
CREATE TABLE documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    title VARCHAR(255) NOT NULL,
    content TEXT,
    metadata JSONB DEFAULT '{}'::jsonb,
    embedding_id VARCHAR(255), -- Reference to vector DB
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for foreign keys and common queries
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_created_at ON tasks(created_at DESC);
CREATE INDEX idx_agent_messages_task_id ON agent_messages(task_id);
CREATE INDEX idx_documents_user_id ON documents(user_id);
```

### 5.2 Redis Caching Strategy

```yaml
# Cache key patterns
Sessions:
  Pattern: "session:{session_id}"
  TTL: 3600 seconds
  
Task Results:
  Pattern: "task:result:{task_id}"
  TTL: 300 seconds
  
Agent Status:
  Pattern: "agent:status:{agent_id}"
  TTL: 60 seconds
  
Rate Limiting:
  Pattern: "ratelimit:{user_id}:{endpoint}"
  TTL: 60 seconds
```

### 5.3 Neo4j Graph Model (Future Integration)

```cypher
// Domain entities as nodes
(u:User {id: UUID, email: STRING})
(a:Agent {id: UUID, name: STRING, type: STRING})
(t:Task {id: UUID, type: STRING, status: STRING})
(d:Document {id: UUID, title: STRING})

// Relationships
(u)-[:CREATED]->(t)
(t)-[:ASSIGNED_TO]->(a)
(a)-[:COMMUNICATES_WITH]->(a)
(u)-[:OWNS]->(d)
(t)-[:REFERENCES]->(d)
```

### 5.4 Vector Database Usage

```python
# ChromaDB for document embeddings
class VectorStore:
    def __init__(self):
        self.client = chromadb.Client()
        self.collection = self.client.get_or_create_collection("documents")
    
    async def add_document(self, doc_id: UUID, text: str, metadata: dict):
        embedding = await self.generate_embedding(text)
        self.collection.add(
            ids=[str(doc_id)],
            embeddings=[embedding],
            metadatas=[metadata]
        )
    
    async def search(self, query: str, k: int = 5):
        query_embedding = await self.generate_embedding(query)
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k
        )
        return results
```

---

## 6. Integration Architecture

### 6.1 Service Mesh Configuration

```yaml
# Kong Gateway Routes (To Be Implemented)
services:
  - name: backend-api
    url: http://backend:8000
    routes:
      - name: api-v1
        paths: ["/api/v1"]
        methods: ["GET", "POST", "PUT", "DELETE"]
    plugins:
      - name: jwt
        config:
          secret_is_base64: false
      - name: rate-limiting
        config:
          minute: 60
          policy: local
      - name: cors
        config:
          origins: ["http://localhost:10011"]

  - name: agent-orchestrator
    url: http://ai-agent-orchestrator:8589
    routes:
      - name: orchestrator
        paths: ["/api/v1/orchestrate"]
    plugins:
      - name: request-transformer
        config:
          add:
            headers:
              - "X-Agent-Type:orchestrator"
```

### 6.2 Message Queue Architecture

```mermaid
sequenceDiagram
    participant User
    participant Backend
    participant RabbitMQ
    participant TaskCoordinator
    participant Agent
    
    User->>Backend: Submit Task
    Backend->>RabbitMQ: Publish to task.queue
    RabbitMQ->>TaskCoordinator: Consume Message
    TaskCoordinator->>TaskCoordinator: Prioritize
    TaskCoordinator->>Agent: Assign Task
    Agent->>Agent: Process
    Agent->>RabbitMQ: Publish Result
    RabbitMQ->>Backend: Result Notification
    Backend->>User: Return Result
```

### 6.3 API Contracts

```yaml
# Standard API Response Format
Response:
  success:
    status: "success"
    data: <payload>
    metadata:
      timestamp: ISO8601
      version: "1.0"
      
  error:
    status: "error"
    error:
      code: ERROR_CODE
      message: "Human readable message"
      details: {}
    metadata:
      timestamp: ISO8601
      trace_id: UUID

# Agent Communication Protocol
AgentMessage:
  id: UUID
  type: TASK | RESULT | ERROR | HEARTBEAT
  source: AgentID
  target: AgentID
  correlation_id: UUID
  payload: {}
  timestamp: ISO8601
```

### 6.4 Event Flows

```mermaid
graph LR
    subgraph "Task Submission Flow"
        A[User Request] --> B[API Validation]
        B --> C[Auth Check]
        C --> D[Task Creation]
        D --> E[Queue Publication]
        E --> F[Agent Assignment]
        F --> G[Processing]
        G --> H[Result Storage]
        H --> I[User Notification]
    end
```

---

## 7. Security Architecture

### 7.1 Authentication & Authorization

```python
# JWT-based authentication (To Be Implemented)
class AuthService:
    async def login(self, email: str, password: str) -> dict:
        user = await self.verify_credentials(email, password)
        access_token = self.create_access_token(user.id)
        refresh_token = self.create_refresh_token(user.id)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer"
        }
    
    def create_access_token(self, user_id: UUID) -> str:
        payload = {
            "sub": str(user_id),
            "exp": datetime.utcnow() + timedelta(minutes=15),
            "iat": datetime.utcnow(),
            "type": "access"
        }
        return jwt.encode(payload, SECRET_KEY, algorithm="HS256")
```

### 7.2 Network Security

```yaml
# Docker Network Isolation
Networks:
  sutazai-network:
    driver: bridge
    internal: false  # Allow external access
    
  sutazai-internal:
    driver: bridge
    internal: true  # No external access
    
# Service placement
Public Services: [Frontend, Backend, Kong]
Internal Services: [Agents, Databases, RabbitMQ]
```

### 7.3 Data Protection

- **Encryption at Rest**: PostgreSQL with encrypted volumes
- **Encryption in Transit**: TLS for all external connections
- **Secrets Management**: Environment variables (temporary)
- **Future**: HashiCorp Vault integration

### 7.4 Security Vulnerabilities (Current)

| Vulnerability | Severity | Mitigation Plan |
|--------------|----------|-----------------|
| No authentication | CRITICAL | Implement JWT auth (Week 1) |
| Hardcoded secrets | HIGH | Move to env vars (Week 1) |
| No input validation | HIGH | Add Pydantic models (Week 2) |
| Open database ports | MEDIUM | Restrict to internal network |
| Default passwords | HIGH | Force password changes |

---

## 8. Deployment Architecture

### 8.1 Docker Compose Structure

```yaml
# Simplified docker-compose.yml structure
version: '3.8'

networks:
  sutazai-network:
    driver: bridge

services:
  # Core Infrastructure
  postgres:
    image: postgres:16.3
    environment:
      POSTGRES_DB: sutazai
      POSTGRES_USER: ${DB_USER:-sutazai}
      POSTGRES_PASSWORD: ${DB_PASSWORD:-change_me}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "10000:5432"
    networks:
      - sutazai-network
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U sutazai"]
      interval: 10s
      timeout: 5s
      retries: 5

  # Application Services
  backend:
    build:
      context: ./backend
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://sutazai:${DB_PASSWORD}@postgres:5432/sutazai
      REDIS_URL: redis://redis:6379
      OLLAMA_URL: http://ollama:10104
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    ports:
      - "10010:8000"
    networks:
      - sutazai-network

volumes:
  postgres_data:
  redis_data:
  ollama_models:
```

### 8.2 Environment Configuration

```bash
# .env.template
# Database
DB_USER=sutazai
DB_PASSWORD=<CHANGE_ME>
DB_NAME=sutazai

# Redis
REDIS_PASSWORD=<CHANGE_ME>

# JWT
JWT_SECRET=<GENERATE_RANDOM_SECRET>
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=15

# Ollama
OLLAMA_BASE_URL=http://ollama:10104
DEFAULT_MODEL=tinyllama

# Monitoring
GRAFANA_ADMIN_PASSWORD=<CHANGE_ME>

# Feature Flags
ENABLE_AUTH=true
ENABLE_RATE_LIMITING=true
ENABLE_VECTOR_SEARCH=false
```

### 8.3 Scaling Strategy

```yaml
# Horizontal scaling approach
Stateless Services:
  - Backend API: Scale to 3-5 instances
  - Agent Services: Scale based on workload
  
Stateful Services:
  - PostgreSQL: Single instance (HA via replication)
  - Redis: Single instance with persistence
  
Load Balancing:
  - Kong Gateway: Round-robin to backend instances
  - Agent Assignment: Queue-based distribution
```

### 8.4 Multi-Architecture Support (ADR-0003)

```dockerfile
# Multi-arch Dockerfile example
FROM --platform=$BUILDPLATFORM python:3.11-slim AS builder
ARG TARGETPLATFORM
ARG BUILDPLATFORM

# Build for both amd64 and arm64
RUN echo "Building for $TARGETPLATFORM on $BUILDPLATFORM"

# Use Docker Buildx
# docker buildx build --platform linux/amd64,linux/arm64 -t sutazai/backend:latest .
```

---

## 9. Operational Architecture

### 9.1 Monitoring Stack

```mermaid
graph TB
    subgraph "Data Collection"
        Exporters[Node/cAdvisor<br/>Exporters]
        Apps[Applications<br/>/metrics]
        Logs[Container<br/>Logs]
    end
    
    subgraph "Storage & Processing"
        Prometheus[Prometheus<br/>:10200]
        Loki[Loki<br/>:10202]
        AlertManager[AlertManager<br/>:10203]
    end
    
    subgraph "Visualization"
        Grafana[Grafana<br/>:10201]
    end
    
    Exporters --> Prometheus
    Apps --> Prometheus
    Logs --> Loki
    Prometheus --> AlertManager
    Prometheus --> Grafana
    Loki --> Grafana
    AlertManager --> |Notifications| Slack[Slack/Email]
```

### 9.2 Logging Strategy

```yaml
# Structured logging configuration
Logging:
  Format: JSON
  Levels:
    Production: INFO
    Development: DEBUG
  
  Fields:
    - timestamp: ISO8601
    - level: INFO|WARN|ERROR
    - service: service_name
    - trace_id: UUID
    - user_id: UUID (if authenticated)
    - message: string
    - extra: object

# Log aggregation via Loki
Sources:
  - Docker container logs
  - Application logs
  - System logs
```

### 9.3 Health Checks

```python
# Standard health check implementation
@app.get("/health")
async def health_check():
    checks = {
        "database": await check_database(),
        "redis": await check_redis(),
        "ollama": await check_ollama(),
        "disk_space": check_disk_space(),
        "memory": check_memory()
    }
    
    status = "healthy" if all(checks.values()) else "degraded"
    
    return {
        "status": status,
        "timestamp": datetime.utcnow().isoformat(),
        "checks": checks,
        "version": APP_VERSION
    }
```

### 9.4 Alerting Rules

```yaml
# Prometheus alerting rules
groups:
  - name: sutazai_alerts
    rules:
      - alert: ServiceDown
        expr: up == 0
        for: 5m
        annotations:
          summary: "Service {{ $labels.job }} is down"
          
      - alert: HighMemoryUsage
        expr: container_memory_usage_bytes > 4e9
        for: 10m
        annotations:
          summary: "Container {{ $labels.name }} high memory"
          
      - alert: DatabaseConnectionFailure
        expr: pg_up == 0
        for: 1m
        annotations:
          summary: "PostgreSQL connection failed"
          severity: critical
```

---

## 10. Migration Path

### 10.1 From Current to Target State

```mermaid
gantt
    title Migration Timeline (6 Weeks)
    dateFormat YYYY-MM-DD
    section Foundation
    Database Schema        :done, 2025-08-09, 2d
    JWT Authentication     :active, 2025-08-11, 3d
    Environment Config     :2025-08-11, 2d
    
    section Agent Development
    Code Assistant Agent   :2025-08-14, 5d
    Doc Assistant Agent    :2025-08-19, 5d
    Agent Communication    :2025-08-21, 3d
    
    section Integration
    Vector DB Pipeline     :2025-08-24, 4d
    Kong Configuration     :2025-08-26, 2d
    RAG Implementation     :2025-08-28, 3d
    
    section Testing & Deployment
    Integration Testing    :2025-08-31, 3d
    Performance Testing    :2025-09-03, 2d
    Documentation         :2025-09-05, 2d
    Production Deploy     :milestone, 2025-09-07, 0d
```

### 10.2 Database Schema Migration

```bash
#!/bin/bash
# Migration script

# Step 1: Backup existing data (if any)
docker exec sutazai-postgres pg_dump -U sutazai sutazai > backup_$(date +%Y%m%d).sql

# Step 2: Apply UUID extension
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\";"

# Step 3: Run migrations
docker exec sutazai-backend alembic upgrade head

# Step 4: Verify schema
docker exec sutazai-postgres psql -U sutazai -d sutazai -c "\dt"
```

### 10.3 Service Mesh Configuration

```bash
# Kong route configuration
curl -X POST http://localhost:8001/services \
  -d name=backend-api \
  -d url=http://backend:8000

curl -X POST http://localhost:8001/services/backend-api/routes \
  -d paths[]=/api/v1 \
  -d methods[]=GET \
  -d methods[]=POST

curl -X POST http://localhost:8001/services/backend-api/plugins \
  -d name=jwt
```

### 10.4 Agent Implementation Strategy

| Phase | Agent | Implementation | Priority |
|-------|-------|---------------|----------|
| 1 | Task Coordinator | Already functional | ✅ Complete |
| 2 | Code Assistant | Implement with tinyllama | Week 2-3 |
| 3 | Doc Assistant | RAG with ChromaDB | Week 3-4 |
| 4 | Orchestrator | Coordinate all agents | Week 4-5 |

---

## 11. Risk Assessment & Mitigation

### 11.1 Technical Risks

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| **TinyLlama limitations** | HIGH | MEDIUM | Design for model swapping, test larger models |
| **Database performance** | MEDIUM | HIGH | Implement proper indexing, connection pooling |
| **Agent communication failures** | MEDIUM | HIGH | Circuit breakers, retry logic, timeouts |
| **Memory constraints** | MEDIUM | MEDIUM | Resource limits, horizontal scaling |
| **Security breaches** | LOW | CRITICAL | Implement auth immediately, security scanning |

### 11.2 Operational Risks

- **Single points of failure**: PostgreSQL, Redis (mitigate with backups)
- **Lack of documentation**: Ongoing documentation updates required
- **Team knowledge gaps**: Training and knowledge transfer sessions

### 11.3 Rollback Strategy

```yaml
Rollback Plan:
  1. Tag all deployments with version numbers
  2. Maintain database migration rollback scripts
  3. Feature flags for new functionality
  4. Blue-green deployment capability
  5. Automated backup before each deployment
  
Recovery Time Objectives:
  - Critical services: < 5 minutes
  - Non-critical services: < 30 minutes
  - Full system restore: < 2 hours
```

---

## 12. Appendices

### A. Port Registry

| Port | Service | Protocol | Status |
|------|---------|----------|--------|
| 10000 | PostgreSQL | TCP | ✅ Running |
| 10001 | Redis | TCP | ✅ Running |
| 10002 | Neo4j Browser | HTTP | ✅ Running |
| 10003 | Neo4j Bolt | TCP | ✅ Running |
| 10005 | Kong Gateway | HTTP | ✅ Running |
| 10006 | Consul | HTTP | ✅ Running |
| 10007 | RabbitMQ AMQP | TCP | ✅ Running |
| 10008 | RabbitMQ Management | HTTP | ✅ Running |
| 10010 | Backend API | HTTP | ✅ Running |
| 10011 | Frontend UI | HTTP | ⚠️ Starting |
| 10100 | ChromaDB | HTTP | ⚠️ Issues |
| 10101 | Qdrant HTTP | HTTP | ✅ Running |
| 10102 | Qdrant gRPC | gRPC | ✅ Running |
| 10103 | FAISS | HTTP | ✅ Running |
| 10104 | Ollama | HTTP | ✅ Running |
| 10200 | Prometheus | HTTP | ✅ Running |
| 10201 | Grafana | HTTP | ✅ Running |
| 10202 | Loki | HTTP | ✅ Running |
| 10203 | AlertManager | HTTP | ✅ Running |

### B. Command Reference

```bash
# System Management
docker-compose up -d                    # Start all services
docker-compose ps                       # Check service status
docker-compose logs -f [service]        # View service logs
docker-compose restart [service]        # Restart a service
docker-compose down                     # Stop all services

# Database Operations
docker exec -it sutazai-postgres psql -U sutazai -d sutazai  # PostgreSQL CLI
docker exec -it sutazai-redis redis-cli                      # Redis CLI

# Testing Endpoints
curl http://localhost:10010/health      # Backend health
curl http://localhost:10104/api/tags    # Ollama models
curl http://localhost:8551/health       # Task Coordinator

# Monitoring Access
http://localhost:10200                  # Prometheus
http://localhost:10201                  # Grafana (admin/admin)
http://localhost:10002                  # Neo4j Browser
```

### C. References

- **Source of Truth**: `/opt/sutazaiapp/CLAUDE.md`
- **Canonical Docs**: `/opt/sutazaiapp/IMPORTANT/10_canonical/`
- **ADRs**: `/opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-*.md`
- **Risk Register**: `/opt/sutazaiapp/IMPORTANT/01_findings/risk_register.md`
- **Engineering Standards**: `/opt/sutazaiapp/IMPORTANT/COMPREHENSIVE_ENGINEERING_STANDARDS.md`

---

## Document Control

- **Version**: 2.0.0
- **Last Updated**: 2025-08-08
- **Next Review**: 2025-09-07
- **Status**: Living Document - Update as system evolves
- **Approval**: System Architecture Team

### Revision History

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 2.0.0 | 2025-08-08 | System Architect | Complete rewrite based on verified reality |
| 1.0.0 | 2025-08-07 | Previous Team | Initial blueprint (contained inaccuracies) |

---

*This blueprint represents the authoritative technical architecture for the SutazAI system. It is based on verified system state and supersedes all previous architecture documents. Regular updates are required as the system evolves from its current proof-of-concept state to production readiness.*