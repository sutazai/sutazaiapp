# System Architecture Design

**Last Updated**: 2025-01-03  
**Version**: 1.0.0  
**Maintainer**: Architecture Team

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [System Components](#system-components)
3. [Service Architecture](#service-architecture)
4. [Data Architecture](#data-architecture)
5. [Network Architecture](#network-architecture)
6. [Security Architecture](#security-architecture)
7. [Scalability Design](#scalability-design)
8. [Deployment Architecture](#deployment-architecture)
9. [Integration Points](#integration-points)
10. [Architecture Decisions](#architecture-decisions)

## Architecture Overview

### System Type
**Hybrid Microservices with Event-Driven Multi-Agent Orchestration**

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Streamlit UI :11000]
        API_Client[API Clients]
    end
    
    subgraph "Gateway Layer"
        Kong[Kong API Gateway :10008-10009]
        LB[Load Balancer]
    end
    
    subgraph "Application Layer"
        FastAPI[FastAPI Backend :10200]
        MCP_Bridge[MCP Bridge :11100]
    end
    
    subgraph "Data Layer"
        PG[(PostgreSQL :10000)]
        Redis[(Redis :10001)]
        Neo4j[(Neo4j :10002)]
        RabbitMQ[RabbitMQ :10004]
    end
    
    subgraph "Vector Layer"
        ChromaDB[(ChromaDB :10100)]
        Qdrant[(Qdrant :10101)]
        FAISS[(FAISS :10103)]
    end
    
    subgraph "Agent Layer"
        Agents[AI Agents :11401-11801]
        MCP_Servers[MCP Servers x18]
    end
    
    UI --> Kong
    API_Client --> Kong
    Kong --> LB
    LB --> FastAPI
    FastAPI --> PG
    FastAPI --> Redis
    FastAPI --> Neo4j
    FastAPI --> RabbitMQ
    FastAPI --> MCP_Bridge
    MCP_Bridge --> Agents
    FastAPI --> ChromaDB
    FastAPI --> Qdrant
    FastAPI --> FAISS
```

### Architecture Principles

1. **Separation of Concerns**: Each service has a single, well-defined responsibility
2. **Loose Coupling**: Services communicate through well-defined interfaces
3. **High Cohesion**: Related functionality is grouped together
4. **Scalability**: Horizontal scaling capability for all services
5. **Resilience**: Failure isolation and graceful degradation
6. **Observability**: Comprehensive logging, monitoring, and tracing

## System Components

### Core Services

| Service | Technology | Port | Purpose | Resources |
|---------|-----------|------|---------|-----------|
| API Backend | FastAPI | 10200 | REST API, WebSocket, business logic | 2GB RAM, 2 CPU |
| Web UI | Streamlit | 11000 | User interface | 1GB RAM, 1 CPU |
| API Gateway | Kong | 10008-10009 | Routing, auth, rate limiting | 1GB RAM, 1 CPU |
| MCP Bridge | Node.js | 11100 | Agent orchestration | 2GB RAM, 2 CPU |

### Data Stores

| Store | Type | Port | Use Case | Capacity |
|-------|------|------|----------|----------|
| PostgreSQL | Relational | 10000 | Primary data, transactions | 10GB, 4GB RAM |
| Redis | Key-Value | 10001 | Cache, sessions, pub/sub | 4GB RAM |
| Neo4j | Graph | 10002-10003 | Relationships, knowledge graph | 8GB RAM |
| RabbitMQ | Message Queue | 10004-10005 | Async tasks, events | 2GB RAM |

### Vector Databases

| Database | Port | Purpose | Index Size |
|----------|------|---------|------------|
| ChromaDB | 10100 | Document embeddings | 5GB |
| Qdrant | 10101-10102 | Semantic search | 8GB |
| FAISS | 10103 | Similarity search | 4GB |

### AI Agent Infrastructure

```yaml
MCP Servers (18 total):
  - filesystem: File system operations
  - memory: Session state management
  - extended-memory: Persistent memory
  - memory-bank: Project memory
  - github: GitHub API integration
  - github-project-manager: Project management
  - claude-flow: Agent orchestration
  - ruv-swarm: Swarm management
  - context7: Documentation lookup
  - playwright: Browser automation
  - sequential-thinking: Multi-step reasoning
  - code-index: Code search and analysis
  - ddg: Web search
  - http-fetch: HTTP requests
  - gitmcp-sutazai: Git operations (local)
  - gitmcp-anthropic: Claude documentation
  - gitmcp-docs: Generic documentation
  - everything: Utility functions
```

## Service Architecture

### Backend Service (FastAPI)

```python
# Service Layer Architecture
app/
├── api/
│   ├── v1/
│   │   ├── router.py           # Main API router
│   │   └── endpoints/          # API endpoints
│   │       ├── auth.py         # Authentication
│   │       ├── users.py        # User management
│   │       ├── agents.py       # Agent control
│   │       ├── vectors.py      # Vector operations
│   │       └── websocket.py    # Real-time connections
│   └── dependencies/
│       ├── auth.py            # Auth dependencies
│       └── database.py        # DB dependencies
├── core/
│   ├── config.py              # Configuration
│   ├── security.py            # Security utilities
│   ├── database.py            # Database setup
│   └── connections.py         # Service connections (SINGLETON)
├── models/
│   ├── user.py               # User models
│   ├── session.py            # Session models
│   └── agent.py              # Agent models
├── services/
│   ├── auth_service.py       # Authentication logic
│   ├── agent_service.py      # Agent management
│   ├── vector_service.py     # Vector operations
│   └── cache_service.py      # Caching logic
└── main.py                    # Application entry
```

### Frontend Service (Streamlit)

```python
# UI Layer Architecture
frontend/
├── app.py                     # Main application
├── pages/
│   ├── dashboard.py          # Dashboard view
│   ├── agents.py             # Agent management
│   ├── analytics.py          # Analytics view
│   └── settings.py           # Settings page
├── components/
│   ├── chat.py               # Chat interface
│   ├── sidebar.py            # Navigation sidebar
│   └── metrics.py            # Metrics display
├── utils/
│   ├── api_client.py         # API communication
│   ├── websocket_client.py   # WebSocket handler
│   └── state_manager.py      # Session state
└── config/
    └── settings.py           # UI configuration
```

### MCP Bridge Architecture

```typescript
// Agent Orchestration Layer
mcp-bridge/
├── src/
│   ├── server.ts             // Main server
│   ├── routers/
│   │   ├── agent.router.ts   // Agent routes
│   │   ├── task.router.ts    // Task management
│   │   └── health.router.ts  // Health checks
│   ├── services/
│   │   ├── orchestrator.ts   // Task orchestration
│   │   ├── registry.ts       // Agent registry
│   │   └── scheduler.ts      // Task scheduling
│   ├── agents/
│   │   ├── base.agent.ts     // Base agent class
│   │   └── specialized/      // Specialized agents
│   └── utils/
│       ├── logger.ts         // Logging utility
│       └── metrics.ts        // Metrics collection
```

## Data Architecture

### Database Schema

```sql
-- PostgreSQL Main Schema
CREATE SCHEMA sutazai;

-- Users table
CREATE TABLE sutazai.users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    is_superuser BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Sessions table
CREATE TABLE sutazai.sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES sutazai.users(id) ON DELETE CASCADE,
    token VARCHAR(500) UNIQUE NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Agent executions table
CREATE TABLE sutazai.agent_executions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES sutazai.users(id),
    agent_type VARCHAR(100) NOT NULL,
    task_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    input_data JSONB,
    output_data JSONB,
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    duration_ms INTEGER
);

-- Vectors table
CREATE TABLE sutazai.vectors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES sutazai.users(id),
    collection_name VARCHAR(255) NOT NULL,
    vector_id VARCHAR(255) NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(collection_name, vector_id)
);
```

### Redis Data Structures

```redis
# Session storage
SESSION:{session_id} -> {
    user_id: UUID,
    expires: timestamp,
    data: JSON
}

# Rate limiting
RATE_LIMIT:{user_id}:{endpoint} -> counter

# Cache patterns
CACHE:users:{user_id} -> User JSON
CACHE:agents:{agent_id} -> Agent status
CACHE:vectors:{collection}:{query_hash} -> Results

# Pub/Sub channels
CHANNEL:agent_updates -> Agent status updates
CHANNEL:task_queue -> Task distribution
CHANNEL:notifications -> User notifications
```

### Neo4j Graph Model

```cypher
// Node types
(User {id: UUID, email: STRING, created: DATETIME})
(Agent {id: UUID, type: STRING, status: STRING})
(Task {id: UUID, type: STRING, status: STRING})
(Document {id: UUID, content: STRING, embedding: LIST})
(Conversation {id: UUID, created: DATETIME})

// Relationships
(User)-[:OWNS]->(Agent)
(User)-[:CREATED]->(Task)
(Agent)-[:EXECUTES]->(Task)
(Task)-[:PRODUCES]->(Document)
(User)-[:PARTICIPATES_IN]->(Conversation)
(Document)-[:SIMILAR_TO {score: FLOAT}]->(Document)
```

## Network Architecture

### Docker Network Configuration

```yaml
networks:
  sutazai-network:
    driver: bridge
    ipam:
      config:
        - subnet: 172.20.0.0/16
          gateway: 172.20.0.1

# IP Allocation Plan
# 172.20.0.1     - Gateway
# 172.20.0.10-29 - Core Services
# 172.20.0.30-49 - Application Services  
# 172.20.0.50-69 - Data Services
# 172.20.0.70-89 - Vector Services
# 172.20.0.100-199 - Agent Services
```

### Service Discovery

```yaml
# Internal DNS names
postgres.sutazai.local -> 172.20.0.10
redis.sutazai.local -> 172.20.0.11
neo4j.sutazai.local -> 172.20.0.12
rabbitmq.sutazai.local -> 172.20.0.13
backend.sutazai.local -> 172.20.0.30
frontend.sutazai.local -> 172.20.0.31
mcp-bridge.sutazai.local -> 172.20.0.32
```

### Load Balancing Strategy

```nginx
# nginx.conf for load balancing
upstream backend_servers {
    least_conn;
    server backend-1:10200 weight=3;
    server backend-2:10200 weight=2;
    server backend-3:10200 weight=1;
    
    keepalive 32;
}

upstream websocket_servers {
    ip_hash;  # Sticky sessions for WebSocket
    server ws-1:10201;
    server ws-2:10201;
}
```

## Security Architecture

### Authentication Flow

```mermaid
sequenceDiagram
    Client->>Gateway: POST /auth/login
    Gateway->>Backend: Forward request
    Backend->>PostgreSQL: Verify credentials
    PostgreSQL-->>Backend: User data
    Backend->>Redis: Store session
    Backend-->>Gateway: JWT token
    Gateway-->>Client: Auth response
    Client->>Gateway: GET /api/data + JWT
    Gateway->>Backend: Verify JWT
    Backend->>Redis: Check session
    Backend-->>Gateway: Data response
    Gateway-->>Client: Protected data
```

### Security Layers

1. **Network Security**
   - Docker network isolation
   - Service mesh with mTLS
   - API Gateway with rate limiting

2. **Application Security**
   - JWT authentication
   - Role-based access control (RBAC)
   - Input validation and sanitization
   - SQL injection prevention
   - XSS protection

3. **Data Security**
   - Encryption at rest (AES-256)
   - Encryption in transit (TLS 1.3)
   - Secrets management (HashiCorp Vault)
   - Database column encryption

4. **Operational Security**
   - Audit logging
   - Security scanning
   - Dependency updates
   - Incident response

## Scalability Design

### Horizontal Scaling Strategy

```yaml
# Auto-scaling configuration
services:
  backend:
    min_replicas: 2
    max_replicas: 10
    target_cpu: 70%
    target_memory: 80%
    
  mcp-bridge:
    min_replicas: 1
    max_replicas: 5
    target_concurrent_tasks: 100
    
  vector-processor:
    min_replicas: 1
    max_replicas: 3
    target_queue_depth: 1000
```

### Caching Strategy

```python
# Multi-layer caching
class CacheStrategy:
    L1_CACHE = "memory"      # In-process cache (100ms)
    L2_CACHE = "redis"       # Redis cache (1-5ms)
    L3_CACHE = "cdn"         # CDN cache (10-50ms)
    
    CACHE_TIMES = {
        "user_profile": 300,      # 5 minutes
        "agent_status": 10,       # 10 seconds
        "vector_search": 3600,    # 1 hour
        "static_content": 86400   # 1 day
    }
```

### Database Sharding

```sql
-- Sharding strategy for high-volume tables
-- Shard by user_id (hash-based)
CREATE TABLE agent_executions_shard_0 PARTITION OF agent_executions
    FOR VALUES WITH (modulus 4, remainder 0);
    
CREATE TABLE agent_executions_shard_1 PARTITION OF agent_executions
    FOR VALUES WITH (modulus 4, remainder 1);
    
CREATE TABLE agent_executions_shard_2 PARTITION OF agent_executions
    FOR VALUES WITH (modulus 4, remainder 2);
    
CREATE TABLE agent_executions_shard_3 PARTITION OF agent_executions
    FOR VALUES WITH (modulus 4, remainder 3);
```

## Deployment Architecture

### Container Orchestration

```yaml
# Kubernetes deployment example
apiVersion: apps/v1
kind: Deployment
metadata:
  name: sutazai-backend
  namespace: sutazai
spec:
  replicas: 3
  selector:
    matchLabels:
      app: backend
  template:
    metadata:
      labels:
        app: backend
    spec:
      containers:
      - name: backend
        image: sutazai/backend:v1.0.0
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        env:
        - name: DB_HOST
          valueFrom:
            secretKeyRef:
              name: db-secret
              key: host
```

### CI/CD Pipeline

```yaml
# GitHub Actions workflow
name: Deploy to Production
on:
  push:
    branches: [main]
    
jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - name: Run tests
        run: |
          docker compose -f docker-compose.test.yml up --abort-on-container-exit
          
  build:
    needs: test
    runs-on: ubuntu-latest
    steps:
      - name: Build and push Docker images
        run: |
          docker build -t sutazai/backend:${{ github.sha }} ./backend
          docker push sutazai/backend:${{ github.sha }}
          
  deploy:
    needs: build
    runs-on: ubuntu-latest
    steps:
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/backend backend=sutazai/backend:${{ github.sha }}
          kubectl rollout status deployment/backend
```

## Integration Points

### External Service Integration

```python
# Integration patterns
class IntegrationPatterns:
    # Circuit Breaker
    @circuit_breaker(failure_threshold=5, recovery_timeout=60)
    async def call_openai_api(self, prompt: str):
        return await openai_client.completions.create(prompt=prompt)
    
    # Retry with exponential backoff
    @retry(max_attempts=3, backoff_factor=2)
    async def call_github_api(self, endpoint: str):
        return await github_client.get(endpoint)
    
    # Rate limiting
    @rate_limit(calls=100, period=60)
    async def call_external_service(self, data: dict):
        return await external_client.post("/api", json=data)
```

### Message Queue Integration

```python
# RabbitMQ integration
async def setup_message_queues():
    connection = await aio_pika.connect_robust(
        "amqp://sutazai:password@rabbitmq:5672/"
    )
    channel = await connection.channel()
    
    # Declare exchanges
    await channel.declare_exchange(
        "agents", ExchangeType.TOPIC, durable=True
    )
    
    # Declare queues
    task_queue = await channel.declare_queue(
        "task_queue", durable=True, arguments={
            "x-max-priority": 10,
            "x-message-ttl": 3600000  # 1 hour
        }
    )
    
    # Bind queues to exchanges
    await task_queue.bind("agents", "task.*")
```

## Architecture Decisions

### ADR-001: Microservices Architecture

**Status**: Accepted  
**Context**: Need for scalable, maintainable system  
**Decision**: Adopt microservices with clear boundaries  
**Consequences**: 
- (+) Independent scaling and deployment
- (+) Technology flexibility
- (-) Increased complexity
- (-) Network overhead

### ADR-002: Event-Driven Communication

**Status**: Accepted  
**Context**: Need for loose coupling between services  
**Decision**: Use RabbitMQ for async communication  
**Consequences**:
- (+) Decoupled services
- (+) Better fault tolerance
- (-) Eventual consistency
- (-) Debugging complexity

### ADR-003: Vector Database Selection

**Status**: Accepted  
**Context**: Need for multiple vector search capabilities  
**Decision**: Use ChromaDB, Qdrant, and FAISS  
**Consequences**:
- (+) Best tool for each use case
- (+) Redundancy and failover
- (-) Maintenance overhead
- (-) Data synchronization

### ADR-004: MCP Server Architecture

**Status**: Accepted  
**Context**: Need for modular AI agent capabilities  
**Decision**: Implement 18 specialized MCP servers  
**Consequences**:
- (+) Specialized functionality
- (+) Independent scaling
- (-) Orchestration complexity
- (-) Resource usage

## Performance Considerations

### Latency Targets

| Operation | Target | Maximum |
|-----------|--------|---------|
| API Response | 200ms | 1000ms |
| Database Query | 50ms | 200ms |
| Cache Lookup | 5ms | 20ms |
| Vector Search | 100ms | 500ms |
| Agent Task | 2s | 10s |

### Throughput Requirements

| Service | RPS Target | Peak RPS |
|---------|-----------|----------|
| API Gateway | 10,000 | 50,000 |
| Backend API | 5,000 | 25,000 |
| WebSocket | 1,000 | 5,000 |
| Vector Search | 100 | 500 |
| Agent Tasks | 50 | 200 |

## Related Documentation

- [API Reference](./api_reference.md)
- [Data Flow](./data_flow.md)
- [Database Schema](./database_schema.md)
- [Security Model](./security_model.md)
- [Scalability Plan](./scalability_plan.md)

## Appendix

### Technology Stack Summary

| Layer | Technology | Version | Purpose |
|-------|-----------|---------|---------|
| Frontend | Streamlit | 1.29.0 | User interface |
| Backend | FastAPI | 0.109.0 | API server |
| Gateway | Kong | 3.5 | API management |
| Database | PostgreSQL | 16.1 | Primary data |
| Cache | Redis | 7.2.3 | Caching/sessions |
| Graph | Neo4j | 5.15.0 | Relationships |
| Queue | RabbitMQ | 3.12.10 | Message broker |
| Vector | ChromaDB | 0.4.22 | Embeddings |
| Container | Docker | 24.0 | Containerization |
| Orchestration | Kubernetes | 1.28 | Container orchestration |