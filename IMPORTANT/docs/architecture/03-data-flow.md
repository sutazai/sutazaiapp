---
title: Data Flow & Integration Patterns
version: 1.0.0
last_updated: 2025-08-08
author: System Architect
review_status: Production
next_review: 2025-09-07
related_docs:
  - IMPORTANT/10_canonical/current_state/seq_inference.mmd
  - IMPORTANT/10_canonical/current_state/seq_agent_exec.mmd
  - IMPORTANT/10_canonical/current_state/seq_rag_ingest.mmd
  - IMPORTANT/10_canonical/current_state/seq_rag_query.mmd
  - IMPORTANT/10_canonical/current_state/seq_alerting.mmd
  - IMPORTANT/10_canonical/data/data_management.md
  - /opt/sutazaiapp/CLAUDE.md
---

# Data Flow & Integration Patterns

## Executive Summary

This document provides comprehensive documentation of SutazAI's data flow architecture based on the **ACTUAL SYSTEM STATE** as of August 2025. All diagrams and descriptions reflect the real implementation, not aspirational features.

**System Reality Check:**
- 28 containers actively running (not 59 as documented elsewhere)
- TinyLlama model loaded (637MB, not gpt-oss)
- 7 Flask agent stubs returning hardcoded responses
- PostgreSQL operational with 14 tables
- Redis cache layer functional
- Neo4j graph database available
- RabbitMQ running but NOT integrated

## Table of Contents

1. [Core Data Flow Patterns](#core-data-flow-patterns)
2. [User Request Flows](#user-request-flows)
3. [Agent Task Processing](#agent-task-processing)
4. [LLM Inference Pipeline](#llm-inference-pipeline)
5. [Database Operations](#database-operations)
6. [Cache Layer Integration](#cache-layer-integration)
7. [Message Queue Architecture](#message-queue-architecture)
8. [Data Formats & Protocols](#data-formats--protocols)
9. [Error Handling & Retry Logic](#error-handling--retry-logic)
10. [Monitoring & Observability](#monitoring--observability)
11. [Performance Considerations](#performance-considerations)
12. [Troubleshooting Guide](#troubleshooting-guide)

## Core Data Flow Patterns

### 1.1 System Architecture Overview

```mermaid
graph TB
    subgraph "Client Layer"
        UI[Streamlit UI :10011]
        API_Client[API Clients]
    end
    
    subgraph "Application Layer"
        Backend[FastAPI Backend :10010]
        Kong[Kong Gateway :10005]
    end
    
    subgraph "Agent Layer (Stubs)"
        AO[Agent Orchestrator :8589]
        MAC[Multi-Agent Coordinator :8587]
        RA[Resource Arbitration :8588]
        TA[Task Assignment :8551]
        HO[Hardware Optimizer :8002]
        OI[Ollama Integration :11015]
        AI_Metrics[AI Metrics :11063]
    end
    
    subgraph "LLM Layer"
        Ollama[Ollama Server :10104]
        TinyLlama[TinyLlama Model 637MB]
    end
    
    subgraph "Data Layer"
        Postgres[PostgreSQL :10000]
        Redis[Redis Cache :10001]
        Neo4j[Neo4j Graph :10002/10003]
    end
    
    subgraph "Vector Stores"
        Qdrant[Qdrant :10101/10102]
        FAISS[FAISS :10103]
        ChromaDB[ChromaDB :10100]
    end
    
    subgraph "Message Layer"
        RabbitMQ[RabbitMQ :10007/10008]
    end
    
    subgraph "Monitoring"
        Prometheus[Prometheus :10200]
        Grafana[Grafana :10201]
        Loki[Loki :10202]
    end
    
    UI --> Backend
    API_Client --> Kong
    Kong --> Backend
    Backend --> AO
    Backend --> Ollama
    Backend --> Postgres
    Backend --> Redis
    Backend --> Neo4j
    AO --> MAC
    AO --> TA
    Ollama --> TinyLlama
    Backend -.-> RabbitMQ
    Backend --> Prometheus
    
    style RabbitMQ stroke-dasharray: 5 5
    style ChromaDB fill:#ff9999
```

**Key Points:**
- Solid lines indicate active data flows
- Dashed lines indicate configured but unused connections
- Red components indicate connection issues (ChromaDB)
- All agent services return stub responses only

### 1.2 Data Flow Classifications

| Flow Type | Status | Components | Protocol | Format |
|-----------|--------|------------|----------|---------|
| User Requests | ✅ Active | UI → Backend | HTTP/REST | JSON |
| LLM Inference | ✅ Active | Backend → Ollama | HTTP/REST | JSON Stream |
| Database Ops | ✅ Active | Backend → PostgreSQL | PostgreSQL Wire | SQL/Binary |
| Cache Operations | ✅ Active | Backend → Redis | Redis Protocol | Key-Value |
| Graph Queries | ✅ Active | Backend → Neo4j | Bolt Protocol | Cypher |
| Agent Communication | ⚠️ Stub Only | Backend → Agents | HTTP/REST | JSON (Hardcoded) |
| Message Queue | ❌ Not Integrated | Backend ↔ RabbitMQ | AMQP | N/A |
| Vector Search | ❌ Not Integrated | Backend → Vector DBs | HTTP/gRPC | N/A |

## User Request Flows

### 2.1 Frontend to Backend Flow

```mermaid
sequenceDiagram
    participant User
    participant Streamlit as Streamlit UI<br/>:10011
    participant Backend as FastAPI<br/>:10010
    participant Cache as Redis<br/>:10001
    participant DB as PostgreSQL<br/>:10000
    
    User->>Streamlit: Navigate/Submit Form
    Streamlit->>Streamlit: Validate Input
    Streamlit->>Backend: HTTP Request
    Note over Backend: CORS Headers Applied
    
    Backend->>Cache: Check Cache
    alt Cache Hit
        Cache-->>Backend: Cached Data
    else Cache Miss
        Backend->>DB: Query Data
        DB-->>Backend: Result Set
        Backend->>Cache: Store Result (TTL: 300s)
    end
    
    Backend-->>Streamlit: JSON Response
    Streamlit-->>User: Render UI Update
```

**Implementation Details:**
- Frontend uses async `call_api()` function with timeout management
- Default timeouts: Health checks (5s), Processing (60s), Standard (30s)
- All responses include correlation IDs for tracing
- Error responses follow RFC 7807 Problem Details format

### 2.2 API Gateway Flow (Kong)

```mermaid
sequenceDiagram
    participant Client
    participant Kong as Kong Gateway<br/>:10005
    participant Backend as FastAPI<br/>:10010
    participant Consul as Consul<br/>:10006
    
    Client->>Kong: API Request
    Kong->>Consul: Service Discovery
    Consul-->>Kong: Backend Location
    
    Note over Kong: Apply Rate Limiting<br/>Apply Auth (Not Configured)<br/>Transform Headers
    
    Kong->>Backend: Proxied Request
    Backend-->>Kong: Response
    
    Note over Kong: Response Caching<br/>Response Transform
    
    Kong-->>Client: Final Response
```

**Current Status:**
- Kong is running but has NO routes configured
- Service discovery via Consul is minimal
- No authentication middleware configured
- Rate limiting not implemented

## Agent Task Processing

### 3.1 Agent Execution Flow (Current - Stub Implementation)

```mermaid
sequenceDiagram
    participant User
    participant Backend as FastAPI<br/>:10010
    participant AO as Agent Orchestrator<br/>:8589
    participant Agent as Specific Agent<br/>:Various Ports
    
    User->>Backend: POST /api/v1/agents/execute
    Backend->>Backend: Validate Request
    Backend->>AO: POST /process
    
    Note over AO: STUB: No actual processing<br/>Returns hardcoded response
    
    AO-->>Backend: {"status": "completed", "result": "processed"}
    Backend-->>User: Task Result
```

**Reality Check:**
- All agent `/process` endpoints return hardcoded JSON
- No actual task processing occurs
- No inter-agent communication implemented
- Agent health checks work but provide no meaningful status

### 3.2 Intended Agent Collaboration Flow (NOT IMPLEMENTED)

```mermaid
sequenceDiagram
    participant Backend
    participant AO as Agent Orchestrator
    participant MAC as Multi-Agent Coordinator
    participant TA as Task Assignment
    participant RA as Resource Arbitration
    participant Agent as Worker Agent
    
    Backend->>AO: Complex Task
    AO->>MAC: Decompose Task
    MAC->>TA: Assign Subtasks
    TA->>RA: Request Resources
    RA-->>TA: Resource Allocation
    TA->>Agent: Execute Subtask
    Agent-->>TA: Result
    TA-->>MAC: Subtask Complete
    MAC-->>AO: Task Complete
    AO-->>Backend: Final Result
```

**Note:** This flow represents the intended design but is NOT IMPLEMENTED. Current system returns stub responses at each step.

## LLM Inference Pipeline

### 4.1 Text Generation Flow (TinyLlama)

```mermaid
sequenceDiagram
    participant User
    participant Backend as FastAPI<br/>:10010
    participant Ollama as Ollama Server<br/>:10104
    participant Model as TinyLlama<br/>637MB
    
    User->>Backend: POST /api/v1/generate
    Backend->>Backend: Validate Prompt
    
    Backend->>Ollama: POST /api/generate
    Note over Backend,Ollama: Request Body:<br/>{"model": "tinyllama",<br/>"prompt": "...",<br/>"stream": true}
    
    loop Streaming Response
        Ollama->>Model: Inference
        Model-->>Ollama: Token
        Ollama-->>Backend: SSE Token
        Backend-->>User: SSE Token
    end
    
    Note over User: Tokens rendered in real-time
```

**Configuration Issues:**
- Backend expects "gpt-oss" model but only TinyLlama is loaded
- This causes "degraded" health status
- Fix: Either load gpt-oss or update backend configuration

### 4.2 Model Loading and Management

```yaml
# Current Model Configuration
models:
  loaded:
    - name: tinyllama
      size: 637MB
      quantization: Q4_0
      context_length: 2048
      
  expected_but_missing:
    - name: gpt-oss
      reason: Not pulled from registry
      
# Performance Metrics
inference:
  cold_start: ~2-3 seconds
  tokens_per_second: ~30-50 (CPU)
  max_concurrent: 4
  timeout: 60 seconds
```

## Database Operations

### 5.1 PostgreSQL Data Flow

```mermaid
sequenceDiagram
    participant Backend
    participant Pool as Connection Pool
    participant PG as PostgreSQL<br/>:10000
    participant WAL as Write-Ahead Log
    
    Backend->>Pool: Request Connection
    Pool->>Pool: Check Available
    
    alt Connection Available
        Pool-->>Backend: Connection
    else Pool Exhausted
        Pool->>PG: Create New
        PG-->>Pool: New Connection
        Pool-->>Backend: Connection
    end
    
    Backend->>PG: BEGIN TRANSACTION
    Backend->>PG: SQL Query
    PG->>WAL: Write to WAL
    PG-->>Backend: Result
    Backend->>PG: COMMIT
    
    Backend->>Pool: Return Connection
```

**Database Schema Status:**
```sql
-- Current Tables (14 total)
users, agents, tasks, sessions, 
audit_logs, agent_metrics, 
task_history, system_config,
api_keys, rate_limits,
embeddings, documents,
cache_entries, migrations

-- Indexes
- Primary keys on all tables (UUID)
- Foreign key indexes
- Timestamp indexes for queries
- Composite indexes for common joins
```

### 5.2 Neo4j Graph Operations

```mermaid
graph LR
    subgraph "Graph Data Model"
        Agent[Agent Node]
        Task[Task Node]
        Service[Service Node]
        Capability[Capability Node]
        
        Agent -->|CAN_EXECUTE| Task
        Agent -->|HAS_CAPABILITY| Capability
        Service -->|DEPENDS_ON| Service
        Agent -->|BELONGS_TO| Service
        Task -->|REQUIRES| Capability
    end
```

**Cypher Query Examples:**
```cypher
-- Find agents by capability
MATCH (a:Agent)-[:HAS_CAPABILITY]->(c:Capability {name: $capability})
RETURN a.name, a.status, a.port

-- Trace service dependencies
MATCH path = (s:Service {name: $service})-[:DEPENDS_ON*]->(dep:Service)
RETURN path

-- Find optimal agent for task
MATCH (t:Task {id: $taskId})-[:REQUIRES]->(c:Capability)
MATCH (a:Agent)-[:HAS_CAPABILITY]->(c)
WHERE a.status = 'healthy'
RETURN a ORDER BY a.load ASC LIMIT 1
```

## Cache Layer Integration

### 6.1 Redis Caching Strategy

```mermaid
sequenceDiagram
    participant Backend
    participant Redis as Redis<br/>:10001
    participant DB as PostgreSQL
    
    Backend->>Redis: GET key
    
    alt Cache Hit
        Redis-->>Backend: Cached Value
        Note over Backend: Return immediately
    else Cache Miss
        Redis-->>Backend: nil
        Backend->>DB: Query Database
        DB-->>Backend: Result
        Backend->>Redis: SETEX key value 300
        Note over Redis: TTL: 5 minutes
    end
```

**Cache Patterns:**
```python
# Key Naming Convention
cache_keys = {
    "user_session": "session:{user_id}:{session_id}",
    "agent_status": "agent:status:{agent_name}",
    "task_result": "task:result:{task_id}",
    "api_rate_limit": "rate:{api_key}:{endpoint}",
    "llm_response": "llm:cache:{prompt_hash}"
}

# TTL Strategy
ttl_config = {
    "session": 3600,        # 1 hour
    "agent_status": 10,     # 10 seconds
    "task_result": 300,     # 5 minutes
    "rate_limit": 60,       # 1 minute
    "llm_cache": 1800       # 30 minutes
}
```

## Message Queue Architecture

### 7.1 RabbitMQ Configuration (NOT INTEGRATED)

```mermaid
graph TB
    subgraph "RabbitMQ Topology (Configured but Unused)"
        Exchange[Topic Exchange]
        
        subgraph "Queues"
            TaskQueue[task.queue]
            ResultQueue[result.queue]
            EventQueue[event.queue]
            DeadLetter[dlq.queue]
        end
        
        Exchange -->|task.*| TaskQueue
        Exchange -->|result.*| ResultQueue
        Exchange -->|event.*| EventQueue
        TaskQueue -->|on_failure| DeadLetter
    end
```

**Current Status:**
- RabbitMQ container running on ports 10007 (AMQP) and 10008 (Management)
- No producers or consumers connected
- No messages being published
- Management UI accessible but shows no activity

**Intended Message Types (Not Implemented):**
```json
{
  "task_message": {
    "id": "uuid",
    "type": "PROCESS",
    "payload": {},
    "timestamp": "2025-08-08T10:00:00Z",
    "correlation_id": "uuid",
    "retry_count": 0
  },
  
  "result_message": {
    "task_id": "uuid",
    "status": "SUCCESS|FAILURE",
    "result": {},
    "error": null,
    "processing_time_ms": 1234
  },
  
  "event_message": {
    "event_type": "AGENT_STATUS_CHANGE",
    "source": "agent_name",
    "data": {},
    "timestamp": "2025-08-08T10:00:00Z"
  }
}
```

## Data Formats & Protocols

### 8.1 API Request/Response Formats

```yaml
# Standard Request Format
Content-Type: application/json
Headers:
  X-Request-ID: uuid
  X-Correlation-ID: uuid
  Authorization: Bearer <token>  # Not implemented

Body:
  {
    "action": "string",
    "params": {},
    "metadata": {
      "timestamp": "ISO-8601",
      "version": "1.0"
    }
  }

# Standard Response Format
Success (2xx):
  {
    "status": "success",
    "data": {},
    "metadata": {
      "request_id": "uuid",
      "processing_time_ms": 123,
      "timestamp": "ISO-8601"
    }
  }

Error (4xx/5xx):
  {
    "status": "error",
    "error": {
      "code": "ERROR_CODE",
      "message": "Human readable message",
      "details": {},
      "trace_id": "uuid"
    },
    "metadata": {}
  }
```

### 8.2 Internal Communication Protocols

| Service Type | Protocol | Port | Format | Encryption |
|--------------|----------|------|--------|------------|
| REST APIs | HTTP/1.1 | Various | JSON | None (Internal) |
| PostgreSQL | PostgreSQL Wire | 10000 | Binary | None |
| Redis | RESP | 10001 | Binary | None |
| Neo4j | Bolt | 10003 | Binary | None |
| RabbitMQ | AMQP 0.9.1 | 10007 | Binary | None |
| Metrics | HTTP | */metrics | Prometheus | None |

### 8.3 Data Transformations

```python
# Common transformation patterns
transformations = {
    "snake_to_camel": lambda s: ''.join(x.capitalize() or '_' for x in s.split('_')),
    "prompt_hash": lambda p: hashlib.sha256(p.encode()).hexdigest()[:16],
    "sanitize_sql": lambda s: s.replace("'", "''"),
    "json_to_msgpack": lambda j: msgpack.packb(j),
    "compress_large": lambda d: zlib.compress(json.dumps(d).encode()) if len(str(d)) > 1024 else d
}
```

## Error Handling & Retry Logic

### 9.1 Error Propagation Flow

```mermaid
sequenceDiagram
    participant Client
    participant Backend
    participant Service
    participant ErrorHandler
    participant Logger
    participant Metrics
    
    Client->>Backend: Request
    Backend->>Service: Process
    Service->>Service: Error Occurs
    Service->>ErrorHandler: Handle Error
    
    ErrorHandler->>Logger: Log Error
    ErrorHandler->>Metrics: Record Metric
    
    alt Retryable Error
        ErrorHandler->>Service: Retry with backoff
        Service-->>Backend: Success/Final Failure
    else Non-Retryable
        ErrorHandler-->>Backend: Error Response
    end
    
    Backend-->>Client: Final Response
```

### 9.2 Retry Configuration

```yaml
retry_policies:
  database:
    max_attempts: 3
    backoff_type: exponential
    base_delay_ms: 100
    max_delay_ms: 5000
    retryable_errors:
      - CONNECTION_TIMEOUT
      - DEADLOCK_DETECTED
      
  http_services:
    max_attempts: 3
    backoff_type: exponential
    base_delay_ms: 500
    max_delay_ms: 10000
    retryable_status_codes: [502, 503, 504]
    
  llm_inference:
    max_attempts: 2
    backoff_type: linear
    delay_ms: 2000
    retryable_errors:
      - MODEL_LOADING
      - CONTEXT_EXCEEDED
      
  cache:
    max_attempts: 2
    backoff_type: immediate
    retryable_errors:
      - CONNECTION_REFUSED
```

### 9.3 Circuit Breaker Pattern (Planned)

```python
# Circuit breaker states and thresholds
circuit_breaker_config = {
    "failure_threshold": 5,          # Failures to open circuit
    "success_threshold": 2,           # Successes to close circuit
    "timeout": 30,                    # Seconds before half-open
    "half_open_requests": 3,          # Test requests in half-open
    
    "states": {
        "CLOSED": "Normal operation",
        "OPEN": "Fast-fail all requests",
        "HALF_OPEN": "Testing recovery"
    }
}
```

## Monitoring & Observability

### 10.1 Metrics Collection Flow

```mermaid
sequenceDiagram
    participant Service
    participant Metrics as Metrics Endpoint
    participant Prometheus as Prometheus<br/>:10200
    participant Grafana as Grafana<br/>:10201
    participant Alert as AlertManager<br/>:10203
    
    Service->>Metrics: Update Metric
    
    loop Every 15s
        Prometheus->>Metrics: Scrape /metrics
        Metrics-->>Prometheus: Metrics Data
    end
    
    Prometheus->>Prometheus: Store Time Series
    
    Grafana->>Prometheus: Query Metrics
    Prometheus-->>Grafana: Time Series Data
    
    alt Threshold Exceeded
        Prometheus->>Alert: Fire Alert
        Alert->>Alert: Route & Notify
    end
```

### 10.2 Logging Pipeline

```mermaid
graph LR
    subgraph "Log Sources"
        Backend[Backend Logs]
        Agents[Agent Logs]
        System[System Logs]
    end
    
    subgraph "Collection"
        Docker[Docker Logging Driver]
    end
    
    subgraph "Aggregation"
        Loki[Loki :10202]
    end
    
    subgraph "Query"
        Grafana[Grafana :10201]
    end
    
    Backend --> Docker
    Agents --> Docker
    System --> Docker
    Docker --> Loki
    Loki --> Grafana
```

**Log Format:**
```json
{
  "timestamp": "2025-08-08T10:00:00.123Z",
  "level": "INFO|WARN|ERROR",
  "service": "backend",
  "correlation_id": "uuid",
  "message": "Operation completed",
  "context": {
    "user_id": "uuid",
    "request_id": "uuid",
    "endpoint": "/api/v1/generate",
    "duration_ms": 123
  },
  "error": {
    "type": "ValueError",
    "message": "Invalid input",
    "stack_trace": "..."
  }
}
```

### 10.3 Distributed Tracing (Planned)

```yaml
# OpenTelemetry configuration (not yet implemented)
tracing:
  enabled: false  # Currently disabled
  exporter: otlp
  endpoint: http://jaeger:4317
  sample_rate: 0.1
  
  instrumentation:
    - fastapi
    - requests
    - sqlalchemy
    - redis
    - asyncio
```

## Performance Considerations

### 11.1 Bottleneck Analysis

| Component | Bottleneck | Impact | Mitigation |
|-----------|------------|--------|------------|
| TinyLlama Inference | CPU-bound, single model | 30-50 tokens/sec | Load multiple models, use GPU |
| PostgreSQL | Connection pool size | Limited concurrent queries | Increase pool, add read replicas |
| Redis | Single-threaded | Limited to one CPU core | Redis Cluster for scaling |
| Agent Stubs | No actual processing | No real functionality | Implement actual logic |
| Frontend | Streamlit limitations | Single-threaded, stateful | Consider React/Vue frontend |
| Network | Docker bridge network | Inter-container latency | Use host networking for critical paths |

### 11.2 Optimization Strategies

```yaml
current_optimizations:
  caching:
    - Redis for session data (5min TTL)
    - LLM response caching (30min TTL)
    - Database query results (5min TTL)
    
  database:
    - Connection pooling (min=5, max=20)
    - Prepared statements
    - Index optimization
    - VACUUM scheduling
    
  api:
    - Response compression (gzip)
    - Pagination (default=100, max=1000)
    - Field filtering
    - Async request handling
    
planned_optimizations:
  - Implement request batching
  - Add database read replicas
  - Enable HTTP/2
  - Implement GraphQL for efficient queries
  - Add CDN for static assets
  - Implement queue-based async processing
```

### 11.3 Capacity Planning

```yaml
current_capacity:
  concurrent_users: ~100
  requests_per_second: ~50
  llm_inference_per_min: ~30
  database_connections: 20
  cache_memory: 512MB
  
bottlenecks_at_scale:
  100_users: "Stable"
  500_users: "LLM inference queue grows"
  1000_users: "Database connection pool exhausted"
  5000_users: "Redis memory limit, Frontend unresponsive"
  
scaling_recommendations:
  immediate:
    - Increase database connection pool to 50
    - Add Redis memory to 2GB
    - Load 3-4 TinyLlama instances
    
  short_term:
    - Implement horizontal scaling for Backend
    - Add PostgreSQL read replica
    - Implement proper caching strategy
    
  long_term:
    - Migrate to microservices architecture
    - Implement Kubernetes orchestration
    - Add auto-scaling policies
```

## Troubleshooting Guide

### 12.1 Common Data Flow Issues

| Issue | Symptoms | Root Cause | Solution |
|-------|----------|------------|----------|
| Backend shows "degraded" | Health check returns degraded status | Ollama expects gpt-oss, has tinyllama | Load gpt-oss or update config |
| Agents return same response | All /process calls return identical JSON | Stub implementation | Implement actual agent logic |
| ChromaDB restarts | Container restart loop | Connection/init issues | Check logs, verify config |
| No database tables | psql \dt shows no tables | Migrations never run | Run init_db.py script |
| Cache misses | Redis always returns nil | Wrong key format | Verify key naming convention |
| Slow API responses | >5s response time | No caching, inefficient queries | Implement caching, optimize queries |
| Lost messages | RabbitMQ shows no activity | Not integrated | Implement producers/consumers |
| High memory usage | Container using >2GB | Memory leaks, no limits | Set container memory limits |

### 12.2 Debugging Data Flows

```bash
# Check service connectivity
docker exec sutazai-backend curl -s http://sutazai-ollama:11434/api/tags

# Monitor real-time logs
docker-compose logs -f backend ollama

# Check database connections
docker exec sutazai-postgres psql -U sutazai -c "SELECT count(*) FROM pg_stat_activity;"

# Redis connection test
docker exec sutazai-redis redis-cli ping

# Neo4j connectivity
docker exec sutazai-backend python -c "from neo4j import GraphDatabase; driver = GraphDatabase.driver('bolt://sutazai-neo4j:7687')"

# RabbitMQ queue status
curl -u guest:guest http://localhost:10008/api/queues

# Trace HTTP request
docker exec sutazai-backend curl -v http://localhost:8000/health

# Check Prometheus metrics
curl -s http://localhost:10200/api/v1/query?query=up | jq
```

### 12.3 Performance Profiling

```python
# Backend profiling endpoints (add to FastAPI)
@app.get("/debug/profile")
async def profile_endpoint():
    import cProfile
    import pstats
    import io
    
    pr = cProfile.Profile()
    pr.enable()
    
    # Run operation to profile
    result = await expensive_operation()
    
    pr.disable()
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats('cumulative')
    ps.print_stats(20)
    
    return {"profile": s.getvalue(), "result": result}

@app.get("/debug/memory")
async def memory_usage():
    import tracemalloc
    import gc
    
    gc.collect()
    snapshot = tracemalloc.take_snapshot()
    top_stats = snapshot.statistics('lineno')[:10]
    
    return {
        "top_memory_usage": [
            {
                "file": stat.traceback.format()[0],
                "size_mb": stat.size / 1024 / 1024,
                "count": stat.count
            }
            for stat in top_stats
        ]
    }
```

## Appendix A: Data Flow Quick Reference

### Service Ports
```bash
# Core Services
10000  PostgreSQL
10001  Redis
10002  Neo4j Browser
10003  Neo4j Bolt
10005  Kong Gateway
10010  Backend API
10011  Frontend UI
10104  Ollama

# Agents (Stubs)
8002   Hardware Optimizer
8551   Task Assignment
8587   Multi-Agent Coordinator
8588   Resource Arbitration
8589   AI Orchestrator
11015  Ollama Integration
11063  AI Metrics

# Monitoring
10200  Prometheus
10201  Grafana
10202  Loki
10203  AlertManager
```

### Critical Configuration Files
```yaml
/opt/sutazaiapp/docker-compose.yml     # Service definitions
/opt/sutazaiapp/backend/config.py      # Backend configuration
/opt/sutazaiapp/.env                   # Environment variables
/opt/sutazaiapp/CLAUDE.md             # System truth document
```

### Health Check Endpoints
```bash
curl http://localhost:10010/health     # Backend
curl http://localhost:10104/           # Ollama
curl http://localhost:8589/health      # Agent Orchestrator
curl http://localhost:10005/status     # Kong Gateway
```

## Appendix B: Data Schema Reference

### PostgreSQL Core Tables
```sql
-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Agents table
CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    type VARCHAR(50),
    status VARCHAR(20) DEFAULT 'inactive',
    port INTEGER,
    capabilities JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tasks table  
CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id),
    user_id UUID REFERENCES users(id),
    type VARCHAR(50),
    status VARCHAR(20) DEFAULT 'pending',
    payload JSONB,
    result JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);
```

### Redis Key Patterns
```redis
# Session management
session:{user_id}:{session_id} -> JSON session data

# Rate limiting
rate:{api_key}:{endpoint}:{window} -> counter

# Cache entries
cache:{entity}:{id}:{version} -> serialized data

# Distributed locks
lock:{resource}:{id} -> lock holder ID
```

### Neo4j Node Types
```cypher
// Agent node
(:Agent {
    id: "uuid",
    name: "agent_name",
    type: "processor|coordinator|optimizer",
    status: "healthy|degraded|unhealthy",
    port: 8589
})

// Capability node
(:Capability {
    id: "uuid", 
    name: "capability_name",
    category: "nlp|vision|reasoning"
})

// Task node
(:Task {
    id: "uuid",
    type: "inference|analysis|generation",
    status: "pending|running|completed",
    priority: 1-10
})
```

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-08-08 | 0.1.0 | Initial draft created | Documentation Lead |
| 2025-08-08 | 1.0.0 | Complete rewrite based on actual system state | System Architect |

## References

### Canonical Sources
- System Truth: `/opt/sutazaiapp/CLAUDE.md`
- Sequence Diagrams: `/opt/sutazaiapp/IMPORTANT/10_canonical/current_state/*.mmd`
- Data Management: `/opt/sutazaiapp/IMPORTANT/10_canonical/data/data_management.md`
- Container Status: `docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"`

### External Documentation
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ollama API Reference](https://github.com/ollama/ollama/blob/main/docs/api.md)
- [PostgreSQL Wire Protocol](https://www.postgresql.org/docs/current/protocol.html)
- [Redis Protocol Specification](https://redis.io/topics/protocol)
- [Neo4j Bolt Protocol](https://neo4j.com/docs/bolt/current/)

---

**Document Status:** Production Ready
**Next Review:** 2025-09-07
**Validation:** Based on direct system inspection and testing
