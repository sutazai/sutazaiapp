---
title: SutazAI Component Architecture  
version: 1.0.0
last_updated: 2025-08-08
author: System Architect
review_status: Draft
next_review: 2025-08-15
related_docs:
  - 01-system-overview.md
  - 03-data-flow.md
  - /IMPORTANT/10_canonical/current_state/components_backend.mmd
  - /IMPORTANT/10_canonical/current_state/erd_current.mmd
  - /IMPORTANT/10_canonical/target_state/components.mmd
---

# SutazAI Component Architecture

## Overview

This document defines the component-level architecture (C4 Level 3) for the SutazAI platform, detailing responsibilities, interfaces, and dependencies of each system component. Components are grouped by architectural layer following Domain-Driven Design principles.

## Prerequisites

- Understanding of C4 model component diagrams
- Familiarity with FastAPI and Flask frameworks
- Knowledge of microservices patterns
- Review of ERD and sequence diagrams in `/IMPORTANT/10_canonical/current_state/`

## Implementation Details

### Backend API Components

The Backend API (FastAPI v17.0.0) serves as the primary orchestration layer:

#### Core Components

| Component | Responsibility | Dependencies | Status |
|-----------|---------------|--------------|--------|
| **API Gateway** | Request routing, validation | FastAPI, Pydantic | ✅ Operational |
| **Auth Service** | JWT tokens, RBAC | Not implemented | ❌ Missing (ISSUE-0005) |
| **Agent Orchestrator** | Task distribution, coordination | RabbitMQ, Agents | ⚠️ Partial (stubs only) |
| **LLM Service** | Ollama integration, prompt management | Ollama API | ✅ Working |
| **Data Service** | CRUD operations, query optimization | PostgreSQL, Redis | ✅ Operational |
| **Vector Service** | Embedding generation, similarity search | Qdrant, ChromaDB | ❌ Not integrated |

#### Component Interactions

```mermaid
graph TD
    subgraph "Backend API"
        Gateway[API Gateway<br/>FastAPI]
        Auth[Auth Service<br/>JWT/RBAC]
        Orchestrator[Agent Orchestrator]
        LLM[LLM Service]
        Data[Data Service]
        Vector[Vector Service]
        
        Gateway --> Auth
        Gateway --> Orchestrator
        Gateway --> Data
        Orchestrator --> LLM
        Orchestrator --> Vector
        Data --> Cache[Redis Cache]
    end
    
    subgraph "External Services"
        Ollama[Ollama<br/>TinyLlama]
        PG[(PostgreSQL)]
        Redis[(Redis)]
        Qdrant[(Qdrant)]
    end
    
    LLM --> Ollama
    Data --> PG
    Cache --> Redis
    Vector --> Qdrant
```

### Agent Components

Seven agent services deployed as Flask applications (currently stubs):

| Agent | Port | Purpose | Implementation Status |
|-------|------|---------|----------------------|
| **Task Assignment Coordinator** | 8551 | Distribute tasks to agents | ⚠️ Basic functionality |
| **AI Agent Orchestrator** | 8589 | High-level coordination | ❌ Stub only |
| **Multi-Agent Coordinator** | 8587 | Inter-agent communication | ❌ Stub only |
| **Resource Arbitration** | 8588 | Resource allocation | ❌ Stub only |
| **Hardware Optimizer** | 8002 | Performance tuning | ❌ Stub only |
| **Ollama Integration** | 11015 | LLM wrapper | ⚠️ May work |
| **AI Metrics Exporter** | 11063 | Metrics collection | ❌ Unhealthy |

#### Agent Communication Pattern

```python
# Current (Stub) Implementation
@app.route('/process', methods=['POST'])
def process():
    return {"status": "success", "result": "processed"}

# Target Implementation
@app.route('/process', methods=['POST'])
def process():
    task = request.json
    # 1. Validate task
    # 2. Process with LLM
    # 3. Store results
    # 4. Emit events
    return {"status": "success", "result": actual_result}
```

### Frontend Components

Streamlit-based UI with these logical components:

| Component | Responsibility | Status |
|-----------|---------------|--------|
| **Auth UI** | Login/logout forms | ❌ Not implemented (ISSUE-0014) |
| **Dashboard** | System overview, metrics | ⚠️ Basic layout only |
| **Agent Interface** | Task submission, results | ⚠️   functionality |
| **Settings** | User preferences, config | ❌ Not implemented |
| **Admin Panel** | System management | ❌ Not implemented |

### Data Layer Components

#### PostgreSQL Schema (Current State)

```sql
-- Current: SERIAL PKs (violates ADR-0001)
CREATE TABLE users (
    id SERIAL PRIMARY KEY,  -- Should be UUID
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP
);

CREATE TABLE agents (
    id SERIAL PRIMARY KEY,  -- Should be UUID
    name VARCHAR(100),
    status VARCHAR(50)
);

CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,  -- Should be UUID
    agent_id INTEGER REFERENCES agents(id),
    payload JSONB,
    status VARCHAR(50)
);
```

#### Target Schema (UUID-based)

```sql
-- Target: UUID PKs per ADR-0001
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE agents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(100) UNIQUE NOT NULL,
    type VARCHAR(50) NOT NULL,
    config JSONB DEFAULT '{}',
    status VARCHAR(50) DEFAULT 'inactive'
);

CREATE TABLE tasks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    agent_id UUID REFERENCES agents(id),
    user_id UUID REFERENCES users(id),
    payload JSONB NOT NULL,
    result JSONB,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW(),
    completed_at TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_tasks_agent_id ON tasks(agent_id);
CREATE INDEX idx_tasks_user_id ON tasks(user_id);
CREATE INDEX idx_tasks_status ON tasks(status);
```

### Infrastructure Components

#### Service Mesh (Kong Gateway)

Currently unconfigured. Target configuration:

```yaml
# kong.yml (target)
services:
  - name: backend-api
    url: http://backend:8000
    routes:
      - name: api-route
        paths: ["/api"]
        methods: ["GET", "POST", "PUT", "DELETE"]
    plugins:
      - name: jwt
      - name: rate-limiting
        config:
          minute: 100
      - name: cors
```

#### Message Queue (RabbitMQ)

```python
# Target implementation for agent communication
import pika

# Publisher (Backend API)
def publish_task(task_id, agent_type, payload):
    connection = pika.BlockingConnection(
        pika.ConnectionParameters('rabbitmq')
    )
    channel = connection.channel()
    channel.queue_declare(queue=f'agent.{agent_type}')
    channel.basic_publish(
        exchange='',
        routing_key=f'agent.{agent_type}',
        body=json.dumps({
            'task_id': task_id,
            'payload': payload
        })
    )

# Consumer (Agent)
def consume_tasks(agent_type):
    channel.basic_consume(
        queue=f'agent.{agent_type}',
        on_message_callback=process_task,
        auto_ack=True
    )
    channel.start_consuming()
```

## Configuration

### Environment Variables Required

```bash
# Database
DATABASE_URL=postgresql://sutazai:password@postgres:5432/sutazai
REDIS_URL=redis://redis:6379

# LLM
OLLAMA_BASE_URL=http://ollama:11434
DEFAULT_MODEL=tinyllama  # Not gpt-oss

# Security (not yet implemented)
JWT_SECRET_KEY=<generate-secure-key>
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24

# Service Discovery
CONSUL_URL=http://consul:8500
KONG_ADMIN_URL=http://kong:8001

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_URL=http://grafana:3000
```

## Testing & Validation

### Component Testing Strategy

| Component | Test Type | Coverage Target | Current |
|-----------|-----------|-----------------|---------|
| API Gateway | Unit + Integration | 90% | 0% |
| Auth Service | Unit + Security | 95% | 0% |
| Agent Orchestrator | Unit + Integration | 85% | 0% |
| Data Service | Unit + Integration | 80% | 0% |
| Agents | Unit + Contract | 75% | 0% |
| Frontend | Component + E2E | 70% | 0% |

### Validation Checklist

- [ ] Each component has defined interfaces
- [ ] Dependencies are explicitly declared
- [ ] Error handling paths documented
- [ ] Performance metrics identified
- [ ] Security boundaries established

## Troubleshooting

### Common Component Issues

| Issue | Component | Root Cause | Solution |
|-------|-----------|------------|----------|
| Circular dependencies | Backend services | Poor separation of concerns | Apply DDD boundaries |
| Memory leaks | Agent services | No connection pooling | Implement connection pools |
| Slow queries | Data service | Missing indexes | Add indexes per queries |
| Message loss | RabbitMQ | No persistence | Enable durable queues |
| Cache misses | Redis | Poor key strategy | Implement cache warming |

### Architectural Smells

1. **Tight Coupling**: Agents directly calling backend DB
2. **Hidden Dependencies**: Hardcoded service URLs
3. **Missing Abstractions**: No repository pattern
4. **Synchronous Everything**: No async processing
5. **No Circuit Breakers**: Services fail in cascade

## References

### Canonical Sources
- [Backend Components Diagram](/opt/sutazaiapp/IMPORTANT/10_canonical/current_state/components_backend.mmd)
- [Current ERD](/opt/sutazaiapp/IMPORTANT/10_canonical/current_state/erd_current.mmd)
- [Target Components](/opt/sutazaiapp/IMPORTANT/10_canonical/target_state/components.mmd)
- [Target ERD](/opt/sutazaiapp/IMPORTANT/10_canonical/target_state/erd_target.mmd)

### Related Documentation
- [System Overview](01-system-overview.md)
- [Data Flow](03-data-flow.md)
- [API Contracts](/opt/sutazaiapp/IMPORTANT/10_canonical/api_contracts/contracts.md)
- [ADR-0001: UUID Strategy](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0001.md)

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-08-08 | 0.1.0 | Initial draft | Documentation Lead |
| 2025-08-08 | 1.0.0 | Complete component architecture with current/target states | System Architect |