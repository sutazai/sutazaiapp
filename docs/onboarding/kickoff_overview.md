# Perfect Jarvis Project - Team Onboarding & Architecture Overview

**Document Version:** 1.0.0  
**Last Updated:** 2025-08-07  
**Document Status:** AUTHORITATIVE - Single Source of Truth for Team Onboarding

---

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Technology Stack](#technology-stack)
3. [Repository Analysis & Integration Points](#repository-analysis--integration-points)
4. [System Architecture](#system-architecture)
5. [Ownership Matrix](#ownership-matrix)
6. [Current System Limitations](#current-system-limitations)
7. [Project Structure & Conventions](#project-structure--conventions)
8. [Onboarding Action Plan](#onboarding-action-plan)

---

## Executive Summary

### Project Vision
Perfect Jarvis is an AI-powered assistant system that synthesizes capabilities from five key repositories to create a comprehensive, locally-hosted AI automation platform. The system integrates voice interaction, multi-model AI processing, task orchestration, and extensive monitoring capabilities.

### Current Reality
- **28 containers** actively running in production
- **7 agent services** deployed (currently stubs requiring implementation)
- **TinyLlama model** (637MB) operational via Ollama
- **Complete monitoring stack** with Prometheus, Grafana, and Loki
- **3 vector databases** ready for integration (ChromaDB, Qdrant, FAISS)

### Immediate Priorities
1. Fix model configuration mismatch (TinyLlama vs gpt-oss)
2. Implement real logic in agent stubs
3. Connect vector databases to backend
4. Establish inter-agent communication via RabbitMQ

---

## Technology Stack

### Core Infrastructure (✅ Running & Verified)

#### Databases
| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| PostgreSQL 16.3 | 10000 | Primary relational database | ✅ HEALTHY (14 tables) |
| Redis 7.2 | 10001 | Cache & session store | ✅ HEALTHY |
| Neo4j 5 | 10002/10003 | Graph database | ✅ HEALTHY |

#### Vector Databases
| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| ChromaDB 0.5.0 | 10100 | Document embeddings | ⚠️ STARTING |
| Qdrant 1.9.2 | 10101/10102 | Vector similarity search | ✅ HEALTHY |
| FAISS | 10103 | Facebook AI Similarity Search | ✅ HEALTHY |

#### Service Mesh
| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| Kong Gateway 3.5 | 10005 | API gateway & load balancing | ✅ HEALTHY |
| Consul | 10006 | Service discovery | ✅ HEALTHY |
| RabbitMQ 3.12 | 10007/10008 | Message queuing | ✅ HEALTHY |

#### AI/ML Infrastructure
| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| Ollama | 10104 | Local LLM inference | ✅ HEALTHY (TinyLlama loaded) |

#### Application Layer
| Service | Port | Purpose | Status |
|---------|------|---------|--------|
| FastAPI Backend | 10010 | Core API (v17.0.0) | ✅ HEALTHY |
| Streamlit Frontend | 10011 | Web UI | ✅ WORKING |

#### Monitoring Stack
| Service | Port | Purpose |
|---------|------|---------|
| Prometheus | 10200 | Metrics collection |
| Grafana | 10201 | Visualization dashboards |
| Loki | 10202 | Log aggregation |
| AlertManager | 10203 | Alert routing |
| Node Exporter | 10220 | System metrics |
| cAdvisor | 10221 | Container metrics |
| Blackbox Exporter | 10229 | Endpoint monitoring |
| Promtail | - | Log shipping to Loki |

### Agent Services (🔧 Require Implementation)

| Agent | Port | Current State | Required Implementation |
|-------|------|---------------|------------------------|
| AI Agent Orchestrator | 8589 | Stub returning `{"status": "healthy"}` | Task planning & coordination logic |
| Multi-Agent Coordinator | 8587 | Basic coordination stub | Inter-agent communication |
| Resource Arbitration | 8588 | Resource allocation stub | Real resource management |
| Task Assignment | 8551 | Task routing stub | Intelligent task distribution |
| Hardware Optimizer | 8002 | Hardware monitoring stub | Actual system metrics collection |
| Ollama Integration | 11015 | Ollama wrapper stub | Full LLM integration |
| AI Metrics Exporter | 11063 | ❌ UNHEALTHY | Fix metrics collection |

---

## Repository Analysis & Integration Points

### 1. Dipeshpal/Jarvis_AI - Foundation Framework
**Repository:** https://github.com/Dipeshpal/Jarvis_AI

**Key Capabilities:**
- Simple Python library with voice/text I/O
- Extensible plugin architecture
- Basic task automation (time, email, screenshots)

**Integration Points:**
- Voice input/output handling → `/backend/jarvis/voice_handler.py`
- Plugin system → `/backend/plugins/`
- Basic automation tasks → `/agents/task_assignment_coordinator/`

**Required Adaptations:**
- Replace English-only limitation with multi-language support
- Integrate with Ollama instead of cloud APIs
- Connect to PostgreSQL for conversation persistence

### 2. Microsoft/JARVIS - Advanced Multimodal AI
**Repository:** https://github.com/microsoft/JARVIS

**Key Capabilities:**
- 4-stage workflow: Planning → Selection → Execution → Response
- LLM controller coordinating expert models
- HuggingFace model ecosystem integration

**Integration Points:**
- Task planning controller → `/backend/jarvis/task_controller.py`
- Model selection logic → `/agents/ai_agent_orchestrator/`
- Response generation → `/backend/app/api/v1/responses.py`

**Required Adaptations:**
- Replace ChatGPT with local Ollama models
- Implement model selection for TinyLlama
- Create task decomposition pipeline

### 3. llm-guy/jarvis - Local LLM Voice Assistant
**Repository:** https://github.com/llm-guy/jarvis

**Key Capabilities:**
- Wake word detection ("Jarvis")
- Fully local processing
- LangChain tool integration

**Integration Points:**
- Wake word detection → `/backend/jarvis/wake_word.py`
- Voice processing pipeline → `/backend/services/voice_service.py`
- Tool calling framework → `/backend/tools/`

**Required Adaptations:**
- Integrate with existing Ollama instance
- Connect to RabbitMQ for async processing
- Link with vector databases for knowledge retrieval

### 4. danilofalcao/jarvis - Multi-Model Coding Assistant
**Repository:** https://github.com/danilofalcao/jarvis

**Key Capabilities:**
- 11 AI model support
- File attachment processing (PDF/Word/Excel)
- Real-time collaboration via WebSocket

**Integration Points:**
- Multi-model manager → `/backend/jarvis/model_manager.py`
- Document processing → `/agents/document_processor/`
- WebSocket integration → `/backend/app/websocket/`

**Required Adaptations:**
- Map models to Ollama-compatible alternatives
- Integrate with ChromaDB for document embeddings
- Connect to existing FastAPI WebSocket support

### 5. SreejanPersonal/JARVIS
**Status:** Repository not accessible (404)
**Action:** Skip integration, focus on accessible repositories

---

## System Architecture

### Current Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User Interface Layer                      │
├─────────────────────────────────────────────────────────────────┤
│  Streamlit Frontend (10011) ←→ FastAPI Backend (10010)          │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      API Gateway & Service Mesh                   │
├─────────────────────────────────────────────────────────────────┤
│  Kong Gateway (10005) ←→ Consul (10006) ←→ RabbitMQ (10007)    │
└─────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
┌─────────────────────────────┐   ┌─────────────────────────────┐
│      Agent Services          │   │    AI/ML Infrastructure     │
├─────────────────────────────┤   ├─────────────────────────────┤
│ • AI Orchestrator (8589)    │   │ • Ollama (10104)           │
│ • Multi-Agent Coord (8587)  │   │   - TinyLlama Model        │
│ • Resource Arb (8588)       │   │ • ChromaDB (10100)         │
│ • Task Assignment (8551)    │   │ • Qdrant (10101)           │
│ • Hardware Opt (8002)       │   │ • FAISS (10103)            │
│ • Ollama Integration (11015)│   │                            │
└─────────────────────────────┘   └─────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Data Layer                               │
├─────────────────────────────────────────────────────────────────┤
│  PostgreSQL (10000) | Redis (10001) | Neo4j (10002/10003)      │
└─────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Monitoring & Observability                   │
├─────────────────────────────────────────────────────────────────┤
│  Prometheus (10200) → Grafana (10201) | Loki (10202) ← Promtail│
│  AlertManager (10203) | Node Exporter (10220) | cAdvisor (10221)│
└─────────────────────────────────────────────────────────────────┘
```

### Service Interactions & Data Flow

#### 1. Request Flow
```
User Request → Frontend (10011) → Backend API (10010) → Kong Gateway (10005)
    → Agent Selection (via Consul) → Task Execution → Response Generation
```

#### 2. Agent Communication Pattern
```
Agent A → RabbitMQ (10007) → Agent B
    ↓                           ↓
PostgreSQL ← Redis Cache → Response Aggregation
```

#### 3. Vector Search Flow
```
Query → Embedding Generation (Ollama) → Vector Store Query
    → ChromaDB/Qdrant/FAISS → Similarity Results → Context Enhancement
```

### API Contracts

#### Core API Endpoints (Backend - Port 10010)
```yaml
/health:
  method: GET
  response: {status: string, services: object}

/api/v1/chat:
  method: POST
  request: {message: string, context?: object}
  response: {response: string, metadata: object}

/api/v1/agents:
  method: GET
  response: {agents: array<Agent>}

/api/v1/tasks:
  method: POST
  request: {type: string, payload: object}
  response: {task_id: string, status: string}
```

#### Agent API Standards
```yaml
/health:
  method: GET
  response: {status: "healthy|degraded|unhealthy"}

/process:
  method: POST
  request: {task: object, context?: object}
  response: {status: string, result: object}

/capabilities:
  method: GET
  response: {capabilities: array<string>}
```

---

## Ownership Matrix

### Module Ownership

| Module/Component | Owner Role | Current State | Priority |
|-----------------|------------|---------------|----------|
| **Core Infrastructure** |
| PostgreSQL Database | Database Administrator | ✅ Running (14 tables) | Maintain |
| Redis Cache | Backend Developer | ✅ Running | Maintain |
| Neo4j Graph DB | Data Engineer | ✅ Running | Enhance |
| **AI/ML Components** |
| Ollama Integration | ML Engineer | ⚠️ Model mismatch | P0 - Critical |
| Vector Databases | ML Engineer | ✅ Running, not integrated | P1 - High |
| Agent Services | Backend Developer | 🔧 Stubs only | P0 - Critical |
| **Application Layer** |
| FastAPI Backend | Senior Backend Dev | ✅ v17.0.0 running | Enhance |
| Streamlit Frontend | Frontend Developer | ✅ Running | Enhance |
| **Service Mesh** |
| Kong Gateway | DevOps Engineer | ✅ Running, no routes | P1 - High |
| Consul | DevOps Engineer | ✅ Running | Configure |
| RabbitMQ | Backend Developer | ✅ Running, not used | P1 - High |
| **Monitoring** |
| Prometheus/Grafana | SRE/DevOps | ✅ Fully operational | Maintain |
| Loki/Promtail | SRE/DevOps | ✅ Running | Maintain |

### Microservice Boundaries

| Service | Responsibility | Dependencies | Communication |
|---------|---------------|--------------|---------------|
| **Backend API** | Core business logic, orchestration | All databases, Ollama | REST API, WebSocket |
| **Agent Orchestrator** | Task planning, agent coordination | Backend API, all agents | RabbitMQ, REST |
| **Hardware Optimizer** | System metrics, resource management | System APIs | REST, Prometheus metrics |
| **Task Assignment** | Task routing, load balancing | Agent registry, RabbitMQ | RabbitMQ, REST |
| **Ollama Integration** | LLM operations, model management | Ollama server | REST, streaming |
| **Vector Services** | Embedding storage/retrieval | ChromaDB, Qdrant, FAISS | REST API |

---

## Current System Limitations

### Technical Constraints
1. **Model Mismatch**: System configured for `gpt-oss` but only `tinyllama` is loaded
2. **CPU-Only**: No GPU acceleration available
3. **Agent Stubs**: All 7 agents return hardcoded responses, no real processing
4. **Database State**: PostgreSQL has schema but unknown data population
5. **Vector DB Integration**: Running but not connected to application
6. **Service Mesh**: Kong/Consul running but no routes configured

### Missing Capabilities (Not Implemented)
- Kubernetes/K3s orchestration
- Terraform infrastructure as code
- HashiCorp Vault secrets management
- Jaeger distributed tracing
- Elasticsearch/ELK stack
- Auto-scaling policies
- Circuit breakers
- Multi-region deployment
- Automated backups

### Performance Limitations
- TinyLlama model: 637MB, limited capabilities vs larger models
- Single-node deployment: No horizontal scaling
- No caching layer for LLM responses
- Synchronous processing: No background job queue active

---

## Project Structure & Conventions

### Directory Structure
```
/opt/sutazaiapp/
├── backend/                 # FastAPI application
│   ├── app/
│   │   ├── api/            # API endpoints
│   │   ├── core/           # Core utilities
│   │   ├── db/             # Database models
│   │   └── services/       # Business logic
│   └── jarvis/             # Jarvis-specific modules (TO CREATE)
├── frontend/               # Streamlit UI
├── agents/                 # Agent services
│   ├── core/              # Shared agent base classes
│   └── [agent_name]/      # Individual agent implementations
├── docker/                # Docker configurations
├── config/                # Configuration files
├── scripts/               # Utility scripts
├── tests/                # Test suite
├── docs/                  # Documentation
│   ├── architecture/      # System design docs
│   ├── onboarding/       # Team onboarding materials
│   └── CHANGELOG.md      # Change tracking
└── IMPORTANT/            # Critical documentation
```

### Naming Conventions
- **Python files**: `snake_case.py`
- **Classes**: `PascalCase`
- **Functions/Variables**: `snake_case`
- **Constants**: `UPPER_SNAKE_CASE`
- **Docker services**: `kebab-case`
- **API endpoints**: `/api/v1/resource_name`

### Module Boundaries
```
components/   # Reusable UI components (Frontend)
services/     # Business logic and external integrations
utils/        # Pure utility functions
hooks/        # React/Streamlit hooks (Frontend)
schemas/      # Data validation schemas (Pydantic)
```

### Development Standards
- **Linting**: Flake8 for Python
- **Formatting**: Black for Python code
- **Type Checking**: mypy for static analysis
- **Testing**: pytest for unit/integration tests
- **Documentation**: Docstrings for all public functions
- **Commit Convention**: `type(scope): description`

---

## Onboarding Action Plan

### Meeting Agenda - Team Kickoff

#### Session 1: System Overview (2 hours)
1. **Introduction** (15 min)
   - Project vision and goals
   - Current system state

2. **Architecture Walkthrough** (45 min)
   - Live system demonstration
   - Service interaction review
   - Database schema exploration

3. **Technology Stack Deep Dive** (30 min)
   - Core technologies
   - Agent framework
   - Monitoring capabilities

4. **Q&A** (30 min)

#### Session 2: Hands-On Setup (2 hours)
1. **Environment Setup** (30 min)
   - Docker installation verification
   - Repository access
   - Local development setup

2. **System Deployment** (45 min)
   ```bash
   # Clone repository
   git clone [repository_url]
   cd sutazaiapp
   
   # Start services
   docker network create sutazai-network
   docker-compose up -d
   
   # Verify health
   docker ps --format "table {{.Names}}\t{{.Status}}"
   ```

3. **Testing Endpoints** (30 min)
   - Backend API testing
   - Agent health checks
   - Monitoring dashboard access

4. **First Task Assignment** (15 min)

### Role-Specific Onboarding Paths

#### Backend Developer
**Week 1 Goals:**
1. Implement one real agent (Hardware Optimizer)
2. Connect one vector database to backend
3. Create basic RabbitMQ integration

**Resources:**
- `/backend/app/` - Core application code
- `/agents/` - Agent implementations
- API documentation at `http://localhost:10010/docs`

#### Frontend Developer
**Week 1 Goals:**
1. Familiarize with Streamlit framework
2. Create agent status dashboard
3. Implement chat interface improvements

**Resources:**
- `/frontend/` - Streamlit application
- UI at `http://localhost:10011`
- Streamlit docs: https://docs.streamlit.io

#### ML Engineer
**Week 1 Goals:**
1. Fix model configuration (TinyLlama vs gpt-oss)
2. Implement embedding generation service
3. Connect vector databases

**Resources:**
- Ollama API: `http://localhost:10104`
- Vector DB APIs (ChromaDB, Qdrant, FAISS)
- `/backend/jarvis/` - ML integration code

#### DevOps Engineer
**Week 1 Goals:**
1. Configure Kong API routes
2. Set up Consul service registration
3. Create backup automation

**Resources:**
- Kong Admin: `http://localhost:8001`
- Consul UI: `http://localhost:10006`
- Monitoring: `http://localhost:10201` (Grafana)

### Success Metrics - Week 1
- [ ] All team members have local environment running
- [ ] Model configuration issue resolved
- [ ] At least 1 agent has real implementation
- [ ] 1 vector database integrated with backend
- [ ] Basic inter-agent communication working
- [ ] Documentation updated with changes

### Expected Outcomes - Month 1
1. **Functional Agents**: All 7 agents with real logic
2. **Voice Integration**: Wake word detection working
3. **Knowledge Base**: Vector databases fully integrated
4. **Task Orchestration**: Multi-agent coordination operational
5. **Production Ready**: Core features stable and tested

---

## Appendix A: Quick Reference Commands

### System Management
```bash
# Check system status
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"

# View logs
docker-compose logs -f [service_name]

# Restart service
docker-compose restart [service_name]

# Database access
docker exec -it sutazai-postgres psql -U sutazai -d sutazai

# Redis CLI
docker exec -it sutazai-redis redis-cli
```

### Testing Endpoints
```bash
# Backend health
curl http://localhost:10010/health | jq

# Agent health
curl http://localhost:8589/health | jq

# Ollama models
curl http://localhost:10104/api/tags | jq

# Generate text
curl -X POST http://localhost:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "Hello"}' | jq
```

### Monitoring Access
- Grafana: http://localhost:10201 (admin/admin)
- Prometheus: http://localhost:10200
- Consul: http://localhost:10006

---

## Appendix B: External Documentation Links

### Core Technologies
- FastAPI: https://fastapi.tiangolo.com/
- Streamlit: https://docs.streamlit.io/
- Ollama: https://github.com/ollama/ollama
- Docker Compose: https://docs.docker.com/compose/

### Vector Databases
- ChromaDB: https://docs.trychroma.com/
- Qdrant: https://qdrant.tech/documentation/
- FAISS: https://github.com/facebookresearch/faiss/wiki

### Service Mesh
- Kong Gateway: https://docs.konghq.com/
- Consul: https://www.consul.io/docs
- RabbitMQ: https://www.rabbitmq.com/documentation.html

### Monitoring
- Prometheus: https://prometheus.io/docs/
- Grafana: https://grafana.com/docs/
- Loki: https://grafana.com/docs/loki/

---

*Document maintained by: System Architecture Team*  
*Next Review Date: 2025-08-14*  
*For questions, refer to CLAUDE.md for system truth*