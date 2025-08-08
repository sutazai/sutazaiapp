---
title: SutazAI Technology Stack
version: 1.0.0
last_updated: 2025-08-08
author: System Architect
review_status: Production Ready
next_review: 2025-09-08
related_docs: 
  - 01-system-overview.md
  - 02-component-architecture.md
  - /IMPORTANT/10_canonical/standards/ADR-0002.md
  - /IMPORTANT/10_canonical/operations/operations.md
  - adrs/0002-technology-choices.md
---

# SutazAI Technology Stack

## Overview

This document provides a comprehensive analysis of the SutazAI technology stack based on verified system inspection (2025-08-08). It documents the ACTUAL technologies in use, their versions, configuration status, and implementation realities rather than aspirational documentation.

**Key Reality Check**: 28 containers running out of 59 defined in docker-compose.yml, with 7 agent services as Flask stubs returning hardcoded responses.

## Prerequisites

- Docker Engine 20.10+ and Docker Compose v2
- Minimum 8GB RAM, 4 CPU cores recommended
- Linux/macOS host (Windows via WSL2)
- Understanding of microservices and containerized architectures

## Technology Layers

### Frontend Technologies

#### Primary UI Framework
| Technology | Version | Status | Purpose | Rationale |
|-----------|---------|---------|---------|-----------|
| **Streamlit** | 1.40.2 | ✅ Operational | Web-based user interface | Rapid prototyping, Python-native, minimal frontend complexity |

**Configuration**: Port 10011, auto-reload enabled
**Performance**: Single-user optimized, 2-5 second cold start
**Limitations**: No multi-user session management, limited customization
**Migration Path**: Consider FastAPI + React for production scalability

#### Supporting Frontend Libraries
```python
streamlit-ace==0.1.1          # Code editor components
streamlit-aggrid==0.3.4       # Data grid functionality  
streamlit-elements==0.1.0     # Custom HTML/JS components
streamlit-webrtc==0.47.7      # WebRTC capabilities (unused)
streamlit-chat==0.1.1         # Chat UI components
plotly==5.24.1                # Interactive visualizations
pandas==2.2.3                 # Data manipulation for UI
```

**Integration Status**: All libraries installed but minimal usage in current UI

### Backend/API Technologies

#### Core Application Framework
| Technology | Version | Status | Purpose | Performance |
|-----------|---------|---------|---------|-------------|
| **FastAPI** | 0.115.6 | ✅ Operational | REST API server | 1000+ req/sec with async support |
| **Uvicorn** | 0.32.1 | ✅ Operational | ASGI server | Production-ready with hot reload |
| **Pydantic** | 2.10.4 | ✅ Operational | Data validation | Type-safe JSON serialization |

**Configuration**: 
- Port: 10010
- Environment: Development mode with auto-reload
- Health endpoint: `/health` (returns degraded due to model mismatch)
- API documentation: `/docs` (Swagger UI available)

#### Data Validation & Serialization
```python
pydantic==2.10.4              # Core validation engine
pydantic-settings==2.8.1      # Configuration management
starlette==0.41.3             # Low-level web framework
python-multipart==0.0.18     # File upload support
```

#### HTTP Client Libraries
```python
requests==2.32.3             # Synchronous HTTP client
httpx==0.27.2                 # Async HTTP client (preferred)
aiohttp==3.11.10              # Alternative async client
websockets==13.0              # WebSocket support
urllib3==2.2.3                # Low-level HTTP library
```

**Usage Pattern**: httpx preferred for inter-service communication, requests for legacy compatibility

### AI/ML Technology Stack

#### Local LLM Infrastructure
| Technology | Version | Status | Model | Reality Check |
|-----------|---------|---------|--------|---------------|
| **Ollama** | Latest | ✅ Healthy | TinyLlama (637MB) | ⚠️ Backend expects gpt-oss |

**Port**: 10104
**API Endpoints**: 
- `/api/tags` - List models
- `/api/generate` - Text generation
- `/api/chat` - Chat completions

**Performance Characteristics**:
- CPU inference: 50-200 tokens/sec (TinyLlama)
- Memory usage: ~1GB RAM
- Cold start: 2-3 seconds
- Model loading: 5-10 seconds

**Configuration Issue**: Backend configured for "gpt-oss" model but only TinyLlama loaded
**Resolution Required**: Either pull gpt-oss model or update backend configuration

#### Agent Services (Current: Flask Stubs)
| Agent | Port | Framework | Status | Actual Functionality |
|-------|------|-----------|---------|---------------------|
| AI Agent Orchestrator | 8589 | Flask | ✅ Running | Hardcoded JSON responses |
| Multi-Agent Coordinator | 8587 | Flask | ✅ Running | Basic coordination stub |
| Resource Arbitration | 8588 | Flask | ✅ Running | Resource allocation stub |
| Task Assignment | 8551 | Flask | ✅ Running | Task routing stub (most functional) |
| Hardware Optimizer | 8002 | Flask | ✅ Running | Hardware monitoring stub |
| Ollama Integration | 11015 | Flask | ✅ Running | Ollama wrapper (may work) |
| AI Metrics Exporter | 11063 | Flask | ❌ Unhealthy | Metrics collection failed |

**Reality**: All agents have `/health` endpoints but return static responses regardless of input

#### ML/AI Libraries
```python
torch==2.5.1                 # PyTorch framework (2.3GB install)
transformers==<latest>        # HuggingFace transformers
sentence-transformers==3.3.1  # Embedding models
huggingface-hub==0.27.0      # Model hub integration
tiktoken==0.8.0              # OpenAI tokenizer
langchain==0.3.11            # LLM orchestration framework
```

**Usage Status**: Libraries installed but minimal integration with agents
**GPU Support**: CPU-only inference currently

### Data Persistence Layer

#### Primary Database
| Technology | Version | Port | Status | Configuration | Use Case |
|-----------|---------|------|---------|---------------|----------|
| **PostgreSQL** | Latest | 10000 | ✅ Healthy | 14 tables with SERIAL PKs | Primary data store |

**Schema Status**: 14 tables exist but use SERIAL primary keys (needs UUID migration per ADR-0001)
**Connection**: `postgresql://sutazai:sutazai@localhost:10000/sutazai`
**Performance**: Single-node, no replication
**Backup Strategy**: None configured (critical gap)

#### Caching Layer
| Technology | Version | Port | Status | Configuration | Performance |
|-----------|---------|------|---------|---------------|-------------|
| **Redis** | Latest | 10001 | ✅ Healthy | Single instance | 50k+ ops/sec |

**Usage**: Session storage, API caching, task queues
**Persistence**: RDB snapshots enabled
**Clustering**: Single node (not production-ready)

#### Graph Database
| Technology | Version | Ports | Status | Purpose | Integration |
|-----------|---------|-------|---------|---------|-------------|
| **Neo4j** | Latest | 10002/10003 | ✅ Healthy | Relationship mapping | Not yet utilized |

**Access**: Browser UI on 10002, Bolt protocol on 10003
**Authentication**: Default neo4j/neo4j
**Data**: Empty database, schema TBD

#### Vector Databases
| Technology | Version | Ports | Status | Integration | Use Case |
|-----------|---------|-------|---------|-------------|----------|
| **Qdrant** | Latest | 10101/10102 | ✅ Healthy | Not integrated | Vector similarity search |
| **FAISS** | Latest | 10103 | ✅ Healthy | Not integrated | Fast similarity search |
| **ChromaDB** | Latest | 10100 | ⚠️ Connection issues | Not integrated | Document embeddings |

**Reality**: All vector databases running but no application integration
**Performance**: Qdrant: 1k+ vectors/sec, FAISS: 10k+ vectors/sec (theoretical)

### Infrastructure/Orchestration

#### Container Orchestration
| Technology | Version | Status | Configuration | Purpose |
|-----------|---------|---------|---------------|----------|
| **Docker Compose** | v2 | ✅ Operational | 28/59 services running | Local orchestration |

**Network**: `sutazai-network` bridge network
**Volumes**: Persistent for databases, logs
**Profiles**: `mesh` profile for worker services (inactive)
**Environment**: Development-focused configuration

#### Service Mesh (Unconfigured)
| Technology | Version | Port | Status | Purpose | Configuration Status |
|-----------|---------|------|---------|---------|---------------------|
| **Kong Gateway** | 3.5 | 10005/8001 | ✅ Running | API gateway | No routes configured |
| **Consul** | 1.17 | 10006 | ✅ Running | Service discovery | Single node, minimal usage |
| **RabbitMQ** | 3.12 | 10007/10008 | ✅ Running | Message queue | Not actively used |

**Reality**: Infrastructure is running but not integrated with application services
**Configuration Gap**: Kong needs route definitions, Consul service registration missing

### Monitoring/Observability Stack

#### Metrics & Monitoring
| Technology | Version | Port | Status | Purpose | Dashboard Status |
|-----------|---------|------|---------|---------|------------------|
| **Prometheus** | Latest | 10200 | ✅ Operational | Metrics collection | Collecting system metrics |
| **Grafana** | Latest | 10201 | ✅ Operational | Visualization | Login: admin/admin, empty dashboards |
| **Node Exporter** | Latest | 10220 | ✅ Operational | System metrics | Full host metrics |
| **cAdvisor** | Latest | 10221 | ✅ Operational | Container metrics | Docker stats collection |

#### Logging & Alerting  
| Technology | Version | Port | Status | Purpose | Configuration |
|-----------|---------|------|---------|---------|---------------|
| **Loki** | Latest | 10202 | ✅ Operational | Log aggregation | Basic log collection |
| **AlertManager** | Latest | 10203 | ✅ Operational | Alert routing | No alerts configured |
| **Blackbox Exporter** | Latest | 10229 | ✅ Operational | Endpoint monitoring | HTTP health checks |

**Dashboard Gap**: Grafana has no pre-configured dashboards (ISSUE-0016)
**Alerting Gap**: No alert rules or notification channels configured

### Development Tools

#### Package Management
```python
# Backend Dependencies (requirements.txt)
- Core: FastAPI, Pydantic, SQLAlchemy
- Database: psycopg2, asyncpg, redis, neo4j
- AI/ML: torch, transformers, langchain
- Security: cryptography, PyJWT, passlib

# Frontend Dependencies (requirements.txt)
- Minimal: streamlit, requests, pandas
- Visualization: plotly
- HTTP: httpx, aiohttp
```

#### Security Libraries
```python
cryptography==43.0.1         # Encryption primitives
PyJWT==2.9.0                 # JSON Web Tokens
passlib[bcrypt]==1.7.4       # Password hashing
bcrypt==4.2.0                # Bcrypt implementation
python-jose[cryptography]==3.3.0  # JOSE/JWS/JWT
email-validator==2.2.0       # Email validation
bleach==6.1.0                # HTML sanitization
```

**Authentication Status**: Libraries installed but no authentication implementation

#### Data Processing
```python
pandas==2.2.3                # Data manipulation
numpy==2.1.3                 # Numerical computing
scipy==1.14.1                # Scientific computing
scikit-learn==1.6.0          # Machine learning
matplotlib==3.10.0           # Basic plotting
plotly==5.24.1               # Interactive visualizations
joblib==1.4.2                # Parallel processing
```

## Technology Decision Rationale

### Core Architecture Decisions

#### ADR-0002: Local LLM Strategy
**Decision**: 100% local execution via Ollama, no external APIs
**Rationale**: Data privacy, cost control, offline capability
**Trade-offs**: Limited model selection, higher resource requirements
**Status**: Accepted, but implementation incomplete (model mismatch)

#### Container-First Approach
**Decision**: Docker Compose for all services
**Rationale**: Consistent deployment, easy local development
**Trade-offs**: Single-host limitation, not production-scalable
**Migration Path**: Consider Kubernetes for multi-node deployment

#### Python-Centric Stack
**Decision**: Python for backend, agents, and frontend
**Rationale**: ML ecosystem compatibility, team expertise
**Trade-offs**: Performance limitations, GIL constraints
**Mitigation**: Async I/O, process-based parallelism

### Database Architecture Decisions

#### Multi-Database Strategy
**PostgreSQL**: Primary relational data
**Redis**: Caching and session management  
**Neo4j**: Graph relationships (planned)
**Vector DBs**: Similarity search (planned)

**Rationale**: Polyglot persistence for optimal data models
**Trade-offs**: Operational complexity, data consistency challenges
**Status**: Infrastructure ready, integration incomplete

## Performance Characteristics

### Throughput Metrics
| Component | Metric | Current | Target | Bottleneck |
|-----------|---------|---------|---------|------------|
| Backend API | Requests/sec | ~100 | 1000+ | Single process |
| LLM Inference | Tokens/sec | 50-200 | 500+ | CPU-bound |
| Database | Queries/sec | ~1000 | 5000+ | Single connection pool |
| Vector Search | Vectors/sec | 0 (not integrated) | 1000+ | Not implemented |

### Resource Usage
| Service | CPU | Memory | Storage | Network |
|---------|-----|---------|---------|---------|
| Backend | 0.1-0.5 cores | 500MB | Minimal | 1-10 Mbps |
| Ollama | 1-4 cores | 1-2GB | 1GB (model) | Minimal |
| PostgreSQL | 0.1-0.2 cores | 200MB | 1GB | Minimal |
| Monitoring | 0.2-0.5 cores | 1GB | 5GB | Minimal |
| **Total** | **2-6 cores** | **3-4GB** | **8GB** | **Low** |

## Known Limitations

### Critical Gaps
1. **Authentication**: No user authentication or authorization
2. **Agent Logic**: All agents return hardcoded responses
3. **Model Mismatch**: Backend expects gpt-oss, TinyLlama loaded
4. **Database Schema**: Uses SERIAL PKs instead of UUIDs
5. **Service Integration**: Service mesh components unconfigured

### Performance Limitations
1. **Single-Node**: No horizontal scaling capability
2. **CPU Inference**: No GPU acceleration configured
3. **Session Management**: No multi-user support
4. **Connection Pooling**: Basic database connection management
5. **Monitoring**: No dashboards or alerts configured

### Security Concerns
1. **No TLS**: HTTP-only communication
2. **Default Passwords**: Unchanged default credentials
3. **No Secrets Management**: Plain-text configuration
4. **Container Security**: Running as root users
5. **Network Isolation**: Minimal network segmentation

## Upgrade/Migration Considerations

### Immediate Priorities (Next 30 days)
1. **Fix Model Configuration**: Align backend with TinyLlama or load gpt-oss
2. **Database Migration**: Implement UUID primary keys (ADR-0001)
3. **Basic Authentication**: Implement user login system
4. **Agent Implementation**: Replace one stub with real logic
5. **Monitoring Setup**: Configure Grafana dashboards

### Medium-term Upgrades (3-6 months)
1. **Production Database**: Multi-node PostgreSQL with replication
2. **Service Mesh**: Configure Kong routes and Consul registration  
3. **Vector Integration**: Connect vector databases to application
4. **GPU Support**: Add NVIDIA container runtime for accelerated inference
5. **Kubernetes Migration**: Multi-node orchestration

### Long-term Architecture Evolution (6-12 months)
1. **Microservices Decomposition**: Break monolithic backend into services
2. **Event-Driven Architecture**: Implement CQRS with RabbitMQ
3. **Multi-Model Support**: Support multiple LLMs simultaneously
4. **Advanced Monitoring**: Distributed tracing, APM integration
5. **Security Hardening**: Zero-trust networking, secrets management

## Integration Points and Dependencies

### Service Dependencies
```mermaid
graph TD
    A[Frontend] --> B[Backend API]
    B --> C[PostgreSQL]
    B --> D[Redis]
    B --> E[Ollama]
    B --> F[Agent Services]
    F --> E
    G[Monitoring] --> B
    G --> C
    G --> D
    H[Kong Gateway] -.-> B
    I[RabbitMQ] -.-> F
```

### Critical Paths
1. **User Request Flow**: Frontend → Backend → Database/Ollama → Response
2. **Agent Processing**: Backend → Agent Service → Ollama → Result
3. **Monitoring Data**: Services → Prometheus → Grafana → Alerts

### Single Points of Failure
1. **Backend API**: Single instance, no redundancy
2. **PostgreSQL**: Single node, no backup
3. **Ollama**: Single model server
4. **Redis**: No persistence configured for critical data

## Technology Roadmap

### Current State (v1.0)
- ✅ Basic containerized services
- ✅ Local LLM capability
- ✅ Monitoring infrastructure
- ⚠️ Stub agent implementations
- ❌ No authentication or security

### Target State (v2.0 - 6 months)
- Production-ready authentication
- Real agent logic implementation  
- Vector database integration
- Configured service mesh
- GPU-accelerated inference
- Comprehensive monitoring dashboards

### Future Vision (v3.0 - 12 months)
- Multi-tenant architecture
- Advanced agent orchestration
- Real-time collaboration features
- Auto-scaling infrastructure
- Enterprise security compliance
- ML pipeline automation

## References

### Architecture Decision Records
- [ADR-0001: UUID Primary Keys](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0001.md)
- [ADR-0002: Local LLM Strategy](/opt/sutazaiapp/IMPORTANT/10_canonical/standards/ADR-0002.md)
- [ADR-0003: Container Strategy](adrs/0001-architecture-principles.md)
- [ADR-0004: Security Architecture](adrs/0002-technology-choices.md)

### System Documentation
- [System Overview](01-system-overview.md)
- [Component Architecture](02-component-architecture.md)
- [System Reality Check](/opt/sutazaiapp/CLAUDE.md)
- [Canonical Infrastructure](/opt/sutazaiapp/IMPORTANT/10_canonical/)

### External References
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Ollama Model Library](https://ollama.ai/library)
- [Docker Compose Reference](https://docs.docker.com/compose/)
- [Prometheus Configuration](https://prometheus.io/docs/prometheus/latest/configuration/)

## Change Log

| Date | Version | Changes | Author |
|------|---------|---------|--------|
| 2025-08-08 | 0.1.0 | Initial draft created | Documentation Lead |
| 2025-08-08 | 1.0.0 | Complete rewrite with comprehensive technology stack analysis based on system verification | System Architect |

---

**Document Status**: Production Ready  
**Verification Date**: 2025-08-08  
**Next Review**: 2025-09-08  
**Reality Check**: Based on direct system inspection, not aspirational documentation

