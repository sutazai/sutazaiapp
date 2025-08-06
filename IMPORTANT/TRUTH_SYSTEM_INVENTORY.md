# SutazAI - Truth System Inventory
**Version:** 1.0  
**Date:** August 6, 2025  
**Status:** FACTUAL DOCUMENTATION BASED ON RUNNING CONTAINERS

## ⚠️ CRITICAL: This Document Contains Only Verified Facts

### Model Reality Check
- **Current Model**: TinyLlama (NOT gpt-oss as claimed in docs)
- **Model Location**: Ollama service on port 10104
- **Model Size**: 637 MB
- **Migration Status**: GPT-OSS migration NOT complete despite documentation claims

### Actually Running Services (Verified via `docker ps`)

#### Core Services (All Healthy)
| Service | Port | Status | Uptime | Purpose |
|---------|------|--------|---------|---------|
| PostgreSQL | 10000 | HEALTHY | 18+ hours | Primary database (no tables created) |
| Redis | 10001 | HEALTHY | 19+ hours | Cache layer |
| Neo4j | 10002/10003 | HEALTHY | 16+ hours | Graph database |
| Ollama | 10104 | HEALTHY | 19+ hours | LLM serving (TinyLlama loaded) |

#### Application Layer
| Service | Port | Status | Notes |
|---------|------|--------|-------|
| Backend | 10010 | STARTING | FastAPI service, takes time to initialize |
| Frontend | 10011 | STARTING | Streamlit UI |

#### Vector Databases
| Service | Port | Status | Notes |
|---------|------|--------|-------|
| Qdrant | 10101/10102 | HEALTHY | Vector search |
| FAISS | 10103 | HEALTHY | Vector similarity |
| ChromaDB | 10100 | STARTING | Connection issues |

#### Monitoring Stack (All Running)
| Service | Port | Purpose |
|---------|------|---------|
| Prometheus | 10200 | Metrics collection |
| Grafana | 10201 | Visualization |
| Loki | 10202 | Log aggregation |
| AlertManager | 10203 | Alert routing |
| Node Exporter | 10220 | Node metrics |
| cAdvisor | 10221 | Container metrics |
| Blackbox Exporter | 10229 | Endpoint probing |

#### Service Mesh Infrastructure
| Service | Port | Status | Reality Check |
|---------|------|--------|---------------|
| Kong Gateway | 10005/8001 | RUNNING | API gateway (basic config) |
| Consul | 10006/8600 | RUNNING | Service discovery (minimal usage) |
| RabbitMQ | 10007/10008 | RUNNING | Message queue (not actively used) |

#### Actually Working AI Agents (7 total)
| Agent | Port | Type | Actual Functionality |
|-------|------|------|---------------------|
| AI Agent Orchestrator | 8589 | Flask app | Basic health endpoint, stub logic |
| Multi-Agent Coordinator | 8587 | Flask app | Basic coordination stub |
| Resource Arbitration | 8588 | Flask app | Resource allocation stub |
| Task Assignment | 8551 | Flask app | Task routing stub |
| Hardware Optimizer | 8002 | Flask app | Hardware monitoring stub |
| Ollama Integration | 11015 | Flask app | Ollama interaction wrapper |
| AI Metrics Exporter | 11063 | Exporter | Metrics collection (UNHEALTHY) |

### What Does NOT Exist (Despite Documentation)

#### Fantasy Services Never Deployed
- HashiCorp Vault (secrets management)
- Jaeger (distributed tracing)
- Elasticsearch (search/analytics)
- Kubernetes orchestration
- Terraform infrastructure
- 60+ additional AI agents

#### Claimed But Non-Functional
- Complex agent communication (agents don't talk to each other)
- Quantum computing modules (complete fiction)
- AGI/ASI capabilities (fantasy)
- Advanced orchestration (basic stubs only)
- Inter-agent message passing (not implemented)

### Code vs Documentation Truth Table

| Component | Documentation Claims | Actual Reality |
|-----------|---------------------|----------------|
| Total Agents | 69-150 agents | 7 Flask stubs |
| Model | GPT-OSS everywhere | TinyLlama only |
| Architecture | Microservices mesh | Docker Compose |
| Communication | Complex routing | Direct HTTP calls |
| Orchestration | Kubernetes/Swarm | docker-compose |
| Data Pipeline | Multi-stage ETL | Basic CRUD |
| AI Capabilities | AGI/Quantum/Advanced | Basic LLM queries |

### Working Endpoints

#### Backend API (port 10010)
```bash
curl http://localhost:10010/health
curl http://localhost:10010/docs  # FastAPI interactive docs
```

#### Frontend (port 10011)
```bash
# Streamlit UI
http://localhost:10011
```

#### Ollama (port 10104)
```bash
# Generate text
curl http://localhost:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "Hello"}'

# List models
curl http://localhost:10104/api/tags
```

### File System Reality

#### Working Code Locations
- `/backend/app/` - FastAPI application (mostly stubs)
- `/frontend/` - Streamlit UI
- `/agents/*/app.py` - Flask stub applications

#### Fantasy/Misleading Locations
- Most of `/opt/sutazaiapp/IMPORTANT/` - Mixed truth and fiction
- `/docs/` - Largely theoretical
- Agent directories claiming advanced AI - All stubs

### Database Status
- PostgreSQL: Running but NO TABLES created
- Redis: Running, basic key-value operations work
- Neo4j: Running, graph database available
- Vector DBs: Running but not integrated with application

### Network Reality
- Docker network: `sutazai-network` exists
- Service communication: Direct port mapping
- No service mesh routing actually configured
- No load balancing or circuit breaking

## Summary: What Actually Works

1. **Basic Infrastructure**: Databases, cache, monitoring
2. **Simple Web App**: FastAPI + Streamlit
3. **Local LLM**: TinyLlama via Ollama (NOT gpt-oss)
4. **Monitoring**: Prometheus/Grafana stack
5. **7 Agent Stubs**: Health endpoints only

## What to Use This System For

✅ **Can Do**:
- Local LLM text generation with TinyLlama
- Basic CRUD operations (once tables created)
- Container monitoring via Grafana
- Simple API development with FastAPI

❌ **Cannot Do**:
- Complex AI agent orchestration
- Distributed AI processing
- Advanced NLP pipelines
- Production workloads
- Any "AGI" or quantum features

## Quick Start Commands

```bash
# Check what's actually running
docker ps

# Test backend
curl http://localhost:10010/health

# Test Ollama
curl http://localhost:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "test"}'

# View logs
docker-compose logs -f backend

# Restart a service
docker-compose restart backend
```

## Next Steps for Reality Alignment

1. Remove all fantasy documentation
2. Update CLAUDE.md with truth
3. Either implement claimed features OR remove claims
4. Migrate to gpt-oss if that's the actual goal
5. Create PostgreSQL tables for backend to work
6. Fix unhealthy services (ai-metrics, chromadb)
7. Remove or implement agent logic beyond stubs