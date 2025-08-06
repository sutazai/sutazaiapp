# CLAUDE.md - SutazAI System Truth Document

This file provides the SINGLE SOURCE OF TRUTH for Claude Code (claude.ai/code) when working with this repository.

**Last Modified:** December 19, 2024  
**Document Sections:** System Reality Check + Comprehensive Codebase Rules

## ⚠️ CRITICAL REALITY CHECK ⚠️

**Last System Verified:** August 6, 2025  
**Verification Method:** Direct container inspection and endpoint testing  
**Major Cleanup Completed:** v56 cleanup operation removed 200+ fantasy documentation files  
**Rules Added:** December 19, 2024 - 19 comprehensive codebase standards

### The Truth About This System
- **59 services defined** in docker-compose.yml
- **28 containers actually running** 
- **7 agent services running** (all Flask stubs with health endpoints only)
- **Model Reality**: TinyLlama 637MB loaded (NOT gpt-oss as docs claim)
- **No AI Logic**: Agents return hardcoded JSON responses, no actual AI processing

## 🔴 What ACTUALLY Works (Verified by Testing)

### Core Infrastructure (All Verified Healthy)
| Service | Port | Status | Reality Check |
|---------|------|--------|---------------|
| PostgreSQL | 10000 | ✅ HEALTHY | Database has 14 tables (users, agents, tasks, etc.) |
| Redis | 10001 | ✅ HEALTHY | Cache layer functional |
| Neo4j | 10002/10003 | ✅ HEALTHY | Graph database available |
| Ollama | 10104 | ✅ HEALTHY | TinyLlama model loaded and working |

### Application Layer
| Service | Port | Status | Notes |
|---------|------|--------|-------|
| Backend API | 10010 | ✅ HEALTHY | FastAPI v17.0.0 - Ollama connected, all services operational |
| Frontend | 10011 | ⚠️ STARTING | Streamlit UI - takes time to initialize |

### Service Mesh (Actually Running)
| Service | Port | Status | Usage |
|---------|------|--------|-------|
| Kong Gateway | 10005/8001 | ✅ RUNNING | API gateway (no routes configured) |
| Consul | 10006 | ✅ RUNNING | Service discovery (minimal usage) |
| RabbitMQ | 10007/10008 | ✅ RUNNING | Message queue (not actively used) |

### Vector Databases
| Service | Port | Status | Integration |
|---------|------|--------|-------------|
| Qdrant | 10101/10102 | ✅ HEALTHY | Not integrated with app |
| FAISS | 10103 | ✅ HEALTHY | Not integrated with app |
| ChromaDB | 10100 | ⚠️ STARTING | Connection issues |

### Monitoring Stack (All Operational)
| Service | Port | Purpose |
|---------|------|---------|
| Prometheus | 10200 | Metrics collection |
| Grafana | 10201 | Visualization dashboards |
| Loki | 10202 | Log aggregation |
| AlertManager | 10203 | Alert routing |
| Node Exporter | 10220 | System metrics |
| cAdvisor | 10221 | Container metrics |

## 🟡 What Are STUBS (Return Fake Responses)

### Running Agent Services (Flask Apps with Health Endpoints Only)
| Agent | Port | Actual Functionality |
|-------|------|---------------------|
| AI Agent Orchestrator | 8589 | Returns `{"status": "healthy"}` |
| Multi-Agent Coordinator | 8587 | Basic coordination stub |
| Resource Arbitration | 8588 | Resource allocation stub |
| Task Assignment | 8551 | Task routing stub |
| Hardware Optimizer | 8002 | Hardware monitoring stub |
| Ollama Integration | 11015 | Ollama wrapper (may work) |
| AI Metrics Exporter | 11063 | Metrics collection (UNHEALTHY) |

**Reality**: These agents have `/health` endpoints but `/process` endpoints return hardcoded JSON regardless of input.

## ❌ What is PURE FANTASY (Doesn't Exist)

### Never Deployed Services
- HashiCorp Vault (secrets management)
- Jaeger (distributed tracing)  
- Elasticsearch (search/analytics)
- Kubernetes orchestration
- Terraform infrastructure
- 60+ additional AI agents mentioned in docs

### Fictional Capabilities
- Quantum computing modules (deleted `/backend/quantum_architecture/`)
- AGI/ASI orchestration (complete fiction)
- Complex agent communication (agents don't talk to each other)
- Inter-agent message passing (not implemented)
- Advanced ML pipelines (not present)
- Self-improvement capabilities (stub only)

## 📍 Accurate Port Registry

```yaml
# Core Services (Running)
10000: PostgreSQL database
10001: Redis cache
10002: Neo4j browser interface
10003: Neo4j bolt protocol
10005: Kong API Gateway
10006: Consul service discovery
10007: RabbitMQ AMQP
10008: RabbitMQ management UI
10010: Backend FastAPI
10011: Frontend Streamlit
10104: Ollama LLM server

# Vector Databases (Running)
10100: ChromaDB (connection issues)
10101: Qdrant HTTP
10102: Qdrant gRPC
10103: FAISS vector service

# Monitoring (Running)
10200: Prometheus metrics
10201: Grafana dashboards
10202: Loki logs
10203: AlertManager
10220: Node Exporter
10221: cAdvisor
10229: Blackbox Exporter

# Agent Services (Stubs)
8002: Hardware Resource Optimizer
8551: Task Assignment Coordinator
8587: Multi-Agent Coordinator
8588: Resource Arbitration Agent
8589: AI Agent Orchestrator
11015: Ollama Integration Specialist
11063: AI Metrics Exporter
```

## 🛠️ Working Commands (Tested & Verified)

### System Management
```bash
# Check what's ACTUALLY running
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"

# Start system (only method that works)
docker network create sutazai-network 2>/dev/null
docker-compose up -d

# View logs for debugging
docker-compose logs -f backend
docker-compose logs -f ollama

# Restart a service
docker-compose restart backend

# Stop everything
docker-compose down
```

### Testing Endpoints
```bash
# Backend API (returns degraded status)
curl http://127.0.0.1:10010/health

# Frontend UI
open http://localhost:10011

# Ollama - check loaded model (shows tinyllama, not gpt-oss)
curl http://127.0.0.1:10104/api/tags

# Test text generation with ACTUAL model
curl http://127.0.0.1:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "Hello world"}'

# Agent health check (stub response)
curl http://127.0.0.1:8589/health

# Monitoring
open http://localhost:10200  # Prometheus
open http://localhost:10201  # Grafana (admin/admin)
```

### Database Access
```bash
# PostgreSQL (no tables exist yet!)
docker exec -it sutazai-postgres psql -U sutazai -d sutazai
# \dt to list tables (will be empty)

# Redis
docker exec -it sutazai-redis redis-cli

# Neo4j browser
open http://localhost:10002
```

## ⚠️ Common Documentation LIES to Ignore

### Lie #1: "69 Agents Deployed"
**Truth**: 7 Flask stubs with health endpoints

### Lie #2: "GPT-OSS Migration Complete"  
**Truth**: TinyLlama is loaded, no gpt-oss model present

### Lie #3: "Complex Service Mesh with Advanced Routing"
**Truth**: Kong/Consul/RabbitMQ running but not configured or integrated

### Lie #4: "Quantum Computing Capabilities"
**Truth**: Complete fiction, code was deleted

### Lie #5: "Self-Improving AI System"
**Truth**: Hardcoded stub responses

### Lie #6: "Production Ready"
**Truth**: No database tables, agents are stubs, basic PoC at best

## 📂 Code Structure Reality

### Working Code Locations
```
/backend/app/          # FastAPI application (partially implemented)
  ├── main.py         # Entry point with feature flags
  ├── api/            # API endpoints (many stubs)
  └── core/           # Core utilities

/frontend/            # Streamlit UI (basic implementation)

/agents/*/app.py     # Flask stub applications
  └── All return:    {"status": "healthy", "result": "processed"}
```

### Post-Cleanup Clean Locations (v56)
```
/backend/app/            # FastAPI application (partially implemented)
  ├── main.py           # Entry point with feature flags
  ├── api/              # API endpoints (many stubs)
  └── core/             # Core utilities

/frontend/              # Streamlit UI (basic implementation)
/agents/core/           # Consolidated agent base classes
/config/                # Clean configuration files
/docker/                # Service container definitions
/scripts/               # Utility and deployment scripts
/tests/                 # Test suite (updated to test real functionality)
```

### Cleaned Up (Removed in v56)
```
❌ /IMPORTANT/*.md       # Fantasy docs removed
❌ /archive/             # Backup directories cleaned
❌ Root-level *_test.py  # Analysis scripts deleted
❌ Root-level *_audit.py # Compliance files removed
❌ Fantasy agent dirs    # Non-functional agent services
❌ Duplicate base agents # Multiple BaseAgent implementations
```

## 🚀 Quick Start (What Actually Works)

### 1. Start the System
```bash
# Ensure network exists
docker network create sutazai-network 2>/dev/null

# Start all services
docker-compose up -d

# Wait for initialization
sleep 30

# Check status
docker-compose ps
```

### 2. Verify Core Services
```bash
# Backend should return degraded (Ollama disconnected)
curl http://127.0.0.1:10010/health | jq

# Ollama should show tinyllama model
curl http://127.0.0.1:10104/api/tags | jq
```

### 3. Test Basic Functionality
```bash
# Generate text with TinyLlama
curl -X POST http://127.0.0.1:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "What is Docker?"}' | jq

# Access frontend
open http://localhost:10011

# View monitoring
open http://localhost:10201  # Grafana dashboards
```

## 🔧 Common Issues & REAL Solutions

### Backend shows "degraded" status
**Cause**: Ollama connection issue  
**Solution**: Backend expects gpt-oss but only tinyllama is loaded
```bash
# Either load gpt-oss model:
docker exec sutazai-ollama ollama pull gpt-oss

# Or update backend to use tinyllama:
# Edit backend config to use "tinyllama" instead of "gpt-oss"
```

### Agents don't do anything intelligent
**Cause**: They're stubs  
**Solution**: Implement actual logic in `/agents/*/app.py` or accept they're placeholders

### PostgreSQL has no tables
**Cause**: Migrations never run  
**Solution**: 
```bash
# Create tables manually or run migrations if they exist
docker exec -it sutazai-backend python -m app.db.init_db
```

### ChromaDB keeps restarting
**Cause**: Connection/initialization issues  
**Solution**: Check logs and ensure proper config
```bash
docker-compose logs -f chromadb
```

## 📋 What This System Can ACTUALLY Do

### ✅ Can Do:
- Local LLM text generation with TinyLlama (637MB model)
- Container monitoring via Prometheus/Grafana
- Basic web UI via Streamlit
- Store data in PostgreSQL/Redis/Neo4j (once tables created)
- Vector similarity search (once integrated)

### ❌ Cannot Do:
- Complex AI agent orchestration (stubs only)
- Distributed AI processing (no real implementation)
- Advanced NLP pipelines (not present)
- Production workloads (too many stubs)
- Any quantum computing (pure fiction)
- AGI/ASI features (marketing fiction)
- Inter-agent communication (not implemented)

## 🎯 Realistic Next Steps

Instead of chasing fantasy features:

1. **Fix Model Mismatch**: Either load gpt-oss or update code to use tinyllama
2. **Create Database Schema**: PostgreSQL has no tables
3. **Implement One Real Agent**: Pick one agent and add actual logic
4. **Fix ChromaDB**: Resolve connection issues
5. **Configure Service Mesh**: Kong has no routes defined
6. **Consolidate Requirements**: Still need to merge 75+ files into 3  
7. **Update Docker Compose**: Remove 31 non-running service definitions

## 🚨 Developer Warning

**Before starting work:**
1. Run `docker ps` to see what's actually running
2. Test endpoints directly with curl
3. Check agent code in `/agents/*/app.py` for actual logic
4. Ignore most documentation files
5. Verify claims before implementing features

**Trust only:**
- Container logs: `docker-compose logs [service]`
- Direct endpoint testing: `curl http://127.0.0.1:[port]/health`
- Actual code files (not documentation)
- This CLAUDE.md file

**After v56 cleanup, you can now trust:**
- This CLAUDE.md file (reflects reality)
- CLEANUP_COMPLETE_REPORT.md (accurate system state)
- Container logs and direct endpoint testing
- Actual code files (fantasy removed)

**Still be wary of:**
- Old documentation in git history
- Claims about complex agent features (agents are stubs)
- GPT-OSS migration completion (still using TinyLlama)
- Production readiness claims (still PoC with missing pieces)

---

## Post-Cleanup Status (v56)

### ✅ Cleanup Achievements  
- **Removed 200+ fantasy documentation files** (quantum, AGI/ASI, non-existent features)
- **Eliminated duplicate code** across BaseAgent implementations
- **Cleaned root directory** of temporary analysis/audit scripts
- **Preserved all working functionality** during cleanup
- **Created truthful documentation** reflecting actual system state

### 🔄 Still Needs Work
- **Requirements consolidation** (75+ files need merging to 3)
- **Docker compose cleanup** (31 non-running services to remove)
- **Database schema creation** (PostgreSQL empty)
- **Model configuration fix** (gpt-oss vs TinyLlama mismatch)
- **Agent logic implementation** (replace stubs with real processing)

## Summary: Clean Foundation for Real Development

This system is now a **clean Docker Compose setup** with:
- Basic web services (FastAPI + Streamlit)  
- Standard databases (PostgreSQL, Redis, Neo4j)
- Local LLM via Ollama (TinyLlama, not gpt-oss)
- Full monitoring stack (Prometheus, Grafana, Loki)
- 7 Flask stub "agents" ready for real implementation
- **Honest documentation** that won't mislead developers

**Post-cleanup, this is a solid foundation for building real AI agent functionality.**

### Development Priority
1. **Fix immediate issues** (model mismatch, database schema)
2. **Implement one real agent** to establish patterns
3. **Build incrementally** on working foundation
4. **Add complexity only after basics work**

See `CLEANUP_COMPLETE_REPORT.md` for full details of what was cleaned and what remains.