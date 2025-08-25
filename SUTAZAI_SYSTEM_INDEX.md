# 🚀 SutazAI System - Complete Project Index
*Generated: 2025-08-25 | Version: 1.0 | Status: Production-Ready*

## 📊 System Overview

SutazAI is a comprehensive multi-agent AI orchestration platform featuring:
- **200+ Specialized AI Agents** with intelligent task routing
- **Service Mesh Architecture** using Consul for discovery
- **Vector Database Integration** (ChromaDB, Qdrant, FAISS)
- **Edge AI Inference** capabilities with Ollama
- **Jarvis Voice Interface** for natural language control
- **MCP Framework** with 17 specialized servers

---

## 🏗️ Architecture Components

### 1. Backend Services (Port 10010)
**Location:** `backend/`
**Technology:** FastAPI, Python 3.11+

#### Core Modules
- `app/main.py` - Application entry point with lifespan management
- `app/core/` - Core services and utilities
  - `database.py` - PostgreSQL with QueuePool (50 connections)
  - `cache.py` - Redis caching with TTL management
  - `connection_pool.py` - HTTP/DB connection pooling
  - `task_queue.py` - Async task processing
  - `claude_agent_executor.py` - Agent execution with memory management
  - `circuit_breaker_integration.py` - Fault tolerance
  - `health_monitoring.py` - System health checks

#### API Endpoints (`app/api/v1/`)
```
GET  /health                  - System health status
GET  /api/v1/agents           - List all agents
POST /api/v1/agents/create    - Create new agent
GET  /api/v1/models           - Available AI models
POST /api/v1/chat             - Chat interaction
POST /api/v1/documents/upload - Document processing
GET  /api/v1/tasks            - Task management
POST /api/v1/jarvis/command   - Jarvis voice commands
```

#### Services (`app/services/`)
- `agent_factory_service.py` - Agent creation and management
- `agent_registry_service.py` - Agent discovery and registration
- `vector_service.py` - Vector database operations
- `document_service.py` - Document processing pipeline
- `llm_service.py` - LLM model management
- `consolidated_ollama_service.py` - Ollama integration

#### Authentication (`app/auth/`)
- `middleware.py` - JWT authentication middleware
- `dependencies.py` - Auth dependency injection
- API key management
- Rate limiting

---

### 2. Frontend Interface (Port 10011)
**Location:** `frontend/`
**Technology:** Streamlit, Python

#### Main Components
- `app.py` - Main Streamlit application
- `components/resilient_ui.py` - Error-resilient UI components
- `utils/resilient_api_client.py` - API client with retry logic

#### Features
- **Dashboard** - System overview and metrics
- **AI Chat** - Interactive chat interface
- **Agent Control** - Agent management panel
- **Hardware Optimizer** - Resource optimization
- **Jarvis Voice** - Voice command interface

---

### 3. Database Layer

#### PostgreSQL (Port 10000)
- **Database:** sutazai
- **Schema:** 12+ tables for users, agents, tasks, chat history
- **Connection Pool:** QueuePool with 20 base + 30 overflow connections

#### Redis (Port 10001)
- **Purpose:** Caching, session management
- **Memory:** 128MB with LRU eviction
- **Features:** TTL support, bulk operations

#### Neo4j (Port 10002-10003)
- **Purpose:** Knowledge graph, relationships
- **Ports:** 10002 (HTTP), 10003 (Bolt)

---

### 4. Vector Databases

#### ChromaDB (Port 10100)
- **Collections:** agent_memories, task_history, knowledge_base, code_snippets
- **Embedding:** all-MiniLM-L6-v2 (384 dimensions)

#### Qdrant (Port 10101-10102)
- **Collections:** semantic_search, code_similarity
- **Features:** HNSW indexing, hybrid search

#### FAISS (Port 10103)
- **Purpose:** CPU-optimized similarity search
- **Index:** IVF with 100 clusters

---

### 5. AI Services

#### Ollama (Port 10104)
- **Models:** TinyLlama (always loaded), Qwen3 (on-demand)
- **Memory:** 4GB allocated
- **API:** REST on port 11434

---

### 6. Service Mesh

#### Consul (Port 10006)
- **Purpose:** Service discovery and health checking
- **UI:** http://localhost:10006

#### Kong API Gateway (Port 10005, 10015)
- **Proxy:** Port 10005
- **Admin:** Port 10015
- **Features:** Rate limiting, load balancing, auth

#### RabbitMQ (Port 10007-10008)
- **AMQP:** Port 10007
- **Management:** Port 10008
- **Queues:** agent.tasks, priority.queue, system.events

---

### 7. MCP Servers (Ports 11100-11199)

#### Core MCP Services
- `sequential-thinking` - Multi-step reasoning
- `claude-flow` - Workflow orchestration
- `ruv-swarm` - Swarm coordination
- `http_fetch` - HTTP operations
- `ddg` - DuckDuckGo search
- `files` - File system access
- `github` - GitHub integration
- `memory-bank-mcp` - Memory persistence
- `extended-memory` - Extended memory
- `context7` - Context management
- `playwright-mcp` - Browser automation
- `compass-mcp` - Navigation
- `knowledge-graph-mcp` - Knowledge graphs
- `language-server` - Language services
- `claude-task-runner` - Task execution

**Configuration:** `mcp-servers-config.json`

---

### 8. Agent System (Ports 11000+)

#### Agent Types
- **Coordinator Agents** - Task orchestration and routing
- **Specialist Agents** - Domain-specific expertise
- **Analysis Agents** - Data and code analysis
- **Security Agents** - Security scanning and validation
- **Documentation Agents** - Documentation generation

#### Key Agents
- `hardware_optimizer` (11019)
- `task_coordinator` (11069)
- `ollama_integration` (11071)
- `ultra_system_architect` (11200)
- `letta_agent` (11300)
- `autogpt_agent` (11301)
- `agent_zero` (11303)

---

### 9. Monitoring & Observability

#### Prometheus (Port 10200)
- **Metrics:** System, application, custom AI metrics
- **Config:** `config/prometheus/prometheus.yml`

#### Grafana (Port 10201)
- **Dashboards:** System overview, AI operations, performance
- **Default login:** admin / {from .env}

#### Node Exporter (Port 10205)
- **Purpose:** System metrics collection

#### Blackbox Exporter (Port 10204)
- **Purpose:** Endpoint availability monitoring

#### AlertManager (Port 10203)
- **Purpose:** Alert routing and management

---

## 📁 Project Structure

```
sutazaiapp/
├── backend/                  # FastAPI backend services
│   ├── app/
│   │   ├── api/v1/          # API endpoints
│   │   ├── core/            # Core services
│   │   ├── services/        # Business logic
│   │   ├── models/          # Database models
│   │   ├── auth/            # Authentication
│   │   └── mesh/            # Service mesh
│   ├── tests/               # Test suites
│   └── scripts/             # Utility scripts
├── frontend/                # Streamlit UI
│   ├── app.py              # Main application
│   ├── components/         # UI components
│   └── utils/              # Utilities
├── docker/                 # Docker configurations
│   ├── docker-compose.yml  # Main compose file
│   └── Dockerfiles/        # Service Dockerfiles
├── config/                 # Configuration files
│   ├── kong/              # API gateway config
│   ├── consul/            # Service discovery
│   ├── prometheus/        # Monitoring config
│   └── services/          # Service configs
├── scripts/               # Deployment & utility scripts
│   ├── deployment/        # Deployment automation
│   ├── monitoring/        # Monitoring scripts
│   └── mcp/              # MCP management
├── tests/                 # Integration tests
├── docs/                  # Documentation
└── SuperClaude_Framework/ # SuperClaude integration

```

---

## 🔧 Configuration Files

### Essential Configuration
- `.env` - Environment variables (create from `.env.example`)
- `docker-compose.yml` - Service orchestration
- `.mcp.json` - MCP server configurations
- `mcp-servers-config.json` - Extended MCP config
- `backend/config/mcp_mesh_registry.yaml` - Service mesh registry

### Testing
- `test_sutazai_system.py` - Comprehensive system test
- `tests/` - Test suites and results

---

## 🚀 Quick Start Commands

```bash
# System Setup
python scripts/setup.py          # Initial setup
docker-compose up -d             # Start all services

# Testing
python test_sutazai_system.py   # Run system tests
curl http://localhost:10010/health  # Check backend
curl http://localhost:10011     # Check frontend

# Monitoring
http://localhost:10201          # Grafana dashboard
http://localhost:10006          # Consul UI
http://localhost:10008          # RabbitMQ management

# Database Access
psql -h localhost -p 10000 -U sutazai -d sutazai
redis-cli -p 10001

# Logs
docker-compose logs -f backend
docker logs sutazai-backend
```

---

## 📊 System Metrics

### Resource Requirements
- **Minimum RAM:** 8GB
- **Recommended RAM:** 16GB
- **CPU Cores:** 4+ cores
- **Disk Space:** 50GB
- **Network:** 100+ Mbps

### Performance Targets
- **Response Time:** <200ms average
- **Concurrent Users:** 1000+
- **Task Processing:** 100+ tasks/minute
- **Model Inference:** <2s for simple, <10s for complex

---

## 🔒 Security Features

- **JWT Authentication** with secure token management
- **API Key Management** for service-to-service auth
- **Rate Limiting** per endpoint and user
- **CORS Protection** with whitelist
- **Input Validation** and sanitization
- **Circuit Breakers** for fault tolerance
- **SSL/TLS** for all communications
- **Environment-based** credentials (no hardcoding)

---

## 📚 Documentation

### Core Documentation
- `CLAUDE.md` - System deep dive
- `CLAUDE-FRONTEND.md` - Frontend documentation
- `CLAUDE-BACKEND.md` - Backend documentation
- `CLAUDE-INFRASTRUCTURE.md` - Infrastructure guide
- `CLAUDE-WORKFLOW.md` - Development workflow
- `CLAUDE-RULES.md` - Development rules

### Additional Resources
- `FIXES_SUMMARY.md` - Recent fixes and improvements
- `SECURITY_AUDIT_REPORT.md` - Security assessment
- `CHANGELOG.md` - Version history
- `TODO.md` - Pending improvements

---

## 🛠️ Development Tools

### SuperClaude Framework
- **Location:** `SuperClaude_Framework/`
- **Version:** 4.0.8
- **Features:** Enhanced Claude Code capabilities
- **Agents:** 14 specialized development agents
- **Modes:** 5 behavioral modes
- **Commands:** 20+ slash commands

### Testing Tools
- `test_sutazai_system.py` - System validation
- `tests/` - Unit and integration tests
- `scripts/test_*.py` - Component tests

---

## 📈 System Status

### Current State (as of 2025-08-25)
- **Backend:** ✅ Operational
- **Frontend:** ✅ Operational
- **Databases:** ✅ Connected with pooling
- **Vector DBs:** ✅ Configured
- **MCP Servers:** ✅ 17 servers configured
- **Authentication:** ✅ JWT implemented
- **Monitoring:** ✅ Prometheus + Grafana

### Known Issues
- Some MCP servers need implementation files
- Docker containers need Windows-specific configuration
- Some unnamed containers need cleanup

---

## 🔗 Related Projects

- [Agent Zero](https://github.com/frdel/agent-zero)
- [Letta](https://github.com/mysuperai/letta)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [LocalAGI](https://github.com/mudler/LocalAGI)
- [Ollama](https://ollama.com)

---

*This index provides a complete overview of the SutazAI system architecture, components, and operational details. For specific component documentation, refer to the individual documentation files listed above.*