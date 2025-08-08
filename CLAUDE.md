# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## System Architecture

SutazAI is a local AI orchestration platform running 28 Docker containers (out of 59 defined). The system uses:
- **Backend**: FastAPI (port 10010) with feature flags for enterprise features
- **Frontend**: Streamlit (port 10011)
- **AI Model**: Ollama with TinyLlama (port 10104) - Note: Backend expects gpt-oss but TinyLlama is loaded
- **Databases**: PostgreSQL (10000), Redis (10001), Neo4j (10002/10003)
- **Vector DBs**: Qdrant (10101/10102), FAISS (10103), ChromaDB (10100 - has connection issues)
- **Monitoring**: Prometheus (10200), Grafana (10201), Loki (10202)
- **Service Mesh**: Kong (10005), Consul (10006), RabbitMQ (10007/10008) - running but not configured

## Development Commands

### System Management
```bash
# Minimal stack (recommended - 8 containers)
make up-minimal
make health-minimal
make down-minimal

# Full system (59 containers defined, 28 running)
docker-compose up -d
docker-compose down

# Service health checks
curl http://localhost:10010/health  # Backend - returns degraded (model mismatch)
curl http://localhost:10104/api/tags  # Ollama - shows tinyllama loaded
```

### Testing & Quality
```bash
# Run tests (target: 80% coverage)
make test-unit
make test-integration
make test-e2e
make test-performance

# Code quality
make lint        # Black, isort, flake8, mypy
make format      # Auto-format code
make security-scan  # Bandit + Safety

# Coverage report
make coverage
make coverage-report
```

### Documentation
```bash
# Generate API docs
python3 scripts/export_openapi.py
python3 scripts/summarize_openapi.py

# Access live docs
open http://localhost:10010/docs  # FastAPI Swagger UI
```

### Service Groups (Makefile targets)
```bash
make dbs-up         # All databases
make mesh-up        # Kong, Consul, RabbitMQ
make monitoring-up  # Prometheus, Grafana, Loki
make core-up        # Ollama, Backend, Frontend
make agents-up      # Agent services (currently stubs)
make stack-up       # Full platform in order
```

## Current System Reality

### ⚠️ Critical Issues
1. **Model Mismatch**: Backend expects `gpt-oss` but `tinyllama` is loaded
   - Fix: Update backend config or load gpt-oss model
2. **Database Schema**: PostgreSQL has no tables created
   - Fix: Run migrations or create schema
3. **Agent Services**: 7 Flask stubs returning hardcoded JSON at:
   - 8002: Hardware Resource Optimizer
   - 8551: Task Assignment Coordinator
   - 8587: Multi-Agent Coordinator
   - 8588: Resource Arbitration Agent
   - 8589: AI Agent Orchestrator
   - 11015: Ollama Integration Specialist
   - 11063: AI Metrics Exporter
4. **ChromaDB**: Connection issues, keeps restarting
5. **Service Mesh**: Kong/Consul/RabbitMQ running but not configured

### Feature Flags (in backend/app/main.py)
```python
SUTAZAI_ENTERPRISE_FEATURES  # Default: "1" (enabled)
SUTAZAI_ENABLE_KNOWLEDGE_GRAPH  # Default: "1" (enabled)
SUTAZAI_ENABLE_COGNITIVE  # Default: "1" (enabled)
ENABLE_FSDP  # Default: false (distributed training)
ENABLE_TABBY  # Default: false (code completion)
```

## API Endpoints

### Core Endpoints
- `POST /api/v1/chat/` - Chat with XSS hardening
- `GET /api/v1/models/` - List available models
- `POST /api/v1/mesh/enqueue` - Task queue via Redis Streams
- `GET /api/v1/mesh/results` - Get task results
- `GET /health` - System health (returns degraded)
- `GET /metrics` - Prometheus metrics

### Enterprise Endpoints (when enabled)
- `/api/v1/agents/*` - Agent management
- `/api/v1/tasks/*` - Task orchestration
- `/api/v1/knowledge-graph/*` - Knowledge graph operations
- `/api/v1/cognitive/*` - Cognitive architecture

## Project Structure

```
/opt/sutazaiapp/
├── backend/           # FastAPI application
│   ├── app/          # Main application code
│   ├── tests/        # Backend tests
│   └── requirements/ # Dependency management
├── frontend/         # Streamlit UI
├── agents/          # Agent services (Flask stubs)
├── docker/          # Container definitions
├── config/          # Service configurations
├── scripts/         # Utility scripts
├── tests/           # Integration tests
└── IMPORTANT/       # Critical documentation
    ├── 00_inventory/  # System inventory
    ├── 01_findings/   # Conflicts and issues
    ├── 02_issues/     # Issue tracking (16 issues)
    └── 10_canonical/  # Source of truth docs
```

## Code Standards

### Python Development
- Python 3.11+ required
- Use Poetry for dependency management (pyproject.toml)
- Black for formatting, isort for imports
- Type hints required for new code
- Async/await patterns for I/O operations
- UUID primary keys for all database tables

### Testing Requirements
- Minimum 80% test coverage
- pytest for all testing
- Use fixtures for database/service mocking
- Integration tests require Docker services

### Git Workflow
- Never commit to main directly
- Feature branches from main
- PR requires passing tests and linting
- Update CHANGELOG.md for all changes

## Common Tasks

### Fix Model Mismatch
```bash
# Option 1: Load gpt-oss model
docker exec sutazai-ollama ollama pull gpt-oss

# Option 2: Update backend to use tinyllama
# Edit backend config to use "tinyllama" instead of "gpt-oss"
```

### Create Database Schema
```bash
# PostgreSQL needs tables created
docker exec -it sutazai-backend python -m app.db.init_db
```

### Convert Agent Stub to Real Implementation
1. Navigate to `/agents/[agent-name]/app.py`
2. Replace Flask with FastAPI
3. Implement actual logic instead of returning hardcoded JSON
4. Integrate with Ollama for AI capabilities
5. Add proper error handling and logging

## Performance Considerations

- Redis caching enabled but not fully utilized
- Connection pooling needed for PostgreSQL
- Agent services consume ~100MB RAM each (stubs)
- Ollama with TinyLlama uses ~2GB RAM
- Total system uses ~15GB RAM (can be optimized to ~6GB)

## Security Notes

- No authentication on PostgreSQL, Redis, Neo4j
- Grafana using default admin/admin
- All services exposed on host network (10000-11200 range)
- JWT authentication stubbed but not implemented
- 8 critical security vulnerabilities documented in ISSUE-0003

## Monitoring

- Grafana dashboards: http://localhost:10201 (admin/admin)
- Prometheus: http://localhost:10200
- Loki for logs: http://localhost:10202
- Custom metrics exposed at `/metrics` endpoint