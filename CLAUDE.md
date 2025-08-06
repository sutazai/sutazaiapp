# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SutazAI is a local AI task automation system that runs entirely on-premises without cloud dependencies. The system uses Docker containers for microservices and Ollama for local LLM inference (exclusively GPT-OSS model).

**⚠️ CRITICAL REALITY CHECK ⚠️**
- **90% Fantasy**: Most documentation describes non-existent features
- **10% Working**: Basic backend, frontend, databases, and Ollama
- **Agent Truth**: Almost all "AI agents" are stub services returning hardcoded responses
- **No Magic**: No quantum computing, AGI, service mesh, or complex orchestration exists

**ALWAYS VERIFY**: Check docker-compose.yml and actual code before trusting any documentation.

## Key Commands

### Development & Testing
```bash
# Start the system (requires Docker network first)
docker network create sutazai-network
docker-compose up -d

# View logs
docker-compose logs -f [service-name]

# Stop system
docker-compose down

# Run tests (uses Makefile with Poetry)
make test           # Run default tests (unit tests)
make test-unit      # Unit tests only
make test-integration # Integration tests (starts services)
make test-e2e       # End-to-end tests with browser automation
make test-performance # Performance and load testing
make test-security  # Security testing suite
make test-docker    # Docker container testing
make test-health    # Health check testing
make test-all       # Complete test suite (quick mode)
make test-comprehensive # Full test suite (long-running)

# Coverage and reporting
make coverage       # Run tests with coverage analysis (80% threshold)
make coverage-report # Generate HTML coverage report
make report-dashboard # Generate comprehensive test dashboard

# Code quality
make lint          # Run linters (black, isort, flake8, mypy)
make format        # Auto-format code (black, isort)
make security-scan # Security analysis (bandit, safety)
make check         # Run complete system check (lint + test + security)
make quality-gate  # Standard quality gate checks
make quality-gate-strict # Strict quality gate with all tests

# CI/CD helpers
make ci-test       # CI test suite (lint + security + unit + coverage)
make ci-test-full  # Full CI test suite with integration tests

# Check system health
curl http://localhost:10010/health

# Docker management
make docker-build  # Build Docker images (backend, frontend)
make docker-test   # Test Docker images
make docker-up     # Start Docker services with docker-compose
make docker-down   # Stop Docker services

# Service management
make services-up   # Start test services (PostgreSQL, Redis)
make services-down # Stop test services
make services-status # Check service status with health checks

# Database operations
make db-migrate    # Run database migrations
make db-rollback   # Rollback last migration
make shell-db      # Open PostgreSQL shell
make shell-backend # Open backend container shell

# Maintenance and cleanup
make clean         # Clean Python caches, test artifacts
make clean-docker  # Clean Docker resources (images, volumes)
make deps-update   # Update dependencies
make deps-audit    # Audit dependencies for security issues

# Development tools
make logs          # Show service logs (all services)
make monitor       # Open monitoring dashboards (Prometheus, Grafana)
make status        # Show project status and versions
make docs          # Generate documentation with Sphinx

# Single test execution
poetry run pytest tests/test_api_endpoints.py -v
poetry run pytest -k "test_function_name" -v
```

### Key Service Ports
- Backend API: `http://localhost:10010` (FastAPI with `/docs` for interactive API docs)
- Frontend: `http://localhost:10011` (React/Streamlit UI)
- Ollama Primary: `http://localhost:11434` (GPT-OSS model serving)
- PostgreSQL: `localhost:10000` (secure instance, user: sutazai, db: sutazai)
- Redis: `localhost:10001` (secure cache instance)
- Prometheus: `http://localhost:10200` (metrics collection)
- Grafana: `http://localhost:10201` (monitoring dashboards)
- AI Metrics Exporter: `http://localhost:10209` (AI-specific metrics)
- Health Monitor: `http://localhost:10210` (system health monitoring)

### Agent Service Ports (11000-11148 range)
- Agent Orchestrator: `http://localhost:11000` (main orchestration)
- AgentZero Coordinator: `http://localhost:11001` (general-purpose agents)
- AI System Architect: `http://localhost:11002` (system design)
- AI Senior Engineers: `http://localhost:11004-11007` (backend, frontend, full-stack)
- Product/QA Management: `http://localhost:11008-11011` (PM, Scrum, QA leads)
- CI/CD & Deployment: `http://localhost:11012-11013` (pipeline, deployment)
- Security & Analysis: `http://localhost:11014` (adversarial detection)

## Architecture & Patterns

### Service Architecture
The system follows a microservices pattern with Docker containers:
- **Core Services**: `backend`, `frontend`, `postgres`, `redis`, `neo4j`, `ollama`
- **Agent Services**: Individual containers for each agent (mostly stubs returning basic responses)
- **Monitoring**: `ai-metrics-exporter`, health monitoring endpoints

### Code Organization
```
/opt/sutazaiapp/
├── backend/           # FastAPI backend service
│   ├── app/          # Main application code
│   ├── requirements.txt # Pinned dependencies
│   └── main.py       # Entry point with enterprise feature flags
├── frontend/         # React frontend
├── agents/           # AI agent implementations (mostly stubs)
│   └── */           # Each agent has app.py + requirements.txt
├── docker/          # Additional Docker services
├── workflows/       # Example automation workflows
├── scripts/         # Deployment and utility scripts
└── docker-compose.yml # Main orchestration file
```

### Key Design Decisions

1. **Local-First**: All AI inference through Ollama, no external API dependencies
2. **Stub Pattern**: Most agents are placeholder implementations - check if actually working before relying on them
3. **Port Registry**: Each service has a designated port in the 10000-12000 range
4. **Shared Network**: All services communicate via `sutazai-network` Docker network
5. **Environment Variables**: Configuration through `.env` file and Docker environment vars

### Agent Implementation Pattern
All agents follow this structure:
```python
# agents/[agent-name]/app.py
from flask import Flask, jsonify, request
app = Flask(__name__)

@app.route('/health')
def health():
    return jsonify({"status": "healthy", "agent": "agent-name"})

@app.route('/process', methods=['POST'])
def process():
    data = request.get_json()
    # Actual implementation or stub
    return jsonify({"result": "processed", "input": data})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
```

### Backend API Pattern
The backend uses FastAPI with async handlers:
```python
# backend/app/api/v1/endpoints/example.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

class RequestModel(BaseModel):
    data: str

@router.post("/process")
async def process(request: RequestModel):
    # Implementation
    return {"status": "success", "result": processed_data}
```

## Working with the Codebase

### Before Making Changes
1. Check if the feature/agent actually exists or is just documented
2. Verify service health: `docker-compose ps`
3. Check logs for errors: `docker-compose logs [service]`
4. Test locally before deploying

### Common Issues & Solutions
- **Port conflicts**: Check `/config/port-registry.yaml` for assigned ports
- **Container restarts**: Usually missing dependencies or incorrect requirements.txt
- **"Agent not working"**: Most agents are stubs - check `app.py` for actual implementation
- **Database connection**: Ensure PostgreSQL is healthy before starting dependent services

### Adding New Features
1. Reuse existing patterns and services where possible
2. Follow the existing Docker service pattern for new agents
3. Pin all dependency versions in requirements.txt
4. Update port registry if adding new services
5. Test with minimal resources (system should run on 8GB RAM)

### Critical Files to Understand
- `backend/app/main.py`: Main API entry point with feature flags
- `docker-compose.yml`: Service orchestration and dependencies
- `agents/compatibility_base_agent.py`: Base class for agent implementations
- `workflows/simple_code_review.py`: Example of practical workflow

## Important Constraints

1. **No Fantasy Features**: Only implement what can actually work today
2. **Resource Limits**: System must run on modest hardware (8GB RAM, 4 cores)
3. **Local Only**: No external API calls, everything runs offline
4. **GPT-OSS Only**: Complete migration to GPT-OSS model completed - no Mistral, TinyLlama, or other models

## Testing Strategy

The project uses pytest with extensive fixtures. Key test patterns:
- Unit tests in `tests/unit/`
- Integration tests in `tests/integration/`
- Mock external services rather than calling them
- Use `pytest-asyncio` for async code testing
- Coverage goal: 80% for new code

## Deployment Notes

Production deployment uses Docker Compose with health checks:
1. Ensure Docker network exists: `docker network create sutazai-network`
2. Start core services first: `docker-compose up -d postgres redis neo4j`
3. Wait for health checks to pass
4. Start remaining services: `docker-compose up -d`
5. Verify all services healthy: `docker-compose ps`

## Security Considerations

- All dependencies are pinned to specific versions
- Regular security scans via `make security-scan`
- No secrets in code - use environment variables
- CORS configured for local development only
- JWT tokens for API authentication (when implemented)

## Universal Deployment Script

Use `/deploy.sh` for ALL deployments (replaces 100+ legacy scripts):
```bash
# Complete system deployment
./deploy.sh deploy local       # Local development with all services
./deploy.sh deploy production  # Production deployment with health checks
./deploy.sh deploy minimal     # Minimal resource deployment

# System management
./deploy.sh status             # Check system status and container health
./deploy.sh health             # Run comprehensive health checks
./deploy.sh logs [service]     # View specific service logs (e.g., backend, ollama)
./deploy.sh stop               # Stop all services gracefully
./deploy.sh restart [service]  # Restart specific service
./deploy.sh rollback           # Rollback to previous version

# Maintenance operations
./deploy.sh cleanup            # Remove unused containers and images
./deploy.sh update             # Update and restart services
./deploy.sh backup             # Backup databases and configurations
./deploy.sh optimize           # Optimize system performance
```

Environment variables (set in `.env` file):
- `DEPLOYMENT_MODE`: Deployment type (local|staging|production|minimal)
- `LIGHTWEIGHT_MODE=true`: Use minimal resources (4GB RAM systems)
- `ENABLE_MONITORING=true`: Include Prometheus/Grafana monitoring stack
- `DEBUG=true`: Enable verbose logging and debugging
- `ENABLE_GPU=false`: GPU acceleration (disabled by default)
- `MAX_AGENTS=30`: Maximum number of agent containers
- `OLLAMA_MODEL=gpt-oss`: Default Ollama model (GPT-OSS only)

## Requirements Management

Each component has its own requirements file:
- `/backend/requirements.txt`: FastAPI backend (security-pinned)
- `/frontend/requirements.txt`: Streamlit frontend
- `/agents/*/requirements.txt`: Per-agent dependencies (many outdated/conflicting)
- `/requirements-base.txt`: Shared base dependencies

⚠️ **Warning**: Agent requirements often conflict. Most agents use Flask==2.3.3 while others use incompatible versions.

## Technology Stack Reference

**📋 AUTHORITATIVE SOURCE: `/opt/sutazaiapp/IMPORTANT/` directory**

Key reference documents (THE SOURCE OF TRUTH):
- **[ACTUAL_SYSTEM_STATUS.md](/opt/sutazaiapp/IMPORTANT/ACTUAL_SYSTEM_STATUS.md)**: Current system reality vs documentation
- **[TECHNOLOGY_STACK_REPOSITORY_INDEX.md](/opt/sutazaiapp/IMPORTANT/TECHNOLOGY_STACK_REPOSITORY_INDEX.md)**: Complete verified technology inventory
- **[ACTUAL_SYSTEM_INVENTORY.md](/opt/sutazaiapp/IMPORTANT/ACTUAL_SYSTEM_INVENTORY.md)**: Detailed component listing

These documents provide:
- Complete verified technology inventory with health status
- Service mesh and infrastructure verification  
- Integration frameworks and available tools
- Implementation priority matrix
- Verification commands for each component

## Working Components vs Fantasy

**✅ ACTUALLY WORKING (Verified per /opt/sutazaiapp/IMPORTANT/):**
- **Backend API**: FastAPI on port 10010 with basic CRUD
- **Frontend**: Streamlit on port 10011
- **Ollama**: Local LLM on port 10104 with GPT-OSS
- **Databases**: PostgreSQL (10000), Redis (10001), Neo4j (10002-10003) - All HEALTHY
- **Vector Databases**: ChromaDB (10100), Qdrant (10101), FAISS (10103) - All DEPLOYED
- **Service Mesh**: Kong Gateway (10005), Consul (10006), RabbitMQ (10007/10008) - VERIFIED OPERATIONAL
- **Agent Orchestration**: AI Agent Orchestrator (8589), Multi-Agent Coordinator (8587), Resource Arbitration (8588) - VERIFIED HEALTHY
- **Full Monitoring Stack**: Prometheus (10200), Grafana (10201), Loki (10202), AlertManager, Node Exporter, cAdvisor - All RUNNING

**⚠️ STUB IMPLEMENTATIONS (Return fake responses):**
- Most agents in `/agents/*` - Flask apps with hardcoded JSON responses
- Individual specialized agents (check actual implementation in app.py)

**❌ COMPLETE FANTASY (Don't exist at all):**
- Kong API Gateway, Consul, RabbitMQ, Vault
- Service mesh, Jaeger tracing, Elasticsearch
- Quantum computing modules (`/backend/quantum_architecture/` - deleted)
- AGI orchestration (`/config/agi_orchestration.yaml` - deleted)
- Complex agent communication (no message passing exists)
- 69 agents (only ~30 defined, <5 actually work)

## How to Verify Reality vs Fantasy

```bash
# 1. Check if service actually exists in Docker:
docker-compose ps | grep [service-name]

# 2. Check if agent has real implementation:
grep -l "stub\|placeholder\|TODO\|not implemented" agents/[agent-name]/app.py

# 3. Test actual functionality:
curl http://localhost:[port]/process -X POST -H "Content-Type: application/json" -d '{"test": "data"}'
# If response is always the same regardless of input = STUB

# 4. Check docker-compose.yml for truth about ports:
grep -A 5 [service-name]: docker-compose.yml

# 5. IGNORE these documentation files (mostly fantasy):
# - /opt/sutazaiapp/IMPORTANT/*.md (except ACTUAL_SYSTEM_STATUS.md)
# - Most files in /docs/
# - Any mention of "69 agents", "quantum", "AGI", "service mesh"
```

## Critical Codebase Rules (from CLAUDE.local.md)

1. **No Fantasy Elements**: No "magic" functions, wizards, or theoretical implementations
2. **Don't Break Working Code**: Always test existing functionality before changes
3. **Codebase Hygiene**: Follow existing patterns, no duplicate code, proper file organization
4. **Reuse Before Creating**: Check for existing solutions before writing new code
5. **Local LLMs Only**: Use Ollama with GPT-OSS, no external AI APIs

## Frequent Issues & Quick Fixes

**Container keeps restarting:**
```bash
docker-compose logs [container-name] | tail -20  # Check error
docker-compose restart [container-name]           # Quick restart
```

**Port already in use:**
```bash
# Find process using port
lsof -i :PORT_NUMBER
# Kill process or change port in docker-compose.yml
```

**Ollama not responding:**
```bash
docker-compose restart ollama
docker exec sutazai-ollama ollama pull gpt-oss  # Ensure GPT-OSS model is downloaded
docker exec sutazai-ollama ollama list          # Verify only GPT-OSS model is present
# Note: System migrated completely to GPT-OSS - no other models should be present
```

**Database connection errors:**
```bash
docker-compose restart postgres redis neo4j
# Wait 30 seconds for initialization
sleep 30
docker-compose ps  # Verify all are healthy
```

**Import errors in backend:**
```bash
# Enterprise features may not be available
# Check ENTERPRISE_FEATURES flag in backend/app/main.py
# Most advanced features are optional and fall back to basic functionality
```

## Debugging Tips

```bash
# Check container logs
docker-compose logs -f backend --tail=100

# Enter container shell for debugging
docker exec -it sutazai-backend /bin/bash

# Check resource usage
docker stats

# Test API endpoint directly
curl -X POST http://localhost:10010/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'

# Check Ollama models and status
docker exec sutazai-ollama ollama list
curl http://localhost:11434/api/tags

# PostgreSQL direct queries
docker exec -it sutazai-postgres psql -U sutazai -d sutazai
```

## Performance Optimization

- Use `LIGHTWEIGHT_MODE=true` environment variable for resource-constrained systems
- Limit Ollama memory: Set `OLLAMA_MAX_LOADED_MODELS=1` 
- Redis caching enabled for frequent queries
- PostgreSQL connection pooling configured in backend
- GPT-OSS is the only supported model

## GPT-OSS Model Migration (Completed)

**COMPLETED MIGRATION**: All model references have been migrated from multiple models (Mistral, TinyLlama, Llama, CodeLlama, Qwen, DeepSeek) to exclusively use GPT-OSS.

### Migration Summary
- **Backend Services**: All FastAPI endpoints now use GPT-OSS
- **Agent Configurations**: All 100+ agent configs updated to gpt-oss model
- **Test Files**: All test files updated to use GPT-OSS exclusively
- **Configuration Files**: Model optimization configs consolidated to GPT-OSS only
- **Scripts**: All deployment and utility scripts updated
- **Documentation**: All references updated (some legacy docs may still exist)

### Model Access
```bash
# GPT-OSS via Ollama (local inference)
curl http://localhost:11434/api/generate \
  -d '{"model": "gpt-oss", "prompt": "Hello, world!"}'

# Backend API (uses GPT-OSS internally)
curl http://localhost:10010/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test prompt"}'
```

### Verification Commands
```bash
# Check Ollama has only GPT-OSS model
docker exec sutazai-ollama ollama list
# Should show only: gpt-oss:latest

# Search for any remaining old model references (should return minimal results)
grep -r "mistral\|tinyllama\|codellama\|qwen\|deepseek" /opt/sutazaiapp --exclude-dir=archive

# Verify agent configs use GPT-OSS
grep -r "gpt-oss" /opt/sutazaiapp/agents/configs/ | wc -l
# Should show high count indicating successful migration
```

## Recent Codebase Improvements (August 2025)

### Completed Consolidation Tasks
1. **Model Migration**: Complete migration from multiple LLM models to GPT-OSS exclusively
2. **Documentation Cleanup**: Moved scattered documentation to organized `/docs/` structure
3. **Docker Compose Consolidation**: Reduced 100+ duplicate compose files to core set
4. **Script Consolidation**: Universal `/deploy.sh` replaces 100+ legacy deployment scripts
5. **Requirements Standardization**: Consolidated conflicting requirements files
6. **Port Registry**: Comprehensive port allocation system (`/config/port-registry.yaml`)

### Archive Locations
- **Legacy Documentation**: `/archive/docs-cleanup-*/` directories contain old documentation
- **Legacy Compose Files**: `/archive/docker-compose-cleanup-*/` contain archived compose variations
- **Legacy Scripts**: Most deployment scripts archived, use `/deploy.sh` exclusively

### Key Cleanup Results
- **Documentation**: ~200 markdown files reorganized from root to `/docs/` structure
- **Docker Compose**: Reduced from 50+ compose files to 3 core files (main, agents, monitoring)
- **Requirements**: Identified and documented dependency conflicts across 150+ requirements files
- **Model References**: 500+ model references updated from legacy models to GPT-OSS