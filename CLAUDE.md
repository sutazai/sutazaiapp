# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SutazAI is a local AI task automation system that runs entirely on-premises without cloud dependencies. The system uses Docker containers for microservices and Ollama for local LLM inference.

**⚠️ CRITICAL REALITY CHECK ⚠️**
- **Model Truth**: Currently using **TinyLlama** (NOT gpt-oss) on port 10104
- **35% Working**: Backend API, Frontend, Databases, Ollama, Monitoring Stack, Service Mesh Infrastructure
- **Agent Truth**: 7 agents running but are Flask stubs with basic health endpoints only
- **Service Mesh EXISTS**: Kong Gateway, Consul, RabbitMQ are VERIFIED RUNNING (not fantasy)
- **No Magic**: No quantum computing, AGI, or ASI capabilities

**ALWAYS VERIFY**: Check docker-compose.yml and actual running containers before trusting documentation.

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

### Key Service Ports (VERIFIED RUNNING)
- Backend API: `http://localhost:10010` (FastAPI v17.0.0 with 70+ endpoints, `/docs` for interactive API docs)
- Frontend: `http://localhost:10011` (Streamlit UI)
- Ollama: `http://localhost:10104` (TinyLlama model loaded - NOT gpt-oss)
- PostgreSQL: `localhost:10000` (HEALTHY but no tables created yet)
- Redis: `localhost:10001` (HEALTHY cache instance)
- Neo4j: `http://localhost:10002` (browser), `bolt://localhost:10003` (HEALTHY)
- Prometheus: `http://localhost:10200` (metrics collection)
- Grafana: `http://localhost:10201` (monitoring dashboards)
- Loki: `http://localhost:10202` (log aggregation)
- AlertManager: `http://localhost:10203` (alert routing)

### Vector Database Ports (VERIFIED RUNNING)
- Qdrant: `http://localhost:10101` (HTTP), `10102` (gRPC) - HEALTHY
- FAISS: `http://localhost:10103` - HEALTHY
- ChromaDB: `http://localhost:10100` - STARTING/Connection issues

### Service Mesh Infrastructure (VERIFIED RUNNING)
- Kong Gateway: `http://localhost:10005` (proxy), `8001` (admin) - OPERATIONAL
- Consul: `http://localhost:10006` (UI), `8600` (DNS) - SERVICE DISCOVERY RUNNING
- RabbitMQ: `localhost:10007` (AMQP), `http://localhost:10008` (management) - MESSAGE QUEUE RUNNING

### Actually Working Agent Ports (7 Flask Stubs)
- AI Agent Orchestrator: `http://localhost:8589` - Basic health endpoint
- Multi-Agent Coordinator: `http://localhost:8587` - Basic coordination stub
- Resource Arbitration: `http://localhost:8588` - Resource allocation stub
- Task Assignment: `http://localhost:8551` - Task routing stub
- Hardware Optimizer: `http://localhost:8002` - Hardware monitoring stub
- Ollama Integration: `http://localhost:11015` - Ollama interaction wrapper
- AI Metrics Exporter: `http://localhost:11063` - Metrics collection (UNHEALTHY)

## Architecture & Patterns

### Service Architecture (ACTUAL RUNNING SYSTEM)
The system runs 26-28 Docker containers:
- **Core Services**: `backend`, `frontend`, `postgres`, `redis`, `neo4j`, `ollama` (ALL HEALTHY)
- **Vector Databases**: `qdrant`, `faiss` (HEALTHY), `chromadb` (CONNECTION ISSUES)
- **Service Mesh**: `kong`, `consul`, `rabbitmq` (ALL VERIFIED OPERATIONAL)
- **Monitoring Stack**: `prometheus`, `grafana`, `loki`, `alertmanager`, `node-exporter`, `cadvisor` (ALL RUNNING)
- **Agent Services**: 7 Flask containers with stub implementations (health endpoints only)

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

1. **Local-First**: All AI inference through Ollama (TinyLlama model), no external API dependencies
2. **Stub Pattern**: ALL 7 agents are Flask stubs - no actual AI logic implemented
3. **Port Registry**: Services use ports 8000-11100 range (not well organized)
4. **Shared Network**: All services communicate via `sutazai-network` Docker network
5. **Environment Variables**: Configuration through `.env` file and Docker environment vars
6. **Database Status**: PostgreSQL running but NO TABLES created yet

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
4. **Model Reality**: Currently using **TinyLlama** (GPT-OSS migration NOT complete despite documentation)

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
- `OLLAMA_MODEL=tinyllama`: Current Ollama model (TinyLlama loaded, not gpt-oss)

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

**✅ ACTUALLY WORKING (Verified Running):**
- **Backend API**: FastAPI v17.0.0 on port 10010 with 70+ endpoints
- **Frontend**: Streamlit on port 10011
- **Ollama**: Local LLM on port 10104 with **TinyLlama** (637 MB model)
- **Databases**: PostgreSQL (10000 - no tables), Redis (10001), Neo4j (10002-10003) - All HEALTHY
- **Vector Databases**: Qdrant (10101), FAISS (10103) HEALTHY; ChromaDB (10100) CONNECTION ISSUES
- **Service Mesh**: Kong Gateway (10005), Consul (10006), RabbitMQ (10007/10008) - VERIFIED OPERATIONAL
- **Full Monitoring Stack**: Prometheus (10200), Grafana (10201), Loki (10202), AlertManager (10203) - All RUNNING
- **Monitoring Tools**: Node Exporter (10220), cAdvisor (10221), Blackbox Exporter (10229)

**⚠️ STUB IMPLEMENTATIONS (Only health endpoints):**
- 7 Working Agent Stubs (ports 8002, 8551, 8587-8589, 11015, 11063)
- All return basic JSON responses, no actual AI logic
- 44 agents defined in docker-compose but only 7 running

**❌ COMPLETE FANTASY (Don't exist at all):**
- HashiCorp Vault (secrets management)
- Jaeger tracing, Elasticsearch
- Kubernetes orchestration, Terraform
- Quantum computing modules
- AGI/ASI capabilities
- 60+ additional AI agents claimed in docs
- Complex agent communication/message passing
- GPT-OSS model (migration incomplete - using TinyLlama)

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

# 5. TRUST these documentation files (verified accurate):
# - /opt/sutazaiapp/IMPORTANT/ACTUAL_SYSTEM_STATUS.md
# - /opt/sutazaiapp/IMPORTANT/TRUTH_SYSTEM_INVENTORY.md
# - docker-compose.yml for actual service definitions
# IGNORE: Most other docs claiming advanced features
```

## Critical Codebase Rules (from CLAUDE.local.md)

1. **No Fantasy Elements**: No "magic" functions, wizards, or theoretical implementations
2. **Don't Break Working Code**: Always test existing functionality before changes
3. **Codebase Hygiene**: Follow existing patterns, no duplicate code, proper file organization
4. **Reuse Before Creating**: Check for existing solutions before writing new code
5. **Local LLMs Only**: Use Ollama with TinyLlama (current), no external AI APIs

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
docker exec sutazai-ollama ollama list          # Should show tinyllama:latest (637 MB)
# Current model: TinyLlama (NOT gpt-oss despite documentation claims)
# To test:
curl http://localhost:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "Hello"}'
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
docker exec sutazai-ollama ollama list  # Shows tinyllama:latest
curl http://localhost:10104/api/tags    # Note: port 10104, not 11434

# PostgreSQL direct queries
docker exec -it sutazai-postgres psql -U sutazai -d sutazai
```

## Performance Optimization

- Use `LIGHTWEIGHT_MODE=true` environment variable for resource-constrained systems
- Limit Ollama memory: Set `OLLAMA_MAX_LOADED_MODELS=1` 
- Redis caching enabled for frequent queries
- PostgreSQL connection pooling configured in backend (but no tables created yet)
- TinyLlama (637 MB) is currently loaded model

## Model Reality Check

**ACTUAL MODEL STATUS**: Despite documentation claims, the system is running **TinyLlama**, NOT gpt-oss

### Current Reality
- **Loaded Model**: TinyLlama (637 MB) on port 10104
- **Documentation Claims**: GPT-OSS migration complete (FALSE)
- **Config Files**: May reference gpt-oss but TinyLlama is what's actually loaded
- **Agent Configs**: Likely configured for gpt-oss but using TinyLlama in practice

### Model Access
```bash
# TinyLlama via Ollama (actual working commands)
curl http://localhost:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "Hello, world!"}'

# Backend API (uses TinyLlama internally despite config)
curl http://localhost:10010/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test prompt"}'
```

### Verification Commands
```bash
# Check what model is ACTUALLY loaded
docker exec sutazai-ollama ollama list
# Shows: tinyllama:latest    637 MB

# Check Ollama API
curl http://localhost:10104/api/tags
# Will show tinyllama model info
```

## Recent Codebase Improvements (August 2025)

### System Reality Summary

#### What's Actually Running (26-28 containers):
1. **Core Infrastructure**: All databases, cache, and Ollama working
2. **Service Mesh**: Kong, Consul, RabbitMQ VERIFIED OPERATIONAL (not fantasy!)
3. **Monitoring Stack**: Full Prometheus/Grafana/Loki stack running
4. **Agent Stubs**: 7 Flask apps with health endpoints only
5. **Model**: TinyLlama (637 MB), NOT gpt-oss

#### Documentation vs Reality Gaps:
- Claims GPT-OSS migration complete → FALSE (using TinyLlama)
- Claims 69-150 agents → Only 7 running (all stubs)
- Claims advanced AI features → Basic LLM queries only
- Claims no service mesh → Kong, Consul, RabbitMQ ACTUALLY RUNNING
- PostgreSQL "working" → Running but NO TABLES created

#### Quick Reality Check Commands:
```bash
# See what's ACTUALLY running
docker ps --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"

# Check actual model
docker exec sutazai-ollama ollama list

# Test service mesh components
curl http://localhost:10005  # Kong
curl http://localhost:10006  # Consul
curl http://localhost:10008  # RabbitMQ Management
```