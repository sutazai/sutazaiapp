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
make test           # Run unit tests
make test-unit      # Unit tests only
make test-integration # Integration tests (starts services)
make test-all       # All test types
make coverage       # Generate coverage report with HTML output

# Code quality
make lint          # Run linters (black, isort, flake8, mypy)
make format        # Auto-format code (black, isort)
make security-scan # Security analysis (bandit, safety)
make check         # Run complete system check (lint + test + security)

# Check system health
curl http://localhost:10010/health

# Database operations
make db-migrate    # Run database migrations
make shell-db      # Open PostgreSQL shell
make shell-backend # Open backend container shell

# Single test execution
poetry run pytest tests/test_api_endpoints.py -v
poetry run pytest -k "test_function_name" -v
```

### Key Service Ports
- Backend API: `http://localhost:10010` (FastAPI with `/docs` for interactive API docs)
- Frontend: `http://localhost:10011`
- Ollama: `http://localhost:11434`
- PostgreSQL: `localhost:5432` (user: sutazai, password: from env, db: sutazai)
- Redis: `localhost:6379`
- Neo4j: `localhost:7474` (browser), `bolt://localhost:7687`
- AI Metrics Exporter: `http://localhost:11068`

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
4. **Ollama Models**: Only GPT-OSS model is used

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
./deploy.sh deploy local       # Local development
./deploy.sh deploy production  # Production deployment
./deploy.sh status             # Check system status
./deploy.sh health             # Run health checks
./deploy.sh logs backend       # View specific service logs
./deploy.sh rollback           # Rollback to previous version
```

Environment variables:
- `SUTAZAI_ENV`: Deployment environment (local|staging|production)
- `LIGHTWEIGHT_MODE=true`: Use minimal resources
- `ENABLE_MONITORING=true`: Include Prometheus/Grafana stack
- `DEBUG=true`: Enable verbose logging

## Requirements Management

Each component has its own requirements file:
- `/backend/requirements.txt`: FastAPI backend (security-pinned)
- `/frontend/requirements.txt`: Streamlit frontend
- `/agents/*/requirements.txt`: Per-agent dependencies (many outdated/conflicting)
- `/requirements-base.txt`: Shared base dependencies

⚠️ **Warning**: Agent requirements often conflict. Most agents use Flask==2.3.3 while others use incompatible versions.

## Working Components vs Fantasy

**✅ ACTUALLY WORKING (Verified):**
- Backend API: FastAPI on port 10010 with basic CRUD
- Frontend: Streamlit on port 10011
- Ollama: Local LLM on port 11434 with GPT-OSS
- Databases: PostgreSQL (10000), Redis (10001), Neo4j (10002-10003)
- Basic Monitoring: Prometheus (10200), Grafana (10201)

**⚠️ STUB IMPLEMENTATIONS (Return fake responses):**
- 95% of agents in `/agents/*` - just Flask apps with hardcoded JSON
- All "orchestration" agents
- All "optimization" agents
- Most "specialist" agents

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
docker exec sutazai-ollama ollama pull gpt-oss  # Ensure model is downloaded
docker exec sutazai-ollama ollama list            # Verify models
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