# Development Workflows Documentation

**Version:** 1.0  
**Last Updated:** August 8, 2025  
**Based On:** CLAUDE.md System Reality + 19 Comprehensive Engineering Standards

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Local Development Workflow](#local-development-workflow)
3. [Code Development Standards](#code-development-standards)
4. [Testing Workflows](#testing-workflows)
5. [Database Development](#database-development)
6. [Agent Development Workflow](#agent-development-workflow)
7. [API Development](#api-development)
8. [Frontend Development](#frontend-development)
9. [Debugging Workflows](#debugging-workflows)
10. [CI/CD Integration](#cicd-integration)
11. [Troubleshooting Guide](#troubleshooting-guide)
12. [Release Workflow](#release-workflow)

---

## 1. Development Environment Setup

### Prerequisites

**Required Software:**
- Docker 20.10+ and Docker Compose 2.0+
- Python 3.11+ with pip
- Git 2.30+
- Node.js 18+ (for MCP server)
- curl (for API testing)

**System Requirements:**
- 16GB RAM minimum (32GB recommended)
- 50GB free disk space
- Linux/macOS/WSL2 on Windows

### Initial Setup

```bash
# Clone the repository
git clone <repository-url>
cd sutazaiapp

# Create required network (MUST be done first)
docker network create sutazai-network 2>/dev/null

# Copy and configure environment file
cp .env.example .env
# Edit .env with your specific values

# Build and start the system
docker-compose up -d

# Wait for initialization (critical step)
sleep 30

# Verify system status
docker-compose ps
```

### Environment Variables (.env file)

```bash
# Database Configuration
POSTGRES_PASSWORD=sutazai_secure_2024
REDIS_PASSWORD=redis_secure_2024
NEO4J_PASSWORD=neo4j_secure_2024

# Ollama Configuration
OLLAMA_MODEL=tinyllama  # Current working model
OLLAMA_HOST=http://ollama:11434

# API Configuration
BACKEND_HOST=0.0.0.0
BACKEND_PORT=10010
FRONTEND_PORT=10011

# Security
JWT_SECRET_KEY=your_jwt_secret_here
ENCRYPTION_KEY=your_encryption_key_here

# Feature Flags
ENABLE_MONITORING=true
ENABLE_AGENTS=true
ENABLE_MCP=false  # Optional feature
```

### IDE Setup

**Recommended IDEs:**
- PyCharm Professional or VS Code with Python extensions
- Docker extension for container management
- REST Client extension for API testing

**VS Code Extensions:**
```json
{
  "recommendations": [
    "ms-python.python",
    "ms-python.black-formatter",
    "ms-python.isort",
    "ms-python.flake8",
    "ms-vscode.docker",
    "humao.rest-client",
    "ms-python.pytest"
  ]
}
```

---

## 2. Local Development Workflow

### Starting the System

```bash
# Start all services (standard workflow)
docker-compose up -d

# Start with specific services only
docker-compose up -d postgres redis ollama backend

# Start in development mode (with logs)
docker-compose up postgres redis ollama backend frontend
```

### Verifying Services

```bash
# Check what's actually running (CRITICAL first step)
docker ps --format "table {{.Names}}\t{{.Ports}}\t{{.Status}}"

# Test core services
curl http://127.0.0.1:10010/health | jq  # Backend API
curl http://127.0.0.1:10104/api/tags | jq  # Ollama models
curl http://127.0.0.1:8589/health | jq    # Agent stub

# Access web interfaces
open http://localhost:10011  # Frontend
open http://localhost:10201  # Grafana (admin/admin)
open http://localhost:10200  # Prometheus
```

### Development Commands

```bash
# View logs for debugging
docker-compose logs -f backend
docker-compose logs -f ollama
docker-compose logs -f [service_name]

# Restart a specific service
docker-compose restart backend

# Rebuild after code changes
docker-compose build backend
docker-compose up -d backend

# Execute commands in containers
docker exec -it sutazai-backend python -c "print('Hello')"
docker exec -it sutazai-postgres psql -U sutazai -d sutazai
```

### Hot Reloading Configuration

**Backend (FastAPI):**
```bash
# Mount source code for live reload
docker-compose -f docker-compose.yml -f docker-compose.dev.yml up -d

# Or run backend locally
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 10010 --reload
```

**Frontend (Streamlit):**
```bash
# Run locally with auto-reload
cd frontend
pip install -r requirements.txt
streamlit run app.py --server.port 10011
```

---

## 3. Code Development Standards

### Python Code Style

**Mandatory Tools (Rule #3):**
```bash
# Install development tools
pip install black isort flake8 mypy pytest pytest-cov

# Format code before commit
black .
isort .
flake8 .
mypy .
```

**Configuration Files:**

**pyproject.toml:**
```toml
[tool.black]
line-length = 88
target-version = ['py311']

[tool.isort]
profile = "black"
multi_line_output = 3

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
addopts = "-v --cov=backend --cov-report=html"
```

### Commit Message Format

**Conventional Commits (Rule #19):**
```bash
# Format: [type]: [description]
# Types: feat, fix, refactor, test, docs, chore

git commit -m "feat: add ollama health check endpoint"
git commit -m "fix: resolve docker container restart loop"
git commit -m "refactor: consolidate agent base classes"
git commit -m "test: add integration tests for database layer"
```

### Branch Naming Conventions

```bash
# Feature branches
feature/agent-health-monitoring
feature/database-migrations

# Bug fixes
fix/ollama-connection-issue
fix/docker-port-conflicts

# Refactoring
refactor/consolidate-requirements
refactor/agent-base-classes

# Documentation
docs/development-workflows
docs/api-specification
```

### Code Review Checklist

**Before Opening PR:**
- [ ] Code follows Black formatting
- [ ] All imports sorted with isort
- [ ] No flake8 violations
- [ ] Type hints added (mypy clean)
- [ ] Tests written and passing
- [ ] Documentation updated
- [ ] No hardcoded secrets
- [ ] CHANGELOG.md updated

---

## 4. Testing Workflows

### Unit Testing with pytest

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend --cov-report=html

# Run specific test files
pytest tests/test_api.py
pytest tests/test_agent_integration.py

# Run tests with verbose output
pytest -v -s

# Run tests in parallel
pytest -n auto
```

### Integration Testing

```bash
# Start test environment
docker-compose -f docker-compose.test.yml up -d

# Run integration tests
pytest tests/integration/

# Clean up test environment
docker-compose -f docker-compose.test.yml down -v
```

### Testing with Docker Containers

```bash
# Test against real containers
docker-compose up -d postgres redis
pytest tests/integration/test_database.py

# Test agent functionality
docker-compose up -d ollama
pytest tests/integration/test_agents.py
```

### Test Data Management

```bash
# Load test fixtures
docker exec -it sutazai-postgres psql -U sutazai -d sutazai -f /sql/test_data.sql

# Clean test data
docker exec -it sutazai-postgres psql -U sutazai -d sutazai -c "TRUNCATE TABLE tasks CASCADE;"
```

### Coverage Requirements

**Minimum Coverage: 80%**
```bash
# Generate coverage report
pytest --cov=backend --cov-report=html --cov-fail-under=80

# View coverage report
open htmlcov/index.html
```

---

## 5. Database Development

### Schema Changes (UUID Primary Keys)

**Current Reality:** PostgreSQL has no tables (needs initialization)

```bash
# Initialize database schema
docker exec -it sutazai-backend python -m app.db.init_db

# Or run SQL directly
docker exec -it sutazai-postgres psql -U sutazai -d sutazai -f /sql/init.sql
```

### Migration Scripts

**Create Migration:**
```python
# migrations/001_initial_schema.py
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID

def upgrade():
    op.create_table('agents',
        sa.Column('id', UUID(as_uuid=True), primary_key=True, default=uuid.uuid4),
        sa.Column('name', sa.String(255), nullable=False),
        sa.Column('status', sa.String(50), nullable=False),
        sa.Column('created_at', sa.DateTime, default=sa.func.now()),
    )
```

### Database Access Patterns

```python
# Use connection pooling
from sqlalchemy.pool import QueuePool

engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=10,
    max_overflow=20
)
```

### Backup and Restore Procedures

```bash
# Backup database
docker exec sutazai-postgres pg_dump -U sutazai sutazai > backup.sql

# Restore database
docker exec -i sutazai-postgres psql -U sutazai sutazai < backup.sql
```

---

## 6. Agent Development Workflow

### Converting Stubs to Real Implementations

**Current Reality:** 7 agent services are Flask stubs returning hardcoded responses

```python
# agents/ai_agent_orchestrator/app.py (current stub)
@app.route('/health')
def health():
    return {"status": "healthy"}

@app.route('/process', methods=['POST'])
def process():
    return {"status": "processed", "result": "hardcoded_response"}
```

**Real Implementation Pattern:**
```python
from agents.core.base_agent_v2 import BaseAgent
from agents.core.ollama_integration import OllamaClient

class AIAgentOrchestrator(BaseAgent):
    def __init__(self):
        super().__init__()
        self.ollama = OllamaClient()
    
    def process(self, task_data):
        # Real processing logic here
        response = self.ollama.generate(
            model="tinyllama",  # Current working model
            prompt=task_data.get('prompt')
        )
        return {"result": response, "status": "completed"}
```

### Testing Agent Functionality

```bash
# Test agent stub
curl -X POST http://127.0.0.1:8589/process \
  -H "Content-Type: application/json" \
  -d '{"task": "test"}'

# Test real agent implementation
pytest tests/agents/test_orchestrator.py -v
```

### Integration with Ollama

```bash
# Verify Ollama connection
curl http://127.0.0.1:10104/api/tags

# Test text generation with current model
curl -X POST http://127.0.0.1:10104/api/generate \
  -H "Content-Type: application/json" \
  -d '{"model": "tinyllama", "prompt": "Hello world"}'
```

### Performance Testing Agents

```bash
# Load test agent endpoint
k6 run scripts/load_test_agent.js

# Monitor agent performance
docker stats sutazai-ai-agent-orchestrator
```

---

## 7. API Development

### FastAPI Endpoint Creation

```python
# backend/app/api/v1/agents.py
from fastapi import APIRouter, Depends
from app.schemas.agent_messages import AgentRequest, AgentResponse

router = APIRouter()

@router.post("/agents/{agent_id}/process", response_model=AgentResponse)
async def process_agent_task(
    agent_id: str,
    request: AgentRequest,
    db: Session = Depends(get_db)
):
    # Implementation here
    return AgentResponse(status="processed")
```

### OpenAPI Documentation

```bash
# Export OpenAPI schema
python scripts/export_openapi.py > docs/backend_openapi.json

# View interactive docs
open http://localhost:10010/docs

# View ReDoc documentation
open http://localhost:10010/redoc
```

### Request/Response Validation

```python
# app/schemas/agent_messages.py
from pydantic import BaseModel, Field
from typing import Optional
from uuid import UUID

class AgentRequest(BaseModel):
    task_id: UUID
    prompt: str = Field(..., min_length=1, max_length=10000)
    model: str = Field(default="tinyllama")
    
class AgentResponse(BaseModel):
    status: str
    result: Optional[str] = None
    error: Optional[str] = None
```

### Error Handling Patterns

```python
from fastapi import HTTPException
from app.core.exceptions import AgentError

@router.post("/agents/process")
async def process_task(request: AgentRequest):
    try:
        result = await agent_service.process(request)
        return result
    except AgentError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
```

---

## 8. Frontend Development

### Streamlit Development Setup

```python
# frontend/app.py
import streamlit as st
import requests

st.set_page_config(
    page_title="SutazAI Dashboard",
    page_icon="🤖",
    layout="wide"
)

def main():
    st.title("SutazAI Agent Dashboard")
    
    # Agent status monitoring
    with st.container():
        st.subheader("Agent Status")
        display_agent_status()
```

### Component Creation

```python
# frontend/components/agent_monitor.py
def display_agent_status():
    try:
        response = requests.get("http://backend:10010/agents/status")
        if response.status_code == 200:
            agents = response.json()
            for agent in agents:
                st.metric(
                    label=agent['name'],
                    value=agent['status'],
                    delta=agent.get('health_score', 0)
                )
    except requests.RequestException as e:
        st.error(f"Failed to fetch agent status: {e}")
```

### State Management

```python
# Use Streamlit session state
if 'agent_data' not in st.session_state:
    st.session_state.agent_data = {}

def update_agent_data(agent_id, data):
    st.session_state.agent_data[agent_id] = data
    st.rerun()
```

---

## 9. Debugging Workflows

### Container Debugging

```bash
# Check container health
docker inspect sutazai-backend | jq '.[0].State'

# Access container shell
docker exec -it sutazai-backend bash

# Debug networking
docker network inspect sutazai-network

# Check resource usage
docker stats --no-stream
```

### Log Analysis

```bash
# View real-time logs
docker-compose logs -f backend

# Search logs for errors
docker-compose logs backend | grep -i error

# Export logs for analysis
docker-compose logs --since="1h" backend > debug_logs.txt
```

### Remote Debugging Setup

```python
# Add to main.py for development
if os.getenv("DEBUG_MODE") == "true":
    import debugpy
    debugpy.listen(("0.0.0.0", 5678))
    debugpy.wait_for_client()  # Optional: wait for debugger
```

### Performance Profiling

```bash
# Profile Python application
python -m cProfile -o profile_output.prof main.py

# Memory profiling
pip install memory-profiler
python -m memory_profiler main.py
```

---

## 10. CI/CD Integration

### Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Setup hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

**.pre-commit-config.yaml:**
```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  - repo: https://github.com/pycqa/isort
    rev: 5.12.0
    hooks:
      - id: isort
  - repo: https://github.com/pycqa/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
```

### Local CI Validation

```bash
# Run full validation suite
./scripts/validate_changes.sh

# Build and test
docker-compose -f docker-compose.test.yml up --build --abort-on-container-exit
```

### Docker Image Building

```bash
# Build optimized images
docker build -f backend/Dockerfile.secure -t sutazai-backend:latest backend/

# Multi-arch builds
docker buildx build --platform linux/amd64,linux/arm64 -t sutazai-backend:latest .
```

### Security Scanning with Trivy

```bash
# Scan container images
trivy image sutazai-backend:latest

# Scan filesystem
trivy fs .

# Generate security report
trivy image --format json --output security-report.json sutazai-backend:latest
```

---

## 11. Troubleshooting Guide

### Common Issues and Solutions

**Issue: Backend shows "degraded" status**
```bash
# Cause: Model mismatch (expects gpt-oss, has tinyllama)
# Solution 1: Use correct model name
curl -X POST http://127.0.0.1:10104/api/generate \
  -d '{"model": "tinyllama", "prompt": "test"}'

# Solution 2: Load expected model
docker exec sutazai-ollama ollama pull gpt-oss
```

**Issue: Agent containers keep restarting**
```bash
# Check logs for specific error
docker logs sutazai-ai-agent-orchestrator

# Common fix: ensure base dependencies
docker-compose build --no-cache ai-agent-orchestrator
```

**Issue: PostgreSQL has no tables**
```bash
# Initialize database
docker exec -it sutazai-backend python -m app.db.init_db

# Or run SQL manually
docker exec -it sutazai-postgres psql -U sutazai -d sutazai -f /sql/init.sql
```

**Issue: Port conflicts**
```bash
# Check what's using ports
netstat -tulpn | grep :10010

# Kill conflicting process
sudo kill -9 $(lsof -t -i:10010)
```

**Issue: ChromaDB keeps restarting**
```bash
# Check logs
docker-compose logs chromadb

# Restart with clean volume
docker-compose down -v
docker-compose up chromadb
```

### Service Dependency Problems

```bash
# Verify service startup order
docker-compose up postgres redis neo4j
sleep 10
docker-compose up backend
sleep 5
docker-compose up frontend
```

---

## 12. Release Workflow

### Version Bumping

```bash
# Update version in all necessary files
./scripts/bump_version.sh 1.2.3

# Files to update:
# - backend/app/__init__.py
# - frontend/app.py
# - pyproject.toml
# - package.json (if applicable)
```

### Changelog Updates (Rule #19)

**Format for CHANGELOG.md:**
```markdown
## [1.2.3] - 2025-08-08

### Added
- Development workflows documentation
- Agent health monitoring endpoints
- Database migration scripts

### Changed
- Updated Ollama integration to use TinyLlama
- Improved error handling in API endpoints

### Fixed
- Resolved Docker container restart loops
- Fixed PostgreSQL initialization scripts

### Security
- Added container security scanning
- Implemented JWT token validation
```

### Testing Checklist

**Pre-release Testing:**
- [ ] All unit tests pass (`pytest`)
- [ ] Integration tests pass
- [ ] Security scan clean (`trivy`)
- [ ] Performance benchmarks acceptable
- [ ] Documentation updated
- [ ] Environment variables documented
- [ ] Migration scripts tested

### Documentation Updates

```bash
# Update API documentation
python scripts/export_openapi.py > docs/api/openapi.json

# Update system architecture diagrams
# Update deployment guides
# Update user manuals
```

### Deployment Preparation

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Run production validation
python scripts/validate_production.py

# Generate deployment artifacts
./scripts/prepare_release.sh 1.2.3
```

---

## System Reality Reminders

**What Actually Works:**
- 59 services defined, 28 containers running
- PostgreSQL/Redis/Neo4j (healthy)
- Ollama with TinyLlama model (not gpt-oss)
- FastAPI backend (partially implemented)
- 7 Flask agent stubs (not real AI)
- Full monitoring stack (Prometheus/Grafana)

**What Are Stubs:**
- All agent `/process` endpoints return hardcoded JSON
- No actual AI logic in agents
- Service mesh exists but not configured

**Current Limitations:**
- Database has no tables (needs initialization)
- Model mismatch (tinyllama vs gpt-oss expectations)
- Agents need real implementation
- Some containers have connection issues

**Follow These Rules:**
1. No conceptual elements - only real, working code
2. Never break existing functionality
3. Analyze everything before changes
4. Reuse before creating
5. Professional approach always
6. Document all changes in CHANGELOG.md

---

## Contact and Support

For questions about these workflows:
- Check CLAUDE.md for system reality
- Review COMPREHENSIVE_ENGINEERING_STANDARDS.md
- Test endpoints directly with curl
- Examine actual code, not documentation claims

Remember: Trust container logs and direct testing over documentation!