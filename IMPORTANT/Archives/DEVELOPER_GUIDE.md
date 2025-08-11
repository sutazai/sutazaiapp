# SutazAI Developer Guide

**Version:** Post-v56 Cleanup  
**Last Updated:** August 6, 2025  
**System Status:** Clean foundation ready for development  

This guide provides practical information for developers working with the cleaned-up SutazAI codebase.

## Quick Start for New Developers

### 1. Understanding the Current System
**Read these files first (in order):**
1. `CLAUDE.md` - Single source of truth about what actually works
2. `CLEANUP_COMPLETE_REPORT.md` - What was cleaned and why
3. This `DEVELOPER_GUIDE.md` - How to work with the codebase

### 2. System Overview
SutazAI is a Docker Compose-based local AI system with:
- **28 running services** out of 59 defined (the rest need cleanup)
- **FastAPI backend** with basic functionality
- **Streamlit frontend** for user interaction  
- **Local LLM** via Ollama (TinyLlama model, 637MB)
- **Full monitoring stack** (Prometheus, Grafana, Loki)
- **7 agent services** (currently stubs returning hardcoded JSON)

### 3. Getting Started
```bash
# Clone and enter directory
cd /opt/sutazaiapp

# Start system (ensure Docker is running)
docker network create sutazai-network 2>/dev/null
docker-compose up -d

# Wait for initialization
sleep 30

# Verify system health
docker ps | grep sutazai
curl http://127.0.0.1:10010/health
curl http://127.0.0.1:10104/api/tags
```

**Expected Result:** Backend shows "degraded" (model mismatch), Ollama shows TinyLlama loaded.

## Directory Structure (Post-Cleanup)

### Core Application Code
```
/backend/                   # FastAPI application
├── app/
│   ├── main.py            # Entry point, feature flags
│   ├── api/               # REST API endpoints
│   │   └── v1/endpoints/  # API route handlers
│   ├── core/              # Configuration and utilities
│   ├── services/          # Business logic services
│   └── tests/             # Backend test suite
├── requirements.txt       # Backend dependencies
└── Dockerfile            # Backend container config

/frontend/                 # Streamlit web UI
├── app.py                # Main UI application
├── requirements.txt      # Frontend dependencies
└── Dockerfile           # Frontend container config

/agents/                  # AI agent services
├── core/                # Base agent classes
│   └── simple_base_agent.py  # Main BaseAgent implementation
├── [agent-name]/        # Individual agent directories
│   ├── app.py          # Flask service (currently stubs)
│   └── Dockerfile      # Agent container config
└── configs/            # Agent configuration files

/docker/                 # Service configurations
├── [service-name]/     # Individual service configs
│   ├── Dockerfile      # Service container definition
│   └── config files   # Service-specific configuration

/config/                # System configuration
├── port-registry.yaml  # Port allocation (needs cleanup)
├── services.yaml       # Service definitions
└── [service-configs]/  # Various service configurations
```

### Supporting Directories
```
/scripts/               # Utility and deployment scripts
├── deployment/        # Deployment automation
├── monitoring/        # Health check and monitoring tools
└── utils/            # General utility scripts

/tests/                # System-wide tests
├── integration/      # Integration test suite
├── load/            # Load testing scripts
└── health/          # Health check tests

/docs/                 # Clean documentation (post-cleanup)
/reports/             # System reports and analysis
/data/                # Persistent data storage
/monitoring/          # Monitoring configuration
```

### Cleaned Up (Removed in v56)
These directories were removed during cleanup:
- `/archive/` - Multiple backup directories
- Root-level analysis scripts (`*_test.py`, `*_audit.py`)
- conceptual documentation files
- Duplicate agent directories
- Non-functional service configurations

## Key Directories and Their Purposes

### `/backend/app/` - Core API Application
**Purpose:** FastAPI-based REST API serving the main application logic  
**Current State:** Partially implemented, has model configuration mismatch  
**Key Files:**
- `main.py` - Application entry point with feature flags
- `core/config.py` - **NEEDS FIX:** Still configured for gpt-oss model
- `services/agent_orchestrator.py` - Agent coordination (basic implementation)
- `api/v1/endpoints/` - REST endpoints (mix of working and stub)

**Common Tasks:**
- Fix model configuration: Change `OLLAMA_MODEL` from "gpt-oss" to "tinyllama"
- Add new API endpoints in `/api/v1/endpoints/`
- Update service logic in `/services/`

### `/frontend/` - User Interface
**Purpose:** Streamlit-based web UI for user interaction  
**Current State:** Basic functionality, slow startup  
**Key Files:**
- `app.py` - Main UI application with multiple pages/tabs

**Common Tasks:**
- Add new UI pages/features to `app.py`
- Update frontend dependencies in `requirements.txt`
- Test UI changes: `docker-compose restart frontend`

### `/agents/` - AI Agent Services
**Purpose:** Individual AI agent services with specialized functionality  
**Current State:** 7 services running as Flask stubs (return hardcoded JSON)  
**Key Files:**
- `core/simple_base_agent.py` - **Use this BaseAgent implementation**
- `[agent-name]/app.py` - Individual agent Flask services

**Working Agent Services (Ports):**
| Agent | Port | Status | Next Steps |
|-------|------|--------|------------|
| AI Agent Orchestrator | 8589 | Stub | **Start here** - main coordination |
| Multi-Agent Coordinator | 8587 | Stub | Inter-agent communication |
| Resource Arbitration | 8588 | Stub | Resource allocation |
| Task Assignment | 8551 | Stub | Task routing |
| Hardware Optimizer | 8002 | Stub | System monitoring |
| Ollama Integration | 11015 | Stub | LLM wrapper (may work) |
| AI Metrics Exporter | 11063 | Unhealthy | Metrics collection |

**Implementing Real Agent Logic:**
1. Pick one agent (recommend AI Agent Orchestrator)
2. Replace hardcoded JSON with actual processing
3. Use BaseAgent class from `/agents/core/simple_base_agent.py`
4. Test with: `curl http://127.0.0.1:8589/process -d '{"task": "test"}'`

### `/config/` - System Configuration
**Purpose:** Centralized configuration management  
**Current State:** Needs cleanup, has 59 service definitions but only 28 running  
**Key Files:**
- `port-registry.yaml` - **NEEDS CLEANUP:** Remove non-running services
- `services.yaml` - Service definitions
- `universal_agents.json` - Agent configurations

**Common Tasks:**
- Update port registry to match reality (28 services, not 59)
- Add new service configurations
- Manage feature flags and environment variables

### `/docker/` - Service Containers
**Purpose:** Docker service definitions and configurations  
**Current State:** Contains definitions for services that may not be running  
**Structure:** Each subdirectory contains Dockerfile and configs for one service

**Common Tasks:**
- Add new services by creating new subdirectory
- Update existing service configurations
- Remove definitions for non-existent services

## How to Add New Features Properly

### 1. Backend API Endpoints
**Location:** `/backend/app/api/v1/endpoints/`

```python
# Create new endpoint file: /backend/app/api/v1/endpoints/my_feature.py
from fastapi import APIRouter
from typing import Dict, Any

router = APIRouter()

@router.post("/my-feature/process")
async def process_my_feature(data: Dict[str, Any]):
    # Implement actual logic here
    return {"result": "processed", "data": data}

@router.get("/my-feature/health")  
async def health_check():
    return {"status": "healthy"}
```

**Register in:** `/backend/app/main.py`
```python
from app.api.v1.endpoints import my_feature
app.include_router(my_feature.router, prefix="/api/v1")
```

### 2. New Agent Services
**Location:** `/agents/my-new-agent/`

1. **Create directory structure:**
```bash
mkdir -p /opt/sutazaiapp/agents/my-new-agent
```

2. **Create Flask service:** `/agents/my-new-agent/app.py`
```python
from flask import Flask, request, jsonify
import sys
import os
sys.path.append('/agents/core')
from simple_base_agent import BaseAgent

app = Flask(__name__)
agent = BaseAgent(
    name="my-new-agent",
    capabilities=["custom-processing"]
)

@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "healthy"})

@app.route('/process', methods=['POST'])  
def process():
    data = request.json
    # Implement real logic here instead of stub
    result = agent.process_task(data.get('task', ''))
    return jsonify({"result": result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8590)  # Use next available port
```

3. **Create Dockerfile:** `/agents/my-new-agent/Dockerfile`
```dockerfile
FROM python:3.11-alpine
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8590
CMD ["python", "app.py"]
```

4. **Add to docker-compose.yml:**
```yaml
  my-new-agent:
    build:
      context: ./agents/my-new-agent
    ports:
      - "8590:8590"
    networks:
      - sutazai-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8590/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

### 3. Frontend UI Features
**Location:** `/frontend/app.py`

Add new tabs/pages to the Streamlit interface:
```python
# Add to main app.py
def my_new_feature_page():
    st.header("My New Feature")
    user_input = st.text_input("Enter your input:")
    
    if st.button("Process"):
        # Call backend API
        response = requests.post(
            "http://backend:10010/api/v1/my-feature/process",
            json={"input": user_input}
        )
        if response.status_code == 200:
            st.json(response.json())

# Add to sidebar navigation
page = st.sidebar.selectbox("Choose a page", [
    "Home", "Agents", "My New Feature"  # Add your feature
])

if page == "My New Feature":
    my_new_feature_page()
```

### 4. Configuration Updates
Always update relevant configuration files:

1. **Port Registry:** Add to `/config/port-registry.yaml`
```yaml
services:
  my-new-agent:
    port: 8590
    internal_port: 8590
    type: agent
    status: active
```

2. **Agent Configuration:** Add to `/agents/configs/my-new-agent_universal.json`
```json
{
  "name": "my-new-agent",
  "capabilities": ["custom-processing"],
  "port": 8590,
  "health_endpoint": "/health",
  "process_endpoint": "/process"
}
```

## Testing Requirements

### 1. Health Check Tests (Required)
Every new service must have a health check test:

```python
# /tests/health/test_my_service.py
import requests
import pytest

def test_my_service_health():
    response = requests.get("http://127.0.0.1:8590/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_my_service_process():
    response = requests.post(
        "http://127.0.0.1:8590/process",
        json={"task": "test"}
    )
    assert response.status_code == 200
    assert "result" in response.json()
```

### 2. Integration Tests
Add integration tests to verify your feature works with the system:

```python
# /tests/integration/test_my_feature_integration.py
import requests

def test_backend_to_agent_integration():
    # Test backend can communicate with your agent
    response = requests.post(
        "http://127.0.0.1:10010/api/v1/my-feature/process",
        json={"input": "test"}
    )
    assert response.status_code == 200
```

### 3. Running Tests
```bash
# Run health checks
python -m pytest tests/health/ -v

# Run integration tests  
python -m pytest tests/integration/ -v

# Run specific test
python -m pytest tests/health/test_my_service.py -v
```

## Working with the Cleaned Codebase

### What Was Preserved
✅ **All working functionality** - Nothing functional was broken  
✅ **Core services** - Backend, frontend, databases, monitoring  
✅ **Agent infrastructure** - Flask services ready for real implementation  
✅ **Docker setup** - All containers and networking  
✅ **Configuration system** - Port registry and service configs  
✅ **Test framework** - Test structure and utilities  

### What Was Removed  
❌ **conceptual documentation** - Quantum computing, AGI/ASI claims  
❌ **Duplicate code** - Multiple BaseAgent implementations  
❌ **Analysis scripts** - Temporary root-level scripts  
❌ **Empty directories** - Non-functional agent services  
❌ **Backup clutter** - Old archive directories  

### Development Best Practices

#### 1. Always Check Reality First
```bash
# Before working on a service, verify it's actually running
docker ps | grep my-service

# Test endpoints directly
curl http://127.0.0.1:PORT/health

# Check logs for real behavior
docker-compose logs -f my-service
```

#### 2. Follow the BaseAgent Pattern
- Use `/agents/core/simple_base_agent.py` as the foundation
- Don't create new BaseAgent implementations
- Follow the Flask service pattern for agents

#### 3. Configuration Management
- Always update port registry when adding services
- Add agent configs to `/agents/configs/`
- Use environment variables for configuration
- Test configuration changes with `docker-compose config`

#### 4. Testing Requirements
- Add health check for every new service
- Test both `/health` and `/process` endpoints  
- Write integration tests for service interactions
- Use pytest for all testing

#### 5. Documentation Standards
- Update this guide when adding major features
- Document API endpoints with examples
- Keep comments honest and reality-based
- No conceptual features or capabilities

## Common Development Tasks

### Fix the Model Configuration Mismatch
**Problem:** Backend expects gpt-oss, but TinyLlama is loaded  
**Solution:**
```bash
# Option 1: Update backend config
sed -i 's/gpt-oss/tinyllama/g' /backend/app/core/config.py

# Option 2: Load gpt-oss model
docker exec sutazai-ollama ollama pull gpt-oss

# Restart backend to pick up changes
docker-compose restart backend
```

### Create Database Schema
**Problem:** PostgreSQL running but no tables exist  
**Solution:**
```bash
# Check if migrations exist
docker exec sutazai-backend find . -name "alembic" -o -name "migrations"

# Run migrations if they exist
docker exec sutazai-backend python -m alembic upgrade head

# Or create tables manually
docker exec -it sutazai-postgres psql -U sutazai -d sutazai
-- Create your tables here
```

### Implement First Real Agent
**Recommended:** Start with AI Agent Orchestrator (port 8589)

1. **Edit the agent:** `/agents/ai-agent-orchestrator/app.py`
2. **Replace stub logic** with real processing using BaseAgent
3. **Test the change:**
```bash
# Restart agent
docker-compose restart ai-agent-orchestrator

# Test with real input
curl http://127.0.0.1:8589/process -d '{"task": "analyze", "data": "test"}'
```

### Add Real Vector Search
**Current State:** Qdrant/FAISS running but not integrated  
**Steps:**
1. Add vector endpoints to backend API
2. Connect to Qdrant client in backend services
3. Implement embedding generation via Ollama
4. Add vector search UI to frontend

### Configure Service Mesh
**Current State:** Kong/Consul running but not configured  
**Steps:**
1. Add Kong routes for API gateway
2. Register services with Consul
3. Set up load balancing rules
4. Test service discovery

## Troubleshooting Guide

### Common Issues

#### Backend Shows "Degraded" Status
**Symptom:** `/health` endpoint returns degraded status  
**Cause:** Ollama model mismatch (expects gpt-oss, has tinyllama)  
**Fix:** Update config or load correct model (see above)

#### Agent Returns Only Stubs  
**Symptom:** All agents return `{"status": "healthy", "result": "processed"}`  
**Cause:** Agents are still stubs, need real implementation  
**Fix:** Replace hardcoded responses with actual logic

#### ChromaDB Keeps Restarting
**Symptom:** ChromaDB container constantly restarting  
**Cause:** Configuration or persistence volume issues  
**Fix:** Check logs and verify mount points

#### Container Won't Start
**Symptom:** Service fails to start with error  
**Debug Steps:**
```bash
# Check logs
docker-compose logs -f service-name

# Check configuration
docker-compose config | grep service-name

# Test port availability
netstat -tlnp | grep PORT
```

#### Port Conflicts
**Symptom:** "Port already in use" errors  
**Fix:** Check port registry and update conflicts
```bash
# Find what's using the port
lsof -i :PORT

# Update port-registry.yaml
# Restart affected services
```

### Performance Issues
- **Slow startup:** Normal for first run, services need initialization time
- **High memory usage:** Expected with 28 services, monitor with `docker stats`
- **Agent timeouts:** Increase timeout values in health checks

## Development Roadmap

### Immediate Tasks (This Week)
1. Fix model configuration mismatch
2. Create PostgreSQL database schema  
3. Resolve ChromaDB connection issues
4. Implement one real agent service

### Short Term (Next 2-4 weeks)  
1. Consolidate requirements files (75+ → 3)
2. Clean up docker-compose.yml (59 → 20-25 services)
3. Implement 2-3 more agent services
4. Add vector search integration

### Medium Term (Next 2-3 months)
1. Build inter-agent communication
2. Add advanced AI processing capabilities
3. Implement service mesh routing
4. Performance optimization

### Long Term (3+ months)
1. Production deployment preparation  
2. Advanced monitoring and alerting
3. Auto-scaling capabilities
4. Complex AI agent orchestration

## Getting Help

### Documentation Priority (Most Reliable → Least)
1. **This Developer Guide** - Practical, tested information
2. **CLAUDE.md** - System truth and reality check  
3. **CLEANUP_COMPLETE_REPORT.md** - What was changed and why
4. **Code comments** - In actual code files
5. **Other .md files** - May contain outdated information

### Testing Changes
- Always test new features with curl/direct API calls
- Use `docker-compose logs` to debug issues
- Verify health endpoints before deploying
- Run test suite after significant changes

### Contributing Guidelines
- Follow existing patterns (BaseAgent, Flask services, FastAPI endpoints)
- Update documentation when adding features
- Test thoroughly before committing
- Keep changes focused and atomic
- Preserve backward compatibility

---

**Remember:** This is a clean foundation ready for real development. Focus on implementing actual functionality rather than chasing complex theoretical features. Build incrementally, test thoroughly, and maintain the honesty about system capabilities that the v56 cleanup established.