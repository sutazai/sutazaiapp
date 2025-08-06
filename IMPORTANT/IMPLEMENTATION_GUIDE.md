# SutazAI Implementation Guide - Ready for Coding

> **📋 Complete Technology Stack**: See `TECHNOLOGY_STACK_REPOSITORY_INDEX.md` for comprehensive technology inventory and implementation priority matrix.

## 🎯 Overview
SutazAI is a local AI automation platform using Docker, FastAPI, and Ollama.

## ✅ What Actually Works (VERIFIED)
- FastAPI Backend (port 10010) - 70+ endpoints operational
- Streamlit Frontend (port 10011) - UI working
- PostgreSQL Database (port 10000) - HEALTHY
- Redis Cache (port 10001) - HEALTHY
- Neo4j Graph DB (ports 10002-10003) - HEALTHY
- Ollama Local LLM (port 10104) - TinyLlama currently loaded
- Service Mesh: Kong (10005), Consul (10006), RabbitMQ (10007-10008)
- Vector Stores: ChromaDB (10100), Qdrant (10101-10102), FAISS (10103)
- Monitoring: Prometheus (10200), Grafana (10201), Loki (10202)
- 5 Active AI Agents with basic health endpoints

## 🚀 Quick Start
```bash
# 1. Clone and setup
cd /opt/sutazaiapp

# 2. Start services
docker-compose up -d

# 3. Access application
open http://localhost:10011
```

## 📁 Project Structure
```
/opt/sutazaiapp/
├── backend/          # FastAPI application
│   ├── app/         # Application code
│   └── main.py      # Entry point
├── frontend/        # Streamlit UI
├── agents/          # AI agents (5 running, mostly stubs)
├── docker/          # Docker services
└── docker-compose.yml  # Service orchestration
```

## 🔧 Development Tasks

### Backend API Endpoints
```python
# File: backend/app/api/v1/endpoints.py

@router.post("/chat")
async def chat(message: str):
    # TODO: Implement Ollama integration
    pass

@router.get("/agents")
async def list_agents():
    # TODO: Return list of working agents
    pass

@router.post("/agents/{agent_id}/execute")
async def execute_agent(agent_id: str, payload: dict):
    # TODO: Execute specific agent
    pass
```

### Database Schema
```sql
-- PostgreSQL Schema
CREATE TABLE tasks (
    id SERIAL PRIMARY KEY,
    title VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE agents (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    type VARCHAR(50),
    endpoint VARCHAR(255),
    is_active BOOLEAN DEFAULT true
);
```

### Frontend Components
```python
# File: frontend/app.py
import streamlit as st

def main():
    st.title("SutazAI Control Panel")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox("Select Page", ["Dashboard", "Agents", "Tasks"])
    
    # Main content
    if page == "Dashboard":
        show_dashboard()
    elif page == "Agents":
        show_agents()
    elif page == "Tasks":
        show_tasks()
```

## 🔌 Integration Points

### Ollama Integration
```python
# File: backend/app/services/ollama_service.py
import httpx

class OllamaService:
    def __init__(self):
        self.base_url = "http://ollama:11434"  # Internal container port
    
    async def generate(self, prompt: str, model: str = "TinyLlama"):
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={"model": model, "prompt": prompt}  # TinyLlama currently loaded
            )
            return response.json()
```

### Agent Communication
```python
# File: backend/app/services/agent_service.py

class AgentService:
    async def call_agent(self, agent_name: str, data: dict):
        agent_url = f"http://{agent_name}:8080/process"
        async with httpx.AsyncClient() as client:
            response = await client.post(agent_url, json=data)
            return response.json()
```

## 📊 Testing Requirements

### Unit Tests
```python
# File: tests/test_api.py
def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200

def test_chat_endpoint():
    response = client.post("/chat", json={"message": "Hello"})
    assert response.status_code == 200
```

### Integration Tests
```python
# File: tests/test_integration.py
def test_ollama_connection():
    # Test Ollama is accessible
    pass

def test_database_connection():
    # Test PostgreSQL connection
    pass
```

## 🔒 Security Requirements
- JWT authentication on all endpoints
- Input validation using Pydantic
- SQL injection prevention via SQLAlchemy
- XSS protection in frontend
- Environment variables for secrets

## 📈 Performance Targets
- API response time < 200ms
- Support 100 concurrent users
- Database query time < 50ms
- Frontend load time < 2 seconds

## 🚢 Deployment Checklist
- [ ] All tests passing
- [ ] Docker images built
- [ ] Environment variables configured
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Backup strategy in place

## 📝 Next Steps
1. Implement missing API endpoints
2. Add Ollama integration
3. Create frontend dashboard
4. Add authentication
5. Write comprehensive tests
6. Deploy to production
