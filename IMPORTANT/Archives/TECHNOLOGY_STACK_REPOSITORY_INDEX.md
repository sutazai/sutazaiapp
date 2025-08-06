# Technology Stack Repository Index - ACTUAL SYSTEM

## ⚠️ REALITY CHECK ⚠️
**Last Verified**: Via `docker ps` command
**Model Reality**: System uses `tinyllama:latest` and  gpt-oss
**Agent Reality**: Only 7 agent containers running, most are stubs

## Overview

This index documents the ACTUAL technology stack currently deployed and verified in the SutazAI system.

## VERIFIED CORE TECHNOLOGIES (Running)

### Service Mesh and API Gateway
- **Kong Gateway 3.5**: API gateway with load balancing (Port 10005)
  - Status: ✅ RUNNING HEALTHY
  - Admin API: Port 8001
  - Docker Image: `kong:3.5`

- **Consul**: Service discovery and configuration (Port 10006) 
  - Status: ✅ RUNNING HEALTHY
  - Docker Image: `hashicorp/consul:latest`
  - UI Available: http://localhost:10006/ui

### Message Queue
- **RabbitMQ 3.12**: Message queuing with management UI (Port 10007)
  - Status: ✅ RUNNING HEALTHY  
  - Management UI: Port 10008
  - Docker Image: `rabbitmq:3.12-management`

### Databases (Production Ready)
- **PostgreSQL 16.3**: Primary relational database (Port 10000)
  - Status: ✅ HEALTHY
  - Docker Image: `postgres:16.3-alpine`
  - Database: sutazai, User: sutazai

- **Redis 7.2**: Caching and session store (Port 10001)
  - Status: ✅ HEALTHY
  - Docker Image: `redis:7.2-alpine`

- **Neo4j 5**: Graph database (Port 10002/10003)
  - Status: ✅ HEALTHY  
  - Docker Image: `neo4j:5-community`
  - Web UI: Port 10002, Bolt: Port 10003

### Vector Databases (AI Ready)
- **ChromaDB 0.5.0**: Document embeddings (Port 10100)
  - Status: ⚠️ HEALTH STARTING
  - Docker Image: `chromadb/chroma:0.5.0`

- **Qdrant 1.9.2**: Vector similarity search (Port 10101)
  - Status: ✅ HEALTHY
  - Docker Image: `qdrant/qdrant:v1.9.2`
  - Web UI: Port 10101, gRPC: Port 10102

- **FAISS**: Facebook AI Similarity Search (Port 10103)
  - Status: ✅ HEALTHY
  - Custom Python service with FAISS library

### AI/ML Infrastructure
- **Ollama**: Local LLM inference server (Port 10104)
  - Status: ✅ HEALTHY (19+ hours uptime)
  - Docker Image: `ollama/ollama:latest`
  - Model: **tinyllama:latest** (637 MB - ONLY model loaded)
  - API: http://localhost:10104
  - ⚠️ **NOTE**: Documentation mentions gpt-oss everywhere but it's NOT loaded

### Application Framework
- **FastAPI Backend**: Python async API framework (Port 10010)
  - Status: ✅ HEALTHY (Version 17.0.0)
  - Custom built image: `sutazaiapp-backend`
  - Swagger UI: http://localhost:10010/docs

- **Streamlit Frontend**: Python web UI framework (Port 10011)
  - Status: ✅ WORKING  
  - Custom built image: `sutazaiapp-frontend`
  - UI: http://localhost:10011

### Monitoring Stack (Full Observability)
- **Prometheus**: Metrics collection (Port 10200)
  - Status: ✅ RUNNING 14+ hours
  - Docker Image: `prom/prometheus:latest`

- **Grafana**: Metrics visualization (Port 10201)
  - Status: ✅ RUNNING 15+ hours
  - Docker Image: `grafana/grafana:latest`

- **Loki**: Log aggregation (Port 10202)
  - Status: ✅ RUNNING 15+ hours
  - Docker Image: `grafana/loki:latest`

- **AlertManager**: Alert handling (Port 10203)
  - Status: ✅ RUNNING 15+ hours
  - Docker Image: `prom/alertmanager:latest`

- **Node Exporter**: System metrics (Port 10220)
  - Status: ✅ RUNNING 15+ hours
  - Docker Image: `prom/node-exporter:latest`

- **cAdvisor**: Container metrics (Port 10221)  
  - Status: ✅ HEALTHY
  - Docker Image: `gcr.io/cadvisor/cadvisor:latest`

- **Blackbox Exporter**: Endpoint monitoring (Port 10229)
  - Status: ✅ RUNNING 15+ hours
  - Docker Image: `prom/blackbox-exporter:latest`

- **Promtail**: Log shipper for Loki
  - Status: ✅ RUNNING 15+ hours
  - Docker Image: `grafana/promtail:latest`

## VERIFIED AI AGENT SERVICES (Actually Running - Most Are Stubs)

### ✅ Currently Running (7 agents total)
- **AI Agent Orchestrator** (Port 8589) - HEALTHY 18+ hours
- **Multi-Agent Coordinator** (Port 8587) - HEALTHY 18+ hours  
- **Hardware Resource Optimizer** (Port 8002) - HEALTHY 20+ hours
- **Resource Arbitration Agent** (Port 8588) - HEALTHY 18+ hours
- **Task Assignment Coordinator** (Port 8551) - HEALTHY 18+ hours
- **Ollama Integration Specialist** (Port 11015) - HEALTHY 16+ hours
- **AI Metrics Exporter** (Port 11063) - ❌ UNHEALTHY

### ⚠️ NOT Running (But defined in docker-compose.yml)
- AgentGPT, AgentZero, Aider, AutoGen, AutoGPT
- Browser Use, CrewAI, Dify, Documind, FinRobot
- FlowiseAI, GPT-Engineer, Langflow, LlamaIndex
- PentestGPT, PrivateGPT, Semgrep, ShellGPT, Skyvern, TabbyML
- And 50+ other "agents" mentioned in docs

## DEVELOPMENT TOOLS (Available)

### Build and Test Tools
- **Make**: Build automation via Makefile
  - Commands: `make test`, `make lint`, `make format`
  - Uses Poetry for Python dependency management

- **Docker Compose**: Container orchestration
  - Network: `sutazai-network` (external)
  - 26+ services defined and running

- **Poetry**: Python dependency management
  - pyproject.toml configuration
  - Virtual environment isolation

## LANGUAGE RUNTIME VERSIONS

### Python 3.11
- Used in all custom agent services
- Base image: `python:3.11-slim`
- Package manager: pip/Poetry

### Node.js (If Needed)
- Not currently deployed
- Available for future frontend enhancements

## VERIFIED EXTERNAL DEPENDENCIES

### Python Packages (Confirmed in use)
- FastAPI: Web framework for backend
- Streamlit: UI framework for frontend  
- asyncio: Async programming
- pydantic: Data validation
- uvicorn: ASGI server
- requests: HTTP client
- redis-py: Redis client
- psycopg2: PostgreSQL client
- neo4j: Graph database client

### System Dependencies
- Docker Engine: Container runtime
- Docker Compose: Service orchestration
- curl: HTTP testing and health checks

## VERIFIED NETWORK CONFIGURATION

### External Network
- Name: `sutazai-network`
- Type: Bridge network
- All services connected

### Port Mapping Strategy
- 10000-10999: Core infrastructure
- 11000-11999: AI agents and specialized services  
- 8000-8999: Internal service communication

## WORKING ENDPOINTS FOR VERIFICATION

```bash
# Infrastructure Health Checks
curl -f http://localhost:10006/v1/status/leader    # Consul
curl -f http://localhost:10005/                    # Kong  
curl -f http://localhost:10010/health              # Backend
curl -f http://localhost:10104/                    # Ollama

# Agent Health Checks  
curl -f http://localhost:8589/health               # Orchestrator
curl -f http://localhost:8587/health               # Coordinator
curl -f http://localhost:8588/health               # Resource Arbitration

# Monitoring Health
curl -f http://localhost:10200/-/healthy           # Prometheus
curl -f http://localhost:10201/api/health          # Grafana
```

## MISSING OR NOT IMPLEMENTED

### Not Currently Available
- Kubernetes/K3s (only Docker Compose)
- Terraform infrastructure as code
- Vault secrets management
- Jaeger distributed tracing
- Elasticsearch/ELK stack
- Apache Kafka (using RabbitMQ instead)

### Planned But Not Implemented
- Auto-scaling policies
- Circuit breakers
- Advanced security policies
- Multi-region deployment
- Backup automation

## COMPREHENSIVE TECHNOLOGY REPOSITORY INDEX

### LLM Orchestration & Models
- **Ollama**: `curl -fsSL https://ollama.com/install.sh | sh`
  - **tinyllama**: `ollama run tinyllama` (Currently loaded and active)
  - **gpt-oss**: `ollama run gpt-oss` (Mentioned for production but not loaded)

### Vector Stores & Data Frameworks
- **ChromaDB**: https://github.com/johnnycode8/chromadb_quickstart (✅ DEPLOYED)
- **FAISS**: https://github.com/facebookresearch/faiss (✅ DEPLOYED)
- **Qdrant**: https://github.com/qdrant/qdrant (✅ DEPLOYED)
- **LlamaIndex**: https://github.com/run-llama/llama_index

### Agent & Automation Frameworks
- **LangChain**: https://github.com/langchain-ai/langchain
- **AutoGen**: https://github.com/ag2ai/ag2
- **CrewAI**: https://github.com/crewAIInc/crewAI
- **AutoGPT**: https://github.com/Significant-Gravitas/AutoGPT
- **AgentGPT**: https://github.com/reworkd/AgentGPT (✅ AVAILABLE IN COMPOSE)
- **Letta**: https://github.com/mysuperai/letta
- **LocalAGI**: https://github.com/mudler/Local
- **AgentZero**: https://github.com/frdel/agent-zero (✅ AVAILABLE IN COMPOSE)
- **BigAGI**: https://github.com/enricoros/big-
- **deep-agent**: https://github.com/soartech/deep-agent
- **PrivateGPT**: https://github.com/zylon-ai/private-gpt

### Workflow & UI Frameworks
- **Langflow**: https://github.com/langflow-ai/langflow
- **FlowiseAI**: https://github.com/FlowiseAI/Flowise
- **Dify**: https://github.com/langgenius/dify
- **Streamlit**: https://github.com/streamlit/streamlit (✅ DEPLOYED)

### Developer, Security & Specialized Tools
- **OpenDevin**: https://github.com/AI-App/OpenDevin.OpenDevin
- **GPT-Engineer**: https://github.com/AntonOsika/gpt-engineer
- **Aider**: https://github.com/Aider-AI/aider (✅ AVAILABLE IN COMPOSE)
- **TabbyML**: https://github.com/TabbyML/tabby
- **Semgrep**: https://github.com/semgrep/semgrep
- **ShellGPT**: https://github.com/TheR1D/shell_gpt
- **PentestGPT**: https://github.com/GreyDGL/PentestGPT
- **Documind**: https://github.com/DocumindHQ/documind
- **FinRobot**: https://github.com/AI4Finance-Foundation/FinRobot
- **Browser Use**: https://github.com/browser-use/browser-use
- **Skyvern**: https://github.com/Skyvern-AI/skyvern

### Core Libraries, Prompts & ML Frameworks
- **PyTorch**: https://github.com/pytorch/pytorch
- **TensorFlow**: https://github.com/tensorflow/tensorflow
- **JAX**: https://github.com/google/jax
- **FSDP**: https://github.com/foundation-model-stack/fms-fsdp
- **context-engineering-framework**: https://github.com/mihaicode/context-engineering-framework
- **awesome-ai-system-prompts**: https://github.com/dontriskit/awesome-ai-system-prompts
- **Awesome-Code-AI**: https://github.com/sourcegraph/awesome-code-ai

### "Jarvis" Composite System (Sources for Synthesis)
- **Microsoft JARVIS**: https://github.com/microsoft/JARVIS
- **LLM-Guy Jarvis**: https://github.com/llm-guy/jarvis
- **Dipeshpal Jarvis AI**: https://github.com/Dipeshpal/Jarvis_AI
- **Danilo Falcao Jarvis**: https://github.com/danilofalcao/jarvis
- **SreejanPersonal JARVIS**: https://github.com/SreejanPersonal/JARVIS

## IMPLEMENTATION PRIORITY MATRIX

### Phase 1: Already Deployed (✅)
- Ollama with tinyllama (currently loaded)
- ChromaDB, Qdrant, FAISS vector stores
- Streamlit frontend framework
- AgentGPT and AgentZero (available but need activation)
- Aider (available but need activation)

### Phase 2: High Priority Integration
- LangChain for agent orchestration
- AutoGen for multi-agent conversations  
- CrewAI for role-based agent teams
- Langflow for visual workflow design
- GPT-Engineer for code generation

### Phase 3: Specialized Tool Integration
- TabbyML for code completion
- Semgrep for security analysis
- FinRobot for financial analysis
- Documind for document processing
- Browser Use for web automation

### Phase 4: Advanced Frameworks
- PyTorch/TensorFlow for custom model training
- JAX for high-performance computing
- Context-engineering-framework for prompt optimization
- AutoGPT for autonomous task execution

This comprehensive technology stack provides the foundation for building a complete AI automation ecosystem.