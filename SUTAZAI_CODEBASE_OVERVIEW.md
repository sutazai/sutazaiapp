# SUTAZAI Codebase Overview - Factual Documentation
**Generated**: August 7, 2025  
**Methodology**: Direct code inspection and runtime verification  
**No speculation or hallucination - only verified facts**

---

## 1. VERIFIED SYSTEM ARCHITECTURE

### 1.1 Container Architecture (docker-compose.yml)
**File**: `/opt/sutazaiapp/docker-compose.yml` (1612 lines)

#### Actually Running Containers (28 verified via `docker ps`):
```
buildx_buildkit_sutazai-builder0
sutazai-ai-agent-orchestrator
sutazai-ai-metrics
sutazai-alertmanager
sutazai-backend
sutazai-blackbox-exporter
sutazai-cadvisor
sutazai-chromadb
sutazai-faiss-vector
sutazai-frontend
sutazai-grafana
sutazai-hardware-resource-optimizer
sutazai-loki
sutazai-multi-agent-coordinator
sutazai-neo4j
sutazai-node-exporter
sutazai-ollama
sutazai-ollama-integration-specialist-new
sutazai-postgres
sutazai-prometheus
sutazai-promtail
sutazai-qdrant
sutazai-redis
sutazai-resource-arbitration-agent
sutazai-task-assignment-coordinator
sutazaiapp-consul
sutazaiapp-kong
sutazaiapp-rabbitmq
```

### 1.2 Network Configuration
**Network Name**: sutazai-network (bridge mode)

### 1.3 Port Mappings (Verified Active)
```yaml
# Core Services
10000: PostgreSQL (sutazai-postgres)
10001: Redis (sutazai-redis)
10002: Neo4j Browser (sutazai-neo4j)
10003: Neo4j Bolt (sutazai-neo4j)
10005: Kong Gateway (sutazaiapp-kong)
10006: Consul (sutazaiapp-consul)
10007: RabbitMQ AMQP (sutazaiapp-rabbitmq)
10008: RabbitMQ Management (sutazaiapp-rabbitmq)
10010: Backend FastAPI (sutazai-backend)
10011: Frontend Streamlit (sutazai-frontend)
10100: ChromaDB (sutazai-chromadb)
10101: Qdrant HTTP (sutazai-qdrant)
10102: Qdrant gRPC (sutazai-qdrant)
10103: FAISS (sutazai-faiss-vector)
10104: Ollama (sutazai-ollama)

# Monitoring Stack
10200: Prometheus (sutazai-prometheus)
10201: Grafana (sutazai-grafana)
10202: Loki (sutazai-loki)
10203: AlertManager (sutazai-alertmanager)
10220: Node Exporter (sutazai-node-exporter)
10221: cAdvisor (sutazai-cadvisor)
10229: Blackbox Exporter (sutazai-blackbox-exporter)

# Agent Services
8002: Hardware Resource Optimizer
8551: Task Assignment Coordinator
8587: Multi-Agent Coordinator
8588: Resource Arbitration Agent
8589: AI Agent Orchestrator
11015: Ollama Integration Specialist
11063: AI Metrics Exporter
```

---

## 2. BACKEND APPLICATION STRUCTURE

### 2.1 Directory Structure
**Base Path**: `/opt/sutazaiapp/backend/app/`

```
backend/app/
├── __init__.py
├── main.py (1953 lines - FastAPI application)
├── core/
│   ├── config.py (Settings configuration)
│   ├── database.py (Database connections)
│   ├── security.py (JWT implementation)
│   ├── middleware.py (Request/response middleware)
│   ├── logging.py (Logging configuration)
│   ├── metrics.py (Prometheus metrics)
│   └── service_registry.py (Service discovery)
├── api/
│   └── v1/
│       ├── agents.py
│       ├── models.py
│       ├── vectors.py
│       └── endpoints/
├── models/
│   ├── agent.py
│   ├── task.py
│   └── user.py
├── services/
│   ├── agent_service.py
│   ├── vector_db_manager.py
│   └── ollama_service.py
└── utils/
    └── logger.py
```

### 2.2 Verified API Endpoints (from main.py)

#### Health & Monitoring
- `GET /health` (line 891) - System health check
- `GET /metrics` (line 1444) - Internal metrics
- `GET /public/metrics` (line 1526) - Public metrics
- `GET /prometheus-metrics` (line 1589) - Prometheus format

#### Core APIs
- `GET /` (line 1943) - Root endpoint
- `GET /agents` (line 953) - List agents
- `GET /models` (line 1613) - List loaded models
- `GET /api/v1/system/status` (line 1681) - System status
- `GET /api/v1/docs/endpoints` (line 1725) - API documentation

#### Chat & Processing
- `POST /chat` (line 1049) - Chat endpoint
- `POST /simple-chat` (line 1637) - Simplified chat
- `POST /public/think` (line 1112) - Public reasoning
- `POST /think` (line 1166) - Private reasoning
- `POST /execute` (line 1229) - Task execution
- `POST /reason` (line 1294) - Reasoning endpoint
- `POST /learn` (line 1356) - Learning endpoint
- `POST /improve` (line 1387) - Improvement endpoint

#### Orchestration APIs
- `POST /api/v1/orchestration/agents` (line 601)
- `POST /api/v1/orchestration/workflows` (line 629)
- `GET /api/v1/orchestration/status` (line 659)

#### Processing APIs
- `POST /api/v1/processing/process` (line 684)
- `GET /api/v1/processing/system_state` (line 719)
- `POST /api/v1/processing/creative` (line 743)

#### Self-Improvement APIs
- `POST /api/v1/improvement/analyze` (line 822)
- `POST /api/v1/improvement/apply` (line 842)

#### Agent Consensus & Models
- `POST /api/v1/agents/consensus` (line 1808)
- `POST /api/v1/models/generate` (line 1871)

### 2.3 Configuration (from core/config.py)
```python
# Verified environment variables used:
DATABASE_URL: postgresql://sutazai:sutazai_secure_2024@postgres:5432/sutazai
REDIS_URL: redis://redis:6379
NEO4J_URI: bolt://neo4j:7687
OLLAMA_BASE_URL: http://ollama:11434
CHROMADB_HOST: chromadb (fixed to port 8000)
QDRANT_HOST: qdrant
QDRANT_PORT: 6333
```

---

## 3. DATABASE SCHEMAS

### 3.1 PostgreSQL Tables (Verified via psql)
**Database**: sutazai  
**User**: sutazai  
**Tables** (14 total):

```sql
- agent_executions
- agent_health
- agents
- api_usage_logs
- chat_history
- knowledge_documents
- model_registry
- orchestration_sessions
- sessions
- system_alerts
- system_metrics
- tasks
- users
- vector_collections
```

### 3.2 Redis Configuration
- **Port**: 6379
- **Usage**: Caching layer (verified connected)
- **No persistence configuration found**

### 3.3 Neo4j Graph Database
- **Bolt Port**: 7687 (internal), 10003 (external)
- **Browser Port**: 7474 (internal), 10002 (external)
- **Auth**: Disabled (NEO4J_AUTH=none)

---

## 4. AGENT IMPLEMENTATIONS

### 4.1 Agent Directory Structure
**Path**: `/opt/sutazaiapp/agents/`

```
agents/
├── configs/         # Configuration files
├── core/           # Core agent classes
│   ├── prompt_optimizer.py
│   └── [other core files]
├── hardware-resource-optimizer/
│   ├── app.py
│   ├── main.py
│   └── Dockerfile
└── [startup scripts]
    ├── container_startup.py
    ├── universal_startup.py
    └── fastapi_wrapper.py
```

### 4.2 Verified Agent Services
**Note**: Examination of running containers shows these are Flask/FastAPI stubs

| Agent | Container Name | Port | Implementation Status |
|-------|---------------|------|----------------------|
| AI Agent Orchestrator | sutazai-ai-agent-orchestrator | 8589 | Stub (health only) |
| Multi-Agent Coordinator | sutazai-multi-agent-coordinator | 8587 | Stub (health only) |
| Resource Arbitration | sutazai-resource-arbitration-agent | 8588 | Stub (health only) |
| Task Assignment | sutazai-task-assignment-coordinator | 8551 | Stub (health only) |
| Hardware Optimizer | sutazai-hardware-resource-optimizer | 8002 | Has app.py file |
| Ollama Integration | sutazai-ollama-integration-specialist-new | 11015 | Stub (health only) |
| AI Metrics Exporter | sutazai-ai-metrics | 11063 | Running |

---

## 5. FRONTEND APPLICATION

### 5.1 Technology Stack
- **Framework**: Streamlit (verified via container)
- **Port**: 10011
- **Status**: Takes time to initialize (verified)

### 5.2 Frontend Files
**Requires manual verification** - Frontend source structure not fully examined

---

## 6. LLM INTEGRATION

### 6.1 Ollama Configuration
- **Container**: sutazai-ollama
- **Port**: 11434 (internal), 10104 (external)
- **Loaded Model**: tinyllama:latest (637MB)
- **Model ID**: 2644915ede35...
- **Quantization**: Q4_0

### 6.2 Vector Databases
| Database | Port | Status | Integration |
|----------|------|--------|-------------|
| ChromaDB | 10100 | Running | Connected (port fix applied) |
| Qdrant | 10101/10102 | Running | Connected |
| FAISS | 10103 | Running | Available |

---

## 7. MONITORING STACK

### 7.1 Prometheus
- **Port**: 10200
- **Config**: `/opt/sutazaiapp/monitoring/prometheus/prometheus.yml`
- **Scrape targets configured for all services**

### 7.2 Grafana
- **Port**: 10201
- **Default credentials**: admin/admin
- **Dashboards location**: `/opt/sutazaiapp/monitoring/grafana/dashboards/`

### 7.3 Loki & Promtail
- **Loki Port**: 10202
- **Promtail**: Log collector running as container
- **Log aggregation from all containers**

---

## 8. SERVICE MESH COMPONENTS

### 8.1 Kong API Gateway
- **Admin Port**: 8001
- **Proxy Port**: 10005
- **Status**: Running but no routes configured

### 8.2 Consul
- **Port**: 10006
- **Usage**: Service discovery (minimal configuration)

### 8.3 RabbitMQ
- **AMQP Port**: 10007
- **Management UI**: 10008
- **Status**: Running but not actively used

---

## 9. DEPLOYMENT SCRIPTS

### 9.1 Verified Shell Scripts in Root
```bash
/opt/sutazaiapp/deploy.sh
/opt/sutazaiapp/health_check.sh
/opt/sutazaiapp/validate_deployment.sh
/opt/sutazaiapp/deploy_security_infrastructure.sh
/opt/sutazaiapp/start_security_dashboard.sh
```

### 9.2 Docker Build Configuration
- **Buildkit**: Active (buildx_buildkit_sutazai-builder0)
- **Compose Version**: Using docker-compose.yml v3.8

---

## 10. VERIFIED GAPS & UNKNOWNS

### 10.1 Confirmed Missing/Stub Implementations
1. **Agent Logic**: All agents return hardcoded JSON responses
2. **Inter-agent Communication**: Not implemented despite RabbitMQ presence
3. **Workflow Engine**: Endpoints exist but no actual workflow execution
4. **Self-improvement Module**: Always returns "inactive"

### 10.2 Configuration Mismatches
1. **Model Loading**: Backend expects "gpt-oss" but only "tinyllama" is loaded
2. **Kong Routes**: Gateway running but no routes configured
3. **Service Discovery**: Consul running but minimal registration

### 10.3 Requires Manual Verification
1. Frontend source code structure and components
2. Detailed database schema definitions
3. CI/CD pipeline configuration (GitHub Actions not examined)
4. Docker image build processes
5. Security configurations and JWT implementation details

---

## 11. SOURCE CODE REFERENCES

### Key Files for Further Investigation:
- `/opt/sutazaiapp/backend/app/main.py` - Main application logic
- `/opt/sutazaiapp/docker-compose.yml` - Service definitions
- `/opt/sutazaiapp/backend/app/core/config.py` - Configuration
- `/opt/sutazaiapp/agents/*/app.py` - Agent implementations
- `/opt/sutazaiapp/monitoring/prometheus/prometheus.yml` - Monitoring config

---

## VERIFICATION METHODOLOGY

This document was created through:
1. Direct examination of docker-compose.yml
2. Runtime verification via `docker ps`
3. API endpoint extraction from source code
4. Database table listing via psql
5. Port verification through health checks
6. File system exploration of project structure
7. No speculation or assumed features

**Document Status**: FACTUAL - Based solely on verified code and runtime state  
**Last Verified**: August 7, 2025, 01:30 UTC