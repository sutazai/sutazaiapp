# SUTAZAI Codebase Overview
Generated: 2025-08-07
Method: Direct repository scan and code inspection (no runtime assumptions)

This document maps the repository’s structure, active services, APIs, data flows, and gaps to guide planning and implementation.

---

## Executive Summary

- Purpose: SutazAI is an AI automation platform composed of a FastAPI backend, a Streamlit frontend, a set of agent services, tooling/services around vector DBs and LLMs, and a full monitoring stack.
- Current state: Backend and frontend are implemented and containerized; many ancillary services exist with real code, but several are optional or disabled-by-default. Messaging primarily uses Redis Streams (lightweight mesh). RabbitMQ artifacts exist but are not wired in docker-compose. DB schemas are minimal in the backend; JWT auth service defines its own schema but is not part of the main compose.
- Core dependencies: Ollama (LLM), Postgres, Redis, Qdrant, ChromaDB, Neo4j. Monitoring via Prometheus, Grafana, Loki, Promtail, cAdvisor, Node Exporter, Blackbox Exporter.
- Key gaps: RabbitMQ not provisioned; multiple services defined but not referenced in the main compose; some “enterprise” features stubbed (e.g., cognitive/reasoning engines simulate responses); DB models and migrations sparse; CI/CD references Dockerfiles not present in this repo.

---

## Repository Top-Level Map

- backend: FastAPI app and supporting modules (routers, services, orchestration, mesh, models).
- frontend: Streamlit UI consuming backend.
- services: Standalone service modules (ai-metrics exporter, FAISS vector service, LlamaIndex app, model/adapter utilities, Jarvis variants, etc.).
- agents: Agent implementations and utilities (hardware-resource-optimizer, orchestration, resource arbitration, Ollama integration helpers, messaging tools).
- monitoring: Monitoring configs and tools (Prometheus rules, Grafana dashboards, Loki/Promtail configs, exporters).
- mcp_server: Node.js MCP server with its own Dockerfile.
- auth: JWT microservice (FastAPI) + RBAC and service-account scaffolding.
- docker: Dockerfiles for many agents/services; compose variants.
- deployment: K8s/manifests, monitoring config, CI/CD templates.
- coordination_bus: Data/README for agent coordination (CSV/messages).
- schemas: Shared Pydantic schema definitions (agent/task/resource/system messages, queue config, Ollama schemas).
- servers: External MCP-style servers workspace (TypeScript/Python services; not wired into compose here).

---

## Container Orchestration (docker-compose.yml)

- Network: `sutazai-network` (external, bridge).
- Core services and published ports:
  - backend (FastAPI): 10010 -> 8000
  - frontend (Streamlit): 10011 -> 8501
  - postgres: 10000 -> 5432 (volume: postgres_data)
  - redis: 10001 -> 6379 (volume: redis_data)
  - qdrant: 10101 -> 6333 HTTP, 10102 -> 6334 gRPC (volume: qdrant_data)
  - chromadb: 10100 -> 8000 (volume: chromadb_data)
  - faiss (FAISS vector service): 10103 -> 8000 (volume: faiss_data)
  - neo4j: 10002 -> 7474, 10003 -> 7687 (volume: neo4j_data)
  - ollama: 10104 -> 10104 (heavy resource limits specified)
  - monitoring: prometheus 10200 -> 9090, grafana 10201 -> 3000, loki 10202 -> 3100, alertmanager 11108 -> 9093, blackbox 10204 -> 9115, node-exporter 10205 -> 9100, cadvisor 10206 -> 8080, postgres-exporter 10207 -> 9187, redis-exporter 10208 -> 9121
  - ai-metrics-exporter: 11068 -> 9200
  - auxiliary agents/tools (examples): crewai 10300, aider 10301, browser-use 10304, shellgpt 10307, llamaindex 10402, flowise 10401, context-framework 10404, autogen 10405, finrobot 10407, code-improver 10408, service-hub 10409, awesome-code-ai 10410, gpt-engineer 11109, hardware-resource-optimizer 11110
- Many services depend_on: postgres, redis, ollama, qdrant, chromadb, neo4j.
- Volumes: agent_workspaces, agent_outputs, per-service data volumes (qdrant_data, chromadb_data, grafana_data, loki_data, models_data, postgres_data, prometheus_data, redis_data).

Notes
- RabbitMQ, Consul, Kong are not present in docker-compose.yml; any references in code/docs are aspirational or legacy.
- Some service definitions are duplicated near the end of the compose (e.g., shellgpt, skyvern, tensorflow), likely a merge artifact. Consider deduplication.

---

## Major Components and APIs

### Backend (FastAPI) — backend/app

- Purpose: Primary API for agents/frontend; integrates with Ollama, vector stores, simple orchestration, mesh, and monitoring hooks.
- Entrypoint: `backend/Dockerfile` runs `uvicorn app.main:app` on 8000.
- Key modules:
  - app/main.py: App assembly, feature flags, router inclusion, startup/shutdown of orchestrator/reasoning/self-improvement/knowledge graph (flags default “on” but many are optional).
  - app/api/v1/endpoints: `chat.py`, `models.py`, `agents.py` (workflows + consensus/delegate), `documents.py`, `monitoring.py`, `network_recon.py`, `mesh.py`, `system.py`, `governance.py`.
  - app/services/model_manager.py: Async Ollama client via aiohttp with preload and generation/chat/embed endpoints.
  - app/mesh/redis_bus.py: Lightweight mesh built on Redis Streams for task/result topics and agent registry.
  - app/core/config.py: Settings (Postgres, Redis, Ollama, Vector DBs, feature flags). OLLAMA_HOST normalized to URL; default models are tinyllama and nomic-embed.
  - processing_engine/reasoning_engine.py: Simulated “reasoning engine” with metrics; returns structured placeholders.

APIs (selected, stable)
- System/health/metrics
  - GET `/health`: System health summary (augmented when enterprise flags enabled).
  - GET `/api/v1/system/health` and GET `/metrics`: Detailed metrics including service checks and simulated AI metrics; public version at `/public/metrics`.
- Chat and models
  - POST `/api/v1/endpoints/chat` (router path `/api/v1/chat/`): Validates inputs via XSS protection and calls ModelManager.chat/generate.
  - GET `/api/v1/endpoints/models` (router path `/api/v1/models/`): Lists Ollama tags; POST `/pull` to pull models.
- Agents/workflows
  - GET `/api/v1/endpoints/agents/`: Lists logical agents and capabilities.
  - POST `/api/v1/endpoints/agents/workflows/code-improvement`: Launches background code-improvement workflow over a directory; GET `/workflows/{id}` and `/workflows/{id}/report` for status/results.
  - POST `/api/v1/endpoints/agents/consensus`: Simulated multi-agent consensus.
  - POST `/api/v1/endpoints/agents/delegate`: Simple static agent selection.
- Vectors (in-memory/simple)
  - POST `/api/v1/vectors/initialize`, `/add`, `/search`, `/stats`, `/optimize`: Simple, in-memory demo manager (not hooked to Qdrant/Chroma at this layer).
- Mesh (Redis Streams)
  - POST `/api/v1/mesh/enqueue` -> Stream enqueue; GET `/results` -> latest results; GET `/agents` -> TTL-based registry; GET `/health`.

Status
- Working backend with real endpoints and I/O to Ollama and Redis; advanced features (knowledge graph, cognitive architecture) are guarded by try/except and environment flags and may be no-ops depending on env. Reasoning engine returns structured mock results.

### Frontend (Streamlit) — frontend

- Purpose: UI for metrics and interactions.
- Entrypoint: `streamlit run app.py` on 8501.
- Uses `BACKEND_URL` to interact with backend.
- Status: Implemented and containerized; relies on backend health.

### AI Metrics Exporter — services/ai-metrics/main.py

- Purpose: Scrapes Ollama, ChromaDB, Qdrant, Neo4j, FAISS and exposes Prometheus metrics.
- Port: 9200 (mapped to 11068).
- Key metrics: `ai_service_up`, `ai_model_loaded`, `ai_requests_total`, `vector_db_size_total`, `graph_db_*`, plus per-service gauges/histograms.
- Status: Implemented. Requires envs for service URLs and access to targets.

### FAISS Vector Service — services/faiss-vector/main.py

- Purpose: High-performance vector index over FAISS with REST API.
- Port: 8000 (mapped to 10103).
- Endpoints:
  - GET `/health`, GET `/metrics`
  - POST `/vectors` (add), POST `/search`, GET `/stats`, DELETE `/reset`
- Storage: Persists `faiss.index` and `metadata.pkl` under `FAISS_INDEX_PATH` and optionally caches stats in Redis.
- Status: Implemented; runs standalone.

### LlamaIndex Service — services/llamaindex/app.py

- Purpose: Minimal wrapper providing indexing/query over documents using LlamaIndex and Ollama.
- Port: 8080 (mapped to 10402).
- Endpoints: GET `/`, GET `/health`, POST `/index`, POST `/query` (mock responses for now).
- Status: Implemented with placeholders; intended for refinement and vector DB persistence.

### Ollama Service Wrapper — services/ollama_service.py

- Purpose: Optimized async client around Ollama with caching (Redis), batching, retries; tailored for tinyllama.
- Not a server by itself; imported by other services.
- Status: Implemented and production-ready as a library.

### Agents — agents/*

- Hardware Resource Optimizer (agents/hardware-resource-optimizer)
  - Purpose: On-demand cleanup/optimization; FastAPI app (mapped 11110 -> 8080).
  - Capabilities: Storage scan/cleanup, container cleanup, compression, etc. Uses Docker and host mounts with `pid: host` and privileged mode in compose.
  - Status: Implemented. Security-sensitive; runs privileged with host mounts.
- Agent orchestration/resource arbitration/coordinator (agents/ai_agent_orchestrator, resource_arbitration_agent, etc.)
  - Purpose: Various coordinators and orchestration helpers; not all are containerized in main compose.
  - Status: Implemented to varying degrees; many scripts and metrics present; integration paths to be validated per service.
- Messaging utilities (agents/core)
  - Includes RabbitMQ client (aio-pika) and message schemas, autoscalers, connection pools.
  - Status: RabbitMQ client implemented, but no RabbitMQ service in compose; consider either adding broker or standardizing on Redis Streams mesh.

### Auth — auth/jwt-service

- Purpose: JWT issuance/validation, service accounts; integrates with Postgres, Redis, Vault, Keycloak.
- Entrypoint: FastAPI app (`/auth/token`, `/auth/validate`, `/auth/revoke`, `/auth/service-accounts`, `/health`, `/metrics`).
- Schema (auto-initialized): `service_accounts`, `jwt_tokens`.
- Status: Implemented but not present in main docker-compose; to deploy, add service and envs.

### MCP Server — mcp_server

- Purpose: Node.js server exposing tools via MCP.
- Files: `index.js`, `Dockerfile`, `database/schema.sql`, `config.example.env`.
- Status: Implemented; included in compose (no port exposed) and depends on backend/ollama/postgres/redis; used as internal component.

### Monitoring Stack — monitoring/* and compose services

- Prometheus, Grafana, Loki, Promtail, Blackbox Exporter, Node Exporter, cAdvisor configured with provisioned dashboards and alerting rules.
- Status: Implemented. Compose mounts configuration from `monitoring/*` and `deployment/monitoring/*`.

---

## Inter-Component Dependencies and Communication

- HTTP/REST: Internal DNS names in compose (e.g., `http://backend:8000`, `http://ollama:10104`, `http://qdrant:6333`, `http://chromadb:8000`).
- Redis: `redis://redis:6379/0` default; used for caching (ollama_service, model manager), mesh bus, and health.
- Postgres: `postgresql://sutazai:${POSTGRES_PASSWORD}@postgres:5432/${POSTGRES_DB}` consumed by backend and selected services; backend’s DB usage is minimal/scaffolded.
- Vector DBs: Qdrant (6333/6334) and ChromaDB (8000) are provisioned. Backend vector endpoints are currently in-memory; FAISS and LlamaIndex run as separate services.
- Graph DB: Neo4j (7474/7687) provisioned for knowledge graph features; backend integrates conditionally via flags.
- LLM: Ollama (10104) is the primary model serving endpoint. Default model references are tinyllama; model manager can list/pull via `/api/tags` and `/api/pull`.
- Messaging: Redis Streams via `app/mesh/redis_bus.py` is active. RabbitMQ client exists but broker is not in compose; no active AMQP paths.
- Monitoring: Exporters scrape services; ai-metrics-exporter queries AI/vector/graph services and exposes `/metrics`.

Ports Registry (external:internal)
- Backend 10010:8000, Frontend 10011:8501, Postgres 10000:5432, Redis 10001:6379, Neo4j 10002:7474 + 10003:7687, ChromaDB 10100:8000, Qdrant 10101:6333 + 10102:6334, FAISS 10103:8000, Ollama 10104:10104, Prometheus 10200:9090, Grafana 10201:3000, Loki 10202:3100, Alertmanager 11108:9093, Blackbox 10204:9115, Node Exporter 10205:9100, cAdvisor 10206:8080, Postgres Exporter 10207:9187, Redis Exporter 10208:9121, AI Metrics 11068:9200, plus assorted agent ports listed earlier.

---

## Implemented Features and User Stories (Evidence in Repo)

- Model operations via Ollama (list, pull, chat, generate) — `app/services/model_manager.py`, `app/api/v1/endpoints/models.py`, `chat.py`.
- Lightweight mesh (enqueue/results/agents) — `app/api/v1/endpoints/mesh.py`, `app/mesh/redis_bus.py`.
- Code improvement workflow — `app/api/v1/endpoints/agents.py` integrates `workflows/code_improvement_workflow` for background analysis and report generation.
- FAISS search and metrics — `services/faiss-vector/main.py` with Prometheus counters/gauges and REST endpoints.
- AI metrics aggregation — `services/ai-metrics/main.py` scrapes AI/vector/graph services and publishes Prometheus metrics.
- Monitoring stack provisioning — `monitoring/*`, `deployment/monitoring/*`, compose services for Grafana/Prometheus/Loki/Promtail.
- Knowledge graph scaffolding — conditional imports and flags in `app/main.py` (Neo4j wired in compose; actual end-to-end usage gated by availability).

Supporting Documentation
- Backlog and priorities — `SUTAZAI_EPICS_AND_STORIES.md` (clear P0/P1/P2 plan), `SUTAZAI_SPRINT_PLAN.md`.
- Architecture analyses — `ARCHITECTURE_ANALYSIS_REPORT.md`, `SUTAZAI_ARCHITECTURE_ENHANCEMENT_PLAN.md`, `CRITICAL_ARCHITECTURE_AUDIT_REPORT.md`.

---

## Gaps, Missing Implementations, and Deprecated Paths

- RabbitMQ path not active: RabbitMQ client and schemas exist, but no broker in compose; all messaging presently via Redis Streams. Decision needed: add RabbitMQ service + bindings or fully standardize on Redis Streams.
- Backend DB persistence is minimal: Pydantic models exist, but no SQLAlchemy models/migrations in active backend paths; several features simulate results (reasoning engine, vectors in-memory). Consider adding a `models/` layer and Alembic migrations.
- Knowledge graph and cognitive architecture are conditionally imported; failures are caught and features degrade gracefully. Verify env flags and dependencies before enabling in production.
- Compose duplication: Some services (e.g., shellgpt, skyvern) defined twice toward the end of docker-compose.yml — clean up duplicates.
- JWT auth service not deployed: The `auth/jwt-service` is standalone and production-ready but missing from compose; DB schema creation is embedded there.
- CI/CD references Dockerfiles not in repo: `.gitlab-ci.yml` mentions `deployment/docker/Dockerfile.*` that aren’t present; align CI with actual Dockerfiles under `docker/` and component dirs.
- Security: Hardware Resource Optimizer runs privileged with host mounts and `pid: host`; ensure strong RBAC/network controls and explicit enablement in production.

---

## Deployment and CI/CD

- Local/dev: docker-compose.yml defines a multi-service stack on `sutazai-network` with healthchecks and resource constraints for several services.
- Monitoring: Prometheus scrapes exporters; Grafana mounts dashboards from repo; Alertmanager configured for Slack/PagerDuty via envs.
- CI/CD: `.gitlab-ci.yml` has stages for validate/build/test/security/performance/staging/production with reusable templates. Notable points:
  - Validate: lint/type-check/hygiene via flake8/mypy/pre-commit scripts.
  - Build: builds base images and “core services”; image paths suggest central registry; some Dockerfile paths do not exist locally and should be corrected.
  - Tests: unit/integration/e2e with docker-compose test env (compose file path referenced `docker-compose.test.yml` not in repo); adjust to current compose or add test compose.
  - Security: bandit, safety, semgrep, trivy, dependency-check, license scan.
  - Deploy: K8s helm-based scripts for staging/production with blue-green strategy. Scripts referenced under `ci-cd/scripts/*` are not present; either add them or update pipeline.

---

## Data Stores and Schemas

- Postgres: Provisioned with `postgres_data` volume. Backend uses async engine scaffolding; auth/jwt-service defines tables `service_accounts` and `jwt_tokens` and creates them on startup.
- Redis: Primary cache and mesh bus (`db 0` by default); FAISS service optionally uses `db 3`.
- Vector stores: Qdrant and ChromaDB run in compose (HTTP endpoints available); backend’s vectors API is an in-memory placeholder; FAISS service persists its own index and metadata.
- Neo4j: Enabled in compose with memory-optimized envs; backend’s knowledge-graph features are guarded by flags.

---

## High-Level Architecture Overview

1. Frontend (Streamlit) calls Backend (FastAPI) for chat, models, vector ops, workflows, and metrics.
2. Backend integrates with:
   - Ollama for LLM (list/pull/generate/chat/embeddings).
   - Redis for caching, mesh messaging, and simple agent registry.
   - Vector stores (currently via separate services: FAISS, LlamaIndex; in-backend vector routes are placeholders).
   - Optional Neo4j/ChromaDB/Qdrant via envs and service URLs.
3. Standalone services (FAISS, ai-metrics-exporter, LlamaIndex) expose their own REST endpoints and Prometheus metrics.
4. Monitoring stack scrapes services and hosts; Grafana dashboards provided.
5. Agents (e.g., hardware optimizer) can be invoked directly or orchestrated through backend workflows.

---

## Suggested Next Steps and Priorities

- Decide on messaging backbone: either add RabbitMQ to compose and wire agents, or remove AMQP code and standardize on Redis Streams mesh.
- Wire backend vector routes to Qdrant/ChromaDB or proxy to FAISS/LlamaIndex services; add persistence.
- Add minimal SQLAlchemy models + Alembic migrations to backend for tasks/agents/logs; align with EPIC 4.
- Add auth/jwt-service to compose (or integrate JWT issuance within backend) and document inter-service auth.
- Clean docker-compose.yml (remove duplicates, ensure only deployable services remain per EPIC 6.2).
- Align CI with repository reality: fix Dockerfile paths, add test compose, prune non-existent scripts.
- Review privileged agents (hardware optimizer) and gate with env and network policies.

---

## Appendix: Selected Config Extracts

Backend Ollama config normalization (app/core/config.py)
```python
@field_validator("OLLAMA_HOST", mode="before")
def validate_ollama_host(cls, v):
    if v == "0.0.0.0" or v == "ollama":
        return "http://sutazai-ollama:11434"
    if not v.startswith("http"):
        return f"http://{v}:11434"
    return v
```

Mesh enqueue example (app/api/v1/endpoints/mesh.py)
```http
POST /api/v1/mesh/enqueue
{ "topic": "code_tasks", "task": {"op": "scan", "path": "/opt/sutazaiapp"} }
-> { "id": "169...-0" }
```

FAISS service endpoints (services/faiss-vector/main.py)
```text
GET /health | GET /metrics | GET /stats
POST /vectors | POST /search | DELETE /reset
```

---

## System Diagram (Mermaid)

```mermaid
graph LR
  subgraph UI
    FE[Frontend (Streamlit) :8501]
  end

  subgraph Core API
    BE[Backend (FastAPI) :8000]
  end

  subgraph AI
    OLL[Ollama :10104]
  end

  subgraph Data Stores
    PG[(Postgres :5432)]
    RD[(Redis :6379)]
    QD[Qdrant :6333]
    CH[ChromaDB :8000]
    N4J[Neo4j :7474/7687]
  end

  subgraph Vector/ML Services
    FAISS[FAISS Service :8000]
    LI[LlamaIndex :8080]
  end

  subgraph Observability
    PR[Prometheus :9090]
    GF[Grafana :3000]
    LK[Loki :3100]
    PT[Promtail]
    NX[Node Exporter :9100]
    CA[cAdvisor :8080]
    BB[Blackbox :9115]
    AIMX[AI Metrics Exporter :9200]
  end

  FE -->|HTTP| BE
  BE -->|HTTP| OLL
  BE -->|HTTP| QD
  BE -->|HTTP| CH
  BE -->|Bolt| N4J
  BE -->|SQL| PG
  BE -->|Cache/Mesh| RD

  BE -->|HTTP| FAISS
  BE -->|HTTP| LI

  AIMX -->|scrape| OLL
  AIMX -->|scrape| QD
  AIMX -->|scrape| CH
  AIMX -->|scrape| FAISS
  AIMX -->|scrape| N4J
  PR -->|scrape| AIMX
  PR -->|scrape| NX
  PR -->|scrape| CA
  PR -->|scrape| BB
  GF --> PR
  PT --> LK
```

---

## Monitoring Flow (Mermaid)

```mermaid
flowchart LR
  subgraph Targets
    OLL[Ollama]
    QD[Qdrant]
    CH[ChromaDB]
    FAISS[FAISS Service]
    N4J[Neo4j]
    NX[Node Exporter]
    CA[cAdvisor]
    BB[Blackbox Exporter]
  end

  AIME[AI Metrics Exporter] -->|scrapes| OLL
  AIME -->|scrapes| QD
  AIME -->|scrapes| CH
  AIME -->|scrapes| FAISS
  AIME -->|scrapes| N4J

  PR[Prometheus] -->|scrapes| AIME
  PR -->|scrapes| NX
  PR -->|scrapes| CA
  PR -->|scrapes| BB

  PT[Promtail] --> LK[Loki]
  GF[Grafana] --> PR
  GF --> LK
```

---

## Service-to-Port Matrix

| Service | Container | External:Internal | Protocol |
|---|---|---|---|
| Backend (FastAPI) | sutazai-backend | 10010:8000 | HTTP |
| Frontend (Streamlit) | sutazai-frontend | 10011:8501 | HTTP |
| Postgres | sutazai-postgres | 10000:5432 | TCP |
| Redis | sutazai-redis | 10001:6379 | TCP |
| Qdrant | sutazai-qdrant | 10101:6333, 10102:6334 | HTTP/gRPC |
| ChromaDB | sutazai-chromadb | 10100:8000 | HTTP |
| FAISS Service | sutazai-faiss | 10103:8000 | HTTP |
| Neo4j | sutazai-neo4j | 10002:7474, 10003:7687 | HTTP/Bolt |
| Ollama | sutazai-ollama | 10104:10104 | HTTP |
| Prometheus | sutazai-prometheus | 10200:9090 | HTTP |
| Grafana | sutazai-grafana | 10201:3000 | HTTP |
| Loki | sutazai-loki | 10202:3100 | HTTP |
| Alertmanager | sutazai-alertmanager | 11108:9093 | HTTP |
| Blackbox Exporter | sutazai-blackbox-exporter | 10204:9115 | HTTP |
| Node Exporter | sutazai-node-exporter | 10205:9100 | HTTP |
| cAdvisor | sutazai-cadvisor | 10206:8080 | HTTP |
| Postgres Exporter | sutazai-postgres-exporter | 10207:9187 | HTTP |
| Redis Exporter | sutazai-redis-exporter | 10208:9121 | HTTP |
| AI Metrics Exporter | sutazai-ai-metrics-exporter | 11068:9200 | HTTP |
| LlamaIndex | sutazai-llamaindex | 10402:8080 | HTTP |
| Hardware Resource Optimizer | sutazai-hardware-resource-optimizer | 11110:8080 | HTTP |

Note: Some services in docker-compose.yml are disabled or duplicated; this matrix lists the core, active ones.

---

## OpenAPI Export

An OpenAPI JSON export for the backend can be generated reliably without running the server using FastAPI’s `app.openapi()`.

1) Script: `scripts/export_openapi.py` (added) writes `docs/backend_openapi.json`.
2) Run:
```
python3 scripts/export_openapi.py
```
3) Output: `docs/backend_openapi.json` (commit or attach as needed).

---

## Endpoints Summary (OpenAPI)

- Generated from `docs/backend_openapi.json`.
- Full listing: see `docs/backend_endpoints.md`.
- Totals: 28 paths, primarily under `/api/v1/*` plus public/system endpoints.

Examples
- GET `/health`, GET `/metrics`, GET `/public/metrics`
- POST `/api/v1/processing/process`, POST `/api/v1/processing/creative`
- GET `/api/v1/orchestration/status`, POST `/api/v1/orchestration/workflows`
- POST `/api/v1/agents/consensus`, GET `/agents`
- POST `/chat`, POST `/simple-chat`


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
