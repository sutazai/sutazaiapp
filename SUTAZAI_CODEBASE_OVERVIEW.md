# SUTAZAI Codebase Overview (Authoritative)

Generated: 2025-08-08
Authoritative source: documents under `IMPORTANT/` (all other docs must align)

---

## Executive Summary

- Identity: Local, privacy-first AI platform (Ollama-only, no external LLMs)
- Primary capabilities: Chatbot, Code Assistant, Research Tool; expandable agents
- Architecture: Dockerized microservices; service-name addressing; single bridge network
- Observability-first: Prometheus/Grafana/Loki; exporters; health/metrics endpoints

Authoritative references: `IMPORTANT/SUTAZAI_PRD.md`, `IMPORTANT/TECHNOLOGY_STACK_REPOSITORY_INDEX.md`, `IMPORTANT/REAL_FEATURES_AND_USERSTORIES.md`

---

## System Purpose and Boundaries

From `IMPORTANT/REAL_FEATURES_AND_USERSTORIES.md`:
- Purpose: 100% local AI assistant platform with privacy-first design
- What it is not: Not cloud-dependent, not a multi-node AGI/ASI, not 69 agents
- Core capability areas:
  - Local LLM via Ollama (chat/code/research)
  - Containerized services with REST APIs
  - Local data persistence and vector search
  - Full local monitoring

---

## Verified Runtime Stack and Ports

As documented in `IMPORTANT/TECHNOLOGY_STACK_REPOSITORY_INDEX.md` (Last verified via docker ps):

- API Gateway & Discovery
  - Kong Gateway 3.5 — 10005 (admin 8001)
  - Consul — 10006 (UI available)
- Message Queue
  - RabbitMQ 3.12 — 10007 (AMQP), 10008 (Mgmt UI)
- Databases
  - PostgreSQL 16.3 — 10000 (DB: sutazai, user: sutazai)
  - Redis 7.2 — 10001
  - Neo4j 5 — 10002 (HTTP), 10003 (Bolt)
- Vector Databases
  - ChromaDB 0.5.0 — 10100
  - Qdrant 1.9.2 — 10101 (HTTP), 10102 (gRPC)
  - FAISS service — 10103 (custom Python service)
- AI/ML
  - Ollama — 10104 (tinyllama loaded; gpt-oss referenced but NOT loaded)
- Application Surfaces
  - Backend (FastAPI) — 10010 (version 17.0.0)
  - Frontend (Streamlit) — 10011
- Monitoring & Logging
  - Prometheus — 10200
  - Grafana — 10201
  - Loki — 10202
  - Alertmanager — 10203
  - Node Exporter — 10220
  - cAdvisor — 10221
  - Blackbox Exporter — 10229
  - Promtail — (running; no external port)

Network: `sutazai-network` (bridge). Port allocation strategy: 10000–10999 core infra; 11000–11999 agents; 8000–8999 internal.

---

## Service & Port Matrix (effective with override)

| Service | Container | Internal | External | Notes |
|---|---|---:|---:|---|
| Backend (FastAPI) | `sutazai-backend` | 8000 | 10010 | Primary API |
| Frontend (Streamlit) | `sutazai-frontend` | 8501 | 10011 | UI |
| Ollama | `sutazai-ollama` | 10104 | 10104 | Local LLMs |
| ChromaDB | `sutazai-chromadb` | 8000 | 10100 | Token auth |
| Qdrant (HTTP) | `sutazai-qdrant` | 6333 | 10101 |  |
| Qdrant (gRPC) | `sutazai-qdrant` | 6334 | 10102 |  |
| FAISS svc | `sutazai-faiss` | 8000 | 10103 | Vector service |
| Postgres | `sutazai-postgres` | 5432 | 10000 | DB `sutazai` |
| Redis | `sutazai-redis` | 6379 | 10001 | AOF disabled |
| Neo4j (HTTP/Bolt) | `sutazai-neo4j` | 7474/7687 | 10002/10003 | Optional |
| Prometheus | `sutazai-prometheus` | 9090 | 10200 | Monitoring |
| Grafana | `sutazai-grafana` | 3000 | 10201 | Dashboards |
| Loki | `sutazai-loki` | 3100 | 10202 | Logs |
| Alertmanager | `sutazai-alertmanager` | 9093 | 10203 | Alerts |
| Node Exporter | `sutazai-node-exporter` | 9100 | 10220 | Host metrics |
| cAdvisor | `sutazai-cadvisor` | 8080 | 10221 | Container metrics |
| Blackbox Exporter | `sutazai-blackbox-exporter` | 9115 | 10229 | Probes |
| Kong (proxy) | `sutazai-kong` | 8000 | 10005 | override |
| Consul (UI) | `sutazai-consul` | 8500 | 10006 | override |
| AI Metrics Exporter | `sutazai-ai-metrics-exporter` | 9200 | 11063 | override |
| Agents (examples) | jarvis-* | 8080 | 11101/11102/11103/11110/11150 | See compose |

---

## Core Components and Roles

- Backend API (FastAPI)
  - Role: Primary API for chatbot/code assistant/research operations; integrates Ollama, vector DBs, message queue, data stores
  - Health: GET http://localhost:10010/health (see verification commands below)
- Frontend (Streamlit)
  - Role: Web UI for the three core functions and metrics
- Ollama (LLM inference)
  - Role: Local text generation; tinyllama is the default/only verified loaded model
- Vector Stores
  - Qdrant, ChromaDB, FAISS service for embeddings, search, and experimentation
- Data Stores
  - Postgres (primary RDBMS), Redis (cache/session), Neo4j (graph operations when enabled)
- API Gateway & Discovery
  - Kong (routing, rate limiting, auth), Consul (service discovery/config)
- Messaging
  - RabbitMQ (primary queueing per IMPORTANT; Kafka not used)
- Observability
  - Prometheus, Grafana, Loki/Promtail, Node Exporter, cAdvisor, Blackbox, Alertmanager

Agent services (runtime reality): several active; others behind profiles (disabled by default in override)

---

## Inter-Component Dependencies & Communication

- Networking
  - Single bridge network: `sutazai-network`. All services communicate via service names (no localhost in inter-service calls).
- Backend service dependencies (from `backend/app/core/config.py` and compose env)
  - Postgres: `postgres:5432` → `DATABASE_URL` or `postgresql+asyncpg://...`
  - Redis: `redis:6379` → `REDIS_URL` (used for mesh, rate limiting, message bus)
  - ChromaDB: `chromadb:8000` (token auth via `CHROMADB_API_KEY`)
  - Qdrant: `qdrant:6333` (HTTP) / `qdrant:6334` (gRPC)
  - Neo4j: `bolt://neo4j:7687` (optional)
  - Ollama: `http://ollama:10104` (models, generation)
- Messaging
  - Primary in-code bus: Redis (see `app/mesh/redis_bus.py`, `app/orchestration/message_bus.py`, and `ai_agents/*` using `redis.asyncio`/`aioredis`).
  - RabbitMQ: provisioned in infrastructure and health-checked by scripts; limited/no direct backend runtime usage observed.
- Observability
  - Health/metrics endpoints implemented in backend; Prometheus scrapes exporters and services listed in compose.
- External integrations
  - None (local-only). LLM via `ollama:10104`.

---

## Working Endpoints (for quick validation)

From `IMPORTANT/TECHNOLOGY_STACK_REPOSITORY_INDEX.md`:

```bash
# Infrastructure
curl -f http://localhost:10006/v1/status/leader   # Consul
curl -f http://localhost:10005/                   # Kong
curl -f http://localhost:10010/health             # Backend
curl -f http://localhost:10104/                   # Ollama

# Agents (examples)
curl -f http://localhost:8589/health              # Orchestrator
curl -f http://localhost:8587/health              # Coordinator
curl -f http://localhost:8588/health              # Resource Arbitration

# Monitoring
curl -f http://localhost:10200/-/healthy          # Prometheus
curl -f http://localhost:10201/api/health         # Grafana
```

---

## Data Layer and Schema Status

From `IMPORTANT/DATABASE_SCHEMA.sql`:
- Current status: Postgres running but EMPTY (no tables yet). The SQL file contains the PLANNED schema.
- Planned tables (SERIAL PKs as documented):
  - users, agents, tasks, chat_history, agent_executions, system_metrics
- Planned indexes for performance are listed in the file
- Seed/insert examples provided for default agents

Note: The schema as documented uses SERIAL keys; migration to UUID and additional indexing can be planned during implementation if required by forthcoming standards.

---

## Deployment & Orchestration

- Compose
  - `docker-compose.yml` declares 25+ services; `docker-compose.override.yml` aligns external ports to IMPORTANT and disables most agents via profiles.
  - Resource limits and healthchecks present for core services; override reduces resource footprints for dev.
- Addressing
  - Services address each other by Docker service names. CORS for frontend uses host ports intentionally.
- Profiles
  - Most agent services are behind `profiles: ["disabled"]` (enable explicitly when needed).
- Volumes
  - Named volumes for data persistence (e.g., `postgres_data`, `chromadb_data`, `qdrant_data`, `ollama_data`).

---

## Product Scope and Roadmap (Authoritative)

From `IMPORTANT/REAL_FEATURES_AND_USERSTORIES.md`:

- What SutazAI Actually Is
  - Local LLM assistant (chat, code assistant, research tool)
  - Strictly local execution and privacy-first boundaries

- Strategic Phases
  - Phase 1 (Foundation, 7 days): Fix model config, stabilize infra, implement one functional agent
  - Phase 2 (Integration, 30 days): Agent comms, authentication, API gateway
  - Phase 3 (Scale, 60 days): Performance, multi-agent workflows, hardening
  - Phase 4 (Maturity, 90 days): Advanced features, ecosystem, enterprise

- MVP Epics (Week 1)
  - Chatbot basic with TinyLlama; add context memory
  - Code assistant basic generation
  - Research tool document ingestion (extract, chunk, embed, store in vector DB)

- Success Metrics (targets)
  - TTFB < 500ms; E2E < 2s; Code gen Pass@1 ≥ 70%; RAG NDCG ≥ 0.8; Availability 99.9%; ≥10 concurrent users

---

## Gaps and Not Implemented (Reality vs Plans)

Per `IMPORTANT/TECHNOLOGY_STACK_REPOSITORY_INDEX.md`:
- Only TinyLlama is loaded; references to gpt-oss exist but are not active
- Several agents listed in various documents are NOT running (AgentGPT, AgentZero, AutoGen, CrewAI, Dify, Documind, FinRobot, FlowiseAI, GPT-Engineer, Langflow, LlamaIndex, PentestGPT, PrivateGPT, Semgrep, ShellGPT, Skyvern, TabbyML, etc.)
- Not currently available: Kubernetes/K3s, Terraform, Vault, Jaeger, ELK/Kafka
- Planned but not implemented: Auto-scaling, circuit breakers, advanced security policies, multi-region, backups

Known misalignments to fix (code vs. IMPORTANT):
- Backend service checks still probe legacy ports in some helpers (e.g., Ollama `11434`, Chroma `8001`) while config uses `10104` and `8000`. Align `app.main` helpers to `http://ollama:10104` and `http://chromadb:8000` for consistency.
- Mixed duplication of models endpoints across `backend/app/api/v1/endpoints/models.py` and `backend/app/api/v1/models.py`; keep one as canonical.

---

## Implementation Priorities (from IMPORTANT)

- Phase 1 (already deployed): Ollama (tinyllama), vector DBs (Qdrant/Chroma/FAISS), Streamlit UI
- Phase 2: Integrate LangChain/AutoGen/CrewAI; configure Kong routes; monitoring dashboards
- Phase 3: Add specialized tools (TabbyML, Semgrep, FinRobot, Documind, Browser Use)
- Phase 4: Advanced frameworks and performance tooling (PyTorch/TensorFlow/JAX, context-engineering)

---

## Operating Procedures (fast checks)

- Verify services with the endpoint list above
- Backend docs at http://localhost:10010/docs once running
- Ollama model management via `docker exec sutazai-ollama ollama list` / `ollama pull <model>`
- Use Prometheus/Grafana for live system metrics and dashboards

---

## API Quick Reference (selection)

- Core
  - GET `/health`
  - GET `/metrics`
  - GET `/prometheus-metrics`
  - POST `/chat`, `/think`, `/execute`, `/reason`, `/learn`
- v1 Namespaces
  - `/api/v1/agents/*`, `/api/v1/models/*`, `/api/v1/documents/*`, `/api/v1/system/*`
  - `/api/v1/orchestration/{agents,workflows,status}`
  - `/api/v1/processing/{process,system_state}`
  - `/api/v1/improvement/{analyze,apply}`

Detailed endpoints (from `backend/app/api/v1/endpoints`):
- Agents
  - GET `/api/v1/agents/`
  - POST `/api/v1/agents/workflows/code-improvement`
  - GET `/api/v1/agents/workflows/{workflow_id}`
  - GET `/api/v1/agents/workflows/{workflow_id}/report`
  - POST `/api/v1/agents/consensus`
  - POST `/api/v1/agents/delegate`
- Models
  - GET `/api/v1/models/`
  - POST `/api/v1/models/pull`
- Mesh (Redis Streams)
  - POST `/api/v1/mesh/enqueue`
  - GET `/api/v1/mesh/results`
  - GET `/api/v1/mesh/agents`
  - GET `/api/v1/mesh/health`
  - POST `/api/v1/mesh/ollama/generate`
- Monitoring
  - GET `/api/v1/monitoring/metrics` (Prometheus format)
  - GET `/api/v1/monitoring/health`
  - GET `/api/v1/monitoring/agents/health`
  - GET `/api/v1/monitoring/agents/{agent_name}/metrics`
  - GET `/api/v1/monitoring/sla/report`

---

### Additional v1 modules (discovered in code)

- Coordinator (`backend/app/api/v1/coordinator.py`)
  - GET `/api/v1/coordinator/status`, `/tasks`, `/agents`, `/agents/status`, `/collective/status`, `/deploy/status`
  - POST `/api/v1/coordinator/task`, `/agents/discover`, `/agents/start-all`, `/agents/activate-agi`, `/agents/stop-all`, `/deploy/mass-activation`, `/deploy/activate-collective`

- Orchestration (`backend/app/api/v1/orchestration.py`)
  - GET `/api/v1/orchestration/agents`, `/agents/healthy`, `/agents/capability/{capability}`, `/tasks/queue/status`, `/tasks/routing/history`, `/load-balancing/algorithms`, `/system/status`, `/system/metrics`, `/health`
  - POST `/api/v1/orchestration/agents/register`, `/agents/discover`, `/tasks/submit`, `/load-balancing/configure`, `/workflows/create`, `/workflows/{workflow_id}/{cancel|pause|resume}`, `/messages/send`, `/messages/broadcast`, `/coordination/consensus`

- Security (`backend/app/api/v1/security.py`)
  - POST `/api/v1/security/login`, `/refresh`, `/logout`, `/encrypt`, `/decrypt`, `/compliance/gdpr/{action}`, `/test/vulnerability-scan`
  - GET `/api/v1/security/report`, `/audit/events`, `/compliance/status`, `/config`

- Vectors (`backend/app/api/v1/vectors.py`)
  - POST `/api/v1/vectors/initialize`, `/add`, `/search`, `/optimize`
  - GET `/api/v1/vectors/stats`

- Models (alt module) (`backend/app/api/v1/models.py`)
  - GET `/api/v1/models/`, `/status`
  - POST `/api/v1/models/pull`, `/generate`, `/chat`, `/embed`

- Self-Improvement (`backend/app/api/v1/self_improvement.py`)
  - POST `/api/v1/self-improvement/start`, `/stop`, `/analyze-file`, `/metrics/add`, `/suggestions/{suggestion_id}/apply`
  - GET `/api/v1/self-improvement/status`, `/report`, `/suggestions/pending`, `/config`
  - PUT `/api/v1/self-improvement/config`

- Feedback (`backend/app/api/v1/feedback.py`)
  - POST `/api/v1/feedback/start`, `/stop`, `/improvements/{improvement_id}/{approve|reject}`
  - GET `/api/v1/feedback/status`, `/metrics`, `/improvements`, `/health`

---

## CI/CD & Quality Gates

- Workflows (key)
  - `hygiene.yml`: banned keywords, IMPORTANT port alignment, localhost scan
  - `important-alignment.yml`: validates ports and localhost alignment on PRs/pushes
  - `docs-audit.yml`: centralizes docs under `docs/` and `IMPORTANT/`, detects duplicates
  - Additional workflows: tests, security scans, nightly runs (see `.github/workflows/*`)
- Scripts enforcing policy
  - `scripts/validate_ports.py`, `scripts/scan_localhost.py`, `scripts/audit_docs.py`
- Policy
  - Service-name addressing; IMPORTANT is source of truth; docs-as-code; observability-first.

---

## API Documentation Maintenance (policy)

- Primary reference: FastAPI’s live OpenAPI at `/docs` and `/openapi.json`. Keep Pydantic models and examples accurate.
- Inline doc comments should follow generator-friendly style (e.g., apidocjs block notation) to keep docs near code [How to keep API documented](https://magda.io/docs/api-documentation-howto.html).
- Follow reference-guide best practices (overview, auth, endpoints, examples, error codes, rate limits) and keep docs iterative and current in CI [API Documentation Done Right](https://www.getambassador.io/blog/api-documentation-done-right-technical-guide).
- CI enforces doc centralization and hygiene via `.github/workflows/docs-audit.yml` and `scripts/audit_docs.py`.

---

## Notes on Code Hygiene and Refactoring (method references)

Given the repository state, tidy and refactor incrementally:
- Favor structural, non-behavioral edits to improve clarity and symmetry; keep tests green; extract helpers from large blocks; commit small steps [Does this code spark joy?](https://medium.com/gusto-engineering/does-this-code-spark-joy-23b1d7706bc0)
- Use a stepwise refactor plan: validate inputs, isolate pure logic, separate persistence/IO, add tests around behavior, then split into modules [How to Refactor Messy Code](https://www.codementor.io/@rctushar/how-to-refactor-messy-code-a-step-by-step-guide-2m4pnsl4yx)

These practices should be applied while keeping IMPORTANT as the single source of truth for scope and target behavior. For legacy-code documentation guidance leveraged in this consolidation, see: [Documenting Legacy Code: A Guide for 2024](https://overcast.blog/documenting-legacy-code-a-guide-for-2024-dbc0b3ba06a7?gi=08dfb6cbd48c). Static analysis and design x‑rays can be supported with SciTools Understand [company overview](https://www.linkedin.com/company/understandbyscitools) and `scitools.com`.

---

## Operational Runbook & Verification Checklist

- Bring-up (core only)
  - `docker-compose up -d postgres redis qdrant chromadb faiss ollama backend frontend prometheus grafana loki` 
- Quick health checks
  - Backend: `curl -f http://localhost:10010/health`
  - Frontend: `curl -f http://localhost:10011`
  - Ollama: `curl -f http://localhost:10104`
  - ChromaDB: `curl -f http://localhost:10100/api/v1/heartbeat`
  - Qdrant: `curl -f http://localhost:10101/cluster`
  - Prometheus: `curl -f http://localhost:10200/-/healthy`
  - Grafana: `curl -f http://localhost:10201/api/health`
- Backend metrics
  - `curl -s http://localhost:10010/prometheus-metrics | head`
- Agent endpoints (enabled per profile)
  - Check each agent on its external port under 111xx (see matrix).

---

## Security & Code Scanning (recommendations)

- GitHub code scanning (CodeQL) integration recommended for Python/TS code and workflows. See GitHub’s guidance on enabling and managing code scanning and alerts [GitHub Code Scanning Docs](https://docs.github.com/en/code-security/code-scanning).
- Optional repo-wide static scanning utility for inventory/metrics: evaluate a containerized scanner approach such as `repo-scanner` to index repos and metadata before CI runs [repo-scanner](https://github.com/trungkh/repo-scanner).
- Existing hygiene workflows already enforce:
  - Port alignment with IMPORTANT and service-name addressing
  - Doc centralization and banned-keyword checks

---

## Single Source of Truth Reminder

This document reflects ONLY what is documented and verified under `/opt/sutazaiapp/IMPORTANT`:
- `SUTAZAI_SYSTEM_ARCHITECTURE_BLUEPRINT*.md`
- `REAL_FEATURES_AND_USERSTORIES.md`
- `TECHNOLOGY_STACK_REPOSITORY_INDEX.md`
- `DATABASE_SCHEMA.sql`
- `SUTAZAI_PRD.md` / `SUTAZAI_MVP.md` / `SUTAZAI_POC.md` (for scope and acceptance)
- `COMPREHENSIVE_ENGINEERING_STANDARDS*.md` (for engineering governance)

Any conflicting files elsewhere in the repository are superseded by the above until expressly reconciled.
