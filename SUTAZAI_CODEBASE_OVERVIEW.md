# SUTAZAI Codebase Overview (Authoritative)

Generated: 2025-08-07
Source of truth: ONLY documents under `/opt/sutazaiapp/IMPORTANT`
Scope: This overview intentionally ignores all other repository files to reflect the verified, intended and actual state documented in IMPORTANT.

---

## Executive Summary

- Identity: Local, privacy-first AI platform (not a distributed AGI with 69 agents)
- Core functions: Chatbot, Code Assistant, Research Tool
- Architecture: Containerized microservices; Ollama for all LLM operations; local databases and monitoring
- Current runtime reality: TinyLlama is the only loaded model; ~7 agent containers are running (most are stubs)
- Verified service mesh: Kong (10005), Consul (10006), RabbitMQ (10007/10008)
- Verified data stores: Postgres (10000), Redis (10001), Neo4j (10002/10003), Qdrant (10101/10102), ChromaDB (10100), FAISS service (10103)
- Application surfaces: Backend FastAPI (10010), Frontend Streamlit (10011), Ollama (10104)
- Observability: Prometheus (10200), Grafana (10201), Loki (10202), plus exporters

References: `IMPORTANT/TECHNOLOGY_STACK_REPOSITORY_INDEX.md`, `IMPORTANT/REAL_FEATURES_AND_USERSTORIES.md`

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

Agent services (runtime reality): 7 agents healthy (or present), mostly stubs — AI Agent Orchestrator (8589), Multi-Agent Coordinator (8587), Hardware Resource Optimizer (8002), Resource Arbitration (8588), Task Assignment Coordinator (8551), Ollama Integration Specialist (11015), AI Metrics Exporter (11063; marked unhealthy).

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

## Notes on Code Hygiene and Refactoring (method references)

Given the repository state, tidy and refactor incrementally:
- Favor structural, non-behavioral edits to improve clarity and symmetry; keep tests green; extract helpers from large blocks; commit small steps [Does this code spark joy?](https://medium.com/gusto-engineering/does-this-code-spark-joy-23b1d7706bc0)
- Use a stepwise refactor plan: validate inputs, isolate pure logic, separate persistence/IO, add tests around behavior, then split into modules [How to Refactor Messy Code](https://www.codementor.io/@rctushar/how-to-refactor-messy-code-a-step-by-step-guide-2m4pnsl4yx)

These practices should be applied while keeping IMPORTANT as the single source of truth for scope and target behavior.

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
