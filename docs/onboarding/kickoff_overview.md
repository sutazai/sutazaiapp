# Perfect Jarvis — Team Kickoff Overview

This onboarding package summarizes the current, verifiable architecture, ownership, and operational conventions for Perfect Jarvis. All details below are sourced from the repository and the IMPORTANT documentation folder. No speculative content is included.

## Technology Stack (Verified)

- Backend: FastAPI, Starlette, Uvicorn (see `requirements.txt`)
- Async HTTP/WS: `httpx`, `aiohttp`, `websockets`
- Data: PostgreSQL (`psycopg2-binary`, `asyncpg`), Redis (`redis`)
- Vector DB clients: `chromadb`, `qdrant-client`, `faiss-cpu`
- Observability: `prometheus-client`; Prometheus and Grafana present in compose files
- Messaging: Celery, RabbitMQ clients (`pika`, `aio-pika`)
- Service discovery: Consul client (`python-consul`)
- Frontend: Streamlit-based UI in `frontend/`
- Containerization: Docker, multiple compose tiers (`docker-compose.*.yml`)

Primary references:
- IMPORTANT/TECHNOLOGY_STACK_REPOSITORY_INDEX.md (verified stack)
- IMPORTANT/SUTAZAI_SYSTEM_ARCHITECTURE_BLUEPRINT.md
- requirements*.txt (pinned versions)
- docker-compose.yml, docker-compose.standard.yml, docker-compose.minimal.yml

Note: The technology stack index is now present at `IMPORTANT/TECHNOLOGY_STACK_REPOSITORY_INDEX.md` and reflects only verified items from this repository.

## Repo Analysis (Five Key External Repos)

The referenced repositories (Dipeshpal, Microsoft, llm-guy, danilofalcao, SreejanPersonal) are not present under `repos/` and no submodules are configured. Without local copies or pinned references, no non‑speculative analysis is possible. Action required:
- Provide commit SHAs or mirror these repositories under `repos/` for validation.

## Modular Boundaries & Folder Conventions

Adopt and enforce the following high‑level structure for new and existing code to maintain clarity and separations of concern:
- `components/`: UI components (React or Streamlit modules if expanded)
- `services/`: Microservices source and Dockerfiles
- `utils/`: Reusable helpers and scripts not tied to a service
- `hooks/`: Frontend hooks (if/when React is introduced)
- `schemas/`: Shared data models and validation schemas

Existing relevant locations:
- `services/` contains service assets and integration code (e.g., `ollama_service.py`)
- `schemas/` exists for shared models

## Integration Points (Current)

- API Gateway: Kong (scripts and compose artifacts present; service-mesh stack was deprecated per CHANGELOG)
- Service discovery: Consul client dependency is present; new scripts added to register services at startup (see `/scripts/register_with_consul.py`).
- Vector DBs: Clients for ChromaDB, Qdrant, FAISS exist; integration points to be implemented where required.
- Observability: Prometheus scraping and Grafana dashboards referenced in docs and compose.

## Ownership Matrix (Initial)

- DevOps & Deployment Scripts: `scripts/` (DevOps team ownership)
- Observability: `monitoring/`, Prometheus/Grafana configs (DevOps + Backend)
- Backend services: `services/` and `backend/` (Backend team)
- Frontend (Streamlit): `frontend/` (Frontend team)
- Security Policies & Audits: `security/`, `docs/security/` (Security team)

Please assign named owners in a shared CODEOWNERS or single source of truth once team rosters are confirmed.

## Current Architecture (Verified Elements)

- Services: Python microservices (FastAPI), Streamlit UI
- Data: Redis + PostgreSQL (compose configs), optional vector DBs
- Gateway: Kong (Admin/API routing), Consul for service registry
- Observability: Prometheus scraping targets, Grafana dashboards (as documented)
- Messaging: RabbitMQ clients present; confirm broker enablement before use

### Architecture Diagram (based on compose and configs)

```mermaid
flowchart LR
  User((User)) --> Frontend[Streamlit UI]
  Frontend --> Gateway[Kong API Gateway]
  Gateway --> Backend[Backend API (FastAPI)]

  Backend -->|reads/writes| Postgres[(PostgreSQL)]
  Backend -->|cache| Redis[(Redis)]
  Backend -->|LLM| Ollama[Ollama + TinyLlama]
  Backend -->|discovery| Consul[(Consul)]
  Backend -->|optional| VectorDBs[ChromaDB / Qdrant / FAISS]

  Prometheus[[Prometheus]] -->|scrapes| Backend
  Prometheus -->|scrapes| Gateway
  Grafana[[Grafana]] --> Prometheus
```

Notes:
- Diagram reflects services and environment values declared in `docker-compose.*.yml` and `.env.*` files.
- Only relationships visible in repository configuration are shown; no speculative edges are added.

### API Contracts (Verified Sources)

- Backend base path: `API_V1_STR=/api/v1` (see `docker-compose.yml` for `backend` environment).
- FastAPI OpenAPI: available at runtime under `/docs` and `/openapi.json` on the backend container.
- Gateway routing: configured via `scripts/configure_kong.sh`. Routes map path prefixes to internal services using Consul DNS.

## Constraints & Limitations (Non‑Speculative)

- Referenced external repos are not available locally; cannot assess modular boundaries or adaptation needs.
- Frontend in this repo is Streamlit, not React; any React component work requires project decision and scaffolding.
- Service mesh compose previously removed per `docs/CHANGELOG.md` (v63.3); new configurations must align with current deployment tiers.

## Onboarding Meeting Agenda

- 0–10 min: Architecture walkthrough (compose tiers, services, data, gateway)
- 10–25 min: Environments & deployment (deploy-tier.sh, validation scripts)
- 25–35 min: Observability (metrics, dashboards, alerts overview)
- 35–45 min: Security & hygiene (pre-commit, scans, no-secrets policy)
- 45–55 min: Roadmap and gaps (vector DB integration, streaming, registry)
- 55–60 min: Q&A and action items

## Role Plans & Expected Outcomes

- Backend: Finalize service registration at startup; add health/metrics endpoints; integrate vector DBs where required.
- DevOps: Wire health checks into CI, enforce fail‑fast on degraded services; maintain deploy scripts and tiered profiles.
- Frontend: Maintain Streamlit UI; coordinate API contracts; accessibility alignment (WCAG) for any new UI.
- QA: Add smoke tests and integration checks; track failures in CI.
- Documentation: Keep READMEs, diagrams, and CHANGELOG current; link external references.

## External References

- IMPORTANT/*.md (PRD, Architecture Blueprints, System Roadmap)
- Root `requirements.txt` (pinned, security‑validated versions)
- `docker-compose.*.yml` (deployment topology)
- `scripts/onboarding/generate_kickoff_deck.py` (generates `docs/onboarding/kickoff_deck_v1.pptx` using `python-pptx`)

## Next Steps

- Provide the five external repos locally or as submodules for concrete analysis.
- Confirm microservice source locations for newly added Dockerfiles (`services/jarvis-*`).
- Approve CI wiring for new health checks and gate merges on failures.

## Appendix

- Stakeholder-provided synthesis plan (unverified): see `docs/onboarding/STAKEHOLDER_SYNTHESIS_PLAN_UNVERIFIED.md`. Treat as input for planning only until repositories and assumptions are validated.
