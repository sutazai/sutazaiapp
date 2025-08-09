title: Documentation Changelog
version: 0.1.0
last_updated: 2025-08-09
author: Coding Agent Team
review_status: Draft
next_review: 2025-09-07
---

# Changelog

All notable changes to the `/docs` and system-wide configuration will be documented here per Rule 19.

## 2025-08-09

### Docker Health Check Standardization (v67.9)
- fix(docker): Fixed hardware-resource-optimizer health check port mismatch
  - docker-compose.yml: hardware-resource-optimizer health check changed from `*backend_health_test` (port 8000) to `["CMD", "curl", "-f", "http://localhost:8080/health"]` (correct port 8080)
  - Issue: Service showed "unhealthy" despite working correctly due to health check testing wrong port
  - Verification: All other agent services already use correct health check patterns for their respective ports

[2025-08-09 16:57 UTC] - [v67.9] - [Docker Infrastructure] - [Fix] - Corrected health check port mismatch for hardware-resource-optimizer service; standardized health checks across agent services. Agent: Infrastructure DevOps Manager (Claude Code). Impact: Service will now correctly report healthy status; no behavior change to service functionality. Dependency: Requires container recreation for health check to take effect.

### Monitoring Scripts (v67.8)
- fix(monitoring): Align live monitor connectivity checks with actual service ports
  - scripts/monitoring/live_logs.sh: API Connectivity now derives backend/frontend ports via `docker port` with fallbacks (backend 10010, frontend 10011); endpoint tester updated accordingly.

[2025-08-09 14:20] - [v67.8] - [Monitoring] - [Fix] - Corrected hardcoded ports (8000/8501) to dynamic discovery for accurate status when backend is healthy on 10010 and frontend on 10011. Agent: Coding Agent (DevOps specialist). Impact: Dashboard no longer reports false negatives; no behavior change to services.

### Security Hardening (v67.7)
- security(config): Removed hardcoded database/password defaults from backend settings
  - backend/app/core/config.py: POSTGRES_PASSWORD, NEO4J_PASSWORD, GRAFANA_PASSWORD no longer default to insecure values; validators enforce strong secrets in staging/production
  - backend/core/config.py: POSTGRES_PASSWORD default removed
  - backend/app/core/connection_pool.py: eliminated hardcoded fallback secret; env required

[2025-08-09 13:40 UTC] - [v67.7] - [Backend Config] - [Security] - Eliminated hardcoded password fallbacks; env-driven secrets enforced with validators. Agent: Coding Agent (backend specialist). Impact: strengthens secret handling; compose supplies required envs. Dependency: Local DB paths need `POSTGRES_PASSWORD` when DB access is used.

### Architecture Hygiene (v67.7)
- refactor(config): Centralized settings to single source in `backend/app/core/config.py`
  - Added backward-compatible shim at `backend/core/config.py` to re-export `AppSettings`, `get_settings`, `settings`
- fix(defaults): Aligned connection defaults with docker-compose service names
  - Defaults now: `postgres`, `redis`, `http://ollama:11434`

[2025-08-09 13:48 UTC] - [v67.7] - [Backend] - [Refactor] - Config shim added; defaults aligned with compose. Agent: Coding Agent (system + backend + API architects). Impact: reduces duplication and misconfig risk; no behavior change for compose deployments.

### Agent Orchestrator Consolidation (v67.7)
- refactor(agents): Centralized orchestrator logic to `app/services/agent_orchestrator.py`
  - `app/agent_orchestrator.py` now re-exports the canonical orchestrator (compatibility shim)
  - Left `app/orchestration/agent_orchestrator.py` intact (distinct workflow engine) to avoid removing advanced functionality
- deps(backend): Rationalized minimal requirements
  - `backend/requirements-minimal.txt` now includes `-r requirements_minimal.txt` (single canonical minimal spec)

[2025-08-09 14:12 UTC] - [v67.7] - [Backend] - [Refactor/Deps] - Unified agent orchestrator import path and reduced duplicate minimal requirement specs. Agent: Coding Agent (backend + API architects). Impact: fewer duplicate codepaths, simpler dependency management; no breaking changes for existing imports or Docker builds.

### Dead Code Cleanup (v67.7)
- cleanup(backend): Removed unused `backend/app/utils/validation.py` (no references across repo/tests)

[2025-08-09 14:20 UTC] - [v67.7] - [Backend] - [Cleanup] - Deleted unused validation helper to reduce surface area. Agent: Coding Agent. Impact: none; file was unreferenced.
