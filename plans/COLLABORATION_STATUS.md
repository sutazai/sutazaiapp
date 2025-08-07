# Collaboration Status — SutazAI (live)

- Updated: 2025-08-07T21:20Z
- Agent: Codex CLI (this session)

## Current Focus (owned by Codex CLI)
- Consolidate requirements usage in Dockerfiles to the repo’s consolidated sets (no per-agent dupes).
- Polish Ollama Integration agent docs/examples (verify `/generate`, `/models`, metrics usage).

## Recently Completed
- Deprecated mesh scripts and archived unused mesh assets per ADR `docs/decisions/2025-08-07-remove-service-mesh.md`.
  - Archived to `archive/service-mesh/`: consul services.json (note), kong.yml (note), service-mesh-init.sh, verify-service-mesh-health.sh, test-service-mesh.sh, service-mesh-resilience.js, README.
  - Removed originals from `config/consul/services.json`, `config/kong/kong.yml`, `scripts/service-mesh-init.sh`, `scripts/verify-service-mesh-health.sh`, `tests/integration/test-service-mesh.sh`, `load-testing/tests/service-mesh-resilience.js`.

## Files Likely To Change Next
- Dockerfiles that `COPY requirements.txt` or bespoke `*-requirements.txt` under `docker/**` and `agents/**`.
- Minor docs under `agents/ollama_integration/` and `docs/mesh/` (only for clarity; no behavior change).

## Please Avoid Editing (to prevent conflicts)
- The three scripts above while we finalize deprecation headers and references.
- Any in-progress Dockerfile adjustments related to requirements until we note done here.

## Hand‑off Notes
- If you’re already modifying Dockerfiles for requirements, ping here by appending your name and target paths so we can divide by directory (e.g., you take `docker/*`, I take `agents/*`).
- If you’re actively wiring RabbitMQ or Consul back, please surface an ADR proposal before reverting the deprecation banners.

## Next Update Window
- I will post the next status update here after scanning and grouping Dockerfile changes (ETA: ~30–45 min).
