# SutazAI Codebase Cleanup Plan (System Architect Phase)

Date: 2025-08-14
Owner: System Architect
Scope: Repository-wide hygiene per Professional Codebase Standards & Hygiene Guide and Rules 1–20

## Objectives
- Enforce Single Source of Truth and eliminate duplication across configs, scripts, and docs.
- Preserve functionality (ZRT) while improving consistency, security, and operability.
- Standardize dependency management and environment configuration.
- Prepare for reliable compose-based orchestration (dev/prod) without ad-hoc steps.

## Prioritized Workstreams
1) Compose + Orchestration Consistency (UDA/DAS)
   - Align service DNS and env across services (use `sutazai-*` aliases consistently).
   - Remove obvious duplication/merge artifacts; avoid changing MCP config.
   - Ensure healthchecks and dependencies reflect actual service names.

2) Dependencies + Python Standards (PPS)
   - Ensure runtime deps required by backend are present in image builds.
   - Consolidate requirements references; avoid divergence between `backend/requirements.txt` and `requirements/*` where practical.
   - Maintain pinned, secure versions; no hardcoded secrets.

3) Scripts Consolidation (SCP)
   - Identify duplicate/legacy scripts per CODEBASE_DUPLICATION_REPORT.md and phase out safely.
   - Keep a single self-updating deployment entry (`scripts/deploy.sh`).

4) Documentation & Change Tracking (DES/Rule 19)
   - Add ADR and cleanup plan docs; update CHANGELOG for every modification.
   - Keep MCP-related docs unchanged (Rule 20).

## Today’s Concrete Changes
- docker-compose.yml: Aligned `x-database-config` hostnames to `sutazai-postgres` and `sutazai-redis` for consistency with live services.
- No MCP configuration changed.

## Next Steps (Proposed)
1) Backend image reliability: update base/build to install `backend/requirements.txt` successfully in CI (investigate prior build failure cause).
2) Frontend bring-up: verify Streamlit container with `BACKEND_URL` pointing to network DNS (`http://backend:8000`) or host override for local loops.
3) Compose image availability: add fallback for `*-secure` images or provide local build targets; document expectations in `README.md`.
4) Script de-dup pass: action items from CODEBASE_DUPLICATION_REPORT.md, archiving obsolete backups under a single `/backups/` namespace or pruning after justification.

## Safeguards
- No changes to `.mcp.json` or MCP wrappers (Rule 20).
- No removal of functional code without investigation and documentation (FPP).
- Changes logged in `docs/CHANGELOG.md` (Rule 19).

