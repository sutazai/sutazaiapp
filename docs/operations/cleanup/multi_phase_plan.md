---
document_id: "DOC-2025-OPS-CLN-0001"
title: "System-Wide Cleanup — Investigation-First Multi-Phase Plan"
created_date: "2025-08-19 21:10:00 UTC"
created_by: "system-architect.md"
last_modified: "2025-08-19 21:10:00 UTC"
status: "active"
owner: "architecture.team@sutazaiapp.local"
category: "process"
---

# Investigation-First Multi-Phase Plan

Authoritative sources: `/opt/sutazaiapp/IMPORTANT` (PortRegistry.md, diagrams/*, docs/*). MCP protection: Rule 20 (no MCP changes without explicit authorization).

## Phase 0 — Authority Lock-in & Scope
- Canonical documents: IMPORTANT/INDEX.md, diagrams/*, PortRegistry.md
- Freeze MCP changes (Rule 20). Audits only.

## Phase 1 — Dependency & Conflict Analysis (Read-only)
- Reconcile PortRegistry vs compose: identify mismatches and "DEFINED BUT NOT RUNNING" items.
- Build docker inventory; group by role (core, AI, monitoring, agents).
- Verify backend docs vs reality for Kong/MCP integration gaps.

## Phase 2 — Module-by-Module Cleanup (Investigate-first)
- Backend: remove proven-dead code only after full grep + usage validation; centralize mesh/MCP wiring in backend/config + backend/app/mesh.
- Frontend: ensure single `/frontend`; remove legacy only after route/usage verification.
- Docker: consolidate configs into `/docker` per diagrams; no new files outside `/docker`.
- Docs: consolidate duplicates; maintain one source of truth per topic.

## Phase 3 — Integration Testing Checkpoints
- Backend unit/integration; API contracts.
- Playwright e2e: derive testDir from playwright.config; ensure non-duplicated placement under `/tests`.

## Phase 4 — Validation & Stabilization
- Performance: ensure no regressions.
- Security: zero high severity.
- Documentation: update CHANGELOGs and cross-references.

## Evidence from Audit (Quotes)
- PortRegistry (Kong 10005, RabbitMQ 10007, backend 10010, frontend 10011): `/IMPORTANT/diagrams/PortRegistry.md` L19–25
- Agents mostly not running: same file L56–61
- Backend docs assert MCP/Kong integration gaps: `/backend/BACKEND_ARCHITECTURE_INVESTIGATION_REPORT.md`

## Quality Gates
- No deletions without traceable investigation and impact analysis.
- MCP configs immutable without explicit approval (Rule 20).
- Docs & CHANGELOG updated in real time.
