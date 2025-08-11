# Codebase Cleanup Execution Plan

Author: System Architect
Date: 2025-08-08
Scope: Full repo cleanup guided exclusively by /docs

## Phase 1 — Dependency Analysis & Conflict Identification

- Objectives: Build a reliable inventory of modules, dependencies, and conflicts.
- Inputs: docs/architecture, docs/backend_openapi.json, requirements/*.txt.
- Outputs:
  - reports/cleanup/docs_inventory.json (generated)
  - reports/cleanup/conflict_map.json
  - reports/cleanup/dependency_graph.json
- Tasks:
  - Walk codebase to index modules, entrypoints, and import graph.
  - Detect version conflicts across requirements, Dockerfiles, and compose.
  - Identify duplicates, deprecated APIs, circular imports, and stub code.
  - Verify ports and services per docs/architecture/01-system-overview.md.
- Checkpoint:
  - Architect approves conflict_map.json before proceeding.

## Phase 2 — Module-by-Module Cleanup

- Objectives: Apply targeted refactors to resolve identified issues.
- Prioritization:
  1) backend API schemas and message buses
  2) agents protocol alignment and registries
  3) docker-compose services coherence and env parity
  4) security baselines (secrets, headers, input validation)
- Tasks:
  - Remove dead code and unify canonical enums/schemas.
  - Replace deprecated APIs; fix circular imports.
  - Normalize logging, configuration, and error handling.
  - Align agents with coordinator contracts and health reporting.
- Deliverables:
  - Change logs per file (reports/cleanup/changes/*.md)
  - Updated configs and minimal diffs per module
- Checkpoint:
  - Architect reviews diffs and change logs; approves batch merges.

## Phase 3 — Integration Testing Checkpoints

- Objectives: Validate behavior across services using existing tests and smoke checks.
- Commands:
  - make test
  - pytest -v backend/tests --cov=backend --cov-fail-under=80
  - curl -f http://localhost:10010/health
  - curl -f http://localhost:10104/
  - curl -f http://localhost:10006/v1/status/leader
- Tasks:
  - Execute unit/integration suites; capture failures with trace IDs.
  - Generate integration reports in reports/cleanup/integration_*.jsonl (schema: schemas/cleanup_report_schema.json).
  - Apply minimal patches to fix regressions; re-run critical paths.
- Checkpoint:
  - Architect signs off on integration report before optimization.

## Phase 4 — Final Validation & Code Stabilization

- Objectives: Ensure performance, security, and maintainability baselines.
- Tasks:
  - Run linters and type checks: black, flake8, mypy.
  - Run performance micro-benchmarks and k6 load tests from docs/testing.
  - Run bandit and dependency scans; remediate high/critical issues.
  - Freeze documentation: update docs/CHANGELOG.md and FINAL_DOCUMENTATION_SUMMARY.md.
- Success Criteria:
  - Conflict-free build, tests pass (>=80% backend coverage).
  - Performance meets or exceeds baselines.
  - Documentation and diagrams updated to reflect cleaned architecture.

## Traceability & Governance

- Task IDs: CLN-YYYYMMDD-XXXX (referenced in commit messages and reports).
- Ledger: reports/cleanup/ledger.jsonl (append-only events: status, directives, heartbeats).
- Communication: coordination_bus/messages/*.jsonl (status, directives, heartbeats).
- Reviews: Architect approval required at each phase checkpoint.

